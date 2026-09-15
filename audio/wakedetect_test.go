package audio

import (
	"sync"
	"testing"
	"time"
)

// fakeScorer stands in for the tflite pipeline so the ring-reading, threshold
// and refractory logic can be tested without cgo or the models.
type fakeScorer struct {
	mu      sync.Mutex
	fed     [][]byte
	scores  []float32 // returned in order; the last value repeats
	resets  int
	closed  bool
	warmups int // hops to report as not-ready before scoring begins
}

func (f *fakeScorer) Feed(pcm []byte) (float32, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	cp := make([]byte, len(pcm))
	copy(cp, pcm)
	f.fed = append(f.fed, cp)

	if len(f.fed) <= f.warmups {
		return 0, false, nil
	}
	if len(f.scores) == 0 {
		return 0, true, nil
	}
	i := len(f.fed) - f.warmups - 1
	if i >= len(f.scores) {
		i = len(f.scores) - 1
	}
	return f.scores[i], true, nil
}

func (f *fakeScorer) Reset() { f.mu.Lock(); f.resets++; f.mu.Unlock() }
func (f *fakeScorer) Close() { f.mu.Lock(); f.closed = true; f.mu.Unlock() }

func (f *fakeScorer) fedCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.fed)
}

// waitFor polls until cond holds or the deadline passes, so the tests do not
// depend on the exact scheduling of the detector's ticker.
func waitFor(t *testing.T, d time.Duration, cond func() bool) bool {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return cond()
}

func newTestRing() *Ring {
	return NewRing(ringSeconds * sampleRate * bytesPerSample)
}

func TestDetectorFeedsWholeFramesInOrder(t *testing.T) {
	ring := newTestRing()
	f := &fakeScorer{}
	d := newDetector(ring, f, WakeConfig{Threshold: 2}) // never fires
	d.Start()
	defer d.Stop()

	// Write five frames as one blob plus a partial sixth; the detector must
	// consume the five and leave the remainder for later.
	blob := make([]byte, 5*frameBytes+100)
	for i := range blob {
		blob[i] = byte(i)
	}
	ring.Write(blob)

	if !waitFor(t, 2*time.Second, func() bool { return f.fedCount() == 5 }) {
		t.Fatalf("expected 5 frames fed, got %d", f.fedCount())
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	for i, got := range f.fed {
		if len(got) != frameBytes {
			t.Fatalf("frame %d is %d bytes, want %d", i, len(got), frameBytes)
		}
		want := blob[i*frameBytes : (i+1)*frameBytes]
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("frame %d differs at byte %d: got %d want %d", i, j, got[j], want[j])
			}
		}
	}
}

func TestDetectorFiresOnceWithinRefractory(t *testing.T) {
	ring := newTestRing()
	// Every hop scores above threshold, as happens for several hops around a
	// real utterance.
	f := &fakeScorer{scores: []float32{0.999}}

	var mu sync.Mutex
	var fires []int64
	d := newDetector(ring, f, WakeConfig{
		Threshold:  0.99,
		Refractory: 10 * time.Second,
		OnDetect: func(offset int64, score float32) {
			mu.Lock()
			fires = append(fires, offset)
			mu.Unlock()
		},
	})
	d.Start()
	defer d.Stop()

	ring.Write(make([]byte, 10*frameBytes))

	if !waitFor(t, 2*time.Second, func() bool { return f.fedCount() == 10 }) {
		t.Fatalf("only %d frames consumed", f.fedCount())
	}

	mu.Lock()
	defer mu.Unlock()
	if len(fires) != 1 {
		t.Fatalf("expected exactly 1 detection inside the refractory window, got %d", len(fires))
	}
	// The offset must be the end of the first qualifying hop, which is where
	// the phrase ended and the command begins.
	if fires[0] != frameBytes {
		t.Fatalf("fired at offset %d, expected %d", fires[0], frameBytes)
	}
}

func TestDetectorFiresAgainAfterRefractory(t *testing.T) {
	ring := newTestRing()
	f := &fakeScorer{scores: []float32{0.999}}

	var mu sync.Mutex
	count := 0
	d := newDetector(ring, f, WakeConfig{
		Threshold:  0.99,
		Refractory: 50 * time.Millisecond,
		OnDetect:   func(int64, float32) { mu.Lock(); count++; mu.Unlock() },
	})
	d.Start()
	defer d.Stop()

	for i := 0; i < 3; i++ {
		ring.Write(make([]byte, frameBytes))
		time.Sleep(120 * time.Millisecond) // longer than the refractory
	}

	if !waitFor(t, 2*time.Second, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return count == 3
	}) {
		mu.Lock()
		got := count
		mu.Unlock()
		t.Fatalf("expected 3 detections once the refractory lapsed, got %d", got)
	}
}

func TestDetectorIgnoresScoresBelowThreshold(t *testing.T) {
	ring := newTestRing()
	// 0.98 is close but must not fire: the deployed model is calibrated at
	// 0.99 precisely because the gap matters.
	f := &fakeScorer{scores: []float32{0.98}}

	fired := false
	var mu sync.Mutex
	d := newDetector(ring, f, WakeConfig{
		Threshold: 0.99,
		OnDetect:  func(int64, float32) { mu.Lock(); fired = true; mu.Unlock() },
	})
	d.Start()
	defer d.Stop()

	ring.Write(make([]byte, 5*frameBytes))
	waitFor(t, 500*time.Millisecond, func() bool { return f.fedCount() == 5 })

	mu.Lock()
	defer mu.Unlock()
	if fired {
		t.Fatal("fired on a score below the threshold")
	}
}

func TestDetectorDoesNotFireWhilePipelineIsWarmingUp(t *testing.T) {
	ring := newTestRing()
	// The real pipeline reports not-ready until its windows fill; a score
	// emitted then must not be treated as a detection.
	f := &fakeScorer{scores: []float32{0.999}, warmups: 4}

	fired := false
	var mu sync.Mutex
	d := newDetector(ring, f, WakeConfig{
		Threshold: 0.99,
		OnDetect:  func(int64, float32) { mu.Lock(); fired = true; mu.Unlock() },
	})
	d.Start()
	defer d.Stop()

	ring.Write(make([]byte, 3*frameBytes))
	waitFor(t, 500*time.Millisecond, func() bool { return f.fedCount() == 3 })

	mu.Lock()
	got := fired
	mu.Unlock()
	if got {
		t.Fatal("fired while the scorer still reported not-ready")
	}
}

func TestDetectorSkipsAheadWhenFarBehind(t *testing.T) {
	ring := newTestRing()
	f := &fakeScorer{}
	d := newDetector(ring, f, WakeConfig{Threshold: 2})

	d.Start()
	defer d.Stop()

	// Dump far more than maxBacklogFrames in one go, as happens when the
	// device stalls and capture keeps running. The loop must abandon it rather
	// than grind through stale audio.
	ring.Write(make([]byte, (maxBacklogFrames+50)*frameBytes))

	// It must reset and skip rather than feed the whole backlog.
	if !waitFor(t, 2*time.Second, func() bool { return f.resets > 0 }) {
		t.Fatal("never reset after falling behind")
	}
	if n := f.fedCount(); n > maxBacklogFrames {
		t.Fatalf("fed %d stale frames instead of skipping to the live edge", n)
	}
}

func TestDetectorStopReleasesScorer(t *testing.T) {
	ring := newTestRing()
	f := &fakeScorer{}
	d := newDetector(ring, f, WakeConfig{Threshold: 2})
	d.Start()
	d.Stop()

	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.closed {
		t.Fatal("Stop did not close the scorer; the models would leak")
	}
}

func TestDefaultWakeConfigMatchesDeployedModel(t *testing.T) {
	// hey_spotify_v3 was measured at 97.7% recall and 0.79 false alarms/hour
	// at 0.99. At 0.5 the same model fires roughly every four minutes, so a
	// change here is a behavioural change, not a tidy-up.
	cfg := DefaultWakeConfig()
	if cfg.Threshold != 0.99 {
		t.Fatalf("default threshold is %v, expected 0.99", cfg.Threshold)
	}
	if cfg.Refractory <= 0 {
		t.Fatal("refractory must be positive or one utterance fires repeatedly")
	}
}
