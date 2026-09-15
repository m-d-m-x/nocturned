package audio

import (
	"encoding/binary"
	"math"
	"testing"
)

// tone fills a buffer with a square wave of the given amplitude, giving a
// predictable RMS without needing real audio.
func tone(seconds float64, amplitude int16) []byte {
	samples := int(seconds * sampleRate)
	out := make([]byte, samples*bytesPerSample)
	for i := 0; i < samples; i++ {
		v := amplitude
		if (i/40)%2 == 0 {
			v = -amplitude
		}
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], uint16(v))
	}
	return out
}

func TestAmbientFloorFindsQuietAudioBeforeSession(t *testing.T) {
	ring := newTestRing()

	// Five seconds of room tone, then a loud second: the shape of a wake-word
	// session, where the phrase immediately precedes the session start.
	ring.Write(tone(5, 8))
	ring.Write(tone(1, 400))
	start := ring.Written()

	floor := ambientFloor(ring, start)
	if math.IsInf(floor, 1) {
		t.Fatal("no floor measured despite six seconds of audio")
	}
	// It must see past the loud tail to the quiet audio behind it.
	if floor > 20 {
		t.Fatalf("floor %.1f reflects the loud phrase, not the room behind it", floor)
	}
}

func TestAmbientFloorIgnoresLoudLeadIn(t *testing.T) {
	ring := newTestRing()
	ring.Write(tone(6, 500)) // nothing quiet anywhere
	floor := ambientFloor(ring, ring.Written())

	// With no quiet audio available the floor is legitimately high; the point
	// is that it is finite and derived from real audio rather than guessed.
	if math.IsInf(floor, 1) {
		t.Fatal("expected a measured floor")
	}
	if floor < 100 {
		t.Fatalf("floor %.1f is implausibly low for uniformly loud audio", floor)
	}
}

func TestAmbientFloorInfiniteWhenRingIsEmpty(t *testing.T) {
	ring := newTestRing()
	if floor := ambientFloor(ring, 0); !math.IsInf(floor, 1) {
		t.Fatalf("expected +Inf with no audio, got %.1f", floor)
	}
}

// The regression this whole change exists for: a session whose audio is speech
// from its very first window.
//
// The failure is temporal, not a simple comparison. Each window is judged
// against a threshold derived from the lowest RMS seen SO FAR, so the first
// window is always measured against three times itself, and a session that
// never contains a quiet window never lowers the bar in time. The sequence
// below reproduces a real failed wake-word session, which reported
// peak=481 floor=110 threshold=329 windows=0.
func TestSeededFloorDetectsSpeechThatStartsImmediately(t *testing.T) {
	windows := []float64{481, 300, 180, 120, 110, 115}

	// Replays watchForSilence's running floor and returns how many windows
	// would have counted as speech.
	countSpeech := func(seedFloor float64) (n int, finalFloor, finalThreshold float64) {
		floor := seedFloor
		for _, rms := range windows {
			if rms < floor {
				floor = rms
			}
			threshold := adaptiveThreshold(floor)
			if rms >= threshold {
				n++
			}
			finalFloor, finalThreshold = floor, threshold
		}
		return n, finalFloor, finalThreshold
	}

	// Old behaviour: the floor is learned from inside the session only.
	n, floor, threshold := countSpeech(math.Inf(1))
	if n != 0 {
		t.Fatalf("expected the unseeded floor to miss all speech, counted %d windows", n)
	}
	// Reproduces the numbers from the device log, confirming this models the
	// real failure rather than an invented one.
	if math.Abs(floor-110) > 1 || math.Abs(threshold-329) > 2 {
		t.Fatalf("simulation drifted from the observed failure: floor=%.0f threshold=%.0f, want 110/329",
			floor, threshold)
	}

	// New behaviour: the floor comes from the room before the phrase.
	n, _, _ = countSpeech(8.0)
	if n != len(windows) {
		t.Fatalf("seeded floor counted %d of %d speech windows", n, len(windows))
	}
}

func TestInitialSilenceGraceExceedsTrailingSilence(t *testing.T) {
	// A wake-word session opens the instant the phrase ends, and people pause
	// before the command. If the pre-speech allowance were not longer than the
	// trailing one, the session would close before they started.
	if initialSilenceAllowed <= silenceRequired {
		t.Fatalf("initial grace %.1fs must exceed trailing silence %.1fs",
			initialSilenceAllowed, silenceRequired)
	}
	if initialSilenceAllowed >= captureMaxSec {
		t.Fatal("initial grace must stay well inside the session cap")
	}
}
