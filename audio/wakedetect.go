package audio

import (
	"errors"
	"log"
	"time"
)

const (
	// One hop of the wake-word pipeline. Everything downstream - the mel
	// frames, the embedding window, the classifier - is defined in multiples
	// of this, so it is not a tunable.
	frameSamples  = 1280 // 80ms at 16kHz
	frameBytes    = frameSamples * bytesPerSample
	featureFrames = 16 // embeddings the classifier scores at once (~1.28s)

	// How often the loop drains the ring. Shorter than a hop so the backlog
	// stays at most one frame in the steady state.
	wakePollInterval = 40 * time.Millisecond

	// If the loop ever falls this far behind - the device was busy, or capture
	// restarted - the audio in between is stale. Skipping to the live edge and
	// resetting is better than spending CPU catching up on speech that has
	// already passed.
	maxBacklogFrames = 40 // 3.2 seconds
)

// ErrWakeUnsupported means this binary was built without the tflite runtime.
// Detection needs cgo and the cross-compiled TensorFlow Lite C library, which
// the default CGO_ENABLED=0 build deliberately does not link - that build
// stays static and is what the Windows deploy script produces. Build with
// `-tags wakeword` (see build-wakeword.sh) to enable detection.
//
// Declared here rather than beside the stub so callers can test for it in
// either build.
var ErrWakeUnsupported = errors.New("wake word detection not compiled into this build")

// ScorerConfig locates the three models the detector chains together.
type ScorerConfig struct {
	MelspectrogramPath string
	EmbeddingPath      string
	ClassifierPath     string
	Threads            int
}

// Scorer turns a stream of audio hops into wake-word scores. The tflite
// implementation is behind the `wakeword` build tag; builds without it get a
// stub so the daemon still compiles and runs with detection disabled.
type Scorer interface {
	// Feed consumes exactly one hop and returns a score. ok is false while the
	// pipeline is still filling its windows.
	Feed(pcm []byte) (score float32, ok bool, err error)
	// Reset discards streaming state after a gap in the audio.
	Reset()
	Close()
}

// WakeConfig tunes detection behaviour.
type WakeConfig struct {
	// Threshold is a property of the trained model, not a general default.
	// hey_spotify_v3 was measured at 97.7% recall and 0.79 false alarms/hour
	// at 0.99; at 0.50 the same model fires every four minutes.
	Threshold float32

	// Refractory suppresses re-triggering while the score stays high. The
	// score remains above threshold for several hops after one utterance, so
	// without this a single "hey spotify" would fire repeatedly.
	Refractory time.Duration

	// OnDetect is called with the ring offset at the end of the hop that
	// crossed the threshold - which is approximately where the phrase ended,
	// and therefore where the command that follows begins.
	OnDetect func(offset int64, score float32)
}

// DefaultWakeConfig matches the deployed hey_spotify_v3 model.
func DefaultWakeConfig() WakeConfig {
	return WakeConfig{Threshold: 0.99, Refractory: 2 * time.Second}
}

// detector drives a Scorer over the live ring.
type detector struct {
	ring   *Ring
	scorer Scorer
	cfg    WakeConfig
	stop   chan struct{}
	done   chan struct{}
}

func newDetector(ring *Ring, scorer Scorer, cfg WakeConfig) *detector {
	if cfg.Threshold <= 0 {
		cfg.Threshold = DefaultWakeConfig().Threshold
	}
	if cfg.Refractory <= 0 {
		cfg.Refractory = DefaultWakeConfig().Refractory
	}
	return &detector{
		ring:   ring,
		scorer: scorer,
		cfg:    cfg,
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
	}
}

func (d *detector) Start() {
	// Capture the starting offset synchronously. Reading it inside the
	// goroutine would make the boundary depend on scheduling: audio written
	// immediately after Start could be skipped or replayed depending on when
	// the goroutine first ran.
	go d.run(d.ring.Written())
}

func (d *detector) Stop() {
	close(d.stop)
	select {
	case <-d.done:
	case <-time.After(2 * time.Second):
		log.Printf("wakeword: detector did not stop cleanly")
	}
	d.scorer.Close()
}

// run consumes the ring from `next` onwards. Starting at the live edge means
// audio captured before detection was armed is ignored, rather than replayed
// and fired on as though it were current speech.
func (d *detector) run(next int64) {
	defer close(d.done)

	var lastFire time.Time

	ticker := time.NewTicker(wakePollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-d.stop:
			return
		case <-ticker.C:
		}

		written := d.ring.Written()
		if backlog := (written - next) / frameBytes; backlog > maxBacklogFrames {
			log.Printf("wakeword: %d frames behind, skipping to live edge", backlog)
			next = written - written%frameBytes
			d.scorer.Reset()
			continue
		}

		for written-next >= frameBytes {
			pcm, err := d.ring.Read(next, next+frameBytes)
			if err != nil {
				if errors.Is(err, ErrEvicted) {
					// Capture outran us; resync rather than give up.
					log.Printf("wakeword: audio evicted before scoring, resyncing")
					next = d.ring.Oldest()
					d.scorer.Reset()
					break
				}
				log.Printf("wakeword: ring read failed: %v", err)
				break
			}
			if len(pcm) < frameBytes {
				break
			}
			next += frameBytes

			score, ok, err := d.scorer.Feed(pcm)
			if err != nil {
				log.Printf("wakeword: scoring failed: %v", err)
				continue
			}
			if !ok || score < d.cfg.Threshold {
				continue
			}
			if time.Since(lastFire) < d.cfg.Refractory {
				continue
			}
			lastFire = time.Now()
			log.Printf("wakeword: detected (score %.3f) at offset %d", score, next)
			if d.cfg.OnDetect != nil {
				d.cfg.OnDetect(next, score)
			}
		}
	}
}
