package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/usenocturne/nocturned/utils"
)

const (
	captureDevice = "default"
	captureRate   = "16000"
	captureFormat = "S16_LE"
	captureChans  = "1"
	captureMaxSec = 30
)

// Silence detection tuning.
//
// A fixed RMS threshold assumes a hot microphone. The Car Thing's PDM mic has
// no gain control (amixer exposes no capture element) and idles around RMS 6-34
// with speech peaking at 300-480, so a fixed 800 meant speech never registered:
// every window counted as silence, captures auto-stopped at the minimum, and
// the speech guard threw the recording away. The threshold is therefore derived
// from the quietest window actually observed in each capture, with the old
// constant kept as the ceiling for genuinely noisy environments.
const (
	checkInterval   = 200 * time.Millisecond
	silenceRequired = 1.4 // seconds of continuous silence before auto-stop
	minRecordTime   = 1.0 // seconds before silence detection activates

	speechFactor   = 3.0   // speech must exceed the noise floor by this much
	thresholdFloor = 20.0  // never so low that mic hiss reads as speech
	thresholdCeil  = 800.0 // never so high that normal speech is missed

	// Speech has to persist, not just spike. RMS is measured over a trailing
	// 1s window, so even a single word elevates several consecutive ticks,
	// while a transient click barely moves a one-second average.
	minSpeechWindows = 2

	// Discarding a real utterance makes voice unusable; letting a silent one
	// through costs one spurious search. So a capture is only thrown away when
	// two independent signals agree there was no speech: no sustained
	// above-threshold windows, AND a peak that never rose meaningfully above
	// the noise floor. Measured: speech registers 5-11 windows (peak 303-480),
	// silence registers 0 (peak 101-183).
	discardPeakFactor = 20.0

	wavHeaderSize  = 44
	sampleRate     = 16000
	bytesPerSample = 2
	windowBytes    = int64(sampleRate * bytesPerSample) // the 1s RMS window

	// How far back a session looks to estimate the room's noise floor, in 1s
	// windows. A wake-word session begins at the end of the phrase, so the
	// second or two immediately before it is the phrase itself; looking
	// further back reaches genuine ambient audio.
	ambientLookbackWindows = 6

	// Silence tolerated before any speech has been heard. A wake-word session
	// opens the moment the phrase ends, and people pause before starting the
	// command, so the original 1.4s closed the session before they spoke.
	// Once speech has been heard, silenceRequired governs the tail.
	initialSilenceAllowed = 4.0

	// Guard band kept either side of detected speech when trimming.
	trimPadBytes = int64(0.25 * sampleRate * bytesPerSample)
	// Never trim down to something too short to be a real utterance.
	minTrimmedBytes = int64(0.6 * sampleRate * bytesPerSample)
)

// ErrNoCapture means there was no session to act on. Callers should treat this
// as a no-op rather than a failure: with silence-driven auto-stop, the daemon
// routinely ends a session before the client gets around to asking it to.
var ErrNoCapture = errors.New("no capture in progress")

func storeFloat(a *atomic.Uint64, v float64) { a.Store(math.Float64bits(v)) }
func loadFloat(a *atomic.Uint64) float64     { return math.Float64frombits(a.Load()) }

// adaptiveThreshold turns an observed noise floor into a speech threshold.
func adaptiveThreshold(floor float64) float64 {
	if math.IsInf(floor, 1) {
		return thresholdCeil
	}
	t := floor * speechFactor
	if t < thresholdFloor {
		t = thresholdFloor
	}
	if t > thresholdCeil {
		t = thresholdCeil
	}
	return t
}

// session is a window onto the always-on ring rather than a recording process.
// It records where in the stream the utterance began; the microphone itself is
// never opened or closed for it.
type session struct {
	startOffset int64
	seedFloor   float64 // ambient RMS measured before the session opened
	provider    string
	apiKey      string
	lang        string
	started     time.Time

	evaluated     atomic.Bool
	sawSpeech     atomic.Bool
	peakRms       atomic.Uint64
	floorRms      atomic.Uint64
	threshold     atomic.Uint64
	speechWindows atomic.Uint32

	// Absolute ring offsets at the first and last tick that measured speech.
	firstSpeechAt atomic.Int64
	lastSpeechAt  atomic.Int64
}

type Manager struct {
	mu      sync.Mutex
	current *session
	wsHub   *utils.WebSocketHub
	capture *Capture

	// Wake-word detection. wakeParams holds the credentials a detected phrase
	// starts a session with; the client supplies them when arming, exactly as
	// it does for a dial-triggered session.
	wake       *detector
	wakeParams StartParams
	// What the running detector was built with, so a re-arm can tell a
	// credential refresh from a genuine configuration change.
	wakeModels    ScorerConfig
	wakeThreshold float32
}

// NewManager builds the manager and opens the microphone. Capture is
// continuous for the lifetime of the daemon: wake-word detection needs the
// stream always running, and sessions read from it rather than starting it.
func NewManager(wsHub *utils.WebSocketHub) *Manager {
	m := &Manager{wsHub: wsHub, capture: NewCapture()}
	m.capture.Start()
	return m
}

// Close stops detection and releases the microphone.
func (m *Manager) Close() {
	m.DisableWake()
	if m.capture != nil {
		m.capture.Stop()
	}
}

// Ring exposes the live audio stream so a wake-word detector can read it
// alongside the session logic, without either owning the microphone.
func (m *Manager) Ring() *Ring { return m.capture.Ring() }

func (m *Manager) broadcastState(state string) {
	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type:    "voice_state",
		Payload: utils.VoiceStatePayload{State: state},
	})
}

func (m *Manager) broadcastError(msg string) {
	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type:    "voice_transcript",
		Payload: utils.VoiceTranscriptPayload{Error: msg},
	})
}

type StartParams struct {
	Provider string
	APIKey   string
	Lang     string
}

// Start opens a session over the live audio stream. It reaches back by
// PreRollBytes so the beginning of the utterance is not lost to trigger latency
// - and, once a wake word drives this, so the words spoken immediately after
// the phrase are already in hand.
func (m *Manager) Start(p StartParams) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.current != nil {
		return fmt.Errorf("capture already in progress")
	}
	if p.Provider == "" {
		p.Provider = "groq"
	}
	if p.APIKey == "" {
		return fmt.Errorf("apiKey is required")
	}
	if !m.capture.Running() {
		return fmt.Errorf("microphone is not capturing")
	}

	ring := m.capture.Ring()
	start := ring.Written() - PreRollBytes()

	return m.startLocked(p, start)
}

// startLocked opens a session beginning at an absolute ring offset. Callers
// must hold m.mu and have already validated p.
//
// The offset matters for the wake-word path: a dial press reaches back by
// PreRollBytes because the user has already started speaking, whereas a
// detected phrase supplies the offset where that phrase ended, so the command
// that follows is captured without the wake word itself being transcribed.
func (m *Manager) startLocked(p StartParams, start int64) error {
	ring := m.capture.Ring()
	if oldest := ring.Oldest(); start < oldest {
		start = oldest
	}
	if written := ring.Written(); start > written {
		start = written
	}

	s := &session{
		startOffset: start,
		seedFloor:   ambientFloor(ring, start),
		provider:    p.Provider,
		apiKey:      p.APIKey,
		lang:        p.Lang,
		started:     time.Now(),
	}
	m.current = s

	log.Printf("audio: session opened at ring offset %d", start)
	m.broadcastState("recording")
	go m.watchForSilence(s)
	return nil
}

func (m *Manager) Stop() error { return m.stop("manual") }

func (m *Manager) stop(reason string) error {
	m.mu.Lock()
	s := m.current
	m.current = nil
	m.mu.Unlock()

	if s == nil {
		return ErrNoCapture
	}

	ring := m.capture.Ring()
	endOffset := ring.Written()
	durationMs := time.Since(s.started).Milliseconds()

	peak := loadFloat(&s.peakRms)
	floor := loadFloat(&s.floorRms)
	threshold := loadFloat(&s.threshold)
	windows := s.speechWindows.Load()

	// Whisper hallucinates confident phrases out of silence ("Thank you."), and
	// the client acts on whatever comes back - so an accidental trigger would
	// start playing music. evaluated gates this: a stop inside minRecordTime
	// never measured anything and must not be mistaken for silence.
	quiet := peak < floor*discardPeakFactor
	if s.evaluated.Load() && !s.sawSpeech.Load() && quiet {
		log.Printf(
			"audio: discarded %dms session - %d speech windows, peak rms %.0f vs threshold %.0f, floor %.0f",
			durationMs, windows, peak, threshold, floor,
		)
		m.wsHub.Broadcast(utils.WebSocketEvent{
			Type: "voice_state",
			Payload: utils.VoiceStatePayload{
				State: "discarded", Reason: reason, DurationMs: durationMs,
				Bytes: endOffset - s.startOffset, PeakRMS: peak,
				FloorRMS: floor, Threshold: threshold, Windows: windows,
			},
		})
		m.broadcastError("no speech detected")
		return nil
	}

	from, to := speechRange(s.startOffset, endOffset, s.firstSpeechAt.Load(), s.lastSpeechAt.Load())
	pcm, err := ring.Read(from, to)
	if err != nil {
		log.Printf("audio: could not read session audio: %v", err)
		m.broadcastError("recording was lost")
		return err
	}
	if len(pcm) == 0 {
		m.broadcastError("recording produced no audio")
		return fmt.Errorf("recording produced no audio")
	}

	wavPath, err := tempWavPath()
	if err != nil {
		m.broadcastError(err.Error())
		return err
	}
	if err := writeWav(wavPath, pcm); err != nil {
		m.broadcastError(err.Error())
		return err
	}

	uploadBytes := int64(wavHeaderSize + len(pcm))
	log.Printf(
		"audio: stopped (%s) after %dms, %d -> %d bytes (%s), %d speech windows, peak rms %.0f, floor %.0f, threshold %.0f",
		reason, durationMs, endOffset-s.startOffset, uploadBytes, durationString(int64(len(pcm))),
		windows, peak, floor, threshold,
	)

	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type: "voice_state",
		Payload: utils.VoiceStatePayload{
			State: "transcribing", Reason: reason, DurationMs: durationMs,
			Bytes: uploadBytes, PeakRMS: peak, FloorRMS: floor,
			Threshold: threshold, Windows: windows,
		},
	})

	go m.transcribeAndBroadcast(s, wavPath)
	return nil
}

func (m *Manager) Cancel() error {
	m.mu.Lock()
	s := m.current
	m.current = nil
	m.mu.Unlock()

	if s == nil {
		return nil
	}
	m.broadcastState("cancelled")
	return nil
}

// watchForSilence reads the tail of the ring rather than polling a growing
// file, so it always sees a complete measurement window and never has to guess
// whether a partially-written buffer is trustworthy.
func (m *Manager) watchForSilence(s *session) {
	silenceSecs := 0.0
	peak := 0.0
	// Seeded from audio captured before the session. Deriving the floor purely
	// from windows inside the session means the threshold is always at least
	// three times the quietest window it contains, so a session that is speech
	// from its first moment can never register any speech at all - which is
	// exactly what a wake-word session looks like.
	floor := s.seedFloor
	ring := m.capture.Ring()

	for {
		time.Sleep(checkInterval)

		m.mu.Lock()
		alive := m.current == s
		m.mu.Unlock()
		if !alive {
			return
		}

		now := ring.Written()
		if time.Since(s.started).Seconds() >= float64(captureMaxSec) {
			log.Printf("audio: reached %ds session cap, stopping", captureMaxSec)
			go m.stop("cap")
			return
		}
		if time.Since(s.started).Seconds() < minRecordTime {
			continue
		}
		// Need a full window of audio before any measurement means anything.
		if now-s.startOffset < windowBytes {
			continue
		}

		pcm, err := ring.Read(now-windowBytes, now)
		if err != nil || len(pcm) < bytesPerSample {
			continue
		}

		rms := rmsOfPCM(pcm)
		if rms > peak {
			peak = rms
		}
		if rms < floor {
			floor = rms
		}
		threshold := adaptiveThreshold(floor)

		s.evaluated.Store(true)
		storeFloat(&s.peakRms, peak)
		storeFloat(&s.floorRms, floor)
		storeFloat(&s.threshold, threshold)

		if rms < threshold {
			silenceSecs += checkInterval.Seconds()
		} else {
			silenceSecs = 0
			s.firstSpeechAt.CompareAndSwap(0, now)
			s.lastSpeechAt.Store(now)
			if s.speechWindows.Add(1) >= minSpeechWindows {
				s.sawSpeech.Store(true)
			}
		}

		// Before any speech, allow the longer grace period; afterwards the
		// short trailing silence is what ends the utterance.
		limit := silenceRequired
		if s.speechWindows.Load() == 0 {
			limit = initialSilenceAllowed
		}
		if silenceSecs >= limit {
			log.Printf("audio: %.1fs silence (rms %.0f < %.0f), auto-stopping", silenceSecs, rms, threshold)
			go m.stop("silence")
			return
		}
	}
}

// ambientFloor estimates the room's noise level from the audio immediately
// preceding a session, scanning backwards in 1s windows and taking the
// quietest. Returns +Inf when the ring holds nothing usable, which leaves the
// caller with the original behaviour of learning the floor as it goes.
func ambientFloor(ring *Ring, before int64) float64 {
	floor := math.Inf(1)
	oldest := ring.Oldest()
	for i := 0; i < ambientLookbackWindows; i++ {
		to := before - int64(i)*windowBytes
		from := to - windowBytes
		if from < oldest {
			break
		}
		pcm, err := ring.Read(from, to)
		if err != nil || len(pcm) < bytesPerSample {
			break
		}
		if r := rmsOfPCM(pcm); r < floor {
			floor = r
		}
	}
	return floor
}

func rmsOfPCM(pcm []byte) float64 {
	samples := len(pcm) / bytesPerSample
	if samples == 0 {
		return 0
	}
	var sumSq float64
	for i := 0; i < samples; i++ {
		v := int16(binary.LittleEndian.Uint16(pcm[i*2 : i*2+2]))
		sumSq += float64(v) * float64(v)
	}
	return math.Sqrt(sumSq / float64(samples))
}

func tempWavPath() (string, error) {
	tmp, err := os.CreateTemp("", "nocturne-voice-*.wav")
	if err != nil {
		return "", fmt.Errorf("failed to create temp wav: %v", err)
	}
	path := tmp.Name()
	tmp.Close()
	return path, nil
}

func (m *Manager) transcribeAndBroadcast(s *session, wavPath string) {
	defer os.Remove(wavPath)

	begin := time.Now()
	text, err := transcribe(s.provider, s.apiKey, s.lang, wavPath)
	elapsed := time.Since(begin).Milliseconds()

	if err != nil {
		log.Printf("audio: transcription failed after %dms: %v", elapsed, err)
		m.broadcastError(err.Error())
		return
	}

	log.Printf("audio: transcript via %s in %dms: %q", s.provider, elapsed, text)
	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type: "voice_transcript",
		Payload: utils.VoiceTranscriptPayload{
			Text: text, Provider: s.provider, ElapsedMs: elapsed,
		},
	})
}

func terminate(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err == nil {
		return syscall.Kill(-pgid, syscall.SIGTERM)
	}
	return cmd.Process.Signal(syscall.SIGTERM)
}
