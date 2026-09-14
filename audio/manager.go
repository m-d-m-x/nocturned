package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
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
// no gain control (amixer exposes no capture element) and idles around RMS 6,
// so an 800 threshold meant speech never registered: every window counted as
// silence, captures auto-stopped at the 3s minimum, and the speech guard threw
// the recording away. The threshold is therefore derived from the quietest
// window actually observed in each capture, with the old constant kept as the
// ceiling for genuinely noisy environments.
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

	wavHeaderSize  = 44
	sampleRate     = 16000
	bytesPerSample = 2
	windowBytes    = int64(sampleRate * bytesPerSample) // the 1s RMS window

	// Guard band kept either side of detected speech when trimming.
	trimPadBytes = int64(0.25 * sampleRate * bytesPerSample)
	// Never trim down to something too short to be a real utterance.
	minTrimmedBytes = int64(0.6 * sampleRate * bytesPerSample)

	// Discarding a real utterance makes voice unusable; letting a silent one
	// through costs one spurious search. So a capture is only thrown away when
	// two independent signals agree there was no speech: no sustained
	// above-threshold windows, AND a peak that never rose meaningfully above
	// the noise floor.
	//
	// This factor was 6 while no speech measurements existed, which was wide
	// enough that silent captures still reached Whisper and came back as
	// hallucinated phrases. Measured since: speech registers 5-11 windows
	// (peak 303-480), silence registers 0 (peak 101-183). The window count
	// separates them cleanly, so this second signal only has to cover the
	// unobserved case where windows undercount on genuinely loud audio.
	discardPeakFactor = 20.0
)

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

// ErrNoCapture means there was no capture to act on. Callers should treat this
// as a no-op rather than a failure: with silence-driven auto-stop, the daemon
// routinely stops a session before the client gets around to asking it to.
var ErrNoCapture = errors.New("no capture in progress")

type captureSession struct {
	cmd      *exec.Cmd
	wavPath  string
	provider string
	apiKey   string
	lang     string
	started  time.Time

	// Set by the silence detector. evaluated means it got far enough to measure
	// anything at all; sawSpeech means at least one window crossed the
	// threshold. Both are read from stop(), on another goroutine.
	evaluated atomic.Bool
	sawSpeech atomic.Bool

	// float64 bits, written by the detector and read by stop().
	peakRms       atomic.Uint64
	floorRms      atomic.Uint64
	threshold     atomic.Uint64
	speechWindows atomic.Uint32

	// File offsets at the first and last tick that measured speech, used to
	// trim the silence the detector deliberately waits through.
	firstSpeechAt atomic.Int64
	lastSpeechAt  atomic.Int64
}

type Manager struct {
	mu      sync.Mutex
	current *captureSession
	wsHub   *utils.WebSocketHub
}

func NewManager(wsHub *utils.WebSocketHub) *Manager {
	return &Manager{wsHub: wsHub}
}

type StartParams struct {
	Provider string
	APIKey   string
	Lang     string
}

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

	tmp, err := os.CreateTemp("", "nocturne-voice-*.wav")
	if err != nil {
		return fmt.Errorf("failed to create temp wav: %v", err)
	}
	wavPath := tmp.Name()
	tmp.Close()
	os.Remove(wavPath)

	cmd := exec.Command(
		"arecord",
		"-q",
		"-D", captureDevice,
		"-f", captureFormat,
		"-r", captureRate,
		"-c", captureChans,
		"-t", "wav",
		"-d", fmt.Sprintf("%d", captureMaxSec),
		wavPath,
	)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start arecord: %v", err)
	}

	m.current = &captureSession{
		cmd:      cmd,
		wavPath:  wavPath,
		provider: p.Provider,
		apiKey:   p.APIKey,
		lang:     p.Lang,
		started:  time.Now(),
	}

	log.Printf("audio capture started: %s", wavPath)
	m.broadcastState("recording")
	go m.startSilenceDetector(m.current)
	return nil
}

func (m *Manager) Stop() error {
	return m.stop("manual")
}

func (m *Manager) stop(reason string) error {
	m.mu.Lock()
	session := m.current
	m.current = nil
	m.mu.Unlock()

	if session == nil {
		return ErrNoCapture
	}

	if err := terminate(session.cmd); err != nil {
		log.Printf("audio: terminate error: %v", err)
	}
	_ = session.cmd.Wait()

	st, err := os.Stat(session.wavPath)
	if err != nil || st.Size() == 0 {
		os.Remove(session.wavPath)
		log.Printf("audio: stop(%s) but recording produced no audio", reason)
		m.broadcastError("recording produced no audio")
		return fmt.Errorf("recording produced no audio")
	}

	durationMs := time.Since(session.started).Milliseconds()

	// Whisper hallucinates confident phrases ("Thank you.", "Thanks for
	// watching!") out of silence, and the client acts on whatever comes back -
	// so an accidental dial press would start playing music. If the detector ran
	// and never once measured audio above the threshold, there is nothing worth
	// transcribing: discard it and skip the API call entirely.
	//
	// evaluated gates this: a manual stop inside minRecordTime never measured
	// anything, and must not be mistaken for silence.
	peak := loadFloat(&session.peakRms)
	floor := loadFloat(&session.floorRms)
	threshold := loadFloat(&session.threshold)
	windows := session.speechWindows.Load()

	quiet := peak < floor*discardPeakFactor
	if session.evaluated.Load() && !session.sawSpeech.Load() && quiet {
		os.Remove(session.wavPath)
		log.Printf(
			"audio: discarded %dms capture - %d speech windows, peak rms %.0f vs threshold %.0f, floor %.0f",
			durationMs, windows, peak, threshold, floor,
		)
		m.wsHub.Broadcast(utils.WebSocketEvent{
			Type: "voice_state",
			Payload: utils.VoiceStatePayload{
				State:      "discarded",
				Reason:     reason,
				DurationMs: durationMs,
				Bytes:      st.Size(),
				PeakRMS:    peak,
				FloorRMS:   floor,
				Threshold:  threshold,
				Windows:    windows,
			},
		})
		m.broadcastError("no speech detected")
		return nil
	}

	uploadBytes := st.Size()
	if trimmed, err := trimWavToSpeech(
		session.wavPath,
		session.firstSpeechAt.Load(),
		session.lastSpeechAt.Load(),
	); err != nil {
		log.Printf("audio: trim failed, sending full capture: %v", err)
	} else {
		uploadBytes = trimmed
	}

	log.Printf(
		"audio: stopped (%s) after %dms, %d -> %d bytes (%s), %d speech windows, peak rms %.0f, floor %.0f, threshold %.0f",
		reason, durationMs, st.Size(), uploadBytes, durationString(uploadBytes),
		windows, peak, floor, threshold,
	)

	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type: "voice_state",
		Payload: utils.VoiceStatePayload{
			State:      "transcribing",
			Reason:     reason,
			DurationMs: durationMs,
			Bytes:      uploadBytes,
			PeakRMS:    peak,
			FloorRMS:   floor,
			Threshold:  threshold,
			Windows:    windows,
		},
	})

	go m.transcribeAndBroadcast(session)
	return nil
}

func (m *Manager) Cancel() error {
	m.mu.Lock()
	session := m.current
	m.current = nil
	m.mu.Unlock()

	if session == nil {
		return nil
	}

	_ = kill(session.cmd)
	_ = session.cmd.Wait()
	os.Remove(session.wavPath)
	m.broadcastState("cancelled")
	return nil
}

func (m *Manager) startSilenceDetector(session *captureSession) {
	chunkBytes := windowBytes
	silenceSecs := 0.0
	startTime := time.Now()
	peak := 0.0
	floor := math.Inf(1)

	for {
		time.Sleep(checkInterval)

		m.mu.Lock()
		alive := m.current == session
		m.mu.Unlock()
		if !alive {
			return
		}

		if time.Since(startTime).Seconds() < minRecordTime {
			continue
		}

		// arecord exits on its own at the -d cap, but nothing else would notice:
		// the file stops growing, so in a noisy car the RMS window stays loud and
		// silence never accumulates. Without this the session hangs forever and
		// a full-length utterance is thrown away instead of transcribed.
		if time.Since(startTime).Seconds() >= float64(captureMaxSec) {
			log.Printf("audio: reached %ds capture cap, stopping", captureMaxSec)
			go m.stop("cap")
			return
		}

		f, err := os.Open(session.wavPath)
		if err != nil {
			continue
		}
		stat, err := f.Stat()
		if err != nil || stat.Size()-int64(wavHeaderSize) < chunkBytes {
			f.Close()
			continue
		}

		offset := stat.Size() - chunkBytes
		if offset < int64(wavHeaderSize) {
			offset = int64(wavHeaderSize)
		}
		if _, err := f.Seek(offset, io.SeekStart); err != nil {
			f.Close()
			continue
		}

		buf := make([]byte, chunkBytes)
		n, _ := f.Read(buf)
		f.Close()

		if n < 2 {
			continue
		}

		samples := n / 2
		var sumSq float64
		for i := 0; i < samples; i++ {
			s := int16(binary.LittleEndian.Uint16(buf[i*2 : i*2+2]))
			sumSq += float64(s) * float64(s)
		}
		rms := math.Sqrt(sumSq / float64(samples))

		if rms > peak {
			peak = rms
		}
		if rms < floor {
			floor = rms
		}
		threshold := adaptiveThreshold(floor)

		session.evaluated.Store(true)
		storeFloat(&session.peakRms, peak)
		storeFloat(&session.floorRms, floor)
		storeFloat(&session.threshold, threshold)

		if rms < threshold {
			silenceSecs += checkInterval.Seconds()
		} else {
			silenceSecs = 0
			session.firstSpeechAt.CompareAndSwap(0, stat.Size())
			session.lastSpeechAt.Store(stat.Size())
			if session.speechWindows.Add(1) >= minSpeechWindows {
				session.sawSpeech.Store(true)
			}
		}

		if silenceSecs >= silenceRequired {
			log.Printf("audio: %.1fs silence (rms %.0f < %.0f), auto-stopping", silenceSecs, rms, threshold)
			go m.stop("silence")
			return
		}
	}
}

func (m *Manager) transcribeAndBroadcast(s *captureSession) {
	defer os.Remove(s.wavPath)

	begin := time.Now()
	text, err := transcribe(s.provider, s.apiKey, s.lang, s.wavPath)
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
			Text:      text,
			Provider:  s.provider,
			ElapsedMs: elapsed,
		},
	})
}

func (m *Manager) broadcastState(state string) {
	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type:    "voice_state",
		Payload: utils.VoiceStatePayload{State: state},
	})
}

func (m *Manager) broadcastError(msg string) {
	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type: "voice_transcript",
		Payload: utils.VoiceTranscriptPayload{
			Error: msg,
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

func kill(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err == nil {
		return syscall.Kill(-pgid, syscall.SIGKILL)
	}
	return cmd.Process.Kill()
}

