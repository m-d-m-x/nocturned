package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"sync"
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

type captureSession struct {
	cmd      *exec.Cmd
	wavPath  string
	provider string
	apiKey   string
	lang     string
	started  time.Time
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
	go m.startSilenceDetector(m.current)
	return nil
}

func (m *Manager) Stop() error {
	m.mu.Lock()
	session := m.current
	m.current = nil
	m.mu.Unlock()

	if session == nil {
		return fmt.Errorf("no capture in progress")
	}

	if err := terminate(session.cmd); err != nil {
		log.Printf("audio: terminate error: %v", err)
	}
	_ = session.cmd.Wait()

	if st, err := os.Stat(session.wavPath); err != nil || st.Size() == 0 {
		os.Remove(session.wavPath)
		m.broadcastError("recording produced no audio")
		return fmt.Errorf("recording produced no audio")
	}

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
	return nil
}

func (m *Manager) startSilenceDetector(session *captureSession) {
	const (
		checkInterval    = 200 * time.Millisecond
		silenceThreshold = 800.0 // RMS (0–32767); tune up for noisy environments
		silenceRequired  = 2.0   // seconds of continuous silence before auto-stop
		minRecordTime    = 1.0   // seconds before silence detection activates
		wavHeaderSize    = 44
		sampleRate       = 16000
		bytesPerSample   = 2
	)
	chunkBytes := int64(sampleRate * bytesPerSample) // 1s window
	silenceSecs := 0.0
	startTime := time.Now()

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

		f, err := os.Open(session.wavPath)
		if err != nil {
			continue
		}
		stat, err := f.Stat()
		if err != nil || stat.Size() <= int64(wavHeaderSize) {
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

		if rms < silenceThreshold {
			silenceSecs += checkInterval.Seconds()
		} else {
			silenceSecs = 0
		}

		if silenceSecs >= silenceRequired {
			log.Printf("audio: %.1fs silence detected (RMS %.0f), auto-stopping", silenceSecs, rms)
			go m.Stop()
			return
		}
	}
}

func (m *Manager) transcribeAndBroadcast(s *captureSession) {
	defer os.Remove(s.wavPath)

	text, err := transcribe(s.provider, s.apiKey, s.lang, s.wavPath)
	if err != nil {
		log.Printf("audio: transcription failed: %v", err)
		m.broadcastError(err.Error())
		return
	}

	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type: "voice_transcript",
		Payload: utils.VoiceTranscriptPayload{
			Text: text,
		},
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

