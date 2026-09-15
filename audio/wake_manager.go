package audio

import (
	"fmt"
	"log"

	"github.com/usenocturne/nocturned/utils"
)

// WakeParams arms detection. The credentials are the same ones a dial-driven
// session uses; the daemon needs them up front because a detected phrase has
// to open a session immediately, with no round trip to the client.
type WakeParams struct {
	Models    ScorerConfig
	Threshold float32
	StartParams
}

// EnableWake loads the models and begins scoring the live stream.
//
// Detection reads the same Ring the sessions read. Nothing about the
// microphone changes: capture was already continuous, and this simply adds a
// second consumer of the stream it was already producing.
//
// Arming is declarative rather than a toggle: enabling while already enabled
// is a credential refresh, not an error. The client re-arms whenever its
// websocket reconnects, which happens routinely, and rejecting that left the
// UI logging failures for something that was in fact working.
func (m *Manager) EnableWake(p WakeParams) error {
	if p.APIKey == "" {
		return fmt.Errorf("apiKey is required")
	}
	if p.Provider == "" {
		p.Provider = "groq"
	}
	if !m.capture.Running() {
		return fmt.Errorf("microphone is not capturing")
	}

	cfg := DefaultWakeConfig()
	if p.Threshold > 0 {
		cfg.Threshold = p.Threshold
	}
	cfg.OnDetect = m.onWakeDetected

	m.mu.Lock()
	if m.wake != nil {
		if m.wakeModels == p.Models && m.wakeThreshold == cfg.Threshold {
			// Same configuration: keep the running detector and its warmed-up
			// audio state, and just take the refreshed credentials.
			m.wakeParams = p.StartParams
			m.mu.Unlock()
			return nil
		}
		// Different models or threshold, so the detector has to be replaced.
		// DisableWake takes the lock itself and must stop the loop outside it.
		m.mu.Unlock()
		m.DisableWake()
	} else {
		m.mu.Unlock()
	}

	// Loading three models takes long enough that it should not block sessions.
	scorer, err := NewScorer(p.Models)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.wake != nil {
		// Another arm landed while we were loading; keep the one that won.
		scorer.Close()
		m.wakeParams = p.StartParams
		return nil
	}

	m.wakeParams = p.StartParams
	m.wakeModels = p.Models
	m.wakeThreshold = cfg.Threshold
	m.wake = newDetector(m.capture.Ring(), scorer, cfg)
	m.wake.Start()

	log.Printf("wakeword: enabled (threshold %.3f, classifier %s)",
		cfg.Threshold, p.Models.ClassifierPath)
	return nil
}

// DisableWake stops detection and releases the models.
func (m *Manager) DisableWake() {
	m.mu.Lock()
	d := m.wake
	m.wake = nil
	m.wakeParams = StartParams{}
	m.wakeModels = ScorerConfig{}
	m.wakeThreshold = 0
	m.mu.Unlock()

	if d == nil {
		return
	}
	// Stop outside the lock: the detector's loop takes m.mu when it fires.
	d.Stop()
	log.Printf("wakeword: disabled")
}

// WakeEnabled reports whether detection is currently running.
func (m *Manager) WakeEnabled() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.wake != nil
}

// onWakeDetected opens a session starting where the phrase ended.
//
// A detection while a session is already running is ignored rather than
// treated as an error: the user is mid-command, and the phrase was almost
// certainly picked up from their own speech.
func (m *Manager) onWakeDetected(offset int64, score float32) {
	m.mu.Lock()
	if m.current != nil {
		m.mu.Unlock()
		log.Printf("wakeword: ignoring detection, session already active")
		return
	}
	params := m.wakeParams
	if params.APIKey == "" {
		m.mu.Unlock()
		log.Printf("wakeword: detected but no credentials armed")
		return
	}
	err := m.startLocked(params, offset)
	m.mu.Unlock()

	if err != nil {
		log.Printf("wakeword: could not start session: %v", err)
		m.broadcastError(fmt.Sprintf("wake word session failed: %v", err))
		return
	}

	// Sent after the session is open so the client can show that it is
	// listening. startLocked has already broadcast "recording"; this carries
	// the detection score alongside it.
	m.wsHub.Broadcast(utils.WebSocketEvent{
		Type:    "voice_state",
		Payload: utils.VoiceStatePayload{State: "wake", Score: score},
	})
}

// WakeModelsFrom builds a ScorerConfig from a directory laid out the way
// deploy-wakeword.sh installs it.
func WakeModelsFrom(dir string) ScorerConfig {
	return ScorerConfig{
		MelspectrogramPath: dir + "/melspectrogram.tflite",
		EmbeddingPath:      dir + "/embedding_model.tflite",
		ClassifierPath:     dir + "/hey_spotify.tflite",
		Threads:            1,
	}
}
