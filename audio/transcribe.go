package audio

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

type providerConfig struct {
	endpoint string
	model    string
}

var providers = map[string]providerConfig{
	"groq": {
		endpoint: "https://api.groq.com/openai/v1/audio/transcriptions",
		model:    "whisper-large-v3-turbo",
	},
	"openai": {
		endpoint: "https://api.openai.com/v1/audio/transcriptions",
		model:    "whisper-1",
	},
}

type transcriptionResponse struct {
	Text string `json:"text"`
}

type errorEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// Groq's upload path is the bottleneck, not its inference: measured from the
// device, 192KB of WAV took 3.7s while the same audio as a 24KB MP3 took 1.1s
// (a plain 180KB POST to an unrelated host took 0.44s, so the link is fine).
// lame ships on the device, so captures are compressed before upload. Speech at
// 32kbps mono is well within what Whisper handles.
func compressForUpload(wavPath string) (string, func()) {
	noop := func() {}

	if _, err := exec.LookPath("lame"); err != nil {
		return wavPath, noop
	}

	mp3Path := strings.TrimSuffix(wavPath, ".wav") + ".mp3"
	cmd := exec.Command("lame", "--quiet", "-m", "m", "-b", "32", wavPath, mp3Path)
	if err := cmd.Run(); err != nil {
		log.Printf("audio: lame encode failed, sending wav: %v", err)
		os.Remove(mp3Path)
		return wavPath, noop
	}

	st, err := os.Stat(mp3Path)
	if err != nil || st.Size() == 0 {
		os.Remove(mp3Path)
		return wavPath, noop
	}

	return mp3Path, func() { os.Remove(mp3Path) }
}

func transcribe(provider, apiKey, lang, wavPath string) (string, error) {
	cfg, ok := providers[provider]
	if !ok {
		return "", fmt.Errorf("unknown provider: %s", provider)
	}

	uploadPath, cleanup := compressForUpload(wavPath)
	defer cleanup()

	f, err := os.Open(uploadPath)
	if err != nil {
		return "", fmt.Errorf("open audio: %v", err)
	}
	defer f.Close()

	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)

	if err := w.WriteField("model", cfg.model); err != nil {
		return "", err
	}
	if err := w.WriteField("response_format", "json"); err != nil {
		return "", err
	}
	if lang != "" {
		if err := w.WriteField("language", lang); err != nil {
			return "", err
		}
	}

	filename := "audio.wav"
	if strings.HasSuffix(uploadPath, ".mp3") {
		filename = "audio.mp3"
	}

	part, err := w.CreateFormFile("file", filename)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(part, f); err != nil {
		return "", err
	}
	if err := w.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", cfg.endpoint, body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", w.FormDataContentType())

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("upload: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var env errorEnvelope
		if err := json.Unmarshal(respBody, &env); err == nil && env.Error.Message != "" {
			return "", fmt.Errorf("%s: %s", provider, env.Error.Message)
		}
		return "", fmt.Errorf("%s returned %s: %s", provider, resp.Status, truncate(string(respBody), 200))
	}

	var parsed transcriptionResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("parse response: %v", err)
	}
	text := strings.TrimSpace(parsed.Text)
	if text == "" {
		return "", fmt.Errorf("empty transcription")
	}
	return text, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
