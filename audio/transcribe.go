package audio

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
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

func transcribe(provider, apiKey, lang, wavPath string) (string, error) {
	cfg, ok := providers[provider]
	if !ok {
		return "", fmt.Errorf("unknown provider: %s", provider)
	}

	f, err := os.Open(wavPath)
	if err != nil {
		return "", fmt.Errorf("open wav: %v", err)
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

	part, err := w.CreateFormFile("file", "audio.wav")
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
