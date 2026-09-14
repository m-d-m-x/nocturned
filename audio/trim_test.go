package audio

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// buildCapture writes a wav of leadSec silence + speechSec of tone + tailSec
// silence, and returns its path plus the file offsets a detector tick would
// have observed at the first and last speech window.
func buildCapture(t *testing.T, leadSec, speechSec, tailSec float64) (string, int64, int64) {
	t.Helper()

	bytesFor := func(sec float64) int64 {
		n := int64(sec * sampleRate * bytesPerSample)
		return n - n%bytesPerSample
	}

	lead, speech, tail := bytesFor(leadSec), bytesFor(speechSec), bytesFor(tailSec)
	pcm := make([]byte, lead+speech+tail)
	for i := lead; i < lead+speech; i += bytesPerSample {
		binary.LittleEndian.PutUint16(pcm[i:i+2], 8000)
	}

	path := filepath.Join(t.TempDir(), "cap.wav")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	f.Write(wavHeader(int64(len(pcm))))
	f.Write(pcm)
	f.Close()

	// The detector only sees a speech window once a full trailing second of
	// audio contains the tone, and keeps seeing it for a second after it ends.
	first := wavHeaderSize + lead + windowBytes
	last := wavHeaderSize + lead + speech
	return path, first, last
}

func TestTrimRemovesTrailingSilence(t *testing.T) {
	path, first, last := buildCapture(t, 1.0, 3.0, 1.4)
	before, _ := os.Stat(path)

	after, err := trimWavToSpeech(path, first, last)
	if err != nil {
		t.Fatalf("trim: %v", err)
	}
	if after >= before.Size() {
		t.Fatalf("expected trim, got %d from %d", after, before.Size())
	}

	st, _ := os.Stat(path)
	if st.Size() != after {
		t.Fatalf("reported %d but file is %d", after, st.Size())
	}

	// The whole utterance must survive: 3s of speech plus guard bands.
	minKeep := int64(3.0*sampleRate*bytesPerSample) + wavHeaderSize
	if after < minKeep {
		t.Fatalf("trimmed too aggressively: %d < %d (%s)", after, minKeep, durationString(after))
	}
	t.Logf("%d -> %d bytes (%s)", before.Size(), after, durationString(after))
}

func TestTrimWritesValidHeader(t *testing.T) {
	path, first, last := buildCapture(t, 1.0, 2.0, 1.4)
	if _, err := trimWavToSpeech(path, first, last); err != nil {
		t.Fatal(err)
	}

	raw, _ := os.ReadFile(path)
	if string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		t.Fatal("not a RIFF/WAVE file")
	}
	declared := binary.LittleEndian.Uint32(raw[40:44])
	if int64(declared) != int64(len(raw))-wavHeaderSize {
		t.Fatalf("data chunk says %d, file holds %d", declared, int64(len(raw))-wavHeaderSize)
	}
	if got := binary.LittleEndian.Uint32(raw[24:28]); got != sampleRate {
		t.Fatalf("sample rate %d", got)
	}
}

func TestTrimLeavesFileAloneWhenNoSpeechRecorded(t *testing.T) {
	path, _, _ := buildCapture(t, 1.0, 2.0, 1.4)
	before, _ := os.Stat(path)

	after, err := trimWavToSpeech(path, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if after != before.Size() {
		t.Fatalf("file should be untouched, got %d want %d", after, before.Size())
	}
}

func TestTrimLeavesFileAloneWhenResultTooShort(t *testing.T) {
	// Speech far shorter than minTrimmedBytes - trimming would produce a stub.
	path, first, last := buildCapture(t, 0.1, 0.05, 0.1)
	before, _ := os.Stat(path)

	after, err := trimWavToSpeech(path, first, last)
	if err != nil {
		t.Fatal(err)
	}
	if after != before.Size() {
		t.Fatalf("expected no trim, got %d want %d", after, before.Size())
	}
}
