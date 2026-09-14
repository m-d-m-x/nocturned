package audio

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

const (
	secBytes = int64(sampleRate * bytesPerSample)
)

// A session running from offset 100000, six seconds long, with speech detected
// between the second and fourth second.
func TestSpeechRangeTrimsTrailingSilence(t *testing.T) {
	start := int64(100000)
	end := start + 6*secBytes
	first := start + 2*secBytes
	last := start + 4*secBytes

	from, to := speechRange(start, end, first, last)

	if from <= start {
		t.Fatalf("expected leading silence trimmed, from=%d start=%d", from, start)
	}
	if to >= end {
		t.Fatalf("expected trailing silence trimmed, to=%d end=%d", to, end)
	}
	// The whole utterance must survive: speech began somewhere in the second
	// before `first`, so the window must be included.
	if from > first-windowBytes {
		t.Fatalf("start %d clips the window before first speech at %d", from, first)
	}
	if to < last {
		t.Fatalf("end %d clips speech that ran to %d", to, last)
	}
	t.Logf("%s session -> %s kept", durationString(end-start), durationString(to-from))
}

func TestSpeechRangeKeepsSampleAlignment(t *testing.T) {
	start := int64(12345) // deliberately odd
	end := start + 6*secBytes
	from, to := speechRange(start, end, start+2*secBytes+1, start+4*secBytes+1)

	if (from-start)%bytesPerSample != 0 {
		t.Fatalf("from is not sample aligned: %d", from-start)
	}
	if (to-start)%bytesPerSample != 0 {
		t.Fatalf("to is not sample aligned: %d", to-start)
	}
}

func TestSpeechRangeUntouchedWhenNoSpeechRecorded(t *testing.T) {
	start := int64(500)
	end := start + 4*secBytes

	from, to := speechRange(start, end, 0, 0)
	if from != start || to != end {
		t.Fatalf("expected full range, got [%d,%d) want [%d,%d)", from, to, start, end)
	}
}

func TestSpeechRangeUntouchedWhenResultTooShort(t *testing.T) {
	start := int64(0)
	end := 4 * secBytes
	// Speech markers so close together that trimming would yield a stub.
	from, to := speechRange(start, end, secBytes, secBytes+10)

	if to-from < minTrimmedBytes {
		t.Fatalf("produced a %d byte stub, should have refused", to-from)
	}
}

func TestSpeechRangeNeverExceedsSession(t *testing.T) {
	start := int64(1000)
	end := start + 3*secBytes
	// Speech right at the edges - padding must not run past the session.
	from, to := speechRange(start, end, start, end)

	if from < start || to > end {
		t.Fatalf("range [%d,%d) escapes session [%d,%d)", from, to, start, end)
	}
}

func TestWriteWavProducesValidHeader(t *testing.T) {
	pcm := make([]byte, 3200)
	for i := range pcm {
		pcm[i] = byte(i % 251)
	}
	path := filepath.Join(t.TempDir(), "out.wav")
	if err := writeWav(path, pcm); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		t.Fatal("not a RIFF/WAVE file")
	}
	if got := binary.LittleEndian.Uint32(raw[40:44]); int(got) != len(pcm) {
		t.Fatalf("data chunk says %d, wrote %d", got, len(pcm))
	}
	if got := binary.LittleEndian.Uint32(raw[24:28]); got != sampleRate {
		t.Fatalf("sample rate %d", got)
	}
	if len(raw) != wavHeaderSize+len(pcm) {
		t.Fatalf("file is %d bytes, expected %d", len(raw), wavHeaderSize+len(pcm))
	}
}
