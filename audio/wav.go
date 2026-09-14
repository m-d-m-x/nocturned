package audio

import (
	"encoding/binary"
	"fmt"
	"os"
)

// wavHeader builds a canonical 44-byte PCM header for a run of raw samples.
func wavHeader(dataLen int64) []byte {
	h := make([]byte, wavHeaderSize)
	copy(h[0:4], "RIFF")
	binary.LittleEndian.PutUint32(h[4:8], uint32(36+dataLen))
	copy(h[8:12], "WAVE")
	copy(h[12:16], "fmt ")
	binary.LittleEndian.PutUint32(h[16:20], 16)
	binary.LittleEndian.PutUint16(h[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(h[22:24], 1) // mono
	binary.LittleEndian.PutUint32(h[24:28], sampleRate)
	binary.LittleEndian.PutUint32(h[28:32], sampleRate*bytesPerSample)
	binary.LittleEndian.PutUint16(h[32:34], bytesPerSample)
	binary.LittleEndian.PutUint16(h[34:36], 8*bytesPerSample)
	copy(h[36:40], "data")
	binary.LittleEndian.PutUint32(h[40:44], uint32(dataLen))
	return h
}

// writeWav writes raw PCM out as a wav file for upload.
func writeWav(path string, pcm []byte) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if _, err := f.Write(wavHeader(int64(len(pcm)))); err != nil {
		f.Close()
		os.Remove(path)
		return err
	}
	if _, err := f.Write(pcm); err != nil {
		f.Close()
		os.Remove(path)
		return err
	}
	return f.Close()
}

func durationString(pcmBytes int64) string {
	return fmt.Sprintf("%.1fs", float64(pcmBytes)/float64(sampleRate*bytesPerSample))
}

// speechRange narrows a session's full offset range down to the speech within
// it, with a guard band either side.
//
// With a ring buffer this replaces the old approach of rewriting the wav file
// on disk: the speech boundaries are already absolute offsets, so trimming is
// just choosing which bytes to read. RMS is measured over a trailing 1s window,
// so a tick that saw speech means the speech lies somewhere in the preceding
// second - hence the window is subtracted from the start.
//
// Returns the original range unchanged whenever anything looks doubtful.
// Sending slightly too much audio is harmless; truncating words is not.
func speechRange(sessionStart, sessionEnd, firstSpeech, lastSpeech int64) (int64, int64) {
	if firstSpeech <= 0 || lastSpeech <= 0 || lastSpeech < firstSpeech {
		return sessionStart, sessionEnd
	}

	start := firstSpeech - windowBytes - trimPadBytes
	if start < sessionStart {
		start = sessionStart
	}
	end := lastSpeech + trimPadBytes
	if end > sessionEnd {
		end = sessionEnd
	}

	// Keep both ends on 16-bit sample boundaries relative to the session start.
	if (start-sessionStart)%bytesPerSample != 0 {
		start++
	}
	if (end-sessionStart)%bytesPerSample != 0 {
		end--
	}

	if end-start < minTrimmedBytes {
		return sessionStart, sessionEnd
	}
	return start, end
}
