package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// trimWavToSpeech rewrites a capture so it contains only the speech region plus
// a short guard band either side.
//
// The detector waits silenceRequired seconds of quiet before auto-stopping, and
// that trailing silence is pure upload cost: it carries no information and
// Groq's ingest is the slow part of the round trip. firstSpeech/lastSpeech are
// the file sizes observed at the first and last tick that measured speech.
//
// RMS is computed over a trailing 1s window, so a tick that saw speech means
// the speech lies somewhere in the preceding second - hence the window is
// subtracted from the start offset.
//
// Returns the resulting file size. On any doubt it leaves the file untouched
// and returns the original size: sending a bit too much audio is harmless,
// truncating someone's words is not.
func trimWavToSpeech(path string, firstSpeech, lastSpeech int64) (int64, error) {
	st, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	total := st.Size()

	if firstSpeech <= 0 || lastSpeech <= 0 {
		return total, nil
	}

	start := firstSpeech - windowBytes - trimPadBytes
	if start < wavHeaderSize {
		start = wavHeaderSize
	}
	end := lastSpeech + trimPadBytes
	if end > total {
		end = total
	}

	// Keep offsets on 16-bit sample boundaries relative to the header.
	if (start-wavHeaderSize)%bytesPerSample != 0 {
		start++
	}
	if (end-wavHeaderSize)%bytesPerSample != 0 {
		end--
	}

	dataLen := end - start
	if dataLen < minTrimmedBytes || dataLen >= total-wavHeaderSize {
		// Nothing meaningful to remove, or the result would be suspiciously
		// short - leave it alone.
		return total, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return total, err
	}
	pcm := make([]byte, dataLen)
	if _, err := f.ReadAt(pcm, start); err != nil && err != io.EOF {
		f.Close()
		return total, err
	}
	f.Close()

	tmp := path + ".trim"
	out, err := os.Create(tmp)
	if err != nil {
		return total, err
	}
	if _, err := out.Write(wavHeader(dataLen)); err != nil {
		out.Close()
		os.Remove(tmp)
		return total, err
	}
	if _, err := out.Write(pcm); err != nil {
		out.Close()
		os.Remove(tmp)
		return total, err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return total, err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return total, err
	}

	return wavHeaderSize + dataLen, nil
}

// wavHeader builds a canonical 44-byte PCM header. arecord's own header has
// stale sizes once the process is terminated mid-write, so it is not reused.
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

func durationString(bytes int64) string {
	return fmt.Sprintf("%.1fs", float64(bytes-wavHeaderSize)/float64(sampleRate*bytesPerSample))
}
