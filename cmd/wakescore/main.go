// Command wakescore runs the wake-word detector over wav files and prints the
// scores, so the Go feature pipeline can be checked against openWakeWord's
// Python implementation on identical audio.
//
// The three-stage chain is easy to get subtly wrong - the melspectrogram wants
// 1760 samples rather than the 1280 of a hop, its output needs a /10+2
// transform, and the embedding window slides 8 mel frames at a time. Any of
// those being off yields a model that loads, runs, and never fires. This exists
// to prove that did not happen.
//
//	wakescore -models /etc/nocturne/wakeword clip.wav [clip.wav ...]
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/usenocturne/nocturned/audio"
)

// readWavPCM returns the samples of a 16-bit mono wav as raw little-endian
// bytes, locating the data chunk rather than assuming a 44-byte header.
func readWavPCM(path string) ([]byte, int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("%s is not a RIFF/WAVE file", path)
	}

	sampleRate := 0
	for off := 12; off+8 <= len(raw); {
		id := string(raw[off : off+4])
		size := int(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		body := off + 8
		if body+size > len(raw) {
			size = len(raw) - body
		}
		switch id {
		case "fmt ":
			if size >= 16 {
				sampleRate = int(binary.LittleEndian.Uint32(raw[body+4 : body+8]))
			}
		case "data":
			return raw[body : body+size], sampleRate, nil
		}
		off = body + size
		if size%2 == 1 {
			off++ // chunks are word-aligned
		}
	}
	return nil, sampleRate, fmt.Errorf("%s has no data chunk", path)
}

func main() {
	dir := flag.String("models", "/etc/nocturne/wakeword", "directory holding the three tflite models")
	verbose := flag.Bool("v", false, "print every hop score, not just the peak")
	threshold := flag.Float64("threshold", 0.99, "score at or above which a clip counts as detected")
	flag.Parse()

	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: wakescore [-models DIR] [-v] clip.wav ...")
		os.Exit(2)
	}

	scorer, err := audio.NewScorer(audio.WakeModelsFrom(*dir))
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		os.Exit(1)
	}
	defer scorer.Close()

	const hopBytes = 1280 * 2
	detected, total := 0, 0
	var peaks []float64

	for _, path := range flag.Args() {
		pcm, sr, err := readWavPCM(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, "  skip:", err)
			continue
		}
		if sr != 0 && sr != 16000 {
			fmt.Fprintf(os.Stderr, "  skip: %s is %d Hz, need 16000\n", path, sr)
			continue
		}

		// Lead-in silence so the first hops have the lookback the pipeline
		// expects, matching how the ring always has preceding audio.
		lead := make([]byte, hopBytes*8)
		pcm = append(lead, pcm...)

		scorer.Reset()
		peak := 0.0
		for off := 0; off+hopBytes <= len(pcm); off += hopBytes {
			score, ok, err := scorer.Feed(pcm[off : off+hopBytes])
			if err != nil {
				fmt.Fprintln(os.Stderr, "  error:", err)
				break
			}
			if !ok {
				continue
			}
			if *verbose {
				fmt.Printf("    %6.2fs  %.4f\n", float64(off)/(16000*2), score)
			}
			if float64(score) > peak {
				peak = float64(score)
			}
		}

		total++
		peaks = append(peaks, peak)
		hit := peak >= *threshold
		if hit {
			detected++
		}
		mark := " "
		if hit {
			mark = "*"
		}
		fmt.Printf("  %s %-44s peak=%.4f\n", mark, filepath.Base(path), peak)
	}

	if total == 0 {
		os.Exit(1)
	}
	sort.Float64s(peaks)
	fmt.Printf("\n  %d/%d detected at threshold %.2f (%.1f%%)\n",
		detected, total, *threshold, 100*float64(detected)/float64(total))
	fmt.Printf("  peak score median=%.4f min=%.4f max=%.4f\n",
		peaks[len(peaks)/2], peaks[0], peaks[len(peaks)-1])
}
