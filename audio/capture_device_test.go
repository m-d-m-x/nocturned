package audio

import (
	"encoding/binary"
	"math"
	"os/exec"
	"testing"
	"time"
)

// These exercise the real microphone and only run where arecord exists, i.e. on
// the device. Run explicitly: ./audiotest -test.run TestCaptureDevice -test.v
func requireArecord(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("arecord"); err != nil {
		t.Skip("arecord not present; not a capture device")
	}
}

func rmsOf(pcm []byte) float64 {
	samples := len(pcm) / bytesPerSample
	if samples == 0 {
		return 0
	}
	var sumSq float64
	for i := 0; i < samples; i++ {
		s := int16(binary.LittleEndian.Uint16(pcm[i*2 : i*2+2]))
		sumSq += float64(s) * float64(s)
	}
	return math.Sqrt(sumSq / float64(samples))
}

func TestCaptureDeviceFillsRingAtRealtime(t *testing.T) {
	requireArecord(t)

	c := NewCapture()
	c.Start()
	defer c.Stop()

	// Give arecord a moment to open the device before timing anything.
	time.Sleep(700 * time.Millisecond)
	start := c.Ring().Written()
	if start == 0 {
		t.Fatal("no audio captured in the first 700ms")
	}

	const window = 2 * time.Second
	time.Sleep(window)
	produced := c.Ring().Written() - start

	expected := int64(window.Seconds()) * sampleRate * bytesPerSample
	ratio := float64(produced) / float64(expected)
	t.Logf("captured %d bytes in %s (expected ~%d, ratio %.2f)", produced, window, expected, ratio)

	// Allow slack for scheduling, but catch a stream running at the wrong rate
	// or a pipe that has stalled.
	if ratio < 0.80 || ratio > 1.20 {
		t.Fatalf("capture rate off: %.2fx realtime", ratio)
	}
}

func TestCaptureDeviceProducesRealAudio(t *testing.T) {
	requireArecord(t)

	c := NewCapture()
	c.Start()
	defer c.Stop()
	time.Sleep(2 * time.Second)

	pcm, from := c.Ring().Tail(int64(sampleRate * bytesPerSample)) // last 1s
	if len(pcm) == 0 {
		t.Fatal("ring produced no audio")
	}

	rms := rmsOf(pcm)
	t.Logf("last 1s: %d bytes from offset %d, rms %.1f", len(pcm), from, rms)

	// A dead pipe or a muted device yields exact zeros. Room tone on this mic
	// measured RMS 7-34, so anything non-zero proves the stream is live.
	if rms == 0 {
		t.Fatal("captured audio is all zeros - stream is not live")
	}
}

func TestCaptureDeviceStopsAndRestarts(t *testing.T) {
	requireArecord(t)

	c := NewCapture()
	c.Start()
	time.Sleep(1200 * time.Millisecond)
	first := c.Ring().Written()
	c.Stop()

	if c.Running() {
		t.Fatal("still running after Stop")
	}
	afterStop := c.Ring().Written()
	time.Sleep(600 * time.Millisecond)
	if c.Ring().Written() != afterStop {
		t.Fatal("ring still growing after Stop")
	}

	// The mic must be usable again - a leaked arecord would hold the device.
	c.Start()
	defer c.Stop()
	time.Sleep(1200 * time.Millisecond)
	if c.Ring().Written() <= afterStop {
		t.Fatal("capture did not resume after restart")
	}
	t.Logf("first run %d bytes, resumed to %d", first, c.Ring().Written())
}
