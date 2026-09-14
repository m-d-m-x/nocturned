package audio

import (
	"io"
	"log"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

const (
	// Enough to hold a full-length capture plus pre-roll, with headroom. At
	// 16kHz mono 16-bit this is 32KB/s, so ~1.2MB total - negligible against
	// the device's memory, and it buys generous wake-word pre-roll.
	ringSeconds = 38

	// How much audio before the trigger a session keeps. A wake word is only
	// recognised after it has been spoken, and people run straight on into the
	// command, so without pre-roll the first syllables are lost.
	preRollSeconds = 1.0

	readChunkBytes = 4096
)

// Capture owns a single long-lived arecord process and pumps its raw PCM into a
// Ring. It replaces the previous design where every voice session spawned its
// own arecord writing to a temp file: wake-word detection needs the microphone
// open continuously, and a session needs access to audio from *before* it
// started, neither of which a per-session file can provide.
type Capture struct {
	ring *Ring

	mu      sync.Mutex
	cmd     *exec.Cmd
	running bool
	stop    chan struct{}
	stopped chan struct{}
}

func NewCapture() *Capture {
	return &Capture{
		ring: NewRing(ringSeconds * sampleRate * bytesPerSample),
	}
}

func (c *Capture) Ring() *Ring { return c.ring }

func (c *Capture) Running() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.running
}

// Start begins capturing and keeps capturing until Stop. The microphone must
// never silently die, so the reader loop restarts arecord on any failure.
func (c *Capture) Start() {
	c.mu.Lock()
	if c.running {
		c.mu.Unlock()
		return
	}
	c.running = true
	c.stop = make(chan struct{})
	c.stopped = make(chan struct{})
	stop, stopped := c.stop, c.stopped
	c.mu.Unlock()

	go c.run(stop, stopped)
}

func (c *Capture) Stop() {
	c.mu.Lock()
	if !c.running {
		c.mu.Unlock()
		return
	}
	c.running = false
	close(c.stop)
	stopped := c.stopped
	cmd := c.cmd
	c.mu.Unlock()

	if cmd != nil {
		_ = terminate(cmd)
	}
	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		log.Printf("audio: capture did not stop cleanly")
	}
}

func (c *Capture) run(stop <-chan struct{}, stopped chan<- struct{}) {
	defer close(stopped)

	backoff := 200 * time.Millisecond
	const maxBackoff = 5 * time.Second

	for {
		select {
		case <-stop:
			return
		default:
		}

		if err := c.pump(stop); err != nil {
			select {
			case <-stop:
				return
			default:
			}
			log.Printf("audio: capture ended (%v), restarting in %s", err, backoff)
			select {
			case <-stop:
				return
			case <-time.After(backoff):
			}
			if backoff < maxBackoff {
				backoff *= 2
			}
			continue
		}
		backoff = 200 * time.Millisecond
	}
}

// pump runs one arecord until it exits or Stop is called.
func (c *Capture) pump(stop <-chan struct{}) error {
	// Raw PCM to stdout rather than a wav file: headers would only get in the
	// way, and the ring records where every byte sits without them.
	cmd := exec.Command(
		"arecord",
		"-q",
		"-D", captureDevice,
		"-f", captureFormat,
		"-r", captureRate,
		"-c", captureChans,
		"-t", "raw",
	)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}

	c.mu.Lock()
	c.cmd = cmd
	c.mu.Unlock()

	log.Printf("audio: continuous capture started (pid %d)", cmd.Process.Pid)

	buf := make([]byte, readChunkBytes)
	for {
		n, readErr := stdout.Read(buf)
		if n > 0 {
			c.ring.Write(buf[:n])
		}
		if readErr != nil {
			_ = terminate(cmd)
			_ = cmd.Wait()
			c.mu.Lock()
			c.cmd = nil
			c.mu.Unlock()
			if readErr == io.EOF {
				return io.EOF
			}
			return readErr
		}

		select {
		case <-stop:
			_ = terminate(cmd)
			_ = cmd.Wait()
			c.mu.Lock()
			c.cmd = nil
			c.mu.Unlock()
			return nil
		default:
		}
	}
}

// PreRollBytes is how far before a trigger a session reaches back.
func PreRollBytes() int64 {
	return int64(preRollSeconds * sampleRate * bytesPerSample)
}
