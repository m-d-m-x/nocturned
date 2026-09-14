package audio

import (
	"bytes"
	"math/rand"
	"sync"
	"testing"
)

func seq(from, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((from + i) % 251) // 251 is prime, so patterns don't align with capacity
	}
	return b
}

func TestRingReadsBackWhatWasWritten(t *testing.T) {
	r := NewRing(1024)
	in := seq(0, 600)
	r.Write(in)

	got, err := r.Read(0, 600)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, in) {
		t.Fatal("round-trip mismatch")
	}
	if r.Written() != 600 {
		t.Fatalf("written = %d, want 600", r.Written())
	}
}

func TestRingWrapsAndKeepsAbsoluteOffsets(t *testing.T) {
	r := NewRing(1000)
	// Write well past capacity in uneven chunks so writes straddle the wrap.
	total := 0
	for _, n := range []int{300, 450, 700, 120, 900} {
		r.Write(seq(total, n))
		total += n
	}

	if r.Written() != int64(total) {
		t.Fatalf("written = %d, want %d", r.Written(), total)
	}

	// The most recent 1000 bytes must still be exact.
	from := int64(total - 1000)
	got, err := r.Read(from, int64(total))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, seq(int(from), 1000)) {
		t.Fatal("wrapped content mismatch")
	}
}

func TestRingEvictsOldAudio(t *testing.T) {
	r := NewRing(500)
	r.Write(seq(0, 1200))

	if _, err := r.Read(0, 100); err != ErrEvicted {
		t.Fatalf("expected ErrEvicted for overwritten range, got %v", err)
	}
	if got := r.Oldest(); got != 700 {
		t.Fatalf("oldest = %d, want 700", got)
	}
	if _, err := r.Read(700, 1200); err != nil {
		t.Fatalf("retained range should be readable: %v", err)
	}
}

func TestRingWriteLargerThanCapacityKeepsTail(t *testing.T) {
	r := NewRing(256)
	r.Write(seq(0, 1000))

	got, err := r.Read(744, 1000)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, seq(744, 256)) {
		t.Fatal("oversized write did not leave the correct tail")
	}
}

func TestRingReadClampsToWritten(t *testing.T) {
	r := NewRing(1024)
	r.Write(seq(0, 100))

	got, err := r.Read(50, 5000)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 50 {
		t.Fatalf("expected clamp to 50 bytes, got %d", len(got))
	}
}

func TestRingTailReportsStartOffset(t *testing.T) {
	r := NewRing(1024)
	r.Write(seq(0, 900))

	got, from := r.Tail(300)
	if from != 600 {
		t.Fatalf("tail start = %d, want 600", from)
	}
	if !bytes.Equal(got, seq(600, 300)) {
		t.Fatal("tail content mismatch")
	}

	// Asking for more than exists yields what there is.
	got, from = r.Tail(5000)
	if from != 0 || len(got) != 900 {
		t.Fatalf("short tail = %d bytes from %d, want 900 from 0", len(got), from)
	}
}

// The capture goroutine writes while the detector reads; this must not race or
// tear. Run with -race on the device.
func TestRingConcurrentWriteAndRead(t *testing.T) {
	r := NewRing(8192)
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 2000; i++ {
			r.Write(seq(i, 64+rand.Intn(64)))
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 2000; i++ {
			b, _ := r.Tail(1024)
			_ = b
			if _, err := r.Read(r.Oldest(), r.Written()); err != nil && err != ErrEvicted {
				t.Errorf("unexpected read error: %v", err)
				return
			}
		}
	}()

	wg.Wait()
}
