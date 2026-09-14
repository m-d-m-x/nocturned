package audio

import (
	"errors"
	"sync"
)

// ErrEvicted means the requested range has already been overwritten. Callers
// should treat it as "that audio is gone" rather than as a failure.
var ErrEvicted = errors.New("requested audio has been overwritten")

// Ring is a fixed-size circular buffer of raw PCM fed by the always-on capture.
//
// Positions are expressed as absolute byte offsets since capture began, not as
// indices into the backing array. That is what makes the wake-word flow
// tractable: a detector can say "the phrase ended at offset N", and a session
// can later ask for "N minus two seconds" without either side reasoning about
// where the write cursor happens to be.
type Ring struct {
	mu       sync.Mutex
	buf      []byte
	capacity int64
	written  int64 // total bytes ever written; only ever increases
}

func NewRing(capacity int) *Ring {
	if capacity <= 0 {
		panic("audio: ring capacity must be positive")
	}
	return &Ring{buf: make([]byte, capacity), capacity: int64(capacity)}
}

func (r *Ring) Capacity() int64 { return r.capacity }

// Written is the absolute offset one past the most recent byte.
func (r *Ring) Written() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.written
}

// Write never blocks and never fails; once full it overwrites the oldest audio.
// Dropping stale audio is always preferable to stalling the capture pipe, which
// would make arecord back up and desynchronise the stream.
func (r *Ring) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	n := int64(len(p))
	if n == 0 {
		return 0, nil
	}

	// A write larger than the whole buffer can only leave its tail. That tail
	// still has to land at its modular position: Read maps absolute offset X to
	// buf[X%capacity], so copying to buf[0] would silently misalign everything.
	if n >= r.capacity {
		p = p[n-r.capacity:]
		start := int((r.written + n) % r.capacity)
		copied := copy(r.buf[start:], p)
		if copied < len(p) {
			copy(r.buf, p[copied:])
		}
		r.written += n
		return int(n), nil
	}

	start := int(r.written % r.capacity)
	copied := copy(r.buf[start:], p)
	if copied < len(p) {
		copy(r.buf, p[copied:])
	}
	r.written += n
	return int(n), nil
}

// oldest returns the lowest offset still held. Caller must hold the lock.
func (r *Ring) oldest() int64 {
	if r.written <= r.capacity {
		return 0
	}
	return r.written - r.capacity
}

// Oldest is the lowest absolute offset still retained.
func (r *Ring) Oldest() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.oldest()
}

// Read copies the absolute range [from, to) out of the ring. It returns
// ErrEvicted if any of that range has already been overwritten, and clamps `to`
// to what has actually been written so far.
func (r *Ring) Read(from, to int64) ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if to > r.written {
		to = r.written
	}
	if from < 0 {
		from = 0
	}
	if from < r.oldest() {
		return nil, ErrEvicted
	}
	if to <= from {
		return []byte{}, nil
	}

	out := make([]byte, to-from)
	start := int(from % r.capacity)
	copied := copy(out, r.buf[start:])
	if copied < len(out) {
		copy(out[copied:], r.buf)
	}
	return out, nil
}

// Tail copies the most recent n bytes and reports the offset they start at.
// Fewer bytes are returned if capture has not produced n yet.
func (r *Ring) Tail(n int64) ([]byte, int64) {
	r.mu.Lock()
	from := r.written - n
	if from < r.oldest() {
		from = r.oldest()
	}
	to := r.written
	r.mu.Unlock()

	out, err := r.Read(from, to)
	if err != nil {
		return []byte{}, to
	}
	return out, from
}
