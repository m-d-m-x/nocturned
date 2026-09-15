//go:build !wakeword

package audio

// NewScorer reports that detection is unavailable. Everything else in the
// daemon behaves normally, so a build without the tag is still fully usable
// via the dial-triggered voice flow.
func NewScorer(cfg ScorerConfig) (Scorer, error) {
	return nil, ErrWakeUnsupported
}
