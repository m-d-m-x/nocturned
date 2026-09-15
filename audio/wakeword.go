//go:build wakeword

package audio

// TensorFlow Lite's C runtime, cross-compiled for this device's 32-bit armhf
// userland. Header and library locations come from CGO_CFLAGS/CGO_LDFLAGS at
// build time rather than #cgo directives, so neither the TensorFlow source
// tree nor a 4MB shared object has to live in this repository.
//
// See build-wakeword.sh for the invocation.

/*
#include <stdlib.h>
#include "tensorflow/lite/c/c_api.h"
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

// Shapes are fixed by the three models and verified against them at load time;
// a mismatch means the wrong file was deployed, which is worth failing loudly
// rather than silently scoring garbage.
const (
	melInputSamples = frameSamples + 3*160 // 1760: the hop plus three frames of lookback
	melFramesPerHop = 8                    // mel frames produced per 80ms hop
	melBins         = 32
	embWindowFrames = 76 // mel frames the embedding model consumes
	embeddingDim    = 96
)

// interpreter is one loaded tflite model plus the handles it owns.
type interpreter struct {
	model  *C.TfLiteModel
	opts   *C.TfLiteInterpreterOptions
	interp *C.TfLiteInterpreter
}

func loadInterpreter(path string, shape []int32, threads int) (*interpreter, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("wakeword: model %s: %w", path, err)
	}

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	it := &interpreter{}
	it.model = C.TfLiteModelCreateFromFile(cpath)
	if it.model == nil {
		return nil, fmt.Errorf("wakeword: could not load %s", path)
	}

	it.opts = C.TfLiteInterpreterOptionsCreate()
	C.TfLiteInterpreterOptionsSetNumThreads(it.opts, C.int32_t(threads))

	it.interp = C.TfLiteInterpreterCreate(it.model, it.opts)
	if it.interp == nil {
		it.Close()
		return nil, fmt.Errorf("wakeword: could not create interpreter for %s", path)
	}

	// The melspectrogram model declares a dynamic input; without a concrete
	// shape its ops cannot be prepared and AllocateTensors fails.
	if len(shape) > 0 {
		if C.TfLiteInterpreterResizeInputTensor(
			it.interp, 0, (*C.int)(unsafe.Pointer(&shape[0])), C.int32_t(len(shape)),
		) != C.kTfLiteOk {
			it.Close()
			return nil, fmt.Errorf("wakeword: could not resize input of %s", path)
		}
	}
	if C.TfLiteInterpreterAllocateTensors(it.interp) != C.kTfLiteOk {
		it.Close()
		return nil, fmt.Errorf("wakeword: could not allocate tensors for %s", path)
	}
	return it, nil
}

func (it *interpreter) Close() {
	if it.interp != nil {
		C.TfLiteInterpreterDelete(it.interp)
		it.interp = nil
	}
	if it.opts != nil {
		C.TfLiteInterpreterOptionsDelete(it.opts)
		it.opts = nil
	}
	// The model has to outlive the interpreter built from it.
	if it.model != nil {
		C.TfLiteModelDelete(it.model)
		it.model = nil
	}
}

// run copies in, invokes, and copies out. Both slices are Go-owned; tflite
// copies through them rather than retaining them.
func (it *interpreter) run(in, out []float32) error {
	inT := C.TfLiteInterpreterGetInputTensor(it.interp, 0)
	if want, got := int(C.TfLiteTensorByteSize(inT)), len(in)*4; want != got {
		return fmt.Errorf("wakeword: input is %d bytes, model wants %d", got, want)
	}
	if C.TfLiteTensorCopyFromBuffer(inT, unsafe.Pointer(&in[0]), C.size_t(len(in)*4)) != C.kTfLiteOk {
		return fmt.Errorf("wakeword: could not copy input")
	}
	if C.TfLiteInterpreterInvoke(it.interp) != C.kTfLiteOk {
		return fmt.Errorf("wakeword: invoke failed")
	}
	outT := C.TfLiteInterpreterGetOutputTensor(it.interp, 0)
	if want, got := int(C.TfLiteTensorByteSize(outT)), len(out)*4; want != got {
		return fmt.Errorf("wakeword: output is %d bytes, model produces %d", got, want)
	}
	if C.TfLiteTensorCopyToBuffer(outT, unsafe.Pointer(&out[0]), C.size_t(len(out)*4)) != C.kTfLiteOk {
		return fmt.Errorf("wakeword: could not read output")
	}
	// in/out must stay reachable across the cgo calls above.
	runtime.KeepAlive(in)
	runtime.KeepAlive(out)
	return nil
}

// tfliteScorer runs openWakeWord's three-stage chain: raw audio becomes mel
// frames, a sliding window of mel frames becomes a speech embedding, and a
// sliding window of embeddings becomes a score.
//
// The buffering here mirrors openwakeword/utils.py::_streaming_features
// exactly - including feeding the melspectrogram 1760 samples rather than the
// 1280 of a hop, and the /10+2 transform applied to its output. Deviating from
// it produces features the classifier was never trained on, which shows up as
// a model that simply never fires.
type tfliteScorer struct {
	mel, emb, cls *interpreter

	raw      []float32 // last melInputSamples samples, as float32
	rawFill  int       // how much of raw is real audio yet
	melBuf   []float32 // embWindowFrames x melBins, oldest first
	featBuf  []float32 // featureFrames x embeddingDim, oldest first
	featFill int       // real embedding frames produced so far

	melOut []float32
	embOut []float32
	clsOut []float32
}

// NewScorer loads the three models. threads is per-interpreter; 1 is right on
// this device, where the loop runs far inside its 80ms budget already.
func NewScorer(cfg ScorerConfig) (Scorer, error) {
	threads := cfg.Threads
	if threads <= 0 {
		threads = 1
	}

	mel, err := loadInterpreter(cfg.MelspectrogramPath, []int32{1, melInputSamples}, threads)
	if err != nil {
		return nil, err
	}
	emb, err := loadInterpreter(cfg.EmbeddingPath, []int32{1, embWindowFrames, melBins, 1}, threads)
	if err != nil {
		mel.Close()
		return nil, err
	}
	cls, err := loadInterpreter(cfg.ClassifierPath, []int32{1, featureFrames, embeddingDim}, threads)
	if err != nil {
		mel.Close()
		emb.Close()
		return nil, err
	}

	s := &tfliteScorer{
		mel: mel, emb: emb, cls: cls,
		raw:     make([]float32, melInputSamples),
		melBuf:  make([]float32, embWindowFrames*melBins),
		featBuf: make([]float32, featureFrames*embeddingDim),
		melOut:  make([]float32, melFramesPerHop*melBins),
		embOut:  make([]float32, embeddingDim),
		clsOut:  make([]float32, 1),
	}
	// openWakeWord seeds its melspectrogram buffer with ones, so the first
	// embeddings see the same padding the training pipeline produced.
	for i := range s.melBuf {
		s.melBuf[i] = 1
	}
	return s, nil
}

func (s *tfliteScorer) Close() {
	s.cls.Close()
	s.emb.Close()
	s.mel.Close()
}

// Feed consumes exactly one hop of 16-bit little-endian PCM and reports the
// score. ok is false until enough audio has accumulated to fill the embedding
// window, which takes featureFrames hops (~1.3s) after start.
func (s *tfliteScorer) Feed(pcm []byte) (float32, bool, error) {
	if len(pcm) != frameSamples*bytesPerSample {
		return 0, false, fmt.Errorf("wakeword: expected %d bytes, got %d",
			frameSamples*bytesPerSample, len(pcm))
	}

	// Slide the raw window and append this hop, keeping the 480 samples of
	// lookback the melspectrogram model expects.
	copy(s.raw, s.raw[frameSamples:])
	base := melInputSamples - frameSamples
	for i := 0; i < frameSamples; i++ {
		v := int16(binary.LittleEndian.Uint16(pcm[i*2 : i*2+2]))
		s.raw[base+i] = float32(v)
	}
	if s.rawFill < melInputSamples {
		s.rawFill += frameSamples
		if s.rawFill < melInputSamples {
			return 0, false, nil // not enough lookback yet
		}
	}

	if err := s.mel.run(s.raw, s.melOut); err != nil {
		return 0, false, err
	}
	// The transform that reconciles this melspectrogram model with the native
	// TensorFlow implementation the embedding model was trained against.
	for i := range s.melOut {
		s.melOut[i] = s.melOut[i]/10 + 2
	}

	// Slide the mel window by the frames just produced.
	shift := melFramesPerHop * melBins
	copy(s.melBuf, s.melBuf[shift:])
	copy(s.melBuf[len(s.melBuf)-shift:], s.melOut)

	if err := s.emb.run(s.melBuf, s.embOut); err != nil {
		return 0, false, err
	}

	copy(s.featBuf, s.featBuf[embeddingDim:])
	copy(s.featBuf[len(s.featBuf)-embeddingDim:], s.embOut)
	if s.featFill < featureFrames {
		s.featFill++
		if s.featFill < featureFrames {
			return 0, false, nil
		}
	}

	if err := s.cls.run(s.featBuf, s.clsOut); err != nil {
		return 0, false, err
	}
	return s.clsOut[0], true, nil
}

// Reset clears the streaming state so a later detection cannot be influenced
// by audio from before it.
func (s *tfliteScorer) Reset() {
	s.rawFill = 0
	s.featFill = 0
	for i := range s.raw {
		s.raw[i] = 0
	}
	for i := range s.melBuf {
		s.melBuf[i] = 1
	}
	for i := range s.featBuf {
		s.featBuf[i] = 0
	}
}
