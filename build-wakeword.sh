#!/bin/bash
# Cross-build nocturned WITH wake-word detection and stage everything the
# device needs into dist-wakeword/.
#
#   wsl -d Ubuntu -- bash /mnt/d/Marshall_Files/MapleLab/Nocturne_Project/nocturned/build-wakeword.sh
#
# The default build (push.ps1) is CGO_ENABLED=0 and fully static. Detection
# needs TensorFlow Lite's C runtime, which is C, so this build enables cgo and
# links the armhf library cross-compiled earlier. The result is dynamically
# linked and needs libtensorflowlite_c.so on the device.
#
# This runs in WSL because the arm cross-compiler, the TFLite build and the
# trained models all live there. It deliberately does NOT deploy: WSL2 is in its
# own network namespace and cannot reach the device's USB network. Everything is
# staged into dist-wakeword/, which is on the Windows filesystem, and
# deploy-wakeword.ps1 copies it across from the Windows side.
set -eu

REPO="${REPO:-/mnt/d/Marshall_Files/MapleLab/Nocturne_Project/nocturned}"
OWW="${OWW:-$HOME/oww}"
GO="${GO:-$HOME/go-toolchain/go/bin/go}"
CC_ARM="${CC_ARM:-arm-linux-gnueabihf-gcc}"

TF_SRC="$OWW/tensorflow"   # provides tensorflow/lite/c/c_api.h
TF_LIB="$OWW/tflite_armhf" # provides libtensorflowlite_c.so (armhf)
OWW_MODELS="$OWW/openWakeWord/openwakeword/resources/models"
CLASSIFIER="${CLASSIFIER:-$OWW/full/hey_spotify_v3/hey_spotify_v3.tflite}"

DIST="$REPO/dist-wakeword"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ -x "$GO" ]                    || fail "no Go toolchain at $GO"
command -v "$CC_ARM" >/dev/null || fail "no $CC_ARM (apt install gcc-arm-linux-gnueabihf)"
[ -f "$TF_SRC/tensorflow/lite/c/c_api.h" ] || fail "no TFLite headers under $TF_SRC"
[ -f "$TF_LIB/libtensorflowlite_c.so" ]    || fail "no libtensorflowlite_c.so in $TF_LIB"
[ -f "$CLASSIFIER" ]            || fail "no classifier at $CLASSIFIER"
for m in melspectrogram.tflite embedding_model.tflite; do
  [ -f "$OWW_MODELS/$m" ] || fail "no $m in $OWW_MODELS"
done

echo "=== toolchain ==="
echo "  go:  $("$GO" version)"
echo "  cc:  $($CC_ARM --version | head -1)"

build() { # output package
  CGO_ENABLED=1 GOOS=linux GOARCH=arm GOARM=7 \
    CC="$CC_ARM" \
    CGO_CFLAGS="-I$TF_SRC" \
    CGO_LDFLAGS="-L$TF_LIB -ltensorflowlite_c" \
    "$GO" build -tags wakeword -ldflags "-s -w" -o "$1" "$2"
}

cd "$REPO"
mkdir -p "$DIST"

echo "=== building nocturned (cgo, armv7, -tags wakeword) ==="
build "$DIST/nocturned" .

echo "=== building wakescore diagnostic ==="
build "$DIST/wakescore" ./cmd/wakescore

echo "=== staging runtime and models ==="
cp -f "$TF_LIB/libtensorflowlite_c.so" "$DIST/"
cp -f "$OWW_MODELS/melspectrogram.tflite" "$DIST/"
cp -f "$OWW_MODELS/embedding_model.tflite" "$DIST/"
# Stable name so the daemon does not care which revision is deployed.
cp -f "$CLASSIFIER" "$DIST/hey_spotify.tflite"

echo "=== dist-wakeword ==="
TOTAL=0
for f in "$DIST"/*; do
  SZ=$(stat -c %s "$f")
  TOTAL=$((TOTAL + SZ))
  printf "  %8s  %s\n" "$(numfmt --to=iec "$SZ")" "$(basename "$f")"
done
printf "  %8s  TOTAL (device needs this much free)\n" "$(numfmt --to=iec "$TOTAL")"
echo
file -b "$DIST/nocturned" | sed 's/^/  /'
echo
echo "Now deploy from Windows (WSL cannot reach the device network):"
echo "  powershell -File D:\\Marshall_Files\\MapleLab\\Nocturne_Project\\nocturned\\deploy-wakeword.ps1"
