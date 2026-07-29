#!/bin/bash
# Test script for MatchaTTSInfer
# Run from repo root: ./test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BIN="$BUILD_DIR/MatchaTTSInfer"

# Models (adjust paths if needed)
MATCHA_MODEL="/home/oem/Basir/TTS/Matcha/Trained/onnx/zahra/zahra-22050-5.onnx"
VOCODER_MODEL="/home/oem/Basir/TTS/vocos22.onnx"
TOKENS="/home/oem/Basir/TTS/Matcha/Matcha-TTS/configs/tokens/tokens_sherpa_with_fa.txt"
ESPEAK_DATA="/home/oem/Basir/TTS/Piper/piper_linux_x86_64/piper/espeak-ng-data"

# Test cases
TESTS=(
    "سلام دنیا"
    "Hello world"
    "امروز هوا خیلی خوبه"
    "This is a test"
    "من فارسی و English حرف میزنم"
)

echo "========================================"
echo " MatchaTTSInfer Test Suite"
echo "========================================"
echo ""

# Build if needed
if [ ! -f "$BIN" ]; then
    echo "[BUILD] Compiling..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. > /dev/null
    cmake --build . -j$(nproc) > /dev/null
    cd "$SCRIPT_DIR"
    echo "[BUILD] Done."
    echo ""
fi

echo "[INFO] Binary: $BIN"
echo "[INFO] Model:  $MATCHA_MODEL"
echo ""

PASSED=0
FAILED=0

for TEXT in "${TESTS[@]}"; do
    OUTPUT="/tmp/matcha_test_$$.wav"
    echo -n "[TEST] \"$TEXT\" ... "

    if "$BIN" \
        --text "$TEXT" \
        --matcha-model "$MATCHA_MODEL" \
        --vocoder-model "$VOCODER_MODEL" \
        --tokens "$TOKENS" \
        --espeak-data "$ESPEAK_DATA" \
        --output "$OUTPUT" \
        > /dev/null 2>&1; then

        if [ -f "$OUTPUT" ] && [ "$(stat -c%s "$OUTPUT")" -gt 1000 ]; then
            echo "PASS ($(stat -c%s "$OUTPUT") bytes)"
            PASSED=$((PASSED + 1))
        else
            echo "FAIL (empty or missing output)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "FAIL (binary error)"
        FAILED=$((FAILED + 1))
    fi

    rm -f "$OUTPUT"
done

echo ""
echo "========================================"
echo " Results: $PASSED passed, $FAILED failed"
echo "========================================"

[ "$FAILED" -eq 0 ] && exit 0 || exit 1
