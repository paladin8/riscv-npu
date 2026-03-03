#!/usr/bin/env bash
# Debug the MNIST firmware with a test image from test_data.py.
#
# Usage:
#   ./scripts/mnist-debug.sh 0       # image index 0 (label: 7)
#   ./scripts/mnist-debug.sh 42      # image index 42
#
# Looks up the image at the given index in firmware/mnist/test_data.py,
# writes 784 bytes into the test_image symbol, and launches the TUI debugger.

set -euo pipefail

ELF="firmware/mnist/mnist.elf"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_index (0-99)>"
    exit 1
fi

if [ ! -f "$ELF" ]; then
    echo "Error: $ELF not found. Run 'cd firmware/mnist && make' first."
    exit 1
fi

INDEX="$1"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Extract the image as 784 raw bytes and print the expected label
LABEL=$(python3 -c "
import struct, sys
sys.path.insert(0, 'firmware/mnist')
from test_data import IMAGES, LABELS

idx = int(sys.argv[1])
if idx < 0 or idx >= len(IMAGES):
    print(f'Error: index {idx} out of range (0-{len(IMAGES)-1})', file=sys.stderr)
    sys.exit(1)

with open('$TMPDIR/image.bin', 'wb') as f:
    f.write(bytes(IMAGES[idx]))

print(LABELS[idx])
" "$INDEX")

echo "Image index: $INDEX  (expected label: $LABEL)"
echo "Launching TUI debugger..."

exec uv run python -m riscv_npu debug "$ELF" \
    --write "test_image:$TMPDIR/image.bin"
