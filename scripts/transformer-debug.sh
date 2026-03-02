#!/usr/bin/env bash
# Debug the transformer firmware with a text input string.
#
# Usage:
#   ./scripts/transformer-debug.sh "the quick "
#   ./scripts/transformer-debug.sh "hello " 5     # generate 5 tokens
#
# The string is converted to token bytes and injected into the firmware's
# test_tokens/test_n_tokens/test_n_generate symbols, then the TUI debugger
# is launched. Max 32 characters (CONTEXT_LEN). Default: generate tokens
# until the context window is full (32 total positions).

set -euo pipefail

ELF="firmware/transformer/transformer.elf"

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"input text (max 32 chars)\" [n_generate]"
    exit 1
fi

if [ ! -f "$ELF" ]; then
    echo "Error: $ELF not found. Run 'cd firmware/transformer && make' first."
    exit 1
fi

INPUT="$1"
N_GEN="${2:-0}"  # 0 = fill context (default)
LEN=${#INPUT}
if [ "$LEN" -gt 32 ]; then
    echo "Error: input must be at most 32 characters (got $LEN)"
    exit 1
fi
if [ "$LEN" -eq 0 ]; then
    echo "Error: input must not be empty"
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Create binary token data, prompt length, and generation count
python3 -c "
import struct, sys
text = sys.argv[1]
n_gen = int(sys.argv[2])
with open('$TMPDIR/tokens.bin', 'wb') as f:
    f.write(text.encode('latin-1').ljust(32, b'\x00'))
with open('$TMPDIR/n_tokens.bin', 'wb') as f:
    f.write(struct.pack('<i', len(text)))
with open('$TMPDIR/n_generate.bin', 'wb') as f:
    f.write(struct.pack('<i', n_gen))
" "$INPUT" "$N_GEN"

if [ "$N_GEN" -eq 0 ]; then
    echo "Prompt: \"$INPUT\" ($LEN tokens, generating until context full)"
else
    echo "Prompt: \"$INPUT\" ($LEN tokens, generating $N_GEN)"
fi
echo "Launching TUI debugger..."

exec uv run python -m riscv_npu debug "$ELF" \
    --write "test_tokens:$TMPDIR/tokens.bin" \
    --write "test_n_tokens:$TMPDIR/n_tokens.bin" \
    --write "test_n_generate:$TMPDIR/n_generate.bin"
