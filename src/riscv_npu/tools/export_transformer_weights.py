"""Export trained transformer weights as float32 C arrays and test data for firmware.

Trains a tiny character-level transformer on TinyShakespeare (~1MB)
and exports raw float32 weights:
  - firmware/transformer/weights.h: C header with float32 weight arrays
  - firmware/transformer/test_data.py: Python module with test sequences + expected outputs

No quantization, no int8, no Q16.16, no fake quantize. Pure float32 throughout.
Uses GPU if available.

Usage:
    uv run --extra torch python -m riscv_npu.tools.export_transformer_weights
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Any

# Torch is an optional dependency -- fail fast with a clear message
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: torch required.  Install with: uv sync --extra torch")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256    # Byte-level
EMBED_DIM = 64
N_HEADS = 4
HEAD_DIM = EMBED_DIM // N_HEADS  # 16
N_LAYERS = 2
CONTEXT_LEN = 32
FF_DIM = 4 * EMBED_DIM  # 256


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm."""
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


DROPOUT = 0.1


class MultiHeadAttention(nn.Module):
    """Multi-head attention with standard nn.Linear projections."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Multi-head attention forward pass.

        Args:
            x: Input tensor (B, T, D).
            mask: Causal mask -- True for positions to ignore (B or 1, T, T).

        Returns:
            Attention output (B, T, D).
        """
        B, T, D = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of V
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        out = self.resid_drop(self.wo(out))
        return out


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feedforward."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.ff_w1 = nn.Linear(dim, ff_dim)
        self.ff_act = nn.GELU()
        self.ff_w2 = nn.Linear(ff_dim, dim)
        self.ff_drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        normed = self.ln1(x)
        attn_out = self.attn(normed, mask=mask)
        x = x + attn_out

        normed = self.ln2(x)
        ff_out = self.ff_w1(normed)
        ff_out = self.ff_act(ff_out)
        ff_out = self.ff_drop(ff_out)
        ff_out = self.ff_w2(ff_out)
        x = x + ff_out

        return x


class TinyTransformer(nn.Module):
    """Tiny character-level transformer for next-token prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, EMBED_DIM)
        self.blocks = nn.ModuleList([
            TransformerBlock(EMBED_DIM, N_HEADS, FF_DIM)
            for _ in range(N_LAYERS)
        ])
        self.ln_final = RMSNorm(EMBED_DIM)
        self.output = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass: tokens (B, T) -> logits (B, T, V)."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)

        x = self.token_embed(tokens) + self.pos_embed(pos)

        # Causal mask: upper-triangular True mask for positions to ignore
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_final(x)
        logits = self.output(x)
        return logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SHAKESPEARE_PATH = DATA_DIR / "shakespeare.txt"

# Fallback corpus used only when shakespeare.txt is missing (e.g. in tests)
_FALLBACK_TEXT = (
    "the quick brown fox jumps over the lazy dog\n"
    "a quick brown fox jumps over a lazy dog\n"
    "the quick red fox jumps over the lazy cat\n"
    "hello world hello world hello world\n"
    "the cat sat on the mat the cat sat on the mat\n"
) * 200


def _load_corpus() -> str:
    """Load training corpus.

    Uses TinyShakespeare (~1MB) if available, otherwise falls back to a
    small built-in corpus.

    Returns:
        Training text as a string.
    """
    if SHAKESPEARE_PATH.exists():
        text = SHAKESPEARE_PATH.read_text(encoding="utf-8")
        print(f"  Loaded {SHAKESPEARE_PATH} ({len(text):,} chars)")
        return text
    print("  Warning: data/shakespeare.txt not found, using fallback corpus")
    return _FALLBACK_TEXT


class CharDataset(Dataset):
    """Character-level dataset for next-token prediction."""

    def __init__(self, text: str, seq_len: int) -> None:
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(epochs: int = 50, lr: float = 3e-4) -> TinyTransformer:
    """Train the character-level transformer on TinyShakespeare.

    Uses GPU if available, AdamW with cosine LR schedule, and gradient
    clipping for stable training on real text.

    Args:
        epochs: Number of training epochs.
        lr: Peak learning rate for AdamW optimizer.

    Returns:
        Trained model in eval mode (on CPU).
    """
    device = _get_device()
    print(f"  Device: {device}")

    corpus = _load_corpus()

    # 90/10 train/val split
    split = int(len(corpus) * 0.9)
    train_dataset = CharDataset(corpus[:split], CONTEXT_LEN)
    val_dataset = CharDataset(corpus[split:], CONTEXT_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, drop_last=True,
        num_workers=0,
    )

    model = TinyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: dict[str, Any] = {}

    model.train()
    for epoch in range(epochs):
        # Training
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
                val_loss += loss.item()
                val_batches += 1
        val_loss /= max(1, val_batches)
        model.train()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch + 1:3d}/{epochs}: "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best checkpoint (val_loss={best_val_loss:.4f})")

    model.eval()
    return model.cpu()


# ---------------------------------------------------------------------------
# Weight extraction (no quantization)
# ---------------------------------------------------------------------------

def extract_weights(model: TinyTransformer) -> dict[str, Any]:
    """Extract all model weights as float32 numpy arrays.

    No quantization, no scaling, no shifting. Just raw float32 weights.

    Args:
        model: Trained model.

    Returns:
        Dict with all float32 weight arrays.
    """
    weights: dict[str, Any] = {}

    with torch.no_grad():
        # Token and position embeddings
        weights["token_embed"] = model.token_embed.weight.data.numpy().astype(np.float32)
        weights["pos_embed"] = model.pos_embed.weight.data.numpy().astype(np.float32)

        # Per-layer weights
        for layer_idx, block in enumerate(model.blocks):
            prefix = f"layer{layer_idx}_"

            # RMSNorm gammas
            weights[prefix + "ln1_gamma"] = block.ln1.weight.data.numpy().astype(np.float32)
            weights[prefix + "ln2_gamma"] = block.ln2.weight.data.numpy().astype(np.float32)

            # Attention projections
            attn = block.attn
            weights[prefix + "wq"] = attn.wq.weight.data.numpy().astype(np.float32)
            weights[prefix + "bq"] = attn.wq.bias.data.numpy().astype(np.float32)
            weights[prefix + "wk"] = attn.wk.weight.data.numpy().astype(np.float32)
            weights[prefix + "bk"] = attn.wk.bias.data.numpy().astype(np.float32)
            weights[prefix + "wv"] = attn.wv.weight.data.numpy().astype(np.float32)
            weights[prefix + "bv"] = attn.wv.bias.data.numpy().astype(np.float32)
            weights[prefix + "wo"] = attn.wo.weight.data.numpy().astype(np.float32)
            weights[prefix + "bo"] = attn.wo.bias.data.numpy().astype(np.float32)

            # FFN weights
            weights[prefix + "w1"] = block.ff_w1.weight.data.numpy().astype(np.float32)
            weights[prefix + "b1"] = block.ff_w1.bias.data.numpy().astype(np.float32)
            weights[prefix + "w2"] = block.ff_w2.weight.data.numpy().astype(np.float32)
            weights[prefix + "b2"] = block.ff_w2.bias.data.numpy().astype(np.float32)

        # Final layer norm
        weights["ln_final_gamma"] = model.ln_final.weight.data.numpy().astype(np.float32)

        # Output projection
        weights["output_proj"] = model.output.weight.data.numpy().astype(np.float32)
        weights["output_bias"] = model.output.bias.data.numpy().astype(np.float32)

    # Print summary
    total_params = sum(v.size for v in weights.values() if isinstance(v, np.ndarray))
    total_bytes = sum(v.nbytes for v in weights.values() if isinstance(v, np.ndarray))
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total weight size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")

    return weights


# ---------------------------------------------------------------------------
# Python reference inference
# ---------------------------------------------------------------------------

def float_inference_python(
    tokens: list[int],
    weights: dict[str, Any],
) -> int:
    """Run float transformer inference in Python.

    Mirrors the firmware behavior for test validation.

    Args:
        tokens: Input token sequence (list of byte values 0-255).
        weights: Float weights dict from extract_weights().

    Returns:
        Predicted next token (0-255).
    """
    from riscv_npu.tools.transformer import (
        TransformerConfig,
        TransformerWeights,
        LayerWeights,
        predict_next_token,
    )

    config = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        n_layers=N_LAYERS,
        context_len=CONTEXT_LEN,
        ff_dim=FF_DIM,
    )

    # Convert numpy arrays to Python lists for the reference implementation
    layers = []
    for i in range(N_LAYERS):
        p = f"layer{i}_"
        layer = LayerWeights(
            ln1_gamma=weights[p + "ln1_gamma"].tolist(),
            wq=weights[p + "wq"].tolist(),
            bq=weights[p + "bq"].tolist(),
            wk=weights[p + "wk"].tolist(),
            bk=weights[p + "bk"].tolist(),
            wv=weights[p + "wv"].tolist(),
            bv=weights[p + "bv"].tolist(),
            wo=weights[p + "wo"].tolist(),
            bo=weights[p + "bo"].tolist(),
            ln2_gamma=weights[p + "ln2_gamma"].tolist(),
            w1=weights[p + "w1"].tolist(),
            b1=weights[p + "b1"].tolist(),
            w2=weights[p + "w2"].tolist(),
            b2=weights[p + "b2"].tolist(),
        )
        layers.append(layer)

    tw = TransformerWeights(
        token_embed=weights["token_embed"].tolist(),
        pos_embed=weights["pos_embed"].tolist(),
        layers=layers,
        ln_final_gamma=weights["ln_final_gamma"].tolist(),
        output_proj=weights["output_proj"].tolist(),
        output_bias=weights["output_bias"].tolist(),
    )

    return predict_next_token(tokens, tw, config)


# ---------------------------------------------------------------------------
# C header export
# ---------------------------------------------------------------------------

def _format_float_array(arr: np.ndarray, name: str, dims: tuple[int, ...]) -> str:
    """Format a float32 numpy array as a C array initializer.

    For 2D arrays, emits nested braces to suppress GCC -Wmissing-braces warnings.

    Args:
        arr: Float32 numpy array.
        name: C variable name.
        dims: Array dimensions (e.g. (256, 64) for 2D).

    Returns:
        C source code for the array declaration.
    """
    dim_str = "".join(f"[{d}]" for d in dims)
    lines = [f"static const float {name}{dim_str} = {{"]

    if len(dims) == 2:
        rows, cols = dims
        for r in range(rows):
            row = arr[r].tolist() if arr.ndim == 2 else arr[r * cols:(r + 1) * cols].tolist()
            row_str = ", ".join(f"{v:.8e}f" for v in row)
            lines.append(f"    {{{row_str}}},")
    else:
        flat = arr.flatten().tolist()
        for i in range(0, len(flat), 10):
            chunk = flat[i:i + 10]
            line = "    " + ", ".join(f"{v:.8e}f" for v in chunk) + ","
            lines.append(line)

    lines.append("};")
    return "\n".join(lines)


def export_c_header(weights: dict[str, Any], path: str) -> None:
    """Write float32 weights as a C header file.

    Args:
        weights: Float weight arrays from extract_weights().
        path: Output file path.
    """
    parts = [
        "#ifndef TRANSFORMER_WEIGHTS_H",
        "#define TRANSFORMER_WEIGHTS_H",
        "",
        f"/* Tiny Transformer (float32): vocab={VOCAB_SIZE}, dim={EMBED_DIM}, "
        f"heads={N_HEADS}, layers={N_LAYERS}, ctx={CONTEXT_LEN}, ff={FF_DIM} */",
        "",
        f"#define VOCAB_SIZE {VOCAB_SIZE}",
        f"#define EMBED_DIM {EMBED_DIM}",
        f"#define N_HEADS {N_HEADS}",
        f"#define HEAD_DIM {HEAD_DIM}",
        f"#define N_LAYERS {N_LAYERS}",
        f"#define CONTEXT_LEN {CONTEXT_LEN}",
        f"#define FF_DIM {FF_DIM}",
        "",
    ]

    # Token and position embeddings
    parts.append("/* Token embedding: (256, 64) */")
    parts.append(_format_float_array(weights["token_embed"], "TOKEN_EMBED", (VOCAB_SIZE, EMBED_DIM)))
    parts.append("")
    parts.append("/* Position embedding: (32, 64) */")
    parts.append(_format_float_array(weights["pos_embed"], "POS_EMBED", (CONTEXT_LEN, EMBED_DIM)))
    parts.append("")

    # Per-layer weights
    for i in range(N_LAYERS):
        p = f"layer{i}_"
        layer_name = f"L{i}"
        parts.append(f"/* === Layer {i} === */")
        parts.append("")

        # RMSNorm gammas
        parts.append(_format_float_array(weights[p + "ln1_gamma"], f"{layer_name}_LN1_GAMMA", (EMBED_DIM,)))
        parts.append(_format_float_array(weights[p + "ln2_gamma"], f"{layer_name}_LN2_GAMMA", (EMBED_DIM,)))
        parts.append("")

        # Attention weights and biases
        for wname in ["wq", "wk", "wv", "wo"]:
            cname = f"{layer_name}_{wname.upper()}"
            parts.append(_format_float_array(weights[p + wname], cname, (EMBED_DIM, EMBED_DIM)))
        parts.append("")

        for bname in ["bq", "bk", "bv", "bo"]:
            cname = f"{layer_name}_{bname.upper()}"
            parts.append(_format_float_array(weights[p + bname], cname, (EMBED_DIM,)))
        parts.append("")

        # FFN weights and biases
        parts.append(_format_float_array(weights[p + "w1"], f"{layer_name}_W1", (FF_DIM, EMBED_DIM)))
        parts.append(_format_float_array(weights[p + "w2"], f"{layer_name}_W2", (EMBED_DIM, FF_DIM)))
        parts.append(_format_float_array(weights[p + "b1"], f"{layer_name}_B1", (FF_DIM,)))
        parts.append(_format_float_array(weights[p + "b2"], f"{layer_name}_B2", (EMBED_DIM,)))
        parts.append("")

    # Final layer norm
    parts.append("/* Final RMSNorm */")
    parts.append(_format_float_array(weights["ln_final_gamma"], "LN_FINAL_GAMMA", (EMBED_DIM,)))
    parts.append("")

    # Output projection
    parts.append("/* Output projection: (256, 64) */")
    parts.append(_format_float_array(weights["output_proj"], "OUTPUT_PROJ", (VOCAB_SIZE, EMBED_DIM)))
    parts.append(_format_float_array(weights["output_bias"], "OUTPUT_BIAS", (VOCAB_SIZE,)))
    parts.append("")

    parts.append("#endif /* TRANSFORMER_WEIGHTS_H */")
    parts.append("")

    Path(path).write_text("\n".join(parts))
    print(f"  Wrote {path} ({Path(path).stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Test data export
# ---------------------------------------------------------------------------

def export_test_data(
    model: TinyTransformer,
    weights: dict[str, Any],
    path: str,
    num_sequences: int = 10,
) -> None:
    """Export test sequences and reference predictions.

    Saves a .py file with:
        SEQUENCES: list of token sequences (input)
        FLOAT_PREDICTIONS: float model predictions (both PyTorch and Python reference)

    Args:
        model: Trained PyTorch model.
        weights: Float weights dict.
        path: Output file path.
        num_sequences: Number of test sequences to export.
    """
    # Generate test sequences from the corpus (use val split for fairness)
    corpus = _load_corpus()
    split = int(len(corpus) * 0.9)
    test_text = corpus[split:]
    test_tokens = [ord(c) for c in test_text]

    sequences = []
    float_preds = []

    for i in range(num_sequences):
        start = i * CONTEXT_LEN
        if start + CONTEXT_LEN >= len(test_tokens):
            break
        seq = test_tokens[start:start + CONTEXT_LEN]
        sequences.append(seq)

        # Float prediction using Python reference
        fpred = float_inference_python(seq, weights)
        float_preds.append(fpred)

    lines = [
        '"""Auto-generated transformer test data. Do not edit manually."""',
        "",
        f"CONTEXT_LEN = {CONTEXT_LEN}",
        "",
        "# Test sequences as byte values [0, 255]",
        f"SEQUENCES = {sequences}",
        "",
        "# Float inference predictions (Python reference, argmax of last-token logits)",
        f"FLOAT_PREDICTIONS = {float_preds}",
        "",
    ]

    Path(path).write_text("\n".join(lines))
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Train and export transformer weights as float32."""
    firmware_dir = Path(__file__).parent.parent.parent.parent / "firmware" / "transformer"
    firmware_dir.mkdir(parents=True, exist_ok=True)

    print("Training tiny transformer on TinyShakespeare...")
    model = train_model(epochs=50)

    print("\nExtracting float32 weights...")
    weights = extract_weights(model)

    print("\nExporting C header...")
    export_c_header(weights, str(firmware_dir / "weights.h"))

    print("\nExporting test data...")
    export_test_data(model, weights, str(firmware_dir / "test_data.py"))

    print("\nDone! Next steps:")
    print("  cd firmware/transformer && make")
    print("  uv run python -m riscv_npu run firmware/transformer/transformer.elf")


if __name__ == "__main__":
    main()
