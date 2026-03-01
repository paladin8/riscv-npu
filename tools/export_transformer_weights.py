"""Export trained transformer weights as C arrays and test data for firmware.

Trains a tiny character-level transformer on a simple text corpus,
quantizes weights to int8, and exports:
  - firmware/transformer/weights.h: C header with weight arrays
  - firmware/transformer/test_data.py: Python module with test sequences + expected outputs

Usage:
    uv run --extra torch python tools/export_transformer_weights.py
"""

from __future__ import annotations

import math
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


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feedforward."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        normed = self.ln2(x)
        x = x + self.ff(normed)
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

# Simple text corpus for training -- repetitive patterns that a tiny model can learn
TRAIN_TEXT = """the quick brown fox jumps over the lazy dog
a quick brown fox jumps over a lazy dog
the quick red fox jumps over the lazy cat
a quick red fox jumps over a lazy cat
the slow brown fox runs past the lazy dog
a slow brown fox runs past a lazy dog
the quick brown dog jumps over the lazy fox
hello world hello world hello world
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz
the cat sat on the mat the cat sat on the mat
""" * 100  # Repeat for sufficient training data


class CharDataset(Dataset):
    """Character-level dataset for next-token prediction."""

    def __init__(self, text: str, seq_len: int) -> None:
        self.data = [ord(c) for c in text]
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(epochs: int = 10, lr: float = 3e-4) -> TinyTransformer:
    """Train the character-level transformer.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.

    Returns:
        Trained model in eval mode.
    """
    dataset = CharDataset(TRAIN_TEXT, CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    model = TinyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)  # (B, T, V)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _quantize_tensor(t: torch.Tensor) -> tuple[np.ndarray, float]:
    """Quantize a float tensor to int8 with symmetric scaling.

    Args:
        t: Float tensor.

    Returns:
        Tuple of (int8 numpy array, scale factor).
    """
    with torch.no_grad():
        t_max = t.abs().max().item()
        if t_max < 1e-8:
            scale = 1.0
        else:
            scale = 127.0 / t_max
        q = torch.clamp(torch.round(t * scale), -128, 127).to(torch.int8)
    return q.numpy(), scale


def quantize_model(model: TinyTransformer) -> dict[str, Any]:
    """Quantize all model weights to int8.

    Returns a dict with all quantized weight arrays and scale factors.
    """
    weights: dict[str, Any] = {}

    with torch.no_grad():
        # Token and position embeddings
        weights["token_embed"], weights["token_embed_scale"] = _quantize_tensor(
            model.token_embed.weight.data
        )
        weights["pos_embed"], weights["pos_embed_scale"] = _quantize_tensor(
            model.pos_embed.weight.data
        )

        # Per-layer weights
        for layer_idx, block in enumerate(model.blocks):
            prefix = f"layer{layer_idx}_"

            # RMSNorm gammas
            weights[prefix + "ln1_gamma"], _ = _quantize_tensor(block.ln1.weight.data)
            weights[prefix + "ln2_gamma"], _ = _quantize_tensor(block.ln2.weight.data)

            # Attention: extract Q, K, V, O projection weights
            # nn.MultiheadAttention stores them combined
            attn = block.attn
            weights[prefix + "wq"], weights[prefix + "wq_scale"] = _quantize_tensor(
                attn.in_proj_weight[:EMBED_DIM]
            )
            weights[prefix + "wk"], weights[prefix + "wk_scale"] = _quantize_tensor(
                attn.in_proj_weight[EMBED_DIM:2*EMBED_DIM]
            )
            weights[prefix + "wv"], weights[prefix + "wv_scale"] = _quantize_tensor(
                attn.in_proj_weight[2*EMBED_DIM:]
            )
            weights[prefix + "wo"], weights[prefix + "wo_scale"] = _quantize_tensor(
                attn.out_proj.weight.data
            )

            # Biases (quantize to int32 in accumulator scale)
            # For simplicity, store raw biases scaled by weight scale
            for name, bias_data, w_scale_key in [
                ("bq", attn.in_proj_bias[:EMBED_DIM], prefix + "wq_scale"),
                ("bk", attn.in_proj_bias[EMBED_DIM:2*EMBED_DIM], prefix + "wk_scale"),
                ("bv", attn.in_proj_bias[2*EMBED_DIM:], prefix + "wv_scale"),
                ("bo", attn.out_proj.bias.data, prefix + "wo_scale"),
            ]:
                # Bias scale = input_scale * weight_scale
                # For simplicity, just scale by 256 (rough approximation)
                b_q = torch.round(bias_data * 256).to(torch.int32)
                weights[prefix + name] = b_q.numpy()

            # Attention scale: 1/sqrt(head_dim) in Q16.16
            weights[prefix + "attn_scale"] = round((1.0 / math.sqrt(HEAD_DIM)) * 65536)

            # FFN weights
            ff = block.ff
            weights[prefix + "w1"], weights[prefix + "w1_scale"] = _quantize_tensor(
                ff[0].weight.data  # Linear 1
            )
            weights[prefix + "w2"], weights[prefix + "w2_scale"] = _quantize_tensor(
                ff[2].weight.data  # Linear 2
            )
            b1_q = torch.round(ff[0].bias.data * 256).to(torch.int32)
            b2_q = torch.round(ff[2].bias.data * 256).to(torch.int32)
            weights[prefix + "b1"] = b1_q.numpy()
            weights[prefix + "b2"] = b2_q.numpy()

            # Shift values (heuristic: based on accumulator scale)
            weights[prefix + "proj_shift"] = 8
            weights[prefix + "ff_shift"] = 8

        # Final layer norm
        weights["ln_final_gamma"], _ = _quantize_tensor(model.ln_final.weight.data)

        # Output projection
        weights["output_proj"], weights["output_proj_scale"] = _quantize_tensor(
            model.output.weight.data
        )
        b_out = torch.round(model.output.bias.data * 256).to(torch.int32)
        weights["output_bias"] = b_out.numpy()

    # Print summary
    total_params = sum(
        v.size for v in weights.values()
        if isinstance(v, np.ndarray)
    )
    total_bytes = sum(
        v.nbytes for v in weights.values()
        if isinstance(v, np.ndarray)
    )
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total weight size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")

    return weights


# ---------------------------------------------------------------------------
# Python reference inference
# ---------------------------------------------------------------------------

def quantized_inference_python(
    tokens: list[int],
    weights: dict[str, Any],
) -> int:
    """Run quantized transformer inference in Python.

    Mirrors the firmware behavior for test validation.

    Args:
        tokens: Input token sequence (list of byte values 0-255).
        weights: Quantized weights dict from quantize_model().

    Returns:
        Predicted next token (0-255).
    """
    from riscv_npu.npu.transformer import (
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
            wk=weights[p + "wk"].tolist(),
            wv=weights[p + "wv"].tolist(),
            wo=weights[p + "wo"].tolist(),
            bq=weights[p + "bq"].tolist(),
            bk=weights[p + "bk"].tolist(),
            bv=weights[p + "bv"].tolist(),
            bo=weights[p + "bo"].tolist(),
            attn_scale=weights[p + "attn_scale"],
            ln2_gamma=weights[p + "ln2_gamma"].tolist(),
            w1=weights[p + "w1"].tolist(),
            w2=weights[p + "w2"].tolist(),
            b1=weights[p + "b1"].tolist(),
            b2=weights[p + "b2"].tolist(),
            proj_shift=weights[p + "proj_shift"],
            ff_shift=weights[p + "ff_shift"],
        )
        layers.append(layer)

    tw = TransformerWeights(
        token_embed=weights["token_embed"].tolist(),
        pos_embed=weights["pos_embed"].tolist(),
        layers=layers,
        ln_final_gamma=weights["ln_final_gamma"].tolist(),
        output_proj=weights["output_proj"].tolist(),
        output_bias=weights["output_bias"].tolist(),
        embed_scale=65536,  # 1.0 in Q16.16
    )

    return predict_next_token(tokens, tw, config)


# ---------------------------------------------------------------------------
# C header export
# ---------------------------------------------------------------------------

def _format_int8_array(arr: np.ndarray, name: str, dims: tuple[int, ...]) -> str:
    """Format an int8 numpy array as a C array initializer."""
    flat = arr.flatten().tolist()
    dim_str = "".join(f"[{d}]" for d in dims)
    lines = [f"static const int8_t {name}{dim_str} = {{"]
    for i in range(0, len(flat), 20):
        chunk = flat[i:i + 20]
        line = "    " + ", ".join(str(v) for v in chunk) + ","
        lines.append(line)
    lines.append("};")
    return "\n".join(lines)


def _format_int32_array(arr: np.ndarray, name: str, dim: int) -> str:
    """Format an int32 numpy array as a C array initializer."""
    flat = arr.flatten().tolist()
    lines = [f"static const int32_t {name}[{dim}] = {{"]
    for i in range(0, len(flat), 10):
        chunk = flat[i:i + 10]
        line = "    " + ", ".join(str(v) for v in chunk) + ","
        lines.append(line)
    lines.append("};")
    return "\n".join(lines)


def export_c_header(weights: dict[str, Any], path: str) -> None:
    """Write quantized weights as a C header file."""
    parts = [
        "#ifndef TRANSFORMER_WEIGHTS_H",
        "#define TRANSFORMER_WEIGHTS_H",
        "#include <stdint.h>",
        "",
        f"/* Tiny Transformer: vocab={VOCAB_SIZE}, dim={EMBED_DIM}, "
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
    parts.append(_format_int8_array(weights["token_embed"], "TOKEN_EMBED", (VOCAB_SIZE, EMBED_DIM)))
    parts.append("")
    parts.append("/* Position embedding: (32, 64) */")
    parts.append(_format_int8_array(weights["pos_embed"], "POS_EMBED", (CONTEXT_LEN, EMBED_DIM)))
    parts.append("")

    # Per-layer weights
    for i in range(N_LAYERS):
        p = f"layer{i}_"
        layer_name = f"L{i}"
        parts.append(f"/* === Layer {i} === */")
        parts.append("")

        # RMSNorm gammas
        parts.append(_format_int8_array(weights[p + "ln1_gamma"], f"{layer_name}_LN1_GAMMA", (EMBED_DIM,)))
        parts.append(_format_int8_array(weights[p + "ln2_gamma"], f"{layer_name}_LN2_GAMMA", (EMBED_DIM,)))
        parts.append("")

        # Attention weights
        for wname in ["wq", "wk", "wv", "wo"]:
            cname = f"{layer_name}_{wname.upper()}"
            parts.append(_format_int8_array(weights[p + wname], cname, (EMBED_DIM, EMBED_DIM)))
        parts.append("")

        # Biases
        for bname in ["bq", "bk", "bv", "bo"]:
            cname = f"{layer_name}_{bname.upper()}"
            parts.append(_format_int32_array(weights[p + bname], cname, EMBED_DIM))
        parts.append("")

        # Attention scale
        parts.append(f"static const int32_t {layer_name}_ATTN_SCALE = {weights[p + 'attn_scale']};")
        parts.append(f"static const int32_t {layer_name}_PROJ_SHIFT = {weights[p + 'proj_shift']};")
        parts.append("")

        # FFN weights
        parts.append(_format_int8_array(weights[p + "w1"], f"{layer_name}_W1", (FF_DIM, EMBED_DIM)))
        parts.append(_format_int8_array(weights[p + "w2"], f"{layer_name}_W2", (EMBED_DIM, FF_DIM)))
        parts.append(_format_int32_array(weights[p + "b1"], f"{layer_name}_B1", FF_DIM))
        parts.append(_format_int32_array(weights[p + "b2"], f"{layer_name}_B2", EMBED_DIM))
        parts.append(f"static const int32_t {layer_name}_FF_SHIFT = {weights[p + 'ff_shift']};")
        parts.append("")

    # Final layer norm
    parts.append("/* Final RMSNorm */")
    parts.append(_format_int8_array(weights["ln_final_gamma"], "LN_FINAL_GAMMA", (EMBED_DIM,)))
    parts.append("")

    # Output projection
    parts.append("/* Output projection: (256, 64) */")
    parts.append(_format_int8_array(weights["output_proj"], "OUTPUT_PROJ", (VOCAB_SIZE, EMBED_DIM)))
    parts.append(_format_int32_array(weights["output_bias"], "OUTPUT_BIAS", VOCAB_SIZE))
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
        FLOAT_PREDICTIONS: PyTorch float model predictions
        QUANT_PREDICTIONS: quantized inference predictions

    Args:
        model: Trained PyTorch model.
        weights: Quantized weights dict.
        path: Output file path.
        num_sequences: Number of test sequences to export.
    """
    # Generate test sequences from the training text
    test_chars = TRAIN_TEXT[:CONTEXT_LEN * num_sequences * 2]
    test_tokens = [ord(c) for c in test_chars]

    sequences = []
    float_preds = []
    quant_preds = []

    for i in range(num_sequences):
        start = i * CONTEXT_LEN
        if start + CONTEXT_LEN >= len(test_tokens):
            break
        seq = test_tokens[start:start + CONTEXT_LEN]
        sequences.append(seq)

        # Float prediction
        with torch.no_grad():
            t = torch.tensor([seq], dtype=torch.long)
            logits = model(t)
            pred = logits[0, -1].argmax().item()
            float_preds.append(pred)

        # Quantized prediction
        qpred = quantized_inference_python(seq, weights)
        quant_preds.append(qpred)

    lines = [
        '"""Auto-generated transformer test data. Do not edit manually."""',
        "",
        f"CONTEXT_LEN = {CONTEXT_LEN}",
        "",
        "# Test sequences as byte values [0, 255]",
        f"SEQUENCES = {sequences}",
        "",
        "# PyTorch float model predictions (argmax of last-token logits)",
        f"FLOAT_PREDICTIONS = {float_preds}",
        "",
        "# Quantized inference predictions (Python reference)",
        f"QUANT_PREDICTIONS = {quant_preds}",
        "",
    ]

    Path(path).write_text("\n".join(lines))
    print(f"  Wrote {path}")

    # Report agreement
    match_count = sum(1 for p, q in zip(float_preds, quant_preds) if p == q)
    print(f"  Float/quant agreement: {match_count}/{len(float_preds)} "
          f"({match_count / max(1, len(float_preds)) * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Train, quantize, and export transformer weights."""
    firmware_dir = Path(__file__).parent.parent / "firmware" / "transformer"
    firmware_dir.mkdir(parents=True, exist_ok=True)

    print("Training tiny transformer (char-level LM)...")
    model = train_model(epochs=10)

    print("\nQuantizing weights to int8...")
    weights = quantize_model(model)

    print("\nExporting C header...")
    export_c_header(weights, str(firmware_dir / "weights.h"))

    print("\nExporting test data...")
    export_test_data(model, weights, str(firmware_dir / "test_data.py"))

    print("\nDone! Next steps:")
    print("  cd firmware/transformer && make")
    print("  uv run python -m riscv_npu run firmware/transformer/transformer.elf")


if __name__ == "__main__":
    main()
