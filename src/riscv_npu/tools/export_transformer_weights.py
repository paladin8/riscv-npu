"""Export trained transformer weights as C arrays and test data for firmware.

Trains a tiny character-level transformer on a simple text corpus,
quantizes weights to int8, and exports:
  - firmware/transformer/weights.h: C header with weight arrays
  - firmware/transformer/test_data.py: Python module with test sequences + expected outputs

Usage:
    uv run --extra torch python -m riscv_npu.tools.export_transformer_weights
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

def _fake_quantize_i8(x: torch.Tensor) -> torch.Tensor:
    """Simulate int8 quantization with straight-through estimator.

    Rounds to int8 grid in the forward pass but passes gradients through.
    Uses per-tensor symmetric quantization with power-of-2 scales
    to match the actual export quantization.
    """
    x_max = x.detach().abs().max().clamp(min=1e-8)
    # Power-of-2 scale matching the export
    shift = torch.floor(torch.log2(127.0 / x_max)).int().item()
    scale = float(2 ** shift)
    x_q = torch.clamp(torch.round(x * scale), -128, 127) / scale
    return x_q


class QATLinear(nn.Module):
    """Linear layer with fake weight quantization for QAT."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.qat_enabled = False

    @property
    def weight(self) -> nn.Parameter:
        """Access underlying weight for quantization export."""
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter:
        """Access underlying bias for quantization export."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional fake weight quantization."""
        if self.qat_enabled:
            w_q = _fake_quantize_i8(self.linear.weight)
            return F.linear(x, w_q, self.linear.bias)
        return self.linear(x)


def _fake_quantize_act_i8(x: torch.Tensor) -> torch.Tensor:
    """Simulate int8 activation quantization (straight-through estimator).

    Uses per-tensor symmetric quantization with power-of-2 scales, matching
    the implicit scaling in the firmware's shift-and-clamp operations.
    """
    x_max = x.detach().abs().max().clamp(min=1e-8)
    shift = torch.floor(torch.log2(127.0 / x_max)).int().item()
    scale = float(2 ** shift)
    x_q = torch.clamp(torch.round(x * scale), -128, 127) / scale
    return x_q


def _fake_quantize_act_u8(x: torch.Tensor) -> torch.Tensor:
    """Simulate uint8 activation quantization (straight-through estimator).

    Maps [0, 1] softmax probabilities to uint8 [0, 255] and back.
    """
    x_q = torch.clamp(torch.round(x * 255), 0, 255) / 255
    return x_q


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


class MultiHeadAttention(nn.Module):
    """Multi-head attention with QATLinear projections.

    Replaces nn.MultiheadAttention so that Q/K/V/O projection weights
    are fake-quantized during QAT training.
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = QATLinear(dim, dim)
        self.wk = QATLinear(dim, dim)
        self.wv = QATLinear(dim, dim)
        self.wo = QATLinear(dim, dim)
        self.qat_enabled = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Multi-head attention forward pass.

        Args:
            x: Input tensor (B, T, D).
            mask: Causal mask — True for positions to ignore (B or 1, T, T).

        Returns:
            Attention output (B, T, D).
        """
        B, T, D = x.shape

        # Project to Q, K, V and fake-quantize activations
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        if self.qat_enabled:
            q = _fake_quantize_act_i8(q)
            k = _fake_quantize_act_i8(k)
            v = _fake_quantize_act_i8(v)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # Simulate uint8 quantization of attention probabilities
        if self.qat_enabled:
            attn_weights = _fake_quantize_act_u8(attn_weights)

        # Weighted sum of V
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Simulate int8 after weighted V sum (firmware clamps to int8)
        if self.qat_enabled:
            out = _fake_quantize_act_i8(out)

        # Output projection
        out = self.wo(out)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feedforward."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.ff_w1 = QATLinear(dim, ff_dim)
        self.ff_act = nn.GELU()
        self.ff_w2 = QATLinear(ff_dim, dim)
        self.qat_enabled = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with residual connections and activation quantization."""
        normed = self.ln1(x)
        if self.qat_enabled:
            normed = _fake_quantize_act_i8(normed)

        attn_out = self.attn(normed, mask=mask)
        if self.qat_enabled:
            attn_out = _fake_quantize_act_i8(attn_out)

        # Residual add + clamp (firmware clamps to int8)
        x = x + attn_out
        if self.qat_enabled:
            x = _fake_quantize_act_i8(x)

        normed = self.ln2(x)
        if self.qat_enabled:
            normed = _fake_quantize_act_i8(normed)

        ff_out = self.ff_w1(normed)
        if self.qat_enabled:
            ff_out = _fake_quantize_act_i8(ff_out)
        ff_out = self.ff_act(ff_out)
        if self.qat_enabled:
            ff_out = _fake_quantize_act_i8(ff_out)
        ff_out = self.ff_w2(ff_out)
        if self.qat_enabled:
            ff_out = _fake_quantize_act_i8(ff_out)

        # Residual add + clamp
        x = x + ff_out
        if self.qat_enabled:
            x = _fake_quantize_act_i8(x)

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
        self.output = QATLinear(EMBED_DIM, VOCAB_SIZE)

    def enable_qat(self) -> None:
        """Enable fake quantization for all weights and activations."""
        for block in self.blocks:
            block.qat_enabled = True
            block.attn.qat_enabled = True
            block.attn.wq.qat_enabled = True
            block.attn.wk.qat_enabled = True
            block.attn.wv.qat_enabled = True
            block.attn.wo.qat_enabled = True
            block.ff_w1.qat_enabled = True
            block.ff_w2.qat_enabled = True
        self.output.qat_enabled = True
        self._qat_enabled = True

    def disable_qat(self) -> None:
        """Disable fake quantization (for float evaluation)."""
        for block in self.blocks:
            block.qat_enabled = False
            block.attn.qat_enabled = False
            block.attn.wq.qat_enabled = False
            block.attn.wk.qat_enabled = False
            block.attn.wv.qat_enabled = False
            block.attn.wo.qat_enabled = False
            block.ff_w1.qat_enabled = False
            block.ff_w2.qat_enabled = False
        self.output.qat_enabled = False
        self._qat_enabled = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass: tokens (B, T) -> logits (B, T, V)."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)

        x = self.token_embed(tokens) + self.pos_embed(pos)

        # Simulate int8 clamp after embedding addition
        if getattr(self, "_qat_enabled", False):
            x = _fake_quantize_act_i8(x)

        # Causal mask: upper-triangular True mask for positions to ignore
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_final(x)

        # Simulate int8 after final layer norm
        if getattr(self, "_qat_enabled", False):
            x = _fake_quantize_act_i8(x)

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

def train_model(
    epochs: int = 10, qat_epochs: int = 10, lr: float = 3e-4,
) -> TinyTransformer:
    """Train the character-level transformer with quantization-aware training.

    Phase 1: Standard float training to converge.
    Phase 2: QAT fine-tuning with simulated int8 quantization so the model
             learns weights that survive quantization.

    Args:
        epochs: Number of float training epochs.
        qat_epochs: Number of QAT fine-tuning epochs.
        lr: Learning rate for Adam optimizer.

    Returns:
        Trained model in eval mode (QAT disabled for float evaluation).
    """
    dataset = CharDataset(TRAIN_TEXT, CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    model = TinyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: Float training
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

    # Phase 2: QAT fine-tuning with higher LR to recover from quantization noise
    print("  Enabling quantization-aware training...")
    model.enable_qat()
    qat_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(qat_epochs):
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            qat_optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            qat_optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"  QAT {epoch + 1}/{qat_epochs}: loss={avg_loss:.4f}")

    model.disable_qat()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _quantize_tensor(
    t: torch.Tensor, *, scale: float | None = None,
) -> tuple[np.ndarray, float, int]:
    """Quantize a float tensor to int8 with power-of-2 scaling.

    Uses a power-of-2 scale so that the corresponding right-shift
    exactly cancels the scale, avoiding systematic magnitude errors
    that compound across layers.

    Args:
        t: Float tensor.
        scale: If provided, use this scale instead of computing from data.

    Returns:
        Tuple of (int8 numpy array, scale factor, shift).
    """
    with torch.no_grad():
        if scale is None:
            t_max = t.abs().max().item()
            if t_max < 1e-8:
                return (torch.zeros_like(t).to(torch.int8).numpy(), 1.0, 0)
            # Use largest power-of-2 scale that keeps values in int8 range
            ideal_scale = 127.0 / t_max
            shift = int(math.log2(ideal_scale))
            scale = float(2 ** shift)
        else:
            shift = int(math.log2(scale))
        q = torch.clamp(torch.round(t * scale), -128, 127).to(torch.int8)
    return q.numpy(), scale, shift


def _calibrate_activations(model: TinyTransformer) -> dict[str, float]:
    """Run calibration data through the model to determine activation magnitudes.

    Records the max absolute activation value at each int8 boundary.
    These are used to compute the combined shift (weight_shift + activation_shift)
    for the firmware's linear layers.

    Args:
        model: Trained model in eval mode.

    Returns:
        Dict mapping boundary name to max absolute activation value.
    """
    activation_maxes: dict[str, float] = {}
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def make_hook(name: str):  # noqa: ANN202
        def hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                val = output.detach().abs().max().item()
            elif isinstance(output, tuple):
                val = output[0].detach().abs().max().item()
            else:
                return
            activation_maxes[name] = max(activation_maxes.get(name, 0.0), val)
        return hook_fn

    # Hook RMSNorm outputs (inputs to linear layers)
    for i, block in enumerate(model.blocks):
        hooks.append(block.ln1.register_forward_hook(make_hook(f"layer{i}_ln1_out")))
        hooks.append(block.ln2.register_forward_hook(make_hook(f"layer{i}_ln2_out")))

    # Run calibration data
    dataset = CharDataset(TRAIN_TEXT, CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            model(x)
            if batch_idx >= 20:  # ~1280 samples is enough
                break

    for h in hooks:
        h.remove()

    print("  Calibrated activation magnitudes:")
    for name, val in sorted(activation_maxes.items()):
        print(f"    {name}: {val:.4f}")

    return activation_maxes


def quantize_model(
    model: TinyTransformer,
    act_maxes: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Quantize all model weights to int8.

    Uses calibrated activation magnitudes to compute combined shifts
    (weight_shift + activation_shift) so that the firmware's linear
    layer outputs fit within int8 range without clipping.

    Args:
        model: Trained model.
        act_maxes: Calibrated activation max values. If None, uses
            weight-only shifts (may cause clipping).

    Returns:
        Dict with all quantized weight arrays and scale factors.
    """
    weights: dict[str, Any] = {}

    with torch.no_grad():
        # Token and position embeddings
        weights["token_embed"], weights["token_embed_scale"], _ = _quantize_tensor(
            model.token_embed.weight.data
        )
        weights["pos_embed"], weights["pos_embed_scale"], _ = _quantize_tensor(
            model.pos_embed.weight.data
        )

        # Per-layer weights
        for layer_idx, block in enumerate(model.blocks):
            prefix = f"layer{layer_idx}_"

            # RMSNorm gammas
            weights[prefix + "ln1_gamma"], _, _ = _quantize_tensor(block.ln1.weight.data)
            weights[prefix + "ln2_gamma"], _, _ = _quantize_tensor(block.ln2.weight.data)

            # Attention: extract Q, K, V, O projection weights from QATLinear
            attn = block.attn
            proj_tensors = [
                attn.wq.weight.data,
                attn.wk.weight.data,
                attn.wv.weight.data,
                attn.wo.weight.data,
            ]
            proj_biases = [
                ("bq", attn.wq.bias.data),
                ("bk", attn.wk.bias.data),
                ("bv", attn.wv.bias.data),
                ("bo", attn.wo.bias.data),
            ]

            # Weight scale: shared across all attention projections
            proj_max = max(t.abs().max().item() for t in proj_tensors)
            proj_shift = int(math.log2(127.0 / proj_max))
            proj_scale = float(2 ** proj_shift)

            # Activation scale: from calibration (input to Q/K/V projections = ln1 output)
            # The bias must be scaled by BOTH weight_scale and activation_scale so
            # that after >> weight_shift, the output preserves the activation scale:
            #   output = (W_int8 @ x_int8 + bias) >> w_shift = S_x * (W @ x + b)
            if act_maxes and f"layer{layer_idx}_ln1_out" in act_maxes:
                act_max = act_maxes[f"layer{layer_idx}_ln1_out"]
                act_scale = float(2 ** int(math.log2(127.0 / max(act_max, 1e-8))))
            else:
                act_scale = 1.0

            for wname, tensor in zip(["wq", "wk", "wv", "wo"], proj_tensors):
                weights[prefix + wname], weights[prefix + wname + "_scale"], _ = (
                    _quantize_tensor(tensor, scale=proj_scale)
                )

            # Biases: scale by weight_scale * activation_scale
            for name, bias_data in proj_biases:
                b_q = torch.round(bias_data * proj_scale * act_scale).to(torch.int32)
                weights[prefix + name] = b_q.numpy()

            # Attention scale: 1/sqrt(head_dim) in Q16.16
            weights[prefix + "attn_scale"] = round((1.0 / math.sqrt(HEAD_DIM)) * 65536)

            # Shift = weight shift only (activation scale is in the bias)
            weights[prefix + "proj_shift"] = proj_shift

            # FFN weights: shared scale across both linear layers
            ff_tensors = [block.ff_w1.weight.data, block.ff_w2.weight.data]
            ff_max = max(t.abs().max().item() for t in ff_tensors)
            ff_shift = int(math.log2(127.0 / ff_max))
            ff_scale = float(2 ** ff_shift)

            # FFN activation scale (input to FFN = ln2 output)
            if act_maxes and f"layer{layer_idx}_ln2_out" in act_maxes:
                ff_act_max = act_maxes[f"layer{layer_idx}_ln2_out"]
                ff_act_scale = float(2 ** int(math.log2(127.0 / max(ff_act_max, 1e-8))))
            else:
                ff_act_scale = 1.0

            weights[prefix + "w1"], weights[prefix + "w1_scale"], _ = (
                _quantize_tensor(ff_tensors[0], scale=ff_scale)
            )
            weights[prefix + "w2"], weights[prefix + "w2_scale"], _ = (
                _quantize_tensor(ff_tensors[1], scale=ff_scale)
            )
            b1_q = torch.round(block.ff_w1.bias.data * ff_scale * ff_act_scale).to(torch.int32)
            b2_q = torch.round(block.ff_w2.bias.data * ff_scale * ff_act_scale).to(torch.int32)
            weights[prefix + "b1"] = b1_q.numpy()
            weights[prefix + "b2"] = b2_q.numpy()

            weights[prefix + "ff_shift"] = ff_shift

        # Final layer norm
        weights["ln_final_gamma"], _, _ = _quantize_tensor(model.ln_final.weight.data)

        # Output projection (no shift applied — logits stay as int32)
        weights["output_proj"], weights["output_proj_scale"], _ = _quantize_tensor(
            model.output.weight.data
        )
        out_scale = weights["output_proj_scale"]
        b_out = torch.round(model.output.bias.data * out_scale).to(torch.int32)
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
    firmware_dir = Path(__file__).parent.parent.parent.parent / "firmware" / "transformer"
    firmware_dir.mkdir(parents=True, exist_ok=True)

    print("Training tiny transformer (char-level LM)...")
    model = train_model(epochs=10)

    print("\nCalibrating activation magnitudes...")
    act_maxes = _calibrate_activations(model)

    print("\nQuantizing weights to int8...")
    weights = quantize_model(model, act_maxes)

    print("\nExporting C header...")
    export_c_header(weights, str(firmware_dir / "weights.h"))

    print("\nExporting test data...")
    export_test_data(model, weights, str(firmware_dir / "test_data.py"))

    print("\nDone! Next steps:")
    print("  cd firmware/transformer && make")
    print("  uv run python -m riscv_npu run firmware/transformer/transformer.elf")


if __name__ == "__main__":
    main()
