"""Quantized transformer inference reference implementation.

Pure-Python implementation of a character-level transformer that matches
the firmware behavior exactly. All operations use int8 weights, int32
intermediates, and Q16.16 fixed-point for softmax/RMSNorm.

This module is used to validate firmware output in integration tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import Q16_ONE, exp_q16_16, rsqrt_q16_16


@dataclass
class TransformerConfig:
    """Configuration for the tiny character-level transformer.

    Attributes:
        vocab_size: Number of unique tokens (256 for byte-level).
        embed_dim: Embedding dimension.
        n_heads: Number of attention heads.
        head_dim: Dimension per head (embed_dim // n_heads).
        n_layers: Number of transformer blocks.
        context_len: Maximum sequence length.
        ff_dim: Feedforward hidden dimension (typically 4 * embed_dim).
    """

    vocab_size: int = 256
    embed_dim: int = 64
    n_heads: int = 4
    head_dim: int = 16  # embed_dim // n_heads
    n_layers: int = 2
    context_len: int = 32
    ff_dim: int = 256  # 4 * embed_dim


@dataclass
class TransformerWeights:
    """Quantized weights for the transformer model.

    All weight matrices are int8 (list of lists of int).
    All bias vectors are int32 (list of int).
    Scale factors are Q16.16 int32 values.

    Attributes:
        token_embed: Token embedding table, shape (vocab_size, embed_dim), int8.
        pos_embed: Positional embedding table, shape (context_len, embed_dim), int8.
        layers: Per-layer weights (list of LayerWeights).
        ln_final_gamma: Final layer norm gamma, shape (embed_dim,), int8.
        output_proj: Output projection, shape (vocab_size, embed_dim), int8.
        output_bias: Output bias, shape (vocab_size,), int32.
        embed_scale: Scale factor for embedding addition, Q16.16.
    """

    token_embed: list[list[int]]
    pos_embed: list[list[int]]
    layers: list[LayerWeights]
    ln_final_gamma: list[int]
    output_proj: list[list[int]]
    output_bias: list[int]
    embed_scale: int


@dataclass
class LayerWeights:
    """Quantized weights for one transformer layer.

    Attributes:
        ln1_gamma: RMSNorm gamma for attention, shape (embed_dim,), int8.
        wq: Query projection, shape (embed_dim, embed_dim), int8.
        wk: Key projection, shape (embed_dim, embed_dim), int8.
        wv: Value projection, shape (embed_dim, embed_dim), int8.
        wo: Output projection, shape (embed_dim, embed_dim), int8.
        bq: Query bias, shape (embed_dim,), int32.
        bk: Key bias, shape (embed_dim,), int32.
        bv: Value bias, shape (embed_dim,), int32.
        bo: Output bias, shape (embed_dim,), int32.
        attn_scale: Attention scale factor (1/sqrt(head_dim)) in Q16.16.
        ln2_gamma: RMSNorm gamma for FFN, shape (embed_dim,), int8.
        w1: FFN first linear, shape (ff_dim, embed_dim), int8.
        w2: FFN second linear, shape (embed_dim, ff_dim), int8.
        b1: FFN first bias, shape (ff_dim,), int32.
        b2: FFN second bias, shape (embed_dim,), int32.
        proj_shift: Right-shift for projection outputs.
        ff_shift: Right-shift for FFN outputs.
    """

    ln1_gamma: list[int]
    wq: list[list[int]]
    wk: list[list[int]]
    wv: list[list[int]]
    wo: list[list[int]]
    bq: list[int]
    bk: list[int]
    bv: list[int]
    bo: list[int]
    attn_scale: int
    ln2_gamma: list[int]
    w1: list[list[int]]
    w2: list[list[int]]
    b1: list[int]
    b2: list[int]
    proj_shift: int
    ff_shift: int


def _clamp_i8(x: int) -> int:
    """Clamp an integer to int8 range [-128, 127]."""
    return max(-128, min(127, x))


def _dot_i8(a: list[int], b: list[int]) -> int:
    """Compute dot product of two int8 vectors, returning int32.

    Args:
        a: First int8 vector.
        b: Second int8 vector (same length as a).

    Returns:
        Sum of element-wise products (Python int, no overflow).
    """
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def rmsnorm_q(x: list[int], gamma: list[int], dim: int) -> list[int]:
    """Apply RMSNorm in the quantized domain.

    Computes: norm(x) = x * gamma * rsqrt(mean(x^2) + eps)

    All values are int8, intermediate computations use int32/Q16.16.

    Args:
        x: Input vector, shape (dim,), int8 values.
        gamma: Learned scale, shape (dim,), int8 values.
        dim: Vector dimension.

    Returns:
        Normalized vector, shape (dim,), int8 values.
    """
    # Compute mean of squares in Q16.16
    # x values are int8 [-128, 127], x^2 is at most 16384
    # sum of squares for dim=64 is at most ~1M, fits in int32
    sum_sq = 0
    for i in range(dim):
        sum_sq += x[i] * x[i]

    # mean_sq in Q16.16: (sum_sq / dim) * Q16_ONE
    # But we need this as Q16.16 for rsqrt input
    # sum_sq is in raw integer scale, mean_sq_float = sum_sq / dim
    # In Q16.16: mean_sq_q = (sum_sq * Q16_ONE) // dim
    mean_sq_q = (sum_sq * Q16_ONE) // dim

    # Add epsilon to avoid division by zero (eps ~ 1e-5 in Q16.16 ~ 1)
    eps_q = max(1, round(1e-5 * Q16_ONE))
    mean_sq_q = (mean_sq_q + eps_q) & 0xFFFFFFFF

    # rsqrt(mean_sq) in Q16.16
    scale = rsqrt_q16_16(mean_sq_q)
    scale_signed = scale - 0x100000000 if scale >= 0x80000000 else scale

    # Apply: out[i] = clamp((x[i] * gamma[i] * scale) >> 16, -128, 127)
    # Two-step: first multiply x[i] * gamma[i] (int8 * int8 -> int16 range)
    # then multiply by scale and shift
    result = []
    for i in range(dim):
        # x[i] * gamma[i] is at most 127*127 = 16129, fits in int32
        xg = x[i] * gamma[i]
        # xg * scale_signed, then >> 16 to get back to int8-ish range
        val = (xg * scale_signed) >> 16
        result.append(_clamp_i8(val))

    return result


def softmax_q(scores: list[int], n: int) -> list[int]:
    """Compute softmax over int32 scores, returning uint8 probabilities.

    Steps:
    1. Find max score (VMAX equivalent)
    2. Subtract max from all scores
    3. Convert to Q16.16 and compute exp (VEXP equivalent)
    4. Sum all exp values (VREDUCE equivalent)
    5. Normalize: prob[i] = (exp[i] * 255) / sum_exp

    The output is uint8 [0, 255] representing probabilities [0.0, 1.0].

    Args:
        scores: Input scores, shape (n,), int32.
        n: Number of elements.

    Returns:
        Probabilities, shape (n,), uint8 [0, 255].
    """
    if n == 0:
        return []

    # Step 1: Find max
    max_score = scores[0]
    for i in range(1, n):
        if scores[i] > max_score:
            max_score = scores[i]

    # Step 2: Subtract max and convert to Q16.16
    # The scores are in int32 accumulator scale. We need to bring them
    # into a reasonable Q16.16 range for exp. Since attention scores
    # are typically small integers, we scale by a fixed factor.
    # For simplicity, treat the raw int32 difference as the Q16.16 input.
    shifted = []
    for i in range(n):
        diff = scores[i] - max_score  # Always <= 0
        # Convert to Q16.16 (diff is already in reasonable range)
        shifted.append(diff & 0xFFFFFFFF)

    # Step 3: Compute exp in Q16.16
    exp_vals = []
    for i in range(n):
        exp_vals.append(exp_q16_16(shifted[i]))

    # Step 4: Sum
    sum_exp = 0
    for v in exp_vals:
        sum_exp += v

    # Step 5: Normalize to uint8 [0, 255]
    if sum_exp == 0:
        return [0] * n

    probs = []
    for i in range(n):
        # prob = (exp_val * 255) / sum_exp, rounded
        p = (exp_vals[i] * 255 + sum_exp // 2) // sum_exp
        probs.append(max(0, min(255, p)))

    return probs


def linear_q(
    x: list[int],
    weight: list[list[int]],
    bias: list[int],
    in_dim: int,
    out_dim: int,
    shift: int,
) -> list[int]:
    """Quantized linear layer: y = clamp((W @ x + b) >> shift).

    Args:
        x: Input vector, shape (in_dim,), int8 values.
        weight: Weight matrix, shape (out_dim, in_dim), int8 values.
        bias: Bias vector, shape (out_dim,), int32 values.
        in_dim: Input dimension.
        out_dim: Output dimension.
        shift: Right-shift for re-quantization.

    Returns:
        Output vector, shape (out_dim,), int8 values.
    """
    result = []
    for i in range(out_dim):
        acc = _dot_i8(weight[i], x) + bias[i]
        acc = acc >> shift
        result.append(_clamp_i8(acc))
    return result


def attention_single_head_q(
    q: list[int],
    k_cache: list[list[int]],
    v_cache: list[list[int]],
    n_tokens: int,
    head_dim: int,
    attn_scale: int,
) -> list[int]:
    """Single-head attention in quantized domain.

    Computes: attn(Q, K, V) = softmax(Q @ K^T * scale) @ V

    Args:
        q: Query vector for current token, shape (head_dim,), int8.
        k_cache: Key vectors for all tokens, shape (n_tokens, head_dim), int8.
        v_cache: Value vectors for all tokens, shape (n_tokens, head_dim), int8.
        n_tokens: Number of tokens in the cache.
        head_dim: Dimension per head.
        attn_scale: Scale factor in Q16.16 (1/sqrt(head_dim)).

    Returns:
        Attention output, shape (head_dim,), int8.
    """
    # Compute attention scores: Q @ K^T
    scores = []
    for t in range(n_tokens):
        score = _dot_i8(q, k_cache[t])
        # Scale by attn_scale (Q16.16) -> shift back
        attn_scale_signed = attn_scale - 0x100000000 if attn_scale >= 0x80000000 else attn_scale
        score = (score * attn_scale_signed) >> 16
        scores.append(score)

    # Softmax over scores -> uint8 probabilities
    probs = softmax_q(scores, n_tokens)

    # Weighted sum of values: output = sum(prob[t] * V[t])
    output = [0] * head_dim
    for t in range(n_tokens):
        for d in range(head_dim):
            output[d] += probs[t] * v_cache[t][d]

    # Rescale output: probs are uint8 [0,255], v values are int8
    # Product range: 255 * 127 * n_tokens ~ manageable in int32
    # Normalize by dividing by 255 (the softmax output scale)
    result = []
    for d in range(head_dim):
        val = (output[d] + 127) // 255  # Round
        result.append(_clamp_i8(val))

    return result


def multi_head_attention_q(
    x: list[int],
    layer_w: LayerWeights,
    k_cache: list[list[int]],
    v_cache: list[list[int]],
    n_tokens: int,
    config: TransformerConfig,
) -> list[int]:
    """Multi-head attention in quantized domain.

    Projects input to Q, K, V, runs per-head attention, concatenates,
    and applies output projection.

    Args:
        x: Input after RMSNorm, shape (embed_dim,), int8.
        layer_w: Layer weights.
        k_cache: Key cache, shape (n_tokens, embed_dim), int8.
        v_cache: Value cache, shape (n_tokens, embed_dim), int8.
        n_tokens: Number of cached tokens (including current).
        config: Model configuration.

    Returns:
        Attention output, shape (embed_dim,), int8.
    """
    dim = config.embed_dim
    n_heads = config.n_heads
    head_dim = config.head_dim

    # Project to Q, K, V
    q_full = linear_q(x, layer_w.wq, layer_w.bq, dim, dim, layer_w.proj_shift)
    k_full = linear_q(x, layer_w.wk, layer_w.bk, dim, dim, layer_w.proj_shift)
    v_full = linear_q(x, layer_w.wv, layer_w.bv, dim, dim, layer_w.proj_shift)

    # Add current K, V to cache
    k_cache.append(k_full)
    v_cache.append(v_full)

    # Per-head attention
    attn_out = [0] * dim
    for h in range(n_heads):
        h_start = h * head_dim
        h_end = h_start + head_dim

        # Extract head slice from Q
        q_head = q_full[h_start:h_end]

        # Extract head slices from K, V caches
        k_head_cache = [kv[h_start:h_end] for kv in k_cache]
        v_head_cache = [vv[h_start:h_end] for vv in v_cache]

        # Run single-head attention
        head_out = attention_single_head_q(
            q_head, k_head_cache, v_head_cache,
            len(k_cache), head_dim, layer_w.attn_scale,
        )

        # Copy into output
        for d in range(head_dim):
            attn_out[h_start + d] = head_out[d]

    # Output projection
    return linear_q(attn_out, layer_w.wo, layer_w.bo, dim, dim, layer_w.proj_shift)


def feedforward_q(
    x: list[int],
    layer_w: LayerWeights,
    config: TransformerConfig,
) -> list[int]:
    """Feedforward network in quantized domain.

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    Uses GELU activation (via the engine's lookup table approach).

    Args:
        x: Input after RMSNorm, shape (embed_dim,), int8.
        layer_w: Layer weights.
        config: Model configuration.

    Returns:
        FFN output, shape (embed_dim,), int8.
    """
    from .engine import GELU_TABLE

    dim = config.embed_dim
    ff_dim = config.ff_dim

    # First linear: x -> ff_dim
    hidden = linear_q(x, layer_w.w1, layer_w.b1, dim, ff_dim, layer_w.ff_shift)

    # GELU activation
    for i in range(ff_dim):
        # Look up in GELU table: index = value + 128
        table_idx = hidden[i] + 128
        table_idx = max(0, min(255, table_idx))
        hidden[i] = GELU_TABLE[table_idx]

    # Second linear: ff_dim -> dim
    return linear_q(hidden, layer_w.w2, layer_w.b2, ff_dim, dim, layer_w.ff_shift)


def add_residual(x: list[int], residual: list[int], dim: int) -> list[int]:
    """Add residual connection with clamping.

    Args:
        x: Current values, shape (dim,), int8.
        residual: Residual values, shape (dim,), int8.
        dim: Vector dimension.

    Returns:
        Sum clamped to int8, shape (dim,).
    """
    result = []
    for i in range(dim):
        val = x[i] + residual[i]
        result.append(_clamp_i8(val))
    return result


def transformer_block_q(
    x: list[int],
    layer_w: LayerWeights,
    k_cache: list[list[int]],
    v_cache: list[list[int]],
    n_tokens: int,
    config: TransformerConfig,
) -> list[int]:
    """One transformer block: attention + FFN with residual connections.

    Architecture:
        x = x + attention(rmsnorm(x))
        x = x + ffn(rmsnorm(x))

    Args:
        x: Input, shape (embed_dim,), int8.
        layer_w: Layer weights.
        k_cache: Key cache for this layer (mutated by appending).
        v_cache: Value cache for this layer (mutated by appending).
        n_tokens: Number of tokens processed so far.
        config: Model configuration.

    Returns:
        Output, shape (embed_dim,), int8.
    """
    dim = config.embed_dim

    # Attention sub-layer
    normed = rmsnorm_q(x, layer_w.ln1_gamma, dim)
    attn_out = multi_head_attention_q(normed, layer_w, k_cache, v_cache, n_tokens, config)
    x = add_residual(attn_out, x, dim)

    # FFN sub-layer
    normed = rmsnorm_q(x, layer_w.ln2_gamma, dim)
    ffn_out = feedforward_q(normed, layer_w, config)
    x = add_residual(ffn_out, x, dim)

    return x


def transformer_forward_q(
    tokens: list[int],
    weights: TransformerWeights,
    config: TransformerConfig,
) -> list[int]:
    """Full transformer forward pass in quantized domain.

    Processes a sequence of tokens and returns logits for the last token.

    Architecture:
        1. Token embedding + positional embedding
        2. N transformer blocks (attention + FFN)
        3. Final RMSNorm
        4. Output projection -> logits

    Args:
        tokens: Input token IDs, shape (seq_len,), each in [0, vocab_size).
        weights: Quantized model weights.
        config: Model configuration.

    Returns:
        Output logits for the last token, shape (vocab_size,), int32.
    """
    seq_len = len(tokens)
    dim = config.embed_dim

    # Initialize KV caches for each layer
    k_caches: list[list[list[int]]] = [[] for _ in range(config.n_layers)]
    v_caches: list[list[list[int]]] = [[] for _ in range(config.n_layers)]

    # Process each token
    last_x: list[int] = [0] * dim

    for pos in range(seq_len):
        token = tokens[pos]

        # Embedding: token_embed[token] + pos_embed[pos]
        x = []
        for d in range(dim):
            val = weights.token_embed[token][d] + weights.pos_embed[pos][d]
            x.append(_clamp_i8(val))

        # Transformer blocks
        for layer_idx in range(config.n_layers):
            x = transformer_block_q(
                x, weights.layers[layer_idx],
                k_caches[layer_idx], v_caches[layer_idx],
                pos, config,
            )

        last_x = x

    # Final RMSNorm
    last_x = rmsnorm_q(last_x, weights.ln_final_gamma, dim)

    # Output projection: logits = last_x @ output_proj^T + output_bias
    logits = []
    for i in range(config.vocab_size):
        acc = _dot_i8(weights.output_proj[i], last_x) + weights.output_bias[i]
        logits.append(acc)

    return logits


def predict_next_token(
    tokens: list[int],
    weights: TransformerWeights,
    config: TransformerConfig,
) -> int:
    """Predict the next token given an input sequence.

    Args:
        tokens: Input token IDs.
        weights: Quantized model weights.
        config: Model configuration.

    Returns:
        Predicted next token ID (argmax of logits).
    """
    logits = transformer_forward_q(tokens, weights, config)
    max_idx = 0
    max_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > max_val:
            max_val = logits[i]
            max_idx = i
    return max_idx
