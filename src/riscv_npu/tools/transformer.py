"""Float32 transformer inference reference implementation.

Pure-Python implementation of a character-level transformer that matches
the firmware behavior exactly. All operations use float32 weights and
activations -- no quantization, no Q16.16, no int8.

This module is used to validate firmware output in integration tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
class LayerWeights:
    """Float weights for one transformer layer.

    Attributes:
        ln1_gamma: RMSNorm gamma for attention, shape (embed_dim,).
        wq: Query projection, shape (embed_dim, embed_dim).
        bq: Query bias, shape (embed_dim,).
        wk: Key projection, shape (embed_dim, embed_dim).
        bk: Key bias, shape (embed_dim,).
        wv: Value projection, shape (embed_dim, embed_dim).
        bv: Value bias, shape (embed_dim,).
        wo: Output projection, shape (embed_dim, embed_dim).
        bo: Output bias, shape (embed_dim,).
        ln2_gamma: RMSNorm gamma for FFN, shape (embed_dim,).
        w1: FFN first linear, shape (ff_dim, embed_dim).
        b1: FFN first bias, shape (ff_dim,).
        w2: FFN second linear, shape (embed_dim, ff_dim).
        b2: FFN second bias, shape (embed_dim,).
    """

    ln1_gamma: list[float]
    wq: list[list[float]]
    bq: list[float]
    wk: list[list[float]]
    bk: list[float]
    wv: list[list[float]]
    bv: list[float]
    wo: list[list[float]]
    bo: list[float]
    ln2_gamma: list[float]
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]


@dataclass
class TransformerWeights:
    """Float weights for the transformer model.

    All weight matrices are float (list of lists of float).
    All bias vectors are float (list of float).

    Attributes:
        token_embed: Token embedding table, shape (vocab_size, embed_dim).
        pos_embed: Positional embedding table, shape (context_len, embed_dim).
        layers: Per-layer weights (list of LayerWeights).
        ln_final_gamma: Final layer norm gamma, shape (embed_dim,).
        output_proj: Output projection, shape (vocab_size, embed_dim).
        output_bias: Output bias, shape (vocab_size,).
    """

    token_embed: list[list[float]]
    pos_embed: list[list[float]]
    layers: list[LayerWeights]
    ln_final_gamma: list[float]
    output_proj: list[list[float]]
    output_bias: list[float]


def dot_f32(a: list[float], b: list[float]) -> float:
    """Dot product of two float vectors.

    Args:
        a: First float vector.
        b: Second float vector (same length as a).

    Returns:
        Sum of element-wise products.
    """
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def rmsnorm(x: list[float], gamma: list[float], dim: int) -> list[float]:
    """RMSNorm: x * gamma * rsqrt(mean(x^2) + eps).

    Normalizes the input vector using root-mean-square normalization
    with a learned scale parameter gamma.

    Args:
        x: Input vector, shape (dim,).
        gamma: Learned scale, shape (dim,).
        dim: Vector dimension.

    Returns:
        Normalized vector, shape (dim,).
    """
    eps = 1e-5

    # Compute mean of squares
    sum_sq = 0.0
    for i in range(dim):
        sum_sq += x[i] * x[i]
    mean_sq = sum_sq / dim

    # Reciprocal square root
    scale = 1.0 / math.sqrt(mean_sq + eps)

    # Apply normalization with gamma scaling
    result = []
    for i in range(dim):
        result.append(x[i] * gamma[i] * scale)
    return result


def softmax(scores: list[float], n: int) -> list[float]:
    """Numerically stable softmax: subtract max, exp, normalize.

    Args:
        scores: Input scores, shape (n,).
        n: Number of elements.

    Returns:
        Probabilities, shape (n,), summing to ~1.0.
    """
    if n == 0:
        return []

    # Find max for numerical stability
    max_score = scores[0]
    for i in range(1, n):
        if scores[i] > max_score:
            max_score = scores[i]

    # Subtract max and compute exp
    exp_vals = []
    for i in range(n):
        exp_vals.append(math.exp(scores[i] - max_score))

    # Sum
    sum_exp = 0.0
    for v in exp_vals:
        sum_exp += v

    # Normalize
    if sum_exp == 0.0:
        return [0.0] * n

    probs = []
    for i in range(n):
        probs.append(exp_vals[i] / sum_exp)
    return probs


def linear(
    x: list[float],
    weight: list[list[float]],
    bias: list[float],
    in_dim: int,
    out_dim: int,
) -> list[float]:
    """Linear layer: y[i] = dot(weight[i], x) + bias[i].

    Args:
        x: Input vector, shape (in_dim,).
        weight: Weight matrix, shape (out_dim, in_dim).
        bias: Bias vector, shape (out_dim,).
        in_dim: Input dimension.
        out_dim: Output dimension.

    Returns:
        Output vector, shape (out_dim,).
    """
    result = []
    for i in range(out_dim):
        acc = dot_f32(weight[i], x) + bias[i]
        result.append(acc)
    return result


def gelu(x: float) -> float:
    """GELU activation.

    Computes the exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).

    Args:
        x: Input value.

    Returns:
        GELU(x).
    """
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def attention_single_head(
    q: list[float],
    k_cache: list[list[float]],
    v_cache: list[list[float]],
    n_tokens: int,
    head_dim: int,
    attn_scale: float,
) -> list[float]:
    """Single-head attention in float domain.

    Computes: attn(Q, K, V) = softmax(Q @ K^T * scale) @ V

    Args:
        q: Query vector for current token, shape (head_dim,).
        k_cache: Key vectors for all tokens, shape (n_tokens, head_dim).
        v_cache: Value vectors for all tokens, shape (n_tokens, head_dim).
        n_tokens: Number of tokens in the cache.
        head_dim: Dimension per head.
        attn_scale: Scale factor (1/sqrt(head_dim)).

    Returns:
        Attention output, shape (head_dim,).
    """
    # Compute attention scores: Q @ K^T * scale
    scores = []
    for t in range(n_tokens):
        score = dot_f32(q, k_cache[t]) * attn_scale
        scores.append(score)

    # Softmax over scores
    probs = softmax(scores, n_tokens)

    # Weighted sum of values
    output = [0.0] * head_dim
    for t in range(n_tokens):
        for d in range(head_dim):
            output[d] += probs[t] * v_cache[t][d]

    return output


def multi_head_attention(
    x: list[float],
    layer_w: LayerWeights,
    k_cache: list[list[float]],
    v_cache: list[list[float]],
    n_tokens: int,
    config: TransformerConfig,
) -> list[float]:
    """Multi-head attention in float domain.

    Projects input to Q, K, V, runs per-head attention, concatenates,
    and applies output projection.

    Args:
        x: Input after RMSNorm, shape (embed_dim,).
        layer_w: Layer weights.
        k_cache: Key cache, shape (n_tokens, embed_dim).
        v_cache: Value cache, shape (n_tokens, embed_dim).
        n_tokens: Number of tokens processed so far.
        config: Model configuration.

    Returns:
        Attention output, shape (embed_dim,).
    """
    dim = config.embed_dim
    n_heads = config.n_heads
    head_dim = config.head_dim

    # Project to Q, K, V
    q_full = linear(x, layer_w.wq, layer_w.bq, dim, dim)
    k_full = linear(x, layer_w.wk, layer_w.bk, dim, dim)
    v_full = linear(x, layer_w.wv, layer_w.bv, dim, dim)

    # Add current K, V to cache
    k_cache.append(k_full)
    v_cache.append(v_full)

    attn_scale = 1.0 / math.sqrt(head_dim)

    # Per-head attention
    attn_out = [0.0] * dim
    for h in range(n_heads):
        h_start = h * head_dim
        h_end = h_start + head_dim

        # Extract head slice from Q
        q_head = q_full[h_start:h_end]

        # Extract head slices from K, V caches
        k_head_cache = [kv[h_start:h_end] for kv in k_cache]
        v_head_cache = [vv[h_start:h_end] for vv in v_cache]

        # Run single-head attention
        head_out = attention_single_head(
            q_head, k_head_cache, v_head_cache,
            len(k_cache), head_dim, attn_scale,
        )

        # Copy into output
        for d in range(head_dim):
            attn_out[h_start + d] = head_out[d]

    # Output projection
    return linear(attn_out, layer_w.wo, layer_w.bo, dim, dim)


def feedforward(
    x: list[float],
    layer_w: LayerWeights,
    config: TransformerConfig,
) -> list[float]:
    """Feedforward network in float domain.

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    Args:
        x: Input after RMSNorm, shape (embed_dim,).
        layer_w: Layer weights.
        config: Model configuration.

    Returns:
        FFN output, shape (embed_dim,).
    """
    dim = config.embed_dim
    ff_dim = config.ff_dim

    # First linear: dim -> ff_dim
    hidden = linear(x, layer_w.w1, layer_w.b1, dim, ff_dim)

    # GELU activation
    for i in range(ff_dim):
        hidden[i] = gelu(hidden[i])

    # Second linear: ff_dim -> dim
    return linear(hidden, layer_w.w2, layer_w.b2, ff_dim, dim)


def transformer_block(
    x: list[float],
    layer_w: LayerWeights,
    k_cache: list[list[float]],
    v_cache: list[list[float]],
    n_tokens: int,
    config: TransformerConfig,
) -> list[float]:
    """One transformer block: attention + FFN with residual connections.

    Architecture:
        x = x + attention(rmsnorm(x))
        x = x + ffn(rmsnorm(x))

    Args:
        x: Input, shape (embed_dim,).
        layer_w: Layer weights.
        k_cache: Key cache for this layer (mutated by appending).
        v_cache: Value cache for this layer (mutated by appending).
        n_tokens: Number of tokens processed so far.
        config: Model configuration.

    Returns:
        Output, shape (embed_dim,).
    """
    dim = config.embed_dim

    # Attention sub-layer
    normed = rmsnorm(x, layer_w.ln1_gamma, dim)
    attn_out = multi_head_attention(
        normed, layer_w, k_cache, v_cache, n_tokens, config,
    )
    # Residual connection (simple float addition)
    x = [x[i] + attn_out[i] for i in range(dim)]

    # FFN sub-layer
    normed = rmsnorm(x, layer_w.ln2_gamma, dim)
    ffn_out = feedforward(normed, layer_w, config)
    # Residual connection
    x = [x[i] + ffn_out[i] for i in range(dim)]

    return x


def transformer_forward(
    tokens: list[int],
    weights: TransformerWeights,
    config: TransformerConfig,
) -> list[float]:
    """Full transformer forward pass in float domain.

    Processes a sequence of tokens and returns logits for the last token.

    Architecture:
        1. Token embedding + positional embedding
        2. N transformer blocks (attention + FFN)
        3. Final RMSNorm
        4. Output projection -> logits

    Args:
        tokens: Input token IDs, shape (seq_len,), each in [0, vocab_size).
        weights: Float model weights.
        config: Model configuration.

    Returns:
        Output logits for the last token, shape (vocab_size,).
    """
    seq_len = len(tokens)
    dim = config.embed_dim

    # Initialize KV caches for each layer
    k_caches: list[list[list[float]]] = [[] for _ in range(config.n_layers)]
    v_caches: list[list[list[float]]] = [[] for _ in range(config.n_layers)]

    # Process each token
    last_x: list[float] = [0.0] * dim

    for pos in range(seq_len):
        token = tokens[pos]

        # Embedding: token_embed[token] + pos_embed[pos]
        x = []
        for d in range(dim):
            x.append(weights.token_embed[token][d] + weights.pos_embed[pos][d])

        # Transformer blocks
        for layer_idx in range(config.n_layers):
            x = transformer_block(
                x, weights.layers[layer_idx],
                k_caches[layer_idx], v_caches[layer_idx],
                pos, config,
            )

        last_x = x

    # Final RMSNorm
    last_x = rmsnorm(last_x, weights.ln_final_gamma, dim)

    # Output projection: logits = last_x @ output_proj^T + output_bias
    logits = []
    for i in range(config.vocab_size):
        acc = dot_f32(weights.output_proj[i], last_x) + weights.output_bias[i]
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
        weights: Float model weights.
        config: Model configuration.

    Returns:
        Predicted next token ID (argmax of logits).
    """
    logits = transformer_forward(tokens, weights, config)
    max_idx = 0
    max_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > max_val:
            max_val = logits[i]
            max_idx = i
    return max_idx
