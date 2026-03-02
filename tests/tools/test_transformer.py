"""Tests for float32 transformer inference reference implementation."""

from __future__ import annotations

import math

from riscv_npu.tools.transformer import (
    TransformerConfig,
    TransformerWeights,
    LayerWeights,
    dot_f32,
    rmsnorm,
    softmax,
    linear,
    gelu,
    attention_single_head,
    transformer_forward,
    predict_next_token,
)


# ==================== Float ops tests ====================


class TestDotF32:
    """Tests for dot_f32."""

    def test_dot_f32_basic(self) -> None:
        """Dot product of known vectors."""
        result = dot_f32([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert abs(result - 32.0) < 1e-6  # 4+10+18

    def test_dot_f32_zero(self) -> None:
        """Zero vector gives zero."""
        result = dot_f32([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert result == 0.0

    def test_dot_f32_negative(self) -> None:
        """Dot product with negatives."""
        result = dot_f32([1.0, -1.0], [1.0, 1.0])
        assert abs(result) < 1e-6

    def test_dot_f32_single(self) -> None:
        """Single element."""
        result = dot_f32([5.0], [7.0])
        assert abs(result - 35.0) < 1e-6

    def test_dot_f32_empty(self) -> None:
        """Empty vectors produce 0."""
        assert dot_f32([], []) == 0.0


# ==================== RMSNorm tests ====================


class TestRMSNorm:
    """Tests for rmsnorm."""

    def test_rmsnorm_ones(self) -> None:
        """All-ones input, gamma=1.0 -> each element ~1.0.

        With x=[1,1,...,1] (dim=4), mean(x^2)=1.0, rsqrt(1.0+eps)~1.0.
        So output = x * gamma * 1.0 ~= [1, 1, 1, 1].
        """
        dim = 4
        x = [1.0] * dim
        gamma = [1.0] * dim
        result = rmsnorm(x, gamma, dim)
        for val in result:
            assert abs(val - 1.0) < 0.01

    def test_rmsnorm_scaling(self) -> None:
        """Known input, verify normalization.

        With x=[1, 2, 3, 4], mean(x^2)=(1+4+9+16)/4=7.5
        rsqrt(7.5+eps) ~ 0.3651
        output[i] = x[i] * gamma[i] * 0.3651
        """
        dim = 4
        x = [1.0, 2.0, 3.0, 4.0]
        gamma = [1.0] * dim
        result = rmsnorm(x, gamma, dim)
        expected_scale = 1.0 / math.sqrt(7.5 + 1e-5)
        for i in range(dim):
            expected = x[i] * expected_scale
            assert abs(result[i] - expected) < 1e-5

    def test_rmsnorm_zero_input(self) -> None:
        """All zeros -> zeros out."""
        dim = 4
        x = [0.0, 0.0, 0.0, 0.0]
        gamma = [1.0, 1.0, 1.0, 1.0]
        result = rmsnorm(x, gamma, dim)
        for val in result:
            assert abs(val) < 1e-5

    def test_rmsnorm_with_gamma(self) -> None:
        """RMSNorm applies gamma scaling."""
        dim = 2
        x = [1.0, 1.0]
        gamma = [2.0, 3.0]
        result = rmsnorm(x, gamma, dim)
        # mean(x^2) = 1.0, scale = 1/sqrt(1.0+eps) ~ 1.0
        assert abs(result[0] - 2.0) < 0.01
        assert abs(result[1] - 3.0) < 0.01


# ==================== Softmax tests ====================


class TestSoftmax:
    """Tests for softmax."""

    def test_softmax_uniform(self) -> None:
        """Equal scores -> equal probabilities."""
        n = 4
        scores = [1.0, 1.0, 1.0, 1.0]
        probs = softmax(scores, n)
        for p in probs:
            assert abs(p - 0.25) < 1e-6

    def test_softmax_one_hot(self) -> None:
        """One large score dominates."""
        n = 3
        scores = [0.0, 100.0, 0.0]
        probs = softmax(scores, n)
        assert probs[1] > 0.99
        assert probs[0] < 0.01
        assert probs[2] < 0.01

    def test_softmax_sums_to_one(self) -> None:
        """Probabilities sum to ~1.0."""
        n = 4
        scores = [1.0, 2.0, 0.5, 1.5]
        probs = softmax(scores, n)
        total = sum(probs)
        assert abs(total - 1.0) < 1e-6

    def test_softmax_single(self) -> None:
        """Softmax of a single element is 1.0."""
        probs = softmax([42.0], 1)
        assert abs(probs[0] - 1.0) < 1e-6

    def test_softmax_empty(self) -> None:
        """Softmax of empty list returns empty."""
        assert softmax([], 0) == []

    def test_softmax_negative_scores(self) -> None:
        """Softmax handles negative scores."""
        n = 3
        scores = [-1.0, -2.0, -3.0]
        probs = softmax(scores, n)
        total = sum(probs)
        assert abs(total - 1.0) < 1e-6
        # Largest (least negative) should have highest probability
        assert probs[0] > probs[1] > probs[2]


# ==================== Linear layer tests ====================


class TestLinear:
    """Tests for linear."""

    def test_linear_identity_weight(self) -> None:
        """Identity weight matrix passes input through."""
        in_dim = 4
        out_dim = 4
        weight = [[0.0] * in_dim for _ in range(out_dim)]
        for i in range(out_dim):
            weight[i][i] = 1.0
        bias = [0.0] * out_dim
        x = [1.0, 2.0, 3.0, 4.0]
        result = linear(x, weight, bias, in_dim, out_dim)
        for i in range(out_dim):
            assert abs(result[i] - x[i]) < 1e-6

    def test_linear_with_bias(self) -> None:
        """Verify bias is added."""
        in_dim = 2
        out_dim = 1
        weight = [[1.0, 1.0]]
        bias = [10.0]
        x = [3.0, 4.0]
        result = linear(x, weight, bias, in_dim, out_dim)
        # 1*3 + 1*4 + 10 = 17
        assert abs(result[0] - 17.0) < 1e-6

    def test_linear_scaling(self) -> None:
        """Linear with scaling weights."""
        in_dim = 2
        out_dim = 2
        weight = [[2.0, 0.0], [0.0, 3.0]]
        bias = [0.0, 0.0]
        x = [5.0, 7.0]
        result = linear(x, weight, bias, in_dim, out_dim)
        assert abs(result[0] - 10.0) < 1e-6
        assert abs(result[1] - 21.0) < 1e-6


# ==================== GELU tests ====================


class TestGelu:
    """Tests for gelu."""

    def test_gelu_zero(self) -> None:
        """GELU(0) = 0."""
        assert abs(gelu(0.0)) < 1e-6

    def test_gelu_positive(self) -> None:
        """GELU of a positive value is positive and less than x."""
        result = gelu(1.0)
        assert result > 0.0
        assert result < 1.0

    def test_gelu_negative(self) -> None:
        """GELU of a negative value is close to zero for large negative."""
        result = gelu(-3.0)
        assert abs(result) < 0.01

    def test_gelu_large_positive(self) -> None:
        """GELU(x) ~= x for large positive x."""
        result = gelu(5.0)
        assert abs(result - 5.0) < 0.01


# ==================== Attention tests ====================


class TestAttention:
    """Tests for attention."""

    def test_attention_single_head(self) -> None:
        """Attention with a single key/value pair returns ~V."""
        head_dim = 4
        q = [1.0, 0.0, 0.0, 0.0]
        k_cache = [[1.0, 0.0, 0.0, 0.0]]
        v_cache = [[5.0, 10.0, 15.0, 20.0]]
        attn_scale = 1.0 / math.sqrt(head_dim)
        result = attention_single_head(q, k_cache, v_cache, 1, head_dim, attn_scale)
        # With single token, softmax gives 1.0 to the only option
        assert len(result) == head_dim
        for d in range(head_dim):
            assert abs(result[d] - v_cache[0][d]) < 1e-5

    def test_attention_deterministic(self) -> None:
        """Same input produces same output."""
        head_dim = 4
        q = [0.5, 0.3, -0.2, 0.1]
        k_cache = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.3, 0.1, -0.1]]
        v_cache = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        attn_scale = 0.5
        result1 = attention_single_head(q, k_cache, v_cache, 2, head_dim, attn_scale)
        result2 = attention_single_head(q, k_cache, v_cache, 2, head_dim, attn_scale)
        for d in range(head_dim):
            assert abs(result1[d] - result2[d]) < 1e-10

    def test_attention_output_length(self) -> None:
        """Attention output has correct dimension."""
        head_dim = 8
        q = [0.1] * head_dim
        k_cache = [[0.1] * head_dim, [0.2] * head_dim]
        v_cache = [[0.3] * head_dim, [0.4] * head_dim]
        attn_scale = 1.0 / math.sqrt(head_dim)
        result = attention_single_head(q, k_cache, v_cache, 2, head_dim, attn_scale)
        assert len(result) == head_dim


# ==================== Full transformer tests ====================


def _make_tiny_config() -> TransformerConfig:
    """Create a minimal config for testing."""
    return TransformerConfig(
        vocab_size=4,
        embed_dim=8,
        n_heads=2,
        head_dim=4,
        n_layers=1,
        context_len=4,
        ff_dim=16,
    )


def _make_tiny_weights(config: TransformerConfig) -> TransformerWeights:
    """Create tiny deterministic float weights for testing.

    Uses deterministic values (not random) for reproducibility.
    """
    dim = config.embed_dim
    ff_dim = config.ff_dim
    vocab = config.vocab_size
    ctx = config.context_len

    def make_matrix(rows: int, cols: int) -> list[list[float]]:
        """Create a small deterministic float matrix."""
        return [
            [((r * 7 + c * 13 + 3) % 21 - 10) / 10.0 for c in range(cols)]
            for r in range(rows)
        ]

    def make_vector(size: int) -> list[float]:
        """Create a small deterministic float vector."""
        return [((i * 11 + 5) % 21 - 10) / 10.0 for i in range(size)]

    layer = LayerWeights(
        ln1_gamma=[1.0] * dim,
        wq=make_matrix(dim, dim),
        bq=make_vector(dim),
        wk=make_matrix(dim, dim),
        bk=make_vector(dim),
        wv=make_matrix(dim, dim),
        bv=make_vector(dim),
        wo=make_matrix(dim, dim),
        bo=make_vector(dim),
        ln2_gamma=[1.0] * dim,
        w1=make_matrix(ff_dim, dim),
        b1=make_vector(ff_dim),
        w2=make_matrix(dim, ff_dim),
        b2=make_vector(dim),
    )

    return TransformerWeights(
        token_embed=make_matrix(vocab, dim),
        pos_embed=make_matrix(ctx, dim),
        layers=[layer],
        ln_final_gamma=[1.0] * dim,
        output_proj=make_matrix(vocab, dim),
        output_bias=make_vector(vocab),
    )


class TestTransformerForward:
    """Tests for the full transformer forward pass."""

    def test_transformer_forward_deterministic(self) -> None:
        """Same tokens + weights -> same logits."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        tokens = [0, 1, 2]
        logits1 = transformer_forward(tokens, weights, config)
        logits2 = transformer_forward(tokens, weights, config)
        assert len(logits1) == config.vocab_size
        for i in range(len(logits1)):
            assert abs(logits1[i] - logits2[i]) < 1e-10

    def test_output_shape(self) -> None:
        """Forward pass returns logits of correct size."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        tokens = [0, 1, 2]
        logits = transformer_forward(tokens, weights, config)
        assert len(logits) == config.vocab_size

    def test_predict_next_token(self) -> None:
        """predict_next_token returns a valid token ID."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        tokens = [0, 1]
        predicted = predict_next_token(tokens, weights, config)
        assert 0 <= predicted < config.vocab_size

    def test_different_inputs_different_outputs(self) -> None:
        """Different input sequences produce different outputs."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        logits1 = transformer_forward([0, 1], weights, config)
        logits2 = transformer_forward([2, 3], weights, config)
        # They should differ
        assert logits1 != logits2

    def test_single_token(self) -> None:
        """Forward pass works with a single token."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        logits = transformer_forward([0], weights, config)
        assert len(logits) == config.vocab_size

    def test_logits_are_float(self) -> None:
        """Logits are float values."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        logits = transformer_forward([0, 1], weights, config)
        for v in logits:
            assert isinstance(v, float)
