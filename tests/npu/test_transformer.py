"""Tests for quantized transformer inference reference implementation."""

from __future__ import annotations

import math

from riscv_npu.npu.engine import Q16_ONE
from riscv_npu.npu.transformer import (
    TransformerConfig,
    TransformerWeights,
    LayerWeights,
    _clamp_i8,
    _dot_i8,
    rmsnorm_q,
    softmax_q,
    linear_q,
    attention_single_head_q,
    add_residual,
    transformer_forward_q,
    predict_next_token,
)


# ==================== Helper tests ====================


class TestClampI8:
    """Tests for _clamp_i8."""

    def test_in_range(self) -> None:
        """Values in [-128, 127] pass through."""
        assert _clamp_i8(0) == 0
        assert _clamp_i8(127) == 127
        assert _clamp_i8(-128) == -128

    def test_overflow(self) -> None:
        """Values outside range are clamped."""
        assert _clamp_i8(200) == 127
        assert _clamp_i8(-200) == -128
        assert _clamp_i8(1000) == 127


class TestDotI8:
    """Tests for _dot_i8."""

    def test_basic(self) -> None:
        """Basic dot product."""
        assert _dot_i8([1, 2, 3], [4, 5, 6]) == 32  # 4+10+18

    def test_negative(self) -> None:
        """Dot product with negatives."""
        assert _dot_i8([1, -1], [1, 1]) == 0

    def test_empty(self) -> None:
        """Empty vectors produce 0."""
        assert _dot_i8([], []) == 0

    def test_single(self) -> None:
        """Single element."""
        assert _dot_i8([5], [7]) == 35


# ==================== RMSNorm tests ====================


class TestRMSNorm:
    """Tests for rmsnorm_q."""

    def test_rmsnorm_uniform_input(self) -> None:
        """RMSNorm with all-ones input and gamma=1.

        With x=[1,1,...,1] (dim=4), mean(x^2)=1.0, rsqrt(1.0)=1.0.
        So output = x * gamma * 1.0 = x * gamma.
        With gamma=[1,1,...,1], output should be [1,1,...,1].
        """
        dim = 4
        x = [1] * dim
        gamma = [1] * dim
        result = rmsnorm_q(x, gamma, dim)
        # Output should be close to 1 (with some Q16.16 rounding)
        for val in result:
            assert abs(val - 1) <= 1

    def test_rmsnorm_scaling(self) -> None:
        """RMSNorm correctly normalizes varying inputs.

        With x=[10, 20, 30, 40], mean(x^2) = (100+400+900+1600)/4 = 750
        rsqrt(750) ~ 0.0365
        output[i] = x[i] * gamma[i] * 0.0365
        """
        dim = 4
        x = [10, 20, 30, 40]
        gamma = [64, 64, 64, 64]  # gamma ~ 0.5 * 128 scale
        result = rmsnorm_q(x, gamma, dim)
        # All values should be non-zero and reasonable
        assert all(-128 <= v <= 127 for v in result)
        # Relative ordering should be preserved: result[3] > result[0]
        assert result[3] >= result[0]

    def test_rmsnorm_zero_input(self) -> None:
        """RMSNorm with all-zero input (edge case)."""
        dim = 4
        x = [0, 0, 0, 0]
        gamma = [64, 64, 64, 64]
        result = rmsnorm_q(x, gamma, dim)
        # All outputs should be 0 (0 * anything = 0)
        assert result == [0, 0, 0, 0]

    def test_rmsnorm_output_in_range(self) -> None:
        """RMSNorm outputs are always in int8 range."""
        dim = 8
        x = [127, -128, 50, -50, 0, 100, -100, 1]
        gamma = [127, 127, 127, 127, 127, 127, 127, 127]
        result = rmsnorm_q(x, gamma, dim)
        for v in result:
            assert -128 <= v <= 127


# ==================== Softmax tests ====================


class TestSoftmaxQ:
    """Tests for softmax_q."""

    def test_softmax_uniform(self) -> None:
        """Softmax of equal values gives uniform distribution."""
        n = 4
        scores = [100, 100, 100, 100]
        probs = softmax_q(scores, n)
        # All probs should be ~255/4 = ~64
        for p in probs:
            assert abs(p - 64) <= 2

    def test_softmax_sums_to_255(self) -> None:
        """Softmax probabilities should approximately sum to 255."""
        n = 4
        scores = [100, 200, 50, 150]
        probs = softmax_q(scores, n)
        total = sum(probs)
        assert abs(total - 255) <= 5  # Allow small rounding error

    def test_softmax_peak(self) -> None:
        """Softmax concentrates on the largest score."""
        n = 3
        # Large difference means softmax should put most mass on index 1
        scores = [0, 1000000, 0]
        probs = softmax_q(scores, n)
        assert probs[1] > 200  # Should be close to 255
        assert probs[0] < 20
        assert probs[2] < 20

    def test_softmax_single(self) -> None:
        """Softmax of a single element is 255."""
        probs = softmax_q([42], 1)
        assert probs[0] == 255

    def test_softmax_empty(self) -> None:
        """Softmax of empty list returns empty."""
        assert softmax_q([], 0) == []

    def test_softmax_all_zero(self) -> None:
        """Softmax of all-zero scores gives uniform."""
        n = 3
        probs = softmax_q([0, 0, 0], n)
        for p in probs:
            assert abs(p - 85) <= 2  # 255/3 ~ 85


# ==================== Linear layer tests ====================


class TestLinearQ:
    """Tests for linear_q."""

    def test_identity_weight(self) -> None:
        """Identity-like weight matrix passes input through (with shift)."""
        in_dim = 4
        out_dim = 4
        # Simple diagonal weight matrix
        weight = [[0] * in_dim for _ in range(out_dim)]
        for i in range(out_dim):
            weight[i][i] = 64  # ~0.5 in int8 scale
        bias = [0] * out_dim
        x = [10, 20, 30, 40]
        result = linear_q(x, weight, bias, in_dim, out_dim, shift=0)
        # output[i] = x[i] * 64 + 0, but this may overflow int8 range
        assert result[0] == 127  # 10*64 = 640 -> clamp to 127

    def test_with_shift(self) -> None:
        """Linear with shift re-quantizes output."""
        in_dim = 2
        out_dim = 1
        weight = [[10, 20]]
        bias = [0]
        x = [5, 5]
        # acc = 10*5 + 20*5 = 150
        # shifted = 150 >> 2 = 37
        result = linear_q(x, weight, bias, in_dim, out_dim, shift=2)
        assert result[0] == 37

    def test_with_bias(self) -> None:
        """Linear with non-zero bias."""
        in_dim = 2
        out_dim = 1
        weight = [[1, 1]]
        bias = [100]
        x = [10, 20]
        # acc = 1*10 + 1*20 + 100 = 130
        # shifted = 130 >> 1 = 65
        result = linear_q(x, weight, bias, in_dim, out_dim, shift=1)
        assert result[0] == 65


# ==================== Attention tests ====================


class TestAttentionSingleHead:
    """Tests for attention_single_head_q."""

    def test_single_token(self) -> None:
        """Attention with a single key/value pair."""
        head_dim = 4
        q = [10, 20, 30, 40]
        k_cache = [[10, 20, 30, 40]]  # Same as query
        v_cache = [[5, 10, 15, 20]]
        attn_scale = Q16_ONE  # 1.0 scale
        result = attention_single_head_q(q, k_cache, v_cache, 1, head_dim, attn_scale)
        # With single token, softmax gives 255 (100%) to the only option
        # Output should be close to v_cache[0]
        assert len(result) == head_dim
        for d in range(head_dim):
            assert abs(result[d] - v_cache[0][d]) <= 1

    def test_output_length(self) -> None:
        """Attention output has correct dimension."""
        head_dim = 8
        q = [1] * head_dim
        k_cache = [[1] * head_dim, [2] * head_dim]
        v_cache = [[3] * head_dim, [4] * head_dim]
        attn_scale = Q16_ONE // 4
        result = attention_single_head_q(q, k_cache, v_cache, 2, head_dim, attn_scale)
        assert len(result) == head_dim


# ==================== Residual tests ====================


class TestAddResidual:
    """Tests for add_residual."""

    def test_basic_add(self) -> None:
        """Simple addition."""
        assert add_residual([10, 20], [5, 10], 2) == [15, 30]

    def test_clamping(self) -> None:
        """Overflow is clamped."""
        assert add_residual([127, -128], [1, -1], 2) == [127, -128]

    def test_zero_residual(self) -> None:
        """Adding zero residual preserves values."""
        assert add_residual([50, -50], [0, 0], 2) == [50, -50]


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
    """Create tiny random-like weights for testing.

    Uses deterministic values (not random) for reproducibility.
    """
    dim = config.embed_dim
    ff_dim = config.ff_dim
    vocab = config.vocab_size
    ctx = config.context_len

    def make_matrix(rows: int, cols: int) -> list[list[int]]:
        """Create a small deterministic matrix."""
        return [
            [((r * 7 + c * 13 + 3) % 21) - 10 for c in range(cols)]
            for r in range(rows)
        ]

    def make_vector_i8(size: int) -> list[int]:
        """Create a small deterministic int8 vector."""
        return [((i * 11 + 5) % 21) - 10 for i in range(size)]

    def make_vector_i32(size: int) -> list[int]:
        """Create a small deterministic int32 bias vector."""
        return [((i * 17 + 3) % 41) - 20 for i in range(size)]

    layer = LayerWeights(
        ln1_gamma=[64] * dim,  # ~0.5 in int8 scale
        wq=make_matrix(dim, dim),
        wk=make_matrix(dim, dim),
        wv=make_matrix(dim, dim),
        wo=make_matrix(dim, dim),
        bq=make_vector_i32(dim),
        bk=make_vector_i32(dim),
        bv=make_vector_i32(dim),
        bo=make_vector_i32(dim),
        attn_scale=Q16_ONE // 2,  # 0.5 = 1/sqrt(4)
        ln2_gamma=[64] * dim,
        w1=make_matrix(ff_dim, dim),
        w2=make_matrix(dim, ff_dim),
        b1=make_vector_i32(ff_dim),
        b2=make_vector_i32(dim),
        proj_shift=4,
        ff_shift=4,
    )

    return TransformerWeights(
        token_embed=make_matrix(vocab, dim),
        pos_embed=make_matrix(ctx, dim),
        layers=[layer],
        ln_final_gamma=[64] * dim,
        output_proj=make_matrix(vocab, dim),
        output_bias=make_vector_i32(vocab),
        embed_scale=Q16_ONE,
    )


class TestTransformerForward:
    """Tests for the full transformer forward pass."""

    def test_output_shape(self) -> None:
        """Forward pass returns logits of correct size."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        tokens = [0, 1, 2]
        logits = transformer_forward_q(tokens, weights, config)
        assert len(logits) == config.vocab_size

    def test_deterministic(self) -> None:
        """Same input produces same output."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        tokens = [0, 1, 2]
        logits1 = transformer_forward_q(tokens, weights, config)
        logits2 = transformer_forward_q(tokens, weights, config)
        assert logits1 == logits2

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
        logits1 = transformer_forward_q([0, 1], weights, config)
        logits2 = transformer_forward_q([2, 3], weights, config)
        # They should differ (not guaranteed but very likely with these weights)
        assert logits1 != logits2

    def test_single_token(self) -> None:
        """Forward pass works with a single token."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        logits = transformer_forward_q([0], weights, config)
        assert len(logits) == config.vocab_size

    def test_logits_are_int32(self) -> None:
        """Logits are plain int values (not clamped to int8)."""
        config = _make_tiny_config()
        weights = _make_tiny_weights(config)
        logits = transformer_forward_q([0, 1], weights, config)
        # Logits can be larger than int8 range
        for v in logits:
            assert isinstance(v, int)
