# Phase 7: Floating-Point Transformer

## Goal
Tiny character-level transformer runs on emulator using FP NPU instructions. All weights and activations are float32 — no quantization. Demonstrates the FP NPU instruction set (opcode 0x2B) as a practical neural network accelerator.

## Design Decisions

1. **Task**: Character-level language model (predict next character). Byte-level vocab (256 entries), no tokenizer needed.
2. **Precision**: All float32. Weights stored as float32 C arrays. Activations are float32. No int8, no Q16.16, no shifts or scales.
3. **Normalization**: RMSNorm (not LayerNorm). Uses FVMAC for sum-of-squares, FVRSQRT for reciprocal sqrt.
4. **NPU instructions used**: All 10 FP NPU instructions (opcode 0x2B) — already implemented.
   - FMACC, FVMAC, FRSTACC — dot products and linear layers
   - FRELU, FGELU — activations
   - FVEXP, FVRSQRT, FVMUL, FVREDUCE, FVMAX — softmax and RMSNorm
5. **Model**: Embedding dim 64, 4 heads (head_dim=16), 2 layers, byte-level vocab (256), context 32 tokens, ~135K params, ~527KB float32.
6. **Softmax strategy**: FVMAX (find max), subtract max, FVEXP, FVREDUCE (sum), set facc = 1/sum, FVMUL (normalize). All native float32.
7. **Memory budget**: ~527KB weights + ~32KB KV cache + scratch < 1MB. Fits in 4MB RAM.

## FP NPU Instruction Summary

All instructions use opcode 0x2B (custom-1), R-type encoding. Already implemented in `src/riscv_npu/npu/fp_instructions.py` with C intrinsics in `firmware/common/npu_fp.h`.

| funct3 | funct7 | Name       | Semantics                                                   |
|--------|--------|------------|-------------------------------------------------------------|
| 0      | 0      | FMACC      | facc += f[rs1] * f[rs2]                                     |
| 0      | 1      | FVMAC      | facc += dot(mem_f32[rs1..+n], mem_f32[rs2..+n])             |
| 0      | 2      | FVEXP      | dst_f32[i] = exp(src_f32[i]) for i in 0..n-1                |
| 0      | 3      | FVRSQRT    | f[rd] = 1/sqrt(mem_f32[rs1])                                |
| 0      | 4      | FVMUL      | dst_f32[i] = src_f32[i] * (float32)facc for i in 0..n-1    |
| 0      | 5      | FVREDUCE   | f[rd] = sum(mem_f32[rs1..+n])                               |
| 0      | 6      | FVMAX      | f[rd] = max(mem_f32[rs1..+n])                               |
| 1      | —      | FRELU      | f[rd] = max(f[rs1], 0.0)                                    |
| 4      | —      | FGELU      | f[rd] = gelu(f[rs1])                                        |
| 5      | —      | FRSTACC    | f[rd] = (float32)facc; facc = 0.0                           |

## Deliverables List

1. **Python float transformer reference** (`src/riscv_npu/tools/transformer.py`)
   - Rewrite: all-float, no quantization, no Q16.16, no clamp/shift
   - Unit tests
2. **Weight exporter** (`src/riscv_npu/tools/export_transformer_weights.py`)
   - Rewrite: train model, export raw float32 weights (no quantization step)
   - Export float32 C arrays and Python test data
3. **C firmware** (`firmware/transformer/main.c`, `firmware/transformer/weights.h`)
   - Rewrite: use `npu_fp.h` FP NPU intrinsics, float32 throughout
   - Makefile for cross-compilation
4. **Integration tests** (`tests/integration/test_transformer.py`)
   - Update: compare firmware output to Python float reference
5. **Documentation updates**
   - `docs/npu-design.md`: update transformer section for FP
   - `docs/isa-reference.md`: verify FP NPU entries are current

## Implementation Details

### Deliverable 1: Python Float Transformer Reference

**File:** `src/riscv_npu/tools/transformer.py`

Rewrite from int8/Q16.16 to pure float. This is the reference implementation that integration tests compare against.

**Data structures:**

```python
@dataclass
class TransformerConfig:
    vocab_size: int = 256
    embed_dim: int = 64
    n_heads: int = 4
    head_dim: int = 16
    n_layers: int = 2
    context_len: int = 32
    ff_dim: int = 256

@dataclass
class LayerWeights:
    ln1_gamma: list[float]           # (embed_dim,)
    wq: list[list[float]]           # (embed_dim, embed_dim)
    bq: list[float]                 # (embed_dim,)
    wk: list[list[float]]           # (embed_dim, embed_dim)
    bk: list[float]                 # (embed_dim,)
    wv: list[list[float]]           # (embed_dim, embed_dim)
    bv: list[float]                 # (embed_dim,)
    wo: list[list[float]]           # (embed_dim, embed_dim)
    bo: list[float]                 # (embed_dim,)
    ln2_gamma: list[float]           # (embed_dim,)
    w1: list[list[float]]           # (ff_dim, embed_dim)
    b1: list[float]                 # (ff_dim,)
    w2: list[list[float]]           # (embed_dim, ff_dim)
    b2: list[float]                 # (embed_dim,)

@dataclass
class TransformerWeights:
    token_embed: list[list[float]]   # (vocab_size, embed_dim)
    pos_embed: list[list[float]]     # (context_len, embed_dim)
    layers: list[LayerWeights]
    ln_final_gamma: list[float]      # (embed_dim,)
    output_proj: list[list[float]]   # (vocab_size, embed_dim)
    output_bias: list[float]         # (vocab_size,)
```

**Key functions:**

```python
def dot_f32(a: list[float], b: list[float]) -> float:
    """Dot product of two float vectors."""

def rmsnorm(x: list[float], gamma: list[float], dim: int) -> list[float]:
    """RMSNorm: x * gamma * rsqrt(mean(x^2) + eps)."""

def softmax(scores: list[float], n: int) -> list[float]:
    """Numerically stable softmax: subtract max, exp, normalize."""

def linear(x: list[float], weight: list[list[float]], bias: list[float],
           in_dim: int, out_dim: int) -> list[float]:
    """y[i] = dot(weight[i], x) + bias[i]."""

def gelu(x: float) -> float:
    """GELU activation."""

def transformer_forward(tokens: list[int], weights: TransformerWeights,
                        config: TransformerConfig) -> list[float]:
    """Full forward pass, returns logits for last token."""

def predict_next_token(tokens: list[int], weights: TransformerWeights,
                       config: TransformerConfig) -> int:
    """Argmax of logits."""
```

### Deliverable 2: Weight Exporter

**File:** `src/riscv_npu/tools/export_transformer_weights.py`

Rewrite: remove all quantization (no QAT, no int8, no calibration, no shifts). Train a standard float model and export weights directly as float32.

**Training:**
- Same model architecture (TinyTransformer with RMSNorm, GELU FFN)
- Remove QATLinear — use standard nn.Linear
- Remove all fake_quantize calls
- Remove enable_qat/disable_qat
- Standard Adam training, ~10 epochs on repeated text corpus

**Export:**
- `weights.h`: float32 C arrays instead of int8
- `test_data.py`: test sequences + float reference predictions

**Key changes from current:**
- `_quantize_tensor()` → removed
- `_calibrate_activations()` → removed
- `quantize_model()` → `extract_weights()` (just copies float tensors)
- `_format_int8_array()` → `_format_float_array()` (float C arrays)
- `_format_int32_array()` → updated for float biases
- `quantized_inference_python()` → `float_inference_python()` (uses float reference)

**C header format:**

```c
#define VOCAB_SIZE 256
#define EMBED_DIM 64
#define N_HEADS 4
#define HEAD_DIM 16
#define N_LAYERS 2
#define CONTEXT_LEN 32
#define FF_DIM 256

static const float TOKEN_EMBED[256][64] = { ... };
static const float POS_EMBED[32][64] = { ... };
static const float L0_LN1_GAMMA[64] = { ... };
static const float L0_WQ[64][64] = { ... };
static const float L0_BQ[64] = { ... };
/* ... etc ... */
```

### Deliverable 3: C Firmware

**File:** `firmware/transformer/main.c`

Rewrite to use FP NPU intrinsics (`npu_fp.h`). All buffers are `float` instead of `int8_t`/`int32_t`.

**Linear layer:**
```c
static void linear(
    const float *input, int in_dim,
    const float *weight, const float *bias,
    float *output, int out_dim
) {
    for (int i = 0; i < out_dim; i++) {
        NPU_FVMAC(&weight[i * in_dim], input, in_dim);
        output[i] = NPU_FRSTACC() + bias[i];
    }
}
```

**RMSNorm:**
```c
static void rmsnorm(
    const float *input, const float *gamma,
    float *output, int dim
) {
    /* sum of squares via FVMAC(input, input) */
    NPU_FVMAC(input, input, dim);
    float sum_sq = NPU_FRSTACC();
    float mean_sq = sum_sq / (float)dim + 1e-5f;

    /* 1/sqrt(mean_sq) */
    float scale = NPU_FVRSQRT(&mean_sq);

    /* output[i] = input[i] * gamma[i] * scale */
    for (int i = 0; i < dim; i++) {
        output[i] = input[i] * gamma[i] * scale;
    }
}
```

**Softmax:**
```c
static void softmax(float *scores, int n, float *probs) {
    float max_score = NPU_FVMAX(scores, n);

    /* subtract max */
    for (int i = 0; i < n; i++)
        scores[i] -= max_score;

    /* exp */
    NPU_FVEXP(scores, probs, n);

    /* sum */
    float sum_exp = NPU_FVREDUCE(probs, n);

    /* normalize: set facc = 1/sum, then FVMUL */
    NPU_FRSTACC();                       /* clear facc */
    float inv = 1.0f / sum_exp;
    float one = 1.0f;
    NPU_FMACC(inv, one);                /* facc = 1/sum */
    NPU_FVMUL(probs, probs, n);         /* probs[i] *= facc */
}
```

**GELU activation:**
```c
static void gelu_activation(float *buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = NPU_FGELU(buf[i]);
}
```

**Attention:**
```c
static void attention(
    const float *x_in, int layer, int pos,
    const float *wq, const float *bq,
    const float *wk, const float *bk,
    const float *wv, const float *bv,
    const float *wo, const float *bo,
    float *output
) {
    /* Project Q, K, V */
    linear(x_in, EMBED_DIM, wq, bq, q_proj, EMBED_DIM);
    linear(x_in, EMBED_DIM, wk, bk, k_proj, EMBED_DIM);
    linear(x_in, EMBED_DIM, wv, bv, v_proj, EMBED_DIM);

    /* Store K, V into cache */
    for (int d = 0; d < EMBED_DIM; d++) {
        k_cache[layer][pos][d] = k_proj[d];
        v_cache[layer][pos][d] = v_proj[d];
    }

    int n_tokens = pos + 1;
    float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);

    /* Per-head attention */
    for (int h = 0; h < N_HEADS; h++) {
        int h_start = h * HEAD_DIM;

        /* Compute scores: Q[h] . K_cache[h] * scale */
        for (int t = 0; t < n_tokens; t++) {
            NPU_FVMAC(&q_proj[h_start], &k_cache[layer][t][h_start], HEAD_DIM);
            scores_buf[t] = NPU_FRSTACC() * attn_scale;
        }

        /* Softmax */
        softmax(scores_buf, n_tokens, probs_buf);

        /* Weighted sum of V */
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < n_tokens; t++) {
                acc += probs_buf[t] * v_cache[layer][t][h_start + d];
            }
            attn_out[h_start + d] = acc;
        }
    }

    /* Output projection */
    linear(attn_out, EMBED_DIM, wo, bo, output, EMBED_DIM);
}
```

**Memory layout:**
- All weight arrays in .rodata as `const float` (compiled into ELF)
- Scratch buffers in .bss as `float`:
  - `x[EMBED_DIM]`, `normed[EMBED_DIM]`, `q_proj[EMBED_DIM]`, etc.
  - `k_cache[N_LAYERS][CONTEXT_LEN][EMBED_DIM]` (float)
  - `v_cache[N_LAYERS][CONTEXT_LEN][EMBED_DIM]` (float)
  - `scores_buf[CONTEXT_LEN]`, `probs_buf[CONTEXT_LEN]`

**Key differences from old firmware:**
- `#include "../common/npu_fp.h"` instead of `"../common/npu.h"`
- No `clamp_i8()`, no shifts, no Q16.16 conversions
- No `NPU_VMAC` / `NPU_RSTACC` / `NPU_CLAMP` / `NPU_GELU` (integer NPU)
- Everything is `float` instead of `int8_t` / `int32_t`
- Attention scale is just `1.0f / sqrtf(HEAD_DIM)` (not Q16.16)
- Softmax outputs float probabilities [0, 1] (not uint8 [0, 255])
- Residual connections are simple float addition (no clamping)

### Deliverable 4: Integration Tests

**File:** `tests/integration/test_transformer.py`

Update to work with float reference. Key changes:
- `_load_test_data()` loads `FLOAT_PREDICTIONS` (no `QUANT_PREDICTIONS`)
- Compare firmware output to float Python reference predictions
- Same structure: load ELF, inject tokens, run CPU, parse output

### Deliverable 5: Documentation

Update `docs/npu-design.md` and `docs/isa-reference.md` to reflect FP transformer usage.

## Test Coverage Requirements

### Deliverable 1 tests (in `tests/tools/test_transformer.py`):

**Float ops:**
- `test_dot_f32_basic`: dot product of known vectors
- `test_dot_f32_zero`: zero vector gives zero

**RMSNorm:**
- `test_rmsnorm_ones`: all-ones input, gamma=1.0 → each element ~1.0
- `test_rmsnorm_scaling`: known input, verify normalization
- `test_rmsnorm_zero_input`: all zeros → zeros out

**Softmax:**
- `test_softmax_uniform`: equal scores → equal probabilities
- `test_softmax_one_hot`: one large score dominates
- `test_softmax_sums_to_one`: probabilities sum to ~1.0

**Linear:**
- `test_linear_identity_weight`: identity-ish weight matrix
- `test_linear_with_bias`: verify bias is added

**Attention:**
- `test_attention_single_head`: small known Q, K, V
- `test_attention_deterministic`: same input → same output

**Forward pass:**
- `test_transformer_forward_deterministic`: same tokens + weights → same logits
- `test_predict_next_token`: returns valid token ID [0, 255]

### Deliverable 4 tests (in `tests/integration/test_transformer.py`):
- `test_single_sequence`: firmware prediction matches Python reference
- `test_multiple_sequences`: majority of predictions match

## Acceptance Criteria

1. `uv run pytest` passes with all existing + new tests
2. Python float transformer reference produces deterministic output for fixed weights
3. Weight exporter trains model and exports float32 C header + test data
4. Firmware compiles with `-march=rv32imf -mabi=ilp32f` and runs on emulator
5. Firmware predictions match Python reference on test sequences
6. No int8, Q16.16, shifts, scales, or clamp logic anywhere in the transformer code
