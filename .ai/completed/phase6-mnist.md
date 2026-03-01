# Phase 6: MNIST Inference

## Goal
Quantized MLP runs on emulator, correctly classifies digits.

## What to build
- tools/export_weights.py: train MLP (784->128 ReLU->10) on MNIST, quantize int8, export as C arrays
- firmware/mnist/weights.h: generated weight arrays
- firmware/mnist/nn_runtime.c: linear layer + activation using NPU_MACC, NPU_RSTACC, NPU_RELU, NPU_CLAMP
- firmware/mnist/main.c: load test image, run inference, print prediction
- tests/integration/test_mnist.py: run multiple images, compare to Python quantized reference

## Network
- Input: 784 uint8 (28x28 flattened, pixel values 0-255)
- Hidden: 128, ReLU
- Output: 10, argmax on raw int32 scores
- ~102KB weights

## Design Decisions

### Quantization scheme
- **Per-layer symmetric int8 quantization**: weights quantized to int8 [-128, 127] using w_scale = 127/max|w|.
- **Layer 1 input**: raw pixel bytes [0, 255] as unsigned int32 in MACC (no int8 conversion, avoids offset bugs).
- **Accumulator rescaling**: arithmetic right shift (SRA) instead of QMUL. QMUL's 8-bit shift is insufficient for the large accumulator scale (255 * w_scale ~ 50000). A calibrated shift amount (shift1 ~ 13) works precisely.
- **Biases stored as int32**: biases pre-scaled to accumulator scale (255 * w_scale for layer 1, hidden_scale * w2_scale for layer 2).
- **Layer 2 output**: raw int32 accumulator values used for argmax (no rescaling needed -- relative ordering preserved).

### Firmware memory layout
- Weights and biases compiled into ELF as const arrays in weights.h (~102KB).
- Test image (784 bytes) at known symbol `test_image` in .bss.
- Test harness writes image bytes into RAM at `test_image` address before running.
- Output: firmware prints predicted digit (0-9) as ASCII followed by newline.

### NPU usage in linear layer
- Layer 1 (linear_relu): RSTACC (clear) -> MACC(pixel_uint8, weight_int8) for all inputs -> RSTACC (read) -> add bias -> SRA shift1 -> CLAMP -> RELU -> store int8.
- Layer 2 (linear_raw): RSTACC -> MACC(hidden_int8, weight_int8) -> RSTACC -> add bias -> store raw int32.
- Argmax on int32 output array to find predicted digit.

### Test approach
- export_weights.py saves: weights.h (C header), test_data.py (Python module with test images and quantized reference predictions).
- Integration test loads test_data.py, writes each image into ELF memory, runs inference, compares to reference.
- Accuracy target: >=95% on 100 test images (achieves ~99% in practice).

## Deliverables List

| # | Deliverable                                   | Dependencies |
|---|-----------------------------------------------|--------------|
| 1 | tools/export_weights.py                       | none         |
| 2 | firmware/mnist/weights.h + test data          | D1           |
| 3 | firmware/mnist/nn_runtime.c + nn_runtime.h    | D2           |
| 4 | firmware/mnist/main.c + Makefile              | D3           |
| 5 | tests/integration/test_mnist.py               | D2, D4       |

## Implementation Details

### D1: tools/export_weights.py
**Files**: `tools/export_weights.py`

**Functions**:
- `train_model(epochs, lr) -> MnistMLP`: Train 784->128->10 MLP on MNIST. Adam, 5 epochs, CrossEntropyLoss. ~97.5% accuracy.
- `quantize_weights(model) -> dict`: Per-layer symmetric int8 quantization with calibrated shift. Returns w1, b1, shift1, w2, b2, and scale metadata.
- `quantized_inference_python(image_uint8, w1, b1, shift1, w2, b2) -> int`: Reference quantized inference matching firmware behavior exactly.
- `export_c_header(weights, path) -> None`: Write weights.h with C arrays.
- `export_test_data(model, weights, path, num_images) -> None`: Save test_data.py with 100 test images + reference predictions.

**Quantization math**:
- `w_scale = 127.0 / max(abs(weight_tensor))`
- `w_q = clamp(round(w_float * w_scale), -128, 127)`
- Layer 1 bias: `b1_q = round(b1_float * 255 * w1_scale)` (accumulator scale)
- Layer 1 shift: `shift1 = round(log2(255 * w1_scale / hidden_scale))` where hidden_scale targets max activation -> 120.
- Layer 2 bias: `b2_q = round(b2_float * hidden_scale_actual * w2_scale)` (layer 2 accumulator scale)

### D2: firmware/mnist/weights.h (generated)
Contains: W1[128][784] (int8), B1[128] (int32), SHIFT1 (int32), W2[10][128] (int8), B2[10] (int32).

### D3: firmware/mnist/nn_runtime.c + nn_runtime.h
**Functions**:
- `linear_relu(input, weights, bias, shift, out_dim, in_dim, output)`: Layer with MACC + SRA + CLAMP + RELU.
- `linear_raw(input, weights, bias, out_dim, in_dim, output)`: Layer with MACC, raw int32 output for argmax.
- `argmax(data, len) -> int`: Index of maximum int32 value.
- `inference(image) -> int`: Full forward pass using exported weights.

### D4: firmware/mnist/main.c + Makefile
- `test_image[784]` buffer in .bss for test harness to write to.
- `main()`: calls `inference(test_image)`, prints digit via write syscall.
- Makefile: links start.o + main.o + nn_runtime.o + syscalls.o -> mnist.elf.

### D5: tests/integration/test_mnist.py
- `test_mnist_single_image`: Run first test image, verify matches quantized reference.
- `test_mnist_accuracy`: Run 100 images, verify >=95% accuracy vs quantized reference.
- `test_mnist_all_digits_represented`: Verify test set covers all 10 digits.
- Uses skipif guards for missing toolchain/ELF/test_data.

## Test Coverage Requirements
- `test_mnist_single_image`: deterministic single-image correctness check.
- `test_mnist_accuracy`: statistical accuracy check (>=95%, typically ~99%).
- `test_mnist_all_digits_represented`: data coverage check.

## Acceptance Criteria
1. `uv run --extra torch python tools/export_weights.py` generates weights.h and test_data.py
2. `cd firmware/mnist && make` produces mnist.elf without errors
3. `uv run python -m riscv_npu run firmware/mnist/mnist.elf` prints a digit and exits cleanly
4. `uv run pytest tests/integration/test_mnist.py -v` passes with >=95% accuracy
5. `uv run pytest` shows all tests passing (555 + 3 new = 558)
