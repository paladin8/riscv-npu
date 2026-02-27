# Phase 6: MNIST Inference

## Goal
Quantized MLP runs on emulator, correctly classifies digits.

## What to build
- tools/export_weights.py: train MLP (784→128 ReLU→10) on MNIST, quantize int8, export as C arrays
- firmware/mnist/weights.h: generated weight arrays
- firmware/mnist/nn_runtime.c: linear layer + activation using NPU_MACC, NPU_RSTACC, NPU_RELU, NPU_CLAMP
- firmware/mnist/main.c: load test image, run inference, print prediction
- tests/integration/test_mnist.py: run multiple images, compare to PyTorch reference

## Network
- Input: 784 int8 (28×28 flattened)
- Hidden: 128, ReLU
- Output: 10, argmax
- ~100KB weights

## Acceptance
uv run python tools/export_weights.py — generates weights.h
cd firmware/mnist && make
uv run python -m riscv_npu run firmware/mnist/mnist.elf — prints correct digit
uv run pytest tests/integration/test_mnist.py — ≥95% accuracy
