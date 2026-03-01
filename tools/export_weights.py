"""Export trained MNIST MLP weights as C arrays and test data for firmware.

Trains a simple 784->128(ReLU)->10 MLP on MNIST, quantizes weights to int8,
and exports:
  - firmware/mnist/weights.h: C header with weight arrays
  - firmware/mnist/test_data.py: Python module with test images + reference predictions

Usage:
    uv run --extra torch python tools/export_weights.py
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
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("ERROR: torch and torchvision required.  Install with: uv sync --extra torch")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class MnistMLP(nn.Module):
    """Two-layer MLP for MNIST: 784 -> 128 (ReLU) -> 10."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(epochs: int = 5, lr: float = 1e-3) -> MnistMLP:
    """Train the MNIST MLP and return the trained model.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.

    Returns:
        Trained MnistMLP model in eval mode.
    """
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1] float
    ])

    train_set = torchvision.datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform,
    )
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    model = MnistMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        acc = correct / total * 100
        avg_loss = total_loss / total
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.1f}%")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_weights(model: MnistMLP) -> dict[str, Any]:
    """Quantize model weights to int8 with per-layer symmetric quantization.

    Quantization scheme:
      - Layer 1 input: raw pixel bytes [0, 255] as unsigned int32.
        Float relationship: pixel_uint8 = x_float * 255.
      - Weights: w_q = round(w_real * w_scale), w_scale = 127/max|w|.
      - After MACC: acc = sum(pixel_uint8_i * w_q_i).
        This is in scale: 255 * w_scale * real_dot_product.
      - Bias: b_q = round(b_float * 255 * w_scale), in accumulator scale.
      - Rescaling: firmware does arithmetic right shift (SRA) by shift1 bits,
        then CLAMP to int8, then RELU.
        shift1 = round(log2(acc_scale / hidden_scale)) where we target
        hidden values in [-120, 120] range.
      - Layer 2: MACC(hidden_q, w2_q) + b2_q, then argmax on raw int32.
        No rescaling needed for argmax (relative ordering preserved).

    Returns dict with:
        w1: int8 ndarray (128, 784), b1: int32 ndarray (128,), shift1: int,
        w2: int8 ndarray (10, 128),  b2: int32 ndarray (10,),
        and various scale factor metadata.
    """
    with torch.no_grad():
        w1_float = model.fc1.weight.data.clone()  # (128, 784)
        b1_float = model.fc1.bias.data.clone()    # (128,)
        w2_float = model.fc2.weight.data.clone()  # (10, 128)
        b2_float = model.fc2.bias.data.clone()    # (10,)

    # --- Calibration: find activation ranges ---
    data_dir = Path(__file__).parent.parent / "data"
    transform = transforms.Compose([transforms.ToTensor()])
    calib_set = torchvision.datasets.MNIST(
        root=str(data_dir), train=True, download=False, transform=transform,
    )
    calib_loader = DataLoader(calib_set, batch_size=5000, shuffle=False)

    with torch.no_grad():
        images, _ = next(iter(calib_loader))
        x = images.view(-1, 784)
        hidden_real = F.relu(model.fc1(x))  # (5000, 128)
        hidden_max = hidden_real.max().item()

    # --- Layer 1 quantization ---
    w1_max = w1_float.abs().max().item()
    w1_scale = 127.0 / w1_max
    w1_q = torch.clamp(torch.round(w1_float * w1_scale), -128, 127).to(torch.int8)

    # Bias in accumulator scale: acc_scale = 255 * w1_scale
    acc1_scale = 255.0 * w1_scale
    b1_q = torch.round(b1_float * acc1_scale).to(torch.int32)

    # Target: hidden_q in [0, ~120] after ReLU.
    # hidden_q = real_value * hidden_scale, where hidden_scale = 120 / hidden_max.
    # acc = real_value * acc1_scale.
    # hidden_q = acc * hidden_scale / acc1_scale = acc >> shift1 (approximately).
    # shift1 = log2(acc1_scale / hidden_scale).
    hidden_target = 120.0
    hidden_scale = hidden_target / hidden_max if hidden_max > 0 else 1.0
    shift1 = round(math.log2(acc1_scale / hidden_scale))
    shift1 = max(0, min(shift1, 31))

    # Verify: after shift1, effective hidden_scale_actual = acc1_scale / 2^shift1
    hidden_scale_actual = acc1_scale / (1 << shift1)

    print(f"  Calibration: hidden_max={hidden_max:.2f}")
    print(f"  w1_scale={w1_scale:.4f}, acc1_scale={acc1_scale:.2f}")
    print(f"  hidden_scale_target={hidden_scale:.4f}, shift1={shift1}")
    print(f"  hidden_scale_actual={hidden_scale_actual:.4f} (max hidden_q ~ {hidden_max * hidden_scale_actual:.1f})")

    # --- Layer 2 quantization ---
    w2_max = w2_float.abs().max().item()
    w2_scale = 127.0 / w2_max
    w2_q = torch.clamp(torch.round(w2_float * w2_scale), -128, 127).to(torch.int8)

    # Layer 2 bias: input to layer 2 has scale hidden_scale_actual,
    # weights have scale w2_scale, so acc2_scale = hidden_scale_actual * w2_scale.
    acc2_scale = hidden_scale_actual * w2_scale
    b2_q = torch.round(b2_float * acc2_scale).to(torch.int32)

    print(f"  w2_scale={w2_scale:.4f}, acc2_scale={acc2_scale:.4f}")

    # --- VMAC bias adjustment for Layer 1 ---
    # Firmware converts uint8 pixels to signed int8: x_signed = x_uint8 - 128.
    # To compensate: bias_adjusted[i] = bias[i] + 128 * sum(weights_row[i])
    w1_row_sums = w1_q.to(torch.int64).sum(dim=1)  # (128,)
    b1_adjusted = b1_q.to(torch.int64) + 128 * w1_row_sums
    b1_adjusted = b1_adjusted.to(torch.int32)
    print(f"  VMAC bias adjustment: max delta = {(128 * w1_row_sums).abs().max().item()}")

    return {
        "w1": w1_q.numpy(), "b1": b1_adjusted.numpy(), "shift1": shift1,
        "w2": w2_q.numpy(), "b2": b2_q.numpy(),
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "hidden_scale_actual": hidden_scale_actual,
        "acc1_scale": acc1_scale,
        "acc2_scale": acc2_scale,
    }


def quantized_inference_python(
    image_uint8: np.ndarray,
    w1: np.ndarray, b1: np.ndarray, shift1: int,
    w2: np.ndarray, b2: np.ndarray,
) -> int:
    """Run quantized inference in pure Python/numpy matching firmware behavior.

    This exactly mirrors what the firmware does:
    Layer 1: convert uint8 to int8 (subtract 128), VMAC(signed_pixel, weight)
             -> RSTACC -> add adjusted bias -> SRA shift1 -> CLAMP -> RELU
    Layer 2: VMAC(hidden, weight) -> RSTACC -> add bias -> argmax (raw int32)

    Biases for layer 1 are pre-adjusted: b1[i] += 128 * sum(w1[i])

    Args:
        image_uint8: Input image as uint8 array of 784 elements (pixel values 0-255).
        w1, b1, shift1: Layer 1 weights (128,784), adjusted biases (128,), right shift.
        w2, b2: Layer 2 weights (10,128), biases (10,).

    Returns:
        Predicted digit 0-9.
    """
    # Convert uint8 pixels to signed int8 (subtract 128)
    pixels_signed = image_uint8.astype(np.int64) - 128

    # Layer 1: hidden = relu(clamp(sra(W1 @ pixels_signed + b1_adjusted, shift1)))
    hidden = np.zeros(128, dtype=np.int64)
    for i in range(128):
        acc = int(np.dot(w1[i].astype(np.int64), pixels_signed)) + int(b1[i])
        acc = acc >> shift1  # Arithmetic right shift
        acc = max(-128, min(127, acc))  # CLAMP
        acc = max(0, acc)  # RELU
        hidden[i] = acc

    # Layer 2: output = W2 @ hidden + b2 (raw int32 for argmax)
    output = np.zeros(10, dtype=np.int64)
    for i in range(10):
        acc = int(np.dot(w2[i].astype(np.int64), hidden)) + int(b2[i])
        output[i] = acc

    return int(np.argmax(output))


# ---------------------------------------------------------------------------
# C header export
# ---------------------------------------------------------------------------

def _format_int8_array(arr: np.ndarray, name: str, dims: tuple[int, ...]) -> str:
    """Format an int8 numpy array as a C array initializer.

    Args:
        arr: numpy int8 array.
        name: C variable name.
        dims: Dimensions for the C array declaration.

    Returns:
        C source string for the array definition.
    """
    flat = arr.flatten().tolist()
    dim_str = "".join(f"[{d}]" for d in dims)

    lines = [f"static const int8_t {name}{dim_str} = {{"]
    # Write 20 values per line
    for i in range(0, len(flat), 20):
        chunk = flat[i:i + 20]
        line = "    " + ", ".join(str(v) for v in chunk) + ","
        lines.append(line)
    lines.append("};")
    return "\n".join(lines)


def _format_int32_array(arr: np.ndarray, name: str, dim: int) -> str:
    """Format an int32 numpy array as a C array initializer.

    Args:
        arr: numpy int32 array.
        name: C variable name.
        dim: Length of the array.

    Returns:
        C source string for the array definition.
    """
    flat = arr.flatten().tolist()
    lines = [f"static const int32_t {name}[{dim}] = {{"]
    for i in range(0, len(flat), 10):
        chunk = flat[i:i + 10]
        line = "    " + ", ".join(str(v) for v in chunk) + ","
        lines.append(line)
    lines.append("};")
    return "\n".join(lines)


def export_c_header(weights: dict[str, Any], path: str) -> None:
    """Write quantized weights as a C header file.

    Args:
        weights: Dict from quantize_weights().
        path: Output file path.
    """
    w1 = weights["w1"]  # (128, 784)
    b1 = weights["b1"]  # (128,)
    w2 = weights["w2"]  # (10, 128)
    b2 = weights["b2"]  # (10,)
    shift1 = weights["shift1"]

    parts = [
        "#ifndef WEIGHTS_H",
        "#define WEIGHTS_H",
        "#include <stdint.h>",
        "",
        f"/* Layer 1: 784 -> 128, right-shift = {shift1} */",
        _format_int8_array(w1, "W1", (128, 784)),
        "",
        _format_int32_array(b1, "B1", 128),
        "",
        f"static const int32_t SHIFT1 = {shift1};",
        "",
        "/* Layer 2: 128 -> 10 (argmax on raw accumulator) */",
        _format_int8_array(w2, "W2", (10, 128)),
        "",
        _format_int32_array(b2, "B2", 10),
        "",
        "#endif /* WEIGHTS_H */",
        "",
    ]

    Path(path).write_text("\n".join(parts))
    print(f"  Wrote {path} ({Path(path).stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Test data export
# ---------------------------------------------------------------------------

def export_test_data(
    model: MnistMLP,
    weights: dict[str, Any],
    path: str,
    num_images: int = 100,
) -> None:
    """Export test images and reference predictions as a Python module.

    Saves a .py file that can be imported to get:
        IMAGES: list of 784-element lists (uint8 pixel values 0-255)
        LABELS: list of true labels
        PREDICTIONS: list of PyTorch model predictions (argmax of float output)
        QUANT_PREDICTIONS: list of quantized inference predictions
        SHIFT1: layer 1 right-shift amount

    Args:
        model: Trained PyTorch model.
        weights: Quantized weights dict.
        path: Output file path.
        num_images: Number of test images to export.
    """
    data_dir = Path(__file__).parent.parent / "data"
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform,
    )
    test_loader = DataLoader(test_set, batch_size=num_images, shuffle=False)

    images_tensor, labels_tensor = next(iter(test_loader))

    # Get PyTorch float predictions
    with torch.no_grad():
        float_preds = model(images_tensor).argmax(dim=1).tolist()

    # Convert images to uint8 [0, 255] -- this is what the firmware receives
    images_uint8 = (images_tensor.view(-1, 784) * 255).round().clamp(0, 255).to(torch.uint8)

    # Run quantized inference in Python
    w1 = weights["w1"]
    b1 = weights["b1"]
    shift1 = weights["shift1"]
    w2 = weights["w2"]
    b2 = weights["b2"]

    quant_preds = []
    for idx in range(num_images):
        img = images_uint8[idx].numpy()
        pred = quantized_inference_python(img, w1, b1, shift1, w2, b2)
        quant_preds.append(pred)

    # Write Python module
    lines = [
        '"""Auto-generated MNIST test data. Do not edit manually."""',
        "",
        "# Layer 1 right-shift for rescaling accumulator",
        f"SHIFT1 = {shift1}",
        "",
        "# Test images as uint8 pixel values [0, 255], shape: (num_images, 784)",
        f"IMAGES = {images_uint8.tolist()}",
        "",
        "# True labels",
        f"LABELS = {labels_tensor.tolist()}",
        "",
        "# PyTorch float model predictions (argmax)",
        f"PREDICTIONS = {float_preds}",
        "",
        "# Quantized inference predictions (Python reference, matches firmware)",
        f"QUANT_PREDICTIONS = {quant_preds}",
        "",
    ]

    Path(path).write_text("\n".join(lines))
    print(f"  Wrote {path} ({Path(path).stat().st_size:,} bytes)")

    # Report accuracy
    labels = labels_tensor.tolist()
    float_correct = sum(1 for p, l in zip(float_preds, labels) if p == l)
    quant_correct = sum(1 for p, l in zip(quant_preds, labels) if p == l)
    match_count = sum(1 for p, q in zip(float_preds, quant_preds) if p == q)
    print(f"  Float model accuracy: {float_correct}/{num_images} ({float_correct / num_images * 100:.1f}%)")
    print(f"  Quantized accuracy:   {quant_correct}/{num_images} ({quant_correct / num_images * 100:.1f}%)")
    print(f"  Float/quant agreement: {match_count}/{num_images} ({match_count / num_images * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Train, quantize, and export MNIST MLP weights."""
    firmware_dir = Path(__file__).parent.parent / "firmware" / "mnist"
    firmware_dir.mkdir(parents=True, exist_ok=True)

    print("Training MNIST MLP (784 -> 128 -> 10)...")
    model = train_model(epochs=5)

    print("\nQuantizing weights to int8...")
    weights = quantize_weights(model)

    print("\nExporting C header...")
    export_c_header(weights, str(firmware_dir / "weights.h"))

    print("\nExporting test data...")
    export_test_data(model, weights, str(firmware_dir / "test_data.py"), num_images=100)

    print("\nDone! Next steps:")
    print("  cd firmware/mnist && make")
    print("  uv run python -m riscv_npu run firmware/mnist/mnist.elf")


if __name__ == "__main__":
    main()
