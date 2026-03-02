accel:
	uv run cythonize -i src/riscv_npu/npu/_accel.pyx

clean-accel:
	rm -f src/riscv_npu/npu/_accel.c src/riscv_npu/npu/_accel*.so

test:
	uv run pytest

bench:
	uv run python scripts/bench.py
