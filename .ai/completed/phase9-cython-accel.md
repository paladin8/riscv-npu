# Phase 9: Cython NPU Acceleration

## Goal

Accelerate NPU vector inner loops with a Cython C extension, operating directly on the RAM bytearray via typed memoryviews. Pure-Python fallback preserved for portability.

## Deliverables

- `_accel.pyx`: Cython module with accelerated vector kernels (int8 and f32)
- Bus `get_device_data()` method for raw buffer access
- Graceful try-import in NPU instruction modules with fast-path dispatch
- Build integration (Makefile target, dev dependency)
- Benchmark validation via `scripts/bench.py`

## Design Decisions

1. **Scope: vector inner loops only.** Scalar NPU ops (RELU, CLAMP, GELU, QMUL, MACC, RSTACC) are already single-element — Python function call overhead is negligible. Only the N-element loops that dominate profiling get accelerated.

2. **Direct buffer access via typed memoryviews.** The Cython functions take `const unsigned char[:]` (the RAM bytearray) plus integer offsets, bypassing the bus/device lookup entirely. This eliminates ~7 Python function calls per vector element.

3. **`MemoryBus.get_device_data(addr)`** returns `(bytearray, base_addr)` for the device owning `addr`. NPU code computes `offset = addr - base`. Both vector addresses are in the same RAM device, so one call suffices.

4. **Try-import with `_USE_ACCEL` flag.** Each NPU instruction module imports from `._accel` at the top. If the .so isn't compiled, everything falls back to the existing pure-Python loops unchanged.

5. **Build via `cythonize -i`.** No build-system changes — hatchling stays, the .so compiles in-place next to the .pyx. A Makefile provides `make accel` and `make clean-accel` targets for convenience.

6. **Writable memoryview for in-place ops.** Functions that write results back to memory (fvexp, fvmul, vmul) take `unsigned char[:]` (non-const) for the output region.

## Deliverables List (ordered, dependency-aware)

1. **D1: `MemoryBus.get_device_data()`** — raw buffer access method
2. **D2: `_accel.pyx`** — Cython module with all vector kernels
3. **D3: Integrate into `instructions.py`** — try-import + fast path for int NPU vector ops
4. **D4: Integrate into `fp_instructions.py`** — try-import + fast path for FP NPU vector ops
5. **D5: Build integration** — Makefile, dev dependency, .gitignore for build artifacts
6. **D6: Benchmark validation** — verify speedup with `scripts/bench.py`

## Implementation Details

### D1: `MemoryBus.get_device_data()`

- **File**: `src/riscv_npu/memory/bus.py`
- New method:
  ```python
  def get_device_data(self, addr: int) -> tuple[bytearray, int]:
      """Get (raw_buffer, device_base_addr) for the device at addr.

      Enables direct buffer access for bulk operations like NPU vector
      instructions, bypassing per-element bus dispatch.

      Args:
          addr: Any address within the target device's range.

      Returns:
          Tuple of (device's backing bytearray, device base address).

      Raises:
          MemoryError: If no device covers the address.
          AttributeError: If the device has no _data buffer.
      """
      mapping = self._find_device(addr, 1)
      return mapping.device._data, mapping.base
  ```

### D2: `_accel.pyx`

- **File**: `src/riscv_npu/npu/_accel.pyx`

**Integer NPU kernels:**

| Function                                              | Replaces loop in          | Semantics                                             |
|-------------------------------------------------------|---------------------------|-------------------------------------------------------|
| `vmac_int8(data, off_a, off_b, n) -> long long`      | `_exec_vmac`              | Sum of int8[a_i] * int8[b_i], returned as 64-bit int  |
| `vmul_int8(data, src, dst, n, scale) -> None`         | `_exec_vmul`              | dst[i] = clamp((src_int8[i] * scale) >> 16, -128,127) |
| `vreduce_int32(data, src, n) -> long long`            | `_exec_vreduce`           | Sum of n signed int32 values                          |
| `vmax_int32(data, src, n) -> int`                     | `_exec_vmax`              | Max of n signed int32 values                          |
| `vexp_int32(data, src, dst, n) -> None`               | `_exec_vexp`              | dst[i] = exp_q16_16(src[i]) for n int32 elements      |

**FP NPU kernels:**

| Function                                              | Replaces loop in          | Semantics                                             |
|-------------------------------------------------------|---------------------------|-------------------------------------------------------|
| `fvmac_f32(data, off_a, off_b, n) -> double`         | `_exec_fvmac`             | Dot product of two f32 arrays, double accumulation    |
| `fvmul_f32(data, src, dst, n, scale_bits) -> None`   | `_exec_fvmul`             | dst[i] = src[i] * scale for n f32 elements            |
| `fvexp_f32(data, src, dst, n) -> None`                | `_exec_fvexp`             | dst[i] = exp(src[i]) for n f32 elements               |
| `fvreduce_f32(data, src, n) -> double`                | `_exec_fvreduce`          | Sum of n f32 values, double accumulation              |
| `fvmax_f32(data, src, n) -> float`                    | `_exec_fvmax`             | Max of n f32 values                                   |

All functions use `cdef` typed locals, `@cython.boundscheck(False)`, `@cython.wraparound(False)` decorators for maximum speed. Float functions use `libc.math.exp`/`libc.math.sqrtf` for C-level math.

### D3: Integrate into `instructions.py`

- **File**: `src/riscv_npu/npu/instructions.py`
- Add try-import block at top of file
- Modify 5 functions to use fast path when `_USE_ACCEL` is True:
  - `_exec_vmac`: call `vmac_int8()`, add result to accumulator
  - `_exec_vmul`: call `vmul_int8()` (writes directly to buffer)
  - `_exec_vreduce`: call `vreduce_int32()`
  - `_exec_vmax`: call `vmax_int32()`
  - `_exec_vexp`: call `vexp_int32()`
- Each function calls `mem.get_device_data(addr)` once to get `(data, base)`, then passes `data` and `addr - base` offsets to the Cython kernel
- Pure-Python fallback loops remain in `else` branches, unchanged

### D4: Integrate into `fp_instructions.py`

- **File**: `src/riscv_npu/npu/fp_instructions.py`
- Same pattern as D3
- Modify 5 functions:
  - `_exec_fvmac`: call `fvmac_f32()`
  - `_exec_fvmul`: call `fvmul_f32()`
  - `_exec_fvexp`: call `fvexp_f32()`
  - `_exec_fvreduce`: call `fvreduce_f32()`
  - `_exec_fvmax`: call `fvmax_f32()`

### D5: Build integration

- **File**: `Makefile` (new, project root)
  ```makefile
  accel:
  	cythonize -i src/riscv_npu/npu/_accel.pyx

  clean-accel:
  	rm -f src/riscv_npu/npu/_accel.c src/riscv_npu/npu/_accel*.so

  test:
  	uv run pytest

  bench:
  	uv run python scripts/bench.py
  ```
- Add `cython` as dev dependency: `uv add --dev cython`
- Add to `.gitignore`: `*.so`, `*.c` (generated), `*.pyc`, `__pycache__/`

### D6: Benchmark validation

- Run `scripts/bench.py` before and after compilation
- Compare `cpu.step (tight loop)` and firmware workload throughput
- Run `--cprofile mnist` to verify vector functions no longer dominate
- Document results in commit message

## Test Coverage Requirements

### D1: get_device_data tests
- **File**: `tests/memory/test_bus.py` (add to existing)
- `test_get_device_data_returns_buffer_and_base` — returns RAM bytearray and base address
- `test_get_device_data_unmapped_address_raises` — MemoryError for unmapped addr

### D2: _accel kernel tests
- **File**: `tests/npu/test_accel.py` (new)
- Test each Cython kernel against the pure-Python equivalent on matching inputs
- Skip all tests if `_accel` not compiled (`pytest.importorskip("riscv_npu.npu._accel")`)
- `test_vmac_int8_basic` — small known dot product
- `test_vmac_int8_negative` — signed int8 values
- `test_vmac_int8_empty` — n=0 returns 0
- `test_fvmac_f32_basic` — f32 dot product
- `test_fvexp_f32_basic` — exp of known values
- `test_fvmul_f32_basic` — scale by known factor
- `test_fvreduce_f32_basic` — sum of known array
- `test_fvmax_f32_basic` — max of known array
- `test_vmul_int8_basic` — int8 scale + clamp
- `test_vreduce_int32_basic` — int32 sum
- `test_vmax_int32_basic` — int32 max
- `test_vexp_int32_basic` — Q16.16 exp of known values

### D3/D4: Integration tests
- Existing NPU tests (`tests/npu/test_instructions.py`, `tests/npu/test_fp_instructions.py`) serve as integration tests — they must pass with and without the .so compiled

## Acceptance Criteria

1. `uv run pytest` — all existing 939 tests pass without .so compiled (pure Python fallback)
2. `make accel` — compiles _accel.so without errors
3. `uv run pytest` — all tests pass with .so compiled (accelerated path exercised)
4. `uv run python scripts/bench.py --cprofile mnist` — NPU vector functions no longer top the profile
5. No changes to existing test files
6. `_USE_ACCEL` flag allows runtime toggle
