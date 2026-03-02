# Performance Profile

## Methodology

Profiled with Python 3.14 `cProfile` and `scripts/bench.py` on firmware workloads and synthetic micro-benchmarks. Run benchmarks with:

```
uv run python scripts/bench.py                 # all workloads
uv run python scripts/bench.py --cprofile mnist # cProfile a specific workload
uv run python scripts/bench.py --micro-only     # just isolated component benchmarks
```

## Current Throughput

With all optimizations (bus delegation, device cache, slots dataclass, Cython NPU):

| Workload      | Instructions/sec | Cycles | Instruction mix           |
|---------------|------------------|--------|---------------------------|
| cpu.step loop |          ~940K   | 200K   | ADDI/BNE tight loop       |
| fibonacci     |          ~710K   | 59     | ADDI 59%, ADD 17%, BNE 17%|
| sort          |          ~750K   | 336    | ADDI 37%, LW 17%, SW 12%  |
| fpu_test      |          ~790K   | 5,348  | ADDI 33%, LBU 20%, BNE 17%|
| mnist         |          ~900K   | 6,847  | ADDI 45%, BNE 14%, SB 13% |

## Where Time Is Spent (current, post-optimization)

The profile is now well-balanced across the CPU step pipeline. On the mnist workload (cProfile, with Cython acceleration compiled):

| Component          | Function                      | Notes                               |
|--------------------|-------------------------------|-------------------------------------|
| CPU step overhead  | `cpu.step()`                  | Per-cycle bookkeeping               |
| Instruction decode | `decode.decode()`             | Bit extraction + Instruction alloc  |
| Memory read (fetch)| `bus.read32()`→`ram.read32()` | Single call via bus delegation      |
| Execute dispatch   | `execute.execute()`           | if/elif chain, relatively cheap     |
| Mnemonic classifier| `instruction_mnemonic()`      | ~3% overhead for stats tracking     |
| NPU vector ops     | `_exec_vmac()` etc.           | Near-zero with Cython, ~50% without |

### Key Observations

1. **The CPU step loop is the bottleneck.** With NPU ops accelerated by Cython, no single component dominates — time is spread evenly across fetch, decode, execute, and per-step bookkeeping. This is the expected balanced state.

2. **NPU vector ops are effectively free with Cython.** The `_accel.pyx` Cython module bypasses the bus entirely, operating directly on the RAM bytearray via typed memoryviews. VMAC dropped from 204K `bus.read8` calls to zero.

3. **Without Cython, NPU vector loops dominate.** In pure Python, `_exec_vmac` alone accounts for ~50% of mnist execution time (204K `bus.read8` calls + 101K `acc_add` calls).

## Pre-optimization Baseline

The original profile (before any optimizations) on a 100K-instruction integer workload:

| Component            | % of time | Function                   | Calls per step |
|----------------------|-----------|----------------------------|----------------|
| Instruction decode   |    15.0%  | `decode.decode()`          | 1              |
| Memory read (fetch)  |    13.4%  | `bus.read32()`             | 1              |
| CPU step overhead    |    14.3%  | `cpu.step()`               | 1              |
| RAM byte reads       |    11.4%  | `ram.read8()`              | 4 (per word)   |
| Instruction dataclass|    10.9%  | `Instruction.__init__()`   | 1              |
| Execute dispatch     |     5.9%  | `execute.execute()`        | 1              |
| Execute handler      |     6.1%  | `_exec_i_arith()` etc.     | 1              |
| Address translation  |     5.5%  | `ram._offset()`            | 4 (per word)   |
| Mnemonic classifier  |     3.4%  | `instruction_mnemonic()`   | 1              |
| Device lookup        |     2.5%  | `bus._find_device()`       | 1              |
| Sign extension       |     2.3%  | `sign_extend()`            | 0-1            |
| Register file        |     3.9%  | `read()` + `write()`       | 1-3            |

The original emulator executed approximately 230,000 instructions per second.

## NPU Instruction Profile

In a neural network workload, NPU instructions dominate execution time differently than the profiled integer loop:

| Operation            | CPU instructions (without NPU) | NPU instructions            | Speedup factor |
|----------------------|--------------------------------|-----------------------------|----------------|
| 128-element dot prod | ~640 (load/mul/add/branch)     | 1 FVMAC + 1 FRSTACC         | ~320x          |
| ReLU (per element)   | 3 (compare + branch + move)    | 1 FRELU                     | ~3x            |
| GELU (per element)   | ~20 (FP erf approximation)     | 1 FGELU                     | ~20x           |
| Softmax (N elems)    | ~8N (exp loop + sum + div)     | FVMAX+FVEXP+FVREDUCE+FVMUL  | ~8x            |
| RMSNorm (N elems)    | ~6N (sq loop + sum + sqrt)     | FVMAC+FRSTACC+FVRSQRT+FVMUL | ~6x            |

The NPU vector instructions (FVMAC, FVEXP, FVMUL) are the highest-value acceleration targets because they collapse O(N) scalar loops into a single instruction dispatch.

## What Hardware Would Accelerate

### Tier 1: Highest Impact

1. **FVMAC / VMAC (vector dot product)**: The inner loop iterates N times in Python, calling `_read_mem_f32()` and `facc_add()` for each element. In hardware, this would be a pipelined multiply-accumulate unit with direct memory access -- completing in N clock cycles with full throughput (1 MAC per cycle).

2. **Memory subsystem**: The current byte-at-a-time `read8()` approach with per-access bounds checking is extremely slow. Hardware would use a word-aligned memory bus with no per-access bounds check.

3. **Instruction decode**: The Python if/elif chain reconstructing immediates from bit fields would be a single-cycle combinational logic block in hardware.

### Tier 2: Medium Impact

4. **FVEXP (vector exponential)**: Currently calls `math.exp()` per element in Python. Hardware could use a lookup table plus polynomial approximation in a dedicated functional unit, achieving 1-4 cycles per element.

5. **FVMUL / VMUL (vector scale)**: Simple multiply loop. Hardware would stream elements through a multiplier pipeline.

6. **FVREDUCE / VREDUCE (vector reduction)**: Accumulation loop. Hardware would use an adder tree for O(log N) latency instead of O(N).

### Tier 3: Lower Impact (Already Fast)

7. **FGELU / GELU**: Single-element operations. The integer GELU is already a table lookup (1 cycle in hardware). FP GELU would need a small polynomial approximation unit.

8. **FRELU / RELU, CLAMP, QMUL**: Simple scalar operations that are already single-instruction in the emulator. Hardware cost is minimal.

## Implemented Optimizations

1. **Bus multi-byte delegation**: The bus detects devices with native `read16`/`read32`/`write16`/`write32` methods at registration time and delegates directly instead of composing from 4x `read8()`. RAM provides these natively via `int.from_bytes()` on its bytearray, so instruction fetch and data loads/stores skip 6 function calls per word access. `load_segment()` also delegates to the device's bulk load when available.

2. **Device lookup fast-path**: `_find_device()` caches the last-hit `DeviceMapping` and checks it before scanning the device list. Since ~99% of accesses target RAM, this turns nearly every lookup into a single comparison.

3. **Instruction `slots=True` dataclass**: Replaced `@dataclass(frozen=True)` with `@dataclass(slots=True)` on the `Instruction` class. The frozen variant uses `object.__setattr__` for each field during `__init__`, adding overhead to every decode cycle. `slots=True` gives the memory benefit of `__slots__` without the per-field cost.

4. **Cython NPU vector acceleration**: The 10 NPU vector inner loops (VMAC, FVMAC, VEXP, FVEXP, VMUL, FVMUL, VREDUCE, FVREDUCE, VMAX, FVMAX) are implemented in Cython (`_accel.pyx`). The Cython kernels access the RAM bytearray directly via typed memoryviews, bypassing the bus/device lookup entirely. This eliminates ~7 Python function calls per vector element. Compile with `make accel`; without compilation, the emulator falls back to pure-Python loops transparently.

   **Impact on mnist (NPU-heavy workload):**

   | Metric               | Pure Python | With Cython | Speedup |
   |----------------------|-------------|-------------|---------|
   | Instructions/sec     | 116K        | 900K        | 7.7x   |
   | `bus.read8` calls    | 204,050     | 786         | 260x   |
   | Total cProfile time  | 0.201s      | 0.027s      | 7.4x   |
   | VMAC in top profile? | #1 (20%)    | Not in top 25 | --   |

## Potential Further Optimizations

These could improve emulator speed further but are not yet implemented:

1. **Dispatch table instead of if/elif**: Replace the opcode if/elif chain with a function pointer array indexed by opcode. Reduces dispatch from O(N opcodes) to O(1).

2. **Optional stats tracking**: Gate `instruction_mnemonic()` behind a flag to remove the ~3% overhead when statistics are not needed.

These remain unimplemented because correctness and readability take priority in an educational emulator. The architecture is designed to mirror how real hardware would work (fetch-decode-execute pipeline), not to maximize Python throughput.
