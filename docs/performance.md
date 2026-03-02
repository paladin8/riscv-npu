# Performance Profile

## Methodology

Profiled with Python 3.14 `cProfile` on a 100,000-cycle hot loop (ADDI/BEQ/JAL) and a 186-cycle Fibonacci program. The emulator executes approximately 230,000 instructions per second on a single core.

## Where Time Is Spent

Profiling a 100,000-instruction workload shows the following breakdown by total time:

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

### Key Observations

1. **Instruction fetch is the biggest single cost**: `read32()` calls `read8()` four times, each of which does bounds checking via `_offset()`. A single 4-byte read path would cut 8 function calls per instruction fetch.

2. **Instruction decoding and dataclass allocation are significant**: Creating a frozen `Instruction` dataclass on every cycle costs ~26% of total time (decode + `__init__`). A mutable struct or tuple return would avoid per-cycle allocation.

3. **Execute dispatch is relatively cheap**: The if/elif chain in `execute()` is fast. Most time is in the actual operation handlers.

4. **The mnemonic classifier adds ~3% overhead**: Acceptable for development; could be disabled in production mode.

## NPU Instruction Profile

In a neural network workload, NPU instructions dominate execution time differently than the profiled integer loop:

| Operation            | CPU instructions (without NPU) | NPU instructions | Speedup factor |
|----------------------|--------------------------------|-------------------|----------------|
| 128-element dot prod | ~640 (load/mul/add/branch)     | 1 FVMAC + 1 FRSTACC | ~320x          |
| ReLU (per element)   | 3 (compare + branch + move)    | 1 FRELU           | ~3x            |
| GELU (per element)   | ~20 (FP erf approximation)     | 1 FGELU           | ~20x           |
| Softmax (N elems)    | ~8N (exp loop + sum + div)     | FVMAX+FVEXP+FVREDUCE+FVMUL | ~8x  |
| RMSNorm (N elems)    | ~6N (sq loop + sum + sqrt)     | FVMAC+FRSTACC+FVRSQRT+FVMUL | ~6x |

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

## Potential Software Optimizations

These optimizations could improve emulator speed without changing the architecture:

1. **Word-aligned memory reads**: Replace 4x `read8()` with a direct `int.from_bytes()` on the RAM bytearray. Estimated 2-3x speedup for instruction fetch.

2. **Dispatch table instead of if/elif**: Replace the opcode if/elif chain with a function pointer array indexed by opcode. Reduces dispatch from O(N opcodes) to O(1).

3. **Avoid Instruction dataclass allocation**: Use a reusable mutable object or return a tuple. Saves one object allocation per cycle (~11% of time).

4. **Inline device lookup**: Since most accesses hit RAM, add a fast path that skips the device list scan for addresses in the RAM range.

5. **Optional stats tracking**: Gate `instruction_mnemonic()` behind a flag to remove the ~3% overhead when statistics are not needed.

These optimizations are not implemented because correctness and readability take priority in an educational emulator. The architecture is designed to mirror how real hardware would work (fetch-decode-execute pipeline), not to maximize Python throughput.
