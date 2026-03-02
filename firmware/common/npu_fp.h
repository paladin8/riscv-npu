#ifndef NPU_FP_H
#define NPU_FP_H

/* FP NPU intrinsics: opcode 0x2B (custom-1)
 *
 * All instructions use R-type encoding.
 * Float operands use "f" constraint, integer operands use "r" constraint.
 */

/* FMACC: facc += f(a) * f(b) — FP multiply-accumulate to float64 accumulator */
#define NPU_FMACC(a, b) \
    asm volatile(".insn r 0x2B, 0x0, 0x00, f0, %0, %1" :: "f"(a), "f"(b))

/* FVMAC: facc += dot(mem_f32[a..+n], mem_f32[b..+n]) — FP vector dot product */
#define NPU_FVMAC(addr_a, addr_b, len) do { \
    register void *_a asm("a0") = (void *)(addr_a); \
    register void *_b asm("a1") = (void *)(addr_b); \
    register int _n asm("a2") = (len); \
    asm volatile(".insn r 0x2B, 0x0, 0x01, a2, a0, a1" \
                 :: "r"(_a), "r"(_b), "r"(_n) : "memory"); \
} while (0)

/* FRSTACC: f[rd] = (float32)facc; facc = 0.0 — read and reset FP accumulator */
static inline float NPU_FRSTACC(void) {
    float result;
    asm volatile(".insn r 0x2B, 0x5, 0x00, %0, f0, f0" : "=f"(result));
    return result;
}

/* FRELU: f[rd] = max(f[rs1], +0.0) — FP ReLU activation */
static inline float NPU_FRELU(float src) {
    float dst;
    asm volatile(".insn r 0x2B, 0x1, 0x00, %0, %1, f0" : "=f"(dst) : "f"(src));
    return dst;
}

/* FGELU: f[rd] = gelu(f[rs1]) — FP GELU activation at full precision */
static inline float NPU_FGELU(float src) {
    float dst;
    asm volatile(".insn r 0x2B, 0x4, 0x00, %0, %1, f0" : "=f"(dst) : "f"(src));
    return dst;
}

/* FVEXP: dst[i] = exp(src[i]) for i in 0..n-1, float32 arrays */
#define NPU_FVEXP(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x02, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)

/* FVRSQRT: returns 1/sqrt(mem_f32[addr]) as float */
static inline float NPU_FVRSQRT(void *addr) {
    float result;
    asm volatile(".insn r 0x2B, 0x0, 0x03, %0, %1, x0" : "=f"(result) : "r"(addr) : "memory");
    return result;
}

/* FVMUL: dst[i] = src[i] * (float32)facc for i in 0..n-1 */
#define NPU_FVMUL(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x04, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)

/* FVREDUCE: returns sum of float32 array[0..n-1] */
static inline float NPU_FVREDUCE(void *addr, int n) {
    float result;
    asm volatile(".insn r 0x2B, 0x0, 0x05, %0, %1, %2" : "=f"(result) : "r"(addr), "r"(n) : "memory");
    return result;
}

/* FVMAX: returns max of float32 array[0..n-1] */
static inline float NPU_FVMAX(void *addr, int n) {
    float result;
    asm volatile(".insn r 0x2B, 0x0, 0x06, %0, %1, %2" : "=f"(result) : "r"(addr), "r"(n) : "memory");
    return result;
}
#endif
