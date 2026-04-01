#ifndef NPU_FP_H
#define NPU_FP_H

/* FP NPU intrinsics: opcode 0x2B (custom-1)
 *
 * All instructions use R-type encoding.
 * Float operands use "f" constraint, integer operands use "r" constraint.
 */

/* FMACC: facc += f(a) * f(b) — FP multiply-accumulate to float64 accumulator */
static inline void NPU_FMACC(float a, float b) {
    asm volatile(".insn r 0x2B, 0x0, 0x00, f0, %0, %1" :: "f"(a), "f"(b));
}

/* FVMAC: facc += dot(mem_f32[a..+n], mem_f32[b..+n]) — FP vector dot product */
static inline void NPU_FVMAC(void *addr_a, void *addr_b, int len) {
    register void *_a asm("a0") = addr_a;
    register void *_b asm("a1") = addr_b;
    register int _n asm("a2") = len;
    asm volatile(".insn r 0x2B, 0x0, 0x01, a2, a0, a1"
                 :: "r"(_a), "r"(_b), "r"(_n) : "memory");
}

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
static inline void NPU_FVEXP(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x02, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

/* FVRSQRT: returns 1/sqrt(mem_f32[addr]) as float */
static inline float NPU_FVRSQRT(void *addr) {
    float result;
    asm volatile(".insn r 0x2B, 0x0, 0x03, %0, %1, x0" : "=f"(result) : "r"(addr) : "memory");
    return result;
}

/* FVMUL: dst[i] = src[i] * (float32)facc for i in 0..n-1 */
static inline void NPU_FVMUL(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x04, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

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

/* --- Phase 11: Arrax vector instructions --- */

/* FVADD: dst[i] = src1[i] + src2[i], i in 0..n-1 — result in-place at src2 */
static inline void NPU_FVADD(void *src1, void *src2, int n) {
    register void *_s1 asm("a0") = src1;
    register void *_s2 asm("a1") = src2;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x07, a2, a0, a1"
                 :: "r"(_s1), "r"(_s2), "r"(_n) : "memory");
}

/* FVSUB: dst[i] = src1[i] - src2[i], i in 0..n-1 — result in-place at src2 */
static inline void NPU_FVSUB(void *src1, void *src2, int n) {
    register void *_s1 asm("a0") = src1;
    register void *_s2 asm("a1") = src2;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x08, a2, a0, a1"
                 :: "r"(_s1), "r"(_s2), "r"(_n) : "memory");
}

/* FVRELU: dst[i] = max(src[i], 0.0), i in 0..n-1 */
static inline void NPU_FVRELU(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x09, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

/* FVGELU: dst[i] = gelu(src[i]), i in 0..n-1 */
static inline void NPU_FVGELU(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x0A, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

/* FVDIV: dst[i] = src[i] / (float32)facc, i in 0..n-1 */
static inline void NPU_FVDIV(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x0B, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

/* FVSUB_SCALAR: dst[i] = src[i] - (float32)facc, i in 0..n-1 */
static inline void NPU_FVSUB_SCALAR(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x2B, 0x0, 0x0C, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

#endif
