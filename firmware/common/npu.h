#ifndef NPU_H
#define NPU_H
#include <stdint.h>

/* MACC: {acc_hi, acc_lo} += signed(a) * signed(b) — integer multiply-accumulate */
static inline void NPU_MACC(int32_t a, int32_t b) {
    asm volatile(".insn r 0x0B, 0x0, 0x00, x0, %0, %1" :: "r"(a), "r"(b));
}

/* VMAC: acc += dot(mem_i8[a..+n], mem_i8[b..+n]) — integer vector dot product */
static inline void NPU_VMAC(void *addr_a, void *addr_b, int len) {
    register void *_a asm("a0") = addr_a;
    register void *_b asm("a1") = addr_b;
    register int _n asm("a2") = len;
    asm volatile(".insn r 0x0B, 0x0, 0x01, a2, a0, a1"
                 :: "r"(_a), "r"(_b), "r"(_n) : "memory");
}

/* RSTACC: rd = acc_lo; acc = 0 — reset integer accumulator */
static inline int32_t NPU_RSTACC(void) {
    int32_t result;
    asm volatile(".insn r 0x0B, 0x5, 0x00, %0, x0, x0" : "=r"(result));
    return result;
}

/* RELU: rd = max(signed(rs1), 0) — integer ReLU activation */
static inline int32_t NPU_RELU(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x1, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

/* GELU: rd = gelu_table[rs1[7:0]] — integer GELU via lookup table */
static inline int32_t NPU_GELU(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x4, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

/* QMUL: rd = (signed(rs1) * signed(rs2)) >> 8 — quantized multiply */
static inline int32_t NPU_QMUL(int32_t a, int32_t b) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x2, 0x00, %0, %1, %2" : "=r"(dst) : "r"(a), "r"(b));
    return dst;
}

/* CLAMP: rd = clamp(signed(rs1), -128, 127) — clamp to int8 */
static inline int32_t NPU_CLAMP(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x3, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

/* Phase 7: Transformer vector instructions */

/* VEXP: dst[i] = exp(src[i]) for i in 0..n-1, Q16.16 fixed-point int32 arrays */
static inline void NPU_VEXP(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x0B, 0x0, 0x02, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

/* VRSQRT: returns 1/sqrt(mem[addr]) in Q16.16 fixed-point */
static inline int32_t NPU_VRSQRT(void *addr) {
    int32_t result;
    asm volatile(".insn r 0x0B, 0x0, 0x03, %0, %1, x0" : "=r"(result) : "r"(addr) : "memory");
    return result;
}

/* VMUL: dst[i] = clamp((src[i] * acc_lo) >> 16, -128, 127) for i in 0..n-1 */
static inline void NPU_VMUL(void *src, void *dst, int n) {
    register void *_s asm("a0") = src;
    register void *_d asm("a1") = dst;
    register int _n asm("a2") = n;
    asm volatile(".insn r 0x0B, 0x0, 0x04, a2, a0, a1"
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory");
}

/* VREDUCE: returns sum of int32 array[0..n-1] */
static inline int32_t NPU_VREDUCE(void *addr, int n) {
    int32_t result;
    asm volatile(".insn r 0x0B, 0x0, 0x05, %0, %1, %2" : "=r"(result) : "r"(addr), "r"(n) : "memory");
    return result;
}

/* VMAX: returns max of int32 array[0..n-1] */
static inline int32_t NPU_VMAX(void *addr, int n) {
    int32_t result;
    asm volatile(".insn r 0x0B, 0x0, 0x06, %0, %1, %2" : "=r"(result) : "r"(addr), "r"(n) : "memory");
    return result;
}
#endif
