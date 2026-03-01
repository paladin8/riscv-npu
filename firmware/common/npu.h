#ifndef NPU_H
#define NPU_H
#include <stdint.h>

#define NPU_MACC(a, b) \
    asm volatile(".insn r 0x0B, 0x0, 0x00, x0, %0, %1" :: "r"(a), "r"(b))

#define NPU_VMAC(addr_a, addr_b, len) do { \
    register void *_a asm("a0") = (void *)(addr_a); \
    register void *_b asm("a1") = (void *)(addr_b); \
    register int _n asm("a2") = (len); \
    asm volatile(".insn r 0x0B, 0x0, 0x01, a2, a0, a1" \
                 :: "r"(_a), "r"(_b), "r"(_n) : "memory"); \
} while (0)

static inline int32_t NPU_RSTACC(void) {
    int32_t result;
    asm volatile(".insn r 0x0B, 0x5, 0x00, %0, x0, x0" : "=r"(result));
    return result;
}

static inline int32_t NPU_RELU(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x1, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

static inline int32_t NPU_GELU(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x4, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

static inline int32_t NPU_QMUL(int32_t a, int32_t b) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x2, 0x00, %0, %1, %2" : "=r"(dst) : "r"(a), "r"(b));
    return dst;
}

static inline int32_t NPU_CLAMP(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x3, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

/* Phase 7: Transformer vector instructions */

/* VEXP: dst[i] = exp(src[i]) for i in 0..n-1, Q16.16 fixed-point int32 arrays */
#define NPU_VEXP(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x0B, 0x0, 0x02, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)

/* VRSQRT: returns 1/sqrt(mem[addr]) in Q16.16 fixed-point */
static inline int32_t NPU_VRSQRT(void *addr) {
    int32_t result;
    asm volatile(".insn r 0x0B, 0x0, 0x03, %0, %1, x0" : "=r"(result) : "r"(addr) : "memory");
    return result;
}

/* VMUL: dst[i] = clamp((src[i] * acc_lo) >> 16, -128, 127) for i in 0..n-1 */
#define NPU_VMUL(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x0B, 0x0, 0x04, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)

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
