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
#endif
