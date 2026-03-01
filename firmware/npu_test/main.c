/* NPU instruction test suite.
 *
 * Exercises each NPU instruction with known inputs and compares
 * to expected outputs. Prints PASS/FAIL per test. Uses the
 * write syscall for output.
 */

#include "../common/npu.h"

long write(int fd, const void *buf, long len);
void _exit(int code);

/* Minimal puts (no libc) */
static void print(const char *s) {
    const char *p = s;
    while (*p) p++;
    write(1, s, p - s);
}

static int test_count = 0;
static int fail_count = 0;

static void check(const char *name, int32_t got, int32_t expected) {
    test_count++;
    if (got == expected) {
        print("PASS ");
        print(name);
        print("\n");
    } else {
        fail_count++;
        print("FAIL ");
        print(name);
        print("\n");
    }
}

int main(void) {
    int32_t result;

    /* ---------- MACC + RSTACC ---------- */
    /* 3 * 7 = 21 */
    NPU_MACC(3, 7);
    result = NPU_RSTACC();
    check("MACC single 3*7=21", result, 21);

    /* Chain: 10*20 * 5 = 1000 */
    NPU_MACC(10, 20);
    NPU_MACC(10, 20);
    NPU_MACC(10, 20);
    NPU_MACC(10, 20);
    NPU_MACC(10, 20);
    result = NPU_RSTACC();
    check("MACC chain 5x(10*20)=1000", result, 1000);

    /* Negative operands: (-5) * 6 = -30 */
    NPU_MACC(-5, 6);
    result = NPU_RSTACC();
    check("MACC negative (-5)*6=-30", result, -30);

    /* ---------- RELU ---------- */
    result = NPU_RELU(42);
    check("RELU positive 42", result, 42);

    result = NPU_RELU(-10);
    check("RELU negative -10->0", result, 0);

    result = NPU_RELU(0);
    check("RELU zero", result, 0);

    /* ---------- QMUL ---------- */
    /* (256 * 256) >> 8 = 256 */
    result = NPU_QMUL(256, 256);
    check("QMUL 256*256>>8=256", result, 256);

    /* (100 * 100) >> 8 = 39 (10000 >> 8 = 39) */
    result = NPU_QMUL(100, 100);
    check("QMUL 100*100>>8=39", result, 39);

    /* Negative: (-128 * 64) >> 8 = -32 */
    result = NPU_QMUL(-128, 64);
    check("QMUL neg (-128*64)>>8=-32", result, -32);

    /* ---------- CLAMP ---------- */
    result = NPU_CLAMP(42);
    check("CLAMP in-range 42", result, 42);

    result = NPU_CLAMP(200);
    check("CLAMP above 200->127", result, 127);

    result = NPU_CLAMP(-200);
    check("CLAMP below -200->-128", result, -128);

    result = NPU_CLAMP(127);
    check("CLAMP boundary 127", result, 127);

    result = NPU_CLAMP(-128);
    check("CLAMP boundary -128", result, -128);

    /* ---------- GELU ---------- */
    result = NPU_GELU(0);
    check("GELU zero", result, 0);

    /* GELU of a large positive should be positive */
    result = NPU_GELU(64);
    check("GELU positive>0", (result > 0) ? 1 : 0, 1);

    /* ---------- RSTACC after clean ---------- */
    result = NPU_RSTACC();
    check("RSTACC clean acc=0", result, 0);

    /* ---------- Summary ---------- */
    if (fail_count == 0) {
        print("ALL PASS\n");
        _exit(0);
    } else {
        print("SOME TESTS FAILED\n");
        _exit(1);
    }

    return 0;
}
