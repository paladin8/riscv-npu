/* Phase 11 NPU vector instruction test suite.
 *
 * Exercises FVADD, FVSUB, FVRELU, FVGELU, FVDIV, and FVSUB_SCALAR
 * with known inputs and compares to expected outputs.
 * Prints PASS/FAIL per test, "ALL PASS" at end.
 */

#include "../common/npu_fp.h"

long write(int fd, const void *buf, long len);
void _exit(int code);

static void print(const char *s) {
    const char *p = s;
    while (*p) p++;
    write(1, s, p - s);
}

static int test_count = 0;
static int fail_count = 0;

static void check_f32(const char *name, float got, float expected, float tol) {
    test_count++;
    float diff = got - expected;
    if (diff < 0) diff = -diff;
    if (diff <= tol) {
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
    float a[4], b[4], dst[4];

    /* ---------- FVADD ---------- */
    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f;
    b[0] = 4.0f; b[1] = 5.0f; b[2] = 6.0f;
    NPU_FVADD(a, b, 3);
    check_f32("FVADD [1+4]=5",  b[0], 5.0f, 0.001f);
    check_f32("FVADD [2+5]=7",  b[1], 7.0f, 0.001f);
    check_f32("FVADD [3+6]=9",  b[2], 9.0f, 0.001f);

    /* ---------- FVSUB ---------- */
    a[0] = 10.0f; a[1] = 20.0f;
    b[0] = 3.0f;  b[1] = 7.0f;
    NPU_FVSUB(a, b, 2);
    check_f32("FVSUB [10-3]=7",  b[0], 7.0f,  0.001f);
    check_f32("FVSUB [20-7]=13", b[1], 13.0f, 0.001f);

    /* ---------- FVRELU ---------- */
    a[0] = -1.0f; a[1] = 0.0f; a[2] = 1.0f;
    NPU_FVRELU(a, dst, 3);
    check_f32("FVRELU [-1]->0", dst[0], 0.0f, 0.001f);
    check_f32("FVRELU [0]->0",  dst[1], 0.0f, 0.001f);
    check_f32("FVRELU [1]->1",  dst[2], 1.0f, 0.001f);

    /* ---------- FVGELU ---------- */
    /* gelu(0) = 0, gelu(1) ~ 0.8413, gelu(-1) ~ -0.1587 */
    a[0] = 0.0f; a[1] = 1.0f; a[2] = -1.0f;
    NPU_FVGELU(a, dst, 3);
    check_f32("FVGELU [0]~0",       dst[0], 0.0f,    0.01f);
    check_f32("FVGELU [1]~0.841",   dst[1], 0.8413f, 0.01f);
    check_f32("FVGELU [-1]~-0.159", dst[2], -0.1587f, 0.01f);

    /* ---------- FVDIV ---------- */
    a[0] = 10.0f; a[1] = 20.0f;
    /* Load 5.0 into facc via FMACC(5.0, 1.0) */
    NPU_FMACC(5.0f, 1.0f);
    NPU_FVDIV(a, dst, 2);
    /* Read and discard facc to clean up */
    (void)NPU_FRSTACC();
    check_f32("FVDIV [10/5]=2", dst[0], 2.0f, 0.001f);
    check_f32("FVDIV [20/5]=4", dst[1], 4.0f, 0.001f);

    /* ---------- FVSUB_SCALAR ---------- */
    a[0] = 10.0f; a[1] = 20.0f;
    /* Load 3.0 into facc */
    NPU_FMACC(3.0f, 1.0f);
    NPU_FVSUB_SCALAR(a, dst, 2);
    (void)NPU_FRSTACC();
    check_f32("FVSUB_SCALAR [10-3]=7",  dst[0], 7.0f,  0.001f);
    check_f32("FVSUB_SCALAR [20-3]=17", dst[1], 17.0f, 0.001f);

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
