/* FPU test: exercises RV32F single-precision floating-point instructions. */

long write(int fd, const void *buf, long len);
void _exit(int code);

static void print(const char *s) {
    const char *p = s;
    while (*p) p++;
    write(1, s, p - s);
}

static void print_int(int val) {
    char buf[12];
    int i = 0;
    int neg = 0;
    unsigned int u;

    if (val < 0) {
        neg = 1;
        u = (unsigned int)(-(val + 1)) + 1;
    } else {
        u = (unsigned int)val;
    }

    if (u == 0) {
        buf[i++] = '0';
    } else {
        while (u > 0) {
            buf[i++] = '0' + (u % 10);
            u /= 10;
        }
    }
    if (neg) buf[i++] = '-';

    /* reverse */
    for (int j = 0; j < i / 2; j++) {
        char tmp = buf[j];
        buf[j] = buf[i - 1 - j];
        buf[i - 1 - j] = tmp;
    }
    write(1, buf, i);
}

static int tests_run = 0;
static int tests_passed = 0;

static void check(const char *name, int condition) {
    tests_run++;
    if (condition) {
        tests_passed++;
        print("  PASS: ");
    } else {
        print("  FAIL: ");
    }
    print(name);
    print("\n");
}

/* Reinterpret float bits as int for exact comparison */
static inline unsigned int float_bits(float f) {
    unsigned int bits;
    __asm__ volatile("fmv.x.w %0, %1" : "=r"(bits) : "f"(f));
    return bits;
}

/* Compare floats with small epsilon */
static int approx_eq(float a, float b) {
    float diff = a - b;
    if (diff < 0.0f) diff = -diff;
    return diff < 0.0001f;
}

static void test_arithmetic(void) {
    print("Arithmetic:\n");
    volatile float a = 3.0f;
    volatile float b = 2.0f;

    check("fadd 3+2=5",       approx_eq(a + b, 5.0f));
    check("fsub 3-2=1",       approx_eq(a - b, 1.0f));
    check("fmul 3*2=6",       approx_eq(a * b, 6.0f));
    check("fdiv 3/2=1.5",     approx_eq(a / b, 1.5f));

    volatile float c = -7.5f;
    volatile float d = 0.25f;
    check("fadd -7.5+0.25=-7.25", approx_eq(c + d, -7.25f));
    check("fmul -7.5*0.25=-1.875", approx_eq(c * d, -1.875f));
}

static void test_sqrt(void) {
    print("Square root:\n");
    volatile float a = 4.0f;
    float result;
    __asm__ volatile("fsqrt.s %0, %1" : "=f"(result) : "f"(a));
    check("fsqrt(4)=2", approx_eq(result, 2.0f));

    volatile float b = 9.0f;
    __asm__ volatile("fsqrt.s %0, %1" : "=f"(result) : "f"(b));
    check("fsqrt(9)=3", approx_eq(result, 3.0f));

    volatile float c = 2.0f;
    __asm__ volatile("fsqrt.s %0, %1" : "=f"(result) : "f"(c));
    check("fsqrt(2)~1.4142", approx_eq(result, 1.41421356f));
}

static void test_comparisons(void) {
    print("Comparisons:\n");
    volatile float a = 1.0f;
    volatile float b = 2.0f;
    volatile float c = 1.0f;

    check("feq 1==1",  a == c);
    check("feq 1!=2",  !(a == b));
    check("flt 1<2",   a < b);
    check("flt !(2<1)", !(b < a));
    check("fle 1<=1",  a <= c);
    check("fle 1<=2",  a <= b);
    check("fle !(2<=1)", !(b <= a));
}

static void test_conversions(void) {
    print("Conversions:\n");

    /* int -> float */
    volatile int ival = 42;
    float fval;
    __asm__ volatile("fcvt.s.w %0, %1" : "=f"(fval) : "r"(ival));
    check("fcvt.s.w 42->42.0", approx_eq(fval, 42.0f));

    volatile int neg = -10;
    __asm__ volatile("fcvt.s.w %0, %1" : "=f"(fval) : "r"(neg));
    check("fcvt.s.w -10->-10.0", approx_eq(fval, -10.0f));

    /* float -> int */
    volatile float fv = 7.9f;
    int result;
    __asm__ volatile("fcvt.w.s %0, %1, rtz" : "=r"(result) : "f"(fv));
    check("fcvt.w.s 7.9->7 (trunc)", result == 7);

    volatile float fv2 = -3.7f;
    __asm__ volatile("fcvt.w.s %0, %1, rtz" : "=r"(result) : "f"(fv2));
    check("fcvt.w.s -3.7->-3 (trunc)", result == -3);
}

static void test_sign_inject(void) {
    print("Sign injection:\n");
    volatile float pos = 5.0f;
    volatile float neg = -3.0f;
    float result;

    /* fsgnj: copy sign from rs2 */
    __asm__ volatile("fsgnj.s %0, %1, %2" : "=f"(result) : "f"(pos), "f"(neg));
    check("fsgnj(5,-3)=-5", approx_eq(result, -5.0f));

    /* fsgnjn: negate sign of rs2 */
    __asm__ volatile("fsgnjn.s %0, %1, %2" : "=f"(result) : "f"(pos), "f"(neg));
    check("fsgnjn(5,-3)=5", approx_eq(result, 5.0f));

    /* fsgnjx: XOR signs — pos ^ neg = neg */
    __asm__ volatile("fsgnjx.s %0, %1, %2" : "=f"(result) : "f"(pos), "f"(neg));
    check("fsgnjx(5,-3)=-5", approx_eq(result, -5.0f));

    /* fneg (pseudo: fsgnjn rd, rs, rs) */
    __asm__ volatile("fneg.s %0, %1" : "=f"(result) : "f"(pos));
    check("fneg(5)=-5", approx_eq(result, -5.0f));

    /* fabs (pseudo: fsgnjx rd, rs, rs) */
    __asm__ volatile("fabs.s %0, %1" : "=f"(result) : "f"(neg));
    check("fabs(-3)=3", approx_eq(result, 3.0f));
}

static void test_min_max(void) {
    print("Min/Max:\n");
    volatile float a = 1.5f;
    volatile float b = 3.5f;
    float result;

    __asm__ volatile("fmin.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
    check("fmin(1.5,3.5)=1.5", approx_eq(result, 1.5f));

    __asm__ volatile("fmax.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
    check("fmax(1.5,3.5)=3.5", approx_eq(result, 3.5f));
}

static void test_fmadd(void) {
    print("Fused multiply-add:\n");
    volatile float a = 2.0f;
    volatile float b = 3.0f;
    volatile float c = 1.0f;
    float result;

    /* fmadd: a*b + c = 7 */
    __asm__ volatile("fmadd.s %0, %1, %2, %3"
                     : "=f"(result) : "f"(a), "f"(b), "f"(c));
    check("fmadd 2*3+1=7", approx_eq(result, 7.0f));

    /* fmsub: a*b - c = 5 */
    __asm__ volatile("fmsub.s %0, %1, %2, %3"
                     : "=f"(result) : "f"(a), "f"(b), "f"(c));
    check("fmsub 2*3-1=5", approx_eq(result, 5.0f));

    /* fnmsub: -(a*b) + c = -5 */
    __asm__ volatile("fnmsub.s %0, %1, %2, %3"
                     : "=f"(result) : "f"(a), "f"(b), "f"(c));
    check("fnmsub -(2*3)+1=-5", approx_eq(result, -5.0f));

    /* fnmadd: -(a*b) - c = -7 */
    __asm__ volatile("fnmadd.s %0, %1, %2, %3"
                     : "=f"(result) : "f"(a), "f"(b), "f"(c));
    check("fnmadd -(2*3)-1=-7", approx_eq(result, -7.0f));
}

static void test_load_store(void) {
    print("Load/Store:\n");
    volatile float mem_val = 42.5f;
    float loaded;

    /* flw / fsw are generated by the compiler for float memory access */
    loaded = mem_val;
    check("flw/fsw roundtrip", approx_eq(loaded, 42.5f));

    volatile float arr[3];
    arr[0] = 1.0f;
    arr[1] = 2.0f;
    arr[2] = 3.0f;
    check("store/load arr[0]", approx_eq(arr[0], 1.0f));
    check("store/load arr[2]", approx_eq(arr[2], 3.0f));
}

int main(void) {
    print("=== FPU Test Suite ===\n");

    test_load_store();
    test_arithmetic();
    test_sqrt();
    test_comparisons();
    test_conversions();
    test_sign_inject();
    test_min_max();
    test_fmadd();

    print("\nResults: ");
    print_int(tests_passed);
    print("/");
    print_int(tests_run);
    print(" passed\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
