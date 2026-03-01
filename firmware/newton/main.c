/* Newton's method: compute square roots using the Babylonian method.
 *
 * x_{n+1} = (x_n + S / x_n) / 2
 *
 * Iterates until convergence, then verifies each result against
 * the hardware FSQRT instruction. Returns 1 if all roots are
 * accurate (relative error < 1e-6), 0 on failure.
 */

int main(void) {
    float inputs[] = {2.0f, 3.0f, 0.25f, 100.0f, 144.0f, 0.01f, 1000000.0f};
    int n = sizeof(inputs) / sizeof(inputs[0]);

    for (int i = 0; i < n; i++) {
        float S = inputs[i];

        /* Initial guess: S/2 (or 1.0 for small values) */
        float x = S > 1.0f ? S * 0.5f : 1.0f;

        /* Iterate until convergence */
        for (int iter = 0; iter < 30; iter++) {
            float next = (x + S / x) * 0.5f;
            /* Converged when update stops changing the value */
            if (next == x)
                break;
            x = next;
        }

        /* Verify against hardware fsqrt */
        float expected;
        __asm__ volatile("fsqrt.s %0, %1" : "=f"(expected) : "f"(S));

        /* Check relative error: |x - expected| / expected < 1e-6 */
        float diff = x - expected;
        if (diff < 0.0f) diff = -diff;
        if (diff > expected * 0.000001f)
            return 0;  /* fail */
    }

    return 1;  /* pass: all roots accurate */
}
