/* Neural network runtime for quantized int8 inference using NPU instructions.
 *
 * Uses NPU_MACC for dot products and NPU_RSTACC to read/clear the accumulator.
 * NPU_CLAMP is used for int8 clamping after rescaling.
 * NPU_RELU is used for ReLU activation.
 */

#include "nn_runtime.h"
#include "weights.h"
#include "../common/npu.h"

void linear_relu(const uint8_t *input, const int8_t *weights,
                 const int32_t *bias, int32_t shift,
                 int out_dim, int in_dim, int8_t *output)
{
    for (int i = 0; i < out_dim; i++) {
        /* Clear accumulator */
        NPU_RSTACC();

        /* Multiply-accumulate: acc += input[j] * weights[i*in_dim+j] */
        const int8_t *row = &weights[i * in_dim];
        for (int j = 0; j < in_dim; j++) {
            NPU_MACC((int32_t)input[j], (int32_t)row[j]);
        }

        /* Read accumulator and add bias */
        int32_t acc = NPU_RSTACC();
        acc += bias[i];

        /* Rescale: arithmetic right shift to bring into int8 range */
        acc = acc >> shift;

        /* Clamp to int8 [-128, 127] */
        acc = NPU_CLAMP(acc);

        /* ReLU: max(0, x) */
        acc = NPU_RELU(acc);

        output[i] = (int8_t)acc;
    }
}

void linear_raw(const int8_t *input, const int8_t *weights,
                const int32_t *bias, int out_dim, int in_dim,
                int32_t *output)
{
    for (int i = 0; i < out_dim; i++) {
        /* Clear accumulator */
        NPU_RSTACC();

        /* Multiply-accumulate */
        const int8_t *row = &weights[i * in_dim];
        for (int j = 0; j < in_dim; j++) {
            NPU_MACC((int32_t)input[j], (int32_t)row[j]);
        }

        /* Read accumulator and add bias */
        int32_t acc = NPU_RSTACC();
        acc += bias[i];

        output[i] = acc;
    }
}

int argmax(const int32_t *data, int len)
{
    int best_idx = 0;
    int32_t best_val = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int inference(const uint8_t *image)
{
    int8_t hidden[128];
    int32_t logits[10];

    /* Layer 1: 784 -> 128, with ReLU */
    linear_relu(image, (const int8_t *)W1, B1, SHIFT1, 128, 784, hidden);

    /* Layer 2: 128 -> 10, raw int32 output for argmax */
    linear_raw(hidden, (const int8_t *)W2, B2, 10, 128, logits);

    /* Argmax to find predicted digit */
    return argmax(logits, 10);
}
