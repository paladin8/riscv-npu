#ifndef NN_RUNTIME_H
#define NN_RUNTIME_H

#include <stdint.h>

/**
 * Apply a linear layer using NPU MACC instructions.
 *
 * Computes: output[i] = clamp((sum_j(input[j] * weights[i*in_dim+j]) + bias[i]) >> shift, -128, 127)
 * for each output neuron i.
 *
 * @param input   Input values (uint8 or int8, promoted to int32 for MACC).
 * @param weights Weight matrix in row-major order, shape (out_dim, in_dim), int8.
 * @param bias    Bias vector, shape (out_dim,), int32 in accumulator scale.
 * @param shift   Right-shift amount to rescale accumulator to int8 range.
 * @param out_dim Number of output neurons.
 * @param in_dim  Number of input features.
 * @param output  Output buffer, shape (out_dim,), int8 after clamp.
 */
void linear_relu(const uint8_t *input, const int8_t *weights,
                 const int32_t *bias, int32_t shift,
                 int out_dim, int in_dim, int8_t *output);

/**
 * Apply a linear layer and return raw int32 outputs (for argmax).
 *
 * Same as linear_relu but without shift/clamp/relu -- returns raw
 * int32 accumulator values for use with argmax.
 *
 * @param input   Input values (int8, promoted to int32 for MACC).
 * @param weights Weight matrix in row-major order, shape (out_dim, in_dim), int8.
 * @param bias    Bias vector, shape (out_dim,), int32.
 * @param out_dim Number of output neurons.
 * @param in_dim  Number of input features.
 * @param output  Output buffer, shape (out_dim,), int32.
 */
void linear_raw(const int8_t *input, const int8_t *weights,
                const int32_t *bias, int out_dim, int in_dim,
                int32_t *output);

/**
 * Find the index of the maximum value in an int32 array.
 *
 * @param data Array of int32 values.
 * @param len  Length of the array.
 * @return     Index of the maximum element (first one if tied).
 */
int argmax(const int32_t *data, int len);

/**
 * Run full MNIST inference: 784 uint8 pixels -> predicted digit 0-9.
 *
 * Uses the pre-exported weights from weights.h.
 *
 * @param image Flattened 28x28 image as uint8 pixel values [0, 255].
 * @return      Predicted digit 0-9.
 */
int inference(const uint8_t *image);

#endif /* NN_RUNTIME_H */
