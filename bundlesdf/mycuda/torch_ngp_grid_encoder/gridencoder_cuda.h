// Torch-free declarations of the CUDA launchers in gridencoder.cu (instantiated for float, double, __half).
#pragma once
#include <stdint.h>
#include <cuda_fp16.h>

template <typename scalar_t>
void grid_encode_forward_cuda(const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, const uint32_t gridtype, const bool align_corners);

template <typename scalar_t>
void grid_encode_backward_cuda(const scalar_t *grad, const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs, const uint32_t gridtype, const bool align_corners);
