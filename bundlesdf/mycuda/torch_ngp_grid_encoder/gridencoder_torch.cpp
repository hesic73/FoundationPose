// torch::Tensor wrappers for gridencoder.cu (moved out of the .cu so it stays torch-free; see gridencoder_cuda.h).
#include <torch/torch.h>
#include "gridencoder.h"
#include "gridencoder_cuda.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

// at::Half -> __half (same 16-bit layout) for the torch-free launchers
template <typename T> struct cuda_t { using type = T; };
template <> struct cuda_t<at::Half> { using type = __half; };
template <typename T> static inline typename cuda_t<T>::type *cptr(const at::Tensor &t) { return reinterpret_cast<typename cuda_t<T>::type *>(t.data_ptr<T>()); }
void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx, const uint32_t gridtype, const bool align_corners) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(dy_dx);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "grid_encode_forward", ([&] {
        grid_encode_forward_cuda<cuda_t<scalar_t>::type>(inputs.data_ptr<float>(), cptr<scalar_t>(embeddings), offsets.data_ptr<int>(), cptr<scalar_t>(outputs), B, D, C, L, S, H, calc_grad_inputs, cptr<scalar_t>(dy_dx), gridtype, align_corners);
    }));
}

void grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs, const uint32_t gridtype, const bool align_corners) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_inputs);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "grid_encode_backward", ([&] {
        grid_encode_backward_cuda<cuda_t<scalar_t>::type>(cptr<scalar_t>(grad), inputs.data_ptr<float>(), cptr<scalar_t>(embeddings), offsets.data_ptr<int>(), cptr<scalar_t>(grad_embeddings), B, D, C, L, S, H, calc_grad_inputs, cptr<scalar_t>(dy_dx), cptr<scalar_t>(grad_inputs), gridtype, align_corners);
    }));

}
