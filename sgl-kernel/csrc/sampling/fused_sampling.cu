/*
 * Copyright (c) 2025 by SGLang team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fused_sampling.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace sglang {
namespace sampling {

/*
 * PyTorch binding for fused sampling kernel
 *
 * This function provides a PyTorch interface to the fused sampling kernel
 * that combines temperature scaling, softmax, top-k/top-p filtering, and sampling
 * into a single CUDA kernel call.
 *
 * Args:
 *   logits: Input logits tensor [batch_size, vocab_size], float32/float16/bfloat16
 *   temperatures: Temperature values [batch_size] or None, float32
 *   top_k: Top-k values [batch_size] or None, int32
 *   top_p: Top-p values [batch_size] or None, float32
 *   uniform_samples: Random numbers for sampling [batch_size], float32
 *
 * Returns:
 *   samples: Sampled token IDs [batch_size], int32
 *
 * Example:
 *   >>> logits = torch.randn(4, 32000, device='cuda')
 *   >>> temperatures = torch.full((4,), 0.7, device='cuda')
 *   >>> top_k = torch.full((4,), 50, dtype=torch.int32, device='cuda')
 *   >>> top_p = torch.full((4,), 0.9, device='cuda')
 *   >>> uniform_samples = torch.rand(4, device='cuda')
 *   >>> samples = fused_sampling(logits, temperatures, top_k, top_p, uniform_samples)
 */
torch::Tensor fused_sampling_from_logits(
    torch::Tensor logits,              // [batch_size, vocab_size]
    torch::optional<torch::Tensor> temperatures,  // [batch_size] or None
    torch::optional<torch::Tensor> top_k,         // [batch_size] or None
    torch::optional<torch::Tensor> top_p,         // [batch_size] or None
    torch::Tensor uniform_samples      // [batch_size]
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(uniform_samples);

    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch_size, vocab_size]");
    TORCH_CHECK(uniform_samples.dim() == 1, "uniform_samples must be 1D [batch_size]");

    const int batch_size = logits.size(0);
    const int vocab_size = logits.size(1);

    TORCH_CHECK(uniform_samples.size(0) == batch_size,
                "uniform_samples batch size must match logits batch size");

    // Check optional tensors
    const float* temperatures_ptr = nullptr;
    if (temperatures.has_value()) {
        auto temp_tensor = temperatures.value();
        CHECK_INPUT(temp_tensor);
        TORCH_CHECK(temp_tensor.dim() == 1 && temp_tensor.size(0) == batch_size,
                    "temperatures must be 1D [batch_size]");
        TORCH_CHECK(temp_tensor.dtype() == torch::kFloat32,
                    "temperatures must be float32");
        temperatures_ptr = temp_tensor.data_ptr<float>();
    }

    const int* top_k_ptr = nullptr;
    if (top_k.has_value()) {
        auto topk_tensor = top_k.value();
        CHECK_INPUT(topk_tensor);
        TORCH_CHECK(topk_tensor.dim() == 1 && topk_tensor.size(0) == batch_size,
                    "top_k must be 1D [batch_size]");
        TORCH_CHECK(topk_tensor.dtype() == torch::kInt32,
                    "top_k must be int32");
        top_k_ptr = topk_tensor.data_ptr<int>();
    }

    const float* top_p_ptr = nullptr;
    if (top_p.has_value()) {
        auto topp_tensor = top_p.value();
        CHECK_INPUT(topp_tensor);
        TORCH_CHECK(topp_tensor.dim() == 1 && topp_tensor.size(0) == batch_size,
                    "top_p must be 1D [batch_size]");
        TORCH_CHECK(topp_tensor.dtype() == torch::kFloat32,
                    "top_p must be float32");
        top_p_ptr = topp_tensor.data_ptr<float>();
    }

    // Allocate output tensor
    auto samples = torch::empty({batch_size},
                                torch::dtype(torch::kInt32).device(logits.device()));

    // Allocate scratch space for probabilities
    auto probs_scratch = torch::empty_like(logits, torch::dtype(torch::kFloat32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch based on logits dtype
    cudaError_t status = cudaSuccess;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "fused_sampling_kernel", [&] {
            status = LaunchFusedSampling<scalar_t, int32_t>(
                logits.data_ptr<scalar_t>(),
                temperatures_ptr,
                top_k_ptr,
                top_p_ptr,
                uniform_samples.data_ptr<float>(),
                samples.data_ptr<int32_t>(),
                probs_scratch.data_ptr<float>(),
                batch_size,
                vocab_size,
                stream
            );
        }
    );

    TORCH_CHECK(status == cudaSuccess,
                "fused_sampling kernel failed with error: ",
                cudaGetErrorString(status));

    return samples;
}

}  // namespace sampling
}  // namespace sglang

// PyBind11 registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sampling_from_logits",
          &sglang::sampling::fused_sampling_from_logits,
          "Fused sampling kernel: temperature scaling + softmax + top-k/top-p + sampling",
          py::arg("logits"),
          py::arg("temperatures") = py::none(),
          py::arg("top_k") = py::none(),
          py::arg("top_p") = py::none(),
          py::arg("uniform_samples")
    );
}
