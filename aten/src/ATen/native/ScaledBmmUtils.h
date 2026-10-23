#pragma once

#include <ATen/core/Tensor.h>
#include <optional>
#include <utility>

namespace at::native {

TORCH_API std::pair<bool, bool> prepare_scaled_bmm_output(
    const Tensor& self, const Tensor& mat2,
    const Tensor& scale_a, const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& scale_result,
    std::optional<ScalarType> out_dtype, Tensor& out);

TORCH_API Tensor& _scaled_bmm_out(
    const Tensor& self, const Tensor& mat2,
    const Tensor& scale_a, const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& scale_result,
    std::optional<ScalarType> out_dtype, bool use_fast_accum, Tensor& out);

} // namespace at::native
