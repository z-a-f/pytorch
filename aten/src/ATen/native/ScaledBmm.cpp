#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/BlasBackend.h>
#include <c10/util/irange.h>
#include <ATen/Context.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ScaledBmmUtils.h>
#include <ATen/native/ScaledBlasUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_bmm_native.h>
#include <ATen/ops/_scaled_mm.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {
namespace {

using at::blas::ScalingType;

void check_inputs(const Tensor& a, const Tensor& b, const std::optional<Tensor>& bias) {
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "_scaled_bmm inputs must be 3D");
  TORCH_CHECK(a.size(0) == b.size(0), "_scaled_bmm inputs must have the same batch size");
  TORCH_CHECK(a.size(2) == b.size(1), "_scaled_bmm contraction dimensions must match");
  TORCH_CHECK(a.device() == b.device(), "_scaled_bmm inputs must be on the same device");
  TORCH_CHECK(isFloat8Type(a.scalar_type()) && isFloat8Type(b.scalar_type()),
      "_scaled_bmm inputs must be Float8 tensors");
  if (bias) {
    TORCH_CHECK(bias->device() == a.device(), "bias must be on the input device");
    TORCH_CHECK((bias->dim() == 1 && bias->size(0) == b.size(2)) ||
        (bias->dim() == 2 && bias->size(0) == a.size(0) && bias->size(1) == b.size(2)),
        "bias must have shape (N,) or (B, N)");
  }
}

// The leading batch dimension is removed only after the per-matrix layout matches.
bool scale_matches(const Tensor& s, int64_t batch, bool batched,
    ScalingType recipe, int64_t outer, bool rhs) {
  if (batched && (s.dim() == 0 || s.size(0) != batch)) {
    return false;
  }
  const auto sizes = s.sizes().slice(batched ? 1 : 0);
  int64_t numel = 1;
  for (const auto size : sizes) {
    numel *= size;
  }
  if (s.scalar_type() != kFloat) {
    return false;
  }
  if (recipe == ScalingType::TensorWise) {
    return numel == 1;
  }
  return sizes.size() == 2 && sizes[rhs ? 1 : 0] == outer &&
      sizes[rhs ? 0 : 1] == 1;
}

std::pair<bool, bool> infer_scales(const Tensor& a, const Tensor& b,
    const Tensor& sa, const Tensor& sb) {
  TORCH_CHECK(sa.device() == a.device() && sb.device() == a.device(),
      "scales must be on the input device");
  std::optional<std::pair<bool, bool>> result;
  for (const auto recipe : {ScalingType::TensorWise, ScalingType::RowWise}) {
    for (const bool ba : {false, true}) {
      for (const bool bb : {false, true}) {
        if (scale_matches(sa, a.size(0), ba, recipe, a.size(1), false) &&
            scale_matches(sb, a.size(0), bb, recipe, b.size(2), true)) {
          const auto candidate = std::make_pair(ba, bb);
          TORCH_CHECK(!result || *result == candidate || a.size(0) == 1,
              "Ambiguous scale shapes for _scaled_bmm; add singleton dimensions "
              "or use explicit recipes with _scaled_bmm_v2");
          if (!result) {
            result = candidate;
          }
        }
      }
    }
  }
  TORCH_CHECK(result, "Invalid scaling configuration for _scaled_bmm");
  return *result;
}

} // namespace

Tensor& _scaled_bmm_out(const Tensor& self, const Tensor& mat2,
    const Tensor& scale_a, const Tensor& scale_b,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& scale_result,
    std::optional<ScalarType> out_dtype, bool use_fast_accum, Tensor& out) {
  check_inputs(self, mat2, bias);
  const auto [batch_a, batch_b] = infer_scales(self, mat2, scale_a, scale_b);
  TORCH_CHECK(out.device() == self.device(), "out must be on the input device");
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  if (scale_result) {
    TORCH_CHECK(scale_result->device() == self.device() && scale_result->scalar_type() == kFloat &&
        (scale_result->numel() == 1 ||
         (scale_result->dim() == 1 && scale_result->size(0) == self.size(0))),
        "scale_result must be a Float scalar or have shape (B,) on the input device");
  }
  assert_no_internal_overlap(out);
  for (const auto& t : {self, mat2, scale_a, scale_b}) {
    assert_no_overlap(out, t);
  }
  if (bias) {
    assert_no_overlap(out, *bias);
  }
  if (scale_result) {
    assert_no_overlap(out, *scale_result);
  }
  resize_output(out, {self.size(0), self.size(1), mat2.size(2)});
  for (const auto i : c10::irange(self.size(0))) {
    auto output = out.select(0, i);
    auto bias_i = bias && bias->dim() == 2 ? std::optional<Tensor>(bias->select(0, i)) : bias;
    auto result_i = scale_result && scale_result->numel() != 1
        ? std::optional<Tensor>(scale_result->select(0, i)) : scale_result;
    output.copy_(at::_scaled_mm(self.select(0, i), mat2.select(0, i),
        batch_a ? scale_a.select(0, i) : scale_a,
        batch_b ? scale_b.select(0, i) : scale_b,
        bias_i, result_i, out.scalar_type(), use_fast_accum));
  }
  return out;
}

Tensor _scaled_bmm(const Tensor& self, const Tensor& mat2,
    const Tensor& scale_a, const Tensor& scale_b,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& scale_result,
    std::optional<ScalarType> out_dtype, bool use_fast_accum) {
  auto out = at::empty({0}, self.options().dtype(out_dtype.value_or(self.scalar_type())));
  return _scaled_bmm_out(self, mat2, scale_a, scale_b, bias, scale_result,
      out_dtype, use_fast_accum, out);
}

} // namespace at::native
