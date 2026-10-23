#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/BlasBackend.h>
#include <c10/util/irange.h>
#include <ATen/ceil_div.h>
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
#include <ATen/ops/_scaled_bmm.h>
#include <ATen/ops/_scaled_bmm_v2_native.h>
#include <ATen/ops/_scaled_mm_v2.h>
#include <ATen/ops/empty_strided.h>
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
  TORCH_CHECK((isFloat8Type(a.scalar_type()) || a.scalar_type() == kFloat4_e2m1fn_x2) &&
      (isFloat8Type(b.scalar_type()) || b.scalar_type() == kFloat4_e2m1fn_x2),
      "_scaled_bmm inputs must be Float8 or Float4 tensors");
  if (bias) {
    TORCH_CHECK(bias->device() == a.device(), "bias must be on the input device");
    TORCH_CHECK((bias->dim() == 1 && bias->size(0) == b.size(2)) ||
        (bias->dim() == 2 && bias->size(0) == a.size(0) && bias->size(1) == b.size(2)),
        "bias must have shape (N,) or (B, N)");
  }
}

// The leading batch dimension is removed only after the per-matrix layout matches.
bool scale_matches(const Tensor& s, int64_t batch, bool batched,
    ScalingType recipe, int64_t outer, bool rhs, int64_t k,
    bool xpu, bool packed, int64_t swizzle = 0, bool explicit_recipe = false) {
  if (batched && (s.dim() == 0 || s.size(0) != batch)) {
    return false;
  }
  const auto sizes = s.sizes().slice(batched ? 1 : 0);
  int64_t numel = 1;
  for (const auto size : sizes) {
    numel *= size;
  }
  if (recipe == ScalingType::TensorWise) {
    return s.scalar_type() == kFloat && numel == 1;
  }
  if (recipe == ScalingType::RowWise) {
    return s.scalar_type() == kFloat &&
        ((sizes.size() == 2 && sizes[rhs ? 1 : 0] == outer && sizes[rhs ? 0 : 1] == 1) ||
         (explicit_recipe && sizes.size() == 1 && numel == outer));
  }
  if (recipe == ScalingType::BlockWise1x128 || recipe == ScalingType::BlockWise128x128) {
    const auto rows = recipe == ScalingType::BlockWise1x128 ? outer : ceil_div(outer, int64_t{128});
    return s.scalar_type() == kFloat && sizes.size() == 2 &&
        sizes[rhs ? 1 : 0] == rows && sizes[rhs ? 0 : 1] == ceil_div(k, int64_t{128});
  }
  const auto round = [](int64_t value, int64_t block) { return ceil_div(value, block) * block; };
  int64_t expected;
  ScalarType dtype;
  if (recipe == ScalingType::BlockWise1x32) {
    dtype = kFloat8_e8m0fnu;
    if (xpu) {
      expected = outer * ceil_div(k, int64_t{32});
    } else if (at::globalContext().hasROCM()) {
      expected = swizzle == 2 ? round(outer, 32) * round(ceil_div(k, int64_t{32}), 8)
          : ceil_div(outer * (packed ? 2 : 1), int64_t{32}) * (packed ? k / 2 : k);
    } else {
      expected = round(outer, 128) * round(ceil_div(k, int64_t{32}), 4);
    }
  } else if (recipe == ScalingType::BlockWise1x16) {
    dtype = kFloat8_e4m3fn;
    expected = xpu ? outer * ceil_div(k, int64_t{16})
        : round(outer, 128) * round(ceil_div(k, int64_t{16}), 4);
  } else {
    return false;
  }
  return s.scalar_type() == dtype && numel == expected;
}

std::pair<bool, bool> infer_scales(const Tensor& a, const Tensor& b,
    const Tensor& sa, const Tensor& sb) {
  TORCH_CHECK(sa.device() == a.device() && sb.device() == a.device(),
      "scales must be on the input device");
  std::optional<std::pair<bool, bool>> result;
  const bool packed = a.scalar_type() == kFloat4_e2m1fn_x2;
  const auto k = a.size(2) * (packed ? 2 : 1);
  const std::pair<ScalingType, ScalingType> recipes[] = {
      {ScalingType::TensorWise, ScalingType::TensorWise},
      {ScalingType::RowWise, ScalingType::RowWise},
      {ScalingType::BlockWise128x128, ScalingType::BlockWise1x128},
      {ScalingType::BlockWise1x128, ScalingType::BlockWise128x128},
      {ScalingType::BlockWise1x128, ScalingType::BlockWise1x128},
      {ScalingType::BlockWise1x32, ScalingType::BlockWise1x32},
      {ScalingType::BlockWise1x16, ScalingType::BlockWise1x16}};
  for (const auto& [recipe_a, recipe_b] : recipes) {
    for (const bool ba : {false, true}) {
      for (const bool bb : {false, true}) {
        if (scale_matches(sa, a.size(0), ba, recipe_a, a.size(1), false, k, a.is_xpu(), packed) &&
            scale_matches(sb, a.size(0), bb, recipe_b, b.size(2), true, k, a.is_xpu(), packed)) {
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

std::vector<bool> explicit_scale_modes(const Tensor& a, const Tensor& b,
    TensorList scales, IntArrayRef recipes, IntArrayRef swizzles, bool rhs) {
  TORCH_CHECK(!recipes.empty() && scales.size() == recipes.size(),
      "scale tensors must match the number of recipes");
  TORCH_CHECK(swizzles.empty() || swizzles.size() == recipes.size(),
      "swizzles must be empty or match the number of recipes");
  std::vector<bool> modes;
  const bool packed = a.scalar_type() == kFloat4_e2m1fn_x2;
  const auto k = a.size(2) * (packed ? 2 : 1);
  for (const auto i : c10::irange(scales.size())) {
    const auto& scale = scales[i];
    TORCH_CHECK(scale.device() == a.device(), "scales must be on the input device");
    const auto recipe = static_cast<ScalingType>(recipes[i]);
    const auto swizzle = swizzles.empty() ? 0 : swizzles[i];
    const auto outer = rhs ? b.size(2) : a.size(1);
    const bool shared = scale_matches(scale, a.size(0), false, recipe, outer, rhs,
        k, a.is_xpu(), packed, swizzle, true);
    const bool batched = scale_matches(scale, a.size(0), true, recipe, outer, rhs,
        k, a.is_xpu(), packed, swizzle, true);
    TORCH_CHECK(shared || batched, "Invalid scale shape or dtype for _scaled_bmm_v2 recipe ", recipes[i]);
    TORCH_CHECK(!shared || !batched || a.size(0) == 1,
        "Ambiguous scale shape for _scaled_bmm_v2; add singleton dimensions");
    modes.push_back(!shared);
  }
  return modes;
}

Tensor matrix_metadata(const Tensor& t, bool batched) {
  const auto offset = batched ? 1 : 0;
  return at::empty_strided(t.sizes().slice(offset), t.strides().slice(offset),
      t.options().device(kMeta));
}

std::vector<Tensor> scale_metadata(TensorList scales, const std::vector<bool>& modes) {
  std::vector<Tensor> result;
  for (const auto i : c10::irange(scales.size())) {
    result.push_back(matrix_metadata(scales[i], modes[i]));
  }
  return result;
}

std::vector<Tensor> batch_scales(TensorList scales, const std::vector<bool>& modes, int64_t i) {
  std::vector<Tensor> result;
  for (const auto j : c10::irange(scales.size())) {
    result.push_back(modes[j] ? scales[j].select(0, i) : scales[j]);
  }
  return result;
}

} // namespace

std::pair<bool, bool> prepare_scaled_bmm_output(const Tensor& self, const Tensor& mat2,
    const Tensor& scale_a, const Tensor& scale_b,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& scale_result,
    std::optional<ScalarType> out_dtype, Tensor& out) {
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
  return {batch_a, batch_b};
}

Tensor& _scaled_bmm_out(const Tensor& self, const Tensor& mat2,
    const Tensor& scale_a, const Tensor& scale_b,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& scale_result,
    std::optional<ScalarType> out_dtype, bool use_fast_accum, Tensor& out) {
  const auto [batch_a, batch_b] = prepare_scaled_bmm_output(
      self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out);
  for (const auto i : c10::irange(self.size(0))) {
    auto output = out.select(0, i);
    auto bias_i = bias && bias->dim() == 2 ? std::optional<Tensor>(bias->select(0, i)) : bias;
    auto result_i = scale_result && scale_result->numel() != 1
        ? std::optional<Tensor>(scale_result->select(0, i)) : scale_result;
    auto batch_out = output.is_contiguous() ? output
        : at::empty({self.size(1), mat2.size(2)}, out.options());
    at::_scaled_mm_out(batch_out, self.select(0, i), mat2.select(0, i),
        batch_a ? scale_a.select(0, i) : scale_a,
        batch_b ? scale_b.select(0, i) : scale_b,
        bias_i, result_i, out_dtype, use_fast_accum);
    if (!output.is_contiguous()) {
      output.copy_(batch_out);
    }
  }
  return out;
}

Tensor _scaled_bmm(const Tensor& self, const Tensor& mat2,
    const Tensor& scale_a, const Tensor& scale_b,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& scale_result,
    std::optional<ScalarType> out_dtype, bool use_fast_accum) {
  auto out = at::empty({0}, self.options().dtype(out_dtype.value_or(self.scalar_type())));
  return at::_scaled_bmm_out(out, self, mat2, scale_a, scale_b, bias, scale_result,
      out_dtype, use_fast_accum);
}

Tensor& _scaled_bmm_v2_out(const Tensor& self, const Tensor& mat2,
    TensorList scale_a, IntArrayRef recipe_a, IntArrayRef swizzle_a,
    TensorList scale_b, IntArrayRef recipe_b, IntArrayRef swizzle_b,
    const std::optional<Tensor>& bias, std::optional<ScalarType> out_dtype,
    IntArrayRef contraction_dim, bool use_fast_accum, Tensor& out) {
  check_inputs(self, mat2, bias);
  TORCH_CHECK(contraction_dim.empty() || (contraction_dim.size() == 2 &&
      (contraction_dim[0] == 2 || contraction_dim[0] == -1) &&
      (contraction_dim[1] == 1 || contraction_dim[1] == -2)),
      "_scaled_bmm_v2 only supports contraction_dim=(2, 1)");
  const auto modes_a = explicit_scale_modes(self, mat2, scale_a, recipe_a, swizzle_a, false);
  const auto modes_b = explicit_scale_modes(self, mat2, scale_b, recipe_b, swizzle_b, true);
  const auto recipes_a = scaled::convert_int_to_enum<ScalingType>(recipe_a);
  const auto recipes_b = scaled::convert_int_to_enum<ScalingType>(recipe_b);
  const auto swizzles_a = scaled::convert_int_to_enum<at::blas::SwizzleType>(swizzle_a);
  const auto swizzles_b = scaled::convert_int_to_enum<at::blas::SwizzleType>(swizzle_b);
  scaled::validate_scaled_mm_v2_inputs(matrix_metadata(self, true), matrix_metadata(mat2, true),
      scale_metadata(scale_a, modes_a), recipes_a, swizzles_a,
      scale_metadata(scale_b, modes_b), recipes_b, swizzles_b, self.device().type());
  TORCH_CHECK(out.device() == self.device(), "out must be on the input device");
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  assert_no_internal_overlap(out);
  assert_no_overlap(out, self);
  assert_no_overlap(out, mat2);
  for (const auto& scale : scale_a) {
    assert_no_overlap(out, scale);
  }
  for (const auto& scale : scale_b) {
    assert_no_overlap(out, scale);
  }
  if (bias) {
    assert_no_overlap(out, *bias);
  }
  resize_output(out, {self.size(0), self.size(1), mat2.size(2)});
  if (self.is_cuda() && recipe_a.size() == 1 && recipe_b.size() == 1 &&
      recipe_a[0] == 0 && recipe_b[0] == 0 && !modes_a[0] && !modes_b[0] &&
      !bias && isFloat8Type(self.scalar_type()) && isFloat8Type(mat2.scalar_type())) {
    return at::_scaled_bmm_out(out, self, mat2, scale_a[0], scale_b[0],
        std::nullopt, std::nullopt, out.scalar_type(), use_fast_accum);
  }
  for (const auto i : c10::irange(self.size(0))) {
    const auto bias_i = bias && bias->dim() == 2 ? std::optional<Tensor>(bias->select(0, i)) : bias;
    auto output = out.select(0, i);
    output.copy_(at::_scaled_mm_v2(self.select(0, i), mat2.select(0, i),
        batch_scales(scale_a, modes_a, i), recipe_a, swizzle_a,
        batch_scales(scale_b, modes_b, i), recipe_b, swizzle_b,
        bias_i, out.scalar_type(), {}, use_fast_accum));
  }
  return out;
}

Tensor _scaled_bmm_v2(const Tensor& self, const Tensor& mat2,
    TensorList scale_a, IntArrayRef recipe_a, IntArrayRef swizzle_a,
    TensorList scale_b, IntArrayRef recipe_b, IntArrayRef swizzle_b,
    const std::optional<Tensor>& bias, std::optional<ScalarType> out_dtype,
    IntArrayRef contraction_dim, bool use_fast_accum) {
  auto out = at::empty({0}, self.options().dtype(out_dtype.value_or(self.scalar_type())));
  return _scaled_bmm_v2_out(self, mat2, scale_a, recipe_a, swizzle_a,
      scale_b, recipe_b, swizzle_b, bias, out_dtype, contraction_dim, use_fast_accum, out);
}

} // namespace at::native
