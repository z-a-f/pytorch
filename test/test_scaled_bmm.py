# Owner(s): ["module: linear algebra"]
import itertools
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_device_type import e4m3_type, instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, instantiate_parametrized_tests, parametrize, run_tests, TestCase


class TestScaledBmm(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def scaled_bmm(self, *args, **kwargs):
        return torch._scaled_bmm(*args, **kwargs)

    def scaled_mm(self, *args, **kwargs):
        return torch._scaled_mm(*args, **kwargs)

    def inputs(self, device, batch=3, m=32, k=32, n=48):
        if device.startswith("cuda") and not PLATFORM_SUPPORTS_FP8:
            self.skipTest("FP8 hardware is required")
        a = torch.randn(batch, m, k, device=device).to(e4m3_type)
        b = torch.randn(batch, n, k, device=device).to(e4m3_type).transpose(1, 2)
        return a, b

    @parametrize("batch", [0, 1, 3])
    @parametrize("rowwise", [False, True])
    def test_scales(self, device, batch, rowwise):
        if rowwise and device.startswith("cuda") and not torch.version.hip and torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("Rowwise FP8 requires SM90 or later")
        a, b = self.inputs(device, batch)
        for ba, bb in itertools.product((False, True), repeat=2):
            sa_shape = (32, 1) if rowwise else ()
            sb_shape = (1, 48) if rowwise else ()
            sa = torch.rand(((batch,) if ba else ()) + sa_shape, device=device) + 0.5
            sb = torch.rand(((batch,) if bb else ()) + sb_shape, device=device) + 0.5
            actual = self.scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16)
            expected = torch.empty(batch, 32, 48, device=device, dtype=torch.bfloat16)
            for i in range(batch):
                expected[i] = self.scaled_mm(a[i], b[i], sa[i] if ba else sa, sb[i] if bb else sb, out_dtype=torch.bfloat16)
            self.assertEqual(actual, expected)
            if not rowwise:
                da = a.float() * (sa.reshape(batch, 1, 1) if ba else sa)
                db = b.float() * (sb.reshape(batch, 1, 1) if bb else sb)
                self.assertEqual(actual.float(), torch.bmm(da, db), atol=0.15, rtol=0.03)

    @parametrize("per_batch", [False, True])
    def test_bias_and_out(self, device, per_batch):
        a, b = self.inputs(device)
        scale = torch.ones((), device=device)
        bias = torch.randn((3, 48) if per_batch else (48,), device=device, dtype=torch.bfloat16)
        out = torch.empty(3, 48, 32, device=device, dtype=torch.bfloat16).transpose(1, 2)
        result = self.scaled_bmm(a, b, scale, scale, bias=bias, out=out)
        self.assertIs(result, out)
        expected = torch.stack([self.scaled_mm(a[i], b[i], scale, scale, bias=bias[i] if per_batch else bias, out_dtype=torch.bfloat16) for i in range(3)])
        self.assertEqual(result, expected)
        resized = torch.empty(0, device=device, dtype=torch.bfloat16)
        self.scaled_bmm(a, b, scale, scale, out=resized)
        self.assertEqual(resized.shape, (3, 32, 48))
        with self.assertRaisesRegex(RuntimeError, "out_dtype must match"):
            self.scaled_bmm(a, b, scale, scale, out=resized, out_dtype=torch.float16)

    @parametrize("shape", [(3, 0, 32, 48), (3, 32, 0, 48), (3, 32, 32, 0)])
    def test_empty(self, device, shape):
        batch, m, k, n = shape
        a, b = self.inputs(device, batch, m, k, n)
        scale = torch.ones((), device=device)
        result = self.scaled_bmm(a, b, scale, scale, out_dtype=torch.bfloat16)
        self.assertEqual(result, torch.zeros(batch, m, n, device=device, dtype=torch.bfloat16))

    def test_invalid(self, device):
        a, b = self.inputs(device)
        scale = torch.ones((), device=device)
        cases = [
            (a[0], b, scale, scale, "must be 3D"),
            (a, b[:1], scale, scale, "same batch size"),
            (a, b[:, :16], scale, scale, "contraction dimensions"),
            (a.float(), b, scale, scale, "Float8"),
            (a, b, torch.ones(2, device=device), scale, "Invalid scal"),
        ]
        for aa, bb, sa, sb, error in cases:
            with self.assertRaisesRegex(RuntimeError, error):
                self.scaled_bmm(aa, bb, sa, sb)
        with self.assertRaisesRegex(RuntimeError, "bias must have shape"):
            self.scaled_bmm(a, b, scale, scale, bias=torch.zeros(2, device=device))

    @parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("fast", [False, True])
    def test_output_dtypes_and_strides(self, device, out_dtype, fast):
        a, b = self.inputs(device, batch=6, m=48)
        a = a[::2, :32, :]
        b = b[::2]
        scale = torch.ones((), device=device)
        result = self.scaled_bmm(a, b, scale, scale, out_dtype=out_dtype, use_fast_accum=fast)
        expected = torch.stack([self.scaled_mm(a[i], b[i], scale, scale, out_dtype=out_dtype, use_fast_accum=fast) for i in range(3)])
        self.assertEqual(result, expected)
        self.assertEqual(result.dtype, out_dtype)
        with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
            self.scaled_bmm(a, b, scale, scale, out=a)

    @parametrize("per_batch", [False, True])
    def test_result_scale(self, device, per_batch):
        a, b = self.inputs(device)
        scale = torch.ones((), device=device)
        result_scale = torch.arange(1, 4, device=device, dtype=torch.float32) if per_batch else scale
        result = torch._scaled_bmm(a, b, scale, scale, scale_result=result_scale)
        expected = torch.stack([self.scaled_mm(a[i], b[i], scale, scale, scale_result=result_scale[i] if per_batch else result_scale) for i in range(3)])
        self.assertEqual(result, expected)

    def test_ambiguous(self, device):
        if device.startswith("cuda") and not torch.version.hip and torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("Rowwise FP8 requires SM90 or later")
        a, b = self.inputs(device, batch=32, m=32, n=32)
        sa = torch.ones(32, 1, device=device)
        sb = torch.ones(32, 1, 32, device=device)
        # Rowwise A can also be interpreted as per-batch scalar A when B=M.
        # Both sides must form a valid pair before it is ambiguous.
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            self.scaled_bmm(a, b[:, :, :1], sa, torch.ones(1, 1, device=device))
        actual = self.scaled_bmm(a, b, sa.unsqueeze(0).expand(32, -1, -1), sb, out_dtype=torch.bfloat16)
        self.assertEqual(actual.shape, (32, 32, 32))


class TestScaledBmmV2(TestScaledBmm):
    hw_classification = HardwareClassification.ACCELERATOR
    test_scales = TestScaledBmm.test_scales
    test_bias_and_out = TestScaledBmm.test_bias_and_out
    test_empty = TestScaledBmm.test_empty
    test_invalid = TestScaledBmm.test_invalid
    test_output_dtypes_and_strides = TestScaledBmm.test_output_dtypes_and_strides

    def scaled_bmm(self, a, b, sa, sb, bias=None, out_dtype=None, use_fast_accum=False, **kwargs):
        rowwise = sa.dim() >= 2 and sa.shape[-2:] == (a.size(1), 1) and sa.numel() != 1
        recipe = [1 if rowwise else 0]
        return torch._scaled_bmm_v2(a, b, [sa], recipe, [], [sb], recipe, [], bias, out_dtype, use_fast_accum=use_fast_accum, **kwargs)

    def scaled_mm(self, a, b, sa, sb, bias=None, out_dtype=None, use_fast_accum=False):
        rowwise = sa.dim() == 2 and sa.shape == (a.size(0), 1) and sa.numel() != 1
        recipe = [1 if rowwise else 0]
        return torch._scaled_mm_v2(a, b, [sa], recipe, [], [sb], recipe, [], bias, out_dtype, use_fast_accum=use_fast_accum)

    def test_ambiguous(self, device):
        if device.startswith("cuda") and not torch.version.hip and torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("Rowwise FP8 requires SM90 or later")
        a, b = self.inputs(device, batch=32, m=32, n=32)
        sa = torch.ones(32, 1, device=device)
        scalar = torch.ones((), device=device)
        actual = torch._scaled_bmm_v2(a, b, [sa], [0], [], [scalar], [0], [], None, torch.bfloat16)
        expected = torch.stack([self.scaled_mm(a[i], b[i], sa[i], scalar, out_dtype=torch.bfloat16) for i in range(32)])
        self.assertEqual(actual, expected)
        actual = torch._scaled_bmm_v2(a, b, [sa], [1], [], [sa.t()], [1], [], None, torch.bfloat16)
        expected = torch.stack([self.scaled_mm(a[i], b[i], sa, sa.t(), out_dtype=torch.bfloat16) for i in range(32)])
        self.assertEqual(actual, expected)

    @parametrize("contraction_dim", [(), (2, 1), (-1, -2)])
    def test_functional(self, device, contraction_dim):
        a, b = self.inputs(device)
        sa = torch.rand(3, device=device)
        sb = torch.ones((), device=device)
        result = torch.nn.functional.scaled_bmm(a, b, sa, torch.nn.functional.ScalingType.TensorWise, sb, torch.nn.functional.ScalingType.TensorWise, contraction_dim=contraction_dim)
        self.assertEqual(result, self.scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "contraction_dim"):
            torch._scaled_bmm_v2(a, b, [sa], [0], [], [sb], [0], [], None, torch.bfloat16, [1, 0])

    def test_invalid_recipes(self, device):
        a, b = self.inputs(device, batch=0)
        scalar = torch.ones((), device=device)
        for recipes, scales, error in [([], [], "number of recipes"), ([99], [scalar], "Invalid scale"), ([0, 0], [scalar], "number of recipes")]:
            with self.assertRaisesRegex(RuntimeError, error):
                torch._scaled_bmm_v2(a, b, scales, recipes, [], [scalar], [0], [], None, torch.bfloat16)


@instantiate_parametrized_tests
class TestScaledBmmMeta(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("fake", [False, True])
    def test_meta(self, fake):
        with FakeTensorMode() if fake else torch.device("meta"):
            a = torch.empty(3, 32, 32, dtype=torch.float8_e4m3fn)
            b = torch.empty(3, 48, 32, dtype=torch.float8_e4m3fn).transpose(1, 2)
            sa = torch.ones(3)
            sb = torch.ones(())
            result = torch._scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16)
            self.assertEqual(result.shape, (3, 32, 48))
            self.assertEqual(result.dtype, torch.bfloat16)


    @parametrize("dtype", [torch.float8_e8m0fnu, torch.float8_e4m3fn])
    def test_block_scales(self, dtype):
        with torch.device("meta"):
            input_dtype = torch.float8_e4m3fn if dtype == torch.float8_e8m0fnu else torch.float4_e2m1fn_x2
            a = torch.empty(3, 32, 32, dtype=input_dtype)
            b = torch.empty(3, 48, 32, dtype=input_dtype).transpose(1, 2)
            scales = torch.empty(3, 512, dtype=dtype)
            result = torch._scaled_bmm(a, b, scales, scales, out_dtype=torch.bfloat16)
            self.assertEqual(result.shape, (3, 32, 48))


    @parametrize("fake", [False, True])
    def test_v2_meta(self, fake):
        with FakeTensorMode() if fake else torch.device("meta"):
            a = torch.empty(3, 32, 32, dtype=torch.float8_e4m3fn)
            b = torch.empty(3, 48, 32, dtype=torch.float8_e4m3fn).transpose(1, 2)
            sa = torch.ones(3)
            sb = torch.ones(())
            result = torch._scaled_bmm_v2(a, b, [sa], [0], [], [sb], [0], [], None, torch.bfloat16)
            self.assertEqual(result.shape, (3, 32, 48))
            self.assertEqual(result.dtype, torch.bfloat16)


    @parametrize("recipe", [(3,), (2,), (2, 0), (4,), (5,)])
    @parametrize("batch", [0, 3])
    def test_v2_block_meta(self, recipe, batch):
        with torch.device("meta"):
            packed = recipe[0] == 2
            a = torch.empty(batch, 128, 64 if packed else 128, dtype=torch.float4_e2m1fn_x2 if packed else torch.float8_e4m3fn)
            b = torch.empty(batch, 128, 64 if packed else 128, dtype=a.dtype).transpose(1, 2)
            rb = [4] if recipe == (5,) else list(recipe)
            if recipe[0] in (4, 5):
                sa = torch.empty(batch, 1 if recipe[0] == 5 else 128, 1)
                sb = torch.empty(1, 128)
                swizzle = []
            else:
                sa = torch.empty(batch, 1024 if packed else 512, dtype=torch.float8_e4m3fn if packed else torch.float8_e8m0fnu)
                sb = torch.empty(1024 if packed else 512, dtype=sa.dtype)
                swizzle = [0 if torch.version.hip else 1] * len(recipe)
            scales_a, scales_b = [sa], [sb]
            if len(recipe) == 2:
                scales_a.append(torch.empty(()))
                scales_b.append(torch.empty(batch))
            result = torch._scaled_bmm_v2(a, b, scales_a, list(recipe), swizzle, scales_b, rb, swizzle, None, torch.bfloat16)
            self.assertEqual(result.shape, (batch, 128, 128))
            with self.assertRaisesRegex(RuntimeError, "Invalid scale"):
                torch._scaled_bmm_v2(a, b, [torch.empty(7), *scales_a[1:]], list(recipe), swizzle, scales_b, rb, swizzle, None, torch.bfloat16)



class TestScaledBmmCompile(TestCase):
    hw_classification = HardwareClassification.CPU

    @parametrize("v2", [False, True])
    def test_compile_dynamic(self, device, v2):
        from torch._dynamo.testing import CompileCounterWithBackend

        def fn(a, b, sa, sb):
            if v2:
                return torch._scaled_bmm_v2(a, b, [sa], [0], [], [sb], [0], [], None, torch.bfloat16)
            return torch._scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16)

        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=True)
        for batch, m, k, n in ((2, 16, 32, 48), (4, 48, 64, 80)):
            a = torch.randn(batch, m, k, device=device).to(torch.float8_e4m3fn)
            b = torch.randn(batch, n, k, device=device).to(torch.float8_e4m3fn).transpose(1, 2)
            sa = torch.rand(batch, device=device)
            sb = torch.ones((), device=device)
            self.assertEqual(compiled(a, b, sa, sb), fn(a, b, sa, sb))
        self.assertEqual(counter.frame_count, 1)

    @parametrize("v2", [False, True])
    def test_export(self, device, v2):
        class ScaledBmm(torch.nn.Module):
            def forward(self, a, b, sa, sb):
                if v2:
                    return torch._scaled_bmm_v2(a, b, [sa], [0], [], [sb], [0], [], None, torch.bfloat16)
                return torch._scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16)

        a = torch.randn(3, 32, 32, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(3, 48, 32, device=device).to(torch.float8_e4m3fn).transpose(1, 2)
        sa = torch.rand(3, device=device)
        sb = torch.ones((), device=device)
        batch = torch.export.Dim("batch", min=2, max=8)
        exported = torch.export.export(ScaledBmm(), (a, b, sa, sb), dynamic_shapes=({0: batch}, {0: batch}, {0: batch}, {}))
        args = (a[:2], b[:2], sa[:2], sb)
        self.assertEqual(exported.module()(*args), ScaledBmm()(*args))


@unittest.skipIf(not PLATFORM_SUPPORTS_FP8 or bool(torch.version.hip), "CUDA FP8 hardware is required")
class TestScaledBmmFastPath(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("v2", [False, True])
    @parametrize("fast", [False, True])
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_batched_launch(self, device, v2, fast, dtype):
        a = torch.randn(4, 128, 128, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(4, 128, 128, device=device).to(torch.float8_e4m3fn).transpose(1, 2)
        scale = torch.ones((), device=device)
        expected = torch.stack([torch._scaled_mm(a[i], b[i], scale, scale, out_dtype=dtype, use_fast_accum=fast) for i in range(4)])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            if v2:
                actual = torch._scaled_bmm_v2(a, b, [scale], [0], [], [scale], [0], [], None, dtype, use_fast_accum=fast)
            else:
                actual = torch._scaled_bmm(a, b, scale, scale, out_dtype=dtype, use_fast_accum=fast)
        self.assertEqual(actual, expected)
        if any(event.key == "aten::_scaled_mm" for event in prof.key_averages()):
            self.skipTest("cuBLASLt has no native batched algorithm for this configuration")
        self.assertEqual(sum(event.count for event in prof.key_averages() if event.key == "aten::_scaled_bmm"), 1 if not v2 else 0)

    def test_per_batch_fallback(self, device):
        a = torch.randn(4, 32, 32, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(4, 48, 32, device=device).to(torch.float8_e4m3fn).transpose(1, 2)
        sa = torch.arange(1, 5, device=device, dtype=torch.float32)
        sb = torch.ones((), device=device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            actual = torch._scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16)
        expected = torch.stack([torch._scaled_mm(a[i], b[i], sa[i], sb, out_dtype=torch.bfloat16) for i in range(4)])
        self.assertEqual(actual, expected)
        self.assertEqual(sum(event.count for event in prof.key_averages() if event.key == "aten::_scaled_mm"), 4)


@unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 hardware is required")
class TestScaledBmmRecipes(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("recipe", ["mxfp8", "mxfp4", "nvfp4", "nvfp4_tw", "deepseek_1d", "deepseek_a2d", "deepseek_b2d"])
    @parametrize("batched", [False, True])
    def test_recipes(self, device, recipe, batched):
        major = torch.cuda.get_device_capability()[0]
        if recipe.startswith("deepseek"):
            if torch.version.hip or major != 9:
                self.skipTest("DeepSeek block scaling requires SM90")
        elif torch.version.hip:
            from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_MX_GEMM
            if recipe.startswith("nv") or not PLATFORM_SUPPORTS_MX_GEMM:
                self.skipTest("Recipe is unsupported on this ROCm device")
        elif major < 10:
            self.skipTest("Block scaling requires SM100 or later")
        batch, m, k, n = 3, 128, 128, 128
        packed = recipe in ("mxfp4", "nvfp4", "nvfp4_tw")
        if packed:
            a = torch.full((batch, m, k // 2), 0x22, device=device, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            b = torch.full((batch, n, k // 2), 0x22, device=device, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).transpose(1, 2)
        else:
            a = torch.randn(batch, m, k, device=device).to(e4m3_type)
            b = torch.randn(batch, n, k, device=device).to(e4m3_type).transpose(1, 2)
        if recipe.startswith("deepseek"):
            ra = 5 if recipe == "deepseek_a2d" else 4
            rb = 5 if recipe == "deepseek_b2d" else 4
            sa = torch.ones((1 if ra == 5 else m, 1), device=device)
            sb = torch.ones((1, 1 if rb == 5 else n), device=device)
            swizzle = []
        else:
            ra = rb = 2 if recipe.startswith("nv") else 3
            count = 1024 if ra == 2 else 512
            sa = torch.ones(count, device=device).to(torch.float8_e4m3fn if ra == 2 else torch.float8_e8m0fnu)
            sb = sa.clone()
            swizzle = [0 if torch.version.hip else 1]
        if batched:
            sa = torch.stack([(sa.float() * (2 ** i)).to(sa.dtype) for i in range(batch)])
            sb = torch.stack([(sb.float() * (2 ** i)).to(sb.dtype) for i in range(batch)])
        recipes = ([ra, 0], [rb, 0]) if recipe == "nvfp4_tw" else ([ra], [rb])
        scales_a = [sa]
        scales_b = [sb]
        if recipe == "nvfp4_tw":
            scales_a.append(torch.tensor(0.5, device=device))
            scales_b.append(torch.arange(1, batch + 1, device=device, dtype=torch.float32))
            swizzle = [1, 0]
        actual = torch._scaled_bmm_v2(a, b, scales_a, recipes[0], swizzle, scales_b, recipes[1], swizzle, None, torch.bfloat16)
        expected = []
        for i in range(batch):
            ai = [sa[i] if batched else sa]
            bi = [sb[i] if batched else sb]
            if recipe == "nvfp4_tw":
                ai.append(scales_a[1])
                bi.append(scales_b[1][i])
            expected.append(torch._scaled_mm_v2(a[i], b[i], ai, recipes[0], swizzle, bi, recipes[1], swizzle, None, torch.bfloat16))
        self.assertEqual(actual, torch.stack(expected))
        if recipe != "nvfp4_tw":
            legacy = torch._scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16)
            self.assertEqual(legacy, actual)


instantiate_device_type_tests(TestScaledBmm, globals(), except_for=("mps",), allow_xpu=True)
instantiate_device_type_tests(TestScaledBmmV2, globals(), allow_mps=True, allow_xpu=True)

instantiate_device_type_tests(TestScaledBmmCompile, globals(), only_for=("cpu",))

instantiate_device_type_tests(TestScaledBmmFastPath, globals(), only_for=("cuda",))

instantiate_device_type_tests(TestScaledBmmRecipes, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
