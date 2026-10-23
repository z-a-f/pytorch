# Owner(s): ["module: linear algebra"]
import itertools
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_device_type import e4m3_type, instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, instantiate_parametrized_tests, parametrize, run_tests, TestCase


class TestScaledBmm(TestCase):
    hw_classification = HardwareClassification.CPU

    def inputs(self, device, batch=3, m=32, k=32, n=48):
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
            actual = torch._scaled_bmm(a, b, sa, sb, out_dtype=torch.bfloat16)
            expected = torch.empty(batch, 32, 48, device=device, dtype=torch.bfloat16)
            for i in range(batch):
                expected[i] = torch._scaled_mm(a[i], b[i], sa[i] if ba else sa, sb[i] if bb else sb, out_dtype=torch.bfloat16)
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
        result = torch._scaled_bmm(a, b, scale, scale, bias=bias, out=out)
        self.assertIs(result, out)
        expected = torch.stack([torch._scaled_mm(a[i], b[i], scale, scale, bias=bias[i] if per_batch else bias, out_dtype=torch.bfloat16) for i in range(3)])
        self.assertEqual(result, expected)
        resized = torch.empty(0, device=device, dtype=torch.bfloat16)
        torch._scaled_bmm(a, b, scale, scale, out=resized)
        self.assertEqual(resized.shape, (3, 32, 48))
        with self.assertRaisesRegex(RuntimeError, "out_dtype must match"):
            torch._scaled_bmm(a, b, scale, scale, out=resized, out_dtype=torch.float16)

    @parametrize("shape", [(3, 0, 32, 48), (3, 32, 0, 48), (3, 32, 32, 0)])
    def test_empty(self, device, shape):
        batch, m, k, n = shape
        a, b = self.inputs(device, batch, m, k, n)
        scale = torch.ones((), device=device)
        result = torch._scaled_bmm(a, b, scale, scale, out_dtype=torch.bfloat16)
        self.assertEqual(result, torch.zeros(batch, m, n, device=device, dtype=torch.bfloat16))

    def test_invalid(self, device):
        a, b = self.inputs(device)
        scale = torch.ones((), device=device)
        cases = [
            (a[0], b, scale, scale, "must be 3D"),
            (a, b[:1], scale, scale, "same batch size"),
            (a, b[:, :16], scale, scale, "contraction dimensions"),
            (a.float(), b, scale, scale, "Float8"),
            (a, b, torch.ones(2, device=device), scale, "Invalid scaling"),
        ]
        for aa, bb, sa, sb, error in cases:
            with self.assertRaisesRegex(RuntimeError, error):
                torch._scaled_bmm(aa, bb, sa, sb)
        with self.assertRaisesRegex(RuntimeError, "bias must have shape"):
            torch._scaled_bmm(a, b, scale, scale, bias=torch.zeros(2, device=device))

    def test_ambiguous(self, device):
        a, b = self.inputs(device, batch=32, m=32, n=32)
        sa = torch.ones(32, 1, device=device)
        sb = torch.ones(32, 1, 32, device=device)
        # Rowwise A can also be interpreted as per-batch scalar A when B=M.
        # Both sides must form a valid pair before it is ambiguous.
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            torch._scaled_bmm(a, b[:, :, :1], sa, torch.ones(1, 1, device=device))
        actual = torch._scaled_bmm(a, b, sa.unsqueeze(0).expand(32, -1, -1), sb, out_dtype=torch.bfloat16)
        self.assertEqual(actual.shape, (32, 32, 32))


@unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 hardware is required")
class TestScaledBmmCUDA(TestScaledBmm):
    hw_classification = HardwareClassification.CUDA


class TestScaledBmmXPU(TestScaledBmm):
    hw_classification = HardwareClassification.XPU


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


instantiate_device_type_tests(TestScaledBmm, globals(), only_for=("cpu",))
instantiate_device_type_tests(TestScaledBmmCUDA, globals(), only_for=("cuda",))
instantiate_device_type_tests(TestScaledBmmXPU, globals(), only_for=("xpu",))

if __name__ == "__main__":
    run_tests()
