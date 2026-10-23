"""Compare native shared-scale CUDA batching with a loop over scaled-mm."""
import argparse

import torch
import torch.utils.benchmark as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    args = parser.parse_args()
    for size in (32, 128, 512):
        for batch in (1, 4, 16, 64):
            a = torch.randn(batch, size, size, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(batch, size, size, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
            scale = torch.ones((), device="cuda")

            def native():
                return torch._scaled_bmm(a, b, scale, scale, out_dtype=torch.bfloat16)

            def loop():
                return torch.stack([torch._scaled_mm(a[i], b[i], scale, scale, out_dtype=torch.bfloat16) for i in range(batch)])

            for name, fn in (("native", native), ("loop", loop)):
                result = benchmark.Timer(stmt="fn()", globals={"fn": fn}, label="scaled_bmm", sub_label=f"B={batch}, M=N=K={size}", description=name).blocked_autorange(min_run_time=args.min_run_time)
                print(result)


if __name__ == "__main__":
    main()
