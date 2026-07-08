import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench_utils import benchmark_forward_with_warmup
from twilight.kernel import (
    top_p_fp32_return_indices,
    top_p_fp32_return_indices_out,
    top_p_fp32_return_mask,
)
from twilight.pyimpl.top_p import top_p_unnormalized


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def random_logits(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    seed: int = 42,
) -> torch.Tensor:
    # Shape: [batch_size, num_heads, 1, seq_len].
    torch.manual_seed(seed)

    logits = torch.empty(
        (batch_size, num_heads, 1, seq_len),
        dtype=torch.float16,
        device="cuda",
    )
    logits[:, :, :, 0].uniform_(1.5, 2.2)
    logits[:, :, :, 1:].uniform_(-9.0, -1.0)
    return logits


@torch.inference_mode()
def bench_top_p(
    batch_sizes: list[int],
    seq_lens: list[int],
    num_heads: int,
    p: float,
    warmups: int,
    repeats: int,
) -> None:
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            logits = random_logits(batch_size, num_heads, seq_len)

            _, timing = benchmark_forward_with_warmup(
                top_p_unnormalized,
                logits,
                p,
                warmups=warmups,
                repeats=repeats,
            )

            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            mask = top_p_fp32_return_mask(probs, p)
            counts = mask.sum(dim=-1).to(torch.int32)
            avg_budget = mask.sum(dim=-1).float().mean().item()
            max_budget = max(1, int(counts.max().item()))

            indices, output_counts = top_p_fp32_return_indices(
                probs, p, max_output_len=max_budget
            )
            assert torch.equal(output_counts, counts)

            output_indices = torch.empty_like(indices)
            prealloc_counts = torch.empty_like(output_counts)

            _, mask_kernel_timing = benchmark_forward_with_warmup(
                top_p_fp32_return_mask,
                probs,
                p,
                warmups=warmups,
                repeats=repeats,
            )
            _, indices_timing = benchmark_forward_with_warmup(
                top_p_fp32_return_indices,
                probs,
                p,
                None,
                max_budget,
                warmups=warmups,
                repeats=repeats,
            )
            _, indices_out_timing = benchmark_forward_with_warmup(
                top_p_fp32_return_indices_out,
                probs,
                output_indices,
                prealloc_counts,
                p,
                warmups=warmups,
                repeats=repeats,
            )

            print(
                f"[top-p] batch_size: {batch_size}, seq_len: {seq_len}, "
                f"p: {p}, avg_budget: {avg_budget:.2f}, "
                f"max_budget: {max_budget}, "
                f"mask+softmax: {timing.mean * 1e6:.2f} us, "
                f"mask_kernel: {mask_kernel_timing.mean * 1e6:.2f} us, "
                f"indices_cap: {indices_timing.mean * 1e6:.2f} us, "
                f"indices_out: {indices_out_timing.mean * 1e6:.2f} us",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Twilight top-p pruning.")
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("4,8,16,32"),
    )
    parser.add_argument(
        "--seq-lens",
        type=parse_int_list,
        default=parse_int_list("4096,8192,16384,32768"),
    )
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--p", type=float, default=0.85)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3000)
    args = parser.parse_args()

    bench_top_p(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        num_heads=args.num_heads,
        p=args.p,
        warmups=args.warmups,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
