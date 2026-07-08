import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench_breakdown import (  # noqa: E402
    PaddedAttentionWrapper,
    VarlenAttentionWrapper,
    benchmark_forward_with_warmup,
    parse_int_list,
    scaled_head_budgets,
)


def group_budgets(head_budgets: list[int], group_size: int) -> list[int]:
    return [
        max(head_budgets[i : i + group_size])
        for i in range(0, len(head_budgets), group_size)
    ]


@torch.inference_mode()
def bench_gqa(
    batch_sizes: list[int],
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    avg_budget: int,
    budget_profile: str,
    warmups: int,
    repeats: int,
) -> None:
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    page_size = 1
    max_num_pages = 32768
    dtype = torch.float16
    device = "cuda:0"
    group_size = num_heads // num_kv_heads
    head_budgets = scaled_head_budgets(num_heads, avg_budget, budget_profile)
    gqa_budgets = group_budgets(head_budgets, group_size)

    flat_kv_cache = torch.randn(
        max_num_pages,
        2,
        page_size,
        1,
        head_size,
        dtype=dtype,
        device=device,
    )
    padded_kv_cache = torch.randn(
        max_num_pages,
        2,
        page_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    for batch_size in batch_sizes:
        flat_lens = head_budgets * batch_size
        gqa_lens = gqa_budgets * batch_size

        per_head = VarlenAttentionWrapper(
            flat_lens,
            flat_kv_cache,
            num_qo_heads=1,
        )
        gqa = VarlenAttentionWrapper(
            gqa_lens,
            flat_kv_cache,
            num_qo_heads=group_size,
        )
        padded = PaddedAttentionWrapper(
            batch_size=batch_size,
            seq_len=max(head_budgets),
            kv_cache=padded_kv_cache,
            num_qo_heads=num_heads,
        )

        print(
            f"batch_size: {batch_size}, avg_budget: {sum(head_budgets) / len(head_budgets):.2f}, "
            f"max_budget: {max(head_budgets)}, group_size: {group_size}"
        )
        _, timing = benchmark_forward_with_warmup(
            per_head.forward, warmups=warmups, repeats=repeats
        )
        print(f"[per-head flattened] time: {timing.mean * 1e6:.2f} us")

        _, timing = benchmark_forward_with_warmup(
            gqa.forward, warmups=warmups, repeats=repeats
        )
        print(f"[gqa grouped] time: {timing.mean * 1e6:.2f} us")

        _, timing = benchmark_forward_with_warmup(
            padded.forward, warmups=warmups, repeats=repeats
        )
        print(f"[padded] time: {timing.mean * 1e6:.2f} us", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark per-head and GQA ragged FlashInfer attention."
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("16,32,64"),
    )
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--avg-budget", type=int, default=256)
    parser.add_argument(
        "--budget-profile",
        choices=("layer4", "uniform"),
        default="layer4",
    )
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()

    bench_gqa(
        batch_sizes=args.batch_sizes,
        head_size=args.head_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        avg_budget=args.avg_budget,
        budget_profile=args.budget_profile,
        warmups=args.warmups,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
