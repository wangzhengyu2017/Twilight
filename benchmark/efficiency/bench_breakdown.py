import argparse
import sys
from pathlib import Path
from typing import Callable

import flashinfer
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FTKA_ROOT = REPO_ROOT / "flash-topk-attention"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FTKA_ROOT))

from bench_utils import benchmark_forward_with_warmup
from ftka.cuda_ops.gemv import batched_sparse_gemv_int4_k, quest_sparse_gemv
from ftka.cuda_ops.topk import raft_topk
from twilight.pyimpl.top_p import (
    top_p_unnormalized,
    top_p_unnormalized_return_indices,
    top_p_unnormalized_return_indices_out,
)

LAYER4_HEAD_BUDGETS = [
    115,
    4,
    162,
    178,
    416,
    444,
    562,
    521,
    254,
    270,
    255,
    228,
    44,
    65,
    14,
    55,
    566,
    231,
    298,
    701,
    179,
    383,
    628,
    793,
    369,
    102,
    290,
    348,
    2719,
    2778,
    2758,
    1700,
]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def scaled_head_budgets(num_heads: int, avg_budget: int, profile: str) -> list[int]:
    if profile == "uniform":
        return [avg_budget] * num_heads
    if profile != "layer4":
        raise ValueError(f"Unknown budget profile: {profile}")

    if num_heads != len(LAYER4_HEAD_BUDGETS):
        raise ValueError("The layer4 budget profile assumes 32 query heads.")

    mean_budget = sum(LAYER4_HEAD_BUDGETS) / len(LAYER4_HEAD_BUDGETS)
    scale = avg_budget / mean_budget
    return [max(1, int(round(budget * scale))) for budget in LAYER4_HEAD_BUDGETS]


def get_indices(batch_size: int, seq_len: int, device: str):
    kv_page_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
    kv_page_indices = kv_page_indices.repeat(batch_size)
    kv_page_indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
    return kv_page_indices, kv_page_indptr, kv_last_page_len


class SpGEMVINT4Wrapper:
    def __init__(self, batch_size: int, seq_len: int, k_cache_int8: torch.Tensor):
        self.dtype = torch.float16
        self.device = k_cache_int8.device
        self.batch_size = batch_size
        self.max_num_pages = k_cache_int8.shape[0]
        self.page_size = k_cache_int8.shape[1]
        self.num_heads = k_cache_int8.shape[2]
        self.head_size = k_cache_int8.shape[3] * 2
        self.k_cache_int8 = k_cache_int8
        self.q = torch.rand(
            (self.batch_size, self.num_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.o = torch.empty(
            (self.batch_size, self.num_heads, seq_len),
            dtype=self.dtype,
            device=self.device,
        )
        self.quant_scales = torch.randn(
            self.max_num_pages,
            self.page_size,
            self.num_heads,
            dtype=self.dtype,
            device=self.device,
        )
        self.quant_zeros = torch.randn(
            self.max_num_pages,
            self.page_size,
            self.num_heads,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_page_indices, self.kv_page_indptr, self.kv_last_page_len = get_indices(
            batch_size, seq_len, str(self.device)
        )

    def forward(self) -> None:
        batched_sparse_gemv_int4_k(
            self.q,
            self.o,
            self.k_cache_int8,
            self.quant_scales,
            self.quant_zeros,
            self.kv_page_indices,
            self.kv_page_indptr,
            self.kv_last_page_len,
        )


class QuestSelectorWrapper:
    def __init__(self, batch_size: int, seq_len: int, k_label: torch.Tensor):
        self.dtype = torch.float16
        self.device = k_label.device
        self.batch_size = batch_size
        self.max_num_pages = k_label.shape[0]
        assert k_label.shape[1] == 1
        self.num_heads = k_label.shape[2]
        self.head_size = k_label.shape[3]
        self.k_label = k_label
        self.q = torch.rand(
            (self.batch_size, self.num_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.o = torch.empty(
            (self.batch_size, self.num_heads, seq_len),
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_page_indices, self.kv_page_indptr, self.kv_last_page_len = get_indices(
            batch_size, seq_len, str(self.device)
        )

    def forward(self) -> None:
        quest_sparse_gemv(
            self.q,
            self.o,
            self.k_label,
            self.k_label,
            self.kv_page_indices,
            self.kv_page_indptr,
            self.kv_last_page_len,
        )


class TopKWrapper:
    def __init__(self, batch_size: int, seq_len: int, topk: int, device: str):
        self.dtype = torch.float16
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.topk = topk
        self.input_data = torch.randn(
            batch_size, seq_len, dtype=self.dtype, device=self.device
        )
        self.input_indices = torch.arange(
            0, self.seq_len, dtype=torch.int32, device=self.device
        ).repeat(self.batch_size, 1)
        self.output_data = torch.randn(
            self.batch_size, self.topk, dtype=self.dtype, device=self.device
        )
        self.output_indices = torch.arange(
            0, self.topk, dtype=torch.int32, device=self.device
        ).repeat(self.batch_size, 1)
        self.topk_buf = torch.zeros(
            (self.batch_size, 8192 * 2 * (2 + 4) // 2 // 48),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self) -> None:
        raft_topk(
            self.input_data,
            self.input_indices,
            self.output_data,
            self.output_indices,
            self.topk_buf,
            self.topk,
        )


class TopPWrapper:
    def __init__(self, batch_size: int, num_heads: int, seq_len: int, p: float):
        self.dtype = torch.float16
        self.device = "cuda"
        self.p = p
        self.logits = self.random_logits(batch_size, num_heads, seq_len)

    def random_logits(self, batch_size: int, num_heads: int, seq_len: int):
        logits = torch.empty(
            (batch_size, num_heads, 1, seq_len),
            dtype=self.dtype,
            device=self.device,
        )
        logits[:, :, :, 0].uniform_(1.5, 2.2)
        logits[:, :, :, 1:].uniform_(-9.0, -1.0)
        return logits

    def forward(self) -> None:
        top_p_unnormalized(self.logits, self.p)


class TopPIndicesWrapper:
    def __init__(self, batch_size: int, num_heads: int, seq_len: int, p: float):
        self.dtype = torch.float16
        self.device = "cuda"
        self.p = p
        self.logits = self.random_logits(batch_size, num_heads, seq_len)

        indices, counts = top_p_unnormalized_return_indices(self.logits, self.p)
        torch.cuda.synchronize()
        self.max_output_len = max(1, int(counts.max().item()))
        self.output_indices = torch.empty(
            (*self.logits.shape[:-1], self.max_output_len),
            dtype=torch.int32,
            device=self.device,
        )
        self.output_counts = torch.empty(
            self.logits.shape[:-1],
            dtype=torch.int32,
            device=self.device,
        )
        top_p_unnormalized_return_indices_out(
            self.logits,
            self.output_indices,
            self.output_counts,
            self.p,
        )
        torch.cuda.synchronize()

    def random_logits(self, batch_size: int, num_heads: int, seq_len: int):
        logits = torch.empty(
            (batch_size, num_heads, 1, seq_len),
            dtype=self.dtype,
            device=self.device,
        )
        logits[:, :, :, 0].uniform_(1.5, 2.2)
        logits[:, :, :, 1:].uniform_(-9.0, -1.0)
        return logits

    def forward(self) -> None:
        top_p_unnormalized_return_indices_out(
            self.logits,
            self.output_indices,
            self.output_counts,
            self.p,
        )


def build_ragged_attention_metadata(
    output_indices: torch.Tensor,
    output_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_counts = output_counts.reshape(-1).to(torch.int32).contiguous()
    flat_indices = output_indices.reshape(flat_counts.numel(), output_indices.shape[-1])
    positions = torch.arange(
        output_indices.shape[-1],
        dtype=torch.int32,
        device=output_indices.device,
    ).unsqueeze(0)
    kv_page_indices = flat_indices[positions < flat_counts.unsqueeze(1)].contiguous()

    kv_page_indptr = torch.empty(
        flat_counts.numel() + 1,
        dtype=torch.int32,
        device=output_indices.device,
    )
    kv_page_indptr[0] = 0
    kv_page_indptr[1:] = torch.cumsum(flat_counts, dim=0, dtype=torch.int32)
    kv_last_page_len = torch.ones_like(flat_counts)
    return kv_page_indices, kv_page_indptr, kv_last_page_len


class PaddedAttentionWrapper:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache: torch.Tensor,
        num_qo_heads: int | None = None,
    ):
        self.dtype = torch.float16
        self.device = "cuda"
        self.batch_size = batch_size
        self.page_size = kv_cache.shape[2]
        self.kv_num_heads = kv_cache.shape[3]
        self.num_qo_heads = num_qo_heads if num_qo_heads is not None else self.kv_num_heads
        self.head_size = kv_cache.shape[4]
        self.q = torch.randn(
            batch_size,
            self.num_qo_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_cache = kv_cache
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.kv_page_indices, self.kv_page_indptr, self.kv_last_page_len = get_indices(
            batch_size, seq_len, self.device
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
        )
        self.decode_wrapper.plan(
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.kv_num_heads,
            self.head_size,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.q.dtype,
            kv_data_type=self.kv_cache.dtype,
        )

    def forward(self) -> None:
        self.decode_wrapper.run(self.q, self.kv_cache)


class VarlenAttentionWrapper:
    def __init__(
        self,
        seq_lens: list[int],
        kv_cache: torch.Tensor,
        num_qo_heads: int = 1,
    ):
        self.dtype = torch.float16
        self.device = "cuda"
        self.batch_size = len(seq_lens)
        self.page_size = kv_cache.shape[2]
        self.num_qo_heads = num_qo_heads
        self.kv_num_heads = kv_cache.shape[3]
        self.head_size = kv_cache.shape[4]
        self.max_seq_len = max(seq_lens)
        self.q = torch.randn(
            self.batch_size,
            self.num_qo_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_cache = kv_cache
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        kv_page_indices = []
        for seq_len in seq_lens:
            kv_page_indices.extend(range(seq_len))
        self.kv_page_indices = torch.tensor(
            kv_page_indices, dtype=torch.int32, device=self.device
        )
        self.kv_page_indptr = torch.zeros(
            self.batch_size + 1, dtype=torch.int32, device=self.device
        )
        for i, seq_len in enumerate(seq_lens):
            self.kv_page_indptr[i + 1] = self.kv_page_indptr[i] + seq_len
        self.kv_last_page_len = torch.ones(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
        )
        self.decode_wrapper.plan(
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.kv_num_heads,
            self.head_size,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.q.dtype,
            kv_data_type=self.kv_cache.dtype,
        )

    def forward(self) -> None:
        self.decode_wrapper.run(self.q, self.kv_cache)


class VarlenAttentionCSRWrapper:
    def __init__(
        self,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        kv_cache: torch.Tensor,
        num_qo_heads: int = 1,
    ):
        self.dtype = torch.float16
        self.device = "cuda"
        self.batch_size = kv_last_page_len.numel()
        self.page_size = kv_cache.shape[2]
        self.num_qo_heads = num_qo_heads
        self.kv_num_heads = kv_cache.shape[3]
        self.head_size = kv_cache.shape[4]
        self.q = torch.randn(
            self.batch_size,
            self.num_qo_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_cache = kv_cache
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.kv_page_indices = kv_page_indices
        self.kv_page_indptr = kv_page_indptr
        self.kv_last_page_len = kv_last_page_len
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
        )
        self.decode_wrapper.plan(
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.kv_num_heads,
            self.head_size,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.q.dtype,
            kv_data_type=self.kv_cache.dtype,
        )

    def forward(self) -> None:
        self.decode_wrapper.run(self.q, self.kv_cache)


def bench_component(name: str, fn: Callable[[], None], warmups: int, repeats: int) -> float:
    _, timing = benchmark_forward_with_warmup(fn, warmups=warmups, repeats=repeats)
    time_us = timing.mean * 1e6
    print(f"[{name}] time: {time_us:.2f} us")
    return time_us


@torch.inference_mode()
def bench_op_breakdown(
    batch_size: int,
    head_size: int,
    num_heads: int,
    seq_len: int,
    b0: int,
    b1: int,
    p: float,
    mode: str,
    budget_profile: str,
    attention_source: str,
    compare_traditional_attention: bool,
    warmups: int,
    repeats: int,
) -> None:
    page_size = 1
    max_num_pages = 32768
    dtype = torch.float16
    device = "cuda:0"

    kv_cache = torch.randn(
        max_num_pages,
        2,
        page_size,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    flat_kv_cache = torch.randn(
        max_num_pages,
        2,
        page_size,
        1,
        head_size,
        dtype=dtype,
        device=device,
    )
    k_cache_int8 = torch.randint(
        -128,
        128,
        (max_num_pages, page_size, num_heads, head_size // 2),
        dtype=torch.int8,
        device=device,
    )

    page_size_quest = 16
    page_num = seq_len // page_size_quest
    k_label = torch.rand(
        (max_num_pages // page_size_quest, 1, num_heads, head_size),
        dtype=dtype,
        device=device,
    )
    quest_selector = QuestSelectorWrapper(batch_size, page_num, k_label)
    topk = TopKWrapper(batch_size, page_num, b0 // page_size_quest, device)

    print(f"batch_size: {batch_size}")
    token_selector_time = bench_component(
        "token selector",
        lambda: (quest_selector.forward(), topk.forward()),
        warmups,
        repeats,
    )

    if mode == "quest":
        print("[spgemv] time: 0.00 us")
        print("[top p] time: 0.00 us")
        attention = PaddedAttentionWrapper(batch_size, b0, kv_cache)
        attention_time = bench_component("attention", attention.forward, warmups, repeats)
        if compare_traditional_attention:
            traditional_attention = PaddedAttentionWrapper(batch_size, seq_len, kv_cache)
            traditional_attention_time = bench_component(
                "traditional attention",
                traditional_attention.forward,
                warmups,
                repeats,
            )
            total_time = token_selector_time + attention_time
            print(
                f"[speedup] attention_only: {traditional_attention_time / attention_time:.2f}x, "
                f"end_to_end: {traditional_attention_time / total_time:.2f}x"
            )
        return

    spgemv = SpGEMVINT4Wrapper(batch_size, b0, k_cache_int8)
    if attention_source == "top-p":
        top_p = TopPIndicesWrapper(batch_size, num_heads, b0, p)
        kv_page_indices, kv_page_indptr, kv_last_page_len = build_ragged_attention_metadata(
            top_p.output_indices,
            top_p.output_counts,
        )
        attention = VarlenAttentionCSRWrapper(
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            flat_kv_cache,
            num_qo_heads=1,
        )
        flat_counts = top_p.output_counts.reshape(-1).float()
        attention_avg_budget = flat_counts.mean().item()
        attention_max_budget = int(flat_counts.max().item())
        flattened_batch = flat_counts.numel()
    elif attention_source == "profile":
        top_p = TopPWrapper(batch_size, num_heads, b0, p)
        head_budgets = scaled_head_budgets(num_heads, b1, budget_profile)
        flat_seq_lens = head_budgets * batch_size
        attention = VarlenAttentionWrapper(flat_seq_lens, flat_kv_cache, num_qo_heads=1)
        attention_avg_budget = sum(head_budgets) / len(head_budgets)
        attention_max_budget = max(head_budgets)
        flattened_batch = len(flat_seq_lens)
    else:
        raise ValueError(f"Unknown attention source: {attention_source}")

    spgemv_time = bench_component("spgemv", spgemv.forward, warmups, repeats)
    top_p_time = bench_component("top p", top_p.forward, warmups, repeats)
    print(
        f"[attention profile] source: {attention_source}, "
        f"avg_budget: {attention_avg_budget:.2f}, "
        f"max_budget: {attention_max_budget}, flattened_batch: {flattened_batch}"
    )
    attention_time = bench_component("attention", attention.forward, warmups, repeats)
    if compare_traditional_attention:
        traditional_attention = PaddedAttentionWrapper(batch_size, seq_len, kv_cache)
        traditional_attention_time = bench_component(
            "traditional attention",
            traditional_attention.forward,
            warmups,
            repeats,
        )
        total_time = token_selector_time + spgemv_time + top_p_time + attention_time
        print(
            f"[speedup] attention_only: {traditional_attention_time / attention_time:.2f}x, "
            f"end_to_end: {traditional_attention_time / total_time:.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Twilight operator-level efficiency breakdown."
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("16,32,64"),
    )
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=30000)
    parser.add_argument("--b0", type=int, default=8192)
    parser.add_argument("--b1", type=int, default=256)
    parser.add_argument("--p", type=float, default=0.85)
    parser.add_argument("--mode", choices=("quest", "quest-twi"), default="quest-twi")
    parser.add_argument(
        "--budget-profile",
        choices=("layer4", "uniform"),
        default="layer4",
        help="Per-head budget profile used when --attention-source=profile.",
    )
    parser.add_argument(
        "--attention-source",
        choices=("top-p", "profile"),
        default="top-p",
        help="Use top-p output indices/counts or synthetic per-head budgets for ragged attention.",
    )
    parser.add_argument(
        "--compare-traditional-attention",
        action="store_true",
        help="Also benchmark dense full-sequence attention and print speedups.",
    )
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()

    for batch_size in args.batch_sizes:
        bench_op_breakdown(
            batch_size=batch_size,
            head_size=args.head_size,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            b0=args.b0,
            b1=args.b1,
            p=args.p,
            mode=args.mode,
            budget_profile=args.budget_profile,
            attention_source=args.attention_source,
            compare_traditional_attention=args.compare_traditional_attention,
            warmups=args.warmups,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
