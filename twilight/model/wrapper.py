import torch
from torch import nn

from typing import List, Optional

from ftka.cuda_ops.gemv import batched_sparse_gemv_int4_k, quest_sparse_gemv
from ftka.cuda_ops.topk import raft_topk

from twilight.pyimpl.top_p import top_p_unnormalized
import flashinfer



def get_indices(batch_size: int, seq_len: int):
    # Construct indices metadata with uniform lengths
    kv_page_indices = torch.arange(seq_len).int().to("cuda")
    kv_page_indices = kv_page_indices.repeat(batch_size)
    kv_page_indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    )
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    return kv_page_indices, kv_page_indptr, kv_last_page_len


class SpGEMVINT4Wrapper:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        k_cache_int8: torch.Tensor,
    ):
        self.dtype = torch.float16
        self.device = k_cache_int8.device
    
        self.batch_size = batch_size
        self.max_num_pages = k_cache_int8.shape[0]
        self.page_size = k_cache_int8.shape[1]
        self.num_heads = k_cache_int8.shape[2]
        self.head_size = k_cache_int8.shape[3] * 2  # int

        self.k_cache_int8 = k_cache_int8

        # NHD
        self.q = torch.rand((self.batch_size, self.num_heads, self.head_size), dtype=self.dtype, device=self.device)
        self.o = torch.empty(
            (self.batch_size, self.num_heads, seq_len), dtype=self.dtype, device=self.device
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
            batch_size, seq_len
        )
    

    def forward(self):
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
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        k_label: torch.Tensor,
    ):
        self.dtype = torch.float16
        self.device = k_label.device

        self.batch_size = batch_size
        self.max_num_pages = k_label.shape[0]
        assert k_label.shape[1] == 1
        self.num_heads = k_label.shape[2]
        self.head_size = k_label.shape[3]

        self.k_label = k_label

        # NHD
        self.q = torch.rand((self.batch_size, self.num_heads, self.head_size), dtype=self.dtype, device=self.device)
        self.o = torch.empty(
            (self.batch_size, self.num_heads, seq_len), dtype=self.dtype, device=self.device
        )

        self.kv_page_indices, self.kv_page_indptr, self.kv_last_page_len = get_indices(
            batch_size, seq_len
        )
    

    def forward(self):
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
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        topk: int,
    ):
        self.dtype = torch.float16
        self.device = "cuda"

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.topk = topk

        self.input_data = torch.randn(batch_size, seq_len, dtype=self.dtype, device=self.device)
        self.input_indices = torch.arange(
            0, self.seq_len, dtype=torch.int32, device=self.device
        ).repeat(self.batch_size, 1)
        self.output_data = torch.randn(self.batch_size, self.topk, dtype=self.dtype, device=self.device)
        self.output_indices = torch.arange(
            0, self.topk, dtype=torch.int32, device=self.device
        ).repeat(self.batch_size, 1)
        self.topk_buf = torch.zeros(
            (self.batch_size, 8192 * 2 * (2 + 4) // 2 // 48), dtype=self.dtype, device=self.device
        )
    

    def forward(self):
        raft_topk(
            self.input_data,
            self.input_indices,
            self.output_data,
            self.output_indices,
            self.topk_buf,
            self.topk,
        )


class TopPWrapper:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        p: float,
    ):
        self.dtype = torch.float16
        self.device = "cuda"
        self.p = p

        self.weight = self.random_weights(batch_size, num_heads, seq_len)
    

    def random_weights(self, bs: int, head_num: int, seq_len: int, seed: int = 42):
        # shape: [bs, head_num, 1, seq_len]

        torch.manual_seed(seed)

        weight = torch.empty((bs, head_num, 1, seq_len), dtype=self.dtype, device=self.device)

        sink_range = [1.5, 2.2]
        other_range = [-9, -1]

        weight[:, :, :, 0].uniform_(sink_range[0], sink_range[1])

        weight[:, :, :, 1:].uniform_(other_range[0], other_range[1])

        weight = nn.functional.softmax(weight, dim=-1)
        return weight

    def forward(self):
        top_p_unnormalized(self.weight, self.p)


class AttentionWrapper:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache: torch.Tensor,
        num_qo_heads: Optional[int] = None,
    ):
        self.dtype = torch.float16
        self.device = "cuda"

        self.batch_size = batch_size
        self.page_size = kv_cache.shape[2]
        self.num_heads = kv_cache.shape[3]
        self.head_size = kv_cache.shape[4]

        self.num_qo_heads = num_qo_heads if num_qo_heads is not None else self.num_heads

        self.q = torch.randn(batch_size, self.num_qo_heads, self.head_size, dtype=self.dtype, device=self.device)
        self.o = torch.empty(batch_size, self.num_heads, seq_len, dtype=self.dtype, device=self.device)
        self.kv_cache = kv_cache

        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.kv_page_indices, self.kv_page_indptr, self.kv_last_page_len = get_indices(batch_size, seq_len)

        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
        )

        self.decode_wrapper.plan(
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_heads,
            self.num_heads,
            self.head_size,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.q.dtype,
            kv_data_type=self.kv_cache.dtype,
        )
    
    def forward(self):
        self.decode_wrapper.run(self.q, self.kv_cache)


class VarlenAttentionWrapper:
    def __init__(
        self,
        seq_lens: List[int],
        kv_cache: torch.Tensor,
        num_qo_heads: int,
    ):
        self.dtype = torch.float16
        self.device = "cuda"

        self.batch_size = len(seq_lens)
        self.page_size = kv_cache.shape[2]
        self.num_qo_heads = num_qo_heads
        self.kv_num_heads = kv_cache.shape[3]
        self.head_size = kv_cache.shape[4]

        self.max_seq_len = max(seq_lens)

        self.q = torch.randn(self.batch_size, self.num_qo_heads, self.head_size, dtype=self.dtype, device=self.device)
        self.o = torch.empty(self.batch_size, self.num_qo_heads, self.max_seq_len, dtype=self.dtype, device=self.device)
        self.kv_cache = kv_cache

        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.kv_page_indices = []
        for seq_len in seq_lens:
            self.kv_page_indices.extend(list(range(seq_len)))
        self.kv_page_indices = torch.tensor(self.kv_page_indices, dtype=torch.int32, device=self.device)
        self.kv_page_indptr = torch.zeros(self.batch_size + 1, dtype=torch.int32, device=self.device)
        for i in range(self.batch_size): 
            self.kv_page_indptr[i + 1] = self.kv_page_indptr[i] + seq_lens[i]
        self.kv_last_page_len = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)

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
    
    def forward(self):
        self.decode_wrapper.run(self.q, self.kv_cache)