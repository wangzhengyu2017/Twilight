from typing import Optional, Tuple
import os
import torch
from torch import nn
import gc
import time

import flashinfer

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaAttention,
    LlamaModel,
    repeat_kv,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    List,
    Union,
    CrossEntropyLoss,
    BaseModelOutputWithPast,
)
import types

from .flashinfer_utils import apply_rope_inplace, enable_flashinfer_rmsnorm


from twilight.ops import (
    BatchDecodeWithPagedKVCacheWrapper as TwilightWrapper,
    batched_sparse_gemv,
    batched_sparse_gemv_int8_k,
    get_label_tensor,
)

from twilight.pyimpl.top_p import top_p_unnormalized

def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # Shape conversion:
    #   Q: [bs, num_heads, q_len, head_dim]
    #   K: [bs, num_kv_heads, q_len, head_dim]
    #   V: [bs, num_kv_heads, q_len, head_dim]
    bs, q_len, _ = hidden_states.size()

    # Prefill stage
    if q_len > 1:
        # return self.flash_forward(
        #     hidden_states,
        #     attention_mask,
        #     position_ids,
        #     past_key_value,
        #     output_attentions,
        #     use_cache,
        #     **kwargs,
        # )
        # Skip prefill
        # self.kv_seq_len = q_len
        return hidden_states, None, None

    # print(q_len)

    query_states = self.q_proj(hidden_states).view(
        bs, q_len, self.num_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        bs, q_len, self.num_key_value_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        bs, q_len, self.num_key_value_heads, self.head_dim
    )

    rope_scale = 1.0
    apply_rope_inplace(
        query_states, key_states, position_ids[:, 0], rope_scale, self.rope_theta
    )

    # Currently we assert Attention Mask to be None
    assert attention_mask is None

    self.kv_seq_len += 1
    kv_seq_len = self.kv_seq_len

    query_states = query_states.view(bs, self.num_heads, self.head_dim)

    # st = torch.cuda.Event(enable_timing=True)
    # ed = torch.cuda.Event(enable_timing=True)
    # st.record()
    # torch.cuda.synchronize()

    page_num = kv_seq_len // 16

    # t0 = torch.cuda.Event(enable_timing=True)
    # t1 = torch.cuda.Event(enable_timing=True)
    # t2 = torch.cuda.Event(enable_timing=True)
    # t3 = torch.cuda.Event(enable_timing=True)

    # t0.record()
    # torch.cuda.synchronize()

    if self.use_quest:
        tmp_scores = batched_sparse_gemv(
            query_states,
            self.k_label,
            *self.page_indices,
            page_num,
        )
        _, label_index = torch.topk(tmp_scores, self.B0 // 16, dim=-1)

        kv_seq_len = self.B0

    # t1.record()
    # torch.cuda.synchronize()

    if self.use_twilight:
        estimated_weights = batched_sparse_gemv_int8_k(
            query_states,
            self.k_cache_int8,
            self.quant_scales,
            self.quant_zeros,
            *self.B0_indices,
            self.B0,
        )
        top_p_unnormalized(estimated_weights, 0.85)

        kv_seq_len = self.B1

    # t2.record()
    # torch.cuda.synchronize()

    kv_page_indices = torch.arange(kv_seq_len).int().to("cuda:0")
    kv_page_indices = kv_page_indices.repeat(bs)
    kv_page_indptr = (
        torch.arange(bs + 1, dtype=torch.int32, device="cuda:0") * kv_seq_len
    )
    kv_last_page_len = torch.ones(bs, dtype=torch.int32, device="cuda:0")

    self.decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        self.num_heads,
        self.num_heads,
        self.head_dim,
        page_size=1,
        pos_encoding_mode="NONE",
        q_data_type=query_states.dtype,
        kv_data_type=self.kv_cache.dtype,
    )
    attn_output = self.decode_wrapper.run(query_states, self.kv_cache).view(bs, 1, -1)

    # t3.record()
    # torch.cuda.synchronize()

    # print(t0.elapsed_time(t1), t1.elapsed_time(t2), t2.elapsed_time(t3), flush=True)

    attn_output = self.o_proj(attn_output)

    # ed.record()
    # torch.cuda.synchronize()
    # print(st.elapsed_time(ed), flush=True)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_twilight(
    model: LlamaForCausalLM,
    use_flash_infer: bool = False,
    use_quest: bool = False,
    use_twilight: bool = False,
    max_num_pages: int = 32768,
    seq_len: int = 1024,
    B0: int = 1024,
    B1: int = 1024,
    bs: int = 32,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    num_heads = 32
    head_size = 128

    kv_cache = torch.randn(
        max_num_pages,
        2,
        1,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    page_size_quest = 16
    page_num = seq_len // page_size_quest
    k_label = torch.rand(
        (max_num_pages // page_size_quest, 1, num_heads, head_size),
        dtype=dtype,
        device=device,
    )

    # print(kv_cache.shape)

    # 2 int4 are packed into 1 int8
    k_cache_int8 = torch.randint(
        -128,
        128,
        (
            max_num_pages,
            1,
            num_heads,
            head_size // 2,
        ),
        dtype=torch.int8,
        device=device,
    )

    quant_scales = torch.randn(
        max_num_pages,
        1,
        num_heads,
        dtype=dtype,
        device=device,
    )

    quant_zeros = torch.randn(
        max_num_pages,
        1,
        num_heads,
        dtype=dtype,
        device=device,
    )

    kv_page_indices_page = torch.arange(page_num).int().to("cuda:0")
    kv_page_indices_page = kv_page_indices_page.repeat(bs)
    kv_page_indptr_page = (
        torch.arange(bs + 1, dtype=torch.int32, device="cuda:0") * page_num
    )
    kv_last_page_len_page = torch.ones(bs, dtype=torch.int32, device="cuda:0")

    kv_page_indices_B0 = torch.arange(B0).int().to("cuda:0")
    kv_page_indices_B0 = kv_page_indices_B0.repeat(bs)
    kv_page_indptr_B0 = torch.arange(bs + 1, dtype=torch.int32, device="cuda:0") * B0
    kv_last_page_len_B0 = torch.ones(bs, dtype=torch.int32, device="cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    for idx, layer in enumerate(model.model.layers):
        # print(idx)
        module = layer.self_attn

        assert isinstance(module, LlamaAttention)

        module.kv_seq_len = seq_len

        module.use_flash_infer = use_flash_infer
        module.use_quest = use_quest
        module.use_twilight = use_twilight

        module.kv_cache = kv_cache
        module.k_label = k_label
        module.k_cache_int8 = k_cache_int8
        module.quant_scales = quant_scales
        module.quant_zeros = quant_zeros
        module.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            "NHD",
            use_tensor_cores=False,
        )
        module.B0 = B0
        module.B1 = B1
        module.page_indices = (
            kv_page_indices_page,
            kv_page_indptr_page,
            kv_last_page_len_page,
        )
        module.B0_indices = (kv_page_indices_B0, kv_page_indptr_B0, kv_last_page_len_B0)

        module.flash_forward = module.forward
        module.forward = types.MethodType(attention_forward, module)
