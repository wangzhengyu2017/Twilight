# TopP


import torch
from torch import nn

from twilight.kernel import (
    top_p_fp32_return_indices,
    top_p_fp32_return_indices_out,
    top_p_fp32_return_mask,
)


def top_p_unnormalized(attn_weights: torch.Tensor, threshold: float) -> torch.Tensor:
    normalized_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    )
    mask = top_p_fp32_return_mask(normalized_weights, threshold)
    return mask


def top_p_unnormalized_return_indices(
    attn_weights: torch.Tensor,
    threshold: float,
    input_indices: torch.Tensor | None = None,
    max_output_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    )
    return top_p_fp32_return_indices(
        normalized_weights, threshold, input_indices, max_output_len
    )


def top_p_unnormalized_return_indices_out(
    attn_weights: torch.Tensor,
    output_indices: torch.Tensor,
    output_counts: torch.Tensor,
    threshold: float,
    input_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    )
    return top_p_fp32_return_indices_out(
        normalized_weights, output_indices, output_counts, threshold, input_indices
    )
