"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from types import SimpleNamespace
from typing import Tuple, Union, Optional

import torch

from flashinfer.jit import (
    has_prebuilt_ops,
    load_cuda_ops,
    FLASHINFER_INCLUDE_DIR,
    FLASHINFER_CSRC_DIR,
)

from flashinfer.utils import (
    get_cuda_stream,
    register_custom_op,
)

from .env import TWILIGHT_CSRC_DIR, TWILIGHT_INCLUDE_DIR


_sampling_module = None


def get_sampling_module():

    global _sampling_module
    if _sampling_module is None:
        if has_prebuilt_ops:
            from . import _kernels

            module = _kernels
        else:
            module = load_cuda_ops(
                "twilight_sampling",
                sources=[
                    TWILIGHT_CSRC_DIR / "sampling.cu",
                ],
                extra_include_paths=[
                    FLASHINFER_INCLUDE_DIR,
                    FLASHINFER_CSRC_DIR,
                    TWILIGHT_INCLUDE_DIR,
                ],
            )

        # Register ops
        @register_custom_op("flashinfer::top_p_fp16_return_mask", mutates_args=())
        def top_p_fp16_return_mask(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            with probs.device as device:
                maybe_top_p_arr = (
                    maybe_top_p_arr if maybe_top_p_arr is not None else None
                )
                mask = torch.zeros_like(probs, dtype=torch.bool)
                module.top_p_fp16_return_mask(
                    probs,
                    mask,
                    maybe_top_p_arr,
                    top_p_val,
                    get_cuda_stream(device),
                )
                return mask

        @register_custom_op("flashinfer::top_p_fp32_return_mask", mutates_args=())
        def top_p_fp32_return_mask(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            with probs.device as device:
                maybe_top_p_arr = (
                    maybe_top_p_arr if maybe_top_p_arr is not None else None
                )
                mask = torch.zeros_like(probs, dtype=torch.bool)
                module.top_p_fp32_return_mask(
                    probs,
                    mask,
                    maybe_top_p_arr,
                    top_p_val,
                    get_cuda_stream(device),
                )
                return mask

        @register_custom_op("flashinfer::top_p_fp16_return_indices", mutates_args=())
        def top_p_fp16_return_indices(
            probs: torch.Tensor,
            maybe_input_indices: Optional[torch.Tensor],
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with probs.device as device:
                indices = torch.empty_like(probs, dtype=torch.int32)
                counts = torch.empty(probs.shape[:-1], dtype=torch.int32, device=device)
                module.top_p_fp16_return_indices(
                    probs,
                    indices,
                    counts,
                    maybe_input_indices,
                    maybe_top_p_arr,
                    top_p_val,
                    get_cuda_stream(device),
                )
                return indices, counts

        @register_custom_op("flashinfer::top_p_fp32_return_indices", mutates_args=())
        def top_p_fp32_return_indices(
            probs: torch.Tensor,
            maybe_input_indices: Optional[torch.Tensor],
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with probs.device as device:
                indices = torch.empty_like(probs, dtype=torch.int32)
                counts = torch.empty(probs.shape[:-1], dtype=torch.int32, device=device)
                module.top_p_fp32_return_indices(
                    probs,
                    indices,
                    counts,
                    maybe_input_indices,
                    maybe_top_p_arr,
                    top_p_val,
                    get_cuda_stream(device),
                )
                return indices, counts

        # Register the module
        _sampling_module = SimpleNamespace(
            _raw_module=module,
            top_p_fp16_return_mask=top_p_fp16_return_mask,
            top_p_fp32_return_mask=top_p_fp32_return_mask,
            top_p_fp16_return_indices=top_p_fp16_return_indices,
            top_p_fp32_return_indices=top_p_fp32_return_indices,
        )

    return _sampling_module


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def top_p_fp16_return_mask(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return get_sampling_module().top_p_fp16_return_mask(
        probs, *_to_tensor_scalar_tuple(top_p)
    )


def top_p_fp32_return_mask(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return get_sampling_module().top_p_fp32_return_mask(
        probs, *_to_tensor_scalar_tuple(top_p)
    )


def top_p_fp16_return_indices(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    input_indices: Optional[torch.Tensor] = None,
    max_output_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact selected indices and per-row true counts.

    Counts are not clipped by max_output_len; choose capacity >= counts.max()
    when the full index list is required.
    """
    output_len = max_output_len if max_output_len is not None else probs.shape[-1]
    indices = torch.empty(
        (*probs.shape[:-1], output_len), dtype=torch.int32, device=probs.device
    )
    counts = torch.empty(probs.shape[:-1], dtype=torch.int32, device=probs.device)
    return top_p_fp16_return_indices_out(probs, indices, counts, top_p, input_indices)


def top_p_fp32_return_indices(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    input_indices: Optional[torch.Tensor] = None,
    max_output_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact selected indices and per-row true counts.

    Counts are not clipped by max_output_len; choose capacity >= counts.max()
    when the full index list is required.
    """
    output_len = max_output_len if max_output_len is not None else probs.shape[-1]
    indices = torch.empty(
        (*probs.shape[:-1], output_len), dtype=torch.int32, device=probs.device
    )
    counts = torch.empty(probs.shape[:-1], dtype=torch.int32, device=probs.device)
    return top_p_fp32_return_indices_out(probs, indices, counts, top_p, input_indices)


def top_p_fp16_return_indices_out(
    probs: torch.Tensor,
    output_indices: torch.Tensor,
    output_counts: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    input_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Write compact selected indices into preallocated output tensors.

    Counts are not clipped by output_indices.shape[-1].
    """
    module = get_sampling_module()
    maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)
    # The registered custom op allocates outputs. Use the JIT module directly for reusable buffers.
    raw_module = getattr(module, "_raw_module", None)
    if raw_module is None:
        raise RuntimeError("raw twilight sampling module is unavailable")
    raw_module.top_p_fp16_return_indices(
        probs,
        output_indices,
        output_counts,
        input_indices,
        maybe_top_p_arr,
        top_p_val,
        get_cuda_stream(probs.device),
    )
    return output_indices, output_counts


def top_p_fp32_return_indices_out(
    probs: torch.Tensor,
    output_indices: torch.Tensor,
    output_counts: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    input_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Write compact selected indices into preallocated output tensors.

    Counts are not clipped by output_indices.shape[-1].
    """
    module = get_sampling_module()
    maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)
    raw_module = getattr(module, "_raw_module", None)
    if raw_module is None:
        raise RuntimeError("raw twilight sampling module is unavailable")
    raw_module.top_p_fp32_return_indices(
        probs,
        output_indices,
        output_counts,
        input_indices,
        maybe_top_p_arr,
        top_p_val,
        get_cuda_stream(probs.device),
    )
    return output_indices, output_counts
