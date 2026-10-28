"""Source-aligned FP16 per-token Q/K INT8 quantization."""
from dataclasses import dataclass, field
import torch

SOURCE_REVISION = '7f7bfb00bcd26b2cef49e7783f51ef610e05abf7'
INPUT_CAST_POLICY = 'native_fp16_allow_underflow_to_zero'
SOURCE_DELTA = ['bf16_to_fp16_input_cast', 'native_fp16_input_underflow_to_zero_allowed', 'zero_group_guard']
QUANTIZATION_COUNT_SEMANTICS = ('Operand elements/groups consumed by each quantizer call after Hq expansion; '
                                'K cache reads are counted again on each attention call, not unique model values.')

@dataclass
class QuantizedTensor:
    codes: torch.Tensor
    scales: torch.Tensor
    dq: torch.Tensor
    diagnostics: dict = field(default_factory=dict)

    def __iter__(self):
        yield self.codes
        yield self.scales
        yield self.dq

def quantize_per_token(x):
    if not x.is_floating_point() or x.ndim != 4:
        raise ValueError('Quantizer requires floating [B,H,L,D] input')
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite quantizer input')
    source = x.to(torch.float16, copy=True)
    nonfinite = ~torch.isfinite(source)
    lost = (x != 0) & (source == 0)
    nonfinite_count = int(nonfinite.sum().item())
    lost_count = int(lost.sum().item())
    examples = []
    if nonfinite_count or lost_count:
        coordinates = (nonfinite | lost).nonzero()[:8].cpu().tolist()
        examples = [{'coordinate': c, 'original': float(x[tuple(c)].item()),
                     'fp16': float(source[tuple(c)].item())} for c in coordinates]
    if nonfinite_count:
        raise ValueError(f'FP16 input cast overflow: source_dtype={x.dtype}, shape={list(x.shape)}, nonfinite_count={nonfinite_count}, examples={examples}')
    xmax = source.abs().max(dim=-1).values
    raw = xmax / 127
    scale = torch.where(xmax == 0, torch.ones_like(raw), raw)
    if not torch.isfinite(scale).all() or (scale == 0).any():
        raise ValueError('Nonzero group scale underflow or nonfinite scale')
    rounded = torch.round(source / scale[..., None])
    if (not torch.isfinite(rounded).all() or (rounded.abs() > 127).any()
            or (rounded != rounded.trunc()).any()):
        raise ValueError('Rounded quantizer codes outside [-127,127]')
    codes = rounded.to(torch.int8)
    diagnostics = {
        'input_cast_policy': INPUT_CAST_POLICY,
        'count_semantics': QUANTIZATION_COUNT_SEMANTICS,
        'source_dtype': str(x.dtype), 'input_dtype': str(source.dtype), 'shape': list(x.shape),
        'input_elements': x.numel(), 'input_nonzero_elements': int((x != 0).sum().item()),
        'input_cast_nonzero_to_zero_count': lost_count,
        'groups': xmax.numel(), 'zero_group_count': int((xmax == 0).sum().item()),
        'examples': examples, 'example_limit': 8, 'examples_truncated': lost_count > len(examples),
    }
    return QuantizedTensor(codes, scale, codes.float() * scale.float()[..., None], diagnostics)

# Explicit compatibility name for independent validators.
quant_pertoken = quantize_per_token
