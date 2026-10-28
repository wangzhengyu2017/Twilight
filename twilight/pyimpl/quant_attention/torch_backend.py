"""Shared FP32 natural-softmax attention with bounded query chunks."""
import math
import torch

def causal_allowed(query_positions, key_positions, device=None):
    q = torch.as_tensor(query_positions, device=device).reshape(-1)
    k = torch.as_tensor(key_positions, device=device).reshape(-1)
    allowed = k[None, :] <= q[:, None]
    if not allowed.any(-1).all():
        raise ValueError('Every query requires a legal key')
    return allowed[None, None]

def validate_mask(attention_mask, allowed):
    if attention_mask is None:
        return
    if attention_mask.ndim != 4 or not attention_mask.is_floating_point():
        raise ValueError('Only HF 4D additive causal masks are supported')
    mask = attention_mask[..., :allowed.shape[-2], :allowed.shape[-1]]
    if mask.shape[-2:] != allowed.shape[-2:]:
        raise ValueError('Attention mask dimensions mismatch')
    domain = allowed.expand_as(mask)
    if not (mask[domain] == 0).all() or not (mask[~domain] <= -1e4).all():
        raise ValueError('Mask differs from causal domain or has extra bias')

def attention(q, k, v, allowed_mask, query_chunk=64, observer=None):
    if query_chunk < 1:
        raise ValueError('query_chunk must be positive')
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Invalid common attention tensor layout')
    q32, k32, v32 = q.float(), k.float(), v.float()
    result = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    for start in range(0, q.shape[-2], query_chunk):
        end = min(start + query_chunk, q.shape[-2])
        allowed = allowed_mask[..., start:end, :]
        logits = torch.matmul(q32[..., start:end, :], k32.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        logits.masked_fill_(~allowed, -torch.inf)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        pv = torch.matmul(probs, v32)
        result[..., start:end, :] = pv
        if observer is not None:
            observer(start, end, logits, probs, pv)
    return result

common_attention = attention
