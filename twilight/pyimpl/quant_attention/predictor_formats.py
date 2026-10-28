"""Format sidecars. Original source quantization objects are read-only."""
import torch


def _bounded_ldexp(value, exponent):
    """Exact power-of-two scaling without an overflowing intermediate factor.

    Legal MX inputs use shifts in [-133,133]. Splitting the shift keeps both
    factors representable in binary32. For conversion, subnormal inputs scale
    upward first; reconstruction starts from small integer/E2M1 values and only
    the final step may enter the subnormal range.
    """
    first=torch.div(exponent,2,rounding_mode='floor')
    return torch.ldexp(torch.ldexp(value,first),exponent-first)


def sm8(codes, bits):
    c = codes.to(torch.int16)
    if (c.abs() > 127).any():
        raise ValueError('SM source outside symmetric INT8 range')
    word = ((c < 0).to(torch.int16) * 128 + c.abs()).to(torch.uint8)
    prefix = (word.to(torch.int16) >> (8-bits)).to(torch.uint8)
    hi = torch.sign(c) * torch.div(c.abs(), 1 << (8-bits), rounding_mode='floor')
    return dict(full_codes=word, prefix_bytes=prefix, hi=hi,
                full_dq=c.float(), predictor_dq=hi.float() * (1 << (8-bits)))


def mx_sidecar(source, bits, fp4=False):
    if source.shape[-1] % 32:
        raise ValueError('MX requires complete contiguous block32 operands')
    shape = source.shape
    blocks = source.float().reshape(*shape[:-1], -1, 32)
    if not torch.isfinite(blocks).all():
        raise ValueError('Nonfinite MX source')
    maximum = blocks.abs().amax(-1)
    _, exponent = torch.frexp(maximum)
    exponent = torch.where(maximum == 0, 0, exponent-1-(2 if fp4 else 0)).to(torch.int32)
    if ((exponent < -127) | (exponent > 127)).any():
        raise ValueError('MX exponent outside finite E8M0 range')
    if fp4:
        u = _bounded_ldexp(blocks, -exponent[..., None])
        table = torch.tensor([0,.5,1,1.5,2,3,4,6], device=source.device, dtype=torch.float32)
        distance = (u.abs()[..., None]-table).abs()
        # Prefer even encoding on an exact tie, then resolve unique minima.
        minimum = distance.amin(-1, keepdim=True)
        tied = distance == minimum
        order = torch.tensor([0,2,4,6,1,3,5,7], device=source.device)
        mag = order[tied[..., order].to(torch.int8).argmax(-1)]
        code = (mag + ((u < 0) & (mag != 0)).long()*8).to(torch.uint8)
        decoded = table[mag] * torch.where(code >= 8, -1., 1.)
        full = _bounded_ldexp(decoded, exponent[..., None])
        prefix, hi, pred = code, decoded, full
        clamped = u.abs() > 6
    else:
        u = _bounded_ldexp(blocks, 6-exponent[..., None])
        rounded = torch.round(u)
        clamped = (rounded < -127) | (rounded > 127)
        code = rounded.clamp(-127,127).to(torch.int8)
        hi = torch.div(code.to(torch.int16),16,rounding_mode='floor') if bits == 4 else code.to(torch.int16)
        prefix = (code.to(torch.int16).bitwise_and(255) >> 4).to(torch.uint8) if bits == 4 else code.view(torch.uint8)
        full = _bounded_ldexp(code.float(), exponent[..., None]-6)
        pred = _bounded_ldexp(hi.float(), exponent[..., None]-(2 if bits == 4 else 6))
    if not torch.isfinite(pred).all() or not torch.isfinite(full).all():
        raise ValueError('Nonfinite reconstructed MX operand')
    return dict(full_codes=code.reshape(shape), prefix_bytes=prefix.reshape(shape),
                hi=hi.reshape(shape), full_dq=full.reshape(shape).contiguous(),
                predictor_dq=pred.reshape(shape).contiguous(), exponent=exponent,
                e8m0=(exponent+127).to(torch.uint8), clamp_mask=clamped.reshape(shape),
                zero_blocks=maximum == 0, underflow_mask=((blocks != 0)&(full == 0)).reshape(shape))


def operands(qq, kk, config):
    fmt = config['predictor_format']
    result = []
    for name, source in (('q',qq),('k',kk)):
        bits = config[name+'_prefix_bits']
        if fmt == 'sm8':
            item = sm8(source.codes,bits)
            item['full_dq'] = source.codes.float()*source.scales.float()[...,None]
            item['predictor_dq'] = item['predictor_dq']*source.scales.float()[...,None]
        else:
            values = source.codes.float()*source.scales.float()[...,None]
            fp4 = fmt == 'mxfp4' or (fmt == 'mxint8q_mxfp4k' and name == 'k')
            item = mx_sidecar(values,bits,fp4)
        result.append(item)
    return result
