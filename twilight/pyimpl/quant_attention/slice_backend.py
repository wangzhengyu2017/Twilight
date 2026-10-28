"""SLICE-v2: exact one-pass Top-p and selected full-code QK recomputation.

Prefix dots use FP32 after integer conversion, never native INT4 MAC. The
executor scatters probabilities and performs a dense, zero-masked floating PV.
"""
import math
import torch


def validate_slice_config(config, method):
    if isinstance(config, dict) and config.get('estimator') == 'format_msb':
        from .format_backend import validate_format_config
        return validate_format_config(config, method)
    required = {'variant_id', 'baseline_name', 'estimator', 'encoding', 'q_prefix_bits',
                'k_prefix_bits', 'selection', 'final_scores', 'final_execution', 'scope', 'diagnostics'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Unknown/missing attention_slice fields')
    if not all(isinstance(config[k], str) and config[k] for k in ('variant_id', 'baseline_name')):
        raise ValueError('Slice identity must be nonempty strings')
    if config['scope'] != 'prefill_and_decode' or config['diagnostics'] != 'targets_only':
        raise ValueError('Unsupported slice scope/diagnostics')
    sel = config['selection']
    if not isinstance(sel, dict) or set(sel) != {'kind', 'p', 'tie_break'}:
        raise ValueError('Unknown/missing selection fields')
    if sel['kind'] != 'top_p' or sel['tie_break'] != 'key_position_ascending':
        raise ValueError('Unsupported selector or tie rule')
    if isinstance(sel['p'], bool) or not isinstance(sel['p'], (float, int)) or not 0 < sel['p'] <= 1:
        raise ValueError('Top-p requires 0 < p <= 1')
    est = config['estimator']
    if est == 'fp_full':
        if method != 'fp_common' or any(config[k] is not None for k in ('encoding', 'q_prefix_bits', 'k_prefix_bits')) or config['final_scores'] != 'fp_full':
            raise ValueError('FP slice must use original floating operands')
    elif est in ('int_full', 'tc_msb'):
        if method != 'intflash_qk8' or config['encoding'] != 'twos_complement' or config['final_scores'] != 'full_int8_qk_dq':
            raise ValueError('INT slice must use full source INT8 codes')
        if any(type(config[k]) is not int or not 1 <= config[k] <= 8 for k in ('q_prefix_bits', 'k_prefix_bits')):
            raise ValueError('Prefix widths must be integers in 1..8')
        if est == 'int_full' and (config['q_prefix_bits'], config['k_prefix_bits']) != (8, 8):
            raise ValueError('INT full estimator requires widths 8/8')
    else:
        raise ValueError('Unsupported estimator')
    if config['final_execution'] != ('selected_full_qk_recompute' if est == 'tc_msb' else 'reuse_full_scores'):
        raise ValueError('Unsupported final execution')


def tc_prefix(codes, bits):
    if type(bits) is not int or not 1 <= bits <= 8:
        raise ValueError('Prefix width must be integer in 1..8')
    a = 1 << (8 - bits)
    hi = torch.div(codes.to(torch.int16), a, rounding_mode='floor')
    return hi, codes.to(torch.int16) - a * hi


def exact_top_p(logits, allowed, p):
    """Return mask, ascending padded indices, counts, validity, full probabilities.

    Key axis is already in ascending original key-position order. Stable sorting
    therefore implements the required ascending-key tie break.
    """
    if not 0 < p <= 1:
        raise ValueError('Top-p requires 0 < p <= 1')
    domain = allowed.expand_as(logits)
    if domain.dtype != torch.bool or not domain.any(-1).all():
        raise ValueError('Every row requires a nonempty boolean legal domain')
    if not torch.isfinite(logits[domain]).all():
        raise ValueError('Nonfinite legal estimator logits')
    scores = logits.masked_fill(~domain, -torch.inf)
    probs = torch.softmax(scores, -1, dtype=torch.float32)
    if p == 1:
        mask = domain.clone()
    else:
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        sorted_probs = probs.gather(-1, order)
        # Include the first crossing item; count preceding cumulative sums < p.
        cumulative = sorted_probs.cumsum(-1)
        previous = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
        chosen = (previous < p) & domain.gather(-1, order)
        mask = torch.zeros_like(domain).scatter_(-1, order, chosen)
    counts = mask.sum(-1)
    width = int(counts.max().item())
    positions = torch.arange(logits.shape[-1], device=logits.device).expand_as(logits)
    indices = positions.masked_fill(~mask, logits.shape[-1]).sort(-1).values[..., :width]
    valid = torch.arange(width, device=logits.device) < counts[..., None]
    indices = indices.masked_fill(~valid, -1)
    return mask, indices, counts, valid, probs


def selected_full_logits(qq, kk, start, end, indices, valid, row_chunk=8):
    """Only gather selected full codes; no unselected full QK is evaluated.

    Small flattened head/query blocks bound the [rows,selected,D] temporary.
    Padding slots are computed and reported as physical work, then discarded.
    """
    b, h, _, d = qq.codes.shape
    rows = end - start
    width = indices.shape[-1]
    kcodes = kk.codes.reshape(b * h, -1, d)
    kscales = kk.scales.reshape(b * h, -1)
    values = torch.empty(indices.shape, device=indices.device, dtype=torch.float32)
    padded_count = 0
    # Parallelize all heads for eight query positions, bounding the temporary
    # to [B*H,8,selected,D] rather than a whole-layer [H,M,N,D].
    for r in range(0, rows, row_chunk):
        stop = min(r + row_chunk, rows)
        local_rows = stop - r
        selected_indices = indices[..., r:stop, :]
        # Repeat a genuinely selected key for padding, never the sentinel or
        # an unrelated unselected key. These extra MACs are counted below.
        safe_indices = torch.where(valid[..., r:stop, :], selected_indices, selected_indices[..., :1])
        ix = safe_indices.reshape(b * h, local_rows * width)
        bh = torch.arange(b * h, device=indices.device)[:, None]
        kdq = kcodes[bh, ix].float() * kscales[bh, ix].float()[..., None]
        kdq = kdq.reshape(b * h * local_rows, width, d)
        qdq = qq.codes[..., start+r:start+stop, :].float() * qq.scales[..., start+r:start+stop, None].float()
        dots = torch.bmm(qdq.reshape(-1, 1, d), kdq.transpose(-1, -2)).reshape(b, h, local_rows, width) / math.sqrt(d)
        values[..., r:stop, :] = dots.masked_fill(~valid[..., r:stop, :], -torch.inf)
        padded_count += b * h * local_rows * width
    return values, padded_count


def attention_slice(q, k, v, allowed, qq, kk, slice_config, query_chunk=64, observer=None, stats=None):
    if slice_config.get('estimator') == 'format_msb':
        from .format_backend import attention_format
        return attention_format(q,k,v,allowed,qq,kk,slice_config,query_chunk,observer,stats)
    if query_chunk < 1 or q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Invalid slice attention layout/chunk')
    est = slice_config['estimator']
    if est != 'fp_full' and (qq is None or kk is None):
        raise ValueError('INT estimator requires actual quantization results')
    qhi = khi = qlo = klo = None
    aq = ak = 1
    if est == 'tc_msb':
        qhi, qlo = tc_prefix(qq.codes, slice_config['q_prefix_bits'])
        khi, klo = tc_prefix(kk.codes, slice_config['k_prefix_bits'])
        aq, ak = 1 << (8 - slice_config['q_prefix_bits']), 1 << (8 - slice_config['k_prefix_bits'])
        qoperand, koperand = qhi.float(), khi.float()
    else:
        qoperand, koperand = (q.float(), k.float()) if est == 'fp_full' else (qq.dq, kk.dq)
    result = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    v32 = v.float()
    d, n = q.shape[-1], k.shape[-2]
    for start in range(0, q.shape[-2], query_chunk):
        end = min(q.shape[-2], start + query_chunk)
        domain = allowed[..., start:end, :]
        dot = torch.matmul(qoperand[..., start:end, :], koperand.transpose(-1, -2))
        dot_hh = None
        if est == 'tc_msb':
            # Conservative exact-integer representability bound for this FP32 dot.
            bound = d * max(128 // aq, 1) * max(128 // ak, 1)
            if bound >= 2 ** 24 or not torch.isfinite(dot).all() or not torch.equal(dot, dot.round()) or (dot.abs() > bound).any():
                raise ValueError('Prefix dot violates exact FP32 integer bounds')
            dot_hh = dot.to(torch.int32)
            logits = ((dot * (aq * ak)) * qq.scales[..., start:end, None].float()) * kk.scales[..., None, :].float() / math.sqrt(d)
        else:
            logits = dot / math.sqrt(d)
        logits = logits.masked_fill(~domain, -torch.inf)
        mask, indices, counts, valid, estimator_probs = exact_top_p(logits, domain, slice_config['selection']['p'])
        padded_count = 0
        if est == 'tc_msb':
            selected, padded_count = selected_full_logits(qq, kk, start, end, indices, valid)
            final = torch.full_like(logits, -torch.inf)
            # Invalid -1 padding must never overwrite any selected or legal entry.
            flat_final = final.reshape(-1, n)
            rr = torch.arange(flat_final.shape[0], device=q.device)[:, None].expand_as(indices.reshape(-1, indices.shape[-1]))
            vi = valid.reshape_as(rr)
            flat_final[rr[vi], indices.reshape_as(rr)[vi]] = selected.reshape_as(rr)[vi]
        else:
            final = logits.masked_fill(~mask, -torch.inf)
        probs = torch.softmax(final, -1, dtype=torch.float32)
        pv = torch.matmul(probs, v32)
        result[..., start:end, :] = pv
        if stats is not None:
            stats.record_slice_statistics(rows=counts.numel(), selected_count=counts.sum(),
                selected_count_squared=(counts * counts).sum(), selected_count_min=counts.min(),
                selected_count_max=counts.max(), allowed_count=domain.expand_as(mask).sum(),
                estimator_qk_macs=dot.numel() * d, selected_full_qk_macs=counts.sum() * d if est == 'tc_msb' else 0,
                actual_selected_full_qk_macs=padded_count * d, actual_pv_macs=probs.numel() * v.shape[-1],
                selection_passes=1, actual_pv='dense_zero_masked', selected_row_chunk=8, heads_parallel=q.shape[1])
        if observer is not None:
            observer(start, end, {'estimator_logits': logits, 'estimator_probs': estimator_probs,
                'selected_mask': mask, 'selected_indices': indices, 'selected_counts': counts, 'selected_valid_mask': valid,
                'counts': counts, 'valid_mask': valid, 'final_selected_logits': final, 'final_selected_probs': probs,
                'dot_hh': dot_hh, 'q_hi': None if qhi is None else qhi[..., start:end, :], 'k_hi': khi,
                'q_lo': None if qlo is None else qlo[..., start:end, :], 'k_lo': klo, 'pv_head': pv})
    return result
