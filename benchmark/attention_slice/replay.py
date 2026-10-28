"""Independent CPU oracle for SLICE-v2 trace evidence.

No production quantizer, selector or attention backend is imported here. Layer
outputs cannot be reprojected without model weights; their capture metadata and
hashes are checked, while the observed attention operands and PV are recomputed.
"""
import argparse
import math
from pathlib import Path

import torch

from benchmark.attention_baseline.common import (canonical_hash, load_json,
    semantic_config, sha256_file, write_json)

ATOL = 1e-5
RTOL = 1e-4
TARGETS = {(l, q, s) for l in (0, 16, 31)
           for q, s in ((127, 'prefill'), (128, 'decode'))}
from .formal import variant_contract, validate_formal_config, variants_for_p

VARIANTS = variants_for_p(.95)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def cpu_dot(q, k):
    """INT64 sum with an explicit D loop, independent of matmul backends."""
    out = torch.zeros(q.shape[:-1] + (k.shape[-2],), dtype=torch.int64)
    for d in range(q.shape[-1]):
        out += q[..., d].long().unsqueeze(-1) * k[..., d].long().unsqueeze(-2)
    return out


def top_p(logits, allowed, positions, p):
    """Stable scalar-row oracle; ties are resolved by original key position."""
    check(0 < p <= 1, 'invalid p')
    allowed = allowed.expand_as(logits)
    check(allowed.any(-1).all(), 'empty legal domain')
    probs = torch.softmax(logits.float().masked_fill(~allowed, -torch.inf), -1)
    selected = torch.zeros_like(allowed)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_allowed = allowed.reshape_as(flat_logits)
    flat_probs = probs.reshape_as(flat_logits)
    flat_selected = selected.reshape_as(flat_logits)
    for i in range(flat_logits.shape[0]):
        legal = [j for j in range(logits.shape[-1]) if bool(flat_allowed[i, j])]
        ordered = sorted(legal, key=lambda j: (-float(flat_logits[i, j]), int(positions[j])))
        if p == 1:
            chosen = ordered
        else:
            # FP32 cumsum matches the specified FP32 selector arithmetic.
            cumulative = torch.cumsum(flat_probs[i, ordered], dim=0)
            hits = (cumulative >= p).nonzero()
            count = int(hits[0]) + 1 if hits.numel() else len(ordered)
            chosen = ordered[:max(1, count)]
        flat_selected[i, chosen] = True
    return selected, probs


def source_quantization(original):
    source = original.to(torch.float16)
    check(torch.isfinite(source).all(), 'source input cast overflow')
    maximum = source.abs().amax(-1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    check(torch.isfinite(scale).all() and (scale > 0).all(), 'source scale underflow')
    codes = torch.round(source / scale[..., None])
    check(torch.isfinite(codes).all() and (codes.abs() <= 127).all(), 'invalid source codes')
    return codes.to(torch.int8), scale


def compare(got, expected, label, exact=False):
    check(torch.is_tensor(got) and got.shape == expected.shape, label + ': shape mismatch')
    if exact or not expected.is_floating_point():
        check(got.dtype == expected.dtype and torch.equal(got, expected), label + ': exact mismatch')
        return {'exact': True}
    check(not torch.isnan(got).any() and not torch.isposinf(got).any(), label + ': NaN/+inf')
    check(torch.equal(torch.isneginf(got), torch.isneginf(expected)), label + ': -inf domain mismatch')
    finite = torch.isfinite(expected)
    check(torch.isfinite(got[finite]).all(), label + ': nonfinite selected values')
    diff = (got[finite].double() - expected[finite].double())
    err = float(diff.abs().max()) if diff.numel() else 0.0
    check(torch.allclose(got[finite], expected[finite], atol=ATOL, rtol=RTOL),
          f'{label}: tolerance failure max_abs={err}')
    return {'max_abs': err}


def check_set_fields(data, mask, positions, prefix='selected_', suffix=''):
    compare(data[prefix + 'mask' + suffix], mask, prefix + 'mask' + suffix, True)
    counts = data[prefix + 'counts' + suffix]
    compare(counts, mask.sum(-1), prefix + 'counts' + suffix, True)
    indices = data[prefix + 'indices' + suffix]
    valid_key = prefix + 'valid_mask' + suffix
    if prefix == 'selected_' and valid_key not in data:
        valid_key = 'valid_mask'
    valid = data[valid_key]
    check(indices.dtype == torch.int64 and valid.dtype == torch.bool and indices.shape == valid.shape,
          'indices/valid mask dtype or shape mismatch')
    check(indices.shape[:-1] == mask.shape[:-1], 'indices row layout mismatch')
    for m, ix, vi in zip(mask.reshape(-1, mask.shape[-1]),
                         indices.reshape(-1, indices.shape[-1]), valid.reshape(-1, valid.shape[-1])):
        keys = positions[m]
        check(torch.equal(vi, torch.arange(ix.numel()) < keys.numel()), 'valid_mask must be prefix')
        check(torch.equal(ix[vi], keys) and (ix[~vi] == -1).all(), 'selected key positions/padding mismatch')


def load_trace_records(run_dir, verify=True):
    root = Path(run_dir).resolve()
    trace = root / 'trace'
    index = load_json(trace / 'index.json')
    check(index.get('schema_version') == 'attention-topp-trace-v2', 'wrong trace schema')
    check(len(index['records']) == 6, 'expected six trace records')
    seen = set()
    result = []
    for entry in index['records']:
        rid = entry['record_id']
        check(rid not in seen, 'duplicate record id')
        seen.add(rid)
        path = (trace / entry.get('path', rid)).resolve()
        check(path.is_relative_to(trace), 'unsafe trace path')
        if verify:
            for name, key in [('actual.pt', 'actual_sha256'), ('derived.pt', 'derived_sha256'), ('meta.json', 'meta_sha256')]:
                check(entry[key] == sha256_file(path / name), f'{rid}/{name}: index hash mismatch')
        meta = load_json(path / 'meta.json')
        if verify:
            for name in ('actual.pt', 'derived.pt'):
                check(meta['files'][name] == sha256_file(path / name), f'{rid}/{name}: meta hash mismatch')
        actual = torch.load(path / 'actual.pt', map_location='cpu', weights_only=True)
        derived = torch.load(path / 'derived.pt', map_location='cpu', weights_only=True)
        result.append((entry, meta, actual, derived))
    identities = [(m['layer_id'], m['query_position'], m['stage']) for _, m, _, _ in result]
    check(len(set(identities)) == 6 and set(identities) == TARGETS, 'trace target identities mismatch')
    return result


def replay_record(entry, meta, a, derived):
    rid = entry['record_id']
    required = ('layer_input', 'layer_output', 'attn_input', 'o_proj_input', 'attn_output',
                'q_post_rope', 'k_visible', 'v_visible', 'q_codes', 'k_codes', 'q_scales', 'k_scales',
                'allowed_mask', 'key_positions', 'selected_mask', 'selected_indices', 'selected_counts',
                'valid_mask', 'estimator_logits', 'estimator_probs', 'final_selected_logits',
                'final_selected_probs', 'pv_head')
    check(all(k in a for k in required), f'{rid}: missing actual fields')
    for name, value in a.items():
        info = meta['fields'].get(name)
        if value is None:
            check(info is None, name + ': null field metadata mismatch')
            continue
        check(torch.is_tensor(value) and info is not None, name + ': invalid field')
        check(info.get('layout') and info.get('capture_point'), name + ': incomplete capture metadata')
        check(info['shape'] == list(value.shape) and str(info['dtype']).removeprefix('torch.') == str(value.dtype).removeprefix('torch.'), name + ': tensor metadata mismatch')
        if value.is_floating_point():
            if name in ('estimator_logits', 'final_selected_logits'):
                check(not torch.isnan(value).any() and not torch.isposinf(value).any(), name + ': NaN/+inf')
            else:
                check(torch.isfinite(value).all(), name + ': nonfinite')
    check(set(derived) == set(meta.get('derived_fields', {})), 'derived tensor inventory mismatch')
    for name, value in derived.items():
        info = meta['derived_fields'][name]
        check(torch.is_tensor(value) and info and info.get('layout') and info.get('capture_point'), name + ': invalid derived metadata')
        check(info['shape'] == list(value.shape) and str(info['dtype']).removeprefix('torch.') == str(value.dtype).removeprefix('torch.'), name + ': derived shape/dtype mismatch')
        if value.is_floating_point():
            check(torch.isfinite(value).all(), name + ': nonfinite derived tensor')
    check((meta['Hq'], meta['Hkv'], meta['D']) == (32, 8, 128), 'architecture mismatch')
    check(meta['head_to_kv'] == [h // 4 for h in range(32)], 'GQA mapping mismatch')
    check(meta['sample_id'] == 0 and meta['window_id'] is None and meta['predicted_target'] == meta['query_position'] + 1, 'sample/target mismatch')
    check(meta['selection_passes'] == 1 and meta['scope'] == 'prefill_and_decode', 'selection scope/pass mismatch')
    check(meta['backend'] == 'torch_qdq' and meta['cache_storage'] == 'original_float'
          and meta['quantization_at'] == 'attention_read', 'execution/cache metadata mismatch')
    check(meta['model_dtype'] == 'torch.bfloat16', 'model dtype mismatch')
    check(meta['cache_length'] == meta['expected_cache_length'] == meta['query_position'] + 1,
          'cache position metadata mismatch')
    selection = meta.get('selection', {})
    p = meta.get('p', selection.get('p'))
    tie = meta.get('tie_break', selection.get('tie_break'))
    expected_p, variants, arithmetic = variant_contract(meta['variant_id'])
    check(p == expected_p and tie == 'key_position_ascending', 'official p/tie mismatch')
    check(selection == {'kind': 'top_p', 'p': p, 'tie_break': tie}, 'selection metadata mismatch')
    check((meta['method'], meta['estimator'], meta['q_prefix_bits'], meta['k_prefix_bits']) == arithmetic,
          'official variant arithmetic mismatch')
    check(meta['set_role'] == ('predicted_k' if meta['estimator'] == 'tc_msb' else 'important_k_reference'),
          'official set role mismatch')
    q, k, v = a['q_post_rope'], a['k_visible'], a['v_visible']
    n = meta['query_position'] + 1
    check(q.shape == (1, 32, 1, 128) and k.shape == v.shape == (1, 32, n, 128), 'Q/K/V dimensions mismatch')
    check(q.dtype == k.dtype == v.dtype == torch.bfloat16, 'original Q/K/V dtype mismatch')
    positions = a['key_positions']
    check(positions.dtype == torch.int64 and torch.equal(positions, torch.arange(n)), 'visible key domain mismatch')
    allowed = a['allowed_mask']
    check(allowed.dtype == torch.bool and allowed.shape == (1, 1, 1, n) and allowed.all(), 'visible allowed domain mismatch')
    errors = {}
    quantized = meta['variant_id'] != variants[0]
    if quantized:
        check(meta['method'] == 'intflash_qk8' and meta['actual_matmul'] == 'fp32_after_dq', 'INT executor mismatch')
        check(meta['source_revision'] == '7f7bfb00bcd26b2cef49e7783f51ef610e05abf7'
              and meta['source_delta'] == ['bf16_to_fp16_input_cast', 'native_fp16_input_underflow_to_zero_allowed', 'zero_group_guard']
              and meta['input_cast_policy'] == 'native_fp16_allow_underflow_to_zero', 'source quantizer provenance mismatch')
        check(meta['zero_point'] == 0 and meta['statistics_axis'] == 'D'
              and meta['scale_broadcast'] == '[B,H,L,1]' and meta['rounding'] == 'torch_round_ties_to_even'
              and meta['quantizer_input_dtype'] == 'float16' and meta['source_input_cast'] == 'bf16_to_fp16', 'source quantizer convention mismatch')
        for name, original in (('q', q), ('k', k)):
            codes, scales = source_quantization(original)
            compare(a[name + '_codes'], codes, name + ' source codes', True)
            compare(a[name + '_scales'], scales, name + ' source scales', True)
            dq = codes.float() * scales.float()[..., None]
            if a.get(name + '_dq') is not None:
                compare(a[name + '_dq'], dq, name + ' DQ', True)
        qc, kc, qs, ks = (a[x] for x in ('q_codes', 'k_codes', 'q_scales', 'k_scales'))
        q_full = qc.float() * qs.float()[..., None]
        k_full = kc.float() * ks.float()[..., None]
        full_dot = cpu_dot(qc, kc)
        if 'int64_dot' in derived:
            compare(derived['int64_dot'], full_dot, 'derived integer dot', True)
        check(int(full_dot.abs().max()) <= 128 * 127 * 127, 'integer dot bound failure')
        full_logits = (q_full @ k_full.transpose(-1, -2)) / math.sqrt(128)
        if meta['variant_id'].startswith('msb_'):
            bq, bk = meta['q_prefix_bits'], meta['k_prefix_bits']
            check((bq, bk) == ((4, 4) if meta['variant_id'] == variants[2] else (8, 4)), 'official prefix mismatch')
            aq, ak = 2 ** (8 - bq), 2 ** (8 - bk)
            qh, kh = qc.long().div(aq, rounding_mode='floor'), kc.long().div(ak, rounding_mode='floor')
            ql, kl = qc.long() - aq * qh, kc.long() - ak * kh
            hh, hl, lh, ll = cpu_dot(qh, kh), cpu_dot(qh, kl), cpu_dot(ql, kh), cpu_dot(ql, kl)
            check(torch.equal(full_dot, aq * ak * hh + aq * hl + ak * lh + ll), 'TC cross terms fail')
            for name, expected in [('q_hi', qh), ('k_hi', kh), ('q_lo', ql), ('k_lo', kl), ('dot_hh', hh)]:
                if a.get(name) is not None:
                    check(torch.equal(a[name], expected), name + ': integer prefix mismatch')
                if name in derived:
                    check(torch.equal(derived[name], expected), name + ': derived integer prefix mismatch')
            check(all(a.get(name) is not None for name in ('q_hi', 'k_hi', 'dot_hh')), 'missing actual prefix operands')
            estimated = (((aq * ak * hh).float() * qs.float()[..., None]) * ks.float().unsqueeze(-2)) / math.sqrt(128)
            reference_mask, reference_probs = top_p(full_logits, allowed, positions, p)
            check_set_fields(derived, reference_mask, positions, 'reference_int_top_p_', '_same_input')
            compare(derived['reference_int_logits_same_input'], full_logits, 'same-input reference logits')
            compare(derived['reference_int_probs_same_input'], reference_probs, 'same-input reference probabilities')
            check(meta['final_operand_origin'] == 'full_codes', 'wrong final operand origin')
        else:
            estimated = full_logits
            check(meta['final_operand_origin'] == 'full_codes', 'wrong INT operand origin')
    else:
        check(meta['method'] == 'fp_common' and meta['actual_matmul'] == 'fp32', 'FP executor mismatch')
        check(all(a[x] is None for x in ('q_codes', 'k_codes', 'q_scales', 'k_scales')), 'FP manufactured codes')
        q_full, k_full = q.float(), k.float()
        full_logits = q_full @ k_full.transpose(-1, -2) / math.sqrt(128)
        estimated = full_logits
        check(meta['final_operand_origin'] == 'original_float', 'wrong FP operand origin')
    estimated = estimated.masked_fill(~allowed, -torch.inf)
    selected, estimator_probs = top_p(estimated, allowed, positions, p)
    check_set_fields(a, selected, positions)
    # Recompute only S with complete operands; do not select a second time.
    final_logits = torch.full_like(full_logits, -torch.inf)
    for h in range(32):
        chosen = selected[0, h, 0]
        final_logits[0, h, 0, chosen] = (q_full[0, h, 0] @ k_full[0, h, chosen].T) / math.sqrt(128)
    final_probs = torch.softmax(final_logits, -1)
    pv = final_probs @ v.float()
    for name, expected in [('estimator_logits', estimated), ('estimator_probs', estimator_probs),
                           ('final_selected_logits', final_logits), ('final_selected_probs', final_probs), ('pv_head', pv)]:
        errors[name] = compare(a[name], expected, name)
        if name in derived:
            errors['derived_' + name] = compare(derived[name], expected, 'derived ' + name)
    check((a['final_selected_probs'][~selected] == 0).all(), 'nonzero probability outside S')
    check(torch.allclose(a['final_selected_probs'].sum(-1), torch.ones((1, 32, 1)), atol=ATOL, rtol=RTOL), 'probabilities do not sum to one')
    # Genuine observation check: flattening PV is exactly the input to o_proj.
    expected_o = a['pv_head'].transpose(1, 2).reshape(1, 1, -1)
    compare(a['o_proj_input'].float(), expected_o.to(a['o_proj_input'].dtype).float(), 'PV/o_proj input')
    return {'record_id': rid, 'integer_exact': quantized, 'errors': errors,
            'layer_outputs': 'hash_and_capture_metadata_only_no_model_weights'}


def replay_run(run_dir):
    root = Path(run_dir).resolve()
    check(load_json(root / 'status.json')['status'] == 'complete', 'run incomplete')
    metrics = load_json(root / 'metrics.json')
    config = load_json(root / 'resolved_config.json')
    algorithm = config.get('algorithm', config)
    validate_formal_config(algorithm)
    assets = load_json(root / 'asset_manifest.json')
    code = load_json(root / 'code_manifest.json')
    check(code['code_sha256'] == metrics['code_sha256'] == canonical_hash({'head': code['head'], 'files': code['files']}), 'invalid code identity')
    check(code['patch_sha256'] == sha256_file(root / 'code.patch'), 'patch hash mismatch')
    check(canonical_hash(assets) == metrics['asset_manifest_sha256'], 'asset hash mismatch')
    reports = []
    for entry, meta, actual, derived in load_trace_records(root):
        check(meta['variant_id'] == metrics['variant_id'] == algorithm['attention_slice']['variant_id'], 'variant identity mismatch')
        slice_config = algorithm['attention_slice']
        check(meta.get('attention_slice') == slice_config, 'trace/resolved slice config mismatch')
        check(all(meta.get(key) == value for key, value in slice_config.items()), 'flattened trace slice config mismatch')
        check(meta.get('p') == slice_config['selection']['p']
              and meta.get('tie_break') == slice_config['selection']['tie_break'], 'trace/resolved selection mismatch')
        for key in ('tokens_sha256', 'code_sha256'):
            check(meta[key] == metrics[key], 'trace provenance mismatch: ' + key)
        check(meta['model_file_hashes'] == assets['model_file_hashes'], 'trace model identity mismatch')
        check(meta['config_sha256'] == metrics['semantic_config_sha256'] == canonical_hash(semantic_config(algorithm)), 'config identity mismatch')
        reports.append(replay_record(entry, meta, actual, derived))
    return {'status': 'complete', 'passed': True, 'run_dir': str(root), 'num_records': len(reports),
            'atol': ATOL, 'rtol': RTOL, 'records': reports,
            'index_sha256': sha256_file(root / 'trace/index.json'),
            'oracle': 'independent_cpu_source_fp16_int64_per_D_stable_top_p_selected_full_dq_pv'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = replay_run(args.run_dir)
    except Exception as exc:
        write_json(args.output, {'status': 'failed', 'passed': False, 'reason': str(exc), 'atol': ATOL, 'rtol': RTOL})
        raise
    write_json(args.output, report)


if __name__ == '__main__':
    main()
