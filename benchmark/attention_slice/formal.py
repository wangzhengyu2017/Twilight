"""Strict threshold-specific identities for the three formal SLICE-v2 matrices."""

THRESHOLDS = {'p095': .95, 'p090': .90, 'p085': .85}
ROLES = ('fp_topp', 'int_topp', 'msb_q4k4_topp', 'msb_q8k4_topp')
ARITHMETIC = (('fp_common', 'fp_full', None, None),
              ('intflash_qk8', 'int_full', 8, 8),
              ('intflash_qk8', 'tc_msb', 4, 4),
              ('intflash_qk8', 'tc_msb', 8, 4))


def variants_for_p(p):
    for suffix, threshold in THRESHOLDS.items():
        if type(p) in (float, int) and p == threshold:
            return tuple(f'{role}_{suffix}' for role in ROLES)
    raise ValueError('Formal TopP threshold must be .95, .90, or .85')


def variant_contract(variant_id):
    for p in THRESHOLDS.values():
        variants = variants_for_p(p)
        if variant_id in variants:
            return p, variants, ARITHMETIC[variants.index(variant_id)]
    raise ValueError('Unknown official variant: ' + str(variant_id))


def matrix_variants(variant_ids):
    ids = tuple(variant_ids)
    if not ids:
        raise ValueError('Formal matrix must contain four variants')
    p, variants, _ = variant_contract(ids[0])
    if ids != variants:
        raise ValueError('Formal matrix requires all four same-threshold variants in fixed order')
    return p, variants


def validate_formal_config(config):
    s = config['attention_slice']
    p, variants, arithmetic = variant_contract(s['variant_id'])
    method, estimator, _, _ = arithmetic
    # Keep replay independent of production quantizer/backend validators while
    # enforcing the complete resolved numerical configuration, not just its ID.
    expected_quantization = {
        'method': method, 'backend': 'torch_qdq', 'scope': 'prefill_and_decode',
        'cache_storage': 'original_float',
        'parameters': {} if method == 'fp_common' else {
            'group_axis': 'head_token', 'quantizer_input_dtype': 'float16',
            'source_input_cast': 'bf16_to_fp16', 'scale_dtype': 'source',
            'rounding': 'torch_round', 'zero_group': 'scale_one'},
        'source_revision': None if method == 'fp_common' else '7f7bfb00bcd26b2cef49e7783f51ef610e05abf7'}
    expected_algorithm = {name: {'type': 'none'} for name in
                          ('compressor', 'selector', 'weight_estimator', 'weight_pruner')}
    expected_algorithm.update(skip_first_two_layers=False, use_estimated_weights_in_attn=False,
                              attention_quantization=expected_quantization, attention_slice=s)
    if config != expected_algorithm:
        raise ValueError('Formal resolved attention protocol differs')
    expected_slice = {
        'variant_id': s['variant_id'], 'baseline_name': s.get('baseline_name'),
        'estimator': estimator, 'encoding': None if estimator == 'fp_full' else 'twos_complement',
        'q_prefix_bits': arithmetic[2], 'k_prefix_bits': arithmetic[3],
        'selection': {'kind': 'top_p', 'p': p, 'tie_break': 'key_position_ascending'},
        'final_scores': 'fp_full' if estimator == 'fp_full' else 'full_int8_qk_dq',
        'final_execution': 'selected_full_qk_recompute' if estimator == 'tc_msb' else 'reuse_full_scores',
        'scope': 'prefill_and_decode', 'diagnostics': 'targets_only'}
    if estimator != 'fp_full' and any(type(s.get(key)) is not int for key in ('q_prefix_bits', 'k_prefix_bits')):
        raise ValueError('Formal prefix widths must be integers')
    if (s != expected_slice or not isinstance(s.get('baseline_name'), str)
            or not s['baseline_name']):
        raise ValueError('Formal resolved slice protocol differs')
    if s['selection'] != {'kind': 'top_p', 'p': p, 'tie_break': 'key_position_ascending'}:
        raise ValueError('Formal variant ID/TopP threshold or tie rule mismatch')
    if (config['attention_quantization']['method'], s['estimator'],
            s['q_prefix_bits'], s['k_prefix_bits']) != arithmetic:
        raise ValueError('Formal variant arithmetic differs from SLICE-v2 matrix')
    return p, variants
