"""Strict format-study identities, independent of backend validation."""
import re
from benchmark.attention_slice.formal import validate_formal_config as validate_slice

THRESHOLDS = {'095': .95, '090': .9, '085': .85}
FORMATS = ('sm8', 'mxint8_tc4', 'mxfp4', 'mxint8q_mxfp4k')

def variant_contract(vid):
    match = re.fullmatch(r'format_(sm8|mxint8_tc4|mxfp4|mxint8q_mxfp4k)_q([48])k4_topp_p(095|090|085)', vid)
    if not match:
        raise ValueError('Unknown or source-blocked format variant: '+str(vid))
    fmt, bits, suffix = match.groups(); bits = int(bits)
    if (fmt == 'mxfp4' and bits != 4) or (fmt == 'mxint8q_mxfp4k' and bits != 8):
        raise ValueError('Native/mixed FP4 shape mismatch')
    return fmt, bits, 4, THRESHOLDS[suffix]

def validate_formal_config(config):
    s = config['attention_slice']; fmt,bq,bk,p = variant_contract(s['variant_id'])
    params = {} if fmt == 'sm8' else {'block_size':32}
    if fmt in ('mxint8_tc4','mxint8q_mxfp4k'):
        params['endpoint_policy']='symmetric_omit_min_negative'
    expected = dict(variant_id=s['variant_id'],baseline_name=s.get('baseline_name'),estimator='format_msb',
        encoding='sign_magnitude' if fmt=='sm8' else 'microscaling', q_prefix_bits=bq,k_prefix_bits=bk,
        selection={'kind':'top_p','p':p,'tie_break':'key_position_ascending'},final_scores='full_int8_qk_dq',
        final_execution='selected_full_qk_recompute',scope='prefill_and_decode',diagnostics='targets_only',
        predictor_format=fmt,format_parameters=params)
    if s != expected or type(s['q_prefix_bits']) is not int or type(s['k_prefix_bits']) is not int:
        raise ValueError('Format configuration differs from frozen contract')
    # Reuse the strict unchanged source protocol through an independently
    # constructed TC control; production validation is deliberately not imported.
    control = dict(config); control_slice = {k:v for k,v in expected.items() if k not in ('predictor_format','format_parameters')}
    control_slice.update(variant_id=f'msb_q{bq}k4_topp_p{next(k for k,v in THRESHOLDS.items() if v==p)}',estimator='tc_msb',encoding='twos_complement')
    control['attention_slice']=control_slice; validate_slice(control)
    return fmt,bq,bk,p
