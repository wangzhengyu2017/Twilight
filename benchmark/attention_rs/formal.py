"""Independently fixed executable RS matrix and reviewed-contract bindings."""
from pathlib import Path
from benchmark.attention_baseline.common import load_json,load_config,sha256_file
from .oracle import check
PROFILE='rsint8_native_source_scale'
PAPER='7c2e22b91543435aeb83f230ac6579a07bcfa8f43a138454b77e18149d2609a7'
SPEC='1b3303d6db0fba09d4470239c97b798f0be6df33131d7d8cd7efa169284614fa'
DESIGN_MATRIX='d0d5f46e068dcb14bcd2e47e8491400898cc1c36c19f5f47015e10b224f4a38c'
TARGET_HASH='e1e7ca7b59def099214be6c1dd429cd6368cf4b61e0357e39df492fff985b8ad'
VARIANTS={'rs_dense_q8k8':('dense_full',8,8,None)}
for mode,shape,q,k in [('full_topp','q8k8',8,8),('msb_topp','q8k4',8,4),('msb_topp','q4k4',4,4)]:
    for label,p in [('095',.95),('090',.9),('085',.85)]:VARIANTS['rs_'+('full' if mode=='full_topp' else 'msb')+'_'+shape+'_topp_p'+label]=(mode,q,k,p)

def expected_config(vid):
    mode,q,k,p=VARIANTS[vid];native=mode!='msb_topp'
    return dict(variant_id=vid,mode=mode,predictor_format=PROFILE,q_bits=q,k_bits=k,selection=dict(kind='none' if p is None else 'top_p',p=p,tie_break='key_position_ascending' if p is not None else None),selection_passes=0 if p is None else 1,
        final_operand_origin='rs_full_decoded' if native else 'original_intflash_codes',final_scores='rs_full_dq' if native else 'full_int8_qk_dq',final_execution='dense_rs_full' if mode=='dense_full' else ('reuse_full_rs_scores' if native else 'selected_full_qk_recompute'),scope='prefill_and_decode',diagnostics='targets_only',
        format_parameters=dict(native_range=[-136,119],source_conversion='integer_clamp_at_source_scale',scale_policy='reuse_original_head_token',zero_point=0,slice_bits=4,packed_layout='post_bsu_H4_L4',paper_sha256=PAPER,codec_version='bada-native-source-scale-v1'))

def validate_formal_config(config):
    check(not any(key in config for key in ('attention_slice','attention_dense_format','attention_full_format')),'RS must be the only attention intervention')
    s=config['attention_rs'];vid=s.get('variant_id');check(vid in VARIANTS,'unknown RS variant');check(s==expected_config(vid),'RS fixed config differs')
    check(all(type(s[key]) is int for key in ('q_bits','k_bits','selection_passes')),'RS integer config types')
    check(config['attention_quantization']['method']=='intflash_qk8','RS wrong source quantizer')
    PARAMETERS={'group_axis':'head_token','quantizer_input_dtype':'float16','source_input_cast':'bf16_to_fp16','scale_dtype':'source','rounding':'torch_round','zero_group':'scale_one'}
    quant=config['attention_quantization'];check(quant==dict(method='intflash_qk8',backend='torch_qdq',scope='prefill_and_decode',cache_storage='original_float',parameters=PARAMETERS,source_revision='7f7bfb00bcd26b2cef49e7783f51ef610e05abf7'),'source contract changed')
    return VARIANTS[vid]

def read_matrix(path):
    path=Path(path).resolve();m=load_json(path);check(m['schema_version']=='rs-execution-v1','RS matrix schema')
    check((m['expected_trace_records_total'],m['expected_heads_per_record'],m['expected_full_targets'],m['expected_full_windows'])==(60,32,289075,282),'RS coverage differs')
    check(m['approved_spec_sha256']==SPEC and m['approved_design_matrix_sha256']==DESIGN_MATRIX,'reviewed design binding')
    targets=(path.parent/m['trace_targets']).resolve();check(sha256_file(targets)==m['trace_targets_sha256']==TARGET_HASH,'RS trace target changed')
    for name,digest in m['specification_files'].items():check(sha256_file(name)==digest,'RS contract changed '+name)
    check(SPEC in m['specification_files'].values() and DESIGN_MATRIX in m['specification_files'].values(),'both reviewed files must be bound')
    configs={}
    for row in m['variants']:
        vid=row['variant_id'];check(vid in VARIANTS and vid not in configs,'unknown/duplicate RS matrix row');file=(path.parent/row['config']).resolve();check(file.is_relative_to(path.parent),'unsafe RS config path')
        check(sha256_file(file)==row['config_sha256'],'RS config bytes changed');config=load_config(file);check(validate_formal_config(config)==VARIANTS[vid] and config['attention_rs']['variant_id']==vid,'RS row/config mismatch');check(row['expected_traces']==6,'RS six actual traces required');configs[vid]=config
    check(list(configs)==list(VARIANTS),'RS fixed ten job order');return m,configs
