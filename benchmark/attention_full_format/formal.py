"""Independent six-cell full-format TopP contract, separate from old MSB study."""
from pathlib import Path
from benchmark.attention_baseline.common import load_json,load_config,sha256_file
from benchmark.attention_dense_format.formal import THRESHOLDS,TARGET_HASH
from benchmark.attention_format.oracle import check

VARIANTS={f'{fmt}_q8k8_topp_p{suffix}':(fmt,p) for fmt in ('sm8','mxint8') for suffix,p in THRESHOLDS.items()}


def validate_formal_config(config):
    check(not any(k in config for k in ('attention_slice','attention_dense_format')),'full-format profile mixed with another executor')
    s=config['attention_full_format'];check(s.get('variant_id') in VARIANTS,'unknown full-format variant');fmt,p=VARIANTS[s['variant_id']]
    expected=dict(variant_id=s['variant_id'],format=fmt,q_bits=8,k_bits=8,selection_mode='top_p',selection_passes=1,selection=dict(kind='top_p',p=p,tie_break='key_position_ascending'),scope='prefill_and_decode',final_operand_profile='decoded_full_format',final_scores='reuse_decoded_full_format_logits',final_execution='reuse_full_scores',format_parameters={} if fmt=='sm8' else dict(block_size=32,endpoint_policy='symmetric_omit_min_negative'))
    check(s==expected and all(type(s[k]) is int for k in ('q_bits','k_bits','selection_passes')),'full-format configuration differs')
    check(config['attention_quantization']['method']=='intflash_qk8','wrong source quantization')
    return fmt,p


def read_matrix(path):
    path=Path(path).resolve();m=load_json(path);check(m['schema_version']=='full-format-topp-execution-v1','wrong full-format execution matrix')
    check((m['expected_full_targets'],m['expected_full_windows'],m['expected_heads_per_record'])==(289075,282,32),'frozen protocol differs')
    check(m['trace_targets_sha256']==TARGET_HASH==sha256_file((path.parent/m['trace_targets']).resolve()),'trace target JSON differs')
    configs={}
    for row in m['variants']:
        vid=row['variant_id'];check(vid in VARIANTS and vid not in configs,'unknown/duplicate full-format row');p=(path.parent/row['config']).resolve();check(p.is_relative_to(path.parent),'unsafe config path');cfg=load_config(p)
        check(validate_formal_config(cfg)==(row['format'],row['p'])==VARIANTS[vid] and row['q_bits']==row['k_bits']==8 and row['selection_passes']==1 and row['expected_traces']==6 and row['final_operand_profile']=='decoded_full_format','matrix/full-format contract differs')
        check(cfg['attention_full_format']['variant_id']==vid,'variant/config differs');configs[vid]=cfg
    check(list(configs)==list(VARIANTS),'six full-format jobs required in fixed order')
    check(len(m['specification_files'])>=5,'approved contract bindings missing')
    for name,digest in m['specification_files'].items():check(sha256_file(name)==digest,'approved contract changed '+name)
    return m,configs
