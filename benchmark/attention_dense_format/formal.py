"""Independent executable identities for the four approved dense measurements."""
from pathlib import Path
from benchmark.attention_baseline.common import load_json,load_config,sha256_file
from benchmark.attention_format.oracle import check

VARIANTS={'dense_sm8_q8k8':('sm8',8,8),'dense_mxint8_q8k8':('mxint8',8,8),'dense_mxint8q_mxfp4k_q8k4':('mxint8q_mxfp4k',8,4),'dense_mxfp4_q4k4':('mxfp4',4,4)}
TARGET_HASH='e1e7ca7b59def099214be6c1dd429cd6368cf4b61e0357e39df492fff985b8ad'
THRESHOLDS={'095':.95,'090':.9,'085':.85}


def validate_formal_config(config):
    check('attention_slice' not in config,'dense cannot contain a forward selector')
    s=config['attention_dense_format'];check(s.get('variant_id') in VARIANTS,'unknown dense variant')
    fmt,bq,bk=VARIANTS[s['variant_id']]
    expected=dict(variant_id=s['variant_id'],format=fmt,q_bits=bq,k_bits=bk,selection_mode='dense',selection_passes=0,final_scores='actual_decoded_format_qk',scope='prefill_and_decode',diagnostic_only_p=[.95,.9,.85],format_parameters={} if fmt=='sm8' else dict(block_size=32,endpoint_policy='symmetric_omit_min_negative'))
    check(s==expected and all(type(s[key]) is int for key in ('q_bits','k_bits','selection_passes')),'dense configuration differs')
    check(config['attention_quantization']['method']=='intflash_qk8','wrong source quantizer')
    return fmt,bq,bk


def read_matrix(path):
    path=Path(path).resolve();m=load_json(path);configs={}
    check(m.get('schema_version')=='dense-format-execution-v1','wrong executable matrix schema')
    check(m['selection_passes']==0 and m['expected_full_targets']==289075 and m['expected_full_windows']==282,'dense protocol differs')
    check(m['diagnostic_only_p']==[.95,.9,.85] and m['expected_heads_per_record']==32,'diagnostic contract differs')
    target=(path.parent/m['trace_targets']).resolve()
    check(m['trace_targets_sha256']==TARGET_HASH==sha256_file(target),'original trace target identity differs')
    for row in m['variants']:
        vid=row['variant_id'];check(vid in VARIANTS and vid not in configs,'unknown/duplicate matrix variant')
        cfgpath=(path.parent/row['config']).resolve();check(cfgpath.is_relative_to(path.parent),'unsafe config path')
        config=load_config(cfgpath);actual=validate_formal_config(config)
        check(actual==(row['format'],row['q_bits'],row['k_bits'])==VARIANTS[vid] and row['expected_traces']==6,'matrix/config differs')
        check(config['attention_dense_format']['variant_id']==vid,'matrix config variant differs');configs[vid]=config
    check(list(configs)==list(VARIANTS),'four dense jobs required in fixed order')
    check(len(m['specification_files'])>=3,'SPEC/MATRIX/REVIEW bindings absent')
    for name,digest in m['specification_files'].items():check(sha256_file(name)==digest,'approved contract changed: '+name)
    return m,configs
