"""Explicit source-final aliases in the two frozen historical trace schemas."""
from .oracle import check

SOURCE_FINAL_ALIASES={'tc':frozenset({'full_codes'}),'sm':frozenset({'original_intflash_codes'})}

def validate_source_final(meta,role):
    check(role in SOURCE_FINAL_ALIASES,'unsupported historical control role')
    s=meta['attention_slice'];origin=meta.get('final_operand_origin')
    check(origin in SOURCE_FINAL_ALIASES[role],'historical '+role+' source-final alias differs')
    check(meta.get('selection_passes')==1 and s['final_scores']=='full_int8_qk_dq' and s['final_execution']=='selected_full_qk_recompute','historical original-source final arithmetic differs')
    check(s['estimator']==('tc_msb' if role=='tc' else 'format_msb'),'historical estimator role differs')
    if role=='sm':check(s['predictor_format']=='sm8','historical SM format role differs')
    return dict(canonical_final_operand_origin='original_intflash_codes',historical_final_operand_origin=origin,historical_schema_role=role)
