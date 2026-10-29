"""Freeze reviewed design, exact ten configs and capture targets before smoke."""
import argparse
import json
import shutil
from pathlib import Path
from benchmark.attention_baseline.common import write_json,sha256_file
from .formal import VARIANTS,expected_config,SPEC,DESIGN_MATRIX,TARGET_HASH,read_matrix

def freeze(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);(out/'configs').mkdir()
    spec=Path('/output/project/experiments/stp-format-study/rs-expanded-spec');prep=Path('/output/project/experiments/stp-rs-study/preparation')
    quant=dict(method='intflash_qk8',backend='torch_qdq',scope='prefill_and_decode',cache_storage='original_float',parameters=dict(group_axis='head_token',quantizer_input_dtype='float16',source_input_cast='bf16_to_fp16',scale_dtype='source',rounding='torch_round',zero_group='scale_one'),source_revision='7f7bfb00bcd26b2cef49e7783f51ef610e05abf7')
    rows=[]
    for vid in VARIANTS:
        cfg={key:{'type':'none'} for key in ('compressor','selector','weight_estimator','weight_pruner')};cfg.update(skip_first_two_layers=False,use_estimated_weights_in_attn=False,attention_quantization=quant,attention_rs=expected_config(vid))
        file=out/'configs'/(vid+'.json');write_json(file,cfg);rows.append(dict(variant_id=vid,config=str(file.relative_to(out)),config_sha256=sha256_file(file),expected_traces=6))
    target=Path(__file__).resolve().parents[1]/'attention_format/trace_targets.json';shutil.copyfile(target,out/'trace_targets.json')
    bindings=[spec/name for name in ('SPEC.md','MATRIX.json','REVIEW.md','SOURCE_EVIDENCE.md','INDEPENDENT_REVIEW_CHECKS.json')]+[prep/'TRACE_ADDENDUM.md']
    matrix=dict(schema_version='rs-execution-v1',approved_spec_sha256=SPEC,approved_design_matrix_sha256=DESIGN_MATRIX,expected_full_targets=289075,expected_full_windows=282,expected_trace_records_total=60,expected_heads_per_record=32,expected_execution_rows=1920,expected_diagnostic_rows=5760,
        trace_targets='trace_targets.json',trace_targets_sha256=TARGET_HASH,specification_files={str(p):sha256_file(p) for p in bindings},variants=rows)
    write_json(out/'MATRIX.json',matrix);read_matrix(out/'MATRIX.json');return matrix

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(freeze(a.output),indent=2))
if __name__=='__main__':main()
