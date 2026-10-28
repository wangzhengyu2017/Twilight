"""Exact preserved-control regression across the format source change."""
import argparse,json
from pathlib import Path
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,semantic_config

CONTROL_CONFIG_ROOT=Path('/output/project/SPEC/remote-gpu-platform-slice-SLICE-v2/configs')
REFERENCE_ROOT=Path('/output/project/experiments/stp-slice/20261028-v2-smoke')
CONTROLS=[('fp_dense_control','fp_common-windows-control'),('int_dense_control','intflash_qk8-windows-control')]+[(v,v+'-windows') for v in ('fp_topp_p095','int_topp_p095','msb_q4k4_topp_p095','msb_q8k4_topp_p095')]

def compare(root):
    root=Path(root).resolve();files={};reports=[];identities=set()
    for cfg,name in CONTROLS:
        reference=REFERENCE_ROOT/name;candidate=root/name
        am,bm=[load_json(r/'metrics.json') for r in (reference,candidate)]
        for r in (reference,candidate):
            if load_json(r/'status.json')['status']!='complete':raise ValueError('Incomplete regression '+str(r))
        keys=('protocol_id','profile','mode','method','backend','model_dtype','tokens_sha256','subset_sha256','scoring_ranges_sha256','num_input_tokens','num_targets','sum_nll','ppl','environment_sha256','asset_manifest_sha256','semantic_config_sha256')
        for key in keys:
            if am[key]!=bm[key]:raise ValueError('Regression metrics mismatch '+name+' '+key)
        for filename in ('asset_manifest.json','method_manifest.json','quantization_diagnostics.json','coverage.json'):
            if load_json(reference/filename)!=load_json(candidate/filename):raise ValueError('Regression artifact mismatch '+name+' '+filename)
        if semantic_config(load_json(reference/'resolved_config.json'))!=semantic_config(load_json(candidate/'resolved_config.json')):raise ValueError('Regression config mismatch')
        ar,br=[[json.loads(x) for x in (r/'window_metrics.jsonl').read_text().splitlines() if x.strip()] for r in (reference,candidate)]
        if not ar or ar!=br or not all(row.get('logits_sha256') for row in ar):raise ValueError('Regression exact row/logits/NLL mismatch '+name)
        for r in (reference,candidate):
            for filename in ('status.json','metrics.json','asset_manifest.json','method_manifest.json','quantization_diagnostics.json','coverage.json','resolved_config.json','window_metrics.jsonl','code_manifest.json','environment.json'):
                p=r/filename;files[str(p)]=sha256_file(p)
        identities.add((bm['code_sha256'],bm['environment_sha256']))
        reports.append(dict(control=name,reference=str(reference),candidate=str(candidate),num_rows=len(ar),exact_logits_and_nll=True))
    if len(identities)!=1:raise ValueError('Regression candidate identity mismatch')
    code,env=identities.pop()
    return dict(status='complete',passed=True,controls=reports,source_files=files,candidate_code_sha256=code,candidate_environment_sha256=env)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:write_json(a.output,compare(a.runs_root))
    except Exception as error:write_json(a.output,dict(status='failed',passed=False,reason=str(error)));raise
if __name__=='__main__':main()
