"""CPU-only selector, orchestration, null-summary and rejection checks."""
from argparse import Namespace
from pathlib import Path
import argparse
import tempfile
from benchmark.attention_baseline.common import load_config,write_json
from .formal import read_matrix
from .oracle import check
from .selector_adapter import resolve_rs
from .run import commands
from .summarize import summarize

def contract_check(matrix,controls,baselines):
    m,configs=read_matrix(matrix);resolutions=[]
    for vid,cfg in configs.items():
        s=cfg['attention_rs'];args=Namespace(encoding='rsint8',msb='on' if s['mode']=='msb_topp' else 'off',top_p='off' if s['mode']=='dense_full' else 'on',p=s['selection']['p'],final_operands='auto',q_bits=s['q_bits'],k_bits=s['k_bits'])
        resolved=resolve_rs(args,matrix);check(resolved['variant_id']==vid and resolved['config']==cfg,'shared-selector-compatible RS resolution');resolutions.append(vid)
    for profile in ('smoke','full'):
        args=Namespace(profile=profile,matrix=str(matrix),handoff_dir=str(Path(matrix).parent),model_path='/hyperai/input/input0',asset_dir='/output/project/assets/tq-wt2-v1',output_root='/tmp/rs-not-executed',smoke_root='/tmp/rs-required-smoke',review_report=None,controls_manifest=str(controls),baselines_manifest=str(baselines),variant=None)
        plan=list(commands(args));window=[(n,a) for n,a in plan if n.startswith('rs_') and n.endswith('-windows' if profile=='smoke' else '-windows-full')];check(len(window)==10,'all ten measured window jobs required')
        if profile=='smoke':check(sum(name.endswith('-decode-on') for name,_ in plan)==10 and sum(name.endswith('-replay') for name,_ in plan)==10,'ten capture and replay companions')
        else:check(all('--rs-gate-report' in argv for _,argv in window),'full RS gate required')
    with tempfile.TemporaryDirectory(prefix='rs-contract-',dir='/tmp') as temporary:
        report=summarize(matrix,Path(temporary)/'missing-runs',Path(temporary)/'summary','full',baselines)
        check(len(report['rows'])==10 and all(row['ppl'] is None and row['sum_nll'] is None and row['num_targets'] is None and row['executed'] is False and row['verified'] is False for row in report['rows']),'unexecuted values must stay null')
        check(len(report['historical_anchors'])==14 and all(not row['verified'] for row in report['historical_anchors'].values()),'read-only old anchors must not inherit RS verification')
    from benchmark.attention_format.regression import CONTROLS,CONTROL_CONFIG_ROOT
    for name,_ in CONTROLS:load_config(CONTROL_CONFIG_ROOT/(name+'.json'))
    return dict(status='passed',scope='CPU_contract_only_no_GPU_or_model_execution',selector_variants=resolutions,window_jobs_per_profile=10,capture_jobs=10,unexecuted_rows_remain_null=True,preserved_configs_read=len(CONTROLS))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','controls-manifest','baselines-manifest','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:report=contract_check(a.matrix,a.controls_manifest,a.baselines_manifest)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
