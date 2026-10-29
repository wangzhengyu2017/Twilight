"""Six full-format TopP jobs using the shared serial engine and semantic events."""
import argparse
import json
from pathlib import Path
import sys
from benchmark.attention_dense_format.run import run as serial_run
from .formal import read_matrix,VARIANTS
from .gate import validate_gate_report


def commands(args):
    matrix_path=Path(args.matrix).resolve();matrix,configs=read_matrix(matrix_path);root=Path(args.output_root).resolve();assets=Path(args.asset_dir).resolve();py=sys.executable
    if args.profile=='full' and not args.smoke_root:raise ValueError('Full requires --smoke-root')
    if args.variant and args.profile!='full':raise ValueError('Single-cell filtering requires full profile; smoke covers the complete six-job suite')
    if args.profile=='smoke':
        yield 'validate',[py,'-m','benchmark.attention_full_format.validate','--matrix',str(matrix_path),'--output',str(root/'validation/validation.json'),'--device','cuda']
        from benchmark.attention_format.regression import CONTROLS,CONTROL_CONFIG_ROOT
        for cfg,name in CONTROLS:
            yield 'regression-'+name,[py,'-m','benchmark.ppl','--model-path',args.model_path,'--tokens',str(assets/'tokens.pt'),'--assets',str(assets/'asset_manifest.json'),'--algo-config-path',str(CONTROL_CONFIG_ROOT/(cfg+'.json')),'--output-dir',str(root/'regression'/name),'--ppl-mode','windows','--max-tokens','4096']
        yield 'regression-compare',[py,'-m','benchmark.attention_format.regression','--runs-root',str(root/'regression'),'--output',str(root/'regression/report.json')]
    for row in matrix['variants']:
        if args.variant and row['variant_id']!=args.variant:continue
        vid=row['variant_id'];base=[py,'-m','benchmark.ppl','--model-path',args.model_path,'--tokens',str(assets/'tokens.pt'),'--assets',str(assets/'asset_manifest.json'),'--algo-config-path',str(matrix_path.parent/row['config'])]
        profiles=[('windows','windows',['--max-tokens','4096']),('decode-off','cached_decode',[]),('decode-on','cached_decode',['--trace-targets',str((matrix_path.parent/matrix['trace_targets']).resolve())])] if args.profile=='smoke' else [('windows-full','windows',['--full-format-gate-report',str(Path(args.smoke_root).resolve()/'gate.json')])]
        for suffix,mode,extra in profiles:yield vid+'-'+suffix,base+['--output-dir',str(root/(vid+'-'+suffix)),'--ppl-mode',mode]+extra
        if args.profile=='smoke':
            on=root/(vid+'-decode-on')
            yield vid+'-compare',[py,'-m','benchmark.attention_baseline.compare','--reference',str(root/(vid+'-decode-off')),'--candidate',str(on),'--output',str(on/'compare.json')]
            yield vid+'-replay',[py,'-m','benchmark.attention_full_format.replay','--run-dir',str(on),'--output',str(on/'replay.json'),'--device','cuda']
    if args.profile=='smoke':
        pair=[py,'-m','benchmark.attention_full_format.pair_sets','--matrix',str(matrix_path),'--runs-root',str(root),'--output-dir',str(root/'pairing')]
        for flag in ('dense_smoke_root','controls_manifest'):
            if getattr(args,flag):pair+=['--'+flag.replace('_','-'),str(Path(getattr(args,flag)).resolve())]
        yield 'pair-diagnostics',pair
        yield 'gate',[py,'-m','benchmark.attention_full_format.gate','--smoke-root',str(root),'--matrix',str(matrix_path),'--model',args.model_path,'--assets',str(assets/'asset_manifest.json'),'--review-report',str(Path(args.review_report).resolve()) if args.review_report else '<required-independent-review-report>','--output',str(root/'gate.json')]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['smoke','full'],required=True)
    for flag in ('matrix','model-path','asset-dir','output-root'):p.add_argument('--'+flag,required=True)
    for flag in ('smoke-root','review-report','events-path','controls-manifest','baselines-manifest','dense-smoke-root','dense-summary-manifest'):p.add_argument('--'+flag)
    p.add_argument('--variant',choices=list(VARIANTS));p.add_argument('--execute',action='store_true');a=p.parse_args()
    from .summarize import summarize
    def summary(matrix,root,out,profile,baselines):return summarize(matrix,root,out,profile,baselines,a.dense_summary_manifest)
    result=serial_run(a,plan_builder=commands,matrix_reader=read_matrix,gate_validator=validate_gate_report,summarizer=summary)
    print(json.dumps(result,indent=2));sys.exit(result['status']=='failed')
if __name__=='__main__':main()
