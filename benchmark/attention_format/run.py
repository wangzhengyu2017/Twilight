"""Serial format study runner. Dry run unless --execute is explicitly supplied."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmark.attention_baseline.common import ROOT, load_json, write_json, sha256_file


def commands(args):
    matrix_path=Path(args.matrix).resolve()
    from .gate import read_matrix
    matrix,configs=read_matrix(matrix_path)
    root=Path(args.output_root).resolve(); assets=Path(args.asset_dir).resolve()
    rows=[r for r in matrix['variants'] if r['variant_id'] in configs]
    if matrix.get('execution_phase','initial')!=args.phase:raise ValueError('Runner/matrix execution phase mismatch')
    if args.phase=='conditional':
        decision=load_json(args.decision) if args.decision else {}
        if decision.get('status')!='complete' or decision.get('activate_mxfp4') is not True:
            raise ValueError('Conditional execution requires a complete affirmative decision')
        activation=matrix['conditional_activation']
        if Path(args.decision).resolve()!=Path(activation['decision_path']).resolve() or sha256_file(args.decision)!=activation['decision_sha256']:raise ValueError('Runner/matrix conditional decision mismatch')
    py=sys.executable
    yield 'validate',[py,'-m','benchmark.attention_format.validate','--matrix',str(matrix_path),'--output',str(root/'validation/validation.json'),'--device','cuda']
    if args.profile=='full' and not args.smoke_root:raise ValueError('Full requires --smoke-root')
    if args.profile=='smoke':
        from .regression import CONTROLS,CONTROL_CONFIG_ROOT
        for cfg,name in CONTROLS:
            yield 'regression-'+name,[py,'-m','benchmark.ppl','--model-path',args.model_path,'--tokens',str(assets/'tokens.pt'),'--assets',str(assets/'asset_manifest.json'),'--algo-config-path',str(CONTROL_CONFIG_ROOT/(cfg+'.json')),'--output-dir',str(root/'regression'/name),'--ppl-mode','windows','--max-tokens','4096']
        yield 'regression-compare',[py,'-m','benchmark.attention_format.regression','--runs-root',str(root/'regression'),'--output',str(root/'regression/report.json')]
    for row in rows:
        vid=row['variant_id']; cfg=matrix_path.parent/row['config']
        base=[py,'-m','benchmark.ppl','--model-path',args.model_path,'--tokens',str(assets/'tokens.pt'),'--assets',str(assets/'asset_manifest.json'),'--algo-config-path',str(cfg)]
        profiles=[('windows','windows',['--max-tokens','4096']),('decode-off','cached_decode',[]),('decode-on','cached_decode',['--trace-targets',str(matrix_path.parent/matrix['trace_targets'])])] if args.profile=='smoke' else [('windows-full','windows',['--slice-gate-report',str(Path(args.smoke_root).resolve()/'gate.json')])]
        for suffix,mode,extra in profiles:
            yield vid+'-'+suffix,base+['--output-dir',str(root/(vid+'-'+suffix)),'--ppl-mode',mode]+extra
        if args.profile=='smoke':
            on=root/(vid+'-decode-on')
            yield vid+'-compare',[py,'-m','benchmark.attention_baseline.compare','--reference',str(root/(vid+'-decode-off')),'--candidate',str(on),'--output',str(on/'compare.json')]
            yield vid+'-replay',[py,'-m','benchmark.attention_format.replay','--run-dir',str(on),'--output',str(on/'replay.json'),'--device','cuda']
            old=Path(args.controls_manifest).resolve() if args.controls_manifest else None
            if old:
                refs=load_json(old)[vid[-3:]]
                yield vid+'-pair',[py,'-m','benchmark.attention_format.pair_sets','--candidate-run',str(on),'--int-run',refs['int_run'],'--fp-run',refs['fp_run'],'--tc-run',refs['q'+str(row['q_total_prefix_bits'])+'_run'],'--output-dir',str(root/'pairing'/vid)]
    if args.profile=='smoke':
        yield 'gate',[py,'-m','benchmark.attention_format.gate','--smoke-root',str(root),'--matrix',str(matrix_path),'--model',args.model_path,'--assets',str(assets/'asset_manifest.json'),'--output',str(root/'gate.json')]


def run(args):
    plan=list(commands(args))
    if not args.execute:return {'status':'dry_run','commands':[{'name':n,'argv':v} for n,v in plan]}
    if args.profile=='smoke' and not args.controls_manifest:raise ValueError('Smoke execution requires --controls-manifest for required cross-run pairing')
    root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=False);(root/'command_logs').mkdir()
    write_json(root/'invocation.json',dict(vars(args),matrix_sha256=sha256_file(args.matrix)))
    if args.profile=='full':
        gate=Path(args.smoke_root).resolve()/'gate.json'
        write_json(root/'gate_reference.json',dict(path=str(gate),sha256=sha256_file(gate)))
    status={'status':'running','completed_commands':[],'failed_commands':[]}
    write_json(root/'status.json',status)
    for i,(name,argv) in enumerate(plan):
        if name=='gate' and status['failed_commands']:continue
        log=root/'command_logs'/f'{i:03d}-{name}.log'
        with (root/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=argv,log=str(log),started_unix=time.time()))+'\n')
        print(json.dumps(dict(stage=name,log=str(log))),flush=True)
        with log.open('w') as f:rc=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy()).returncode
        status['failed_commands' if rc else 'completed_commands'].append(dict(name=name,returncode=rc,log=str(log)))
        write_json(root/'status.json',status)
        if rc and name=='validate':break
    status['status']='failed' if status['failed_commands'] else 'completed_available_source_blocked'
    write_json(root/'status.json',status);return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['smoke','full'],required=True)
    p.add_argument('--phase',choices=['initial','conditional'],default='initial')
    for field in ['matrix','model-path','asset-dir','output-root']:p.add_argument('--'+field,required=True)
    for field in ['smoke-root','controls-manifest','decision']:p.add_argument('--'+field)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();result=run(a);print(json.dumps(result,indent=2));sys.exit(result['status']=='failed')
if __name__=='__main__':main()
