"""Serial ten-job native RS execution; dry run by default, no duplicate monitoring loop."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmark.attention_baseline.common import ROOT,load_json,write_json,sha256_file,code_provenance,canonical_hash
from .formal import read_matrix
from .oracle import check

REGISTERED_EVENTS=Path('/output/project/experiments/stp-rs-study/events.jsonl')
SEMANTIC_KINDS={'experiment_started','experiment_failed','validation_failed','summary_complete'}


def commands(args):
    matrix_path=Path(args.matrix).resolve();matrix,configs=read_matrix(matrix_path);root=Path(args.output_root).resolve();assets=Path(args.asset_dir).resolve();py=sys.executable
    check(Path(args.matrix).resolve().parent==Path(args.handoff_dir).resolve(),'--handoff-dir must own exact matrix')
    if args.profile=='full':check(bool(args.smoke_root),'full requires --smoke-root')
    if args.profile=='smoke':
        yield 'validate',[py,'-m','benchmark.attention_rs.validate','--matrix',str(matrix_path),'--output',str(root/'validation/validation.json'),'--device','cuda']
        from benchmark.attention_format.regression import CONTROLS,CONTROL_CONFIG_ROOT
        for cfg,name in CONTROLS:
            yield 'regression-'+name,[py,'-m','benchmark.ppl','--model-path',args.model_path,'--tokens',str(assets/'tokens.pt'),'--assets',str(assets/'asset_manifest.json'),'--algo-config-path',str(CONTROL_CONFIG_ROOT/(cfg+'.json')),'--output-dir',str(root/'regression'/name),'--ppl-mode','windows','--max-tokens','4096']
        yield 'regression-compare',[py,'-m','benchmark.attention_format.regression','--runs-root',str(root/'regression'),'--output',str(root/'regression/report.json')]
    variant=getattr(args,'variant',None)
    if variant is not None:check(variant in configs,'unknown RS selected variant')
    for row in matrix['variants']:
        if variant is not None and row['variant_id']!=variant:continue
        vid=row['variant_id'];base=[py,'-m','benchmark.ppl','--model-path',args.model_path,'--tokens',str(assets/'tokens.pt'),'--assets',str(assets/'asset_manifest.json'),'--algo-config-path',str(matrix_path.parent/row['config'])]
        profiles=[('windows','windows',['--max-tokens','4096']),('decode-off','cached_decode',[]),('decode-on','cached_decode',['--trace-targets',str((matrix_path.parent/matrix['trace_targets']).resolve())])] if args.profile=='smoke' else [('windows-full','windows',['--rs-gate-report',str(Path(args.smoke_root).resolve()/'gate.json')])]
        for suffix,mode,extra in profiles:yield vid+'-'+suffix,base+['--output-dir',str(root/(vid+'-'+suffix)),'--ppl-mode',mode]+extra
        if args.profile=='smoke':
            on=root/(vid+'-decode-on')
            yield vid+'-compare',[py,'-m','benchmark.attention_baseline.compare','--reference',str(root/(vid+'-decode-off')),'--candidate',str(on),'--output',str(on/'compare.json')]
            yield vid+'-replay',[py,'-m','benchmark.attention_rs.replay','--run-dir',str(on),'--output',str(on/'replay.json'),'--device','cuda']
    if args.profile=='smoke' and getattr(args,'variant',None) is None:
        pair=[py,'-m','benchmark.attention_rs.pair_sets','--matrix',str(matrix_path),'--runs-root',str(root),'--output-dir',str(root/'pairing')]
        check(bool(args.controls_manifest),'RS pairing requires --controls-manifest with all old TC/SM/dense controls')
        if args.controls_manifest:pair+=['--controls-manifest',str(Path(args.controls_manifest).resolve())]
        yield 'pair-diagnostics',pair
        yield 'gate',[py,'-m','benchmark.attention_rs.gate','--smoke-root',str(root),'--matrix',str(matrix_path),'--model',args.model_path,'--assets',str(assets/'asset_manifest.json'),'--review-report',str(Path(args.review_report).resolve()) if args.review_report else '<required-independent-review-report>','--output',str(root/'gate.json')]


def emit(args,kind,**fields):
    row=dict(kind=kind,unix=time.time(),study='stp-rs-study',profile=args.profile,runner_pid=os.getpid(),**fields)
    print(json.dumps(row,sort_keys=True,allow_nan=False),flush=True)
    if args.events_path and kind in SEMANTIC_KINDS:
        path=Path(args.events_path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')


def run(args,*,plan_builder=commands):
    if getattr(args,'baseline_root',None):
        baseline=Path(args.baseline_root).resolve()
        if not args.controls_manifest:args.controls_manifest=str(baseline/'controls_manifest.json')
        if not args.baselines_manifest:args.baselines_manifest=str(baseline/'baselines_manifest.json')
    if args.execute and args.profile=='full':check(bool(args.baselines_manifest),'RS full reporting requires all frozen PPL anchors')
    plan=list(plan_builder(args))
    if not args.execute:return dict(status='dry_run',commands=[dict(name=n,argv=v) for n,v in plan])
    check(bool(args.events_path) and Path(args.events_path).resolve()==REGISTERED_EVENTS,'execution requires registered --events-path '+str(REGISTERED_EVENTS))
    check(bool(args.review_report),'execution requires --review-report')
    review=load_json(args.review_report);code=code_provenance()
    check(review.get('status')=='complete' and review.get('passed') is True and review.get('independent_reviewer') is True and review.get('code_sha256')==code['code_sha256'],'independent implementation acceptance missing/stale')
    matrix,configs=read_matrix(args.matrix)
    check(review.get('matrix_sha256')==sha256_file(args.matrix),'independent RS matrix acceptance missing/stale')
    if args.profile=='full':
        from .gate import validate_gate_report
        for cfg in configs.values():validate_gate_report(Path(args.smoke_root).resolve()/'gate.json',args.model_path,Path(args.asset_dir)/'asset_manifest.json',cfg)
    root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=False);(root/'command_logs').mkdir()
    code_provenance(root);write_json(root/'invocation.json',dict(vars(args),matrix_sha256=sha256_file(args.matrix),review_report_sha256=sha256_file(args.review_report)))
    if args.profile=='full':
        gate=Path(args.smoke_root).resolve()/'gate.json';write_json(root/'gate_reference.json',dict(path=str(gate),sha256=sha256_file(gate)))
    status=dict(status='running',profile=args.profile,completed_commands=[],failed_commands=[],pending_commands=[n for n,_ in plan]);write_json(root/'status.json',status)
    emit(args,'batch_started',run=str(root),argv=sys.argv,source_code_sha256=code['code_sha256'])
    child=None
    try:
        for i,(name,argv) in enumerate(plan):
            check(code_provenance()['code_sha256']==code['code_sha256'],'source changed during batch')
            log=root/'command_logs'/f'{i:03d}-{name}.log';started=time.time()
            is_experiment=argv[1:3]==['-m','benchmark.ppl'] and not name.startswith('regression-')
            with log.open('w') as stream:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
                event=dict(name=name,argv=argv,pid=child.pid,run=str(root),log=str(log),started_unix=started)
                with (root/'commands.jsonl').open('a') as history:history.write(json.dumps(event,sort_keys=True)+'\n')
                status['current_command']=event;write_json(root/'status.json',status);emit(args,'command_started',**event)
                if is_experiment:
                    emit(args,'experiment_started',name=name,argv=argv,pid=child.pid,run=argv[argv.index('--output-dir')+1],batch_root=str(root),log=str(log),source_code_sha256=code['code_sha256'])
                rc=child.wait()
            completed=dict(name=name,returncode=rc,pid=child.pid,log=str(log),elapsed_seconds=time.time()-started)
            status['pending_commands'].remove(name);status['failed_commands' if rc else 'completed_commands'].append(completed);status.pop('current_command',None)
            write_json(root/'status.json',status);emit(args,'command_failed' if rc else 'command_completed',run=str(root),**completed)
            if rc:
                emit(args,'experiment_failed' if is_experiment else 'validation_failed',run=argv[argv.index('--output-dir')+1] if is_experiment else str(root),argv=argv,**completed)
                break
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        failure=dict(name='runner',returncode=1,reason=str(error),exception=type(error).__name__)
        status['failed_commands'].append(failure);status.pop('current_command',None)
        emit(args,'experiment_failed',run=str(root),failure_scope='runner',**failure)
    status['status']='failed' if status['failed_commands'] else 'complete';write_json(root/'status.json',status)
    # Missing or failed measurements remain null even when a batch stops early.
    from .summarize import summarize
    summarize(args.matrix,root,root/'summary',args.profile,args.baselines_manifest)
    emit(args,'summary_complete',run=str(root),result_table=str(root/'summary/BASELINE.csv'),summary_manifest=str(root/'summary/summary_manifest.json'),batch_status=status['status'],independently_audited=False)
    emit(args,'batch_failed' if status['failed_commands'] else 'batch_completed',run=str(root),completed_commands=len(status['completed_commands']),failed_commands=len(status['failed_commands']))
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['smoke','full'],required=True)
    for flag in ('matrix','handoff-dir','model-path','asset-dir','output-root'):p.add_argument('--'+flag,required=True)
    for flag in ('smoke-root','review-report','events-path','controls-manifest','baselines-manifest','baseline-root','variant'):p.add_argument('--'+flag)
    p.add_argument('--execute',action='store_true');a=p.parse_args();result=run(a);print(json.dumps(result,indent=2));sys.exit(result['status']=='failed')
if __name__=='__main__':main()
