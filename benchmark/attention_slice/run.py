"""Execute the complete SLICE-v2 matrix using the existing frozen-token scorer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from benchmark.attention_baseline.common import ROOT, load_json, write_json, sha256_file, configure_runtime
from benchmark.attention_slice.gate import read_matrix, build_gate, validate_gate_report


def run(args):
    # Gate identity includes deterministic/TF32 settings, also in this parent
    # process before it validates evidence and spawns the scoring children.
    configure_runtime()
    root=Path(args.output_root).resolve()
    if root.exists(): raise FileExistsError(f'Refusing to overwrite {root}')
    root.mkdir(parents=True);logs=root/'command_logs';logs.mkdir()
    matrix_path=Path(args.matrix).resolve();handoff=Path(args.handoff_dir).resolve()
    matrix,configs=read_matrix(matrix_path)
    if matrix_path.parent!=handoff: raise ValueError('Matrix must belong to the specified handoff')
    asset_dir=Path(args.asset_dir).resolve();assets=asset_dir/'asset_manifest.json'
    write_json(root/'invocation.json',dict(vars(args),matrix_sha256=sha256_file(matrix_path),python=sys.executable))
    status={'status':'running','profile':args.profile,'completed_commands':0,'failed_commands':[]}
    write_json(root/'status.json',status)
    number=0
    def command(name,argv,optional=False):
        nonlocal number
        number+=1;path=logs/f'{number:02d}-{name}.log'
        with (root/'commands.jsonl').open('a') as f:
            f.write(json.dumps({'name':name,'argv':argv,'cwd':str(ROOT),'log':str(path),'started_unix':time.time()})+'\n')
        print(json.dumps({'stage':name,'log':str(path)}),flush=True)
        with path.open('w') as out:
            completed=subprocess.run(argv,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
        if completed.returncode:
            status['failed_commands'].append({'name':name,'returncode':completed.returncode,'log':str(path)})
            write_json(root/'status.json',status)
            if not optional: raise RuntimeError(f'{name} failed ({completed.returncode}), see {path}')
            return False
        status['completed_commands']+=1;write_json(root/'status.json',status)
        return True
    def ppl(vid,config,suffix,mode,max_tokens=None,trace=False,gate=None,optional=False):
        argv=[sys.executable,'-m','benchmark.ppl','--model-path',args.model_path,
            '--tokens',str(asset_dir/'tokens.pt'),'--assets',str(assets),'--algo-config-path',str(config),
            '--output-dir',str(root/f'{vid}-{suffix}'),'--ppl-mode',mode]
        if max_tokens:argv+=['--max-tokens',str(max_tokens)]
        if trace:argv+=['--trace-targets',str(handoff/matrix['trace_targets'])]
        if gate:argv+=['--slice-gate-report',str(gate)]
        return command(f'{vid}-{suffix}',argv,optional)
    try:
        command('verify-handoff',[sys.executable,str(handoff/'verify_handoff.py')])
        current_assets=load_json(assets)
        imported=[]
        for method in ('fp_common','intflash_qk8'):
            old=Path(args.baseline_root).resolve()/'20261027-v5-full'/f'{method}-windows-full'
            if not old.is_dir():
                imported.append({'method':method,'status':'missing','path':str(old)})
                continue
            old_status=load_json(old/'status.json'); old_metrics=load_json(old/'metrics.json')
            if load_json(old/'asset_manifest.json')!=current_assets or old_status.get('status')!='complete':
                raise ValueError('Historical dense baseline assets/status could not be verified')
            imported.append({'method':method,'status':'verified_historical_anchor','path':str(old),
                'sum_nll':old_metrics['sum_nll'],'num_targets':old_metrics['num_targets'],'ppl':old_metrics['ppl'],
                'files':{str(old/name):sha256_file(old/name) for name in ('status.json','metrics.json','asset_manifest.json','environment.json','code_manifest.json')}})
        write_json(root/'baseline_import.json',{'source':str(Path(args.baseline_root).resolve()),'runs':imported,
            'scope':'Historical dense quality anchors only; fresh controls and all four TopP experiments run independently.'})
        if args.profile=='smoke':
            command('validate',[sys.executable,'-m','benchmark.attention_slice.validate','--output-dir',str(root/'validation'),'--config-dir',str(handoff/'configs')])
            for method,filename in [('fp_common','fp_dense_control.json'),('intflash_qk8','int_dense_control.json')]:
                ppl(method,handoff/'configs'/filename,'windows-control','windows',4096)
            for row in matrix['variants']:
                vid=row['variant_id'];cfg=handoff/row['config']
                ppl(vid,cfg,'windows','windows',4096)
                ppl(vid,cfg,'decode-off','cached_decode')
                ppl(vid,cfg,'decode-on','cached_decode',trace=True)
                on=root/f'{vid}-decode-on'
                command(f'{vid}-compare',[sys.executable,'-m','benchmark.attention_baseline.compare','--reference',str(root/f'{vid}-decode-off'),'--candidate',str(on),'--output',str(on/'compare.json')])
                command(f'{vid}-replay',[sys.executable,'-m','benchmark.attention_slice.replay','--run-dir',str(on),'--output',str(on/'replay.json')])
            command('pair-sets',[sys.executable,'-m','benchmark.attention_slice.pair_sets','--matrix',str(matrix_path),'--runs-root',str(root),'--output-dir',str(root/'pairing')])
            command('summarize',[sys.executable,'-m','benchmark.attention_slice.summarize','--matrix',str(matrix_path),'--runs-root',str(root),'--output',str(root/'comparison.csv')])
            write_json(root/'gate.json',build_gate(root,matrix_path,args.model_path,assets))
        else:
            if not args.smoke_root: raise ValueError('Full requires --smoke-root')
            gate=Path(args.smoke_root).resolve()/'gate.json'
            for config in configs.values():validate_gate_report(gate,args.model_path,assets,config)
            write_json(root/'gate_reference.json',{'path':str(gate),'sha256':sha256_file(gate)})
            for row in matrix['variants']:
                ppl(row['variant_id'],handoff/row['config'],'windows-full','windows',gate=gate,optional=True)
            command('summarize',[sys.executable,'-m','benchmark.attention_slice.summarize','--matrix',str(matrix_path),'--runs-root',str(root),'--output',str(root/'comparison.csv')],optional=bool(status['failed_commands']))
            if status['failed_commands']: raise RuntimeError('One or more required variants failed; preserved logs and other completed variants')
        status['status']='complete';write_json(root/'status.json',status)
        return status
    except Exception as error:
        status.update(status='failed',reason=str(error));write_json(root/'status.json',status)
        (root/'orchestration-failure.log').write_text(traceback.format_exc())
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','full'),required=True)
    for name in ('matrix','handoff-dir','model-path','asset-dir','baseline-root','output-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--smoke-root')
    args=p.parse_args();print(json.dumps(run(args)),flush=True)

if __name__=='__main__':main()
