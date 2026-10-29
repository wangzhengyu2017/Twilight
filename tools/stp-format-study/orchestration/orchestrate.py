"""Persist a complete serial smoke/full/report plan; --execute releases commands."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--study-root',required=True);p.add_argument('--matrix',required=True);p.add_argument('--model-path',required=True);p.add_argument('--asset-dir',required=True);p.add_argument('--controls-manifest',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
root=Path(a.study_root).resolve();tools=Path(__file__).resolve().parent;py=sys.executable
common=['--matrix',a.matrix,'--model-path',a.model_path,'--asset-dir',a.asset_dir]
plan=[['freeze',[py,str(tools/'freeze.py'),'--matrix',a.matrix,'--assets',str(Path(a.asset_dir)/'asset_manifest.json'),'--output-dir',str(root/'freeze')]],['smoke',[py,'-m','benchmark.attention_format.run','--profile','smoke',*common,'--controls-manifest',a.controls_manifest,'--output-root',str(root/'smoke'),'--execute']],['full',[py,'-m','benchmark.attention_format.run','--profile','full',*common,'--smoke-root',str(root/'smoke'),'--output-root',str(root/'full'),'--execute']]]
summary=[py,'-m','benchmark.attention_format.summarize','--matrix',a.matrix,'--output',str(root/'BASELINE.csv')]
for path in [root/'full',root/'smoke','/output/project/experiments/stp-slice/20261028-v2-full','/output/project/experiments/stp-slice-thresholds/p090-full','/output/project/experiments/stp-slice-thresholds/p085-full','/output/project/experiments/stp-baselines/20261027-v5-full']:summary+=['--runs-root',str(path)]
plan.append(['summary',summary])
if not a.execute:print(json.dumps(plan,indent=2));sys.exit()
root.mkdir(parents=True,exist_ok=False);(root/'orchestration_logs').mkdir();(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for name,argv in plan:
    with (root/'orchestration_logs'/(name+'.log')).open('w') as log:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT).returncode
    (root/'orchestration_status.json').write_text(json.dumps(dict(stage=name,returncode=rc,status='failed' if rc else 'running'))+'\n')
    if rc:
        # Preserve failed full cells and still produce the report.
        if name=='full':subprocess.run(summary,check=False)
        sys.exit(rc)
decision=json.loads((root/'BASELINE.decision.json').read_text())
if decision.get('status')!='complete':raise ValueError('Conditional decision unavailable; preserve and investigate evidence')
if decision['activate_mxfp4']:
    import shutil
    for suffix in ('csv','decision.json','manifest.json'):shutil.copyfile(root/('BASELINE.'+suffix),root/('INITIAL_BASELINE.'+suffix))
    activate=[py,str(tools/'activate_conditional.py'),'--matrix',a.matrix,'--decision',str(root/'INITIAL_BASELINE.decision.json'),'--decision-manifest',str(root/'INITIAL_BASELINE.manifest.json'),'--output-dir',str(root/'conditional-handoff')]
    subprocess.run(activate,check=True)
    matrix=str(root/'conditional-handoff/matrix.json')
    summary[summary.index('--matrix')+1]=matrix
    for profile in ('smoke','full'):
        argv=[py,'-m','benchmark.attention_format.run','--profile',profile,'--phase','conditional','--decision',str(root/'INITIAL_BASELINE.decision.json'),'--matrix',matrix,'--model-path',a.model_path,'--asset-dir',a.asset_dir,'--output-root',str(root/('conditional-'+profile)),'--controls-manifest',a.controls_manifest,'--execute']
        if profile=='full':argv+=['--smoke-root',str(root/'conditional-smoke')]
        with (root/'orchestration_logs'/('conditional-'+profile+'.log')).open('w') as log:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:
            subprocess.run(summary+['--runs-root',str(root/'conditional-smoke'),'--runs-root',str(root/'conditional-full')],check=False)
            raise RuntimeError('Conditional '+profile+' failed; preserved evidence')
    summary+=['--runs-root',str(root/'conditional-smoke'),'--runs-root',str(root/'conditional-full')]
    subprocess.run(summary,check=True)
(root/'orchestration_status.json').write_text(json.dumps(dict(status='complete',conditional_activated=decision['activate_mxfp4']))+'\n')
subprocess.run([py,str(tools/'index_artifacts.py'),str(root),'--output',str(root/'artifact-index.json')],check=True)
