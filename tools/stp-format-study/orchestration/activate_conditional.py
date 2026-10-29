"""Prepare a distinct conditional matrix only from a complete affirmative decision."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--decision',required=True);p.add_argument('--decision-manifest',required=True);p.add_argument('--matrix',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
d=Path(a.decision).resolve();report=json.loads(d.read_text())
if report.get('status')!='complete' or report.get('activate_mxfp4') is not True:raise ValueError('No valid affirmative conditional decision')
manifest_path=Path(a.decision_manifest).resolve();manifest=json.loads(manifest_path.read_text())
for name,digest in manifest['source_files'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed decision source '+name)
if manifest.get('decision_sha256')!=hashlib.sha256(d.read_bytes()).hexdigest():raise ValueError('Decision manifest mismatch')
source=Path(a.matrix).resolve();m=json.loads(source.read_text());out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
for row in m['variants']:
    row['required']=row.get('status')=='conditional_not_activated'
    if row['required']:row['status']='conditional_activated_pending_smoke'
m['execution_phase']='conditional';m['conditional_activation']={'trigger_verified':True,'decision_path':str(d),'decision_sha256':hashlib.sha256(d.read_bytes()).hexdigest(),'decision_manifest_path':str(manifest_path),'decision_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
shutil.copytree(source.parent/'configs',out/'configs');shutil.copyfile(source.parent/m['trace_targets'],out/m['trace_targets']);(out/'matrix.json').write_text(json.dumps(m,indent=2)+'\n')
print(out/'matrix.json')
