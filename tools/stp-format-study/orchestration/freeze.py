"""Archive exact source, patch, environment, assets, matrix and adopted specs."""
import argparse,shutil,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from benchmark.attention_baseline.common import ROOT,code_provenance,environment,configure_runtime,load_json,write_json,sha256_file
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--matrix',required=True);p.add_argument('--assets',required=True);a=p.parse_args()
out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copytree(Path(__file__).resolve().parent,out/'tools',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
configure_runtime();code=code_provenance(out);write_json(out/'environment.json',environment())
for name,digest in code['files'].items():
    source=ROOT/name;target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    if sha256_file(target)!=digest:raise ValueError('Source changed while freezing '+name)
matrix=Path(a.matrix).resolve();shutil.copytree(matrix.parent,out/'handoff',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));m=load_json(matrix)
for name,digest in m['specification_files'].items():
    source=Path(name);target=out/'spec'/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    if sha256_file(target)!=digest:raise ValueError('Specification changed '+name)
diagnosis=m['diagnosis_manifest'];diagnosis_path=Path(diagnosis['path'])
if sha256_file(diagnosis_path)!=diagnosis['sha256']:raise ValueError('Changed diagnosis manifest')
shutil.copytree(diagnosis_path.parent,out/'diagnosis-distributions')
for entry in load_json(diagnosis_path)['inputs']+load_json(diagnosis_path)['outputs']:
    if sha256_file(entry['path'])!=entry['sha256']:raise ValueError('Changed diagnosis evidence '+entry['path'])
shutil.copyfile(a.assets,out/'asset_manifest.json')
write_json(out/'freeze.json',dict(status='complete',head=code['head'],code_sha256=code['code_sha256'],matrix_sha256=sha256_file(matrix),asset_manifest_sha256=sha256_file(a.assets),python=sys.executable,source_root=str(ROOT)))
