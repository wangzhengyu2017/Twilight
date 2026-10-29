"""Index closed experiment files with absolute paths; explicitly exclude active logs."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);p.add_argument('--exclude',action='append',default=[]);a=p.parse_args()
root=Path(a.root).resolve();out=Path(a.output).resolve();excluded={Path(x).resolve() for x in a.exclude};files={}
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.resolve()==out or path.resolve() in excluded:continue
    if any(part in ('.git','__pycache__','tmp') for part in path.parts) or path.suffix in ('.tmp','.pyc'):continue
    files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
out.write_text(json.dumps({'schema_version':'format-artifact-index-v1','root':str(root),'files':files,'excluded':[str(x) for x in excluded]},indent=2)+'\n')
