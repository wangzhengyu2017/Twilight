"""Rebind adopted spec digests before source freeze, never after smoke begins."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--matrix',required=True);a=p.parse_args();path=Path(a.matrix);matrix=json.loads(path.read_text())
for name in matrix['specification_files']:matrix['specification_files'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
path.write_text(json.dumps(matrix,indent=2)+'\n')
