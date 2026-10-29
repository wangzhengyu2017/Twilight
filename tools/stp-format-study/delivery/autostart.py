"""Wait for terminal GPU execution and invoke the independently reviewed writer."""
import json
import subprocess
import sys
import time
from pathlib import Path

study = Path('/output/project/experiments/stp-format-study')
execution = study / 'execution-v1'
state = study / 'preparation/delivery-autostart.json'
log = study / 'preparation/delivery-autostart.log'
state.parent.mkdir(exist_ok=True)
state.write_text(json.dumps({'status': 'waiting_for_execution', 'execution': str(execution)}) + '\n')
while True:
    try:
        status = json.loads((execution / 'orchestration_status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(55)
        continue
    if status.get('status') == 'failed':
        state.write_text(json.dumps({'status': 'execution_failed', 'execution_status': status}) + '\n')
        sys.exit(1)
    if status.get('status') == 'complete':
        try:
            json.loads((execution / 'artifact-index.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(55)
            continue
        argv = [sys.executable, str(Path(__file__).parent / 'write_report.py'),
                '--execution-root', str(execution), '--study-root', str(study),
                '--output-dir', str(study / 'delivery-v1')]
        with log.open('w') as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
        state.write_text(json.dumps({'status': 'complete' if result.returncode == 0 else 'writer_refused',
                                    'returncode': result.returncode, 'argv': argv, 'log': str(log)}) + '\n')
        sys.exit(result.returncode)
    time.sleep(55)
