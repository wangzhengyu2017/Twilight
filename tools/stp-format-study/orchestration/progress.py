"""Read current run logs and final window metrics without modifying a run."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
for log in sorted(Path(a.root).rglob('run.log')):
    run=log.parent;metrics=run/'window_metrics.jsonl';final=run/'metrics.json'
    lines=log.read_text(errors='replace').splitlines()
    report={'run':str(run),'last_log':lines[-1] if lines else '', 'final':final.exists()}
    if metrics.exists():report['completed_windows']=sum(bool(x.strip()) for x in metrics.read_text().splitlines())
    else:
        windows={}
        for line in lines:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if 'window_id' in row and 'num_targets' in row:windows[row['window_id']]=row
        report['completed_windows']=len(windows)
        report['scored_targets_so_far']=sum(row['num_targets'] for row in windows.values())
        report['progress_source']='run.log until window_metrics.jsonl finalizes'
    if final.exists():report['metrics']=json.loads(final.read_text())
    print(json.dumps(report))
