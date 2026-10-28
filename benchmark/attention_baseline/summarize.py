"""Produce one comparison per protocol/profile without inventing missing metrics."""
import argparse
import csv
import math
from pathlib import Path
from .common import load_json,canonical_hash


def public_config(config):
    config=dict(config)
    if 'algorithm' in config:config['algorithm']=public_config(config['algorithm'])
    config.pop('attention_quantization',None)
    for key in ('trace','trace_targets','output_dir','capture'):config.pop(key,None)
    return config


def summarize(runs_root,output):
    root=Path(runs_root);groups={};captures={}
    for status_path in root.rglob('status.json'):
        run=status_path.parent
        if not (run/'resolved_config.json').exists() or any(s in run.name for s in ('preflight','validation')):continue
        status=load_json(status_path);mp=run/'metrics.json'
        metrics=load_json(mp) if mp.exists() else {}
        config=load_json(run/'resolved_config.json');method=config.get('algorithm',config)['attention_quantization']['method']
        mode=metrics.get('mode','cached_decode' if 'decode' in run.name else 'windows')
        profile=metrics.get('profile','full' if 'full' in run.name else 'smoke')
        # off/on decode are the same measurement; choose off, preserving failed capture evidence separately.
        if mode=='cached_decode' and (config.get('trace_targets') or (run/'trace').exists()):
            capture=captures.setdefault((profile,mode),{})
            assert method not in capture, f'duplicate capture {profile}/{mode}/{method}'
            capture[method]=(run,status)
            continue
        key=(profile,mode); group=groups.setdefault(key,{})
        assert method not in group, f'duplicate {profile}/{mode}/{method}; summarize one run root at a time'
        group[method]=(run,status,metrics,config)
    expected=[('full','windows'),('smoke','windows'),('smoke','cached_decode')]
    rows=[]
    for profile,mode in expected:
        group=groups.get((profile,mode),{});row={'profile':profile,'mode':mode}
        valid={}
        for method in ('fp_common','intflash_qk8'):
            data=group.get(method);prefix='fp' if method=='fp_common' else 'int'
            state='not_run' if data is None else data[1].get('status','failed')
            row[prefix+'_status']=state
            capture=captures.get((profile,mode),{}).get(method)
            if capture:
                row[prefix+'_capture_status']=capture[1].get('status','failed')
                row[prefix+'_capture_run']=str(capture[0])
                row[prefix+'_capture_reason']=capture[1].get('reason','')
            if data:
                row[prefix+'_run']=str(data[0]);row[prefix+'_reason']=data[1].get('reason','')
            if state=='complete':
                m=data[2];assert math.isfinite(m['sum_nll']) and math.isfinite(m['ppl']) and m['ppl']>=1 and m['num_targets']>0,'invalid completed metrics'
                if mode=='windows':assert m['num_targets']==m['num_input_tokens']-1, 'incomplete windows coverage'
                row[prefix+'_sum_nll']=m['sum_nll'];row[prefix+'_ppl']=m['ppl'];row[prefix+'_num_targets']=m['num_targets'];valid[method]=data
        if len(valid)==2:
            a,b=(valid[m] for m in ('fp_common','intflash_qk8'))
            for key in ('protocol_id','model_dtype','backend','tokens_sha256','subset_sha256','scoring_ranges_sha256','num_input_tokens','num_targets','asset_manifest_sha256','code_sha256','environment_sha256'):
                assert a[2][key]==b[2][key], f'{profile}/{mode}: incompatible {key}'
            assert canonical_hash(public_config(a[3]))==canonical_hash(public_config(b[3])), 'common algorithm configuration mismatch'
            assert load_json(a[0]/'coverage.json')['scoring_ranges']==load_json(b[0]/'coverage.json')['scoring_ranges'], 'coverage mismatch'
            row['delta_nll']=b[2]['sum_nll']-a[2]['sum_nll'];row['delta_ppl']=b[2]['ppl']-a[2]['ppl']
        rows.append(row)
    names=['profile','mode']+[p+'_'+s for p in ('fp','int') for s in ('status','run','reason','sum_nll','num_targets','ppl','capture_status','capture_run','capture_reason')]+['delta_nll','delta_ppl']
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();summarize(a.runs_root,a.output)

if __name__=='__main__':main()
