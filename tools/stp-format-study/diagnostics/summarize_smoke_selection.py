"""Aggregate saved actual pairing records only; no tensors/models/predictors loaded."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

ROOT = Path('/output/project/experiments/stp-format-study')
OUT = ROOT / 'diagnosis-math'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def writecsv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

sources=[];ownrows=[];crossrows=[]
for directory in sorted((ROOT/'execution-v1/smoke/pairing').iterdir()):
    if not directory.name.startswith(('format_sm8_', 'format_mxint8_tc4_')): continue
    jp=directory/'paired_sets.jsonl';mp=directory/'pairing_manifest.json'
    m=json.loads(mp.read_text())
    assert m['passed'] and m['status']=='complete'
    actualsha=sha(jp)
    assert m['artifacts'][str(jp)]==actualsha
    sources.extend([{'path':str(jp),'sha256':actualsha},{'path':str(mp),'sha256':sha(mp)}])
    grouped={}
    with jp.open() as f:
        for line in f:
            r=json.loads(line);grouped.setdefault(r['reference_role'],[]).append(r)
    own=grouped['own_source_int_full'];tc=grouped['tc']
    assert len(own)==len(tc)==192
    assert len({(r['predicted_record_id'],r['head_id']) for r in own})==192
    assert all(r['comparison_kind']=='same_input_estimator' for r in own)
    assert all(r['comparison_kind']=='cross_run_trajectory' for r in tc)
    replay=m['replay_reports']['candidate']
    assert replay['passed'] and replay['num_records']==6
    stats=[s for rec in replay['records'] for s in rec['same_input_logit_statistics']]
    assert len(stats)==192
    statmap={(rec['record_id'],s['head_id']):s for rec in replay['records'] for s in rec['same_input_logit_statistics']}
    assert all(r['num_predicted']==statmap[(r['predicted_record_id'],r['head_id'])]['selected_count'] for r in own)
    p=own[0]['p'];shape='Q4K4' if '_q4k4_' in directory.name else 'Q8K4'
    base={'variant':directory.name,'format':'SM' if directory.name.startswith('format_sm8_') else 'MXINT8_TC4','shape':shape,'p':p,'heads':192}
    row=dict(base)
    for field in ['num_predicted','num_reference','num_intersection','precision','recall','jaccard','retained_mass','reference_retained_mass','shortfall']:
        row['mean_'+field]=fmean(r[field] for r in own)
    row['ratio_mean_counts']=sum(r['num_predicted'] for r in own)/sum(r['num_reference'] for r in own)
    row['mean_head_count_ratio']=fmean(r['num_predicted']/r['num_reference'] for r in own)
    row['heads_more_than_reference']=sum(r['num_predicted']>r['num_reference'] for r in own)
    row['heads_below_threshold']=sum(r['retained_mass']<p for r in own)
    row['min_retained_mass']=min(r['retained_mass'] for r in own)
    row['mean_centered_logit_rmse']=fmean(s['centered_rmse'] for s in stats)
    row['mean_stable_rank_correlation']=fmean(s['stable_rank_correlation'] for s in stats)
    row['mean_predictor_tie_fraction']=fmean(s['predictor_tie_fraction'] for s in stats)
    ownrows.append(row)
    cross=dict(base)
    cross['comparison_kind']='cross_run_trajectory';cross['reference_role']='tc'
    for field in ['num_predicted','num_reference','num_intersection','precision','recall','jaccard']:
        cross['mean_'+field]=fmean(r[field] for r in tc)
    cross['ratio_mean_counts']=sum(r['num_predicted'] for r in tc)/sum(r['num_reference'] for r in tc)
    crossrows.append(cross)
assert len(ownrows)==12
writecsv('smoke_selection_own_source.csv',ownrows)
writecsv('smoke_selection_cross_tc.csv',crossrows)
manifest={'scope':'Existing actual smoke pairing JSONL and manifests only; aggregation, no new estimator or experiment','weighting':'192 equally weighted head/query records per cell; 6 captures x 32 heads','source_files':sources,'artifacts':{name:sha(OUT/name) for name in ['summarize_smoke_selection.py','smoke_selection_own_source.csv','smoke_selection_cross_tc.csv']},'checks':['12 complete passed manifests','paired_sets SHA256 matches source manifest','192 unique own-source identities/cell','192 cross-TC records/cell','6 passed candidate replay captures/cell','saved selection count equals saved replay selection count for every own-source row']}
if (OUT/'SMOKE_SELECTION_BUDGET.md').exists():
    manifest['artifacts']['SMOKE_SELECTION_BUDGET.md']=sha(OUT/'SMOKE_SELECTION_BUDGET.md')
if (OUT/'layer0_input_equality_manifest.json').exists():
    manifest['related_layer0_equality_manifest']={'path':'layer0_input_equality_manifest.json','sha256':sha(OUT/'layer0_input_equality_manifest.json')}
(OUT/'smoke_selection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r,c in zip(ownrows,crossrows):
    print(r['format'],r['shape'],r['p'],'counts',round(r['mean_num_predicted'],4),round(r['mean_num_reference'],4),'ratio',round(r['ratio_mean_counts'],4),'mass',round(r['mean_retained_mass'],6),'shortfall',round(r['mean_shortfall'],6),'J',round(r['mean_jaccard'],6),'recall',round(r['mean_recall'],6),'precision',round(r['mean_precision'],6),'below',r['heads_below_threshold'],'RMSE',round(r['mean_centered_logit_rmse'],6),'rank',round(r['mean_stable_rank_correlation'],6),'TCcount',round(c['mean_num_reference'],4))
