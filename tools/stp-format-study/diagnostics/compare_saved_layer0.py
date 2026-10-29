"""Exact equality and membership checks on saved tensors; no estimator evaluated."""
import csv,hashlib,json
from pathlib import Path
import torch

torch.set_num_threads(2)
ROOT=Path('/output/project/experiments/stp-format-study');OUT=ROOT/'diagnosis-math'
FIELDS=['q_codes','k_codes','q_scales','k_scales','q_post_rope','k_visible','q_dq','k_dq','v_visible','allowed_mask','key_positions']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

goldroot=Path('/output/project/experiments/stp-slice/20261028-v2-smoke/msb_q4k4_topp_p095-decode-on/trace')
golds={};sources=[];checks=[];heads=[]
for record in ['layer00_query127_prefill','layer00_query128_decode']:
 p=goldroot/record/'actual.pt';golds[record]=torch.load(p,map_location='cpu',weights_only=False)
 sources.append({'path':str(p),'sha256':sha(p)})
for d in sorted((ROOT/'execution-v1/smoke/pairing').iterdir()):
 if not d.name.startswith(('format_sm8_','format_mxint8_tc4_')):continue
 mp=d/'pairing_manifest.json';m=json.loads(mp.read_text());sources.append({'path':str(mp),'sha256':sha(mp)})
 for record,gold in golds.items():
  p=Path(m['replay_reports']['candidate']['run_dir'])/'trace'/record/'actual.pt'
  actualsha=sha(p);assert actualsha==m['source_files'][str(p)]
  x=torch.load(p,map_location='cpu',weights_only=False)
  equal={f:torch.equal(x[f],gold[f]) for f in FIELDS}
  checks.append({'variant':d.name,'record':record,'all_equal':all(equal.values()),'fields':equal})
  sources.append({'path':str(p),'sha256':actualsha})
  if record=='layer00_query128_decode':
   with (d/'paired_sets.jsonl').open() as f:
    row=next(r for r in map(json.loads,f) if r['reference_role']=='own_source_int_full' and r['predicted_record_id']==record and r['head_id']==2)
   keys=row['predicted_selected_key_positions'];headmask=x['selected_mask'][0,2,0]
   savedkeys=x['key_positions'][headmask].tolist();assert savedkeys==keys
   heads.append({'variant':d.name,'p':row['p'],'num_predicted':row['num_predicted'],'num_reference':row['num_reference'],'retained_mass':row['retained_mass'],'shortfall':row['shortfall'],'key127_selected':127 in keys,'key128_selected':128 in keys,'key126_selected':126 in keys,'selected_keys':json.dumps(keys)})
assert len(checks)==24 and all(c['all_equal'] for c in checks)
name='layer0_query128_head2.csv'
with (OUT/name).open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(heads[0]));w.writeheader();w.writerows(heads)
gold=golds['layer00_query128_decode'];goldkeys=gold['key_positions'][gold['selected_mask'][0,2,0]].tolist()
assert goldkeys==[128]
result={'scope':'Saved-tensor exact comparisons only; no logits/probabilities/estimators recomputed','gold':'historical TC Q4K4 p=.95, same position','compared_fields':FIELDS,'checks':checks,'all_24_candidate_captures_equal_to_position_matched_gold':True,'historical_tc_q4_p095_query128_head2_selected_keys':goldkeys,'source_files':sources,'artifacts':{name:sha(OUT/name),'compare_saved_layer0.py':sha(OUT/'compare_saved_layer0.py')}}
(OUT/'layer0_input_equality_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
for r in heads:
 if r['p']==.95:print({k:v for k,v in r.items() if k!='selected_keys'})
print('24 layer0 captures: all 11 saved input/domain/value tensor fields exactly equal to position-matched TC p=.95 gold.')
