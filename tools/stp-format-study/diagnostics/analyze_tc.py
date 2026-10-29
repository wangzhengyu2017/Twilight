"""CPU-only posthoc diagnosis; reads frozen smoke traces, creates new artifacts only."""
import os
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
import torch, math, csv, json, hashlib
from pathlib import Path
import numpy as np
torch.set_num_threads(2); torch.set_num_interop_threads(2)
OUT=Path(__file__).resolve().parent
ROOTS={'.95':Path('/output/project/experiments/stp-slice/20261028-v2-smoke'),'.90':Path('/output/project/experiments/stp-slice-thresholds/p090-smoke'),'.85':Path('/output/project/experiments/stp-slice-thresholds/p085-smoke')}
rows=[]; codes=[]; inputs=[]
def top(z,p):
 prob=z.softmax(-1); ix=torch.argsort(z,descending=True,stable=True); sp=prob.gather(-1,ix); prev=sp.cumsum(-1)-sp; mask=torch.zeros_like(z,dtype=torch.bool).scatter_(-1,ix,prev<p); return mask,prob

def metrics(z,ref,v,p):
 mask,probs=top(z,p); rm,rp=top(ref,p); err=z-ref; ec=err-err.mean(-1,keepdim=True); rc=ref-ref.mean(-1,keepdim=True); zc=z-z.mean(-1,keepdim=True)
 rankz=torch.argsort(torch.argsort(z,stable=True),stable=True).float(); rankr=torch.argsort(torch.argsort(ref,stable=True),stable=True).float(); rz=rankz-rankz.mean(-1,keepdim=True); rr=rankr-rankr.mean(-1,keepdim=True)
 final=ref.masked_fill(~mask,-torch.inf).softmax(-1); oracle=ref.masked_fill(~rm,-torch.inf).softmax(-1); pv=torch.bmm(final[:,None,:],v).squeeze(1); opv=torch.bmm(oracle[:,None,:],v).squeeze(1)
 out={'rmse':err.square().mean(-1).sqrt(),'centered_rmse':ec.square().mean(-1).sqrt(),'mean_logit_error':err.mean(-1),'logit_std_ratio':zc.square().mean(-1).sqrt()/rc.square().mean(-1).sqrt().clamp_min(1e-12),'spearman':(rz*rr).sum(-1)/(rz.square().sum(-1)*rr.square().sum(-1)).sqrt(),'selected':mask.sum(-1),'reference_selected':rm.sum(-1),'retained_full_mass':(rp*mask).sum(-1),'predicted_mass':(probs*mask).sum(-1),'tv':(probs-rp).abs().sum(-1)/2,'entropy':-(probs*probs.clamp_min(1e-30).log()).sum(-1),'reference_entropy':-(rp*rp.clamp_min(1e-30).log()).sum(-1),'jaccard':(mask&rm).sum(-1)/(mask|rm).sum(-1),'top1_missed':~mask.gather(-1,ref.argmax(-1,keepdim=True)).squeeze(-1),'missing_mass':(rp*(rm&~mask)).sum(-1),'pv_relative_error':(pv-opv).norm(dim=-1)/opv.norm(dim=-1).clamp_min(1e-12)}
 return out
for pstr,root in ROOTS.items():
 p=float(pstr); tag='p'+str(round(p*100)).zfill(3)
 for source in ['int','msb_q4k4','msb_q8k4']:
  td=root/f'{source}_topp_{tag}-decode-on'/'trace'
  if not td.exists():continue
  for ap in sorted(td.glob('*/actual.pt')):
   a=torch.load(ap,map_location='cpu',weights_only=True); q=a['q_codes'].float().reshape(32,1,128); k=a['k_codes'].float().reshape(32,-1,128); qs=a['q_scales'].float().reshape(32,1,1); ks=a['k_scales'].float().reshape(32,-1,1); v=a['v_visible'].float().reshape(32,-1,128)
   factor=qs*ks.transpose(-1,-2)/math.sqrt(128)
   dot=lambda x,y:(x@y.transpose(-1,-2)*factor).squeeze(1)
   ref=dot(q,k); qh=(q/16).floor()*16; kh=(k/16).floor()*16; qr=q-qh; kr=k-kh
   exact=dot(qh,kh)+dot(qh,kr)+dot(qr,kh)+dot(qr,kr)
   assert (ref-exact).abs().max()<1e-4
   estimators={'full_int':ref,'q4k4':dot(qh,kh),'q8k4':dot(q,kh)}
   for name,z in estimators.items():
    mm=metrics(z,ref,v,p)
    for h in range(32): rows.append({'p':p,'source':source,'trace':ap.parent.name,'layer':int(ap.parent.name[5:7]),'head':h,'estimator':name,**{key:float(val[h]) for key,val in mm.items()}})
   for operand,x,res,sc in [('q',q,qr,qs),('k',k,kr,ks)]:
    for h in range(32):
     xx=x[h].flatten(); rr=res[h].flatten(); hh=(xx/16).floor();
     codes.append({'p':p,'source':source,'trace':ap.parent.name,'head':h,'operand':operand,'code_mean':float(xx.mean()),'code_std':float(xx.std()),'residual_mean':float(rr.mean()),'residual_std':float(rr.std()),'zero_code_fraction':float((xx==0).float().mean()),'zero_hi_fraction':float((hh==0).float().mean()),'negative_hi_fraction':float((hh<0).float().mean()),'truncation_bias_dequant':float((-res[h]*sc[h].expand_as(res[h])).mean()),'scale_mean':float(sc[h].mean())})
   terms={'q_hi_k_lo':dot(qh,kr),'q_lo_k_hi':dot(qr,kh),'q_lo_k_lo':dot(qr,kr),'total_missing':ref-estimators['q4k4'],'q_offset_k':dot(torch.ones_like(q)*7.5,k),'q_k_offset':dot(q,torch.ones_like(k)*7.5),'offset_offset':dot(torch.ones_like(q)*7.5,torch.ones_like(k)*7.5)}
   for name,z in terms.items():
    centered=z-z.mean(-1,keepdim=True); target=ref-estimators['q4k4']; tc=target-target.mean(-1,keepdim=True)
    for h in range(32):inputs.append({'p':p,'source':source,'trace':ap.parent.name,'head':h,'term':name,'mean':float(z[h].mean()),'centered_rms':float(centered[h].square().mean().sqrt()),'corr_total_missing':float((centered[h]*tc[h]).sum()/(centered[h].norm()*tc[h].norm()).clamp_min(1e-12))})
for name,data in [('head_metrics.csv',rows),('code_statistics.csv',codes),('decomposition.csv',inputs)]:
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
import pandas as pd
df=pd.DataFrame(rows); numeric=[k for k in rows[0] if k not in ['source','trace','estimator','head','layer','p']]
for group,name in [(['p','source','estimator'],'summary.csv'),(['p','source','layer','estimator'],'layers.csv')]:df.groupby(group)[numeric].mean().reset_index().to_csv(OUT/name,index=False)
# Cross trajectory stats: original captured operands have upstream divergence.
tr=[]
for pstr,root in ROOTS.items():
 tag='p'+str(round(float(pstr)*100)).zfill(3); it=root/f'int_topp_{tag}-decode-on/trace'
 for source in ['msb_q4k4','msb_q8k4']:
  for ap in sorted((root/f'{source}_topp_{tag}-decode-on/trace').glob('*/actual.pt')):
   a=torch.load(ap,map_location='cpu',weights_only=True); b=torch.load(it/ap.parent.name/'actual.pt',map_location='cpu',weights_only=True)
   am=a['selected_mask'].reshape(32,-1);bm=b['selected_mask'].reshape(32,-1)
   for h in range(32):
    r={'p':float(pstr),'source':source,'trace':ap.parent.name,'head':h,'set_jaccard':float((am[h]&bm[h]).sum()/(am[h]|bm[h]).sum())}
    for n in ['q_post_rope','k_visible','layer_input','layer_output']:
     x=a[n].float();y=b[n].float()
     if n in ['q_post_rope','k_visible']:x=x[:,h];y=y[:,h]
     r[n+'_relative_difference']=float((x-y).norm()/y.norm().clamp_min(1e-12))
    tr.append(r)
pd.DataFrame(tr).to_csv(OUT/'trajectory.csv',index=False)
manifest={'threads':2,'gpu_used':False,'model_loaded':False,'roots':{k:str(v) for k,v in ROOTS.items()},'head_metric_rows':len(rows),'scope':'six real traces x32 heads x3 source trajectories x3 thresholds; posthoc CPU diagnostics, not new PPL','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
print(df.query("p==.95 and source=='msb_q4k4'").groupby('estimator')[['centered_rmse','spearman','selected','retained_full_mass','tv','pv_relative_error']].mean().round(4).to_string())
