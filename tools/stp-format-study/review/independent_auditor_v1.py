import json,math,time,hashlib,traceback
from pathlib import Path
R=Path('/output/project/experiments/stp-format-study/execution-v1'); V=Path('/output/project/experiments/stp-format-study/review'); S=Path('/output/project/Twilight-STP_test_Format')
log=(V/'execution-v1-independent-watch.log').open('a',buffering=1);seen=set();gates={};full={};dv=False;last=None

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(v,m):
 if not v:raise ValueError(m)
def emit(kind,**d):
 t=json.dumps(dict(unix=time.time(),kind=kind,**d),sort_keys=True);print(t,flush=True);log.write(t+'\n')
def bind(fs):
 for n,h in fs.items():ck(Path(n).is_file() and sha(n)==h,'changed hash '+n)
def gate(phase):
 if phase in gates:return gates[phase]
 p=R/phase/'gate.json';d=read(p);ck(d['passed'] and d['status']=='complete','gate incomplete');ck(d['trace_records']==(72 if phase=='smoke' else 36),'trace count');bind(d['evidence_files'])
 for n,h in d['source_files'].items():ck(sha(S/n)==h,'source changed '+n)
 ck(d['head']=='124b78ae3e542bba3dc66bb35e58048373892ced','source head');gates[phase]=d;emit('gate_verified',phase=phase,traces=d['trace_records'],evidence_files=len(d['evidence_files']));return d

def audit(p,phase):
 m=read(p/'metrics.json');c=read(p/'coverage.json');a=read(p/'asset_manifest.json');rows=[json.loads(x) for x in (p/'window_metrics.jsonl').read_text().splitlines()];gp=R/('smoke' if phase=='full' else 'conditional-smoke')/'gate.json';g=gate(gp.parent.name);N=a['num_tokens']-1
 ck(m['profile']=='full' and m['mode']=='windows' and m['num_targets']==N==289075,'full target/profile');ck((m['context_length'],m['stride'],m['query_chunk'])==(2048,1024,64),'dimensions');ck(len(rows)==len(c['scoring_ranges'])==282,'window count');cursor=1
 for i,(row,rng) in enumerate(zip(rows,c['scoring_ranges'])):
  ck(row['window_id']==i and row['lo']==rng['lo']==cursor,'gap/order');ck(all(row[k]==rng[k] for k in ('b','e','lo','target_end')),'ranges');ck(row['num_targets']==row['e']-row['lo'] and row['target_end']==row['e'] and row['e']-row['b']<=2048,'row targets');ck(row['logits_sha256'] and math.isfinite(row['sum_nll']),'row data');cursor=row['target_end']
 ck(cursor==a['num_tokens'] and sum(x['num_targets'] for x in rows)==N,'tail/coverage');ck(sum(x['sum_nll'] for x in rows)==m['sum_nll'],'NLL sum');ck(math.exp(m['sum_nll']/N)==m['ppl'],'PPL');ck(c['num_targets']==N and c['all_layers_covered'] and c['expected_layer_count']==32 and set(c['layers'])==set(map(str,range(32))),'layers');ck(all(v=={'intflash_qk8':{'prefill':282}} for v in c['layers'].values()),'layer calls')
 for k in ('code_sha256','environment_sha256','asset_manifest_sha256','tokens_sha256'):ck(m[k]==g[k],'gate identity '+k)
 ck(m['gate_report_sha256']==sha(gp),'gate digest');ck(read(p/'costs.json')['selection_passes']==m['selection_passes']==1,'selection passes');ck(read(p/'method_manifest.json')['final_operand_origin']=='full_codes','final source');ck(m['capture_config_sha256'] is None,'trace enabled')
 d=dict(variant_id=m['variant_id'],phase=phase,targets=N,windows=len(rows),sum_nll=m['sum_nll'],ppl=m['ppl'],forward_nll_seconds=m['forward_nll_seconds'],end_to_end_seconds=m['end_to_end_seconds'],metrics_sha256=sha(p/'metrics.json'));full[str(p)]=d;emit('full_cell_verified',**d);(V/'execution-v1-full-independent-audit.json').write_text(json.dumps(dict(status='partial_until_terminal_verified',verified_cells=list(full.values())),indent=2)+'\n')

def decision():
 p=R/'INITIAL_BASELINE.decision.json'
 if not p.exists():p=R/'BASELINE.decision.json'
 if not p.exists():return False
 d=read(p);ck(d['status']=='complete','decision unavailable');ws={};roots={'095':Path('/output/project/experiments/stp-slice/20261028-v2-full'),'090':Path('/output/project/experiments/stp-slice-thresholds/p090-full'),'085':Path('/output/project/experiments/stp-slice-thresholds/p085-full')}
 def ce(p):
  m=read(p/'metrics.json');ck(read(p/'status.json')['status']=='complete' and m['num_targets']==289075,'decision source');return m['sum_nll']/m['num_targets']
 for q in (4,8):
  mx=sum(ce(R/'full'/f'format_mxint8_tc4_q{q}k4_topp_p{p}-windows-full') for p in ('095','090','085'))/3;tc=sum(ce(roots[p]/f'msb_q{q}k4_topp_p{p}-windows-full') for p in ('095','090','085'))/3;it=sum(ce(roots[p]/f'int_topp_p{p}-windows-full') for p in ('095','090','085'))/3;a=mx>=tc;b=mx-it>math.log(1.01);w=d['widths'][f'q{q}k4'];ck(w['mean_ce']==dict(mx=mx,tc=tc,int=it) and w['no_tc_improvement']==a and w['above_int_margin']==b,'decision numeric mismatch');ws[f'q{q}k4']=dict(no_tc_improvement=a,above_int_margin=b)
 ck(d['activate_mxfp4']==(ws['q4k4']['no_tc_improvement'] or ws['q4k4']['above_int_margin']),'activation');emit('conditional_decision_verified',activate_mxfp4=d['activate_mxfp4'],widths=ws);return True

emit('independent_auditor_started',scope='CPU-only; writes review notes only')
for tick in range(2400):
 try:
  for phase in ('smoke','conditional-smoke'):
   p=R/phase/'gate.json'
   if p.exists() and read(p).get('passed'):gate(phase)
  for phase in ('full','conditional-full'):
   for p in sorted((R/phase).glob('format_*-windows-full')):
    if str(p) in seen or not (p/'status.json').exists():continue
    st=read(p/'status.json')
    if st['status']=='failed':emit('model_run_failed',run=str(p),details=st);seen.add(str(p))
    if st['status']=='complete':audit(p,phase);seen.add(str(p))
  if not dv and len([x for x in full.values() if x['phase']=='full'])==12:dv=decision()
  st=read(R/'orchestration_status.json')
  if st!=last:emit('orchestration_status',details=st);last=st
  if st['status']=='failed':emit('terminal_failure_preserved',details=st);break
  idx=R/'artifact-index.json'
  if st['status']=='complete' and idx.exists():
   d=read(idx);bind(d['files']);ck(dv,'decision not verified');ck(len(full)==(18 if st['conditional_activated'] else 12),'terminal full count');out=dict(status='terminal_artifacts_independently_verified',verified_full_cells=list(full.values()),full_cell_count=len(full),initial_trace_records=72,conditional_trace_records=36 if st['conditional_activated'] else 0,artifact_index_files=len(d['files']),artifact_index_sha256=sha(idx),conditional_activated=st['conditional_activated'],conditional_decision_independently_recomputed=True,source_head='124b78ae3e542bba3dc66bb35e58048373892ced');(V/'execution-v1-full-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');emit('terminal_independent_audit_passed',full_cell_count=len(full),artifact_index_files=len(d['files']));break
 except (json.JSONDecodeError,FileNotFoundError) as e:emit('transient_incomplete_write',reason=str(e))
 except Exception as e:emit('INDEPENDENT_AUDIT_FAILURE',reason=str(e),traceback=traceback.format_exc());break
 time.sleep(60)
