#!/usr/bin/env python3
"""CPU-only reconstruction of saved non-RS K-position ID set sizes.
No model, GPU, repository imports or RS artifacts are read or changed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import torch

torch.set_num_threads(1)
ROOT = Path('/output/project/experiments')
OUT = ROOT / 'stp-nonrs-cancelled-summary-v1'
HIST = ROOT / 'stp-format-study/scientific-synthesis-v1/MEASURED_RESULTS.csv'
REPORT = ROOT / 'stp-format-study/paired-expanded-v1/report-v6'
BASE = ROOT / 'stp-baselines/20261027-v5-smoke/intflash_qk8-decode-on'
LAYERS = (0, 16, 31)
SLOTS = tuple(f'layer{l:02d}_query{q}_{s}' for l in LAYERS for q,s in ((127,'prefill'),(128,'decode')))
INPUTS = {}
CHECKS = []
GATES = {}
TRACE_CACHE = {}
IDENTITY = ('protocol_id','tokens_sha256','subset_sha256','asset_manifest_sha256','sample_id','window_id','stage','layer_id','query_position','capture_config_sha256')


def sha(p):
    p=Path(p)
    if str(p) not in INPUTS:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
        INPUTS[str(p)]=h.hexdigest()
    return INPUTS[str(p)]


def readjson(p):
    sha(p)
    return json.loads(Path(p).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checkhash(p, expected):
    assert expected and sha(p)==expected, f'hash mismatch: {p}'


def gate(p):
    p=str(p)
    if p not in GATES:
        d=readjson(p)
        assert d['passed'] is True and d['status']=='complete', f'gate incomplete: {p}'
        GATES[p]=d
    return GATES[p]


def loadtrace(run, gate_path=None, report_manifest=None):
    run=Path(run)
    if str(run) in TRACE_CACHE: return TRACE_CACHE[str(run)]
    index_path=run/'trace/index.json'
    index=readjson(index_path)
    assert {r['record_id'] for r in index['records']}==set(SLOTS), f'capture slots mismatch: {run}'
    g=gate(gate_path) if gate_path else None
    if g:
        checkhash(index_path,g['evidence_files'][str(index_path)])
    if report_manifest:
        checkhash(index_path, report_manifest['inputs'][str(index_path)])
    traces={}
    for r in index['records']:
        p=run/'trace'/r['path']
        for name in ('meta','actual','derived'):
            file=p/('meta.json' if name=='meta' else f'{name}.pt')
            checkhash(file,r[f'{name}_sha256'])
            if g:
                checkhash(file,g['evidence_files'][str(file)])
            if report_manifest:
                checkhash(file,report_manifest['inputs'][str(file)])
        meta=readjson(p/'meta.json')
        actual=torch.load(p/'actual.pt',map_location='cpu',weights_only=False)
        derived=torch.load(p/'derived.pt',map_location='cpu',weights_only=False)
        assert all(not isinstance(v,torch.Tensor) or v.device.type=='cpu' for v in actual.values())
        traces[r['record_id']]=(meta,actual,derived,p)
    TRACE_CACHE[str(run)]=traces
    return traces


def sets_from_tensors(data,keys,mask_name,index_name,valid_name,count_name):
    """Indices are original key-position IDs, not dense offsets; padding is ignored."""
    mask=data[mask_name].bool()
    assert tuple(mask.shape[:3])==(1,32,1)
    indices=data[index_name]
    valid=data[valid_name].bool()
    counts=data[count_name]
    out=[]
    assert indices.shape==valid.shape
    for h in range(32):
        selected=keys[mask[0,h,0]].tolist()
        saved=indices[0,h,0][valid[0,h,0]].tolist()
        assert len(selected)==len(set(selected))==len(saved)==len(set(saved))
        assert set(selected)==set(saved), f'mask/indices membership mismatch head {h}'
        assert len(selected)==int(counts[0,h,0])
        out.append(sorted(selected))
    return out


def allowed_sets(actual):
    keys=actual['key_positions'].long().flatten()
    assert len(keys)==len(set(keys.tolist()))
    allowed=actual['allowed_mask'].bool().expand(1,32,1,len(keys))
    return keys,[sorted(keys[allowed[0,h,0]].tolist()) for h in range(32)]


def addcheck(vid,slot,h,kind,selected,allowed,mask_path):
    assert set(selected)<=set(allowed), f'noncausal IDs: {vid} {slot} h{h}'
    CHECKS.append({'variant_id':vid,'record_id':slot,'head_id':h,'kind':kind,
                   'selected_count':len(selected),'baseline_allowed_count':len(allowed),
                   'selected_ids_sha256':digest(selected),'baseline_allowed_ids_sha256':digest(allowed),
                   'selected_subset_of_baseline_allowed':True,'mask_indices_valid_membership_exact':True,
                   'tensor_path':str(mask_path)})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old={r['variant_id']:r for r in csv.DictReader(HIST.open())}; sha(HIST)
    order=list(csv.DictReader((OUT/'TABLE_PPL.csv').open()))
    assert len(order)==42 and len({r['variant_id'] for r in order})==42
    # IDs/order only; no PPL values enter any K calculation.
    sha(OUT/'TABLE_PPL.csv')
    manifest=readjson(REPORT/'manifest.json')
    report_selections=defaultdict(dict)
    sha(REPORT/'selections.jsonl')
    for line in (REPORT/'selections.jsonl').open():
        r=json.loads(line)
        assert r['family'] in ('original18','nonRS_dense4','nonRS_fullQ8K8_TopP6')
        key=(r['record_id'],r['head_id'],int(round(100*r['p'])))
        assert key not in report_selections[r['variant']]
        report_selections[r['variant']][key]=r
    fam={v:f for f in manifest['families'] if f['family']!='RS10' for v in f.get('variants',[])}
    # Reading the non-RS entries of report-v6 does not read any RS files listed there.
    for f in manifest['families']:
        if f['family']=='RS10': continue
        assert f['status']=='accepted_saved_traces'
        checkhash(f['gate_path'],f['gate_sha256'])
        review=f.get('independent_review_provenance')
        if isinstance(review,dict): checkhash(review['path'],review['sha256'])
        amendment=f.get('verification_amendment')
        if amendment:
            approval=amendment['independent_approval']; checkhash(approval['path'],approval['sha256'])
    base_status=readjson(BASE/'status.json'); assert base_status['status']=='complete'
    base_replay=readjson(BASE/'replay.json'); assert base_replay['passed'] is True
    checkhash(BASE/'trace/index.json',base_replay['index_sha256'])
    validation=readjson(BASE.parent/'validation/report.json'); assert validation['passed'] is True
    baseline=loadtrace(BASE)
    baseline_sets={}
    for slot,(m,a,d,p) in baseline.items():
        assert m['method']=='intflash_qk8'
        assert not any(k in a for k in ('selected_mask','selected_indices','execution_mask'))
        _,allowed=allowed_sets(a)
        assert all(allowed[h]==list(range(m['query_position']+1)) for h in range(32))
        baseline_sets[slot]=allowed
    rows=[]; summaries=[]
    for info in order:
        vid=info['variant_id']; before=len(CHECKS); errors=[]; result={}
        is_dense=vid.startswith('dense_')
        historical_dense=vid in ('fp_common','intflash_qk8')
        source=None; gpath=None
        if historical_dense:
            status='N/A: no saved important K ID set (dense execution uses all legal keys)'
            if vid=='intflash_qk8': source=str(BASE)
        else:
            try:
                if vid in fam:
                    f=fam[vid]; gpath=f['gate_path']; g=gate(gpath)
                    first=next(iter(report_selections[vid].values()))
                    source=str(Path(first['actual_path']).parents[2])
                    traces=loadtrace(source,gpath,manifest)
                    if is_dense:
                        assert g['selection_mode']=='dense' and g['selection_passes']==0
                        assert g['diagnostic_sets_unused_in_forward'] is True
                    else:
                        assert all(r['set_semantics']=='actual_forward_selected_keys' for r in report_selections[vid].values())
                else:
                    gpath=old[vid]['gate_path']; gate(gpath)
                    source=str(Path(gpath).parent/f'{vid}-decode-on')
                    traces=loadtrace(source,gpath)
                # Historical controls additionally retain their original per-head set sidecar.
                side={}
                if vid not in fam:
                    index=readjson(Path(source)/'trace/index.json')
                    sidepath=Path(index['selection_sets']['path'])
                    checkhash(sidepath,index['selection_sets']['sha256'])
                    for line in sidepath.open():
                        r=json.loads(line); key=(r['record_id'],r['head_id'])
                        assert key not in side; side[key]=r
                    assert len(side)==192
                for slot,(m,a,d,p) in traces.items():
                    bm=baseline[slot][0]
                    for k in IDENTITY: assert m.get(k)==bm.get(k), f'identity {k}: {vid} {slot}'
                    assert m['layer_id'] in LAYERS
                    keys,local_allowed=allowed_sets(a)
                    assert local_allowed==baseline_sets[slot], f'allowed ID mismatch: {vid} {slot}'
                    assert a['q_post_rope'].shape[1:3]==torch.Size([32,1])
                    if is_dense:
                        sets=sets_from_tensors(a,keys,'execution_mask','execution_indices','execution_valid_mask','execution_counts')
                        assert sets==local_allowed
                        kinds={'actual_forward_dense_all_legal':(sets,p/'actual.pt')}
                        for percent in (95,90,85):
                            name=f'diagnostic_p{percent:03d}'
                            ds=sets_from_tensors(d,keys,name+'_mask',name+'_indices',name+'_valid_mask',name+'_counts')
                            kinds[f'offline_diagnostic_p{percent:03d}_unused_in_forward']=(ds,p/'derived.pt')
                    else:
                        valid='selected_valid_mask' if 'selected_valid_mask' in a else 'valid_mask'
                        sets=sets_from_tensors(a,keys,'selected_mask','selected_indices',valid,'selected_counts')
                        kinds={'actual_forward_selected_keys':(sets,p/'actual.pt')}
                    result[slot]={}
                    for kind,(sets,tensor_path) in kinds.items():
                        for h,s in enumerate(sets):
                            if kind.startswith('offline_diagnostic') or (vid in fam and not is_dense):
                                percent=int(kind.split('_p')[1][:3]) if kind.startswith('offline_diagnostic') else int(vid.rsplit('_p',1)[1])
                                r=report_selections[vid][(slot,h,percent)]
                                assert r['set_semantics']==('offline_diagnostic_unused_in_forward' if is_dense else 'actual_forward_selected_keys')
                                assert r['actual_path']==str(p/'actual.pt') and r['derived_path']==str(p/'derived.pt')
                                assert r['gate_path']==gpath
                                assert r['selected_key_positions']==s and r['selected_count']==len(s)
                                assert r['allowed_key_positions']==local_allowed[h] and r['allowed_count']==len(local_allowed[h])
                                assert r['selected_mask']==(d[f'diagnostic_p{percent:03d}_mask'] if is_dense else a['selected_mask'])[0,h,0].tolist()
                            elif vid not in fam:
                                r=side[(slot,h)]
                                assert r['selected_key_positions']==s and r['num_selected']==len(s)
                                checkhash(p/'actual.pt',r['actual_sha256'])
                                assert r['selected_mask_semantics']=='True means selected for final execution'
                            addcheck(vid,slot,h,kind,s,baseline_sets[slot][h],tensor_path)
                        result[slot][kind]=sum(map(len,sets))
                status='verified_saved_dense_execution_and_unused_diagnostics' if is_dense else 'verified_actual_saved_selection'
            except (AssertionError,KeyError,FileNotFoundError,StopIteration) as e:
                errors.append(f'{type(e).__name__}: {e}')
                del CHECKS[before:]
                result={}
                status='N/A: required evidence missing or inconsistent'
        for layer in LAYERS:
            slots=[s for s in SLOTS if s.startswith(f'layer{layer:02d}_')]
            denom=sum(sum(map(len,baseline_sets[s])) for s in slots)
            row={'variant_id':vid,'layer':layer,'query_slots':2,'heads_per_slot':32,'baseline_allowed_count':denom,
                 'actual_selected_count':'','actual_selected_ratio':'','diagnostic_p095_count':'','diagnostic_p095_ratio':'',
                 'diagnostic_p090_count':'','diagnostic_p090_ratio':'','diagnostic_p085_count':'','diagnostic_p085_ratio':'','status':status}
            if result:
                kind='actual_forward_dense_all_legal' if is_dense else 'actual_forward_selected_keys'
                n=sum(result[s][kind] for s in slots); row.update(actual_selected_count=n,actual_selected_ratio=n/denom)
                if is_dense:
                    for percent in (95,90,85):
                        n=sum(result[s][f'offline_diagnostic_p{percent:03d}_unused_in_forward'] for s in slots)
                        row[f'diagnostic_p{percent:03d}_count']=n; row[f'diagnostic_p{percent:03d}_ratio']=n/denom
            rows.append(row)
        summaries.append({'variant_id':vid,'source_run':source,'gate_path':gpath,'status':status,'errors':errors,'checked_head_sets':len(CHECKS)-before})
        # Keep CPU memory bounded: discard per-variant tensor captures after membership checks.
        if source and source!=str(BASE): TRACE_CACHE.pop(source,None)
    assert len(rows)==126
    with (OUT/'TABLE_K_RATIOS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT/'K_SET_CHECKS.jsonl').open('w') as f:
        for r in CHECKS: f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
    lines=['# 第二组：42 组非 RS 实验的保存 K 集合比例','',
        'CPU-only；只读取已保存证据。每组一张小表，行仅为 Llama 层 0、16、31。', '',
        '比例 = Σ保存所选 K 位置 ID 的集合大小 / Σ INT-Flash 稠密基线的因果合法 K 位置 ID 集合大小；每层对两个捕获槽（query 127 prefill、query 128 decode）×32 个 query heads 加权合计。相同位置在不同槽/head 分别计数，不取跨 head 并集。每层分母是 32×(128+129)=8224。', '',
        'INT-Flash 基线没有保存重要性掩码或 TopP 集合；分母仅来自其 actual.pt 的 allowed_mask/key_positions，绝不把它称作 INT-Flash TopP 集合。这里的 ID 是 token 位置 ID，不是词表 token ID。', '',
        '36 组 TopP 的“实际所选”来自 actual.pt 的 selected_mask，并逐 head 核对 selected_indices、valid_mask（或 selected_valid_mask）、selected_counts 与 JSONL 集合。新增稠密 4 组的实际前向 execution_mask 使用所有合法 K，比例为 1；它们的 p=.95/.90/.85 列来自保存的 derived.pt 诊断集合，**未用于前向**。历史浮点/INT 稠密 2 组未保存重要 K 集合，填 N/A；不从概率重新制造集合。', '',
        '只描述六个 smoke 捕获槽，不是全数据集/全部层/所有 query 的比例；轨迹之间隐状态可能不同，该比例不是集合相似度、召回率或同输入估计器质量。完整性、成员关系和 SHA-256 核对见 K_MANIFEST.json 与 K_SET_CHECKS.jsonl。未读取或修改任何 RS 实验文件。', '']
    by=defaultdict(list)
    for r in rows: by[r['variant_id']].append(r)
    def fmt(r,key):
        ratio=r[key+'_ratio']; n=r[key+'_count']
        return 'N/A' if ratio=='' else f'{ratio:.6f} ({n}/{r["baseline_allowed_count"]})'
    for i,info in enumerate(order,1):
        vid=info['variant_id']; lines.extend([f'## {i:02d}. {vid}', '',info['name_zh'],''])
        if vid.startswith('dense_'):
            lines.extend(['| Llama 层 | 实际前向：全合法 K / INT 合法 K | p=.95 诊断（未用于前向） | p=.90 诊断（未用于前向） | p=.85 诊断（未用于前向） |','|---|---:|---:|---:|---:|'])
            for r in by[vid]: lines.append('| '+str(r['layer'])+' | '+' | '.join(fmt(r,k) for k in ('actual_selected','diagnostic_p095','diagnostic_p090','diagnostic_p085'))+' |')
        else:
            lines.extend(['| Llama 层 | 保存的实际重要 K 集合 / INT 合法 K |','|---|---:|'])
            for r in by[vid]: lines.append(f'| {r["layer"]} | {fmt(r,"actual_selected")} |')
        lines.append('')
        if vid in ('fp_common','intflash_qk8'): lines.extend(['N/A：历史稠密前向使用全部合法 K，但没有保存重要 K ID 集合。',''])
        err=next(s['errors'] for s in summaries if s['variant_id']==vid)
        if err: lines.extend(['N/A 证据问题：'+'; '.join(err),''])
    text='\n'.join(lines)+'\n'; assert text.count('| Llama 层 |')==42
    (OUT/'TABLE_K_RATIOS.md').write_text(text)
    metrics={'groups':42,'layer_rows':126,'top_p_groups':36,'new_dense_groups':4,'historical_dense_na_groups':2,
             'checked_head_sets':len(CHECKS),'baseline_capture_slots':len(baseline),'baseline_allowed_count_per_layer':8224,
             'evidence_error_groups':sum(bool(s['errors']) for s in summaries)}
    result_manifest={'schema':'nonrs-saved-k-ratios-v1','cpu_only':True,'model_loaded':False,'gpu_used':False,
        'rs_files_read_or_modified':False,'aggregation':'sum selected per-head set cardinalities / sum INT-Flash causal-allowed per-head set cardinalities',
        'denominator_semantics':'saved allowed_mask/key_positions; no INT-Flash importance/TopP set exists',
        'baseline_run':str(BASE),'capture_slots':list(SLOTS),'heads':list(range(32)),'metrics':metrics,'groups':summaries,
        'input_sha256':dict(sorted(INPUTS.items())), 'script_sha256':sha(Path(__file__)),
        'output_sha256':{n:sha(OUT/n) for n in ('TABLE_K_RATIOS.md','TABLE_K_RATIOS.csv','K_SET_CHECKS.jsonl')}}
    (OUT/'K_MANIFEST.json').write_text(json.dumps(result_manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(metrics,sort_keys=True)); print('ERRORS',[(r['variant_id'],r['errors']) for r in summaries if r['errors']])
    vals=[r['actual_selected_ratio'] for r in rows if r['actual_selected_ratio']!='' and not r['variant_id'].startswith('dense_')]
    print('TopP layer ratio range',min(vals) if vals else None,max(vals) if vals else None)

if __name__=='__main__': main()
