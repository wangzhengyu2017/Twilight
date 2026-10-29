"""Read-only hashes and saved-set validation for the historical RS controls."""
import argparse
from pathlib import Path
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from benchmark.attention_dense_format.pair_sets import historical_records,identity,bind_run
from .oracle import check,compare,selection,source_quantization
from .historical import validate_source_final

def audit(manifest):
    import torch
    torch.set_num_threads(2)
    controls=load_json(manifest);sources={str(Path(manifest).resolve()):sha256_file(manifest)};groups={};reports=[];unavailable_dense={}
    for vid,mapping in controls['top_p'].items():
        for role,path in mapping.items():
            root=Path(path).resolve()
            if role=='tc':
                from benchmark.attention_slice.replay import load_trace_records
            else:
                from benchmark.attention_format.replay import load_trace_records
            records=load_trace_records(root);bind_run(root,records,sources);keys=set()
            check(load_json(root/'status.json')['status']=='complete','old control incomplete');code=load_json(root/'code_manifest.json');metrics=load_json(root/'metrics.json')
            check(code['code_sha256']==metrics['code_sha256']==canonical_hash(dict(head=code['head'],files=code['files'])) and sha256_file(root/'code.patch')==code['patch_sha256'],'old control code provenance')
            aliases=[]
            for entry,meta,a,d in records:
                aliases.append(validate_source_final(meta,role))
                keys.add(canonical_hash(identity(meta,a)));p=meta['attention_slice']['selection']['p'];mask=selection(a['estimator_logits'],a['allowed_mask'],p)[0];compare(a['selected_mask'],mask,'old actual selector membership',True)
                for prefix,original in [('q',a['q_post_rope']),('k',a['k_visible'])]:
                    c,s=source_quantization(original);compare(a[prefix+'_codes'],c,'old source codes',True);compare(a[prefix+'_scales'],s,'old source scales',True)
            groups[vid+':'+role]=keys;reports.append(dict(candidate=vid,role=role,path=str(root),actual_records=len(records),saved_sets_validated=True,source_final_aliases=aliases,code_sha256=metrics['code_sha256']))
    for role,path in controls['dense'].items():
        root=Path(path).resolve();records=historical_records(root);bind_run(root,records,sources);dense_keys={canonical_hash(identity(meta,a)) for _,meta,a,_ in records}
        if dense_keys!=next(iter(groups.values())):unavailable_dense[role]=dict(reason='strict environment/protocol mismatch',driver=load_json(root/'environment.json').get('driver'),environment_sha256=records[0][1]['environment_sha256'],source_run=str(root))
        else:groups[role]=dense_keys
        reports.append(dict(role=role,path=str(root),actual_records=6,important_sets_already_saved=False,strict_cross_run_compatible=role not in unavailable_dense,diagnostic_derivation_policy='only offline from actual saved scores/probabilities; no claim of preexisting sets'))
    first=next(iter(groups.values()));check(all(keys==first for keys in groups.values()),'old controls full protocol identities differ')
    return dict(status='complete',passed=True,read_only=True,gpu_used=False,model_runs=0,compatible_control_groups=len(groups),unavailable_dense_cross_run=unavailable_dense,actual_existing_records=sum(row['actual_records'] for row in reports),reports=reports,compatible_environment_sha256=load_json(next(iter(controls['top_p'].values()))['tc']+'/environment.json')['environment_sha256'],compatible_driver=load_json(next(iter(controls['top_p'].values()))['tc']+'/environment.json')['driver'],source_files=sources)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:report=audit(a.manifest)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
