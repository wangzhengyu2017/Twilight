"""Dense measurement table; failed/unexecuted values stay null."""
import argparse
import csv
import json
import math
from pathlib import Path
from benchmark.attention_baseline.common import load_json,write_json,sha256_file
from .formal import read_matrix
from .oracle import check

DEFINITIONS={'sm8':'Full sign plus seven magnitude bits; actual stored byte decoding and original per-token FP16 source scale',
             'mxint8':'Block32 symmetric MXINT8 full signed codes plus E8M0; reconstructed c*2**(e-6)',
             'mxint8q_mxfp4k':'Independent block32 full MXINT8 Q and native E2M1 FP4 K, each with E8M0',
             'mxfp4':'Native E2M1 FP4 Q/K codes in uint8 containers, block32 E8M0; no MSB truncation'}


def summarize(matrix,runs_root,output_dir,profile='full',baselines_manifest=None):
    _,configs=read_matrix(matrix);root=Path(runs_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[];evidence={};suffix='windows-full' if profile=='full' else 'windows'
    for vid,cfg in configs.items():
        s=cfg['attention_dense_format'];run=root/(vid+'-'+suffix);status=load_json(run/'status.json') if (run/'status.json').is_file() else dict(status='not_executed')
        row=dict(variant_id=vid,format=s['format'],q_bits=s['q_bits'],k_bits=s['k_bits'],selection_mode='dense',selection_passes=0,profile=profile,status=status['status'],measurement_state='not_measured',validation_state='not_accepted',sum_nll=None,num_targets=None,ppl=None,representation=DEFINITIONS[s['format']],run_dir=str(run))
        if status['status']=='complete':
            m=load_json(run/'metrics.json');expected=289075 if profile=='full' else 4095
            check(m['num_targets']==expected and m['selection_passes']==0 and m['selection_mode']=='dense','measurement protocol differs')
            check(math.isfinite(m['sum_nll']) and m['ppl']==math.exp(m['sum_nll']/expected),'measurement PPL differs')
            row.update(sum_nll=m['sum_nll'],num_targets=m['num_targets'],ppl=m['ppl'],measurement_state='measured',source_code_sha256=m['code_sha256'],semantic_config_sha256=m['semantic_config_sha256'])
            gate=(load_json(root/'gate_reference.json') if (root/'gate_reference.json').is_file() else None)
            if gate:
                check(sha256_file(gate['path'])==gate['sha256'],'full gate binding changed');row['validation_state']='passed_smoke_gate_bound'
            elif (root/'gate.json').is_file() and load_json(root/'gate.json').get('passed') is True:row['validation_state']='passed_smoke_gate'
            for name in ('metrics.json','status.json','resolved_config.json','method_manifest.json','coverage.json','selection_stats.json','costs.json','code_manifest.json'):
                path=run/name;evidence[str(path)]=sha256_file(path)
        rows.append(row)
    controls=load_json(baselines_manifest) if baselines_manifest else {role:str(Path('/output/project/experiments/stp-baselines/20261027-v5-full')/(method+'-windows-full')) for role,method in [('fp','fp_common'),('source_int','intflash_qk8')]}
    for role in ('fp','source_int'):
        path=Path(controls[role]).resolve() if controls.get(role) else None
        row=dict(variant_id=role+'_dense_control',format='original_float' if role=='fp' else 'original_source_INT8',q_bits=None,k_bits=None,selection_mode='dense',selection_passes=0,profile='full',status='not_supplied',measurement_state='not_measured',validation_state='historical_control_not_supplied',sum_nll=None,num_targets=None,ppl=None,representation='Existing frozen dense control; never rerun by dense format summarizer',run_dir=str(path) if path else None)
        if path:
            cfg=load_json(path/'resolved_config.json')['algorithm']
            check('attention_slice' not in cfg and 'attention_dense_format' not in cfg and cfg['attention_quantization']['method']==('fp_common' if role=='fp' else 'intflash_qk8'),'historical baseline role must be genuine dense FP/source-INT')
            check(load_json(path/'status.json')['status']=='complete','historical baseline incomplete');m=load_json(path/'metrics.json');check(m['num_targets']==289075 and m['profile']=='full' and m['mode']=='windows','historical full baseline protocol')
            row.update(status='complete',measurement_state='historical_measured',validation_state='frozen_control',sum_nll=m['sum_nll'],num_targets=m['num_targets'],ppl=m['ppl'],source_code_sha256=m['code_sha256'])
            for name in ('status.json','metrics.json','resolved_config.json','code_manifest.json'):
                evidence[str(path/name)]=sha256_file(path/name)
        rows.append(row)
    csvpath=out/'BASELINE.csv'
    with csvpath.open('w',newline='') as stream:
        fields=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    markdown=['# Dense format measurements','',f'Profile: {profile}. Diagnostic TopP sets at .95/.90/.85 are unused in forward. No native low-bit kernel or speedup is claimed.','', '| Variant | Status | Validation | Targets | NLL | PPL |','|---|---|---|---:|---:|---:|']
    for r in rows:
        cell=lambda x:'null' if x is None else (f'{x:.10g}' if isinstance(x,float) else str(x))
        markdown.append('| '+' | '.join(cell(r[k]) for k in ('variant_id','status','validation_state','num_targets','sum_nll','ppl'))+' |')
    markdown+=['','Representation definitions:','']+[f"- {vid}: {DEFINITIONS[cfg['attention_dense_format']['format']]}" for vid,cfg in configs.items()]
    markdown+=['','Same-input codec counterfactuals and cross-run trajectories are stored as separate comparison kinds in the pairing evidence. Missing measurements remain null.']
    (out/'BASELINE.md').write_text('\n'.join(markdown)+'\n')
    manifest=dict(schema_version='dense-format-summary-v1',status='complete',profile=profile,rows=rows,source_files=evidence,artifacts={str(p):sha256_file(p) for p in (csvpath,out/'BASELINE.md')})
    write_json(out/'summary_manifest.json',manifest);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('matrix','runs-root','output-dir'):p.add_argument('--'+flag,required=True)
    p.add_argument('--profile',choices=['smoke','full'],default='full');p.add_argument('--baselines-manifest');a=p.parse_args();summarize(a.matrix,a.runs_root,a.output_dir,a.profile,a.baselines_manifest)
if __name__=='__main__':main()
