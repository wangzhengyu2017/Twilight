"""Combine immutable run metrics and evaluate the predeclared MX decision."""
import argparse
import csv
import json
import math
from pathlib import Path
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from benchmark.attention_slice.summarize import comparable,valid_metrics


def verified_full(run, files):
    """Rebind archived full evidence without requiring today's source to equal history."""
    def bind(path):
        path=Path(path).resolve()
        if not path.is_file():raise ValueError('Missing evidence '+str(path))
        files[str(path)]=sha256_file(path)
        return load_json(path)
    m=bind(run/'metrics.json');status=bind(run/'status.json')
    if status.get('status')!='complete' or m.get('profile')!='full' or m.get('mode')!='windows':raise ValueError('Incomplete full run')
    valid_metrics(m,'windows');assets=bind(run/'asset_manifest.json');coverage=bind(run/'coverage.json')
    if m['num_targets']!=assets['num_tokens']-1 or m['num_targets']!=assets['windows']['expected_targets']:raise ValueError('Asset target coverage')
    if canonical_hash(assets)!=m['asset_manifest_sha256'] or assets['tokens_sha256']!=m['tokens_sha256']:raise ValueError('Asset identity')
    ranges=coverage['scoring_ranges']
    if canonical_hash(ranges)!=m['scoring_ranges_sha256'] or coverage['num_targets']!=m['num_targets'] or not coverage['all_layers_covered']:raise ValueError('Coverage identity')
    next_target=1
    for row in ranges:
        if row['lo']!=next_target or row['target_end']<=row['lo']:raise ValueError('Coverage gap/overlap')
        next_target=row['target_end']
    if next_target!=assets['num_tokens']:raise ValueError('Tail missing')
    rows_path=run/'window_metrics.jsonl';files[str(rows_path.resolve())]=sha256_file(rows_path)
    rows=[json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
    if len(rows)!=len(ranges):raise ValueError('Window count mismatch')
    for i,(row,expected) in enumerate(zip(rows,ranges)):
        if any(row[k]!=v for k,v in expected.items()) or row['num_targets']!=expected['target_end']-expected['lo'] or not row.get('logits_sha256'):raise ValueError('Window evidence mismatch')
    if sum(r['num_targets'] for r in rows)!=m['num_targets'] or sum(r['sum_nll'] for r in rows)!=m['sum_nll']:raise ValueError('Full totals mismatch')
    code=bind(run/'code_manifest.json');env=bind(run/'environment.json');config=bind(run/'resolved_config.json')['algorithm']
    patch=run/'code.patch';files[str(patch.resolve())]=sha256_file(patch)
    if canonical_hash({'head':code['head'],'files':code['files']})!=code['code_sha256'] or sha256_file(patch)!=code['patch_sha256'] or canonical_hash({k:v for k,v in env.items() if k!='environment_sha256'})!=env['environment_sha256']:raise ValueError('Invalid archived code/environment digest')
    if code['code_sha256']!=m['code_sha256'] or env['environment_sha256']!=m['environment_sha256'] or canonical_hash(config)!=m['semantic_config_sha256']:raise ValueError('Full provenance mismatch')
    from benchmark.attention_slice.gate import check_run
    identities={k:m[k] for k in ('code_sha256','environment_sha256','asset_manifest_sha256','tokens_sha256')}
    check_run(run,config,'full','windows',identities,assets['num_tokens']-1)
    for name in ('method_manifest.json','quantization_diagnostics.json','selection_stats.json','costs.json'):
        if (run/name).exists():bind(run/name)
    gate_paths=[]
    ref=run.parent/'gate_reference.json'
    if ref.exists():gate_paths.append(Path(bind(ref)['path']))
    commands=run.parent/'commands.jsonl'
    if commands.exists():
        files[str(commands.resolve())]=sha256_file(commands)
        for line in commands.read_text().splitlines():
            argv=json.loads(line)['argv']
            if '--output-dir' in argv and Path(argv[argv.index('--output-dir')+1]).resolve()==run.resolve():
                for flag in ('--slice-gate-report','--gate-report'):
                    if flag in argv:gate_paths.append(Path(argv[argv.index(flag)+1]))
    matches=[path for path in gate_paths if path.is_file() and sha256_file(path)==m.get('gate_report_sha256')]
    if not matches:raise ValueError('No matching archived full release gate')
    gate=bind(matches[0])
    if gate.get('status')!='complete' or gate.get('passed') is not True:raise ValueError('Full gate failed')
    for key in ('code_sha256','environment_sha256','asset_manifest_sha256','tokens_sha256'):
        if gate[key]!=m[key]:raise ValueError('Gate/full identity '+key)
    hashes=gate.get('semantic_config_sha256')
    if isinstance(hashes,dict) and hashes.get(m.get('variant_id',m.get('method')))!=canonical_hash(config):raise ValueError('Config not in gate')
    for name,digest in gate.get('evidence_files',{}).items():
        path=Path(name)
        if not path.is_file() or sha256_file(path)!=digest:raise ValueError('Changed gate evidence '+name)
        files[str(path.resolve())]=digest
    m=dict(m,verified=True,expected_targets=assets['num_tokens']-1,num_windows=len(ranges))
    return m


def decision(records):
    try:
        lookup={}
        for r in records:
            if r.get('verified') is True:
                if r['variant_id'] in lookup:raise ValueError('Duplicate verified variant '+r['variant_id'])
                lookup[r['variant_id']]=r
        values={}
        for q in (4,8):
            groups={}
            for kind,pattern in [('mx',f'format_mxint8_tc4_q{q}k4_topp_p'),('tc',f'msb_q{q}k4_topp_p'),('int','int_topp_p')]:
                rows=[lookup[pattern+p] for p in ('095','090','085')]
                for r in rows:
                    if r['num_targets']!=r['expected_targets'] or not math.isfinite(r['sum_nll']):raise ValueError('invalid full coverage')
                groups[kind]=sum(r['sum_nll']/r['num_targets'] for r in rows)/3
            for p in ('095','090','085'):
                comparable(lookup[f'format_mxint8_tc4_q{q}k4_topp_p'+p],lookup[f'msb_q{q}k4_topp_p'+p]);comparable(lookup[f'format_mxint8_tc4_q{q}k4_topp_p'+p],lookup['int_topp_p'+p])
            values[f'q{q}k4']=dict(mean_ce=groups,delta_ce_vs_tc=groups['mx']-groups['tc'],delta_ce_vs_int=groups['mx']-groups['int'],ppl_ratio_vs_tc=math.exp(groups['mx']-groups['tc']),ppl_ratio_vs_int=math.exp(groups['mx']-groups['int']),no_tc_improvement=groups['mx']>=groups['tc'],above_int_margin=groups['mx']-groups['int']>math.log(1.01))
        environments={r['environment_sha256'] for key,r in lookup.items() if key.startswith(('format_mxint8_tc4_','msb_','int_topp_'))}
        if len(environments)!=1:raise ValueError('Decision environment mismatch')
        primary=values['q4k4'];return dict(status='complete',activate_mxfp4=primary['no_tc_improvement'] or primary['above_int_margin'],primary='q4k4',widths=values)
    except (KeyError,ValueError,TypeError,OverflowError) as error:return dict(status='decision_unavailable',reason=str(error),activate_mxfp4=None)


def failed_record(run,status,metrics,files):
    resolved=load_json(run/'resolved_config.json') if (run/'resolved_config.json').exists() else {}
    cfg=resolved.get('algorithm',{});sl=cfg.get('attention_slice',{})
    row=dict(variant_id=sl.get('variant_id',metrics.get('variant_id',metrics.get('method',cfg.get('attention_quantization',{}).get('method','unknown')))),status=status.get('status','failed'),verified=False,reason=status.get('reason','incomplete execution'),stage=status.get('stage',''),run=str(run.resolve()),log=str((run/'run.log').resolve()),profile=metrics.get('profile',resolved.get('profile','unknown')),mode=metrics.get('mode',resolved.get('ppl_mode','unknown')))
    for name in ('status.json','resolved_config.json','metrics.json','run.log'):
        path=run/name
        # A running log remains mutable and cannot join a final evidence index.
        if path.exists() and (name!='run.log' or status.get('status')=='failed'):files[str(path.resolve())]=sha256_file(path)
    return row


def summarize(matrix,runs_roots,output):
    spec=load_json(matrix);records=[];files={};seen=set()
    for root in runs_roots:
        for mp in sorted(Path(root).rglob('metrics.json')):
            run=mp.parent
            if str(run.resolve()) in seen:continue
            seen.add(str(run.resolve()))
            status=load_json(run/'status.json');m=load_json(mp)
            if status.get('status')!='complete':
                records.append(failed_record(run,status,m,files))
                continue
            valid_metrics(m,m.get('mode','windows'))
            if m.get('mode')!='windows' or m.get('profile')!='full':continue
            # Completed scoring is distinct from accepted smoke/replay evidence.
            verified=False
            try:
                m=verified_full(run,files);verified=True
            except (ValueError,KeyError,AssertionError,FileNotFoundError) as error:
                m=dict(m,verification_error=str(error))
            row=dict(m,run=str(run.resolve()),status='complete',verified=verified)
            if not row.get('variant_id'):row['variant_id']=m.get('method','unknown')
            stats_path=run/'selection_stats.json'
            if stats_path.exists():
                stats=load_json(stats_path)
                for key in ('num_rows','selected_count','allowed_count','mean_num_selected','selected_fraction'):
                    if key in stats:row['selection_'+key]=stats[key]
            records.append(row)
            for path in run.iterdir():
                if path.is_file() and path.name not in ('run.log',):files[str(path.resolve())]=sha256_file(path)
    for root in runs_roots:
        for status_path in Path(root).rglob('status.json'):
            run=status_path.parent
            if (run/'metrics.json').exists() or str(run.resolve()) in seen:continue
            seen.add(str(run.resolve()))
            if not (run/'resolved_config.json').exists():continue
            status=load_json(status_path);resolved=load_json(run/'resolved_config.json')
            if status.get('status')=='complete':continue
            cfg=resolved['algorithm'];sl=cfg.get('attention_slice',{})
            records.append(failed_record(run,status,{},files))
            files[str(status_path.resolve())]=sha256_file(status_path)
            files[str((run/'resolved_config.json').resolve())]=sha256_file(run/'resolved_config.json')
    existing={r.get('variant_id') for r in records}
    for row in spec['variants']:
        if row['variant_id'] not in existing:records.append(dict(variant_id=row['variant_id'],status=row['status'],verified=False))
    lookup={}
    for row in records:
        if row.get('verified'):
            vid=row['variant_id']
            if vid in lookup:raise ValueError('Ambiguous duplicate full variant '+vid)
            lookup[vid]=row
    for variant in spec['variants']:
        for row in records:
            if row['variant_id']!=variant['variant_id']:continue
            for key in ('predictor_format','p','q_total_prefix_bits','k_total_prefix_bits'):row[key]=variant[key]
            if not row.get('verified'):continue
            suffix=variant['variant_id'][-3:]
            for label,refid in [('int','int_topp_p'+suffix),('tc',f'msb_q{variant["q_total_prefix_bits"]}k4_topp_p'+suffix)]:
                ref=lookup.get(refid)
                if ref:
                    comparable(row,ref)
                    if row['environment_sha256']!=ref['environment_sha256']:raise ValueError('Reference environment mismatch')
                    row['delta_nll_vs_'+label]=row['sum_nll']-ref['sum_nll']
                    row['delta_ppl_vs_'+label]=row['ppl']-ref['ppl']
    for root in runs_roots:
        for path in Path(root).rglob('pairing_manifest.json'):
            report=load_json(path)
            if report.get('status')!='complete' or report.get('passed') is not True:continue
            for name,digest in {**report['source_files'],**report['artifacts']}.items():
                if sha256_file(name)!=digest:raise ValueError('Changed pairing evidence '+name)
                files[str(Path(name).resolve())]=digest
            files[str(path.resolve())]=sha256_file(path)
            for row in records:
                if row.get('variant_id')!=report['candidate_variant_id']:continue
                for role,metrics in report['summary'].items():
                    for metric,stats in metrics.items():
                        for stat,value in stats.items():row[f'trace_{role}_{metric}_{stat}']=value
                row['trace_scope']='192 heads per reference; separate from corpus scoring'
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    fields=['variant_id','status','verified','sum_nll','num_targets','ppl','run']
    fields+=sorted(set().union(*(set(r) for r in records))-set(fields))
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    write_json(out.with_suffix('.decision.json'),decision(records))
    write_json(out.with_suffix('.manifest.json'),dict(matrix_sha256=sha256_file(matrix),source_files=files,output_sha256=sha256_file(out),decision_sha256=sha256_file(out.with_suffix('.decision.json'))))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--runs-root',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();summarize(a.matrix,a.runs_root,a.output)
if __name__=='__main__':main()
