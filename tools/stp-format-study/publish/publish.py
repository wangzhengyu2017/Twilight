#!/usr/bin/env python3
"""Guarded post-delivery Git publisher. Default: validate and print an exact plan."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HEAD = '124b78ae3e542bba3dc66bb35e58048373892ced'
BRANCH = 'STP_test_Format'
ORIGIN = 'git@github.com:wangzhengyu2017/Twilight.git'
WRITER_SHA = 'c67f194b6c8b190c13c39a05d85769766fd2444c418338e01d07bad7f7bd6239'
MESSAGE = 'results: format study baselines and diagnostics'
DOC = Path('docs/stp-format-study')
TOOLS = Path('tools/stp-format-study')
PRODUCTS = ('REPORT.md', 'PPL_TABLE.csv', 'FULL_TABLE.csv', 'PAIRED_SETS.csv')
TEXT_SOURCES = {
    'diagnosis/REPORT.md': 'MSB_INFORMATION_ANALYSIS.md',
    'diagnosis-distributions/REPORT.md': 'SOURCE_DISTRIBUTIONS.md',
    'diagnosis-math/NOTE.md': 'MATHEMATICAL_NOTE.md',
    'diagnosis-math/SMOKE_SELECTION_BUDGET.md': 'SMOKE_SELECTION_BUDGET.md',
    'diagnosis-math/METRIC_DEFINITIONS.md': 'METRIC_DEFINITIONS.md',
    'supplement-proposal/SPEC.md': 'SUPPLEMENT_PROPOSAL.md',
    'supplement-proposal/REASONING.md': 'SUPPLEMENT_REASONING.md',
    'supplement-proposal/REVIEW.md': 'SUPPLEMENT_REVIEW.md',
}
CODE_SOURCES = {
    **{f'tools/{x}': f'orchestration/{x}' for x in ('README.md','orchestrate.py','freeze.py','activate_conditional.py','index_artifacts.py','progress.py','rebind_spec.py','controls.json')},
    **{f'delivery-tools/{x}': f'delivery/{x}' for x in ('README.md','DOC_RECONCILIATION.md','write_report.py','test_write_report.py','autostart.py')},
    'review-tools/independent_auditor_v1.py': 'review/independent_auditor_v1.py',
    'review-tools/independent_auditor_v1.provenance.json': 'review/independent_auditor_v1.provenance.json',
    'diagnosis/analyze_tc.py': 'diagnostics/analyze_tc.py',
    'diagnosis-math/compare_saved_layer0.py': 'diagnostics/compare_saved_layer0.py',
    'diagnosis-math/summarize_smoke_selection.py': 'diagnostics/summarize_smoke_selection.py',
    'publish-tools/publish.py': 'publish/publish.py',
    'publish-tools/README.md': 'publish/README.md',
    'publish-tools/test_publish.py': 'publish/test_publish.py',
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args], text=True).strip()


def git_guard(repo):
    need(git(repo,'branch','--show-current') == BRANCH, 'unexpected branch')
    need(git(repo,'rev-parse','HEAD') == HEAD, 'unexpected HEAD; preserve user changes')
    origin_guard(repo)
    need(not git(repo,'status','--porcelain=v1','--untracked-files=all'), 'dirty checkout; preserve user changes')


def origin_guard(repo):
    need(git(repo,'remote','get-url','--all','origin').splitlines() == [ORIGIN], 'unexpected fetch origins')
    need(git(repo,'remote','get-url','--push','--all','origin').splitlines() == [ORIGIN], 'unexpected push origins')


def merge_hashes(destination, mapping):
    for name,digest in mapping.items():
        need(name not in destination or destination[name] == digest, 'conflicting bound hash: '+name)
        destination[name] = digest


def safe_source(source, root):
    need(root in source.resolve().parents and root in source.parents, 'copy source escapes study: '+str(source))
    need(source.is_file() and not source.is_symlink() and not root.is_symlink(), 'missing/unsafe copy source: '+str(source))
    need(not any(parent.is_symlink() for parent in source.parents if root in parent.parents), 'symlink source parent: '+str(source))


def expected_cells(active):
    cells = {f'format_{kind}_q{q}k4_topp_p{threshold}':'full'
             for kind in ('sm8','mxint8_tc4') for q in (4,8) for threshold in ('095','090','085')}
    if active:
        cells.update({f'format_{kind}_topp_p{threshold}':'conditional-full'
                      for kind in ('mxfp4_q4k4','mxint8q_mxfp4k_q8k4') for threshold in ('095','090','085')})
    return cells


def audit_metrics(execution, active, cells, indexed, rows):
    expected = expected_cells(active)
    need(len(cells) == len(expected) and {c['variant_id'] for c in cells} == set(expected), 'audit exact variant set mismatch')
    lookup = {r['variant_id']:r for r in rows}
    need(len(lookup) == len(rows), 'duplicate delivery variant')
    for cell in cells:
        variant = cell['variant_id']
        phase = expected[variant]
        need(cell.get('phase') == phase, 'audit phase mismatch: '+variant)
        run = execution/phase/(variant+'-windows-full')
        path = run/'metrics.json'
        need(indexed.get(str(path)) == cell.get('metrics_sha256') == sha(path), 'audit metrics SHA mismatch: '+variant)
        metric = read(path)
        need(metric.get('variant_id') == variant, 'metric variant mismatch')
        for key in ('sum_nll','ppl','forward_nll_seconds','end_to_end_seconds'):
            need(cell.get(key) == metric.get(key), 'audit metric value mismatch: '+key)
        need(cell.get('targets') == metric.get('num_targets') == 289075 and cell.get('windows') == 282, 'audit target/window count mismatch')
        window_path = run/'window_metrics.jsonl'
        need(indexed.get(str(window_path)) == sha(window_path), 'audit window file not indexed')
        need(sum(bool(line.strip()) for line in window_path.read_text().splitlines()) == cell['windows'], 'audit actual window count mismatch')
        row = lookup.get(variant,{})
        need(row.get('status') == 'complete' and row.get('verified') == 'yes', 'audited cell missing/unverified in delivery: '+variant)
        need(row.get('metrics_path') == str(path), 'delivery metrics path mismatch')
        for column,key in (('sum_nll','sum_nll'),('num_targets','num_targets'),('ppl','ppl')):
            need(row.get(column) == str(metric[key]), 'delivery raw metric mismatch: '+column)


def hashes(mapping):
    need(isinstance(mapping,dict) and mapping, 'empty hash map')
    for name,digest in mapping.items():
        need(Path(name).is_absolute() and sha(name) == digest, 'hash mismatch: '+name)


def validate(study, repo):
    git_guard(repo)
    execution, delivery = study/'execution-v1', study/'delivery-v1'
    status = read(execution/'orchestration_status.json')
    need(status.get('status') == 'complete', 'execution not terminal complete')
    active = status.get('conditional_activated')
    need(type(active) is bool, 'missing conditional decision')
    closure = {}
    def bound_read(path):
        merge_hashes(closure,{str(path):sha(path)})
        return read(path)
    index = bound_read(execution/'artifact-index.json')
    need(index.get('schema_version') == 'format-artifact-index-v1' and Path(index['root']).resolve() == execution, 'invalid root artifact index')
    hashes(index['files'])
    merge_hashes(closure,index['files'])
    audit_path = study/'review/execution-v1-full-independent-audit.json'
    audit = bound_read(audit_path)
    expected_count = 18 if active else 12
    need(audit.get('status') == 'terminal_artifacts_independently_verified', 'independent audit not terminal pass')
    need(audit.get('conditional_activated') is active and audit.get('conditional_decision_independently_recomputed') is True, 'audit decision mismatch')
    need(audit.get('source_head') == HEAD and audit.get('artifact_index_sha256') == sha(execution/'artifact-index.json'), 'audit provenance/index mismatch')
    need(audit.get('artifact_index_files') == len(index['files']) and audit.get('full_cell_count') == expected_count, 'audit counts mismatch')
    cells = audit.get('verified_full_cells', [])
    need(len(cells) == expected_count and len({r['variant_id'] for r in cells}) == expected_count, 'audit cells incomplete/duplicate')
    need(audit.get('initial_trace_records') == 72 and audit.get('conditional_trace_records') == (36 if active else 0), 'audit trace counts mismatch')
    writer = study/'delivery-tools/write_report.py'
    need(sha(writer) == WRITER_SHA, 'writer differs from independently approved hash')
    merge_hashes(closure,{str(writer):WRITER_SHA})
    manifest = bound_read(delivery/'DELIVERY.manifest.json')
    need(manifest.get('schema') == 'stp-format-delivery-v1' and manifest.get('test_data_only') is False and manifest.get('writer_sha256') == WRITER_SHA, 'delivery not real reviewed writer bundle')
    hashes(manifest['source_files']); hashes(manifest['artifacts'])
    merge_hashes(closure,manifest['source_files']); merge_hashes(closure,manifest['artifacts'])
    with (delivery/'FULL_TABLE.csv').open(newline='') as stream:
        audit_metrics(execution,active,cells,index['files'],list(csv.DictReader(stream)))
    need(set(manifest['artifacts']) == {str(delivery/p) for p in PRODUCTS}, 'unexpected delivery artifacts')
    need(manifest['source_files'].get(str(execution/'artifact-index.json')) == sha(execution/'artifact-index.json'), 'delivery index binding absent')
    # Rebuild the complete report in disposable storage and compare every delivered byte.
    spec = importlib.util.spec_from_file_location('reviewed_writer', writer)
    module = importlib.util.module_from_spec(spec)
    # Disable bytecode writes: never add __pycache__ to frozen/external source trees.
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    with tempfile.TemporaryDirectory(prefix='stp-publish-validation-') as tmp:
        rebuilt = module.build(execution, study, Path(tmp)/'rebuilt')
        for name in PRODUCTS:
            need((rebuilt/name).read_bytes() == (delivery/name).read_bytes(), 'delivery differs from regenerated output: '+name)
        need(read(rebuilt/'DELIVERY.manifest.json')['source_files'] == manifest['source_files'], 'delivery source closure mismatch')
    need(read(execution/'freeze/code_manifest.json')['head'] == HEAD, 'execution source commit mismatch')
    # Require an actual completed writer process, in addition to the independently rebuilt bundle.
    state = bound_read(study/'preparation/delivery-autostart.json')
    need(state.get('status') == 'complete' and state.get('returncode') == 0, 'writer process success missing')
    hashes(closure)
    return status, audit_path, closure


def external_links(text, source):
    """Copied diagnostic prose points to its real external evidence, never guessed copies."""
    def replace(match):
        target = match.group(2)
        if target.startswith(('/', '#', 'http:', 'https:', 'mailto:')):
            return match.group(0)
        return match.group(1)+str((source.parent/target).resolve())+match.group(3)
    return re.sub(r'(\]\()([^\s)]+)(\))', replace, text)


def rebase_history_links(text):
    def replace(match):
        target = match.group(2)
        if target.startswith(('/', '#', 'http:', 'https:', 'mailto:')):
            return match.group(0)
        return match.group(1)+'../'+target+match.group(3)
    return re.sub(r'(\]\()([^\s)]+)(\))',replace,text)


def plan(study, repo, status, audit_path, closure=None):
    delivery = study/'delivery-v1'
    files, sources = {}, dict(closure or {})
    def copy(src, dst, rewrite=False):
        safe_source(src,study)
        merge_hashes(sources,{str(src):sha(src)})
        files[str(dst)] = external_links(src.read_text(),src).encode() if rewrite else src.read_bytes()
    for source,target in TEXT_SOURCES.items():
        copy(study/source, DOC/target, True)
    for source,target in CODE_SOURCES.items():
        copy(study/source, TOOLS/target)
    for name in PRODUCTS:
        copy(delivery/name,DOC/'results'/name)
    copy(delivery/'DELIVERY.manifest.json',DOC/'results/DELIVERY.manifest.json')
    copy(audit_path,DOC/'results/INDEPENDENT_AUDIT.json')
    copy(study/'review/delivery-writer-review.md',DOC/'results/WRITER_REVIEW.md',True)
    old_readme = repo/DOC/'README.md'
    need(old_readme.is_file() and not old_readme.is_symlink(), 'missing/unsafe previous README')
    # The prior README is bound by the required Git parent and copied-file digest.
    # Do not claim its old bytes remain at the current README path after publication.
    files[str(DOC/'history/pre-delivery.md')] = rebase_history_links(old_readme.read_text()).encode()
    with (delivery/'FULL_TABLE.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for row in rows:
        key = row['status']+'; verified='+row['verified']
        counts[key] = counts.get(key,0)+1
    summary = '\n'.join(f'- {k}: {v}' for k,v in sorted(counts.items()))
    intro = (f'# Format study: delivered results\n\nExecution is terminal complete. Conditional FP4 activated: {str(status["conditional_activated"]).lower()}. '
             f'Independent terminal audit verified {18 if status["conditional_activated"] else 12} format full cells. '
             'RS-INT8 remains blocked; proposed other-cause supplements remain unapproved and unexecuted.\n\n'
             f'Execution source commit: `{HEAD}`. The later delivery commit records documentation and reproducibility tools; it is not the source used by these runs.\n\n'
             f'{summary}\n\n'
             '[Actual report and per-threshold signed CE changes](results/REPORT.md), '
             '[unified PPL table](results/PPL_TABLE.csv), [full precision/status table](results/FULL_TABLE.csv), '
             '[paired sets](results/PAIRED_SETS.csv), [delivery hashes](results/DELIVERY.manifest.json), '
             '[independent audit](results/INDEPENDENT_AUDIT.json).\n\n'
             '[Main cause analysis](MSB_INFORMATION_ANALYSIS.md), [source distributions](SOURCE_DISTRIBUTIONS.md), '
             '[mathematics](MATHEMATICAL_NOTE.md), [selection budget](SMOKE_SELECTION_BUDGET.md), '
             '[metric definitions](METRIC_DEFINITIONS.md), [proposed supplemental specification](SUPPLEMENT_PROPOSAL.md), '
             '[proposal reasoning](SUPPLEMENT_REASONING.md), [proposal review](SUPPLEMENT_REVIEW.md), '
             '[history](history/README.md).\n\n'
             f'Authoritative external execution: `{study}/execution-v1`; delivery: `{delivery}`. '
             'Machine-local absolute evidence links and manifests require those archived artifacts. '
             'Historical dense rows retain their parser-matched gate limitation. '
             'Same-input estimator comparisons and cross-run trajectories have separate scopes; no unique corpus-level causal attribution or native integer speedup is claimed. '
             'Original full Q8/K8 final operands and one Top-p selection remain the execution semantics.\n')
    files[str(DOC/'README.md')] = intro.encode()
    files[str(DOC/'history/README.md')] = ('# Format study history\n\n'
        '[Preparation snapshot](pre-delivery.md) records implemented versus pending scope before delivery. '
        '[Current completed delivery](../README.md) supersedes that status using validated actual evidence. '
        f'Execution was frozen at `{HEAD}`; this delivery is a later documentation/code snapshot, not a new experiment round.\n').encode()
    provenance = {'schema':'stp-format-publication-v1','execution_head':HEAD,'publication_parent':HEAD,
                  'conditional_activated':status['conditional_activated'],'source_files':sources,
                  'files':{p:hashlib.sha256(b).hexdigest() for p,b in sorted(files.items())}}
    files[str(DOC/'results/PUBLICATION.manifest.json')] = (json.dumps(provenance,indent=2,sort_keys=True)+'\n').encode()
    return files, sources


def approval(path):
    need(path is not None, '--execute requires root review approval JSON')
    value = read(path)
    need(value.get('approved') is True and value.get('scope') == 'execute-format-study-publication', 'root approval scope missing')
    need(value.get('publisher_sha256') == sha(__file__), 'root approval does not bind current publisher')


def execute(study, repo, files, sources, receipt_dir):
    need(not receipt_dir.exists() and repo not in receipt_dir.parents and study/'execution-v1' not in receipt_dir.parents, 'receipt directory must be new and outside checkout/execution')
    git_guard(repo)
    hashes(sources)
    # Refuse overwrite of every newly introduced destination before creating any file.
    tracked = set(git(repo,'ls-files').splitlines())
    for name in files:
        dest = repo/name
        need(not dest.is_symlink() and (name in tracked or not dest.exists()), 'destination collision: '+name)
        need(repo in dest.resolve().parents, 'destination escapes repository')
        need(not any(parent.is_symlink() for parent in dest.parents if parent != repo and repo in parent.parents), 'symlink destination parent: '+name)
    receipt_dir.mkdir(parents=True)
    (receipt_dir/'plan.json').write_text(json.dumps({'paths':sorted(files),'source_files':sources},indent=2)+'\n')
    for name,data in files.items():
        dest = repo/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        need(sha(dest) == hashlib.sha256(data).hexdigest(), 'copied file hash mismatch: '+name)
    hashes(sources)
    need(git(repo,'rev-parse','HEAD') == HEAD and git(repo,'branch','--show-current') == BRANCH, 'checkout identity changed during copy')
    changed = set(git(repo,'diff','--name-only').splitlines()) | set(git(repo,'diff','--cached','--name-only').splitlines()) | set(git(repo,'ls-files','--others','--exclude-standard').splitlines())
    need(changed <= set(files), 'concurrent user changes outside whitelist; stop preserving changes')
    git(repo,'add','--',*sorted(files))
    staged = set(git(repo,'diff','--cached','--name-only').splitlines())
    need(staged and staged <= set(files), 'unexpected staged paths; stop preserving changes')
    identity = git(repo,'show','-s','--format=%an%n%ae',HEAD).splitlines()
    opts = []
    for key,value in zip(('user.name','user.email'),identity):
        result = subprocess.run(['git','-C',str(repo),'config','--get',key],capture_output=True,text=True)
        if result.returncode != 0 or not result.stdout.strip():
            opts += ['-c',key+'='+value]
    need(git(repo,'rev-parse','HEAD') == HEAD and git(repo,'branch','--show-current') == BRANCH, 'checkout identity changed before commit')
    git(repo,*opts,'commit','-m',MESSAGE)
    commit = git(repo,'rev-parse','HEAD')
    need(git(repo,'rev-parse','HEAD^') == HEAD, 'unexpected publication parent')
    need(not git(repo,'status','--porcelain=v1','--untracked-files=all'), 'checkout dirty after commit; no push')
    (receipt_dir/'committed.json').write_text(json.dumps({'status':'committed_push_pending','commit':commit,'execution_head':HEAD},indent=2)+'\n')
    origin_guard(repo)
    git(repo,'push','origin',BRANCH)
    remote = git(repo,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    need(remote == commit, 'remote HEAD differs from publication commit')
    status = git(repo,'status','--porcelain=v1','--untracked-files=all')
    receipt = {'status':'pushed_verified','commit':commit,'remote_head':remote,'execution_head':HEAD,
               'branch':BRANCH,'origin':ORIGIN,'git_status':status,'files':{p:sha(repo/p) for p in files}}
    (receipt_dir/'publication.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    need(not status,'dirty checkout after push')
    print(json.dumps(receipt,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,default=Path('/output/project/experiments/stp-format-study'))
    parser.add_argument('--repo',type=Path,default=Path('/output/project/Twilight-STP_test_Format'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--root-review',type=Path)
    parser.add_argument('--receipt-dir',type=Path)
    args = parser.parse_args()
    try:
        study,repo = args.study_root.resolve(),args.repo.resolve()
        need(study != repo and repo not in study.parents and study not in repo.parents, 'study and checkout must be separate')
        if args.execute:
            approval(args.root_review)
            need(args.receipt_dir is not None,'--execute requires --receipt-dir')
        status,audit,closure = validate(study,repo)
        files,sources = plan(study,repo,status,audit,closure)
        print(json.dumps({'mode':'execute' if args.execute else 'dry-run','paths':sorted(files),
                          'planned_sha256':{p:hashlib.sha256(b).hexdigest() for p,b in files.items()}},indent=2))
        if args.execute:
            execute(study,repo,files,sources,args.receipt_dir.resolve())
        return 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as exc:
        print('REFUSED: '+str(exc),file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
