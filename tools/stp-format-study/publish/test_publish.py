#!/usr/bin/env python3
"""CPU-only negative guards; all fixtures live in disposable TEST-labelled directories."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('publisher',Path(__file__).with_name('publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class Guards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='TEST-stp-publisher-')
        self.root = Path(self.tmp.name)
        self.study = self.root/'study'
        (self.study/'execution-v1').mkdir(parents=True)
    def tearDown(self):
        self.tmp.cleanup()
    def put(self,name,value):
        path = self.study/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    def validate(self):
        with patch.object(p,'git_guard'):
            return p.validate(self.study,self.root/'repo')
    def terminal(self):
        self.put('execution-v1/orchestration_status.json',dict(status='complete',conditional_activated=False))
    def index(self):
        path=self.study/'execution-v1/orchestration_status.json'
        self.put('execution-v1/artifact-index.json',dict(schema_version='format-artifact-index-v1',root=str(path.parent),files={str(path):p.sha(path)}))
    def test_running_refused(self):
        self.put('execution-v1/orchestration_status.json',dict(status='running'))
        with self.assertRaisesRegex(ValueError,'not terminal'):self.validate()
        self.assertFalse((self.study/'delivery-v1').exists())
    def test_missing_terminal_index_refused(self):
        self.terminal()
        with self.assertRaises(FileNotFoundError):self.validate()
    def test_index_hash_corruption_refused(self):
        self.terminal();self.index()
        self.put('execution-v1/orchestration_status.json',dict(status='complete',conditional_activated=True))
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.validate()
    def test_partial_audit_refused(self):
        self.terminal();self.index()
        self.put('review/execution-v1-full-independent-audit.json',dict(status='partial_until_terminal_verified'))
        with self.assertRaisesRegex(ValueError,'not terminal pass'):self.validate()
    def test_approval_required(self):
        with self.assertRaisesRegex(ValueError,'requires root'):p.approval(None)
    def test_changed_publisher_approval_refused(self):
        path=self.root/'review.json'
        path.write_text(json.dumps(dict(approved=True,scope='execute-format-study-publication',publisher_sha256='0'*64)))
        with self.assertRaisesRegex(ValueError,'does not bind'):p.approval(path)
    def test_wrong_branch_head_dirty_origin_refused(self):
        base={('branch','--show-current'):p.BRANCH,('rev-parse','HEAD'):p.HEAD,
              ('remote','get-url','--all','origin'):p.ORIGIN,('remote','get-url','--push','--all','origin'):p.ORIGIN,
              ('status','--porcelain=v1','--untracked-files=all'):''}
        for key,bad in [(('branch','--show-current'),'user-branch'),(('rev-parse','HEAD'),'bad'),
                        (('remote','get-url','--all','origin'),'https://example.invalid/repo'),
                        (('remote','get-url','--push','--all','origin'),'https://example.invalid/repo'),
                        (('status','--porcelain=v1','--untracked-files=all'),' M user.txt')]:
            answers=dict(base);answers[key]=bad
            with self.subTest(key=key),patch.object(p,'git',side_effect=lambda repo,*args:answers[args]):
                with self.assertRaises(ValueError):p.git_guard(self.root)
    def test_extra_origin_urls_refused(self):
        for fetch,push in [(p.ORIGIN+'\nhttps://example.invalid/repo',p.ORIGIN),
                           (p.ORIGIN,p.ORIGIN+'\ngit@example.invalid:other/repo')]:
            with patch.object(p,'git',side_effect=lambda repo,*args:push if '--push' in args else fetch):
                with self.assertRaises(ValueError):p.origin_guard(self.root)
    def audit_fixture(self,active=False):
        cells,rows,indexed=[],[],{}
        for variant,phase in p.expected_cells(active).items():
            run=self.study/'execution-v1'/phase/(variant+'-windows-full')
            run.mkdir(parents=True)
            metric=dict(variant_id=variant,sum_nll=123.25,ppl=2.25,num_targets=289075,forward_nll_seconds=3.25,end_to_end_seconds=4.25)
            path=run/'metrics.json';path.write_text(json.dumps(metric))
            windows=run/'window_metrics.jsonl';windows.write_text('{}\n'*282)
            indexed[str(path)]=p.sha(path);indexed[str(windows)]=p.sha(windows)
            cells.append(dict(variant_id=variant,phase=phase,metrics_sha256=p.sha(path),targets=289075,windows=282,**{k:metric[k] for k in ('sum_nll','ppl','forward_nll_seconds','end_to_end_seconds')}))
            rows.append(dict(variant_id=variant,status='complete',verified='yes',metrics_path=str(path),sum_nll='123.25',ppl='2.25',num_targets='289075'))
        return cells,indexed,rows
    def test_exact_initial_and_conditional_audit_binding(self):
        cells,indexed,rows=self.audit_fixture(True)
        p.audit_metrics(self.study/'execution-v1',True,cells,indexed,rows)
        cells[-1]['phase']='full'
        with self.assertRaisesRegex(ValueError,'phase mismatch'):p.audit_metrics(self.study/'execution-v1',True,cells,indexed,rows)
    def test_unexpected_audit_variant_refused(self):
        cells,indexed,rows=self.audit_fixture()
        cells[-1]['variant_id']='format_unknown'
        with self.assertRaisesRegex(ValueError,'exact variant set'):p.audit_metrics(self.study/'execution-v1',False,cells,indexed,rows)
    def test_stale_cached_audit_metrics_refused(self):
        cells,indexed,rows=self.audit_fixture()
        path=Path(rows[0]['metrics_path'])
        metric=p.read(path);metric['sum_nll']=456.5;path.write_text(json.dumps(metric))
        indexed[str(path)]=p.sha(path)
        with self.assertRaisesRegex(ValueError,'metrics SHA mismatch'):p.audit_metrics(self.study/'execution-v1',False,cells,indexed,rows)
    def test_audit_numeric_value_mismatch_refused(self):
        cells,indexed,rows=self.audit_fixture()
        cells[0]['ppl']=3.5
        with self.assertRaisesRegex(ValueError,'metric value mismatch'):p.audit_metrics(self.study/'execution-v1',False,cells,indexed,rows)
    def test_missing_unverified_or_wrong_delivery_cell_refused(self):
        cells,indexed,rows=self.audit_fixture()
        for field,bad in [('verified','no'),('metrics_path','/wrong'),('sum_nll','456.5')]:
            changed=[dict(r) for r in rows];changed[0][field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):p.audit_metrics(self.study/'execution-v1',False,cells,indexed,changed)
        with self.assertRaisesRegex(ValueError,'missing/unverified'):p.audit_metrics(self.study/'execution-v1',False,cells,indexed,rows[1:])
    def test_source_parent_symlink_refused(self):
        outside=self.root/'outside';outside.mkdir();(outside/'REPORT.md').write_text('TEST')
        (self.study/'diagnosis').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'escapes study'):p.safe_source(self.study/'diagnosis/REPORT.md',self.study)
    def test_internal_source_parent_symlink_refused(self):
        real=self.study/'real';real.mkdir();(real/'REPORT.md').write_text('TEST')
        (self.study/'diagnosis').symlink_to(real,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlink source parent'):p.safe_source(self.study/'diagnosis/REPORT.md',self.study)
    def test_history_relative_links_rebased(self):
        result=p.rebase_history_links('[spec](SPEC.md) [baseline](../stp-baselines/THRESHOLD_BASELINE.md) [web](https://example.com)')
        self.assertIn('[spec](../SPEC.md)',result)
        self.assertIn('[baseline](../../stp-baselines/THRESHOLD_BASELINE.md)',result)
        self.assertIn('[web](https://example.com)',result)
    def test_bound_underlying_source_change_after_copy_refused(self):
        repo=self.root/'repo';repo.mkdir()
        source=self.study/'underlying-metrics.json';source.write_text('TEST original')
        sources={str(source):p.sha(source)}
        original=p.hashes;calls=[]
        def changing_hashes(mapping):
            original(mapping);calls.append(True)
            if len(calls)==1:source.write_text('TEST changed')
        with patch.object(p,'git_guard'),patch.object(p,'git',return_value='') as git_mock,patch.object(p,'hashes',side_effect=changing_hashes):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                p.execute(self.study,repo,{'docs/test.md':b'TEST'},sources,self.root/'receipt')
        self.assertTrue((repo/'docs/test.md').exists())
        self.assertFalse(any('add' in call.args for call in git_mock.call_args_list))
    def test_conflicting_closure_source_refused(self):
        with self.assertRaisesRegex(ValueError,'conflicting bound hash'):p.merge_hashes({'/same':'old'},{'/same':'new'})
    def test_copied_diagnostic_links_absolute(self):
        source=self.study/'diagnosis-distributions/REPORT.md'
        text=p.external_links('[main](../diagnosis/REPORT.md) [data](histograms.jsonl)',source)
        self.assertIn(str(self.study/'diagnosis/REPORT.md'),text)
        self.assertIn(str(source.parent/'histograms.jsonl'),text)
    def test_default_cli_refuses_missing_fixture_without_writes(self):
        result=subprocess.run([sys.executable,str(Path(p.__file__)),'--study-root',str(self.study),'--repo',str(self.root/'absent')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertFalse((self.root/'absent').exists())
    def test_source_change_refuses_before_receipt_or_write(self):
        receipt=self.root/'receipt'
        src=self.root/'script.py';src.write_text('actual')
        with patch.object(p,'git_guard'):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                p.execute(self.study,self.root/'repo',{'docs/new.md':b'test'},{str(src):'0'*64},receipt)
        self.assertFalse(receipt.exists())
        self.assertFalse((self.root/'repo').exists())

if __name__=='__main__':unittest.main()
