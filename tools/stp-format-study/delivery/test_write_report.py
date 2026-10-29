"""Synthetic temporary evidence only; never reads actual execution/model artifacts."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import write_report as report


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


CONDITIONAL=[f'{prefix}{p}' for prefix in ('format_mxfp4_q4k4_topp_p','format_mxint8q_mxfp4k_q8k4_topp_p') for p in ('095','090','085')]
FIELDS=['variant_id','status','verified','sum_nll','num_targets','ppl','run']


def index(root):
    files={str(p):report.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact-index.json'}
    put(root/'artifact-index.json',dict(schema_version='format-artifact-index-v1',root=str(root),files=files,excluded=[]))


def rows_write(root,rows):
    with (root/'BASELINE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)


def manifest(root,matrix):
    # Reseal intentionally edited synthetic source identities so structural
    # negative tests reach the intended validator rather than an earlier hash.
    for path in root.rglob('pairing_manifest.json'):
        doc=report.read(path)
        doc['source_files']={name:report.sha(name) for name in doc['source_files']}
        put(path,doc)
    sources={str(p):report.sha(p) for p in root.rglob('*') if p.is_file() and ('runs' in p.parts or p.name=='source-TEST.json')}
    put(root/'BASELINE.manifest.json',dict(output_sha256=report.sha(root/'BASELINE.csv'),decision_sha256=report.sha(root/'BASELINE.decision.json'),matrix_sha256=report.sha(matrix),source_files=sources))


def measured(root,vid,ppl=math.e,verified=True):
    run=root/'runs'/vid;run.mkdir(parents=True)
    assets=dict(num_tokens=report.TARGETS+1,windows=dict(expected_targets=report.TARGETS),tokens_sha256='TEST-TOKENS')
    ranges=[dict(lo=1,target_end=report.TARGETS+1)]
    coverage=dict(scoring_ranges=ranges,num_targets=report.TARGETS,all_layers_covered=True)
    config={'TEST':True}
    code=dict(head='TEST-COMMIT-NOT-REAL',files={});code['code_sha256']=report.canonical(code)
    (run/'code.patch').write_text('TEST ONLY');code['patch_sha256']=report.sha(run/'code.patch')
    env={'TEST':True};env['environment_sha256']=report.canonical(env)
    metrics=dict(variant_id=vid,profile='full',mode='windows',protocol_id='TEST',subset_sha256='TEST-TOKENS',num_input_tokens=report.TARGETS+1,model_dtype='bfloat16',backend='torch_qdq',num_targets=report.TARGETS,sum_nll=math.log(ppl)*report.TARGETS,ppl=ppl,asset_manifest_sha256=report.canonical(assets),tokens_sha256=assets['tokens_sha256'],scoring_ranges_sha256=report.canonical(ranges),code_sha256=code['code_sha256'],environment_sha256=env['environment_sha256'],semantic_config_sha256=report.canonical(config))
    gate=put(run/'gate.json',dict(passed=True,status='complete',**{k:metrics[k] for k in ('code_sha256','environment_sha256','asset_manifest_sha256','tokens_sha256')}))
    metrics['gate_report_sha256']=report.sha(gate)
    put(run/'metrics.json',metrics);put(run/'status.json',{'status':'complete'})
    for name,value in [('asset_manifest',assets),('coverage',coverage),('code_manifest',code),('environment',env),('resolved_config',{'algorithm':config})]:
        put(run/(name+'.json'),value)
    (run/'window_metrics.jsonl').write_text(json.dumps(dict(**ranges[0],num_targets=report.TARGETS,sum_nll=metrics['sum_nll'],logits_sha256='TEST-LOGITS'))+'\n')
    if vid.startswith('format_'):
        pair=root/'smoke/pairing'/vid;pair.mkdir(parents=True)
        artifacts=[]
        for name in ('paired_sets.jsonl','set_comparison.csv'):
            path=pair/name;path.write_text('TEST DATA ONLY');artifacts.append(path)
        summary={role:{key:{'mean':0.9} for key in ('jaccard','precision','recall','retained_mass','shortfall')} for role in ('int','fp','tc','own_source_int_full')}
        put(pair/'pairing_manifest.json',dict(candidate_variant_id=vid,status='complete',passed=True,num_rows=768,counts={'same_input_estimator':192,'cross_run_trajectory':576},summary=summary,source_files={str(run/'metrics.json'):report.sha(run/'metrics.json')},artifacts={str(path):report.sha(path) for path in artifacts}))
    return {**{k:metrics[k] for k in ('variant_id','sum_nll','num_targets','ppl')},'status':'complete','verified':verified,'run':str(run)}


class ReportTests(unittest.TestCase):
    def fixture(self,base,conditional=False):
        root,study=base/'execution-TEST',base/'study-TEST';root.mkdir()
        put(root/'orchestration_status.json',{'status':'complete','conditional_activated':conditional})
        put(root/'source-TEST.json',{'synthetic':True})
        variants=[{'variant_id':f'format_{kind}_q{q}k4_topp_p{p}','status':'initial'} for kind in ('sm8','mxint8_tc4') for q in (4,8) for p in ('095','090','085')]
        variants += [{'variant_id':vid,'status':'conditional_not_activated'} for vid in CONDITIONAL]
        matrix=put(root/'freeze/handoff/matrix.json',{'variants':variants})
        assets=put(root/'freeze/asset_manifest.json',{'synthetic':True})
        patch=root/'freeze/code.patch';patch.write_text('TEST ONLY')
        code={'head':'TEST-COMMIT-NOT-REAL','files':{}}
        code['code_sha256']=hashlib.sha256(json.dumps(code,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        code['patch_sha256']=report.sha(patch);put(root/'freeze/code_manifest.json',code)
        put(root/'freeze/freeze.json',dict(code_sha256=code['code_sha256'],head=code['head'],source_root=str(base/'nonexistent-TEST'),matrix_sha256=report.sha(matrix),asset_manifest_sha256=report.sha(assets)))
        for filename in ('diagnosis/REPORT.md','diagnosis-distributions/REPORT.md','supplement-proposal/SPEC.md'):
            path=study/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('TEST DATA ONLY')
        rows=[measured(root,v['variant_id']) for v in variants if v['status']=='initial']
        rows += [measured(root,f'msb_q{q}k4_topp_p{p}',math.e if conditional else math.e*1.005) for q in (4,8) for p in ('095','090','085')]
        rows += [measured(root,'int_topp_p'+p) for p in ('095','090','085')]
        rows += [measured(root,vid,value,False) for vid,value in report.DENSE.items()]
        decision=dict(status='complete',primary='q4k4',widths={})
        lookup={r['variant_id']:r for r in rows}
        for q in (4,8):
            means={role:sum(lookup[prefix+p]['sum_nll']/lookup[prefix+p]['num_targets'] for p in ('095','090','085'))/3 for role,prefix in [('mx',f'format_mxint8_tc4_q{q}k4_topp_p'),('tc',f'msb_q{q}k4_topp_p'),('int','int_topp_p')]}
            dt,di=means['mx']-means['tc'],means['mx']-means['int']
            decision['widths'][f'q{q}k4']=dict(mean_ce=means,delta_ce_vs_tc=dt,delta_ce_vs_int=di,ppl_ratio_vs_tc=math.exp(dt),ppl_ratio_vs_int=math.exp(di),no_tc_improvement=means['mx']>=means['tc'],above_int_margin=di>math.log(1.01))
        decision['activate_mxfp4']=conditional
        put(root/'BASELINE.decision.json',decision)
        rows += [dict(variant_id=vid,status='conditional_not_activated',verified=False) for vid in CONDITIONAL]
        rows_write(root,rows);manifest(root,matrix)
        if conditional:
            # Snapshot the actual activation authority before conditional outputs.
            for suffix in ('csv','decision.json','manifest.json'):
                shutil.copyfile(root/('BASELINE.'+suffix),root/('INITIAL_BASELINE.'+suffix))
            initial=root/'INITIAL_BASELINE.manifest.json';dec=root/'INITIAL_BASELINE.decision.json'
            transformed=[dict(v,required=v['status']=='conditional_not_activated',status='conditional_activated_pending_smoke' if v['status']=='conditional_not_activated' else v['status']) for v in variants]
            matrix=put(root/'conditional-handoff/matrix.json',dict(variants=transformed,execution_phase='conditional',conditional_activation=dict(trigger_verified=True,decision_path=str(dec),decision_sha256=report.sha(dec),decision_manifest_path=str(initial),decision_manifest_sha256=report.sha(initial))))
            rows=[row for row in rows if row['variant_id'] not in CONDITIONAL]+[measured(root,vid) for vid in CONDITIONAL]
            rows_write(root,rows);manifest(root,matrix)
        index(root)
        return root,study

    def test_synthetic_report_precision_and_nonactivation(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            base=Path(temp);root,study=self.fixture(base)
            output=report.build(root,study,base/'delivery-TEST',test_data=True)
            text=(output/'REPORT.md').read_text()
            self.assertIn('TEST DATA ONLY',text);self.assertIn('6.484759140305276',text)
            self.assertIn('no parser-matched full gate',text);self.assertIn('source blocked',text)
            self.assertNotIn('| MXFP4 Q4K4 |',text)
            with (output/'FULL_TABLE.csv').open() as stream:full=list(csv.DictReader(stream))
            with (root/'BASELINE.csv').open() as stream:original={r['variant_id']:r for r in csv.DictReader(stream)}
            for row in full:
                if row['status']=='complete':
                    for key in ('sum_nll','ppl'):self.assertEqual(row[key],original[row['variant_id']][key])

    def test_historical_pairing_stays_bound_but_is_not_format_output(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            base=Path(temp);root,study=self.fixture(base)
            historical=base/'historical-TEST/pairing/pairing_manifest.json'
            leaf=historical.parent/'paired_sets.jsonl'
            leaf.parent.mkdir(parents=True);leaf.write_text('HISTORICAL TEST DATA ONLY')
            put(historical,dict(schema_version='slice-pairing-v2',status='complete',passed=True,
                counts={'cross_run_trajectory':768,'same_input_estimator':384},num_rows=1152,
                artifacts={str(leaf):report.sha(leaf)}))
            put(root/'source-TEST.json',{'source_files':{str(historical):report.sha(historical)}})
            manifest(root,root/'freeze/handoff/matrix.json');index(root)
            output=report.build(root,study,base/'delivery-TEST',test_data=True)
            delivery=report.read(output/'DELIVERY.manifest.json')
            for path in (historical,leaf):
                self.assertEqual(delivery['source_files'][str(path)],report.sha(path))
            with (output/'PAIRED_SETS.csv').open() as stream:pairing=list(csv.DictReader(stream))
            with (root/'BASELINE.csv').open() as stream:
                accepted_formats={row['variant_id'] for row in csv.DictReader(stream)
                                  if row['variant_id'].startswith('format_') and report.accepted(row)}
            self.assertEqual(len(pairing),4*len(accepted_formats))
            self.assertEqual({row['variant_id'] for row in pairing},accepted_formats)
            self.assertNotIn(str(historical),{row['manifest'] for row in pairing})
            # Historical records remain integrity evidence even though they are
            # outside the current-format pairing table and have another schema.
            leaf.write_text('CORRUPTED HISTORICAL TEST DATA')
            with self.assertRaisesRegex(ValueError,'Changed evidence'):
                report.build(root,study,base/'rejected-TEST',test_data=True)
            self.assertFalse((base/'rejected-TEST').exists())

    def test_build_rejects_invalid_current_format_pairing(self):
        for case in ('counts','rows','references','metric','completion','artifacts'):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
                base=Path(temp);root,study=self.fixture(base)
                path=root/'smoke/pairing/format_sm8_q4k4_topp_p095/pairing_manifest.json'
                document=report.read(path)
                if case=='counts':document['counts']['same_input_estimator']=384
                elif case=='rows':document['num_rows']=1152
                elif case=='references':del document['summary']['own_source_int_full']
                elif case=='metric':document['summary']['int']['recall']['mean']=1.1
                elif case=='completion':document['passed']=False
                else:document['artifacts']={}
                put(path,document);manifest(root,root/'freeze/handoff/matrix.json');index(root)
                with self.assertRaisesRegex(ValueError,'Pairing|pairing'):
                    report.build(root,study,base/'rejected-TEST',test_data=True)
                self.assertFalse((base/'rejected-TEST').exists())

    def test_conditional_positive(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            base=Path(temp);root,study=self.fixture(base,True)
            _,decision,_,_=report.load_bundle(root,report.Evidence())
            self.assertIs(decision['activate_mxfp4'],True)
            output=report.build(root,study,base/'delivery-TEST',test_data=True)
            self.assertIn('| MXFP4 Q4K4 |',(output/'REPORT.md').read_text())

    def test_conditional_wrong_final_decision(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp),True)
            put(root/'BASELINE.decision.json',{'status':'complete','activate_mxfp4':False,'widths':{}})
            manifest(root,root/'conditional-handoff/matrix.json');index(root)
            with self.assertRaisesRegex(ValueError,'differs from immutable initial'):
                report.load_bundle(root,report.Evidence())

    def test_activation_hash_corruption(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp),True);matrix=root/'conditional-handoff/matrix.json'
            doc=report.read(matrix);doc['conditional_activation']['decision_manifest_sha256']='0'*64;put(matrix,doc)
            manifest(root,matrix);index(root)
            with self.assertRaisesRegex(ValueError,'Changed evidence'):
                report.load_bundle(root,report.Evidence())

    def test_initial_only_nested_leaf_corruption(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp),True)
            leaf=put(root/'initial-only-leaf.json',{'TEST':1})
            nested=put(root/'initial-only-nested.json',{'source_files':{str(leaf):report.sha(leaf)}})
            initial=root/'INITIAL_BASELINE.manifest.json';doc=report.read(initial);doc['source_files'][str(nested)]=report.sha(nested);put(initial,doc)
            matrix=root/'conditional-handoff/matrix.json';doc=report.read(matrix);doc['conditional_activation']['decision_manifest_sha256']=report.sha(initial);put(matrix,doc)
            manifest(root,matrix)
            put(leaf,{'CORRUPTED':True});index(root)
            with self.assertRaisesRegex(ValueError,'Changed evidence'):
                report.load_bundle(root,report.Evidence())

    def test_status_conflict_and_missing_initial(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp),True)
            put(root/'orchestration_status.json',{'status':'complete','conditional_activated':False});index(root)
            with self.assertRaisesRegex(ValueError,'conflicts with conditional'):
                report.load_bundle(root,report.Evidence())
            put(root/'orchestration_status.json',{'status':'complete','conditional_activated':True});(root/'INITIAL_BASELINE.manifest.json').unlink();index(root)
            with self.assertRaises(FileNotFoundError):report.load_bundle(root,report.Evidence())

    def test_index_and_missing_run_evidence(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp));(root/'source-TEST.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed evidence'):report.load_bundle(root,report.Evidence())
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp));(root/'artifact-index.json').unlink()
            with self.assertRaises(FileNotFoundError):report.load_bundle(root,report.Evidence())
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp))
            with (root/'BASELINE.csv').open() as stream:rows=list(csv.DictReader(stream))
            rows[0]['run']='';rows_write(root,rows);manifest(root,root/'freeze/handoff/matrix.json');index(root)
            with self.assertRaisesRegex(ValueError,'lacks absolute run evidence'):report.load_bundle(root,report.Evidence())

    def test_dense_absent_and_failed_never_emit_constants(self):
        for failed in (False,True):
            with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
                base=Path(temp);root,study=self.fixture(base)
                with (root/'BASELINE.csv').open() as stream:rows=list(csv.DictReader(stream))
                if failed:
                    for row in rows:
                        if row['variant_id'] in report.DENSE:row.update(status='failed',run='')
                else:
                    rows=[r for r in rows if r['variant_id'] not in report.DENSE]
                rows_write(root,rows);manifest(root,root/'freeze/handoff/matrix.json');index(root)
                output=report.build(root,study,base/'delivery-TEST',test_data=True)
                text=(output/'REPORT.md').read_text()
                for value in report.DENSE.values():self.assertNotIn(repr(value),text)
                self.assertIn('historical evidence absent',text)

    def test_coverage_totals_and_shared_identity_rejected(self):
        for case in ('window_nll','gate_identity','shared_tokens'):
            with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
                root,_=self.fixture(Path(temp));run=root/'runs/format_sm8_q4k4_topp_p095'
                if case=='window_nll':
                    path=run/'window_metrics.jsonl';doc=json.loads(path.read_text());doc['sum_nll']+=1;path.write_text(json.dumps(doc)+'\n')
                elif case=='gate_identity':
                    path=run/'gate.json';doc=report.read(path);doc['tokens_sha256']='WRONG';put(path,doc)
                    path=run/'metrics.json';doc=report.read(path);doc['gate_report_sha256']=report.sha(run/'gate.json');put(path,doc)
                else:
                    # Preserve individually valid full evidence while changing the
                    # common subset identity to test comparison across methods.
                    path=run/'metrics.json';doc=report.read(path);doc['subset_sha256']='DIFFERENT-SUBSET';put(path,doc)
                manifest(root,root/'freeze/handoff/matrix.json');index(root)
                with self.assertRaises(ValueError):report.load_bundle(root,report.Evidence())

    def test_conditional_transformation_and_unactivated_numeric(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp),True);matrix=root/'conditional-handoff/matrix.json'
            doc=report.read(matrix);doc['variants'][-1]['required']=False;put(matrix,doc);manifest(root,matrix);index(root)
            with self.assertRaisesRegex(ValueError,'activation transformation'):report.load_bundle(root,report.Evidence())
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp))
            with (root/'BASELINE.csv').open() as stream:rows=list(csv.DictReader(stream))
            rows=[r for r in rows if r['variant_id']!=CONDITIONAL[0]]+[measured(root,CONDITIONAL[0])]
            rows_write(root,rows);manifest(root,root/'freeze/handoff/matrix.json');index(root)
            with self.assertRaisesRegex(ValueError,'Unactivated conditional'):report.load_bundle(root,report.Evidence())

    def test_decision_formula_and_schema_rejected(self):
        for case in ('empty_widths','changed_mean','wrong_trigger'):
            with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
                root,_=self.fixture(Path(temp));path=root/'BASELINE.decision.json';doc=report.read(path)
                if case=='empty_widths':doc['widths']={}
                elif case=='changed_mean':doc['widths']['q4k4']['mean_ce']['mx']+=0.001
                else:doc['widths']['q4k4']['no_tc_improvement']=True
                put(path,doc);manifest(root,root/'freeze/handoff/matrix.json');index(root)
                with self.assertRaisesRegex(ValueError,'frozen binary64 CE rule'):report.load_bundle(root,report.Evidence())

    def test_initial_csv_must_match_validated_actual_runs(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp),True);path=root/'INITIAL_BASELINE.csv'
            with path.open() as stream:rows=list(csv.DictReader(stream))
            for row in rows:
                if row['variant_id']=='format_mxint8_tc4_q4k4_topp_p095':row['sum_nll']=str(float(row['sum_nll'])+1)
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
            initial=root/'INITIAL_BASELINE.manifest.json';doc=report.read(initial);doc['output_sha256']=report.sha(path);put(initial,doc)
            matrix=root/'conditional-handoff/matrix.json';doc=report.read(matrix);doc['conditional_activation']['decision_manifest_sha256']=report.sha(initial);put(matrix,doc)
            manifest(root,matrix);index(root)
            with self.assertRaisesRegex(ValueError,'differs from validated final evidence'):report.load_bundle(root,report.Evidence())

    def test_missing_pairing_and_changed_index_rejected(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp));(root/'smoke/pairing/format_sm8_q4k4_topp_p095/pairing_manifest.json').unlink()
            manifest(root,root/'freeze/handoff/matrix.json');index(root)
            with self.assertRaisesRegex(ValueError,'unique bound pairing'):report.load_bundle(root,report.Evidence())
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp));path=root/'artifact-index.json';doc=report.read(path);doc['files'][str(root/'orchestration_status.json')]='0'*64;put(path,doc)
            with self.assertRaisesRegex(ValueError,'Changed evidence'):report.load_bundle(root,report.Evidence())

    def test_negative_nll_and_noninteger_target_rejected(self):
        for case in ('negative_nll','noninteger_target'):
            with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
                root,_=self.fixture(Path(temp));run=root/'runs/format_sm8_q4k4_topp_p095'
                path=run/'metrics.json';doc=report.read(path)
                if case=='negative_nll':doc.update(sum_nll=-report.TARGETS,ppl=math.exp(-1))
                else:doc['num_targets']=float(report.TARGETS)
                put(path,doc)
                with (root/'BASELINE.csv').open() as stream:rows=list(csv.DictReader(stream))
                for row in rows:
                    if row['variant_id']=='format_sm8_q4k4_topp_p095':
                        for key in ('sum_nll','ppl','num_targets'):row[key]=doc[key]
                rows_write(root,rows);manifest(root,root/'freeze/handoff/matrix.json');index(root)
                with self.assertRaises(ValueError):report.load_bundle(root,report.Evidence())

    def test_output_corruption_and_incomplete(self):
        with tempfile.TemporaryDirectory(prefix='stp-TEST-') as temp:
            root,_=self.fixture(Path(temp))
            with (root/'BASELINE.csv').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'Changed evidence'):report.load_bundle(root,report.Evidence())
            put(root/'orchestration_status.json',{'status':'running'})
            with self.assertRaisesRegex(ValueError,'not terminal'):report.load_bundle(root,report.Evidence())

class ArchivedSourceTests(unittest.TestCase):
    def fixture(self, base):
        root = base/'stp-format-study/execution-v1'
        current = root/'freeze'
        historical = base/'stp-slice-thresholds/preparation/execution-source'
        gates = []
        for archive, directory, content in ((current, 'source', 'CURRENT'), (historical, 'files', 'HISTORICAL')):
            source = archive/directory/'STP_Test/MSBTop_p.py'
            source.parent.mkdir(parents=True)
            source.write_text(content)
            code = {'head': 'TEST', 'files': {'STP_Test/MSBTop_p.py': report.sha(source)}}
            code['code_sha256'] = report.canonical(code)
            put(archive/'code_manifest.json', code)
            gate = put(root/(content.lower()+'-gate.json'), dict(
                status='complete', passed=True, code_sha256=code['code_sha256'],
                source_files=code['files'], artifacts={}, evidence_files={}))
            gates.append((gate, source))
        evidence = report.Evidence()
        evidence.configure_sources(root)
        for gate, _ in gates:
            evidence.bind(gate, report.sha(gate))
        return evidence, gates

    def test_real_relative_gate_schema_closure_independent_of_cwd(self):
        with tempfile.TemporaryDirectory(prefix='stp-source-TEST-') as temp:
            base = Path(temp)
            evidence, gates = self.fixture(base)
            unrelated = base/'unrelated';unrelated.mkdir()
            # A plausible but wrong CWD file must never be consulted.
            decoy = unrelated/'STP_Test/MSBTop_p.py';decoy.parent.mkdir();decoy.write_text('DECOY')
            previous = Path.cwd()
            try:
                os.chdir(unrelated)
                evidence.closure()
            finally:
                os.chdir(previous)
            for _, source in gates:
                self.assertEqual(evidence.files[str(source)], report.sha(source))
            self.assertNotIn(str(decoy), evidence.files)

    def test_source_corruption_and_gate_wrong_digest_refused(self):
        for corrupt_archive in (False, True):
            with tempfile.TemporaryDirectory(prefix='stp-source-TEST-') as temp:
                evidence, gates = self.fixture(Path(temp))
                gate, source = gates[0]
                if corrupt_archive:
                    source.write_text('WRONG')
                    with self.assertRaisesRegex(ValueError, 'Changed evidence'):
                        evidence.closure()
                else:
                    doc = report.read(gate);doc['source_files']['STP_Test/MSBTop_p.py'] = '0'*64
                    with self.assertRaisesRegex(ValueError, 'differs from archived manifest'):
                        evidence.maps(doc, owner=gate)

    def test_missing_ambiguous_identity_and_invalid_manifest_refused(self):
        for case in ('missing', 'ambiguous', 'invalid', 'no_identity'):
            with tempfile.TemporaryDirectory(prefix='stp-source-TEST-') as temp:
                evidence, gates = self.fixture(Path(temp));gate, _ = gates[0]
                doc = report.read(gate)
                if case == 'missing':doc['code_sha256'] = '0'*64
                elif case == 'no_identity':doc.pop('code_sha256')
                elif case == 'ambiguous':evidence.source_archives += (evidence.source_archives[0],)
                else:
                    path, _ = evidence.source_archives[0];manifest = report.read(path);manifest['head'] = 'WRONG';put(path, manifest)
                with self.assertRaises(ValueError):evidence.maps(doc, owner=gate)

    def test_traversal_symlink_and_relative_artifacts_refused(self):
        with tempfile.TemporaryDirectory(prefix='stp-source-TEST-') as temp:
            base = Path(temp);root = base/'snapshot';root.mkdir()
            outside = base/'outside.py';outside.write_text('TEST')
            (root/'escape.py').symlink_to(outside)
            for name in ('../outside.py', 'a/../../outside.py', './outside.py', '/outside.py', 'escape.py'):
                with self.assertRaises(ValueError):report.Evidence.source_path(root, name)
            for key in ('artifacts', 'evidence_files'):
                with self.assertRaisesRegex(ValueError, 'must be absolute'):
                    report.Evidence().maps({key: {'relative.json': '0'*64}})

if __name__=='__main__':unittest.main()
