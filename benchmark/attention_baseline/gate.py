"""Bind successful smoke evidence to immutable code, environment and assets."""
import argparse
from pathlib import Path
from .common import load_json,write_json,canonical_hash,sha256_file,load_config,semantic_config,code_provenance,environment,validate_assets,configure_runtime,load_tokens,tokens_hash
from .compare import compare_runs
from .replay import replay_run
from .summarize import public_config
from .preflight import HF_EAGER_THRESHOLDS

from twilight.pyimpl.quant_attention.intflash import SOURCE_DELTA, INPUT_CAST_POLICY, QUANTIZATION_COUNT_SEMANTICS

METHODS=('fp_common','intflash_qk8')

def _passed(path):
    value=load_json(path)
    assert value.get('status')=='complete' and value.get('passed',True) is True, f'unsuccessful evidence: {path}'
    return value


def validate_quantization_evidence(directory, method, layer_calls, recorded):
    diagnostics=load_json(directory/'quantization_diagnostics.json')
    assert recorded['quantization_diagnostics']==diagnostics, 'diagnostics report mismatch'
    assert diagnostics['input_cast_policy']==INPUT_CAST_POLICY, 'diagnostic cast policy changed'
    assert diagnostics['count_semantics']==QUANTIZATION_COUNT_SEMANTICS, 'diagnostic count semantics changed'
    assert len(diagnostics['events'])<=diagnostics['event_limit']==32, 'invalid diagnostic event budget'
    by_layer=diagnostics['by_layer_operand']
    if method=='fp_common':
        assert by_layer=={} and diagnostics['events']==[], 'FP run unexpectedly quantized'
    else:
        assert set(by_layer)==set(layer_calls), 'quantization diagnostic layer coverage mismatch'
        for layer,operands in by_layer.items():
            assert set(operands)=={'q','k'}, 'quantization diagnostic operand coverage mismatch'
            for counts in operands.values():
                assert counts['calls']==sum(layer_calls[layer][method].values()), 'quantizer call count mismatch'
                assert 0<=counts['input_cast_nonzero_to_zero_count']<=counts['input_nonzero_elements']<=counts['input_elements'], 'invalid cast counts'
                assert 0<=counts['zero_group_count']<=counts['groups'], 'invalid zero-group counts'
    return diagnostics


def validate_gate_report(report,model_path,assets,config,query_chunk=64):
    report=load_json(report) if not isinstance(report,dict) else report
    assert report.get('status')=='complete' and report.get('passed') is True, 'smoke gate did not pass'
    assert query_chunk==report['query_chunk']==64, 'query chunk changed; rerun smoke'
    assert code_provenance()['code_sha256']==report['code_sha256'], 'code changed; rerun smoke'
    assert environment()['environment_sha256']==report['environment_sha256'], 'environment changed; rerun smoke'
    ap=Path(assets)
    manifest=validate_assets(model_path,ap.parent/'tokens.pt',ap)
    assert canonical_hash(manifest)==report['asset_manifest_sha256'], 'assets changed; rerun smoke'
    if isinstance(config,(str,Path)):config=load_config(config)
    method=config['attention_quantization']['method']
    assert canonical_hash(semantic_config(config))==report['semantic_config_sha256'][method], 'configuration changed; rerun smoke'
    for name,digest in report['evidence_files'].items():
        assert sha256_file(name)==digest, f'smoke evidence changed: {name}'
    return report


def build_gate(smoke_root,model_path,assets,config_dir):
    root=Path(smoke_root).resolve();configure_runtime()
    code=code_provenance();env=environment();ap=Path(assets)
    manifest=validate_assets(model_path,ap.parent/'tokens.pt',ap)
    prefix=load_tokens(ap.parent/'tokens.pt')[:130]
    assert prefix.numel()==130, 'preflight requires 130 frozen tokens'
    expected_input={'source_policy':'frozen_wikitext_first_130','num_tokens':130,'tokens_sha256':manifest['tokens_sha256'],'subset_sha256':tokens_hash(prefix),'asset_manifest_sha256':canonical_hash(manifest)}
    configs={m:canonical_hash(semantic_config(load_config(Path(config_dir)/(m+'.json')))) for m in METHODS}
    def auxiliary(directory):
        _passed(directory/'status.json')
        recorded=load_json(directory/'code_manifest.json')
        assert recorded['code_sha256']==code['code_sha256'], f'{directory}: stale code'
        assert recorded['files']==code['files'] and recorded['head']==code['head'], f'{directory}: code sources mismatch'
        assert recorded['patch_sha256']==sha256_file(directory/'code.patch'), f'{directory}: patch hash mismatch'
        assert load_json(directory/'environment.json')['environment_sha256']==env['environment_sha256'], f'{directory}: stale environment'
    auxiliary(root/'validation')
    _passed(root/'validation'/'report.json')
    evidence={}
    def collect(directory):
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.patch'):
                evidence[str(p.resolve())]=sha256_file(p)
    collect(root/'validation')
    metrics={}
    for method in METHODS:
        pre=root/(method+'-preflight');auxiliary(pre);collect(pre)
        assert load_json(pre/'input_manifest.json')==expected_input, 'preflight frozen input identity changed'
        assert canonical_hash(load_json(pre/'asset_manifest.json'))==canonical_hash(manifest), 'preflight asset manifest mismatch'
        assert load_json(pre/'preflight.json')['input_manifest']==expected_input, 'preflight report input mismatch'
        assert canonical_hash(semantic_config(load_json(pre/'resolved_config.json')))==configs[method], 'preflight config changed'
        check=_passed(pre/'preflight.json');model=load_json(pre/'model_manifest.json')
        if method=='fp_common':
            assert check['hf_eager_comparison']['thresholds']==HF_EAGER_THRESHOLDS, 'preflight HF eager thresholds changed; rerun smoke'
        validate_quantization_evidence(pre,method,check['layer_calls'],check)
        assert check['method']==method and check['config_sha256']==configs[method], 'preflight identity mismatch'
        assert model['model_file_hashes']==manifest['model_file_hashes'], 'preflight model files mismatch'
        assert check['model_files_sha256']==model['model_files_sha256']==canonical_hash(manifest['model_file_hashes']), 'preflight model hash mismatch'
        assert check['all_logits_finite'] is True and check['patch_count']==32, 'preflight coverage incomplete'
        assert check['cache_storage']=='original_float' and check['quantization_at']=='attention_read', 'preflight cache contract mismatch'
        for label in ('windows','decode-off','decode-on'):
            run=root/(method+'-'+label);_passed(run/'status.json')
            m=load_json(run/'metrics.json');metrics[method,label]=m
            assert m['method']==method and m['profile']=='smoke', 'wrong smoke method/profile'
            assert m['mode']==('windows' if label=='windows' else 'cached_decode'), 'wrong smoke mode'
            for key,value in [('context_length',2048),('stride',1024),('query_chunk',64),('model_dtype','bfloat16'),('prefill_tokens',128),('decode_targets',16)]:
                assert m[key]==value, f'{run}: unsupported {key}'
            assert canonical_hash(semantic_config(load_json(run/'resolved_config.json')['algorithm']))==configs[method], 'run config mismatch'
            assert m['num_input_tokens']==(4096 if label=='windows' else 144), 'wrong smoke input count'
            assert m['num_targets']==(4095 if label=='windows' else 16), 'wrong smoke target count'
            for key,value in [('code_sha256',code['code_sha256']),('environment_sha256',env['environment_sha256']),('asset_manifest_sha256',canonical_hash(manifest)),('semantic_config_sha256',configs[method])]:
                assert m[key]==value, f'{run}: stale {key}'
            manifest_method=load_json(run/'method_manifest.json')
            expected_delta=list(SOURCE_DELTA) if method=='intflash_qk8' else []
            expected_policy=INPUT_CAST_POLICY if method=='intflash_qk8' else None
            assert manifest_method['source_delta']==m['source_delta']==expected_delta, 'run source delta mismatch'
            assert manifest_method['input_cast_policy']==m['input_cast_policy']==expected_policy, 'run cast policy mismatch'
            coverage=load_json(run/'coverage.json')
            diagnostics=validate_quantization_evidence(run,method,coverage['layers'],m)
            assert coverage['quantization_diagnostics']==diagnostics and coverage['input_cast_policy']==expected_policy, 'coverage diagnostic mismatch'
            auxiliary(run)
            assert canonical_hash(load_json(run/'asset_manifest.json'))==canonical_hash(manifest), 'asset manifest mismatch'
            collect(run)
        on=root/(method+'-decode-on');_passed(on/'compare.json');_passed(on/'replay.json')
        compare_runs(root/(method+'-decode-off'),on);replay_run(on)
    assert public_config(load_config(Path(config_dir)/'fp_common.json'))==public_config(load_config(Path(config_dir)/'intflash_qk8.json')), 'common config mismatch'
    for label in ('windows','decode-off','decode-on'):
        a,b=(metrics[m,label] for m in METHODS)
        for key in ('tokens_sha256','subset_sha256','scoring_ranges_sha256','num_input_tokens','num_targets','model_dtype','backend'):
            assert a[key]==b[key], f'method mismatch: {label}/{key}'
    return {'status':'complete','passed':True,'smoke_root':str(root),'code_sha256':code['code_sha256'],'environment_sha256':env['environment_sha256'],'asset_manifest_sha256':canonical_hash(manifest),'semantic_config_sha256':configs,'query_chunk':64,'context_length':2048,'stride':1024,'evidence_files':evidence}


def main():
    p=argparse.ArgumentParser()
    for arg in ('smoke-root','model-path','assets','config-dir','output'):p.add_argument('--'+arg,required=True)
    a=p.parse_args()
    try:r=build_gate(a.smoke_root,a.model_path,a.assets,a.config_dir)
    except Exception as e:
        write_json(a.output,{'status':'failed','passed':False,'reason':str(e)});raise
    write_json(a.output,r)

if __name__=='__main__':main()
