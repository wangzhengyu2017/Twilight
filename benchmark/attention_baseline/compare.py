"""Require capture to preserve every scored logits byte and NLL."""
import argparse
import json
from pathlib import Path
from .common import load_json, write_json, canonical_hash


def compare_runs(reference, candidate):
    a, b = Path(reference), Path(candidate)
    for root in (a, b):
        assert load_json(root / 'status.json')['status'] == 'complete', f'incomplete run: {root}'
    am, bm = [load_json(p / 'metrics.json') for p in (a,b)]
    keys = ('protocol_id','profile','mode','method','backend','model_dtype','tokens_sha256','subset_sha256','scoring_ranges_sha256','num_input_tokens','num_targets','sum_nll','ppl','code_sha256','environment_sha256','asset_manifest_sha256','semantic_config_sha256')
    for key in keys:
        assert key in am and am[key] == bm.get(key), f'metrics mismatch: {key}'
    for name in ('asset_manifest.json','method_manifest.json','quantization_diagnostics.json','coverage.json'):
        assert load_json(a/name) == load_json(b/name), f'{name} mismatch'
    ac,bc = [load_json(p/'resolved_config.json') for p in (a,b)]
    # Recording options are not numerical configuration.
    from .common import semantic_config
    assert semantic_config(ac) == semantic_config(bc), 'semantic configuration mismatch'
    rows = []
    for filename in ('window_metrics.jsonl','target_metrics.jsonl'):
        if (a/filename).exists() or (b/filename).exists():
            assert (a/filename).exists() and (b/filename).exists(), 'missing scoring rows'
            ar,br = [[json.loads(s) for s in (p/filename).read_text().splitlines() if s] for p in (a,b)]
            assert ar and ar == br, f'exact per-call logits/NLL comparison failed: {filename}'
            assert all(any('logits' in k and 'hash' in k or k == 'logits_sha256' for k in r) for r in ar), 'missing original-dtype logits hash'
            rows.extend(ar)
    assert rows, 'no scoring rows'
    return {'status':'complete','passed':True,'reference':str(a.resolve()),'candidate':str(b.resolve()),'rows_compared':len(rows),'exact_logits_and_nll':True,'metrics_sha256':canonical_hash(am)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    try: result=compare_runs(args.reference,args.candidate)
    except Exception as e:
        write_json(args.output,{'status':'failed','passed':False,'reason':str(e)});raise
    write_json(args.output,result)

if __name__ == '__main__': main()
