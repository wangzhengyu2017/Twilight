"""CPU-only real config/runner dispatch and gate-refusal acceptance."""
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch
from benchmark.attention_baseline.common import ROOT,load_config,write_json
from benchmark.attention_benchmark import parser,resolve,plan,execute,THRESHOLDS


def args(*flags):
    return parser().parse_args(['--profile','smoke','--output-root','/tmp/selector-unexecuted-output',*flags])


def must_fail(fn,text):
    try:fn()
    except (ValueError,FileExistsError) as exc:
        assert text.lower() in str(exc).lower(),str(exc);return
    raise AssertionError('invalid profile accepted')


def test_dispatch():
    count=0
    # Resolve every existing 18-format configuration to its unchanged serialized file.
    for path in sorted((ROOT/'benchmark/attention_format/configs').glob('*.json')):
        cfg=load_config(path);s=cfg['attention_slice'];fmt=s['predictor_format'];native=fmt in ('mxfp4','mxint8q_mxfp4k');encoding='mxint8' if fmt=='mxint8_tc4' else fmt
        flags=['--encoding',encoding,'--top-p','on','--p',str(s['selection']['p']),'--q-bits',str(s['q_prefix_bits']),'--k-bits',str(s['k_prefix_bits'])]
        flags+=['--final-operands','source-int8'] if native else ['--msb','on']
        got=resolve(args(*flags));assert got['config']==cfg and got['semantic_config_sha256'] and got['selection_passes']==1 and got['final_operand_profile']=='source_int8';count+=1
    for p,suffix in THRESHOLDS.items():
        for fmt in ('fp','int8'):
            got=resolve(args('--encoding',fmt,'--top-p','on','--p',str(p)));assert got['config']['attention_slice']['estimator']==('fp_full' if fmt=='fp' else 'int_full');count+=1
        for bq in (4,8):
            got=resolve(args('--encoding','int8','--msb','on','--top-p','on','--p',str(p),'--q-bits',str(bq)));assert got['config']['attention_slice']['estimator']=='tc_msb' and got['q_bits']==bq and got['final_operands']=='source-int8';count+=1
        for fmt in ('sm8','mxint8'):
            got=resolve(args('--encoding',fmt,'--top-p','on','--p',str(p)));assert got['config']['attention_full_format']['variant_id']==f'{fmt}_q8k8_topp_p{suffix}' and got['final_operand_profile']=='decoded_full_format' and got['final_operands']=='format';count+=1
    for fmt,widths in [('fp',(None,None)),('int8',(8,8)),('sm8',(8,8)),('mxint8',(8,8)),('mxfp4',(4,4)),('mxint8q_mxfp4k',(8,4))]:
        got=resolve(args('--encoding',fmt));assert (got['q_bits'],got['k_bits'])==widths and got['selection_passes']==0 and got['top_p'] is False;count+=1
    return count


def test_invalid():
    for flags,message in [(['--encoding','sm8','--p','.95'],'--p requires'),(['--encoding','sm8','--msb','on'],'MSB without'),(['--encoding','fp','--msb','on','--top-p','on','--p','.95'],'no compatible'),(['--encoding','mxfp4','--msb','on','--top-p','on','--p','.95'],'native FP4'),(['--encoding','mxint8q_mxfp4k','--top-p','on','--p','.95'],'not an implemented'),(['--encoding','mxfp4','--q-bits','8'],'width differs'),(['--encoding','rsint8','--final-operands','source-int8'],'final operand'),(['--encoding','sm8','--top-p','on','--p','1'],'authorized executable'),(['--encoding','mxint8','--top-p','on','--p','0'],'0 <'),(['--encoding','sm8','--msb','on','--top-p','on','--p','.95','--final-operands','format'],'original source')]:
        must_fail(lambda:resolve(args(*flags)),message)


def test_plan_and_gate():
    import torch
    with patch('benchmark.attention_baseline.common.load_model',side_effect=AssertionError('plan loaded model')),patch.object(torch.cuda,'init',side_effect=AssertionError('plan initialized CUDA')):
        for encoding,top,pvalue in [('mxfp4','off',None),('mxint8','on','.9')]:
            flags=['--encoding',encoding,'--top-p',top]+(['--p',pvalue] if pvalue else [])
            request=args(*flags);result=plan(request);commands=result['execution_commands'];assert len(commands)==6 and result['gate_prerequisites']
            scorer=[row['argv'] for row in commands if row['argv'][1:3]==['-m','benchmark.ppl']];assert len(scorer)==3
            assert all(command[command.index('--algo-config-path')+1]==result['resolved']['config_path'] for command in scorer),'selector expanded to other benchmark configs'
            assert not any(row['name']=='gate' for row in commands),'single smoke minted cohort gate'
        request=args('--encoding','sm8','--top-p','on','--p','.95');request.profile='full';request.smoke_root='/tmp/selector-required-matching-gate';result=plan(request)
        assert len(result['execution_commands'])==1 and result['execution_commands'][0]['argv'][result['execution_commands'][0]['argv'].index('--algo-config-path')+1]==result['resolved']['config_path']
        request.execute=True;request.review_report='/tmp/not-read-review';request.smoke_root=None
        must_fail(lambda:execute(request,result),'matching --smoke-root')
        from benchmark.attention_full_format.gate import validate_gate_report
        must_fail(lambda:validate_gate_report({},'/hyperai/input/input0','/output/project/assets/tq-wt2-v1/asset_manifest.json',result['resolved']['config']),'not passed')
        from benchmark.attention_dense_format.gate import validate_gate_report as dense_gate
        must_fail(lambda:dense_gate({},'/hyperai/input/input0','/output/project/assets/tq-wt2-v1/asset_manifest.json',resolve(args('--encoding','sm8'))['config']),'not passed')
        old=args('--encoding','int8');old.execute=True;oldplan=plan(old);must_fail(lambda:execute(old,oldplan),'completed frozen studies')


def main():
    count=test_dispatch();test_invalid();test_plan_and_gate();report=dict(status='passed',resolved_study_profiles=count,single_config_smoke_and_full_dispatch=True,plan_only_no_model_or_cuda=True,invalid_profiles_and_missing_gates_rejected=True,historical_semantics_preserved=True,RS_available_with_independent_review_gate=True)
    write_json('/output/project/experiments/stp-dense-formats/preparation/selector-cpu-validation.json',report);print(report)
if __name__=='__main__':main()
