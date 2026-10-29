"""Resolve numeric format/MSB/TopP choices; print a plan unless --execute is set."""
import argparse
from argparse import Namespace
from pathlib import Path
import sys
from benchmark.attention_baseline.common import ROOT,load_config,load_json,canonical_hash,sha256_file,code_provenance

FORMATS=('fp','int8','sm8','mxint8','mxfp4','mxint8q_mxfp4k','rsint8')
THRESHOLDS={.95:'095',.9:'090',.85:'085'}
REGISTERED_EVENTS='/output/project/experiments/stp-dense-formats/events.jsonl'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoding',choices=FORMATS,required=True);p.add_argument('--msb',choices=['off','on'],default='off');p.add_argument('--top-p',choices=['off','on'],default='off');p.add_argument('--p',type=float)
    p.add_argument('--q-bits',type=int);p.add_argument('--k-bits',type=int);p.add_argument('--final-operands',choices=['auto','format','source-int8'],default='auto')
    p.add_argument('--profile',choices=['smoke','full'],required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--model-path',default='/hyperai/input/input0');p.add_argument('--asset-dir',default='/output/project/assets/tq-wt2-v1')
    for flag in ('smoke-root','review-report','controls-manifest','baselines-manifest','dense-smoke-root','dense-summary-manifest'):p.add_argument('--'+flag)
    p.add_argument('--events-path');p.add_argument('--execute',action='store_true');return p


def slice_config_root(suffix):
    return Path('/output/project/SPEC/remote-gpu-platform-slice-SLICE-v2/configs') if suffix=='095' else Path('/output/project/experiments/stp-slice-thresholds/handoffs')/('p'+suffix)/'configs'


def resolve(args):
    fmt=args.encoding;msb=args.msb=='on';topp=args.top_p=='on'
    if fmt=='rsint8':
        from benchmark.attention_rs.selector_adapter import resolve_rs
        return resolve_rs(args,ROOT/'benchmark/attention_rs/matrix.json')
    if not topp and args.p is not None:raise ValueError('--p requires --top-p on; dense diagnostic thresholds are separate and unused in forward')
    if msb and not topp:raise ValueError('MSB without TopP is undefined for the predictor study')
    if topp:
        if args.p is None or not 0<args.p<=1:raise ValueError('TopP requires explicit 0 < --p <= 1')
        if args.p not in THRESHOLDS:raise ValueError('Only .95/.90/.85 are authorized executable study thresholds; p=1 is never a dense alias')
    if msb and fmt in ('fp','mxfp4','mxint8q_mxfp4k'):raise ValueError('This format has no compatible INT-MSB predictor; native FP4 is not MSB truncation')
    final=('source-int8' if msb else 'format') if args.final_operands=='auto' else args.final_operands
    if msb and final!='source-int8':raise ValueError('Reviewed MSB predictors require original source-INT8 final operands')
    suffix=THRESHOLDS.get(args.p);suite=None;matrix=None;historical=False
    if not topp:
        if final!='format':raise ValueError('Dense format attention requires actual decoded format operands')
        widths={'fp':(None,None),'int8':(8,8),'sm8':(8,8),'mxint8':(8,8),'mxfp4':(4,4),'mxint8q_mxfp4k':(8,4)}[fmt]
        if fmt in ('fp','int8'):
            path=slice_config_root('095')/('fp_dense_control.json' if fmt=='fp' else 'int_dense_control.json');suite='historical_baseline';historical=True
        else:
            vid={'sm8':'dense_sm8_q8k8','mxint8':'dense_mxint8_q8k8','mxfp4':'dense_mxfp4_q4k4','mxint8q_mxfp4k':'dense_mxint8q_mxfp4k_q8k4'}[fmt]
            path=ROOT/'benchmark/attention_dense_format/configs'/(vid+'.json');suite='attention_dense_format';matrix=ROOT/'benchmark/attention_dense_format/matrix.json'
    elif msb:
        widths=(8 if args.q_bits is None else args.q_bits,4 if args.k_bits is None else args.k_bits)
        if widths not in ((4,4),(8,4)):raise ValueError('Reviewed MSB predictor widths are Q4K4 and Q8K4')
        bq,bk=widths
        if fmt=='int8':path=slice_config_root(suffix)/(f'msb_q{bq}k4_topp_p{suffix}.json');suite='historical_slice'
        else:path=ROOT/'benchmark/attention_format/configs'/(f'format_{"mxint8_tc4" if fmt=="mxint8" else fmt}_q{bq}k4_topp_p{suffix}.json');suite='historical_format'
        historical=True
    else:
        widths={'fp':(None,None),'int8':(8,8),'sm8':(8,8),'mxint8':(8,8),'mxfp4':(4,4),'mxint8q_mxfp4k':(8,4)}[fmt]
        if final=='source-int8':
            if fmt not in ('mxfp4','mxint8q_mxfp4k'):raise ValueError('Explicit source-int8 without MSB is reserved for the historical native-FP4 predictor profiles')
            path=ROOT/'benchmark/attention_format/configs'/(f'format_{fmt}_q{widths[0]}k{widths[1]}_topp_p{suffix}.json');suite='historical_format';historical=True
        elif fmt in ('sm8','mxint8'):
            vid=f'{fmt}_q8k8_topp_p{suffix}';path=ROOT/'benchmark/attention_full_format/configs'/(vid+'.json');suite='attention_full_format';matrix=ROOT/'benchmark/attention_full_format/matrix.json'
        elif fmt in ('fp','int8'):
            path=slice_config_root(suffix)/(('fp' if fmt=='fp' else 'int')+f'_topp_p{suffix}.json');suite='historical_slice';historical=True
        else:raise ValueError('Native/mixed FP4 full-format TopP is not an implemented authorized study profile; use --final-operands source-int8 only to resolve its historical predictor profile')
    for requested,expected,name in ((args.q_bits,widths[0],'Q'),(args.k_bits,widths[1],'K')):
        if requested is not None and requested!=expected:raise ValueError(name+' width differs from the selected numeric format/profile')
    config=load_config(path)
    if 'attention_full_format' in config:
        c=config['attention_full_format'];vid=c['variant_id'];origin='stored_format_bytes';profile='decoded_full_format';passes=1
    elif 'attention_dense_format' in config:
        c=config['attention_dense_format'];vid=c['variant_id'];origin='stored_format_bytes';profile='decoded_full_format';passes=0
    elif 'attention_slice' in config:
        c=config['attention_slice'];vid=c['variant_id'];origin='original_float' if fmt=='fp' else 'original_source_INT8';profile='original_float' if fmt=='fp' else 'source_int8';passes=1
    else:
        vid='fp_dense_control' if fmt=='fp' else 'int_dense_control';origin='original_float' if fmt=='fp' else 'original_source_INT8';profile='original_float' if fmt=='fp' else 'source_int8';passes=0
    return dict(encoding=fmt,msb=msb,top_p=topp,p=args.p,q_bits=widths[0],k_bits=widths[1],final_operands=final,final_operand_profile=profile,final_operand_origin=origin,selection_passes=passes,variant_id=vid,suite=suite,config_path=str(path.resolve()),config_sha256=sha256_file(path),semantic_config_sha256=canonical_hash(config),config=config,matrix_path=str(matrix.resolve()) if matrix else None,historical_completed_profile=historical,execution_authorized=not historical)


def forwarded_args(args,resolved,**extra):
    rs=resolved['suite']=='attention_rs'
    values=dict(vars(args),matrix=resolved['matrix_path'],variant=resolved['variant_id'] if rs or args.profile=='full' else None,**extra)
    values['events_path']=args.events_path or ('/output/project/experiments/stp-rs-study/events.jsonl' if rs else REGISTERED_EVENTS)
    if rs:
        values['handoff_dir']=str(Path(resolved['matrix_path']).parent)
        values['baseline_root']=None
    return Namespace(**values)

def plan(args):
    resolved=resolve(args);output=Path(args.output_root).resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite '+str(output))
    assets=Path(args.asset_dir).resolve();asset_path=assets/'asset_manifest.json'
    asset=load_json(asset_path);source=code_provenance()
    base=[sys.executable,'-m']
    if resolved['historical_completed_profile']:
        argv=base+['benchmark.ppl','--model-path',args.model_path,'--tokens',str(assets/'tokens.pt'),'--assets',str(asset_path),'--algo-config-path',resolved['config_path'],'--output-dir',str(output),'--ppl-mode','windows']
        if args.profile=='smoke':argv+=['--max-tokens','4096']
        else:argv+=['--slice-gate-report' if resolved['top_p'] else '--gate-report',str(Path(args.smoke_root).resolve()/'gate.json') if args.smoke_root else '<required-matching-historical-gate>']
        effective_scope='historical completed profile; plan only under current no-rerun authorization'
    else:
        events=args.events_path or ('/output/project/experiments/stp-rs-study/events.jsonl' if resolved['suite']=='attention_rs' else REGISTERED_EVENTS)
        argv=base+['benchmark.'+resolved['suite']+'.run','--profile',args.profile,'--matrix',resolved['matrix_path'],'--model-path',args.model_path,'--asset-dir',str(assets),'--output-root',str(output),'--events-path',events]
        for key in ('smoke_root','review_report','controls_manifest','baselines_manifest'):
            if getattr(args,key):argv+=['--'+key.replace('_','-'),str(Path(getattr(args,key)).resolve())]
        if resolved['suite']=='attention_rs':argv+=['--handoff-dir',str(Path(resolved['matrix_path']).parent)]
        if resolved['suite']=='attention_full_format':
            for key in ('dense_smoke_root','dense_summary_manifest'):
                if getattr(args,key):argv+=['--'+key.replace('_','-'),str(Path(getattr(args,key)).resolve())]
        if args.profile=='full':argv+=['--variant',resolved['variant_id']]
        effective_scope='one selected benchmark smoke; no cohort gate is minted' if args.profile=='smoke' else 'one selected full cell bound to the complete matching smoke gate'
    execution_commands=[]
    prerequisites=[]
    if not resolved['historical_completed_profile']:
        forwarded=forwarded_args(args,resolved)
        if resolved['suite']=='attention_rs':
            from benchmark.attention_rs.run import commands as build_commands
        elif resolved['suite']=='attention_dense_format':
            from benchmark.attention_dense_format.run import commands as build_commands
        else:
            from benchmark.attention_full_format.run import commands as build_commands
        selected=[(name,command) for name,command in build_commands(forwarded) if args.profile=='full' or name=='validate' or name.startswith(resolved['variant_id']+'-')]
        execution_commands=[dict(name=name,argv=command) for name,command in selected]
        if args.profile=='smoke':
            argv=next(row['argv'] for row in execution_commands if row['name']==resolved['variant_id']+'-windows')
            prerequisites=['This single-config smoke does not mint the cohort gate; full execution requires the complete independently accepted matching matrix smoke from benchmark.'+resolved['suite']+'.run']
    return dict(status='dry_run',resolved=resolved,argv=argv,execution_commands=execution_commands,gate_prerequisites=prerequisites,output_root=str(output),profile=args.profile,effective_scope=effective_scope,source=dict(head=source['head'],code_sha256=source['code_sha256']),frozen_assets=dict(path=str(asset_path),sha256=sha256_file(asset_path),tokens_sha256=asset['tokens_sha256'],model_file_hashes=asset['model_file_hashes']),model_path=str(Path(args.model_path).resolve()),measurement_state='not_executed')


def execute(args,resolved_plan):
    resolved=resolved_plan['resolved']
    if resolved['historical_completed_profile']:raise ValueError('This profile belongs to the completed frozen studies; this authorization excludes rerunning those measurements')
    if not args.review_report:raise ValueError('--execute requires independent --review-report')
    if args.profile=='full' and not args.smoke_root:raise ValueError('Full execution requires matching --smoke-root before any model load')
    forwarded=forwarded_args(args,resolved,selector_resolution=resolved_plan)
    if resolved['suite']=='attention_rs':
        from benchmark.attention_rs.run import run as rs_run,commands as rs_commands
        def selected_rs(a):return ((name,argv) for name,argv in rs_commands(a) if args.profile=='full' or name=='validate' or name.startswith(resolved['variant_id']+'-'))
        return rs_run(forwarded,plan_builder=selected_rs)
    from benchmark.attention_dense_format.run import run as serial_run
    if resolved['suite']=='attention_dense_format':
        if args.profile=='full':return serial_run(forwarded)
        from benchmark.attention_dense_format.run import commands as dense_commands
        def selected_dense(a):return ((n,v) for n,v in dense_commands(a) if n=='validate' or n.startswith(resolved['variant_id']+'-'))
        return serial_run(forwarded,plan_builder=selected_dense)
    from benchmark.attention_full_format.run import commands
    from benchmark.attention_full_format.formal import read_matrix
    from benchmark.attention_full_format.gate import validate_gate_report
    from benchmark.attention_full_format.summarize import summarize
    def summary(matrix,root,out,profile,baselines):return summarize(matrix,root,out,profile,baselines,args.dense_summary_manifest)
    def selected_full(a):return ((n,v) for n,v in commands(a) if args.profile=='full' or n=='validate' or n.startswith(resolved['variant_id']+'-'))
    return serial_run(forwarded,plan_builder=selected_full,matrix_reader=read_matrix,gate_validator=validate_gate_report,summarizer=summary)


def main():
    import json
    args=parser().parse_args();resolved_plan=plan(args)
    result=execute(args,resolved_plan) if args.execute else resolved_plan
    print(json.dumps(result,indent=2,allow_nan=False));sys.exit(result.get('status')=='failed')
if __name__=='__main__':main()
