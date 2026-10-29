"""Stable seam for benchmark.attention_benchmark after accepted non-RS merge."""
from pathlib import Path
from benchmark.attention_baseline.common import sha256_file,canonical_hash
from .formal import read_matrix
from .oracle import check

def resolve_rs(args,matrix_path):
    """Use the shared selector's exact --encoding/--msb/--top-p/--p flags."""
    matrix_path=Path(matrix_path).resolve();matrix,configs=read_matrix(matrix_path);check(args.encoding=='rsint8','RS encoding must be explicit')
    msb=args.msb=='on';topp=args.top_p=='on';p=args.p
    check(not msb or topp,'RS MSB requires Top-p')
    final=('source-int8' if msb else 'format') if args.final_operands=='auto' else args.final_operands
    check(final==('source-int8' if msb else 'format'),'RS final operand origin differs from reviewed mode')
    if not topp:
        check(p is None,'dense RS has no Top-p threshold');q,k=8,8;vid='rs_dense_q8k8'
    else:
        check(type(p) is float and p in (.95,.9,.85),'RS requires explicit .95/.90/.85')
        q=(8 if args.q_bits is None else args.q_bits) if msb else 8;k=4 if msb else 8
        check((q,k) in ((8,4),(4,4)) if msb else (q,k)==(8,8),'RS widths differ')
        suffix={.95:'095',.9:'090',.85:'085'}[p];vid='rs_'+('msb' if msb else 'full')+'_q'+str(q)+'k'+str(k)+'_topp_p'+suffix
    check(args.q_bits in (None,q) and args.k_bits in (None,k),'RS requested widths differ')
    config=configs[vid];s=config['attention_rs'];file=matrix_path.parent/next(row['config'] for row in matrix['variants'] if row['variant_id']==vid)
    return dict(encoding='rsint8',msb=msb,top_p=topp,p=p,q_bits=q,k_bits=k,final_operands=final,final_operand_profile='source_int8' if msb else 'rs_native_full_source_scale',final_operand_origin=s['final_operand_origin'],selection_passes=s['selection_passes'],variant_id=vid,suite='attention_rs',config_path=str(file),config_sha256=sha256_file(file),semantic_config_sha256=canonical_hash(config),config=config,matrix_path=str(matrix_path),historical_completed_profile=False,execution_authorized=True,implementation_acceptance='requires_current_independent_review_report',gpu_verification='not_established_by_selector')
