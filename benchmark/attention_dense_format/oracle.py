"""Independent scalar byte reconstruction; production codec is never imported."""
import math
import torch
from benchmark.attention_format.oracle import check,compare,source_quantization,mx_sidecar,sm8,top_p,ATOL,RTOL


def formats(config):
    fmt=config['format'];return ('mxint8','mxfp4') if fmt=='mxint8q_mxfp4k' else (fmt,fmt)


def scalar_decode(codes,fmt,scales=None,e8m0=None):
    shape=codes.shape;result=[]
    if fmt=='sm8':
        check(codes.dtype==torch.uint8 and scales.shape==shape[:-1],'invalid SM stored bytes')
        for row,scale in zip(codes.reshape(-1,shape[-1]).tolist(),scales.reshape(-1).tolist()):
            for word in row:result.append((-(word&127) if word&128 and word&127 else word&127)*scale)
    else:
        fp4=fmt=='mxfp4';check(fmt in ('mxint8','mxfp4'),'unknown scalar codec')
        check(codes.dtype==(torch.uint8 if fp4 else torch.int8),'invalid stored code dtype')
        check(e8m0.dtype==torch.uint8 and e8m0.shape==(*shape[:-1],shape[-1]//32),'invalid stored E8M0 layout')
        for block,encoded_e in zip(codes.reshape(-1,32).tolist(),e8m0.reshape(-1).tolist()):
            check(encoded_e!=255,'nonfinite E8M0');e=encoded_e-127
            for word in block:
                if fp4:
                    check(0<=word<=15,'invalid stored E2M1');value=(0.,.5,1.,1.5,2.,3.,4.,6.)[word&7]
                    if word&8 and value:value=-value
                    result.append(math.ldexp(value,e))
                else:
                    check(-127<=word<=127,'symmetric endpoint violated');result.append(math.ldexp(word,e-6))
    output=torch.tensor(result,dtype=torch.float32).reshape(shape)
    check(torch.isfinite(output).all(),'nonfinite scalar byte decode');return output


def scalar_encode(codes,scales,fmt):
    if fmt=='sm8':
        encoded=sm8(codes,8);return dict(full_codes=encoded['full_codes'],full_dq=scalar_decode(encoded['full_codes'],fmt,scales=scales))
    fp4=fmt=='mxfp4';encoded=mx_sidecar(codes.float()*scales.float()[...,None],4 if fp4 else 8,fp4)
    result={k:encoded[k] for k in ('full_codes','exponent','e8m0','clamp_mask','zero_blocks','underflow_mask')}
    result['full_dq']=scalar_decode(result['full_codes'],fmt,e8m0=result['e8m0']);return result


def check_sets(data,mask,positions,prefix):
    compare(data[prefix+'mask'],mask,prefix+'mask',True)
    counts=mask.sum(-1);compare(data[prefix+'counts'],counts,prefix+'counts',True)
    indices=data[prefix+'indices'];valid=data[prefix+'valid_mask']
    check(indices.dtype==torch.int64 and valid.dtype==torch.bool and indices.shape==valid.shape,'diagnostic index layout')
    check(torch.equal(valid.sum(-1),counts) and (indices[~valid]==-1).all(),'diagnostic index counts/padding')
    reconstructed=((indices[...,None]==positions)&valid[...,None]).any(-2)
    compare(reconstructed,mask,prefix+'indices agree',True)
