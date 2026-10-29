"""BADA post-BSU signed-nibble codec; source conversion is a separate clamp."""
import torch

PROFILE='rsint8_native_source_scale'
PAPER_SHA256='7c2e22b91543435aeb83f230ac6579a07bcfa8f43a138454b77e18149d2609a7'
VERSION='bada-native-source-scale-v1'
PARAMETERS=dict(native_range=[-136,119],source_conversion='integer_clamp_at_source_scale',scale_policy='reuse_original_head_token',zero_point=0,slice_bits=4,packed_layout='post_bsu_H4_L4',paper_sha256=PAPER_SHA256,codec_version=VERSION)
INTEGER_DTYPES=(torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)

def _integer(x,lo,hi,name):
    if not isinstance(x,torch.Tensor) or x.dtype not in INTEGER_DTYPES or ((x.to(torch.int64)<lo)|(x.to(torch.int64)>hi)).any():
        raise ValueError(name+' requires integer tensor in ['+str(lo)+','+str(hi)+']')
    return x.to(torch.int16)

def source_to_native(codes):
    c=_integer(codes,-127,127,'source')
    native=c.clamp(-136,119)
    return native,native!=c,c-native

def encode_native(native):
    r=_integer(native,-136,119,'native encoder')
    raw=r&255;low=raw&15;high=((raw>>4)+(low>>3))&15
    return ((high<<4)|low).to(torch.uint8)

def signed4(n):
    return torch.where(n<8,n,n-16).to(torch.int16)

def decode_native(words):
    w=_integer(words,0,255,'packed decoder')
    return 16*signed4(w>>4)+signed4(w&15)

def decode_prefix4(prefix):
    p=_integer(prefix,0,15,'prefix decoder')
    return 16*signed4(p)

def raw_from_words(words):
    w=_integer(words,0,255,'packed inverse');low=w&15
    return (((((w>>4)-(low>>3))&15)<<4)|low).to(torch.uint8)

def sidecar(source,bits):
    if type(bits) is not int or bits not in (4,8):raise ValueError('RS width must be 4 or 8')
    c,s=source.codes,source.scales
    if c.ndim!=4 or c.dtype!=torch.int8 or s.shape!=c.shape[:-1] or s.dtype!=torch.float16 or not torch.isfinite(s).all() or (s<=0).any():
        raise ValueError('RS requires original INT8 codes and positive unchanged FP16 [B,H,L] scales')
    native,saturation,delta=source_to_native(c)
    words=encode_native(native)
    # All numerical operands below depend on the actual packed post-BSU byte.
    decoded=decode_native(words);high=signed4(words.to(torch.int16)>>4);low=signed4(words.to(torch.int16)&15)
    retained=(words>>4) if bits==4 else words
    predictor=decode_prefix4(retained) if bits==4 else decode_native(retained)
    return dict(raw_input_bytes=raw_from_words(words),rs_words=words,H=high,L=low,retained_bytes=retained,
        native_int=decoded,predictor_int=predictor,scales=s,full_dq=decoded.float()*s.float()[...,None],
        predictor_dq=predictor.float()*s.float()[...,None],saturation_mask=saturation,conversion_residual=delta,
        prefix_residual=decoded-decode_prefix4(words>>4),total_prefix_residual=c.to(torch.int16)-decode_prefix4(words>>4))

def allocation_report(source,side,bits):
    """Logical decode information and actual prototype allocations are distinct."""
    fields={name:dict(bytes=x.numel()*x.element_size(),dtype=str(x.dtype),shape=list(x.shape),aliases_source_scale=(name=='scales' and x.data_ptr()==source.scales.data_ptr()),aliases=[other for other,y in side.items() if other!=name and x.data_ptr()==y.data_ptr()]) for name,x in side.items()}
    unique={x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for name,x in side.items() if name!='scales'}
    return dict(logical_payload_bits=source.codes.numel()*bits,required_source_scale_bits=source.scales.numel()*source.scales.element_size()*8,
        bits_per_element_with_scale=bits+source.scales.element_size()*8/source.codes.shape[-1],fields=fields,unique_sidecar_allocation_bytes=sum(unique.values()),
        source_code_coexistence_bytes=source.codes.numel()*source.codes.element_size(),source_dq_coexistence_bytes=source.dq.numel()*source.dq.element_size(),
        storage_policy='uint8 prefix containers; unpacked int16 decode; FP32 DQ; original floating cache and V coexist',native_int4_mac_claim=False)
