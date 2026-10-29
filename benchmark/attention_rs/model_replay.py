"""Pinned model-weight replay of the captured row; independent of custom forward."""
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from benchmark.attention_baseline.common import load_json
from .oracle import check,compare


class ModelRowReplay:
    def __init__(self,model_path,device):
        self.root=Path(model_path);self.device=device
        self.config=AutoConfig.from_pretrained(str(self.root),local_files_only=True,trust_remote_code=False)
        index=self.root/'model.safetensors.index.json'
        self.mapping=load_json(index)['weight_map'] if index.is_file() else None
        self.cache={}

    def weight(self,layer,name):
        key=f'model.layers.{layer}.'+name
        if key not in self.cache:
            shard=self.root/self.mapping[key] if self.mapping else next(self.root.glob('*.safetensors'))
            with safe_open(str(shard),framework='pt',device='cpu') as reader:self.cache[key]=reader.get_tensor(key)
        return self.cache[key].to(self.device)

    def replay(self,meta,a):
        device=self.device;layer=meta['layer_id'];D=meta['D'];h=meta['Hq'];hkv=meta['Hkv']
        width=len(meta['global_query_positions']);local=meta['local_query_position']
        check(width>0 and 0<=local<width and meta['global_query_positions'][local]==meta['query_position'],'authentic original projection row geometry')
        def embed(row):
            check(row.shape[0]==1 and row.shape[1]==1,'captured authentic row layout')
            full=torch.zeros((1,width,row.shape[-1]),device=device,dtype=row.dtype)
            full[:,local:local+1]=row.to(device)
            return full
        def row(full):return full[:,local:local+1]
        def linear(x,name):return F.linear(x,self.weight(layer,name+'.weight'))
        def norm(x,name):
            z=x.float();z=z*torch.rsqrt(z.pow(2).mean(-1,keepdim=True)+self.config.rms_norm_eps)
            return self.weight(layer,name+'.weight')*z.to(x.dtype)
        hidden=embed(a['layer_input']);normalized=norm(hidden,'input_layernorm')
        compare(a['attn_input'],row(normalized).cpu(),'independent input RMSNorm',True)
        q=row(linear(normalized,'self_attn.q_proj')).view(1,1,h,D).transpose(1,2)
        k=row(linear(normalized,'self_attn.k_proj')).view(1,1,hkv,D).transpose(1,2)
        v=row(linear(normalized,'self_attn.v_proj')).view(1,1,hkv,D).transpose(1,2)
        rope=getattr(self.config,'rope_scaling',None) or {};kind=rope.get('rope_type',rope.get('type','default'))
        inverse,scale=ROPE_INIT_FUNCTIONS[kind](self.config,torch.device(device))
        position=torch.tensor([meta['query_position']],device=device,dtype=torch.float32)
        frequency=position[:,None]*inverse.float()[None,:];angle=torch.cat([frequency,frequency],-1)
        cosine=(angle.cos()*scale).to(q.dtype)[None,None];sine=(angle.sin()*scale).to(q.dtype)[None,None]
        rotate=lambda x:torch.cat([-x[...,D//2:],x[...,:D//2]],-1)
        q=q*cosine+rotate(q)*sine;k=k*cosine+rotate(k)*sine
        compare(a['q_post_rope'],q.cpu(),'weight projection and one RoPE Q',True)
        compare(a['k_visible'][...,-1:,:],k.repeat_interleave(h//hkv,1).cpu(),'current cache K projected/RoPE/GQA',True)
        compare(a['v_visible'][...,-1:,:],v.repeat_interleave(h//hkv,1).cpu(),'current cache V projected/GQA',True)
        projected=linear(embed(a['o_proj_input']),'self_attn.o_proj')
        compare(a['attn_output'],row(projected).cpu(),'independent o_proj output',True)
        residual=hidden+projected;mlp_input=norm(residual,'post_attention_layernorm')
        mlp=linear(F.silu(linear(mlp_input,'mlp.gate_proj'))*linear(mlp_input,'mlp.up_proj'),'mlp.down_proj')
        compare(a['layer_output'],row(residual+mlp).cpu(),'independent residual/MLP layer output',True)
        return dict(status='passed',projection_and_RoPE_exact=True,o_proj_exact=True,layer_output_exact=True,model_definition='pinned_HF_RoPE_frequency_initialization; full_original_M_zero_embedding_authentic_row; independent_linear_norm_rotate_MLP',original_projection_rows=width,authentic_local_row=local)
