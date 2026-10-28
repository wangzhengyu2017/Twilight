"""Independent SLICE-v2 numerical acceptance, on CPU and CUDA when available."""
import argparse
import copy
import math
from pathlib import Path
import torch
from benchmark.attention_baseline.common import configure_runtime, load_config, write_json, code_provenance, environment, canonical_hash
from .formal import matrix_variants, validate_formal_config
from twilight.pyimpl.quant_attention.intflash import quantize_per_token
from twilight.pyimpl.quant_attention.slice_backend import attention_slice, exact_top_p, tc_prefix


def integer_dot_cpu(q, k):
    out = torch.zeros((*q.shape[:-1], k.shape[-2]), dtype=torch.int64)
    for d in range(q.shape[-1]):
        out += q[..., d].to(torch.int64)[..., None] * k[..., d].to(torch.int64)[..., None, :]
    return out


def check_device(device, configs):
    p, variants = matrix_variants(sorted(configs, key=lambda name: (
        0 if name.startswith('fp_') else 1 if name.startswith('int_') else 2 if name.startswith('msb_q4') else 3)))
    for config in configs.values():
        validate_formal_config(config)
    checks = []
    codes = torch.arange(-128, 128, dtype=torch.int16, device=device)
    for bits in range(1, 9):
        hi, lo = tc_prefix(codes, bits)
        a = 1 << (8 - bits)
        assert torch.equal(a * hi + lo, codes)
        assert ((lo >= 0) & (lo < a)).all()
        expected = torch.tensor([int(x) // a for x in range(-128, 128)], dtype=torch.int16, device=device)
        assert torch.equal(hi, expected)
    checks.append('all_256_tc_codes_all_widths')
    generator = torch.Generator(device=device).manual_seed(123)
    q = torch.randn((1, 2, 7, 128), device=device, generator=generator)
    k = torch.randn((1, 2, 11, 128), device=device, generator=generator)
    v = torch.randn((1, 2, 11, 128), device=device, generator=generator)
    qq, kk = quantize_per_token(q), quantize_per_token(k)
    for bq, bk in ((4, 4), (8, 4), (8, 8), (1, 1), (3, 6)):
        qh, ql = tc_prefix(qq.codes, bq)
        kh, kl = tc_prefix(kk.codes, bk)
        aq, ak = 1 << (8-bq), 1 << (8-bk)
        hh = integer_dot_cpu(qh.cpu(), kh.cpu())
        hl = integer_dot_cpu(qh.cpu(), kl.cpu())
        lh = integer_dot_cpu(ql.cpu(), kh.cpu())
        ll = integer_dot_cpu(ql.cpu(), kl.cpu())
        full = integer_dot_cpu(qq.codes.cpu(), kk.codes.cpu())
        assert torch.equal(aq * ak * hh + aq * hl + ak * lh + ll, full)
        actual_hh = torch.matmul(qh.float(), kh.float().transpose(-1, -2)).cpu()
        assert torch.equal(actual_hh.to(torch.int64), hh)
    checks.append('independent_int64_per_dimension_prefix_and_all_cross_terms')
    # Deliberate equal-score keys and an illegal leading/trailing position.
    tied = torch.zeros((1, 1, 1, 6), device=device)
    legal = torch.tensor([[[[False, True, True, True, True, False]]]], device=device)
    mask, ix, counts, valid, _ = exact_top_p(tied, legal, .5)
    assert torch.equal(mask, torch.tensor([[[[False, True, True, False, False, False]]]], device=device))
    assert ix.tolist() == [[[[1, 2]]]] and counts.item() == 2 and valid.all()
    assert torch.equal(exact_top_p(tied, legal, 1.)[0], legal)
    checks.append('stable_ties_ascending_key_positions_mask_shortest_prefix_p1')
    from .replay import top_p
    scalar_logits = torch.tensor([[[[99., 4., 3., 2., 1., 0., -99.]]]], device=device)
    scalar_allowed = torch.tensor([[[[False, True, True, True, True, True, False]]]], device=device)
    for threshold in (.95, .90, .85):
        for scores in (scalar_logits, torch.zeros_like(scalar_logits)):
            selected = exact_top_p(scores, scalar_allowed, threshold)[0]
            expected, probs = top_p(scores.cpu(), scalar_allowed.cpu(), torch.arange(7), threshold)
            assert torch.equal(selected.cpu(), expected)
            legal = [i for i in range(7) if scalar_allowed.flatten()[i]]
            ordered = sorted(legal, key=lambda i: (-float(scores.flatten()[i]), i))
            count = int(selected.sum())
            cumulative = probs.flatten()[ordered].cumsum(0)
            assert float(cumulative[count-1]) >= threshold
            assert count == 1 or float(cumulative[count-2]) < threshold
    checks.append('scalar_oracle_all_formal_thresholds_ties_mask_shortest_prefix')
    allowed = (torch.arange(11, device=device)[None, :] <= torch.arange(4, 11, device=device)[:, None])[None, None]
    captured = {}
    max_error = 0.
    for name, original in configs.items():
        cfg = copy.deepcopy(original['attention_slice'])
        chunks = []
        pv = attention_slice(q, k, v, allowed, None if cfg['estimator']=='fp_full' else qq,
                             None if cfg['estimator']=='fp_full' else kk, cfg, query_chunk=3,
                             observer=lambda start,end,payload: chunks.append((start,end,payload)))
        fq, fk = (q.float(), k.float()) if cfg['estimator']=='fp_full' else (qq.dq, kk.dq)
        reference_full = torch.matmul(fq, fk.transpose(-1, -2)) / math.sqrt(128)
        for start, end, payload in chunks:
            mask = payload['selected_mask']
            assert not (mask & ~allowed[..., start:end, :]).any()
            assert torch.equal(mask.sum(-1), payload['counts'])
            restored = torch.zeros_like(mask)
            # Validate -1 padding without using negative-index gather.
            inds, valid = payload['selected_indices'], payload['valid_mask']
            for b in range(mask.shape[0]):
                for h in range(mask.shape[1]):
                    for row in range(mask.shape[2]):
                        restored[b,h,row,inds[b,h,row][valid[b,h,row]]] = True
            assert torch.equal(restored, mask)
            expected_logits = reference_full[..., start:end, :].masked_fill(~mask, -torch.inf)
            actual = payload['final_selected_logits']
            torch.testing.assert_close(actual, expected_logits, atol=1e-5, rtol=1e-4)
            max_error = max(max_error, float((actual[mask]-expected_logits[mask]).abs().max()))
            expected_pv = torch.matmul(torch.softmax(expected_logits, -1), v.float())
            torch.testing.assert_close(pv[..., start:end, :], expected_pv, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(payload['final_selected_probs'].sum(-1), torch.ones_like(payload['counts'],dtype=torch.float32))
            assert (payload['final_selected_probs'][~mask] == 0).all()
            if cfg['estimator']=='tc_msb':
                hh = integer_dot_cpu(payload['q_hi'].cpu(), payload['k_hi'].cpu())
                assert torch.equal(payload['dot_hh'].cpu().to(torch.int64), hh)
                aq,ak = 1 << (8-cfg['q_prefix_bits']),1 << (8-cfg['k_prefix_bits'])
                est = ((payload['dot_hh'].float()*(aq*ak))*qq.scales[...,start:end,None].float())*kk.scales[...,None,:].float()/math.sqrt(128)
                torch.testing.assert_close(payload['estimator_logits'][allowed[...,start:end,:].expand_as(est)],est[allowed[...,start:end,:].expand_as(est)],atol=0,rtol=0)
        cfg['selection']['p'] = 1.
        p1 = attention_slice(q,k,v,allowed,None if cfg['estimator']=='fp_full' else qq,None if cfg['estimator']=='fp_full' else kk,cfg,query_chunk=3)
        dense = torch.matmul(torch.softmax(reference_full.masked_fill(~allowed,-torch.inf),-1),v.float())
        torch.testing.assert_close(p1,dense,atol=1e-5,rtol=1e-4)
        captured[name] = (pv, chunks)
    checks.append('all_four_variants_selected_full_scores_probability_pv_and_p1_dense')
    cfg=copy.deepcopy(configs[variants[2]]['attention_slice'])
    cfg['q_prefix_bits']=cfg['k_prefix_bits']=8
    out=[]
    pv=attention_slice(q,k,v,allowed,qq,kk,cfg,query_chunk=3,observer=lambda s,e,p:out.append(p))
    int_pv,int_chunks=captured[variants[1]]
    for payload,(_,_,ref) in zip(out,int_chunks):
        assert torch.equal(payload['selected_mask'],ref['selected_mask'])
    torch.testing.assert_close(pv,int_pv,atol=1e-5,rtol=1e-4)
    checks.append('b8_msb_matches_int_topp_same_input')
    return {'status':'passed','device':str(device),'checks':checks,'selected_recompute_max_absolute_error':max_error,'atol':1e-5,'rtol':1e-4}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--config-dir',default=str(Path(__file__).resolve().parents[3] / 'SPEC/remote-gpu-platform-slice-SLICE-v2/configs'))
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    configure_runtime()
    configs={}
    for path in sorted(Path(args.config_dir).glob('*.json')):
        config=load_config(path)
        if 'attention_slice' in config:
            vid = config['attention_slice']['variant_id']
            if vid in configs: raise ValueError('Duplicate validation variant: ' + vid)
            configs[vid]=config
    if len(configs) != 4: raise ValueError('Validation requires all four formal configs')
    p, variants = validate_formal_config(next(iter(configs.values())))
    if set(configs) != set(variants): raise ValueError('Validation requires one coherent formal threshold matrix')
    for config in configs.values():
        validate_formal_config(config)
    report={'schema':'attention-slice-validation-v2','status':'running','cpu':None,'gpu':None,
            'top_p':p, 'variant_ids':list(variants),
            'semantic_config_sha256':{vid:canonical_hash(configs[vid]) for vid in variants},
            'code_sha256':code_provenance()['code_sha256'],
            'environment_sha256':environment()['environment_sha256']}
    write_json(output/'validation.json',report)
    try:
        report['cpu']=check_device(torch.device('cpu'),configs)
        report['gpu']=check_device(torch.device('cuda'),configs) if torch.cuda.is_available() else {'status':'not_run','reason':'CUDA unavailable'}
        report['status']='passed' if report['gpu']['status']=='passed' else 'cpu_passed_gpu_not_run'
    except Exception as error:
        report['status']='failed';report['error']=repr(error)
        write_json(output/'validation.json',report)
        raise
    write_json(output/'validation.json',report)
    print(report['status'])


if __name__=='__main__': main()
