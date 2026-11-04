"""R41-F: R40 r40ext.py + reference mode "encoder_lr_variant" (optimizer compared except lr_encoder).
R40 additions, byte-identical in both code trees (r40f Flickr60, r40c CC152K).

 - extra_references(cfg): read-only reused references besides the R38 N-FROZEN unit (cfg['reference_unit'], R38 rule): the R39 N-JOINT
   unit of the same parent/objective ('reference_joint': head+null init hash, student init-state hash, batch-index hash chain) and, for the
   Flickr oracle-JOINT cells, the R39 ORACLE-FROZEN unit ('reference_oracle': head+null init hash, batch chain). Asserted in-unit.
 - present_recheck(d, present): independent recomputation of the fixed N_PRESENT_ONLY normaliser p_d from the full FIT observed map and the
   frozen candidate tables (all FIT records through the production null_lists/valid_rows path; no batch, no truth) == data36.presence_probs.
 - masked_rows_grad(gl, masks): N_PRESENT_ONLY rows with no observed-positive mass on a real slot must have exactly zero dL/dg (all 33 slots).
 - one_step_rule(...): active-path check of update 1 (the first update with a nonzero critic output): every declared-active tensor (head,
   null, enabled student modality) must change unless its update-1 gradient is exactly zero or its predicted AdamW step (from the optimizer
   state after the update) is below the fp32 spacing of every element; frozen-modality tensors must be requires_grad False, without .grad,
   outside the optimizer and bit-unchanged. Per-tensor values recorded.
"""
from __future__ import annotations
import numpy as np
import torch

from common import PEERS, sha256, read_json


# ============================================================================ references

def extra_references(cfg):
    out = {}
    for key in ('reference_joint', 'reference_oracle'):
        ref = cfg.get(key)
        if not ref:
            continue
        for k in ('result_json', 'train_history_json'):
            assert sha256(ref[k]) == ref[k + '_sha256'], f'{key} file changed: {ref[k]}'
        res = read_json(ref['result_json']); hist = read_json(ref['train_history_json'])
        rc = res['config']
        assert rc['ckpt_steps'] == cfg['ckpt_steps'] and rc['sampler_seed'] == cfg['sampler_seed'] and rc['init_seed'] == cfg['init_seed'] and rc['steps'] == cfg['steps']
        skip = ('lr_encoder',) if ref.get('mode') == 'encoder_lr_variant' else ()   # R41-F: the variant differs from the reference only in the student lr
        assert rc['objective'] == cfg['objective'] and rc['cond'] == cfg['cond'] and {k: v for k, v in rc['optimizer'].items() if k not in skip} == {k: v for k, v in cfg['optimizer'].items() if k in rc['optimizer'] and k not in skip}
        out[key] = dict(unit=ref['unit_id'], init=res['phases'][0]['identity']['init_state_sha256'], chain=list(hist['batch_hash']),
                        student_init=ref.get('student_init_state_sha256'))
    return out


# ============================================================================ N_PRESENT_ONLY

def present_recheck(d, present, null_lists, valid_rows, chunk=20000):
    """p_d over all FIT records via the training sampler's own list builder (identity permutation; validity is permutation invariant)."""
    obs = d['obs_np']; n = len(obs); k = d['cand_i2t'].shape[1] + 1
    cnt_i = cnt_t = 0
    for s in range(0, n, chunk):
        r = np.arange(s, min(n, s + chunk))
        perm = np.broadcast_to(np.arange(k), (len(r), k)).copy()
        lists = null_lists(r, r, obs, d['cand_i2t'], d['cand_t2i'], perm, perm)
        mi, mt = valid_rows(lists)
        cnt_i += int(mi.sum()); cnt_t += int(mt.sum())
    rec = dict(p_i2t_recomputed=cnt_i / n, p_t2i_recomputed=cnt_t / n, p_i2t_production=present['p_i2t'], p_t2i_production=present['p_t2i'],
               n_fit_records=int(n), inputs='FIT observed map (cache observed_map / noise file) + frozen natural top-32 candidate tables only; no batch statistic, no truth')
    rec['abs_diff_i2t'] = abs(rec['p_i2t_recomputed'] - rec['p_i2t_production']); rec['abs_diff_t2i'] = abs(rec['p_t2i_recomputed'] - rec['p_t2i_production'])
    rec['passed'] = bool(rec['abs_diff_i2t'] <= 1e-12 and rec['abs_diff_t2i'] <= 1e-12 and 0 < present['p_i2t'] <= 1 and 0 < present['p_t2i'] <= 1)
    return rec


def masked_rows_grad(gl, masks):
    """returns (n_violations, n_masked_rows, n_valid_rows_with_nonzero_grad) over peers and directions."""
    viol = 0; nm = 0; nz = 0
    for gi, gt in gl:
        for g, m in ((gi, masks[0]), (gt, masks[1])):
            m = torch.as_tensor(m, device=g.device)
            G = g.grad if g.grad is not None else torch.zeros_like(g)
            viol += int((G[~m] != 0).any(1).sum()); nm += int((~m).sum()); nz += int((G[m] != 0).any(1).sum())
    return viol, nm, nz


# ============================================================================ one-step active-path rule

def active_named(heads, nulls, stud):
    out = []
    for p in PEERS:
        out += [(f'head.{p}.{n}', q) for n, q in heads[p].named_parameters()]
        if nulls is not None:
            out.append((f'null.{p}', nulls.v[p]))
        if stud is not None:
            out += [(f'student.{p}.{n}', q) for n, q in stud[p].named_parameters() if q.requires_grad]
    return out


def frozen_modality_named(stud):
    out = []
    if stud is None:
        return out
    for p in PEERS:
        fm = tuple(m + '.' for m in ('img', 'txt') if m not in stud[p].modalities)
        out += [(f'student.{p}.{n}', q) for n, q in stud[p].named_parameters() if n.startswith(fm)] if fm else []
    return out


@torch.no_grad()
def snapshot(named):
    return {n: q.detach().clone() for n, q in named}


@torch.no_grad()
def grad_record(named):
    return {n: (None if q.grad is None else float(q.grad.abs().max())) for n, q in named}


@torch.no_grad()
def predicted_step(opt, q, before):
    """AdamW update just applied to q, from the post-update optimizer state (torch AdamW: p *= 1 - lr wd; p -= lr/bc1 * m / (sqrt(v)/sqrt(bc2) + eps))."""
    grp = next(g for g in opt.param_groups if any(x is q for x in g['params']))
    st = opt.state.get(q)
    if not st:   # never had a .grad: torch AdamW skips it entirely (no decay, no step)
        z = torch.zeros_like(before, dtype=torch.float64)
        return z, z, dict(adam_step=0.0, lr=grp['lr'])
    t = float(st['step']); b1, b2 = grp['betas']; lr = grp['lr']; wd = grp['weight_decay']; eps = grp['eps']
    m = st['exp_avg'].double(); v = st['exp_avg_sq'].double(); p0 = before.double()
    delta = -lr * wd * p0 - (lr / (1 - b1 ** t)) * m / (v.sqrt() / (1 - b2 ** t) ** .5 + eps)
    spacing = torch.from_numpy(np.asarray(np.spacing(np.abs(before.detach().cpu().numpy().astype(np.float32))), dtype=np.float64)).to(delta.device)   # 0-d safe
    ratio = delta.abs() / spacing
    return delta, ratio, dict(adam_step=t, lr=lr)


@torch.no_grad()
def one_step_rule(opt, active, before, grads1, frozen_named, frozen_ref):
    opt_ids = {id(x) for g in opt.param_groups for x in g['params']}
    per = {}; unexplained = []; changed_by_peer = {p: 0 for p in PEERS}
    for n, q in active:
        ch = not torch.equal(q.detach(), before[n])
        delta, ratio, info = predicted_step(opt, q, before[n])
        actual = (q.detach().double() - before[n].double())
        rec = dict(changed=bool(ch), grad1_max_abs=grads1.get(n), grad1_exactly_zero=bool(grads1.get(n) in (None, 0.0)),
                   predicted_step_max_abs=float(delta.abs().max()), predicted_step_over_fp32_spacing_max=float(ratio.max()),
                   frac_elements_predicted_ge_spacing=float((ratio >= 1).double().mean()), actual_step_max_abs=float(actual.abs().max()),
                   in_optimizer=id(q) in opt_ids, **info)
        rec['explained'] = bool(ch or rec['grad1_exactly_zero'] or rec['predicted_step_over_fp32_spacing_max'] < 1.0)
        if not rec['explained'] or not rec['in_optimizer']:
            unexplained.append(n)
        if ch:
            changed_by_peer[n.split('.')[1]] += 1
        per[n] = rec
    fz = {}
    for n, q in frozen_named:
        fz[n] = dict(requires_grad=bool(q.requires_grad), has_grad=q.grad is not None, in_optimizer=id(q) in opt_ids, bit_unchanged_vs_parent=bool(torch.equal(q.detach(), frozen_ref[n])))
    fz_ok = all(not r['requires_grad'] and not r['has_grad'] and not r['in_optimizer'] and r['bit_unchanged_vs_parent'] for r in fz.values())
    student_changed = {p: sum(1 for n, r in per.items() if n.startswith(f'student.{p}.') and r['changed']) for p in PEERS}
    head_changed = {p: sum(1 for n, r in per.items() if n.startswith(f'head.{p}.') and r['changed']) for p in PEERS}
    out = dict(rule='R40: every active tensor (head, null, enabled student modality) changes at update 1 unless its update-1 gradient is exactly zero or the '
                    'predicted AdamW step is below the fp32 spacing of every element; frozen-modality tensors requires_grad False, no .grad, not in the '
                    'optimizer, bit-unchanged; >= 1 head tensor and (if a student exists) >= 1 student tensor changed per peer',
               n_active=len(per), n_changed=sum(r['changed'] for r in per.values()), unexplained_unchanged=unexplained,
               unchanged_explained={n: dict(grad1_exactly_zero=r['grad1_exactly_zero'], predicted_step_over_fp32_spacing_max=r['predicted_step_over_fp32_spacing_max'])
                                    for n, r in per.items() if not r['changed']},
               head_tensors_changed=head_changed, student_tensors_changed=student_changed, n_frozen_modality_tensors=len(fz), frozen_modality_ok=bool(fz_ok),
               per_tensor=per, frozen_modality=fz)
    has_stud = any(n.startswith('student.') for n in per)
    out['passed'] = bool(not unexplained and fz_ok and all(v > 0 for v in head_changed.values()) and (not has_stud or all(v > 0 for v in student_changed.values())))
    return out
