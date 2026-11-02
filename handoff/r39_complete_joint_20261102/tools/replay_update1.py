"""R39X diagnostic (not a formal unit): replay updates 0-1 of one JOINT config in a scratch dir with the in-unit phase-1 checks bypassed
(phase=2 label, no readout, no reference), save checkpoints at steps 1 and 2, and for every student tensor report: update-1 gradient
(recovered exactly from AdamW exp_avg after update 1: g0 = 0 because the head output layer is zero at update 0, so exp_avg = (1-b1) g1),
the predicted update-1 AdamW step, the fp32 spacing of the tensor's values, and whether the tensor actually changed.
Usage: python -B tools/replay_update1.py --tree r39xc --unit R39X_CC152K_R37S_s1001_N_joint_global_J"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tree', required=True); ap.add_argument('--unit', required=True); a = ap.parse_args()
    sys.path.insert(0, str(W / a.tree))
    import torch, common as C, train_unit as TU
    C.install_guards(role='gate'); TU.set_determinism()
    cfg = TU.cfg_of(a.unit); cfg['_config_sha256'] = C.sha256(C.ROOT / 'configs' / f'{a.unit}.json')
    cfg.update(reference_unit=None, phase=2)   # phase label only: skips the phase-1 in-unit asserts; no parent_ckpt -> fresh start as the unit
    dev = torch.device('cuda')
    ctx = TU.Ctx(cfg['cond'], dev, need_local=False); ctx.ensure_joint()
    sdir = C.OUT / 'diag' / 'replay_update1' / a.unit; shutil.rmtree(sdir, ignore_errors=True)
    TU.run_unit(dict(cfg), ctx, out_dir=sdir, steps=2, ckpt_steps=[0, 1, 2], no_readout=True)
    c1 = torch.load(sdir / 'ckpt_step1.pt', map_location='cpu', weights_only=False); c2 = torch.load(sdir / 'ckpt_step2.pt', map_location='cpu', weights_only=False)
    o = cfg['optimizer']; b1, b2 = o['betas']; eps = o['eps']; wd = o['weight_decay']
    lr1 = c2['optimizer']['param_groups'][1]['lr']
    # optimizer param ids: group 0 = heads+nulls, group 1 = student trainable params in stud[p].trainable() order (A then B)
    stud_names = []
    s1 = c1['student']
    from r39ext import build_students
    st = build_students(ctx.joint.ck, torch.device('cpu'))
    for p in C.PEERS:
        stud_names += [f'{p}.{n}' for n, q in st[p].named_parameters() if q.requires_grad]
    ids = c2['optimizer']['param_groups'][1]['params']; assert len(ids) == len(stud_names)
    rows = []
    for name, pid in zip(stud_names, ids):
        stt = c2['optimizer']['state'][pid]; t = float(stt['step'])
        assert t == 2.0, t
        m, v = stt['exp_avg'].double(), stt['exp_avg_sq'].double()
        g1 = m / (1 - b1)   # exact up to fp32 rounding of exp_avg (g0 == 0)
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        th1 = s1[name].double()
        step = lr1 * mh / (vh.sqrt() + eps) + lr1 * wd * th1
        th1f = th1.float(); spacing = (torch.nextafter(th1f.abs(), torch.tensor(float('inf'))) - th1f.abs()).double()
        th2 = c2['student'][name]
        changed = not torch.equal(th2, s1[name])
        ratio = (step.abs() / spacing).max().item()
        rows.append(dict(tensor=name, numel=int(th1.numel()), grad1_absmax=float(g1.abs().max()), grad1_norm=float(g1.norm()), pred_step_absmax=float(step.abs().max()),
                         spacing_min=float(spacing.min()), max_step_over_spacing=ratio, n_elem_step_gt_half_spacing=int((step.abs() > 0.5 * spacing).sum()),
                         changed=changed, abs_value_min=float(th1.abs().min())))
    out = dict(unit=a.unit, lr_update1=lr1, eps=eps, rows=rows, gpu=torch.cuda.get_device_name(0),
               unchanged=[r for r in rows if not r['changed']], note='diagnostic replay; not a formal unit; outputs under OUT/diag')
    C.write_json(sdir / 'REPLAY_UPDATE1.json', out); C.write_json(W / 'DIAG_REPLAY_UPDATE1_R37S_J.json', out)
    print('R39X_REPLAY unchanged', json.dumps(out['unchanged']))
    for r in sorted(rows, key=lambda r: r['max_step_over_spacing'])[:5]:
        print('lowest step/spacing', json.dumps(r))


if __name__ == '__main__':
    main()
