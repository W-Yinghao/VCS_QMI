"""CR4 read-only CE-decomposition / gradient-component audit (no training update, no truth).

For each fixed graph and checkpoint: a fixed diagnostic draw stream (seed 20261104061, 20 batches x 64 edges),
CR3 COMPLETE packing, scores from the checkpoint model; (1) per-view arrays -> package tools/audit_logits.py
(CE = separation + H + KL, positive score-down fraction/mass, PC partial signs; per direction);
(2) G-parameter gradients of the weighted separation, KL, CE and PC terms (norms, cosines, CE = sep + KL check).
--when before: checkpoint step 0 (common init) and CR3 COMPLETE-CE steps 250/8000.  --when after: CR4 PC steps 250/8000.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr4_student as S                                                   # noqa: E402
C1 = S.C1
from reference.support_complete import FixedRelation                      # noqa: E402
from reference.positive_exclusive import ce_decomposition, packed_loss    # noqa: E402
from cr1.models import GlobalScorer                                       # noqa: E402

CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR3S = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104/students_A1')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104'); PY = '/home/infres/yinwang/CS_QMI/env_nr1/bin/python'
TOOL = HERE.parent/'pkg'/'tools'/'audit_logits.py'


def cos(a, b): return float((a @ b)/(a.norm()*b.norm()+1e-30))


def audit_one(fz, unit, ckpt, tag, dev, nb=20):
    c = fz['units'][unit]['condition']; rel = fz['units'][unit]['graph_path']
    bank, prep, graph = C1.get_prepared(CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'); X, Y = C1.features(bank, dev)
    m = GlobalScorer(768).to(dev); m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)['model'])
    mat = load_npz(rel).tocsr(); relation = C1.ExposedRelation(mat); fr = FixedRelation(mat); rng = np.random.default_rng(20261104061)
    S_, A_, M_, W_, D_ = [], [], [], [], []; G = {k: None for k in ('sep', 'kl', 'ce', 'pc')}; ent = 0.0
    params = list(m.parameters())
    for _ in range(nb):
        b = relation.batch_edges(rng, 64); pb = fr.build(fr.from_edge_indices(b['edges']), 'COMPLETE')
        ps = S.score_pairs(m, X, Y, pb.pairs).double()
        g = torch.as_tensor(pb.gather, device=dev); mk = torch.as_tensor(pb.mask, device=dev); a = torch.as_tensor(pb.targets, device=dev)
        w = torch.as_tensor(pb.query_weights, device=dev) / nb
        s = ps[g].masked_fill(~mk, 0.0)
        parts = ce_decomposition(s, a, mk)
        terms = {'sep': parts['separation'].dot(w), 'kl': parts['within_positive_kl'].dot(w), 'ce': parts['ce'].dot(w), 'pc': packed_loss(ps, pb, kind='PC')/nb}
        ent += float(parts['entropy'].dot(w))
        for k, v in terms.items():
            gr = torch.autograd.grad(v, params, retain_graph=True)
            gv = torch.cat([x.flatten() for x in gr]).double()
            G[k] = gv if G[k] is None else G[k]+gv
        S_.append(s.detach().cpu().numpy()); A_.append(pb.targets); M_.append(pb.mask); W_.append(pb.query_weights/nb)
        D_.append(np.array([0 if v.direction == 'i2t' else 1 for v in pb.views]))
    width = max(x.shape[1] for x in S_)
    pad = lambda arrs, fill, dt: np.concatenate([np.pad(x, ((0, 0), (0, width-x.shape[1])), constant_values=fill).astype(dt) for x in arrs])
    npz = OUT/'audit'/f'{tag}.npz'; npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, scores=pad(S_, 0.0, np.float64), targets=pad(A_, 0.0, np.float64), mask=pad(M_, False, bool), query_weights=np.concatenate(W_), directions=np.concatenate(D_))
    js = OUT/'audit'/f'{tag}.logits.json'
    if js.exists(): js.unlink()
    subprocess.run([PY, '-I', '-B', str(TOOL), '--npz', str(npz), '--out', str(js)], check=True, capture_output=True)
    npz.unlink()   # keep outputs small; the JSON report is the deliverable
    gs = {k: v for k, v in G.items()}
    return {'unit': unit, 'checkpoint': str(ckpt), 'batches': nb, 'diag_seed': 20261104061, 'logits_report': json.loads(js.read_text()),
            'param_grad': {'norm_sep': float(gs['sep'].norm()), 'norm_kl': float(gs['kl'].norm()), 'norm_ce': float(gs['ce'].norm()), 'norm_pc': float(gs['pc'].norm()),
                           'cos_sep_kl': cos(gs['sep'], gs['kl']), 'cos_ce_pc': cos(gs['ce'], gs['pc']), 'cos_pc_sep': cos(gs['pc'], gs['sep']), 'cos_ce_sep': cos(gs['ce'], gs['sep']),
                           'ce_minus_sep_minus_kl_max_abs': float((gs['ce']-gs['sep']-gs['kl']).abs().max()), 'weighted_target_entropy': ent},
            'note': 'read-only fixed-score diagnostic at a fixed checkpoint; not a mechanism proof or a ranking result'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--when', choices=['before', 'after'], required=True); ap.add_argument('--condition', default=None); a = ap.parse_args()
    fz = json.loads((HERE.parents[1]/'PROTOCOL_FROZEN_CR4.json').read_text()); dev = 'cuda'; C1.setup_determinism(dev); res = {}
    for u, info in fz['units'].items():
        if a.condition and info['condition'] != a.condition: continue
        c, r = info['condition'], info['relation']
        if a.when == 'before':
            cr3 = CR3S/c/f'CR3_{c}_{r}_COMPLETE_s1001'
            cks = [('init_step0', cr3/'step_000000.pt'), ('CR3_COMPLETE_CE_step250', cr3/'step_000250.pt'), ('CR3_COMPLETE_CE_step8000', cr3/'step_008000.pt')]
        else:
            d = Path(info['result_dir']); uc = d/'UNIT_COMPLETE.json'
            if uc.exists() and json.loads(uc.read_text()).get('reused_dir'): d = Path(json.loads(uc.read_text())['reused_dir'])
            cks = [('CR4_PC_step250', d/'step_000250.pt'), ('CR4_PC_step8000', d/'step_008000.pt')]
        for tag, ck in cks:
            if ck.exists(): res[f'{u}:{tag}'] = audit_one(fz, u, ck, f'{a.when}_{u}_{tag}', dev); print(u, tag, res[f'{u}:{tag}']['param_grad'], flush=True)
    p = OUT/'audit'/f"GRAD_COMPONENT_AUDIT_{a.when}{'_'+a.condition if a.condition else ''}.json"; p.write_text(json.dumps(res, indent=1)); print('wrote', p)


if __name__ == '__main__': main()
