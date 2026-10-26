"""KD relation-vector headroom check on fixed epoch-240 checkpoints (FIT only, eval-mode, no augmentation, no training).

Relation vector of sample x w.r.t. a fixed FIT reference set A (2 per class, 200, disjoint from queries):
    r(x) = [cos(h(x), h(a_j))]_j  with h = pooled penultimate features (student WRN16-2, teacher WRN40-2).
Questions (review 2026-10-25, KD section): does the teacher's hidden relation structure carry information its own logits do not,
and is it already transferred to the distilled students? Plus the VCS P/Q separability of relation vectors (saturation).
"""
import hashlib, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'handoff/r22_other_apps_20261023/engine'))
import guard  # noqa: E402
import torch  # noqa: E402
from kd import runner as K, decision as D  # noqa: E402
sys.path.insert(0, str(BASE / 'handoff/r22_kd_link_20261025'))
import run as LINK  # noqa: E402  (fit_link / t_stats / identity_j, frozen in that package)

RUNS = BASE / 'handoff/r22_other_apps_20261023/runs'
NAMES = [f'kd_full_{a}_{m}_l{l}_s{s}' for a in ('ce', 'dkd') for m, l in (('none', 0), ('cosine', 10), ('vcs', 10), ('bce', 10 if a == 'ce' else 1))
         for s in (31415, 16180, 57721)]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def write(p, d):
    tmp = Path(str(p) + f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + '\n'); tmp.replace(p)


def rowwise_pearson(a, b):
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def identity_stats(S):
    n = S.shape[-1]; mask = ~torch.eye(n, dtype=torch.bool); p, qq = S.diagonal(dim1=-2, dim2=-1).reshape(-1), S[:, mask].reshape(-1)
    return dict(P_mean=float(p.mean()), Q_mean=float(qq.mean()), P_abs_ge099=float((p.abs() >= .99).double().mean()),
                Q_abs_ge099=float((qq.abs() >= .99).double().mean()), P_grad_factor_mean=float((1 - p).mean()))


def extract(model, loader, device):
    model.eval(); L, H, I, Y = [], [], [], []
    with torch.no_grad():
        for images, labels, ids in loader:
            logits, feat = model(images.to(device)); L.append(logits.double().cpu()); H.append(feat.double().cpu()); I.append(ids); Y.append(labels)
    return torch.cat(L), torch.cat(H), torch.cat(I), torch.cat(Y)


def relations(H, Lg, q, a):
    hn = torch.nn.functional.normalize(H, dim=1); ln = D.coordinates(Lg)
    return hn[q] @ hn[a].T, ln[q] @ ln[a].T


def main():
    assert os.environ.get('SLURM_JOB_ID'); torch.set_num_threads(4); started = time.time(); device = torch.device('cuda')
    out = HERE / 'COMPLETE.json'; assert not out.exists()
    cfg = json.loads((RUNS / NAMES[0] / 'resolved_config.json').read_text()); guard.install(guard.kd_roots(cfg))
    manifest = K.read_manifest(cfg['data']['manifest'])
    fit = K.RoleDataset(cfg['data']['root'], manifest, 'fit', augment=False, seed=0)
    loader = K._loader(fit, cfg, False, 0)
    teacher, _, _ = K._teacher(cfg, manifest, device)
    LT, HT, I, Y = extract(teacher, loader, device)
    # reference set: per class the 2 FIT ids with smallest sha256(id); queries = all other FIT samples
    key = torch.tensor([int(hashlib.sha256(str(int(i)).encode()).hexdigest()[:12], 16) for i in I])
    anchors = []
    for c in Y.unique():
        idx = torch.nonzero(Y == c)[:, 0]; anchors += idx[key[idx].argsort()[:2]].tolist()
    a = torch.tensor(sorted(anchors)); qmask = torch.ones(len(I), dtype=torch.bool); qmask[a] = False; q = torch.nonzero(qmask)[:, 0]
    RT, LRT = relations(HT, LT, q, a)
    teacher_internal = float(rowwise_pearson(RT, LRT).mean())
    results = dict(n_queries=len(q), n_anchors=len(a), anchor_ids_sha256=hashlib.sha256(I[a].numpy().tobytes()).hexdigest(),
                   teacher_hidden_vs_own_logit_relation=teacher_internal, students={})
    for name in NAMES:
        d = RUNS / name; rec = json.loads((d / 'run.json').read_text()); assert rec['status'] == 'complete' and rec['epochs_completed'] == 240
        scfg = json.loads((d / 'resolved_config.json').read_text()); seed = scfg['seed']
        ckpt = torch.load(d / 'last.pt', map_location='cpu', weights_only=True); assert ckpt['epoch'] == 239
        model = K.create_model(scfg['student'], manifest['num_classes'], seed); K.feature_dim(model)
        model = model.to(device); model.load_state_dict(ckpt['student'])
        LS, HS, IS, _ = extract(model, loader, device); assert torch.equal(IS, I)
        RS, LRS = relations(HS, LS, q, a)
        # VCS P/Q on centered-L2 relation vectors: fit (a,b) on even query ids, evaluate on odd, fixed 64-batches
        us, vt = D.coordinates(RS), D.coordinates(RT); stacks = {}
        for h in (0, 1):
            sel = torch.nonzero(I[q] % 2 == h)[:, 0]; sel = sel[torch.randperm(len(sel), generator=torch.Generator().manual_seed(seed + h))]
            m = (len(sel) // 64) * 64; sel = sel[:m].reshape(-1, 64); stacks[h] = us[sel] @ vt[sel].transpose(-1, -2)
        fitv = LINK.fit_link(stacks[0], 'vcs_link')
        results['students'][name] = dict(
            seed=seed, checkpoint_sha256=sha(d / 'last.pt'), final_cal_top1=json.loads((d / 'history.json').read_text())[-1]['cal']['top1'],
            hidden_relation_agreement=float(rowwise_pearson(RS, RT).mean()),
            logit_relation_agreement=float(rowwise_pearson(LRS, LRT).mean()),
            student_logit_vs_teacher_hidden_relation=float(rowwise_pearson(LRS, RT).mean()),
            relation_J_identity_heldout=float(LINK.identity_j(stacks[1])),
            relation_identity_stats=identity_stats(stacks[1]),
            relation_link_fit=fitv, relation_link_heldout=LINK.t_stats(stacks[1], fitv['a'], fitv['b']))
        s = results['students'][name]
        print(name, 'hidden %.3f logit %.3f sLogit-tHidden %.3f J_id %.3f' % (s['hidden_relation_agreement'], s['logit_relation_agreement'],
              s['student_logit_vs_teacher_hidden_relation'], s['relation_J_identity_heldout']), flush=True)
        del model, ckpt; torch.cuda.empty_cache()
    status = guard.status(); assert not status.get('forbidden_hits', []), status
    results.update(status='COMPLETE', job=os.environ['SLURM_JOB_ID'], seconds=time.time() - started, gpu=torch.cuda.get_device_name(),
                   guard=status, test_read=False, protocol_sha256=sha(HERE / 'PROTOCOL.json'))
    write(out, results); print('KD_RELATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
