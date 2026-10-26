"""DKD + relation-vector term on a fixed FIT reference set (process-local wrapper; the frozen engine is not edited).

r(x) = [cos(h(x), h(a_j))]_{j=1..200}, h = pooled penultimate features, A = 2 FIT samples per class (smallest sha256(id)), the
same rule as check_relation.py. Student reference features are recomputed every step with gradient in TRAIN-mode BN using the reference
batch's own statistics, with all BN running buffers restored afterwards (v2; v1 used eval-mode BN and diverged, see v1 postmortem); teacher reference features are computed once (frozen teacher, eval, no_grad).
Coordinates: u = centered-L2(r_S), v = centered-L2(r_T) over the 200 references. Terms (all weight 10):
  direct : 1 - <u, v>                      (direct relation matching, the non-VCS control)
  vcs    : -J with T = <u,v>, diagonal P, ordered off-diagonal within-batch Q, exact moments (decision.quadratic_u)
  bce    : bounded-score balanced BCE with P(+) = (1+T)/2 (same as the original decision 'bce', on relation coordinates)
P/Q are conditional on the fixed reference set A. Queries that are themselves references keep their (augmented) self-similarity
column; 200 of 44992 samples, disclosed rather than masked so that data order and batches stay identical to the controls.
"""
import hashlib
import torch
import torch.nn.functional as F
from kd import runner as K, decision as D
from kd import losses as L

METHODS = ('direct', 'vcs', 'bce')
STATE = {}
ORIGINAL_CREATE, ORIGINAL_TEACHER, ORIGINAL_FORWARD = K.create_model, K._teacher, L.DistillationLoss.forward


def reference_items(dataset):
    """Positions (dataset items) of the 2 per class FIT samples with the smallest sha256(str(id))."""
    ids = list(dataset.ids); labels = [int(dataset.labels[i]) for i in ids]; best = {}
    for item, (i, y) in enumerate(zip(ids, labels)):
        best.setdefault(y, []).append((hashlib.sha256(str(int(i)).encode()).hexdigest(), item))
    return sorted(item for y in best for _, item in sorted(best[y])[:2])


def coordinates(r):
    return F.normalize(r - r.mean(1, keepdim=True), dim=1, eps=1e-8)


def relation_loss(u, v, method):
    p = (u * v).sum(1)
    if method == 'direct': return (1 - p).mean()
    if method == 'vcs': return -D.quadratic_u(u, v)
    n = len(u); scores = u @ v.T; mask = ~torch.eye(n, dtype=torch.bool, device=u.device)
    return -torch.log(((1 + p) / 2).clamp(1e-6, 1 - 1e-6)).mean() - torch.log(((1 - scores[mask]) / 2).clamp(1e-6, 1 - 1e-6)).mean()


def train_mode_frozen_stats(model, x):
    """v2: reference features in TRAIN-mode BN (normalized by the reference batch's own statistics, like the query features).
    BN momentum is set to 0 for this pass, so the running update r*(1-0) + b*0 leaves running_mean/var bit-for-bit unchanged
    (no in-place restore, which would break autograd); the integer batch counter is restored (not used for gradients)."""
    assert model.training, 'relation term is only used inside training steps'
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [(m.momentum, m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None) for m in bns]
    assert all(mom is not None for mom, _ in saved), 'cumulative-average BN not supported'
    for m in bns: m.momentum = 0.0
    try:
        out = model(x)
    finally:
        for m, (mom, nbt) in zip(bns, saved):
            m.momentum = mom
            if nbt is not None: m.num_batches_tracked.copy_(nbt)
    return out


def create_model(model_cfg, num_classes, seed):
    m = ORIGINAL_CREATE(model_cfg, num_classes, seed)
    STATE.setdefault('models', []).append((model_cfg.get('name'), m))
    return m


def teacher(cfg, manifest, device):
    out = ORIGINAL_TEACHER(cfg, manifest, device); STATE['teacher'] = out[0]; STATE['cfg'] = cfg; return out


def forward(self, logits_s, logits_t, feat_s, feat_t, labels, epoch):
    total, parts = ORIGINAL_FORWARD(self, logits_s, logits_t, feat_s, feat_t, labels, epoch)
    rel = self.cfg.get('relation', {})
    if rel.get('method', 'none') == 'none': return total, parts
    if 'student' not in STATE:  # the model K.train created last under the student name (after any preflight probe)
        STATE['student'] = [m for n, m in STATE['models'] if n == STATE['cfg']['student']['name']][-1]
    student = STATE['student']; device = feat_s.device
    if 'ref_images' not in STATE:
        cfg = STATE['cfg']; ds = K.RoleDataset(cfg['data']['root'], K.read_manifest(cfg['data']['manifest']), 'fit', augment=False, seed=0)
        items = reference_items(ds); assert len(items) == 200
        STATE['ref_items'] = items; STATE['ref_images'] = torch.stack([ds[i][0] for i in items]).to(device)
        with torch.no_grad():
            STATE['teacher'].eval(); STATE['ref_t'] = F.normalize(STATE['teacher'](STATE['ref_images'])[1], dim=1)
    ref_s = F.normalize(train_mode_frozen_stats(student, STATE['ref_images'])[1], dim=1)
    u = coordinates(F.normalize(feat_s, dim=1) @ ref_s.T)
    v = coordinates(F.normalize(feat_t.detach(), dim=1) @ STATE['ref_t'].T)
    aux = relation_loss(u, v, rel['method'])
    total = total + rel['weight'] * aux
    parts['relation'] = float(aux.detach())
    with torch.no_grad():
        self.decision_diagnostics = dict(method='relation_' + rel['method'], relation_positive_cosine=float((u * v).sum(1).mean()),
                                         raw_auxiliary=float(aux), raw_j=float(D.quadratic_u(u, v)), references=200,
                                         q='off-diagonal ordered cross-instance, conditional on the fixed reference set')
    # last_representation_loss stays None: the runner's decision-space probe differentiates w.r.t. logits only.
    return total, parts


def install():
    K.create_model = create_model; K._teacher = teacher; L.DistillationLoss.forward = forward
