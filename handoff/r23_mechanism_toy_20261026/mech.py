"""R23 capacity mechanism toy: a broad imperfect factor S and a sparse discriminative factor R compete for a unit-vector
representation; compare exact-moment VCS, InfoNCE (tau grid) and learnable-logit BCE across descriptor dimension d."""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
K_R, P_INFO, FLIP, E_DIM, NOISE = 1024, 0.15, 0.05, 32, 0.05
OBJECTIVES = ('vcs', 'bce', 'nce03', 'nce07', 'nce15', 'nce30')
TAU = {'nce03': .03, 'nce07': .07, 'nce15': .15, 'nce30': .30}


def write(p, d):
    tmp = Path(str(p) + '.tmp'); tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + '\n'); tmp.replace(p)


# ---------------------------------------------------------------- theory (factor-level, no learning)
def theory(rng=np.random.default_rng(0)):
    def supj(P, Q): return float(((P - Q) ** 2 / (2 * (P + Q))).sum())
    def js(P, Q):
        M = (P + Q) / 2; f = lambda X: float(np.sum(np.where(X > 0, X * np.log(np.where(X > 0, X, 1) / M), 0)))
        return (f(P) + f(Q)) / 2
    def nce(P, px, py, K, n=None):
        n = n or (400000 if K <= 128 else 80000)
        r = P / np.outer(px, py); x, y = np.unravel_index(rng.choice(P.size, size=n, p=P.ravel()), P.shape)
        neg = rng.choice(len(py), size=(n, K - 1), p=py); num = r[x, y]
        return float(np.mean(np.log(num / ((num + r[x[:, None], neg].sum(1)) / K))))
    PS = np.array([[1 - FLIP, FLIP], [FLIP, 1 - FLIP]]) / 2; pS = np.array([.5, .5])
    p = np.r_[1 - P_INFO, np.full(K_R, P_INFO / K_R)]; PR = np.diag(p)
    PJ = np.kron(PS, PR); pJ = np.kron(pS, p)
    pI = np.array([1 - P_INFO, P_INFO]); PI = np.diag(pI)            # R coarsened to 'informative or not' (shared)
    PSI = np.kron(PS, PI); pSI = np.kron(pS, pI)
    out = {}
    for name, P, px in (('S', PS, pS), ('R', PR, p), ('joint', PJ, pJ), ('R_indicator', PI, pI), ('S_plus_R_indicator', PSI, pSI)):
        Q = np.outer(px, px); out[name] = dict(supJ=supj(P, Q), JS=js(P, Q), InfoNCE128=nce(P, px, px, 128), InfoNCE512=nce(P, px, px, 512))
    return out


# ---------------------------------------------------------------- synthetic world
class World:
    def __init__(self, seed):
        g = torch.Generator().manual_seed(10_000 + seed)
        unit = lambda n: F.normalize(torch.randn(n, E_DIM, generator=g), dim=1)
        self.ES = [unit(2), unit(2)]; self.ER = [unit(K_R + 1), unit(K_R + 1)]

    def make(self, sA, sB, r, g):
        xa = torch.cat([self.ES[0][sA], self.ER[0][r]], 1); xb = torch.cat([self.ES[1][sB], self.ER[1][r]], 1)
        return xa + NOISE * torch.randn(xa.shape, generator=g), xb + NOISE * torch.randn(xb.shape, generator=g)

    def sample(self, n, g, informative_only=False):
        sA = torch.randint(0, 2, (n,), generator=g); sB = sA ^ (torch.rand(n, generator=g) < FLIP).long()
        if informative_only: r = torch.randint(1, K_R + 1, (n,), generator=g)
        else: r = torch.where(torch.rand(n, generator=g) < P_INFO, torch.randint(1, K_R + 1, (n,), generator=g), torch.zeros(n, dtype=torch.long))
        xa, xb = self.make(sA, sB, r, g)
        return xa, xb, sA, sB, r

    def r_probe(self, per_class, g):
        r = torch.arange(1, K_R + 1).repeat_interleave(per_class); n = len(r)
        sA = torch.randint(0, 2, (n,), generator=g); sB = sA ^ (torch.rand(n, generator=g) < FLIP).long()
        xa, xb = self.make(sA, sB, r, g); return xa, xb, r


def tower(d):
    return nn.Sequential(nn.Linear(2 * E_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d))


class Model(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fa, self.fb = tower(d), tower(d)
        self.a = nn.Parameter(torch.ones(())); self.b = nn.Parameter(torch.zeros(()))   # used by bce only

    def forward(self, xa, xb):
        return F.normalize(self.fa(xa), dim=1), F.normalize(self.fb(xb), dim=1)


def objective(name, m, u, v):
    """Returns (loss, own value): VCS J_hat, InfoNCE log B - loss, BCE JS_hat."""
    n = len(u); p = (u * v).sum(1)
    if name == 'vcs':
        j = (p - .5 * p ** 2).mean() - u.mean(0) @ v.mean(0) - .5 * ((u.T @ u / n) * (v.T @ v / n)).sum()
        return -j, j
    s = u @ v.T
    if name == 'bce':
        z = m.a * s + m.b; off = ~torch.eye(n, dtype=torch.bool)
        loss = F.softplus(-z.diagonal()).mean() + F.softplus(z[off]).mean()
        return loss, (math.log(4) - loss) / 2
    t = TAU[name]; lab = torch.arange(n)
    loss = .5 * (F.cross_entropy(s / t, lab) + F.cross_entropy(s.T / t, lab))
    return loss, math.log(n) - loss


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def embed(m, xa, xb):
    m.eval(); u, v = m(xa, xb); m.train(); return u, v


def logistic_bacc(ztr, ytr, zte, yte):
    w = torch.zeros(ztr.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(ztr @ w + b, ytr.float()) + 1e-4 * w.square().sum(); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        pred = (zte @ w + b) > 0
        return float(.5 * ((pred[yte == 1]).float().mean() + (~pred[yte == 0]).float().mean()))


def ncm_acc(zmean_src, rtr, zq, rq):
    means = torch.zeros(K_R + 1, zmean_src.shape[1]).index_add_(0, rtr, zmean_src)[1:]
    means = F.normalize(means, dim=1)
    return float(((zq @ means.T).argmax(1) + 1 == rq).float().mean())


def evaluate(world, m, name, batch, seed):
    g = torch.Generator().manual_seed(2_000_000 + seed)
    vals, top1 = [], []
    for _ in range(50):
        xa, xb, *_ = world.sample(batch, g); u, v = embed(m, xa, xb)
        vals.append(float(objective(name, m, u, v)[1])); top1.append(float(((u @ v.T).argmax(1) == torch.arange(batch)).float().mean()))
    xa, xb, sA, sB, r1 = world.sample(8192, g); ua, vb = embed(m, xa, xb)
    xa2, xb2, sA2, sB2, r2 = world.sample(8192, g); ua2, vb2 = embed(m, xa2, xb2)
    s_a = logistic_bacc(ua, sA, ua2, sA2); s_b = logistic_bacc(vb, sB, vb2, sB2)
    i_a = logistic_bacc(ua, (r1 > 0).long(), ua2, (r2 > 0).long())   # coarse 'R informative or not' indicator
    xa, xb, rtr = world.r_probe(16, g); ua, vb = embed(m, xa, xb)
    xq, xqb, _, _, rq = world.sample(8192, g, informative_only=True); uq, vq = embed(m, xq, xqb)
    r_a = ncm_acc(ua, rtr, uq, rq); r_cross = ncm_acc(vb, rtr, uq, rq)
    norm_s = lambda x: (x - .5) / .5; norm_r = lambda x: (x - 1 / K_R) / (1 - 1 / K_R)
    return dict(own_value=float(np.mean(vals)), retrieval_top1=float(np.mean(top1)), S_bacc_A=s_a, S_bacc_B=s_b,
                R_ncm_A=r_a, R_ncm_cross=r_cross, I_bacc_A=i_a, S_ret=norm_s(.5 * (s_a + s_b)), R_ret=norm_r(r_a), I_ret=norm_s(i_a))


def raw_check(seed=1):
    world = World(seed); g = torch.Generator().manual_seed(7)
    xa, xb, sA, sB, _ = world.sample(8192, g); xa2, xb2, sA2, sB2, _ = world.sample(8192, g)
    s = logistic_bacc(xa, sA, xa2, sA2)
    xr, _, rtr = world.r_probe(16, g); xq, _, _, _, rq = world.sample(8192, g, informative_only=True)
    return dict(S_bacc_raw_A=s, R_ncm_raw_A=ncm_acc(F.normalize(xr, dim=1), rtr, F.normalize(xq, dim=1), rq))


def train(name, d, batch, seed, lr, steps, checkpoints=()):
    torch.manual_seed(seed); world = World(seed); m = Model(d)
    opt = torch.optim.Adam(m.parameters(), lr=lr); g = torch.Generator().manual_seed(1_000_000 + seed); curve = []
    for step in range(1, steps + 1):
        xa, xb, *_ = world.sample(batch, g); u, v = m(xa, xb); loss, _ = objective(name, m, u, v)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        assert math.isfinite(float(loss.detach()))
        if step in checkpoints: curve.append(dict(step=step, **evaluate(world, m, name, batch, seed)))
    return m, world, curve


def sanity():
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))); t0 = time.time(); runs = {}
    cps = tuple(range(1000, 6001, 1000))
    for lr in (3e-4, 1e-3, 3e-3):
        for name in ('vcs', 'nce07', 'bce'):
            _, _, curve = train(name, 16, 128, 1, lr, 6000, cps); runs[f'{name}_lr{lr:g}'] = curve
            print('sanity', name, lr, [round(c['own_value'], 4) for c in curve], flush=True)
    # pre-registered rule (PROTOCOL.json budget)
    wins = {lr: 0 for lr in (3e-4, 1e-3, 3e-3)}
    for name in ('vcs', 'nce07', 'bce'):
        best = max(wins, key=lambda lr: (runs[f'{name}_lr{lr:g}'][-1]['own_value'], -lr)); wins[best] += 1
    lr = min((lr for lr in wins if wins[lr] == max(wins.values())))
    steps = 6000
    for cp in cps:
        if all(runs[f'{n}_lr{lr:g}'][cps.index(cp)]['own_value'] >= .98 * runs[f'{n}_lr{lr:g}'][-1]['own_value'] for n in ('vcs', 'nce07', 'bce')):
            steps = cp; break
    write(HERE / 'SANITY.json', dict(status='COMPLETE', job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - t0, raw_input_check=raw_check(),
                                     runs=runs, lr_wins={f'{k:g}': v for k, v in wins.items()}, chosen_lr=lr, chosen_steps=steps,
                                     rule='PROTOCOL.json budget rule, applied mechanically'))
    print('SANITY chosen lr', lr, 'steps', steps, flush=True)


def main():
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))); t0 = time.time()
    s = json.loads((HERE / 'SANITY.json').read_text()); lr, steps = s['chosen_lr'], s['chosen_steps']
    out = HERE / 'runs.jsonl'; done = set()
    if out.exists(): done = {(r['objective'], r['d'], r['batch'], r['seed']) for r in map(json.loads, out.read_text().splitlines())}
    cells = [(n, d, 128, sd) for d in (2, 4, 8, 16, 32, 64) for n in OBJECTIVES for sd in (1, 2, 3)] + [(n, 16, 512, sd) for n in OBJECTIVES for sd in (1, 2, 3)]
    for n, d, b, sd in cells:
        if (n, d, b, sd) in done: continue
        t1 = time.time(); m, world, _ = train(n, d, b, sd, lr, steps)
        row = dict(objective=n, d=d, batch=b, seed=sd, lr=lr, steps=steps, seconds=time.time() - t1, **evaluate(world, m, n, b, sd))
        if n == 'bce': row.update(a=float(m.a), b_bias=float(m.b))
        with out.open('a') as f: f.write(json.dumps(row) + '\n')
        print(n, d, b, sd, {k: round(row[k], 4) for k in ('own_value', 'S_ret', 'R_ret', 'retrieval_top1')}, flush=True)
    print('MAIN done', time.time() - t0, flush=True)


def analyze():
    rows = [json.loads(l) for l in (HERE / 'runs.jsonl').read_text().splitlines()]; th = theory()
    mean = lambda xs: float(np.mean(xs))
    def cell(n, d, b, key): return [r[key] for r in sorted(rows, key=lambda r: r['seed']) if r['objective'] == n and r['d'] == d and r['batch'] == b]
    table = {}
    for b, ds in ((128, (2, 4, 8, 16, 32, 64)), (512, (16,))):
        for d in ds:
            ent = {n: {k: mean(cell(n, d, b, k)) for k in ('S_ret', 'R_ret', 'I_ret', 'own_value', 'retrieval_top1', 'S_bacc_A', 'R_ncm_A', 'R_ncm_cross')} for n in OBJECTIVES}
            best = max(TAU, key=lambda n: ent[n]['retrieval_top1'])
            diff = {k: [x - y for x, y in zip(cell('vcs', d, b, k), cell(best, d, b, k))] for k in ('S_ret', 'R_ret', 'I_ret')}
            table[f'b{b}_d{d}'] = dict(means=ent, best_nce_by_retrieval=best,
                                       vcs_minus_best_nce={k: dict(per_seed=v, mean=mean(v)) for k, v in diff.items()},
                                       vcs_minus_bce={k: mean([x - y for x, y in zip(cell('vcs', d, b, k), cell('bce', d, b, k))]) for k in ('S_ret', 'R_ret', 'I_ret')})
    sanity = json.loads((HERE / 'SANITY.json').read_text())
    write(HERE / 'RESULT.json', dict(status='COMPLETE', theory=th, raw_input_check=sanity['raw_input_check'], lr=sanity['chosen_lr'],
                                     steps=sanity['chosen_steps'], n_runs=len(rows), table=table, test_read=False))
    for k, v in table.items():
        print(k, 'bestNCE', v['best_nce_by_retrieval'], {n: (round(v['means'][n]['S_ret'], 3), round(v['means'][n]['R_ret'], 3)) for n in OBJECTIVES},
              'VCS-bestNCE S/R', round(v['vcs_minus_best_nce']['S_ret']['mean'], 3), round(v['vcs_minus_best_nce']['R_ret']['mean'], 3))
    print(json.dumps(th, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('theory', 'sanity', 'main', 'analyze')); a = p.parse_args()
    {'theory': lambda: print(json.dumps(theory(), indent=1)), 'sanity': sanity, 'main': main, 'analyze': analyze}[a.mode]()
