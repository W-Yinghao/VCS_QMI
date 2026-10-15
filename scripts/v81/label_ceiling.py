"""V8.1 supplementary diagnostic (main session, 2026-10-15): how high can within-prompt pair accuracy go against 3- or 6-vote labels?

Question (V81_03 section 4 left it open): the eight V8 arms reach a pooled within-prompt pair accuracy of about 0.70 on DEV and do not
improve on held-out roles after the first epoch.  Is that close to the ceiling that noisy 3-vote human means allow, or is there room?

Three independent estimates of the pair accuracy an oracle that knows each image's true expected rating would reach against the
observed vote means (informative pairs only, exact ties excluded, oracle ties count 0.5 -- the same scoring as V81_03):
  A. Gaussian attenuation: reliability R = 1 - E[noise var of d] / E[d^2] over within-prompt pairs, noise var from each image's unbiased
     vote variance / n; ceiling = 1/2 + asin(sqrt(R)) / pi (sign agreement of a bivariate normal).  Continuous approximation.
  B. Empirical-Bayes simulation: each image's true vote histogram = (counts + alpha * prompt-pooled histogram) / (n + alpha), alpha chosen
     so that the implied within-prompt variance of observed means equals the observed one; votes re-drawn with the same n; the oracle is
     the true histogram's mean.  Reported with alpha / 2 and 2 alpha as sensitivity.
  C. Split-half on the 6-vote images: each 6-vote image's votes split at random into two halves of 3; agreement of the two halves'
     orderings over pairs informative in both halves gives the 3-vote reliability r = sin(pi (agreement - 1/2)); implied oracle ceiling
     against 3-vote labels = 1/2 + asin(sqrt(r)) / pi, against 6-vote labels via Spearman-Brown r6 = 2r / (1 + r).

Model pair accuracy is recomputed on exactly the same pair sets from the normalised V8 rows (fork 2,
outputs_v81/normalized/combined/<role>_v8fit.jsonl), with exact rational comparisons of the human means.

Data: development roles of the EvalMuse public TRAIN split only (V7 manifest), the labels V81_02-04 already use.  No test role, no
GenAI / TIFA160 label, no new label source.  CPU only.  A development diagnostic: estimates rest on the stated noise models and are
not a pre-registered quantity.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MANIFEST = Path('/home/infres/yinwang/CS_QMI/outputs/V7/data/manifest_images.jsonl')
NORM = Path('/projects/EEG-foundation-model/yinghao/outputs_v81/normalized/combined')
ROLES = {'DEV': 'dev_v8fit.jsonl', 'CAL_SELECT': 'cal_select_v8fit.jsonl', 'CAL_PROB': 'cal_prob_v8fit.jsonl'}
LEVELS = np.arange(1, 6, dtype=float)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_role(role: str):
    imgs = []
    for line in open(MANIFEST, encoding='utf-8'):
        r = json.loads(line)
        if r.get('role') == role and r.get('overall_votes'):
            v = [int(x) for x in r['overall_votes']]
            imgs.append({'image_id': str(r['image_id']), 'prompt_id': str(r['prompt_id']), 'votes': v})
    imgs.sort(key=lambda r: (r['prompt_id'], r['image_id']))
    n = np.array([len(r['votes']) for r in imgs], dtype=np.int64)
    S = np.array([sum(r['votes']) for r in imgs], dtype=np.int64)
    counts = np.zeros((len(imgs), 5))
    for k, r in enumerate(imgs):
        for x in r['votes']:
            counts[k, x - 1] += 1
    s2 = np.array([np.var(r['votes'], ddof=1) for r in imgs])
    pid = np.array([r['prompt_id'] for r in imgs])
    I, J = [], []
    start = 0
    for p in sorted(set(pid)):
        idx = np.flatnonzero(pid == p)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                I.append(idx[a]); J.append(idx[b])
        start += len(idx)
    return imgs, n, S, counts, s2, pid, np.array(I), np.array(J)


def exact_sign(S, n, I, J):
    """sign(m_i - m_j) with m = S / n, compared exactly by cross-multiplication."""
    return np.sign(S[I] * n[J] - S[J] * n[I])


def pa_from_signs(pred_sign, label_sign):
    ok = label_sign != 0
    if not ok.any():
        return None, 0
    hit = np.where(pred_sign[ok] == 0, 0.5, (pred_sign[ok] == label_sign[ok]).astype(float))
    return float(hit.mean()), int(ok.sum())


def gaussian_ceiling(S, n, s2, I, J, mask=None):
    if mask is not None:
        I, J = I[mask], J[mask]
    m = S / n
    d = m[I] - m[J]
    v = s2[I] / n[I] + s2[J] / n[J]
    R = 1.0 - v.mean() / (d * d).mean()
    return {'reliability_R': float(R), 'mean_d2': float((d * d).mean()), 'mean_noise_var': float(v.mean()),
            'ceiling_pair_accuracy': float(0.5 + math.asin(math.sqrt(max(R, 0.0))) / math.pi), 'n_pairs_all': int(len(I))}


def within_prompt_var(x, pid):
    out = []
    for p in np.unique(pid):
        xs = x[pid == p]
        if len(xs) > 1:
            out.append(xs.var(ddof=1))
    return float(np.mean(out))


def eb_hist(counts, n, pid, alpha):
    pooled = {}
    for p in np.unique(pid):
        c = counts[pid == p].sum(0)
        pooled[p] = c / c.sum()
    prior = np.stack([pooled[p] for p in pid])
    return (counts + alpha * prior) / (n[:, None] + alpha)


def implied_var(hist, n, pid):
    mu = hist @ LEVELS
    var_i = hist @ (LEVELS ** 2) - mu ** 2
    # within-prompt variance of the simulated observed means = spread of true means + mean sampling noise
    return within_prompt_var(mu, pid) + float(np.mean(var_i / n))


def calibrate_alpha(counts, n, S, pid):
    target = within_prompt_var(S / n, pid)
    lo, hi = math.log(1e-3), math.log(1e3)
    f = lambda a: implied_var(eb_hist(counts, n, pid, math.exp(a)), n, pid) - target
    grid = np.linspace(lo, hi, 61)
    vals = [f(a) for a in grid]
    sign_change = [k for k in range(60) if vals[k] * vals[k + 1] <= 0]
    if not sign_change:
        return None, target, vals
    k = sign_change[0]
    a, b = grid[k], grid[k + 1]
    for _ in range(60):
        c = 0.5 * (a + b)
        if f(a) * f(c) <= 0:
            b = c
        else:
            a = c
    return math.exp(0.5 * (a + b)), target, vals


def simulate(hist, n, I, J, reps, seed):
    rng = np.random.default_rng(seed)
    mu = hist @ LEVELS
    osign = np.sign(mu[I] - mu[J])
    out = []
    for _ in range(reps):
        Ssim = (rng.multinomial(n, hist) @ np.arange(1, 6)).astype(np.int64)
        pa, _ = pa_from_signs(osign, exact_sign(Ssim, n, I, J))
        out.append(pa)
    out = np.array(out)
    return {'mean': float(out.mean()), 'p2.5': float(np.quantile(out, .025)), 'p97.5': float(np.quantile(out, .975)), 'reps': reps}


def split_half(imgs, n, pid, I, J, reps, seed):
    six = n >= 6
    sel = six[I] & six[J]
    Ii, Jj = I[sel], J[sel]
    rng = np.random.default_rng(seed)
    agr, used = [], []
    votes = [np.array(r['votes']) for r in imgs]
    for _ in range(reps):
        S1 = np.zeros(len(imgs), dtype=np.int64); S2 = np.zeros(len(imgs), dtype=np.int64)
        for k in np.flatnonzero(six):
            perm = rng.permutation(votes[k])
            S1[k] = perm[:3].sum(); S2[k] = perm[3:6].sum()
        a = np.sign(S1[Ii] - S1[Jj]); b = np.sign(S2[Ii] - S2[Jj])
        ok = (a != 0) & (b != 0)
        agr.append(float((a[ok] == b[ok]).mean())); used.append(int(ok.sum()))
    agree = float(np.mean(agr))
    r = math.sin(math.pi * (agree - 0.5))
    r6 = 2 * r / (1 + r) if r > -1 else float('nan')
    return {'n_pairs_both_6_votes': int(sel.sum()), 'mean_pairs_informative_in_both_halves': float(np.mean(used)),
            'half_vs_half_agreement': agree, 'half_vs_half_agreement_p2.5': float(np.quantile(agr, .025)),
            'half_vs_half_agreement_p97.5': float(np.quantile(agr, .975)), 'reliability_3vote_r': r,
            'implied_oracle_ceiling_vs_3vote': float(0.5 + math.asin(math.sqrt(max(r, 0))) / math.pi),
            'reliability_6vote_spearman_brown': r6,
            'implied_oracle_ceiling_vs_6vote': float(0.5 + math.asin(math.sqrt(max(r6, 0))) / math.pi), 'reps': reps,
            'caveat': '6-vote images may be a non-random subset (e.g. extra votes for contested images); used as a check, not the main estimate'}


def validate_gaussian(hist, n, I, J, mask, reps, seed):
    """Estimator A applied to synthetic votes drawn i.i.d. from `hist` (3-vote pairs), next to the oracle PA known in that simulation."""
    rng = np.random.default_rng(seed + 2)
    mu = hist @ LEVELS
    Im, Jm = I[mask], J[mask]
    osign = np.sign(mu[Im] - mu[Jm])
    est, true = [], []
    for _ in range(min(reps, 50)):
        c = rng.multinomial(n, hist)
        Ssim = (c @ np.arange(1, 6)).astype(np.int64)
        ex2 = (c @ (np.arange(1, 6) ** 2)).astype(float)
        s2sim = (ex2 - Ssim ** 2 / n) / np.maximum(n - 1, 1)
        est.append(gaussian_ceiling(Ssim, n, s2sim, Im, Jm)['ceiling_pair_accuracy'])
        true.append(pa_from_signs(osign, exact_sign(Ssim, n, Im, Jm))[0])
    return {'estimator_A_mean': float(np.mean(est)), 'oracle_pa_mean': float(np.mean(true)), 'bias_A_minus_oracle': float(np.mean(est) - np.mean(true)),
            'reps': min(reps, 50), 'model': 'votes i.i.d. from the empirical-Bayes histograms (alpha calibrated), 3-vote pairs'}


def split_half_synthetic(hist, n, pid, I, J, reps, seed):
    """Same estimator on synthetic 6-vote data drawn i.i.d. from `hist`: exchangeable votes must give agreement >= 0.5."""
    rng = np.random.default_rng(seed + 1)
    six = n >= 6
    fake = []
    for k in range(len(n)):
        c = rng.multinomial(int(n[k]), hist[k]) if six[k] else np.zeros(5, dtype=int)
        fake.append({'votes': [lev for lev, cnt in zip(range(1, 6), c) for _ in range(int(cnt))] if six[k] else [3, 3, 3]})
    r = split_half(fake, n, pid, I, J, reps, seed)
    return {k: r[k] for k in ('half_vs_half_agreement', 'half_vs_half_agreement_p2.5', 'half_vs_half_agreement_p97.5', 'implied_oracle_ceiling_vs_3vote')}


def model_pa(role_file: Path, imgs, n, S, I, J):
    idx = {r['image_id']: k for k, r in enumerate(imgs)}
    scores = collections.defaultdict(lambda: np.full(len(imgs), np.nan))
    for line in open(role_file, encoding='utf-8'):
        r = json.loads(line)
        key = (r['method'], int(r['seed']), r['checkpoint_id'])
        scores[key][idx[r['image_id']]] = float(r['score'])
    lab = exact_sign(S, n, I, J)
    six = (n[I] >= 6) & (n[J] >= 6)
    three = (n[I] == 3) & (n[J] == 3)
    rows = []
    for (meth, seed, ck), s in sorted(scores.items()):
        if np.isnan(s).any():
            raise SystemExit(f'missing predictions for {meth} s{seed}')
        ps = np.sign(s[I] - s[J])
        pa_all, n_all = pa_from_signs(ps, lab)
        pa6, n6 = pa_from_signs(ps[six], lab[six])
        pa3, n3 = pa_from_signs(ps[three], lab[three])
        rows.append({'method': meth, 'seed': seed, 'checkpoint_id': ck, 'pair_accuracy_all': pa_all, 'n_informative_all': n_all,
                     'pair_accuracy_both6': pa6, 'n_informative_both6': n6, 'pair_accuracy_both3': pa3, 'n_informative_both3': n3})
    arm = collections.defaultdict(list)
    for r in rows:
        arm[r['method']].append(r)
    arm_mean = {m: {k: float(np.mean([r[k] for r in rs])) for k in ('pair_accuracy_all', 'pair_accuracy_both6', 'pair_accuracy_both3')}
                for m, rs in sorted(arm.items())}
    return rows, arm_mean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=2718)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    res = {'script': 'scripts/v81/label_ceiling.py', 'status': 'development_diagnostic_not_preregistered',
           'inputs': {'manifest': str(MANIFEST), 'manifest_sha256': sha256(MANIFEST)}, 'reps': a.reps, 'seed': a.seed, 'roles': {}}
    for role, fname in ROLES.items():
        imgs, n, S, counts, s2, pid, I, J = load_role(role)
        lab = exact_sign(S, n, I, J)
        alpha, target, _ = calibrate_alpha(counts, n, S, pid)
        sims = {}
        if alpha is not None:
            for tag, al in (('alpha', alpha), ('alpha_half', alpha / 2), ('alpha_double', alpha * 2)):
                sims[tag] = {'alpha': al, **simulate(eb_hist(counts, n, pid, al), n, I, J, a.reps, a.seed)}
        rows, arm_mean = model_pa(NORM / fname, imgs, n, S, I, J)
        res['inputs'][f'{role}_rows'] = str(NORM / fname)
        res['inputs'][f'{role}_rows_sha256'] = sha256(NORM / fname)
        res['roles'][role] = {
            'n_images': len(imgs), 'n_prompts': int(len(np.unique(pid))), 'votes_per_image': {str(k): int(v) for k, v in collections.Counter(n.tolist()).items()},
            'n_pairs_all': int(len(I)), 'n_pairs_informative': int((lab != 0).sum()),
            'A_gaussian': gaussian_ceiling(S, n, s2, I, J),
            'A_gaussian_both3': gaussian_ceiling(S, n, s2, I, J, (n[I] == 3) & (n[J] == 3)),
            'A_gaussian_both6': gaussian_ceiling(S, n, s2, I, J, (n[I] >= 6) & (n[J] >= 6)),
            'B_note': 'B uses smoothed plug-in histograms, whose vote variance underestimates the sampling variance (factor (n-1)/n): it overstates the true spread and so the ceiling; A (unbiased s^2) is the main estimate',
            'B_empirical_bayes': {'observed_within_prompt_var_of_means': target, **sims} if sims else {'status': 'alpha not identified'},
            'C_split_half_6vote': split_half(imgs, n, pid, I, J, a.reps, a.seed),
            'A_validation_on_synthetic_3vote': validate_gaussian(eb_hist(counts, n, pid, alpha), n, I, J, (n[I] == 3) & (n[J] == 3), a.reps, a.seed) if alpha is not None else None,
            'C_synthetic_iid_check': split_half_synthetic(eb_hist(counts, n, pid, alpha), n, pid, I, J, a.reps, a.seed) if alpha is not None else None,
            'model_pair_accuracy_rows': rows, 'model_pair_accuracy_arm_mean': arm_mean}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    for role, r in res['roles'].items():
        B = r['B_empirical_bayes']
        print(f"{role}: pairs {r['n_pairs_informative']} informative | A {r['A_gaussian']['ceiling_pair_accuracy']:.3f} (R {r['A_gaussian']['reliability_R']:.3f})"
              f" | B {B.get('alpha', {}).get('mean', float('nan')):.3f} [alpha {B.get('alpha', {}).get('alpha', float('nan')):.2f}; half {B.get('alpha_half', {}).get('mean', float('nan')):.3f}, double {B.get('alpha_double', {}).get('mean', float('nan')):.3f}]"
              f" | A3 {r['A_gaussian_both3']['ceiling_pair_accuracy']:.3f} A6 {r['A_gaussian_both6']['ceiling_pair_accuracy']:.3f}"
              f" | A-validation est {r['A_validation_on_synthetic_3vote']['estimator_A_mean']:.3f} vs oracle {r['A_validation_on_synthetic_3vote']['oracle_pa_mean']:.3f}"
              f" | C agree real {r['C_split_half_6vote']['half_vs_half_agreement']:.3f} vs synthetic-iid {r['C_synthetic_iid_check']['half_vs_half_agreement']:.3f}"
              f" | U-CS PA all {r['model_pair_accuracy_arm_mean']['U-CS']['pair_accuracy_all']:.3f}, both6 {r['model_pair_accuracy_arm_mean']['U-CS']['pair_accuracy_both6']:.3f}, both3 {r['model_pair_accuracy_arm_mean']['U-CS']['pair_accuracy_both3']:.3f}")


if __name__ == '__main__':
    main()
