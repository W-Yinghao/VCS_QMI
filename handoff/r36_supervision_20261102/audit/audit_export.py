"""R36 EVALUATOR-ONLY candidate-source / false-negative audit (package SERVER_PROMPT item 1, SERVER_PLAN section 6).
Separate process; never imported by any R36 trainer (train_unit.py / train_oracle.py refuse to run if this module
is loaded, and the static gate scan checks that no trainer imports it). Reads the benchmark truth relation
(Flickr30K precomp: caption record k belongs to image k//5) ONLY here. Never opens TEST (R36 guard installed).

For each condition (0%, 60%) and each direction, over ALL 145000 FIT records (and the frozen R35 B-TRUST subset for
60%), exports evaluator NPZs for the package tool tools/audit_candidate_lists.py (run by path with python -I -B):
  forced  = R35 C list: observed positive + first 31 other frozen-H top-32 candidates (data35.build_lists, no permutation)
  native  = R36 N list: unmodified frozen-H top-32 + null slot (33 slots; null native g = 0 = N init, truth False)
and computes, itself (numbers only):
  * frozen-H rank of the observed positive and of the true partner over the full FIT pool (outside top-32/top-100)
  * score-label association inside the lists (native score of inserted vs natural observed positives vs A=0 slots,
    per-row AUC of native g for A>0 vs A=0, fraction of rows where the observed positive is the lowest slot)
  * stored R35 C-G n60 checkpoint scores (step 0..8000 J, step 8000 BCE/NCE, n00 J 8000) on a locked random
    subsample of records (denominator reported): skip/residual decomposition, Spearman(g, native), argmax hub
    concentration; NPZs with checkpoint_g for the package tool (slot-class gradients)
  * DEV: frozen R35 C/D final scores re-ranked by -g (read-only diagnostic; not a method, not used for tuning)
Usage: python -B audit_export.py --stage export|summarize"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'r36'))
from common import (ROOT, OUT, PREP, R34_OUT, PEERS, CONDS, N_TRAIN_CAP, N_TRAIN_IMG, R35_OUT, PACKAGE, install_guards, guard_record,
                    sha256, write_json, read_json, utc)
from data35 import load_cond, build_lists
from heads import build_heads
import evals as EV

AUD = OUT / 'audit'            # evaluator-only directory (truth-bearing NPZs); no trainer reads it
K = 32
SUB_N = 8192                   # locked checkpoint subsample per direction
SUB_SEED = 20261136


def truth_img(rec):
    return np.asarray(rec, np.int64) // 5


@torch.no_grad()
def native_passes(d):
    """Full-pool frozen-H peer-mean fp32 scores (same arithmetic as R35 prep.candidate_tables). Returns ranks and the
    native scores of the forced and native list slots, for every record in both directions."""
    obs = d['obs_np']; ci2t = d['cand_i2t'].astype(np.int64); ct2i = d['cand_t2i'].astype(np.int64)
    rec = np.arange(N_TRAIN_CAP)
    (qi, caps_f, ai_f), (qc, imgs_f, at_f) = build_lists(rec, rec, obs, ci2t, ct2i,
                                                        np.tile(np.arange(K), (N_TRAIN_CAP, 1)), np.tile(np.arange(K), (N_TRAIN_CAP, 1)))
    caps_n = ci2t[obs]; imgs_n = ct2i                      # native lists (i2t query image = obs[r])
    xs = [d[('fit', 'img', p)] for p in PEERS]; ys = [d[('fit', 'cap', p)] for p in PEERS]
    out = dict(caps_f=caps_f, imgs_f=imgs_f, ai_f=ai_f, at_f=at_f, caps_n=caps_n, imgs_n=imgs_n,
               i2t_obs_rank=np.zeros(N_TRAIN_CAP, np.int64), i2t_true_best_rank_img=np.zeros(N_TRAIN_IMG, np.int64),
               t2i_obs_rank=np.zeros(N_TRAIN_CAP, np.int64), t2i_true_rank=np.zeros(N_TRAIN_CAP, np.int64),
               s_i_f=np.zeros((N_TRAIN_CAP, K), np.float32), s_i_n=np.zeros((N_TRAIN_CAP, K), np.float32),
               s_t_f=np.zeros((N_TRAIN_CAP, K), np.float32), s_t_n=np.zeros((N_TRAIN_CAP, K), np.float32))
    order = np.argsort(obs, kind='stable'); obs_sorted = obs[order]
    CH = 250
    for i0 in range(0, N_TRAIN_IMG, CH):
        i1 = min(i0 + CH, N_TRAIN_IMG)
        sc = sum(x[i0:i1] @ y.T for x, y in zip(xs, ys)) / 2                       # [n_img, 145000]
        # best true caption rank per image (truth: captions 5q..5q+4)
        q = torch.arange(i0, i1)
        tr = sc[torch.arange(i1 - i0)[:, None], 5 * q[:, None] + torch.arange(5)[None]]    # [n,5]
        rk = torch.stack([(sc > tr[:, k:k + 1]).sum(1) for k in range(5)], 1)
        out['i2t_true_best_rank_img'][i0:i1] = rk.min(1).values.numpy()
        lo, hi = np.searchsorted(obs_sorted, i0), np.searchsorted(obs_sorted, i1)
        rr = order[lo:hi]
        if len(rr):
            loc = torch.as_tensor(obs[rr] - i0)
            for j0 in range(0, len(rr), 512):
                rj = rr[j0:j0 + 512]; lj = loc[j0:j0 + 512]
                R = sc[lj]                                                            # [m,145000]
                pos = R[torch.arange(len(rj)), torch.as_tensor(rj)]
                out['i2t_obs_rank'][rj] = (R > pos[:, None]).sum(1).numpy()
                out['s_i_f'][rj] = torch.gather(R, 1, torch.as_tensor(caps_f[rj])).numpy()
                out['s_i_n'][rj] = torch.gather(R, 1, torch.as_tensor(caps_n[rj])).numpy()
    CT = 2000
    for j0 in range(0, N_TRAIN_CAP, CT):
        j1 = min(j0 + CT, N_TRAIN_CAP)
        sc = sum(y[j0:j1] @ x.T for x, y in zip(xs, ys)) / 2                        # [n_cap, 29000]
        ar = torch.arange(j1 - j0)
        po = sc[ar, torch.as_tensor(obs[j0:j1])]; pt = sc[ar, torch.as_tensor(np.arange(j0, j1) // 5)]
        out['t2i_obs_rank'][j0:j1] = (sc > po[:, None]).sum(1).numpy()
        out['t2i_true_rank'][j0:j1] = (sc > pt[:, None]).sum(1).numpy()
        out['s_t_f'][j0:j1] = torch.gather(sc, 1, torch.as_tensor(imgs_f[j0:j1])).numpy()
        out['s_t_n'][j0:j1] = torch.gather(sc, 1, torch.as_tensor(imgs_n[j0:j1])).numpy()
    return out


def tool_arrays(P, d, rows, kind, a0, b0, g_ckpt=None):
    """package-tool NPZ arrays for record subset `rows` (both directions stacked: i2t rows then t2i rows)."""
    obs = d['obs_np']
    t = truth_img(rows); mism = obs[rows] != t
    if kind == 'forced':
        caps, imgs = P['caps_f'][rows], P['imgs_f'][rows]
        pi, pt = P['ai_f'][rows].astype(np.float64), P['at_f'][rows].astype(np.float64)
        pi /= pi.sum(1, keepdims=True); pt /= pt.sum(1, keepdims=True)
        ti = (caps // 5) == obs[rows][:, None]                # i2t query image = obs[r]; truth caption c iff c//5 == obs[r]
        tt = imgs == t[:, None]
        si, st = P['s_i_f'][rows], P['s_t_f'][rows]
        # inserted = observed positive record outside the frozen-H top-32 of its query (slot holding caption r / image obs[r])
        in_i = (P['caps_n'][rows] == rows[:, None]).any(1); in_t = (P['imgs_n'][rows] == obs[rows][:, None]).any(1)
        oi = (caps == rows[:, None]) & ~in_i[:, None]
        ot = (imgs == obs[rows][:, None]) & ~in_t[:, None]
        g_i, g_t = a0 * si.astype(np.float64) + b0, a0 * st.astype(np.float64) + b0
    else:
        caps, imgs = P['caps_n'][rows], P['imgs_n'][rows]
        mi = obs[caps] == obs[rows][:, None]; mt = imgs == obs[rows][:, None]
        def tgt(m):
            n = m.sum(1, keepdims=True)
            return np.concatenate([m / np.maximum(n, 1), (n == 0).astype(np.float64)], 1)
        pi, pt = tgt(mi), tgt(mt)
        ti = np.concatenate([(caps // 5) == obs[rows][:, None], np.zeros((len(rows), 1), bool)], 1)
        tt = np.concatenate([imgs == t[:, None], np.zeros((len(rows), 1), bool)], 1)
        z = np.zeros((len(rows), 1))
        g_i = np.concatenate([a0 * P['s_i_n'][rows].astype(np.float64) + b0, z], 1)
        g_t = np.concatenate([a0 * P['s_t_n'][rows].astype(np.float64) + b0, z], 1)
        oi = np.zeros_like(ti); ot = np.zeros_like(tt)
    arr = dict(observed_p=np.concatenate([pi, pt]), truth_mask=np.concatenate([ti, tt]), native_g=np.concatenate([g_i, g_t]),
               outside_native_topk=np.concatenate([oi, ot]), anchor_mismatch=np.concatenate([mism, mism]),
               direction=np.r_[np.zeros(len(rows), np.int64), np.ones(len(rows), np.int64)])
    if g_ckpt is not None:
        arr['checkpoint_g'] = g_ckpt
    return arr


def own_stats(P, d, rows, a0, b0):
    """rank / provenance / score-label association (numbers only), by direction x anchor flag."""
    obs = d['obs_np']; t = truth_img(rows); mism = obs[rows] != t
    res = {}
    ranks = dict(i2t=dict(obs=P['i2t_obs_rank'][rows], true=P['i2t_true_best_rank_img'][obs[rows]]),
                 t2i=dict(obs=P['t2i_obs_rank'][rows], true=P['t2i_true_rank'][rows]))
    for dn in ('i2t', 't2i'):
        if dn == 'i2t':
            caps = P['caps_f'][rows]; A = P['ai_f'][rows] > 0; s = P['s_i_f'][rows].astype(np.float64)
            ins = ~(P['caps_n'][rows] == rows[:, None]).any(1); pos_slot = caps == rows[:, None]
            truth_f = (caps // 5) == obs[rows][:, None]
            An = obs[P['caps_n'][rows]] == obs[rows][:, None]; truth_n = (P['caps_n'][rows] // 5) == obs[rows][:, None]
            sn = P['s_i_n'][rows].astype(np.float64)
        else:
            imgs = P['imgs_f'][rows]; A = P['at_f'][rows] > 0; s = P['s_t_f'][rows].astype(np.float64)
            ins = ~(P['imgs_n'][rows] == obs[rows][:, None]).any(1); pos_slot = imgs == obs[rows][:, None]
            truth_f = imgs == t[:, None]
            An = P['imgs_n'][rows] == obs[rows][:, None]; truth_n = P['imgs_n'][rows] == t[:, None]
            sn = P['s_t_n'][rows].astype(np.float64)
        g = a0 * s + b0
        # per-row AUC of native score: A>0 slots vs A=0 slots (ties 0.5); rows with both classes
        def row_auc(score, lab):
            num = np.zeros(len(lab)); den = np.zeros(len(lab))
            for c0 in range(0, len(lab), 8192):
                sc, lb = score[c0:c0 + 8192], lab[c0:c0 + 8192]
                valid = lb[:, :, None] & ~lb[:, None, :]
                cmp = (sc[:, :, None] > sc[:, None, :]) + .5 * (sc[:, :, None] == sc[:, None, :])
                num[c0:c0 + 8192] = (cmp * valid).sum((1, 2)); den[c0:c0 + 8192] = valid.sum((1, 2))
            ok = den > 0
            return num[ok] / den[ok], ok
        for flag, nm in ((None, 'all'), (False, 'anchor_correct'), (True, 'anchor_mismatch')):
            ix = np.ones(len(rows), bool) if flag is None else (mism == flag)
            n = int(ix.sum())
            if not n:
                continue
            ro, rt = ranks[dn]['obs'][ix], ranks[dn]['true'][ix]
            auc_f, okf = row_auc(g[ix], A[ix])
            auc_n, okn = row_auc(sn[ix], An[ix])
            sl = s[ix]; Ai = A[ix]; insi = ins[ix]; ps = pos_slot[ix]
            inserted_pos = sl[ps & insi[:, None]]; natural_pos = sl[Ai & ~(ps & insi[:, None])]; zero = sl[~Ai]
            lowest = (np.where(ps, sl, np.inf).min(1) <= sl.min(1))
            gr = dict(rows=n,
                      observed_pos_native_rank_ge32_fraction=float((ro >= 32).mean()),
                      observed_pos_native_rank_ge100_fraction=float((ro >= 100).mean()),
                      observed_pos_native_rank_median=float(np.median(ro)),
                      true_partner_best_native_rank_ge32_fraction=float((rt >= 32).mean()),
                      true_partner_best_native_rank_ge100_fraction=float((rt >= 100).mean()),
                      forced_inserted_fraction=float(insi.mean()),
                      forced_true_in_list_fraction=float(truth_f[ix].any(1).mean()),
                      forced_true_slots_mean=float(truth_f[ix].sum(1).mean()),
                      forced_true_but_A0_row_fraction=float((truth_f[ix] & ~Ai).any(1).mean()),
                      forced_true_but_A0_slots_mean=float((truth_f[ix] & ~Ai).sum(1).mean()),
                      forced_observed_pos_slots_mean=float(Ai.sum(1).mean()),
                      forced_wrong_pos_slots_mean=float((Ai & ~truth_f[ix]).sum(1).mean()),
                      forced_observed_pos_is_lowest_native_slot_fraction=float(lowest.mean()),
                      forced_native_s_inserted_pos_mean=float(inserted_pos.mean()) if inserted_pos.size else None,
                      forced_native_s_natural_pos_mean=float(natural_pos.mean()) if natural_pos.size else None,
                      forced_native_s_A0_mean=float(zero.mean()),
                      forced_rowAUC_native_g_Apos_vs_A0_mean=float(auc_f.mean()) if auc_f.size else None,
                      forced_rowAUC_rows=int(okf.sum()),
                      native_null_label_fraction=float((~An[ix].any(1)).mean()),
                      native_observed_pos_slots_mean=float(An[ix].sum(1).mean()),
                      native_true_in_list_fraction=float(truth_n[ix].any(1).mean()),
                      native_true_but_A0_row_fraction=float((truth_n[ix] & ~An[ix]).any(1).mean()),
                      native_true_but_A0_slots_mean=float((truth_n[ix] & ~An[ix]).sum(1).mean()),
                      native_wrong_pos_slots_mean=float((An[ix] & ~truth_n[ix]).sum(1).mean()),
                      native_effective_positive_rows_fraction=float(An[ix].any(1).mean()),
                      native_rowAUC_native_s_Apos_vs_A0_mean=float(auc_n.mean()) if auc_n.size else None,
                      native_rowAUC_rows=int(okn.sum()))
            res[f'{dn}/{nm}'] = gr
    return res


def ckpt_scores(cond, unit, step, d, P, rows):
    """peer-mean C-G checkpoint g on the forced lists of `rows` (i2t rows then t2i rows), plus decomposition."""
    ck = torch.load(R35_OUT / 'runs' / unit / f'ckpt_step{step}.pt', map_location='cpu', weights_only=False)
    a0, b0 = d['a0b0']['a0'], d['a0b0']['b0']
    heads = build_heads('G', a0, b0, 1001); heads.load_state_dict(ck['heads']); heads.eval()
    obs = d['obs_np']
    img = np.concatenate([np.repeat(obs[rows], K), P['imgs_f'][rows].ravel()])
    cap = np.concatenate([P['caps_f'][rows].ravel(), np.repeat(rows, K)])
    g, skip = 0, 0
    with torch.no_grad():
        for p in PEERS:
            X, Y = d[('fit', 'img', p)], d[('fit', 'cap', p)]
            outs, sk = [], []
            for i in range(0, len(img), 65536):
                x, y = X[torch.as_tensor(img[i:i + 65536])], Y[torch.as_tensor(cap[i:i + 65536])]
                s = (x * y).sum(-1)
                sk.append(heads[p].affine(s).double()); outs.append(heads[p].paired(x, y, s).double())
            g = g + torch.cat(outs); skip = skip + torch.cat(sk)
    g = (g / 2).numpy(); skip = (skip / 2).numpy()
    n = len(rows) * K
    G = np.concatenate([g[:n].reshape(-1, K), g[n:].reshape(-1, K)])
    S = np.concatenate([skip[:n].reshape(-1, K), skip[n:].reshape(-1, K)])
    return G, S, dict(path=str(R35_OUT / 'runs' / unit / f'ckpt_step{step}.pt'), step=int(ck['step']),
                      heads_ab={p: [float(t) for t in heads[p].affine.ab()] for p in PEERS})


def spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).astype(np.float64); rb = b.argsort(1).argsort(1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return (ra * rb).sum(1) / np.where(den > 0, den, np.nan)


def export():
    install_guards(role='audit'); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '16')))
    AUD.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); summary = dict(created_utc=utc(), scope='EVAL_ONLY; truth = benchmark caption k -> image k//5; FIT only; TEST not read',
                                    job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename, conds={}, npz={})
    rng = np.random.default_rng(SUB_SEED); sub = np.sort(rng.choice(N_TRAIN_CAP, SUB_N, replace=False))
    for cond in ('n00', 'n60'):
        d = load_cond(cond, torch.device('cpu'))
        a0, b0 = d['a0b0']['a0'], d['a0b0']['b0']
        tc = time.time(); P = native_passes(d)
        cs = dict(native_pass_seconds=time.time() - tc, a0=a0, b0=b0, candidate_table_sha256=d['cand_sha256'],
                  cache_manifest_sha256=d['manifest_sha256'], prep_manifest_sha256=d['prep_manifest_sha256'],
                  observed_map_mismatch_fraction=float((d['obs_np'] != truth_img(np.arange(N_TRAIN_CAP))).mean()))
        # consistency with R35 prep coverage diagnostics (observed map only)
        cs['check_i2t_frac_images_with_observed_caption_in_top32'] = float((d['obs_np'][d['cand_i2t'].astype(np.int64)] == np.arange(N_TRAIN_IMG)[:, None]).any(1).mean())
        cs['check_t2i_frac_obs_in_top32_from_rank'] = float((P['t2i_obs_rank'] < 32).mean())
        cs['check_t2i_frac_obs_in_top32_from_table'] = float((d['cand_t2i'].astype(np.int64) == d['obs_np'][:, None]).any(1).mean())
        pools = {'all': np.arange(N_TRAIN_CAP)}
        if cond == 'n60':
            pools['trust'] = d['trust_subset']
            cs['trust_subset_size'] = int(len(d['trust_subset'])); cs['trust_sha256'] = d['trust_sha256']
            cs['trust_anchor_truth_clean_fraction'] = float((d['obs_np'][d['trust_subset']] == truth_img(d['trust_subset'])).mean())
        cs['own'] = {pn: own_stats(P, d, rows, a0, b0) for pn, rows in pools.items()}
        for pn, rows in pools.items():
            for kind in ('forced', 'native'):
                f = AUD / f'tool_{cond}_{pn}_{kind}.npz'
                np.savez(f, **tool_arrays(P, d, rows, kind, a0, b0)); summary['npz'][f.name] = sha256(f)
        # checkpoint states (C-G, stored R35 checkpoints), locked subsample
        states = []
        if cond == 'n60':
            states = [('r35_C_n60_s1001_G_J', s) for s in (0, 250, 500, 1000, 2000, 4000, 8000)] + [('r35_C_n60_s1001_G_BCE', 8000), ('r35_C_n60_s1001_G_NCE', 8000)]
        else:
            states = [('r35_C_n00_s1001_G_J', 8000), ('r35_C_n00_s1001_G_NCE', 8000)]
        cs['checkpoints'] = {}
        nat_sub = tool_arrays(P, d, sub, 'forced', a0, b0)['native_g']
        for unit, step in states:
            G, S, info = ckpt_scores(cond, unit, step, d, P, sub)
            R = G - S
            sp = spearman_rows(G, nat_sub)
            am = G.argmax(1)
            ids = np.concatenate([P['caps_f'][sub], P['imgs_f'][sub]])[np.arange(len(G)), am]
            dirn = np.r_[np.zeros(SUB_N, int), np.ones(SUB_N, int)]
            hub = {}
            for dd, dn in ((0, 'i2t'), (1, 't2i')):
                c = np.bincount(ids[dirn == dd]); hub[f'{dn}_max_argmax_count'] = int(c.max()); hub[f'{dn}_n_distinct_argmax'] = int((c > 0).sum())
                hub[f'{dn}_top10_share'] = float(np.sort(c)[::-1][:10].sum() / SUB_N)
            cs['checkpoints'][f'{unit}@{step}'] = dict(**info, rows=int(len(G)), g_mean=float(G.mean()), g_std_within_row_mean=float(G.std(1).mean()),
                                                       skip_std_within_row_mean=float(S.std(1).mean()), residual_mean=float(R.mean()),
                                                       residual_std_within_row_mean=float(R.std(1).mean()),
                                                       spearman_g_vs_native_row_mean=float(np.nanmean(sp)), spearman_rows_nan=int(np.isnan(sp).sum()),
                                                       frac_rows_spearman_negative=float(np.nanmean(sp < 0)), **hub)
            f = AUD / f'tool_{cond}_sub_forced_{unit}_step{step}.npz'
            np.savez(f, **tool_arrays(P, d, sub, 'forced', a0, b0, g_ckpt=G)); summary['npz'][f.name] = sha256(f)
        cs['subsample'] = dict(n_per_direction=SUB_N, seed=SUB_SEED, rule='numpy default_rng(seed).choice(145000, n, replace=False), sorted; same records for both directions/conditions')
        summary['conds'][cond] = cs
        del d, P
    # DEV: stored R35 C/D final scores re-ranked by -g (diagnostic readout only)
    dev = {}
    for cond in ('n00', 'n60'):
        nat = np.load(R34_OUT / 'cache' / cond / 'native_dev_sims.npy')
        tab = EV.dev_tables(nat)
        for u in sorted((R35_OUT / 'runs').glob(f'r35_[CD]_{cond}_*')):
            fs = sorted(u.glob('dev_scores_step*.npz'), key=lambda f: int(f.stem.split('step')[1]))
            fs = [f for f in fs if f.stem != 'dev_scores_step0']
            if not fs:
                continue
            z = np.load(fs[-1])
            m, _, _, _ = EV.composite_metrics(nat, tab, z['gi_top'], z['gt_top'])
            mr, _, _, _ = EV.composite_metrics(nat, tab, -z['gi_top'], -z['gt_top'])
            nat_i = np.take_along_axis(nat, tab['i_top'], 1); nat_t = np.take_along_axis(nat.T, tab['t_top'], 1)
            dev[u.name] = dict(scores=fs[-1].name, rsum=m['rsum'], rsum_reverse_sign=mr['rsum'],
                               spearman_g_vs_native_top100_i2t=float(np.nanmean(spearman_rows(z['gi_top'], nat_i))),
                               spearman_g_vs_native_top100_t2i=float(np.nanmean(spearman_rows(z['gt_top'], nat_t))),
                               g_std_within_row_i2t=float(z['gi_top'].std(1).mean()), g_std_within_row_t2i=float(z['gt_top'].std(1).mean()))
    summary['dev_reverse_sign'] = dev
    summary['test_guard'] = guard_record(); summary['elapsed_seconds'] = time.time() - t0
    write_json(AUD / 'AUDIT_EXPORT.json', summary)
    print('R36_AUDIT_EXPORT_DONE', time.time() - t0)


def run_tool(py):
    tool = PACKAGE / 'tools' / 'audit_candidate_lists.py'
    ex = read_json(AUD / 'AUDIT_EXPORT.json')
    res = {}
    for name in sorted(ex['npz']):
        f = AUD / name; o = AUD / (name[:-4] + '.tool.json')
        assert sha256(f) == ex['npz'][name]
        r = subprocess.run([py, '-I', '-B', str(tool), '--input', str(f), '--output', str(o)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        res[name] = read_json(o)
    write_json(AUD / 'AUDIT_TOOL.json', dict(created_utc=utc(), tool=str(tool), tool_sha256=sha256(tool), results=res))
    print('R36_AUDIT_TOOL_DONE', len(res))


def summarize():
    """numbers-only tables into ROOT (CANDIDATE_CONFUSION.csv, PROVENANCE_AUDIT.csv, SCORE_GRAD_AUDIT.csv, AUDIT_RESULTS.json)."""
    ex = read_json(AUD / 'AUDIT_EXPORT.json'); tl = read_json(AUD / 'AUDIT_TOOL.json')['results']
    rows = []
    for cond, cs in ex['conds'].items():
        for pool, gs in cs['own'].items():
            for grp, v in gs.items():
                rows.append(dict(cond=cond, pool=pool, group=grp, **v))
    with open(ROOT / 'PROVENANCE_AUDIT.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    conf = []
    for name, r in tl.items():
        if '_sub_' in name:
            continue
        cond, pool, kind = name[5:-4].split('_')
        for grp, v in r['groups'].items():
            conf.append(dict(cond=cond, pool=pool, list=kind, group=grp, **{k: x for k, x in v.items() if not isinstance(x, dict)}))
    with open(ROOT / 'CANDIDATE_CONFUSION.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(conf[0])); w.writeheader(); w.writerows(conf)
    grad = []
    for name, r in tl.items():
        state = 'native_H' if '_sub_' not in name else name.split('_forced_')[1][:-4]
        cond, pool = name[5:].split('_')[0], name[5:].split('_')[1]
        kind = 'forced' if 'forced' in name else 'native'
        for grp, v in r['groups'].items():
            for k, x in v.items():
                if isinstance(x, dict):
                    score = 'native_g' if k.startswith('native_g') else 'checkpoint_g'
                    rest = k[len(score) + 1:]; obj, cl = rest.split('_', 1)
                    grad.append(dict(cond=cond, pool=pool, list=kind, state=state if score == 'checkpoint_g' else 'native_H', score=score, group=grp,
                                     objective=obj, slot_class=cl, **x))
    with open(ROOT / 'SCORE_GRAD_AUDIT.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(grad[0])); w.writeheader(); w.writerows(grad)
    out = dict(created_utc=utc(), export_job=ex['job'], conds={c: {k: v for k, v in cs.items() if k not in ('own',)} for c, cs in ex['conds'].items()},
               dev_reverse_sign=ex['dev_reverse_sign'], test_guard=ex['test_guard'],
               files=dict(PROVENANCE_AUDIT='PROVENANCE_AUDIT.csv', CANDIDATE_CONFUSION='CANDIDATE_CONFUSION.csv', SCORE_GRAD_AUDIT='SCORE_GRAD_AUDIT.csv'),
               npz_sha256=ex['npz'], tool_sha256=read_json(AUD / 'AUDIT_TOOL.json')['tool_sha256'])
    write_json(ROOT / 'AUDIT_RESULTS.json', out)
    print('R36_AUDIT_SUMMARY_DONE', len(rows), len(conf), len(grad))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['export', 'tool', 'summarize', 'all'])
    a = ap.parse_args()
    if a.stage in ('export', 'all'):
        export()
    if a.stage in ('tool', 'all'):
        run_tool(sys.executable)
    if a.stage in ('summarize', 'all'):
        summarize()
