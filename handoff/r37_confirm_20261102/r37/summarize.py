"""R37 summary (numbers only; evaluator process). Reads runs/<unit>/RESULT.json + eval_step*.json + dev_ranks_step8000.npz
(R37 and the R36 seed1001 T_G/N_G reference units, read-only), parent caches (native DEV sims), prep tables, gate/CPU
records. Replays every unit's sampler on CPU (verified against the training batch-index hash) for participation /
positive mass / repeats; Flickr injected truth (record//5) enters ONLY this evaluator for the truth-correctness columns.
Writes RESULTS_FIXED.csv, CURVES.csv, PAIRED_THREE_PARENT.csv, STRUCTURE_B.csv, NULL_READOUTS.csv, PARTICIPATION.csv,
BOOTSTRAP.csv, COST_LEDGER.csv, RESULTS.md. Missing units are listed as NA, never filled.  Usage: python -B summarize.py"""
from __future__ import annotations
import csv, hashlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, R36_ROOT, R35_PREP, CONDS, N_TRAIN_CAP, N_DEV_IMG, sha256, read_json, write_json, utc, install_guards
import evals as EV
import data36 as D36

R36_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r36_supervision_20261102/runs')
R6 = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
PSEED = {'n60': 1001, 'n60s1002': 1002, 'n60s1003': 1003}
RNAME = {'TRUST_HARD': 'T_G', 'OBS_MISSING_NULL': 'N_G', 'TRUST_NATIVE_NULL': 'TRUST_NATIVE_NULL_G', 'N_PRESENT_ONLY': 'N_PRESENT_ONLY_G'}


def wcsv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(ROOT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def units():
    out = []
    for u in read_json(ROOT / 'configs' / 'UNITS.json'):
        c = read_json(ROOT / 'configs' / f'{u}.json'); out.append(dict(uid=u, cfg=c, dir=RUNS / u, src='R37'))
    for sup, r in (('TRUST_HARD', 'T'), ('OBS_MISSING_NULL', 'N')):
        for o in ('J', 'BCE', 'NCE'):
            u = f'r36_{r}_n60_G_{o}'; out.append(dict(uid=u, cfg=read_json(R36_ROOT / 'configs' / f'{u}.json'), dir=R36_RUNS / u, src='R36 (seed1001 reference, read-only)'))
    return out


def minimal_d(cond):
    c = CONDS[cond]; pdir = Path(c['prep_dir'])
    obs = np.load(c['noise_file']).astype(np.int64)
    cand = np.load(pdir / f"candidates_{c['prep_key']}.npz"); T = np.load(pdir / c['trust_file'])
    return dict(obs_np=obs, cand_i2t=cand['i2t_top32'], cand_t2i=cand['t2i_top32'], trust_subset=T['subset'].astype(np.int64))


def replay(cfg, d, steps):
    """re-draw the unit's training lists (same sampler, same seed) and recompute the batch-index hash chain."""
    sup = cfg['supervision']; rng = np.random.default_rng(cfg['sampler_seed'])
    pool = d['trust_subset'] if sup in D36.TRUST_SUPS else None
    chain = b''; truth = np.arange(N_TRAIN_CAP) // 5
    acc = dict(rows=0, i2t_present=0, t2i_present=0, i2t_posmass_real=0.0, t2i_posmass_real=0.0, i2t_posmass_true=0.0, t2i_posmass_true=0.0,
               i2t_rows_with_true_slot=0, t2i_rows_with_true_slot=0)
    cnt_i = np.zeros(N_TRAIN_CAP, np.int32); cnt_t = np.zeros(N_TRAIN_CAP, np.int32); cnt_vi = np.zeros(N_TRAIN_CAP, np.int32); cnt_vt = np.zeros(N_TRAIN_CAP, np.int32)
    empty_i = empty_t = 0
    for _ in range(steps):
        lists, idx = D36.sample(sup, rng, None, d, cfg['n_queries'], pool)
        (qi, caps, ai), (qc, imgs, at) = lists
        ri, rt = caps != D36.NULL_ID, imgs != D36.NULL_ID
        img = np.concatenate([np.repeat(qi, caps.shape[1]).reshape(caps.shape)[ri], imgs[rt]])
        cap = np.concatenate([caps[ri], np.repeat(qc, imgs.shape[1]).reshape(imgs.shape)[rt]])
        h = hashlib.sha256(chain)
        for x in (img, cap, idx[2], idx[3]):
            h.update(np.ascontiguousarray(x).tobytes())
        chain = h.digest()
        rec_i, rec_t = idx[0], idx[1]
        vi = ((ai > 0) & ri).any(1); vt = ((at > 0) & rt).any(1)
        acc['rows'] += len(qi); acc['i2t_present'] += int(vi.sum()); acc['t2i_present'] += int(vt.sum())
        acc['i2t_posmass_real'] += float((ai * ri).sum()); acc['t2i_posmass_real'] += float((at * rt).sum())
        # evaluator-only truth: i2t slot caption c is true iff c//5 == query image; t2i slot image is true iff == record//5
        ti = ri & (np.where(ri, caps, 0) // 5 == qi[:, None]); tt = rt & (imgs == truth[qc][:, None])
        acc['i2t_posmass_true'] += float((ai * ti).sum()); acc['t2i_posmass_true'] += float((at * tt).sum())
        acc['i2t_rows_with_true_slot'] += int(ti.any(1).sum()); acc['t2i_rows_with_true_slot'] += int(tt.any(1).sum())
        np.add.at(cnt_i, rec_i, 1); np.add.at(cnt_t, rec_t, 1)
        if sup == 'N_PRESENT_ONLY':
            np.add.at(cnt_vi, rec_i[vi], 1); np.add.at(cnt_vt, rec_t[vt], 1); empty_i += int(vi.sum() == 0); empty_t += int(vt.sum() == 0)
    n = acc['rows']
    out = dict(steps=steps, query_rows_per_direction=n,
               i2t_rows_observed_partner_in_list=acc['i2t_present'] / n, t2i_rows_observed_partner_in_list=acc['t2i_present'] / n,
               i2t_positive_mass_on_real_slots=acc['i2t_posmass_real'] / n, t2i_positive_mass_on_real_slots=acc['t2i_posmass_real'] / n,
               EVAL_i2t_positive_mass_on_true_pairs=acc['i2t_posmass_true'] / n, EVAL_t2i_positive_mass_on_true_pairs=acc['t2i_posmass_true'] / n,
               EVAL_i2t_true_share_of_real_positive_mass=acc['i2t_posmass_true'] / max(acc['i2t_posmass_real'], 1e-12),
               EVAL_t2i_true_share_of_real_positive_mass=acc['t2i_posmass_true'] / max(acc['t2i_posmass_real'], 1e-12),
               EVAL_i2t_rows_with_true_partner_in_list=acc['i2t_rows_with_true_slot'] / n, EVAL_t2i_rows_with_true_partner_in_list=acc['t2i_rows_with_true_slot'] / n,
               unique_query_records_i2t=int((cnt_i > 0).sum()), unique_query_records_t2i=int((cnt_t > 0).sum()),
               max_repeats_query_record_i2t=int(cnt_i.max()), max_repeats_query_record_t2i=int(cnt_t.max()),
               mean_repeats_per_used_record_i2t=float(cnt_i[cnt_i > 0].mean()), mean_repeats_per_used_record_t2i=float(cnt_t[cnt_t > 0].mean()))
    if sup == 'N_PRESENT_ONLY':
        out.update(valid_unique_records_i2t=int((cnt_vi > 0).sum()), valid_unique_records_t2i=int((cnt_vt > 0).sum()),
                   valid_max_repeats_i2t=int(cnt_vi.max()), valid_max_repeats_t2i=int(cnt_vt.max()),
                   empty_valid_batches_i2t=empty_i, empty_valid_batches_t2i=empty_t)
    return out, chain.hex()


def boot(rk, rk_nat, B=2000, seed=20261102):
    """image-grouped bootstrap of rSum(unit) - rSum(H): resample DEV images; i2t = the image query, t2i = its 5 captions."""
    rng = np.random.default_rng(seed)
    ri, rn = rk['ranks_i'], rk_nat['ranks_i']; ti = rk['ranks_t'].reshape(N_DEV_IMG, 5); tn = rk_nat['ranks_t'].reshape(N_DEV_IMG, 5)
    def rs(sel):
        a = sum(100 * (ri[sel] < k).mean() for k in (1, 5, 10)) + sum(100 * (ti[sel] < k).mean() for k in (1, 5, 10))
        b = sum(100 * (rn[sel] < k).mean() for k in (1, 5, 10)) + sum(100 * (tn[sel] < k).mean() for k in (1, 5, 10))
        return a - b
    full = rs(np.arange(N_DEV_IMG)); bs = np.array([rs(rng.integers(0, N_DEV_IMG, N_DEV_IMG)) for _ in range(B)])
    return dict(diff=full, boot_mean=float(bs.mean()), ci2_5=float(np.quantile(bs, .025)), ci97_5=float(np.quantile(bs, .975)), frac_boot_gt0=float((bs > 0).mean()), B=B)


def main():
    install_guards(role='audit')
    U = units(); fixed, curves, nulls, part, bt, cost, missing = [], [], [], [], [], [], []
    nat_rk, dcache = {}, {}
    for u in U:
        c = u['cfg']; cond = c['cond']; res_f = u['dir'] / 'RESULT.json'
        base = dict(unit=u['uid'], source=u['src'], parent_seed=PSEED[cond], route=RNAME[c['supervision']], objective=c['objective'])
        if not res_f.exists():
            missing.append(u['uid']); fixed.append(dict(base, status='NA (not complete)')); continue
        r = read_json(res_f); fin = r['final']; dr = fin['dev_retrieval']; H = r['native_H_dev']; tr = fin['transitions_vs_H']
        fixed.append(dict(base, status='COMPLETE', step=fin['step'], **{k: dr[k] for k in R6}, rsum=dr['rsum'], H_rsum=H['rsum'],
                          **{f'd_{k}': dr[k] - H[k] for k in R6}, d_rsum=dr['rsum'] - H['rsum'],
                          i2t_rescued_r1=tr['i2t_rescued_r1'], i2t_lost_r1=tr['i2t_lost_r1'], t2i_rescued_r1=tr['t2i_rescued_r1'], t2i_lost_r1=tr['t2i_lost_r1'],
                          rsum_tie_pess=fin['dev_retrieval_tie_bounded']['rsum_pessimistic'], rsum_tie_opt=fin['dev_retrieval_tie_bounded']['rsum_optimistic'],
                          gpu=r['env']['gpu'], job=r['env']['job']))
        cv = {k['step']: k for k in r['phases'][0]['curves']}
        for s, ev in sorted(cv.items()):
            curves.append(dict(base, step=s, rsum=ev['dev_retrieval']['rsum'], d_rsum_vs_H=ev['dev_retrieval']['rsum'] - H['rsum'],
                               i2t_r1=ev['dev_retrieval']['i2t_r1'], t2i_r1=ev['dev_retrieval']['t2i_r1'],
                               train_loss_window=ev.get('train_window', {}).get('loss_mean'), grad_norm_window=ev.get('train_window', {}).get('grad_norm_mean')))
        fixed[-1]['change_2000_to_8000'] = cv[8000]['dev_retrieval']['rsum'] - cv[2000]['dev_retrieval']['rsum']
        fixed[-1]['rsum_steps_2000_4000_8000'] = ' / '.join(f"{cv[s]['dev_retrieval']['rsum']:.2f}" for s in (2000, 4000, 8000))
        nr = fin.get('fit_null_lists_observed_J_raw')
        if nr:
            nulls.append(dict(base, **{f'{dn}_{k}': nr[dn][k] for dn in ('i2t', 't2i') for k in ('J_total', 'J_real_part', 'J_null_part', 'null_label_fraction', 'null_rank_mean')},
                              J_total_bidirectional=nr['J_total_bidirectional'], null_params=str(nr['null_params'])))
        if cond not in dcache:
            dcache[cond] = minimal_d(cond)
            nat = np.load(Path(CONDS[cond]['cache_dir']) / 'native_dev_sims.npy'); nat_rk[cond] = EV.full_metrics(nat)[2]
        rp, h = replay(c, dcache[cond], r['phases'][0]['steps'])
        part.append(dict(base, replay_batch_hash_matches_training=h == r['phases'][0]['identity']['batch_index_sha256'], **rp,
                         **{f'train_{k}': v for k, v in r['batch_stats']['phase1'].items() if k.startswith(('valid_', 'empty_', 'effective_', 'null_label'))}))
        rk = np.load(u['dir'] / 'dev_ranks_step8000.npz')
        bt.append(dict(base, **boot(rk, nat_rk[cond])))
        cost.append(dict(base, gpu=r['env']['gpu'], host=r['env']['host'], job=r['env']['job'], train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'],
                         unit_wall_seconds=r['cost']['unit_wall_seconds_this_process'], params_trainable=r['cost']['params_trainable'],
                         peak_gpu_gb=r['cost']['peak_gpu_memory_bytes'] / 2 ** 30))
    # paired tables
    idx = {(f['parent_seed'], f['route'], f['objective']): f for f in fixed}
    paired = []
    for route in ('T_G', 'N_G'):
        for o in ('J', 'BCE', 'NCE'):
            ds = {s: idx.get((s, route, o), {}).get('d_rsum') for s in (1001, 1002, 1003)}
            row = dict(route=route, objective=o, **{f'd_rsum_vs_H_s{s}': (v if v is not None else 'NA') for s, v in ds.items()})
            vals = [v for v in ds.values() if v is not None]
            row['mean_3_parents'] = float(np.mean(vals)) if len(vals) == 3 else 'NA'
            row['n_parents_positive'] = sum(v > 0 for v in vals) if len(vals) == 3 else 'NA'
            paired.append(row)
    struct = []
    for o in ('J', 'BCE', 'NCE'):
        g = lambda rt: idx.get((1001, rt, o), {}).get('rsum')
        row = dict(objective=o, H=idx.get((1001, 'N_G', o), {}).get('H_rsum'), T_G=g('T_G'), N_G=g('N_G'), TRUST_NATIVE_NULL_G=g('TRUST_NATIVE_NULL_G'), N_PRESENT_ONLY_G=g('N_PRESENT_ONLY_G'))
        for a, b in (('TRUST_NATIVE_NULL_G', 'N_G'), ('TRUST_NATIVE_NULL_G', 'T_G'), ('N_PRESENT_ONLY_G', 'N_G')):
            row[f'{a}-{b}'] = (row[a] - row[b]) if (row[a] is not None and row[b] is not None) else 'NA'
        struct.append(row)
    # prep / gate costs
    other = []
    for cond in ('n60s1002', 'n60s1003'):
        m = Path(CONDS[cond]['cache_dir']) / 'CACHE_MANIFEST.json'
        if m.exists():
            mm = read_json(m); other.append(dict(item=f'cache {cond}', seconds=mm['elapsed_seconds'], gpu=mm['env']['gpu'], job=mm['env']['job']))
    pm = Path(CONDS['n60s1002']['prep_dir']) / 'PREP_MANIFEST.json'
    if pm.exists():
        p = read_json(pm); other.append(dict(item='prep n60s1002+n60s1003 (calibration, top-32 tables, B-TRUST eval_train)', seconds=p['elapsed_seconds'], gpu=p['env']['gpu'], job=p['env']['job'],
                                             trust_seconds=str({k: v['seconds'] for k, v in p['trust'].items()})))
    for f, nm in (('CPU_TEST.json', 'cpu test'), ('GPU_GATE.json', 'gpu gate')):
        if (ROOT / f).exists():
            g = read_json(ROOT / f); other.append(dict(item=nm, seconds=g.get('elapsed_seconds', g.get('seconds')), gpu=g.get('gpu', 'cpu'), job=g['job']))
    other.append(dict(item='parent H training (not R37 cost; existing R30 runs)', seconds='R30 PEAK wall: s1001 10986, s1002 10979, s1003 13455 (co-train only; + 5-epoch warmup parents)'))
    wcsv('RESULTS_FIXED.csv', fixed); wcsv('CURVES.csv', curves); wcsv('PAIRED_THREE_PARENT.csv', paired); wcsv('STRUCTURE_B.csv', struct)
    wcsv('NULL_READOUTS.csv', nulls); wcsv('PARTICIPATION.csv', part); wcsv('BOOTSTRAP.csv', bt); wcsv('COST_LEDGER.csv', cost + other)

    def tbl(rows, cols, fmt='{:.2f}'):
        out = ['|' + '|'.join(cols) + '|', '|' + '|'.join('---' for _ in cols) + '|']
        for r in rows:
            out.append('|' + '|'.join((fmt.format(r[c]) if isinstance(r.get(c), float) else str(r.get(c, ''))) for c in cols) + '|')
        return '\n'.join(out)
    bind = read_json(ROOT / 'PARENT_BINDING.json')
    md = [f'# R37 结果（仅数字）', '', f'生成 {utc()}；DEV 全查询，两阶段（同父模型 H-top100 重排）；固定终点 8000 步；TEST 未读。seed1001 行为 R36 已完成单元（只读引用）。',
          f'未完成单元：{missing if missing else "无"}。CC152K 6 项未运行：无同系列 R30-H 父模型（见 PARENT_BINDING.json）。', '',
          '## 父模型 H（同父模型基线）', tbl([dict(parent=k, unit=v['parent_unit'], H_rsum=v['best_rsum_logged'], best_epoch=v['best_epoch'], gpu=v['train_gpu']) for k, v in bind['conditions'].items()],
                                          ['parent', 'unit', 'H_rsum', 'best_epoch', 'gpu']), '',
          '## A：相对同父模型 H 的 rSum 差（固定 8000 步）', tbl(paired, list(paired[0])), '',
          '## B：同一 H1001 的结构对照（rSum，8000 步）', tbl(struct, list(struct[0])), '',
          '## 固定终点完整表', tbl([f for f in fixed if f.get('status') == 'COMPLETE'], ['unit', 'parent_seed', 'route', 'objective'] + R6 + ['rsum', 'H_rsum', 'd_rsum', 'change_2000_to_8000', 'rsum_steps_2000_4000_8000',
                                                                                     'i2t_rescued_r1', 'i2t_lost_r1', 't2i_rescued_r1', 't2i_lost_r1']), '',
          '## 按图像分组 bootstrap（同父模型，rSum 差，B=2000，描述性）', tbl(bt, ['unit', 'diff', 'boot_mean', 'ci2_5', 'ci97_5', 'frac_boot_gt0']), '',
          '## N 类：J_real / J_null（FIT 锁定列表，观测标签）', tbl(nulls, ['unit', 'i2t_J_real_part', 'i2t_J_null_part', 't2i_J_real_part', 't2i_J_null_part', 'i2t_null_label_fraction', 't2i_null_label_fraction']), '',
          '## 训练监督：参与比例、正质量、重复（采样重放；EVAL_ 列仅评价器使用注入真值）',
          tbl(part, ['unit', 'replay_batch_hash_matches_training', 'i2t_rows_observed_partner_in_list', 't2i_rows_observed_partner_in_list', 'i2t_positive_mass_on_real_slots',
                     't2i_positive_mass_on_real_slots', 'EVAL_i2t_true_share_of_real_positive_mass', 'EVAL_t2i_true_share_of_real_positive_mass',
                     'unique_query_records_t2i', 'max_repeats_query_record_t2i', 'empty_valid_batches_i2t', 'empty_valid_batches_t2i'], '{:.4f}'), '',
          '## 成本', tbl(cost + other, ['unit', 'item', 'gpu', 'train_seconds', 'eval_seconds', 'seconds'], '{:.1f}'), '',
          '文件：RESULTS_FIXED.csv、CURVES.csv、PAIRED_THREE_PARENT.csv、STRUCTURE_B.csv、NULL_READOUTS.csv、PARTICIPATION.csv、BOOTSTRAP.csv、COST_LEDGER.csv。']
    (ROOT / 'RESULTS.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('R37_SUMMARY', len(fixed) - len(missing), 'complete; missing', missing)


if __name__ == '__main__':
    main()
