"""R37C summary (numbers only; evaluator process). Reads runs/<unit>/RESULT.json + eval_step*.json + dev_ranks_step8000.npz,
the parent cache (native DEV sims) and prep tables. Replays every unit's sampler on CPU (verified against the training
batch-index hash) for participation / positive mass / repeats. CC152K has no injected truth: no truth column exists.
All gains are relative to the same CC152K parent (R31 H0_default cc_s1001); no cross-dataset absolute rSum comparison.
Writes RESULTS_FIXED.csv, CURVES.csv, NULL_READOUTS.csv, PARTICIPATION.csv, BOOTSTRAP.csv, COST_LEDGER.csv, RESULTS.md.
Missing units are listed as NA, never filled.  Usage: python -B summarize.py"""
from __future__ import annotations
import csv, hashlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, CONDS, N_TRAIN_CAP, N_DEV_IMG, PER_CAP, read_json, utc, install_guards
import evals as EV
import data36 as D36

R6 = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
RNAME = {'TRUST_HARD': 'T_G', 'OBS_MISSING_NULL': 'N_G'}


def wcsv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(ROOT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def minimal_d():
    c = CONDS['cc']; pdir = Path(c['prep_dir'])
    obs = np.load(Path(c['cache_dir']) / 'observed_map.npy').astype(np.int64)
    cand = np.load(pdir / f"candidates_{c['prep_key']}.npz"); T = np.load(pdir / c['trust_file'])
    return dict(obs_np=obs, cand_i2t=cand['i2t_top32'], cand_t2i=cand['t2i_top32'], trust_subset=T['subset'].astype(np.int64))


def replay(cfg, d, steps):
    sup = cfg['supervision']; rng = np.random.default_rng(cfg['sampler_seed'])
    pool = d['trust_subset'] if sup in D36.TRUST_SUPS else None
    chain = b''
    acc = dict(rows=0, i2t_present=0, t2i_present=0, i2t_posmass_real=0.0, t2i_posmass_real=0.0, i2t_null_mass=0.0, t2i_null_mass=0.0)
    cnt_i = np.zeros(N_TRAIN_CAP, np.int32); cnt_t = np.zeros(N_TRAIN_CAP, np.int32)
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
        vi = ((ai > 0) & ri).any(1); vt = ((at > 0) & rt).any(1)
        acc['rows'] += len(qi); acc['i2t_present'] += int(vi.sum()); acc['t2i_present'] += int(vt.sum())
        acc['i2t_posmass_real'] += float((ai * ri).sum()); acc['t2i_posmass_real'] += float((at * rt).sum())
        acc['i2t_null_mass'] += float((ai * ~ri).sum()); acc['t2i_null_mass'] += float((at * ~rt).sum())
        np.add.at(cnt_i, idx[0], 1); np.add.at(cnt_t, idx[1], 1)
    n = acc['rows']
    return dict(steps=steps, query_rows_per_direction=n,
                i2t_rows_observed_partner_in_list=acc['i2t_present'] / n, t2i_rows_observed_partner_in_list=acc['t2i_present'] / n,
                i2t_positive_mass_on_real_slots=acc['i2t_posmass_real'] / n, t2i_positive_mass_on_real_slots=acc['t2i_posmass_real'] / n,
                i2t_positive_mass_on_null=acc['i2t_null_mass'] / n, t2i_positive_mass_on_null=acc['t2i_null_mass'] / n,
                unique_query_records_i2t=int((cnt_i > 0).sum()), unique_query_records_t2i=int((cnt_t > 0).sum()),
                max_repeats_query_record_i2t=int(cnt_i.max()), max_repeats_query_record_t2i=int(cnt_t.max()),
                mean_repeats_per_used_record_i2t=float(cnt_i[cnt_i > 0].mean()), mean_repeats_per_used_record_t2i=float(cnt_t[cnt_t > 0].mean())), chain.hex()


def boot(rk, rk_nat, B=2000, seed=20261102):
    """image-grouped bootstrap of rSum(unit) - rSum(H): resample DEV images; i2t = the image query, t2i = its caption (per 1)."""
    rng = np.random.default_rng(seed)
    ri, rn = rk['ranks_i'], rk_nat['ranks_i']; ti = rk['ranks_t'].reshape(N_DEV_IMG, PER_CAP); tn = rk_nat['ranks_t'].reshape(N_DEV_IMG, PER_CAP)
    def rs(sel):
        a = sum(100 * (ri[sel] < k).mean() for k in (1, 5, 10)) + sum(100 * (ti[sel] < k).mean() for k in (1, 5, 10))
        b = sum(100 * (rn[sel] < k).mean() for k in (1, 5, 10)) + sum(100 * (tn[sel] < k).mean() for k in (1, 5, 10))
        return a - b
    full = rs(np.arange(N_DEV_IMG)); bs = np.array([rs(rng.integers(0, N_DEV_IMG, N_DEV_IMG)) for _ in range(B)])
    return dict(diff=full, boot_mean=float(bs.mean()), ci2_5=float(np.quantile(bs, .025)), ci97_5=float(np.quantile(bs, .975)), frac_boot_gt0=float((bs > 0).mean()), B=B)


def main():
    install_guards(role='audit')
    uids = read_json(ROOT / 'configs' / 'UNITS.json')
    fixed, curves, nulls, part, bt, cost, missing = [], [], [], [], [], [], []
    c = CONDS['cc']; nat = np.load(Path(c['cache_dir']) / 'native_dev_sims.npy'); nat_m, _, nat_rk = EV.full_metrics(nat)
    d = minimal_d()
    for u in uids:
        cfg = read_json(ROOT / 'configs' / f'{u}.json'); udir = RUNS / u; res_f = udir / 'RESULT.json'
        base = dict(unit=u, route=RNAME[cfg['supervision']], objective=cfg['objective'])
        if not res_f.exists():
            missing.append(u); fixed.append(dict(base, status='NA (not complete)')); continue
        r = read_json(res_f); fin = r['final']; dr = fin['dev_retrieval']; H = r['native_H_dev']; tr = fin['transitions_vs_H']
        s0 = read_json(udir / 'eval_step0.json')['step0_native_check']
        row = dict(base, status='COMPLETE', step=fin['step'], **{k: dr[k] for k in R6}, rsum=dr['rsum'], H_rsum=H['rsum'],
                   **{f'd_{k}': dr[k] - H[k] for k in R6}, d_rsum=dr['rsum'] - H['rsum'],
                   **{f'{dn}_{w}_r{k}': tr[f'{dn}_{w}_r{k}'] for dn in ('i2t', 't2i') for w in ('rescued', 'lost') for k in (1, 5, 10)},
                   rsum_tie_pess=fin['dev_retrieval_tie_bounded']['rsum_pessimistic'], rsum_tie_opt=fin['dev_retrieval_tie_bounded']['rsum_optimistic'],
                   step0_rank_diff_i2t=s0['n_i2t_rank_diff'], step0_rank_diff_t2i=s0['n_t2i_rank_diff'], step0_passed=s0['passed'],
                   coverage_i2t=fin['candidate_coverage']['coverage_i2t'], coverage_t2i=fin['candidate_coverage']['coverage_t2i'],
                   gpu=r['env']['gpu'], job=r['env']['job'])
        cv = {k['step']: k for k in r['phases'][0]['curves']}
        row['change_2000_to_8000'] = cv[8000]['dev_retrieval']['rsum'] - cv[2000]['dev_retrieval']['rsum']
        row['rsum_steps_0_2000_4000_8000'] = ' / '.join(f"{cv[s]['dev_retrieval']['rsum']:.2f}" for s in (0, 2000, 4000, 8000))
        fixed.append(row)
        for s, ev in sorted(cv.items()):
            curves.append(dict(base, step=s, rsum=ev['dev_retrieval']['rsum'], d_rsum_vs_H=ev['dev_retrieval']['rsum'] - H['rsum'],
                               **{k: ev['dev_retrieval'][k] for k in R6},
                               fit_list_observed_J_raw=ev['fit_list_observed_J_raw']['J'], dev_list_J_raw=ev['dev_list_clean_J_raw']['J'],
                               train_loss_window=ev.get('train_window', {}).get('loss_mean'), grad_norm_window=ev.get('train_window', {}).get('grad_norm_mean')))
        nr = fin.get('fit_null_lists_observed_J_raw')
        if nr:
            nulls.append(dict(base, **{f'{dn}_{k}': nr[dn][k] for dn in ('i2t', 't2i') for k in ('J_total', 'J_real_part', 'J_null_part', 'null_label_fraction', 'null_rank_mean', 'observed_pos_real_slots_mean')},
                              J_total_bidirectional=nr['J_total_bidirectional'], null_params=str(nr['null_params'])))
        rp, h = replay(cfg, d, r['phases'][0]['steps'])
        part.append(dict(base, replay_batch_hash_matches_training=h == r['phases'][0]['identity']['batch_index_sha256'], **rp,
                         presence_p_i2t=r['present_probs']['p_i2t'], presence_p_t2i=r['present_probs']['p_t2i'],
                         **{f'train_{k}': v for k, v in r['batch_stats']['phase1'].items() if k.startswith(('effective_', 'null_label', 'observed_pos'))}))
        rk = np.load(udir / 'dev_ranks_step8000.npz')
        bt.append(dict(base, **boot(rk, nat_rk)))
        cost.append(dict(base, gpu=r['env']['gpu'], host=r['env']['host'], job=r['env']['job'], train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'],
                         unit_wall_seconds=r['cost']['unit_wall_seconds_this_process'], params_trainable=r['cost']['params_trainable'],
                         peak_gpu_gb=r['cost']['peak_gpu_memory_bytes'] / 2 ** 30))
    other = []
    m = read_json(Path(c['cache_dir']) / 'CACHE_MANIFEST.json')
    other.append(dict(item='cache cc (native DEV reproduction + FIT encoding)', seconds=m['elapsed_seconds'], gpu=m['env']['gpu'], job=m['env']['job']))
    p = read_json(Path(c['prep_dir']) / 'PREP_MANIFEST.json')
    other.append(dict(item='prep cc (locks, calibration, top-32 tables, B-TRUST eval_train)', seconds=p['elapsed_seconds'], gpu=p['env']['gpu'], job=p['env']['job']))
    if (ROOT / 'CPU_TEST.json').exists():
        g = read_json(ROOT / 'CPU_TEST.json'); other.append(dict(item='cpu test', seconds=g.get('seconds'), gpu='cpu', job=g['job']))
    other.append(dict(item='parent H training (not R37C cost; existing R31 run cc_s1001/H0_default, 1+40 epochs)', seconds='see R31 RUN/EPOCHS'))
    wcsv('RESULTS_FIXED.csv', fixed); wcsv('CURVES.csv', curves); wcsv('NULL_READOUTS.csv', nulls); wcsv('PARTICIPATION.csv', part)
    wcsv('BOOTSTRAP.csv', bt); wcsv('COST_LEDGER.csv', cost + other)

    def tbl(rows, cols, fmt='{:.2f}'):
        out = ['|' + '|'.join(cols) + '|', '|' + '|'.join('---' for _ in cols) + '|']
        for r_ in rows:
            out.append('|' + '|'.join((fmt.format(r_[k]) if isinstance(r_.get(k), float) else str(r_.get(k, ''))) for k in cols) + '|')
        return '\n'.join(out)
    bind = read_json(ROOT / 'PARENT_BINDING.json')['conditions']['cc']
    pc = p['calibration_init']['cc']; tr_ = p['trust']['cc']; ca = p['candidates']['cc']
    comp = [f for f in fixed if f.get('status') == 'COMPLETE']
    md = ['# R37C 结果（CC152K，仅数字；开发实验，本线未读 TEST）', '',
          f'生成 {utc()}；DEV 1000 图 × 1000 文（每图 1 文），全查询，两阶段（同父模型 H-top100 重排）；固定终点 8000 步。与 Flickr 分开报告：只报相对同一父模型的增益，不跨数据集比较绝对 rSum。',
          f'未完成单元：{missing if missing else "无"}。', '',
          '## 父模型 H（R31 IN²R 原生，作者默认 warmup 1 / K 1 / i2t,t2i 关；与 Flickr R30-H 不同系列）',
          tbl([dict(unit=bind['parent_unit'], best_rsum_logged=bind['best_rsum_logged'], best_epoch=bind['best_epoch'], native_rsum_reproduced=m['native_dev']['rsum'],
                    exact=m['native_dev']['reproduces_checkpoint_best_rsum_exactly'], gpu=m['env']['gpu'], **{k: nat_m[k] for k in R6})],
              ['unit', 'best_rsum_logged', 'best_epoch', 'native_rsum_reproduced', 'exact', 'gpu'] + R6), '',
          '## 重建的 CC152K 监督构件（观测关系 = 原生恒等，记录 k ↔ 图 k ↔ 文 k）',
          tbl([dict(a0=pc['a0'], b0=pc['b0'], trust_n=tr_['n_subset'], trust_retention=tr_['retention'], i2t_obs_in_top32=ca['i2t_frac_images_with_an_observed_caption_in_top32'],
                    t2i_obs_in_top32=ca['t2i_frac_captions_observed_image_in_top32'], i2t_obs_rank1=ca['i2t_frac_observed_partner_at_rank1'], t2i_obs_rank1=ca['t2i_frac_observed_partner_at_rank1'])],
              ['a0', 'b0', 'trust_n', 'trust_retention', 'i2t_obs_in_top32', 't2i_obs_in_top32', 'i2t_obs_rank1', 't2i_obs_rank1'], '{:.4f}'), '',
          '## 固定终点（8000 步），相对同父模型 H',
          tbl(comp, ['unit', 'route', 'objective'] + R6 + ['rsum', 'H_rsum', 'd_rsum', 'rsum_steps_0_2000_4000_8000', 'step0_passed']), '',
          '## 逐方向 R@K 差（单元 − H）', tbl(comp, ['unit'] + [f'd_{k}' for k in R6] + ['d_rsum']), '',
          '## 救回 / 丢失（相对 H，R@1/5/10）',
          tbl(comp, ['unit'] + [f'{dn}_{w}_r{k}' for dn in ('i2t', 't2i') for k in (1, 5, 10) for w in ('rescued', 'lost')], '{:.0f}'), '',
          '## 按图像分组 bootstrap（rSum 差，B=2000，描述性）', tbl(bt, ['unit', 'diff', 'boot_mean', 'ci2_5', 'ci97_5', 'frac_boot_gt0']), '',
          '## N 类：J_real / J_null（FIT 锁定列表，观测标签）',
          tbl(nulls, ['unit', 'i2t_J_real_part', 'i2t_J_null_part', 't2i_J_real_part', 't2i_J_null_part', 'i2t_null_label_fraction', 't2i_null_label_fraction'], '{:.4f}'), '',
          '## 训练监督：参与比例、正质量、重复（采样重放，与训练批哈希核对；CC152K 无注入真值，无真值列）',
          tbl(part, ['unit', 'replay_batch_hash_matches_training', 'i2t_rows_observed_partner_in_list', 't2i_rows_observed_partner_in_list', 'i2t_positive_mass_on_real_slots',
                     't2i_positive_mass_on_real_slots', 'i2t_positive_mass_on_null', 't2i_positive_mass_on_null', 'unique_query_records_t2i', 'max_repeats_query_record_t2i'], '{:.4f}'), '',
          '## 成本', tbl(cost + other, ['unit', 'item', 'gpu', 'job', 'train_seconds', 'eval_seconds', 'unit_wall_seconds', 'seconds', 'peak_gpu_gb'], '{:.1f}'), '',
          '文件：RESULTS_FIXED.csv、CURVES.csv、NULL_READOUTS.csv、PARTICIPATION.csv、BOOTSTRAP.csv、COST_LEDGER.csv。']
    (ROOT / 'RESULTS.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('R37C_SUMMARY', len(comp), 'complete; missing', missing)


if __name__ == '__main__':
    main()
