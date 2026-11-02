"""R37S critic summary (numbers only; evaluator process, no TEST file is opened). Reads the 18 R37S critic units (OUT/runs/<unit>) on
the single R37S CC152K R30-style H parent, the parent binding and the cache/prep. Adapted from R38 tools/summarize.py (same tables,
restricted to one parent; no reused reference units; no cross-parent pooling; never mixed with R31-parent results).
Writes critic_summary/*.csv and RESULTS.md. Missing units are NA, never filled. Usage: python -I -B tools/summarize_critic.py"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r37s_critic'))
from common import read_json, utc, install_guards, OUT, CONDS

PY = sys.executable
R6 = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
RUNS = OUT / 'runs'; PK = 'cc_R37S_s1001'; COND = 'cc_r37s'; PER, NIMG = 1, 1000
CKS = [0, 250, 500, 1000, 2000, 4000, 8000]
SD = W / 'critic_summary'


def wcsv(name, rows):
    SD.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(SD / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def tbl(rows, cols, fmt='{:.2f}'):
    out = ['|' + '|'.join(cols) + '|', '|' + '|'.join('---' for _ in cols) + '|']
    for r in rows:
        out.append('|' + '|'.join((fmt.format(r[k]) if isinstance(r.get(k), float) else ('NA' if r.get(k) is None else str(r.get(k)))) for k in cols) + '|')
    return '\n'.join(out)


def boot_pair(rka, rkb, per=PER, n_img=NIMG, B=2000, seed=20261102):
    """image-grouped paired bootstrap of rSum(a) - rSum(b) (R38 rule)."""
    rng = np.random.default_rng(seed)
    ia, ta = rka['ranks_i'], rka['ranks_t'].reshape(n_img, per); ib, tb = rkb['ranks_i'], rkb['ranks_t'].reshape(n_img, per)
    def rs(sel):
        f = lambda i, t: sum(100 * (i[sel] < k).mean() for k in (1, 5, 10)) + sum(100 * (t[sel] < k).mean() for k in (1, 5, 10))
        return f(ia, ta) - f(ib, tb)
    full = rs(np.arange(n_img)); bs = np.array([rs(rng.integers(0, n_img, n_img)) for _ in range(B)])
    return dict(diff=float(full), ci2_5=float(np.quantile(bs, .025)), ci97_5=float(np.quantile(bs, .975)), frac_boot_gt0=float((bs > 0).mean()), B=B)


def trans(rka, rkb, k=1):
    out = {}
    for d, key in (('i2t', 'ranks_i'), ('t2i', 'ranks_t')):
        a, b = rka[key] < k, rkb[key] < k
        out[f'{d}_a_only_r{k}'] = int((a & ~b).sum()); out[f'{d}_b_only_r{k}'] = int((~a & b).sum()); out[f'{d}_both_r{k}'] = int((a & b).sum())
        out[f'{d}_neither_r{k}'] = int((~a & ~b).sum()); out[f'{d}_net_r{k}'] = out[f'{d}_a_only_r{k}'] - out[f'{d}_b_only_r{k}']
        if k == 1:
            out[f'{d}_top1_changed'] = int((rka[key.replace('ranks', 'top1')] != rkb[key.replace('ranks', 'top1')]).sum())
    return out


def main():
    install_guards(role='audit')
    U = {}
    for u in read_json(W / 'configs' / 'UNITS.json'):
        c = read_json(W / 'configs' / f'{u}.json')
        U[(c['route'], c['architecture'], c['optimizer']['schedule'], c['objective'])] = dict(unit=u, dir=RUNS / u, block=c['block'])
    tmp = OUT / 'summary'; tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / f'native_ranks_{COND}.npz'
    if not f.exists():
        subprocess.run([PY, '-I', '-B', str(W / 'tools' / 'native_ranks.py'), str(W / 'r37s_critic'), COND, str(f)], check=True)
    natrk = dict(np.load(f)); natm = json.loads(Path(str(f) + '.json').read_text())['metrics']
    pb = read_json(W / 'PARENT_BINDING.json')
    R, RK = {}, {}
    fixed, curves, dirrec, step0, nulls, gdiag, cost, missing, failures = [], [], [], [], [], [], [], [], []
    for key, u in sorted(U.items()):
        route, arch, sch, obj = key
        base = dict(unit=u['unit'], block=u['block'], route=route, architecture=arch, schedule=sch, objective=obj)
        for ff in sorted(u['dir'].glob('FAILED_*.json')):
            x = read_json(ff); failures.append(dict(unit=u['unit'], file=ff.name, utc=x.get('utc'), last_line=x['traceback'].strip().splitlines()[-1][:200]))
        rf = u['dir'] / 'RESULT.json'
        if not rf.exists():
            missing.append(u['unit']); fixed.append(dict(base, status='NA (not complete)')); continue
        r = read_json(rf); R[key] = r; fin = r['final']; dr = fin['dev_retrieval']; H = r['native_H_dev']
        RK[key] = dict(np.load(u['dir'] / 'dev_ranks_step8000.npz'))
        cv = {e['step']: e for e in r['phases'][0]['curves']}
        s0 = cv[0].get('step0_native_check') or read_json(u['dir'] / 'eval_step0.json').get('step0_native_check')
        h38 = r.get('r38') or {}; gc = h38.get('step1_graph_check') or {}
        rs_curve = {s: cv[s]['dev_retrieval']['rsum'] for s in CKS if s in cv}
        fixed.append(dict(base, status='COMPLETE', **{k: dr[k] for k in R6}, rsum=dr['rsum'], H_rsum=H['rsum'], **{f'd_{k}': dr[k] - H[k] for k in R6}, d_rsum=dr['rsum'] - H['rsum'],
                          change_2000_to_8000=rs_curve[8000] - rs_curve[2000], curve_max_rsum_secondary=max(rs_curve.values()), curve_argmax_step=max(rs_curve, key=rs_curve.get),
                          rsum_curve=' / '.join(f'{rs_curve[s]:.2f}' for s in CKS), step0_passed=s0['passed'] if s0 else None, graph_check_passed=gc.get('passed'),
                          lr_last_update=(h38.get('lr_at_readout') or {}).get('8000'), gpu=r['env']['gpu'], job=r['env']['job'],
                          init_shared_keys_sha256=(h38.get('init_state_sha256_shared_keys') or '')[:16], batch_index_sha256=r['phases'][0]['identity']['batch_index_sha256'][:16]))
        for s, e in sorted(cv.items()):
            curves.append(dict(base, step=s, rsum=e['dev_retrieval']['rsum'], d_rsum_vs_H=e['dev_retrieval']['rsum'] - H['rsum'], **{k: e['dev_retrieval'][k] for k in R6},
                               fit_list_observed_J_raw=e['fit_list_observed_J_raw']['J'], dev_list_clean_J_raw=e['dev_list_clean_J_raw']['J'],
                               train_loss_window=e.get('train_window', {}).get('loss_mean'), batch_J_window=e.get('train_window', {}).get('batch_J_mean'),
                               grad_norm_window=e.get('train_window', {}).get('grad_norm_mean'), lr_last_update=(h38.get('lr_at_readout') or {}).get(str(s)),
                               T_sat_frac=e.get('T_saturation_dev_top100', {}).get('frac_absT_ge_0p999'),
                               t2i_pairacc_Hhard10=e['pair_win_rates']['t2i_pairacc_Hhard10'], i2t_pairacc_Hhard10=e['pair_win_rates']['i2t_pairacc_Hhard10']))
        for d in ('i2t', 't2i'):
            dirrec.append(dict(base, direction=d, **{f'r{k}': dr[f'{d}_r{k}'] for k in (1, 5, 10)}, **{f'H_r{k}': H[f'{d}_r{k}'] for k in (1, 5, 10)},
                               **{f'd_r{k}': dr[f'{d}_r{k}'] - H[f'{d}_r{k}'] for k in (1, 5, 10)}, medr=dr[f'{d}_medr'], meanr=dr[f'{d}_meanr'],
                               **{f'{w}_r{k}_vs_H': fin['transitions_vs_H'][f'{d}_{w}_r{k}'] for w in ('rescued', 'lost') for k in (1, 5, 10)},
                               coverage=fin['candidate_coverage'][f'coverage_{d}'], cond_r1_covered=fin['candidate_coverage'][f'cond_r1_{d}_covered']))
        step0.append(dict(base, **({k: s0[k] for k in s0} if s0 else {}), graph_routing=json.dumps(gc.get('routing')) if gc else None,
                          second_update_passed=(gc.get('second_update') or {}).get('passed') if gc else None, graph_passed=gc.get('passed') if gc else None))
        nr = fin.get('fit_null_lists_observed_J_raw')
        if nr:
            nulls.append(dict(base, **{f'{d}_{k}': nr[d][k] for d in ('i2t', 't2i') for k in ('J_total', 'J_real_part', 'J_null_part', 'null_label_fraction')}, null_params=json.dumps(nr['null_params'])))
        for s, dg in sorted((h38.get('dir_grad_diag') or {}).items(), key=lambda x: int(x[0])):
            for p, rec in dg.items():
                gdiag.append(dict(base, step=int(s), peer=p, cos_trunk=rec.get('trunk', {}).get('cos'), cos_all=rec['all']['cos'], norm_i2t_all=rec['all']['norm_i2t'], norm_t2i_all=rec['all']['norm_t2i']))
        cost.append(dict(base, item='unit', gpu=r['env']['gpu'], host=r['env']['host'], job=r['env']['job'], train_seconds=r['cost']['train_seconds'],
                         eval_seconds=r['cost']['eval_seconds'], unit_wall_seconds=r['cost']['unit_wall_seconds_this_process'], params_trainable=r['cost']['params_trainable'],
                         peak_gpu_gb=r['cost']['peak_gpu_memory_bytes'] / 2 ** 30))
    g = lambda key, m: R[key]['final']['dev_retrieval'][m] if key in R else None
    df = lambda a, b: None if a is None or b is None else a - b
    axs, objc, tsch, boots, rtr, vsH = [], [], [], [], [], []
    for key in sorted(R):   # every unit vs this parent's native H (bootstrap + R@1 transitions)
        boots.append(dict(contrast='unit - native H', a=R[key]['unit_id'], b='native H (this parent)', **boot_pair(RK[key], natrk)))
        rtr.append(dict(contrast='unit vs native H', a=R[key]['unit_id'], b='native H', **trans(RK[key], natrk, 1)))
    for obj in ('J', 'BCE', 'NCE'):   # N: head x schedule
        k = {(a, s): ('N', a, s, obj) for a in ('shared', 'directional') for s in ('constant', 'cosine')}
        row = dict(objective=obj, H_rsum=natm['rsum'], H_i2t_r1=natm['i2t_r1'], H_t2i_r1=natm['t2i_r1'])
        for m in ('rsum', 'i2t_r1', 't2i_r1', 'i2t_r5', 't2i_r5'):
            v = {ks: g(kk, m) for ks, kk in k.items()}
            for (a, s), x in v.items():
                row[f'{m}_{a}_{s}'] = x
            row[f'{m}_dir_minus_shared_constant'] = df(v['directional', 'constant'], v['shared', 'constant'])
            row[f'{m}_dir_minus_shared_cosine'] = df(v['directional', 'cosine'], v['shared', 'cosine'])
            row[f'{m}_cos_minus_const_shared'] = df(v['shared', 'cosine'], v['shared', 'constant'])
            row[f'{m}_cos_minus_const_dir'] = df(v['directional', 'cosine'], v['directional', 'constant'])
            row[f'{m}_interaction'] = df(row[f'{m}_cos_minus_const_dir'], row[f'{m}_cos_minus_const_shared'])
        axs.append(row)
        for (a1, s1), (a2, s2), lab in ((('directional', 'constant'), ('shared', 'constant'), 'dir-shared|constant'), (('directional', 'cosine'), ('shared', 'cosine'), 'dir-shared|cosine'),
                                         (('shared', 'cosine'), ('shared', 'constant'), 'cos-const|shared'), (('directional', 'cosine'), ('directional', 'constant'), 'cos-const|dir')):
            ka, kb = k[a1, s1], k[a2, s2]
            if ka in RK and kb in RK:
                boots.append(dict(objective=obj, contrast=f'N {lab}', a=R[ka]['unit_id'], b=R[kb]['unit_id'], **boot_pair(RK[ka], RK[kb])))
                rtr.append(dict(objective=obj, contrast=f'N {lab}', a=R[ka]['unit_id'], b=R[kb]['unit_id'], **trans(RK[ka], RK[kb], 1)))
    for cell in sorted({k[:3] for k in U}):   # objective contrasts within each (route, arch, schedule) cell
        v = {o: cell + (o,) for o in ('J', 'BCE', 'NCE')}
        row = dict(route=cell[0], architecture=cell[1], schedule=cell[2])
        for m in ('rsum', 'i2t_r1', 't2i_r1'):
            for o in ('J', 'BCE', 'NCE'):
                row[f'{m}_{o}'] = g(v[o], m)
            row[f'{m}_J_minus_BCE'] = df(row[f'{m}_J'], row[f'{m}_BCE']); row[f'{m}_J_minus_NCE'] = df(row[f'{m}_J'], row[f'{m}_NCE'])
        for o in ('BCE', 'NCE'):
            if v['J'] in RK and v[o] in RK:
                lab = f'J-{o}|{cell[0]},{cell[1]},{cell[2]}'
                boots.append(dict(objective=f'J-{o}', contrast=lab, a=R[v['J']]['unit_id'], b=R[v[o]]['unit_id'], **boot_pair(RK[v['J']], RK[v[o]])))
                rtr.append(dict(objective=f'J-{o}', contrast=lab, a=R[v['J']]['unit_id'], b=R[v[o]]['unit_id'], **trans(RK[v['J']], RK[v[o]], 1)))
        objc.append(row)
    for obj in ('J', 'BCE', 'NCE'):   # T schedule (+ N at the same schedule)
        kc, kk, nc, nk = ('T', 'shared', 'cosine', obj), ('T', 'shared', 'constant', obj), ('N', 'shared', 'cosine', obj), ('N', 'shared', 'constant', obj)
        row = dict(objective=obj, H_rsum=natm['rsum'], T_constant_rsum=g(kk, 'rsum'), T_cosine_rsum=g(kc, 'rsum'), T_cos_minus_const=df(g(kc, 'rsum'), g(kk, 'rsum')),
                   N_constant_rsum=g(nk, 'rsum'), N_cosine_rsum=g(nc, 'rsum'), T_minus_N_constant=df(g(kk, 'rsum'), g(nk, 'rsum')), T_minus_N_cosine=df(g(kc, 'rsum'), g(nc, 'rsum')))
        for kx, lab in ((kk, 'T_constant'), (kc, 'T_cosine')):
            if kx in R:
                cv = {e['step']: e['dev_retrieval']['rsum'] for e in R[kx]['phases'][0]['curves']}
                row[f'{lab}_curve'] = ' / '.join(f'{cv[s]:.1f}' for s in CKS); row[f'{lab}_max_secondary'] = max(cv.values())
        tsch.append(row)
        if kc in RK and kk in RK:
            boots.append(dict(objective=obj, contrast='T cos-const', a=R[kc]['unit_id'], b=R[kk]['unit_id'], **boot_pair(RK[kc], RK[kk])))
            rtr.append(dict(objective=obj, contrast='T cos-const', a=R[kc]['unit_id'], b=R[kk]['unit_id'], **trans(RK[kc], RK[kk], 1)))
    ident = []
    for route in ('N', 'T'):
        us = [R[k] for k in R if k[0] == route]
        if us:
            sk = {(r_.get('r38') or {}).get('init_state_sha256_shared_keys') for r_ in us}; ch = {r_['phases'][0]['identity']['batch_index_sha256'] for r_ in us}
            ident.append(dict(route=route, n_units=len(us), n_distinct_init_shared_keys=len(sk), n_distinct_batch_chains=len(ch), identical=len(sk) == 1 and len(ch) == 1))
    c = CONDS[COND]; man = read_json(Path(c['cache_dir']) / 'CACHE_MANIFEST.json'); pm = read_json(Path(c['prep_dir']) / 'PREP_MANIFEST.json')
    other = [dict(item='cache (build_cache)', seconds=man['elapsed_seconds'], gpu=man['env']['gpu'], job=man['env']['job']),
             dict(item='prep (locks/calibration/top-32/trust)', seconds=pm['elapsed_seconds'], gpu=pm['env']['gpu'], job=pm['env']['job'])]
    if (W / 'CRITIC_CPU_TEST.json').exists():
        x = read_json(W / 'CRITIC_CPU_TEST.json'); other.append(dict(item='CRITIC_CPU_TEST', seconds=x.get('seconds'), gpu='cpu', job=x.get('job')))
    wcsv('RESULTS_FIXED.csv', fixed); wcsv('CURVES.csv', curves); wcsv('DIRECTIONAL_RECALL.csv', dirrec); wcsv('RANK_TRANSITIONS.csv', rtr)
    wcsv('ARCH_X_SCHEDULE_CONTRASTS.csv', axs); wcsv('OBJECTIVE_CONTRASTS.csv', objc); wcsv('T_SCHEDULE.csv', tsch); wcsv('BOOTSTRAP.csv', boots)
    wcsv('GRAD_DIRECTION_DIAG.csv', gdiag); wcsv('STEP0_CHECK.csv', step0); wcsv('NULL_READOUTS.csv', nulls); wcsv('COST_LEDGER.csv', cost + other); wcsv('IDENTITY_CROSSCHECK.csv', ident)
    wcsv('FAILURES.csv', failures)
    comp = [x for x in fixed if x.get('status') == 'COMPLETE']
    gsum = {}
    for rr in gdiag:
        gsum.setdefault((rr['route'], rr['architecture'], rr['schedule'], rr['objective']), []).append(rr)
    gtab = [dict(route=k[0], architecture=k[1], schedule=k[2], objective=k[3], **{f'cos_trunk_step{s}': float(np.nanmean([x['cos_trunk'] for x in v if x['step'] == s and x['cos_trunk'] is not None] or [np.nan])) for s in (1, 250, 1000, 4000)})
            for k, v in sorted(gsum.items())]
    curv_tab = []
    for x in comp:
        curv_tab.append(dict(unit=x['unit'][len('r37s_cc_r30h_s1001_'):], **{f's{s}': next(e['rsum'] for e in curves if e['unit'] == x['unit'] and e['step'] == s) for s in CKS}))
    tot = sum(c_['unit_wall_seconds'] for c_ in cost)
    pc = pb['cost']; pk_ = pb['cost']['peak_files']
    ptab = [dict(item='warmup (5 epochs, both peers)', value=f"DEV rSum by epoch {' / '.join(f'{v:.1f}' for v in pb['warmup_parent']['warmup_dev_rsum'])}", seconds=pc['warmup_seconds_sum']),
            dict(item='co-train H0 (40 epochs)', value=f"epochs done {pb['epochs_done']}; best epoch {pb['best_cotrain_epoch']}; best DEV rSum {pb['best_dev_rsum']:.1f}", seconds=pc['cotrain_seconds_sum']),
            dict(item='best-epoch GPU / GPU block', value=f"{pb['best_epoch_gpu']} / {pb['gpu_block']['gpu']} (attempts: {', '.join(a['job'] + ' ' + str(a['gpu']) for a in pb['attempts'])})", seconds=None)]
    for n, v in pk_.items():
        ptab.append(dict(item=n, value=f"peak alloc {v.get('peak_mem_allocated_GiB', float('nan')):.2f} GiB, reserved {v.get('peak_mem_reserved_GiB', float('nan')):.2f} GiB, {v.get('gpu')}", seconds=v.get('wall_seconds')))
    rs_ep = pb['dev_rsum_by_epoch']
    md = ['# R37S 结果（仅数字；开发实验，本线未读 TEST）', '',
          f'生成 {utc()}。父模型：CC152K 上按 R30 方案训练的 H（{pb["parent_unit"]}；warmup 5 + H0 co-train 40，A1，K5，i2t/t2i 开，seed 1001，DEV-best）。'
          '与 R37C/R38 的 R31 H0_default 父模型属不同系列，不在同一表中混合；所有差值只相对本父模型。冻结 encoder 的 critic fits；固定主终点 8000 步（曲线最大值只作次要读数）。', '',
          f'未完成单元：{missing if missing else "无"}。完成 {len(comp)}/18（批准 6 项 shared/constant + 用户补充 12 项 R38 对应格）。失败记录：{len(failures)} 条。', '',
          '## 父模型训练', tbl(ptab, ['item', 'value', 'seconds'], '{:.0f}'), '',
          'co-train 每轮 DEV rSum（epoch 0..39）：' + ' / '.join(f'{v:.1f}' for v in rs_ep), '',
          f"cache：原生 DEV 复算 rSum {man['native_dev']['rsum']:.1f}（与 checkpoint best_rsum 完全一致={man['native_dev']['reproduces_checkpoint_best_rsum_exactly']}，GPU {man['env']['gpu']}）；"
          f"prep：a0 {pm['calibration_init'][c['prep_key']]['a0']:.4f}、b0 {pm['calibration_init'][c['prep_key']]['b0']:.4f}；trust 子集 {pm['trust'][c['prep_key']]['n_subset']}（保留率 {pm['trust'][c['prep_key']]['retention']:.4f}）；"
          f"候选：t2i 观测图在 top-32 的比例 {pm['candidates'][c['prep_key']]['t2i_frac_captions_observed_image_in_top32']:.4f}，i2t {pm['candidates'][c['prep_key']]['i2t_frac_images_with_an_observed_caption_in_top32']:.4f}。", '',
          '## 父模型 H（native DEV，按 cache 复算）', tbl([dict(**{k: natm[k] for k in R6}, rsum=natm['rsum'])], R6 + ['rsum']), '',
          '## 全部单元固定终点（8000 步；Δ 相对本父模型 H）',
          tbl(comp, ['unit', 'route', 'architecture', 'schedule', 'objective'] + R6 + ['rsum', 'd_rsum', 'd_i2t_r1', 'd_t2i_r1', 'change_2000_to_8000', 'curve_max_rsum_secondary', 'step0_passed', 'graph_check_passed', 'gpu']), '',
          '## N：结构 × 日程（rSum 与双向 R@1）',
          tbl(axs, ['objective', 'H_rsum', 'rsum_shared_constant', 'rsum_shared_cosine', 'rsum_directional_constant', 'rsum_directional_cosine',
                    'rsum_dir_minus_shared_constant', 'rsum_dir_minus_shared_cosine', 'rsum_cos_minus_const_shared', 'rsum_cos_minus_const_dir', 'rsum_interaction']), '',
          tbl(axs, ['objective', 'H_i2t_r1', 'i2t_r1_shared_constant', 'i2t_r1_shared_cosine', 'i2t_r1_directional_constant', 'i2t_r1_directional_cosine', 'i2t_r1_dir_minus_shared_constant', 'i2t_r1_dir_minus_shared_cosine']), '',
          tbl(axs, ['objective', 'H_t2i_r1', 't2i_r1_shared_constant', 't2i_r1_shared_cosine', 't2i_r1_directional_constant', 't2i_r1_directional_cosine', 't2i_r1_dir_minus_shared_constant', 't2i_r1_dir_minus_shared_cosine']), '',
          '## T 日程（shared）', tbl(tsch, ['objective', 'H_rsum', 'T_constant_rsum', 'T_cosine_rsum', 'T_cos_minus_const', 'N_constant_rsum', 'N_cosine_rsum', 'T_minus_N_constant', 'T_minus_N_cosine', 'T_constant_curve', 'T_cosine_curve']), '',
          '## 目标对比（同路线、同结构、同日程）',
          tbl(objc, ['route', 'architecture', 'schedule', 'rsum_J', 'rsum_BCE', 'rsum_NCE', 'rsum_J_minus_BCE', 'rsum_J_minus_NCE', 'i2t_r1_J_minus_BCE', 't2i_r1_J_minus_BCE', 'i2t_r1_J_minus_NCE', 't2i_r1_J_minus_NCE']), '',
          '## 曲线（DEV rSum，步 0/250/500/1000/2000/4000/8000）', tbl(curv_tab, ['unit'] + [f's{s}' for s in CKS]), '',
          '## 同路线单元的初始化 / 抽样一致性（共享键 init 哈希与批索引哈希链）', tbl(ident, ['route', 'n_units', 'n_distinct_init_shared_keys', 'n_distinct_batch_chains', 'identical']), '',
          '## R@1 转移（a 对 b：仅 a 对 / 仅 b 对 / 净；top1 改变数）',
          tbl(rtr, ['objective', 'contrast', 'a', 'i2t_a_only_r1', 'i2t_b_only_r1', 'i2t_net_r1', 'i2t_top1_changed', 't2i_a_only_r1', 't2i_b_only_r1', 't2i_net_r1', 't2i_top1_changed'], '{:.0f}'), '',
          '## 按图像分组配对 bootstrap（rSum 差，B=2000，描述性）', tbl(boots, ['objective', 'contrast', 'a', 'diff', 'ci2_5', 'ci97_5', 'frac_boot_gt0']), '',
          '## 方向梯度夹角（训练批，trunk 上 i2t 与 t2i 梯度 cos，两 peer 平均）', tbl(gtab, ['route', 'architecture', 'schedule', 'objective', 'cos_trunk_step1', 'cos_trunk_step250', 'cos_trunk_step1000', 'cos_trunk_step4000'], '{:.3f}'), '',
          '## 成本', f'critic 单元 GPU 墙钟合计 {tot:.0f} s（unit_wall_seconds，不含作业启动与 cache 载入）；父模型训练成本见上（PEAK / EPOCHS 秒数）。',
          tbl(cost + other, ['unit', 'item', 'gpu', 'job', 'train_seconds', 'eval_seconds', 'unit_wall_seconds', 'seconds', 'peak_gpu_gb'], '{:.1f}'), '',
          '## 失败', tbl(failures, ['unit', 'file', 'utc', 'last_line']) if failures else '无。', '',
          '文件：critic_summary/ 下 RESULTS_FIXED.csv、CURVES.csv、DIRECTIONAL_RECALL.csv、RANK_TRANSITIONS.csv、ARCH_X_SCHEDULE_CONTRASTS.csv、OBJECTIVE_CONTRASTS.csv、T_SCHEDULE.csv、BOOTSTRAP.csv、GRAD_DIRECTION_DIAG.csv、STEP0_CHECK.csv、NULL_READOUTS.csv、COST_LEDGER.csv、IDENTITY_CROSSCHECK.csv、FAILURES.csv。']
    (W / 'RESULTS.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('R37S_SUMMARY', len(comp), 'complete; missing', missing)


if __name__ == '__main__':
    main()
