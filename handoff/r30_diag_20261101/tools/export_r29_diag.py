#!/usr/bin/env python3
"""R30 plan §6.1: export the records R29 already logged (read-only; no recomputation from states).

For every R29 unit (runs/<cell>/<arm>/) copies verbatim:
  R29_WARMUP_DIAG.json (whole file), EPOCHS.json warmup DEV records, every co-train epoch record
  (lr, DEV, is_best, split_A/split_B = GMM split quality written by r25_co_train.split_quality, train.A/train.B = all logged
  per-peer statistics), RUN_<job>.json env (gpu/host/job), PEAK_*.json, DONE.json, R29_ATTEMPTS.jsonl.
A field the original code did not log for a unit/epoch is written as "NA". Nothing is reconstructed from final states.
Outputs: R29_DIAG_EXPORT.json (complete), R29_DIAG_EXPORT_EPOCHS.csv (flat, one row per unit x co-train epoch x peer),
R29_DIAG_EXPORT.md (numbers-only tables).
Usage: export_r29_diag.py RUNS_ROOT OUT_DIR [R29_QUEUE_SAVED_STATES.json]
"""
import csv, glob, hashlib, json, os, sys

ORDER_CELLS = ('f00_s1001', 'f60_s1001', 'f60_s1002', 'f60_s1003')
ORDER_ARMS = ('h_native', 'n_infonce', 'p_pointwise_vcs', 'l_listwise_vcs', 'b_listwise_bce')
NA = 'NA'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load(p):
    return json.load(open(p))


def unit_record(d):
    files = {}
    rec = {}
    for name in ('R29_WARMUP_DIAG.json', 'EPOCHS.json', 'DONE.json', 'R29_IDENTITY.json'):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            files[name] = sha(p)
    rec['warmup_diag'] = load(os.path.join(d, 'R29_WARMUP_DIAG.json')) if 'R29_WARMUP_DIAG.json' in files else NA
    h = load(os.path.join(d, 'EPOCHS.json'))
    rec['warmup_epochs'] = h['warmup']
    rec['cotrain_epochs'] = h['cotrain']
    rec['epochs_extra_keys'] = {k: v for k, v in h.items() if k not in ('warmup', 'cotrain')}
    rec['done'] = load(os.path.join(d, 'DONE.json')) if 'DONE.json' in files else NA
    rec['run_env'] = [{'file': os.path.basename(p), **{k: load(p)['env'].get(k, NA) for k in ('gpu', 'host', 'job', 'torch', 'cuda', 'cudnn', 'dtype')}}
                      for p in sorted(glob.glob(os.path.join(d, 'RUN_*.json')))]
    rec['peak'] = [load(p) for p in sorted(glob.glob(os.path.join(d, 'PEAK_*.json')))]
    ap = os.path.join(d, 'R29_ATTEMPTS.jsonl')
    rec['attempts'] = [json.loads(l) for l in open(ap)] if os.path.isfile(ap) else NA
    rec['source_files_sha256'] = files
    return rec


# ---------------------------------------------------------------------------------------------------- markdown helpers

COUNT_KEYS = {'steps', 'diag_n', 'skip_labeled_lt2', 'i2t_n', 't2i_n', 'i2t_no_proto', 't2i_no_proto', 'n_pred_clean'}


def f(x, nd=4, count=False):
    if x is None or x == NA:
        return NA
    if count and isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, bool):
        return '1' if x else '0'
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:
            return 'nan'
        if abs(x) >= 1000:
            return f'{x:.1f}'
        return f'{x:.{nd}f}'
    return str(x)


def g(dct, k):
    if not isinstance(dct, dict):
        return NA
    return dct.get(k, NA)


WARM_COLS = [('steps', '更新步数'), ('diag_n', '诊断批数'), ('align_loss', '主项loss'), ('gnorm_align', '主项梯度范数'),
             ('Jlist_i2t', 'J列表 i2t'), ('Jlist_t2i', 'J列表 t2i'), ('top1pos_i2t', '正例top1 i2t'), ('top1pos_t2i', '正例top1 t2i'),
             ('pimass_i2t', '正例质量 i2t'), ('pimass_t2i', '正例质量 t2i'), ('Jpoint_i2t', 'J点对 i2t'), ('Jpoint_t2i', 'J点对 t2i'),
             ('multi_pos_rows', '多正例行比例'), ('r29_b_end', 'P偏置b')]
TRAIN_COLS = [('steps', '实际更新数'), ('skip_labeled_lt2', '跳过B<2'), ('r29_align_loss_mean', '主项loss'),
              ('r29_gnorm_align_mean', '主项梯度范数'), ('gnorm_enc_clean_mean', 'clean梯度范数'), ('gnorm_enc_noisy_mean', 'noisy梯度范数'),
              ('gnorm_refiner_mean', 'refiner梯度范数'), ('r29_Jlist_i2t_mean', 'J列表 i2t'), ('r29_Jlist_t2i_mean', 'J列表 t2i'),
              ('r29_top1pos_i2t_mean', '正例top1 i2t'), ('r29_top1pos_t2i_mean', '正例top1 t2i'), ('r29_pimass_i2t_mean', '正例质量 i2t'),
              ('r29_pimass_t2i_mean', '正例质量 t2i'), ('r29_Jpoint_i2t_mean', 'J点对 i2t'), ('r29_Jpoint_t2i_mean', 'J点对 t2i'),
              ('r29_b_end', 'P偏置b'), ('i2t_n', 'noisy i2t步'), ('t2i_n', 'noisy t2i步'), ('i2t_no_proto', 'i2t无候选'), ('t2i_no_proto', 't2i无候选'),
              ('i2t_raw_loss_mean', 'noisy i2t loss'), ('t2i_raw_loss_mean', 'noisy t2i loss'), ('clean_loss_mean', 'clean总loss'),
              ('i2t_usable_frac_mean', '对方队列ID可用比例 i2t'), ('t2i_usable_frac_mean', '对方队列ID可用比例 t2i'), ('train_seconds', '该peer训练秒')]
MD_EPOCHS = (0, 1, 2, 5, 10, 14, 15, 20, 25, 30, 35, 39)


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    runs, out_dir = sys.argv[1], sys.argv[2]
    qpath = sys.argv[3] if len(sys.argv) > 3 else None
    units = {}
    for c in ORDER_CELLS:
        for a in ORDER_ARMS:
            d = os.path.join(runs, c, a)
            if os.path.isdir(d):
                units[f'{c}__{a}'] = unit_record(d)
    queue = json.load(open(qpath)) if qpath and os.path.isfile(qpath) else None
    for uid, u in units.items():
        u['queue_at_saved_states'] = [r for r in queue['rows'] if r['unit'] == uid] if queue else NA
        wall = [p.get('wall_seconds') for p in u['peak']] if u['peak'] else []
        u['cost'] = {'gpu_wall_seconds_recorded': sum(w for w in wall if w is not None) if wall else NA,
                     'gpu_hours_recorded': (sum(w for w in wall if w is not None) / 3600) if wall else NA,
                     'gpus': [e.get('gpu') for e in u['run_env']], 'attempts': len(u['attempts']) if isinstance(u['attempts'], list) else NA,
                     'warmup_seconds_sum': sum(w.get('seconds', 0) for w in u['warmup_epochs']),
                     'cotrain_seconds_sum': sum(e.get('seconds', 0) for e in u['cotrain_epochs']),
                     'note': 'PEAK_<job>.json wall_seconds of the driver per attempt; scheduler/staging time not included; sacct unavailable'}
    meta = {'what': 'R30 plan §6.1 verbatim export of R29 logged records; NA = not logged by the original code; nothing reconstructed from states',
            'runs_root': runs, 'n_units': len(units),
            'field_notes': {
                'split_A / split_B': 'r25_co_train.split_quality(pred_X, clean_labels) of the GMM split scored by peer X (post-hoc vs injected-noise truth; never fed to training). n_pred_clean = number of samples in the clean (labeled) split. Peer A trains on the labeled loader of pred_B and vice versa.',
                'train.A / train.B': 'per-peer co-train statistics as logged (sums/means of R27/R29 diagnostics every R27.TIME_EVERY steps; steps = optimizer steps of that peer in the epoch).',
                'r29_gnorm_align_mean': 'encoder gradient norm of the main alignment term alone (autograd.grad on the retained graph, report-only).',
                'gnorm_enc_clean_mean / gnorm_enc_noisy_mean': 'encoder grad norm after the labeled backward (main + intra) / increment added by the noisy branch.',
                'r29_Jlist_* / r29_top1pos_* / r29_pimass_*': 'listwise J (L critic), positive-top1 rate, softmax mass on observed positives of the training batch (all arms, detached).',
                'warmup_diag': 'R29_WARMUP_DIAG.json verbatim, keys <warmup epoch>_<peer>.'},
            'units': units}
    json.dump(meta, open(os.path.join(out_dir, 'R29_DIAG_EXPORT.json'), 'w'), ensure_ascii=False, indent=1)

    # flat CSV
    tkeys = sorted({k for u in units.values() for e in u['cotrain_epochs'] for p in ('A', 'B') for k in e['train'][p]})
    dkeys = sorted({k for u in units.values() for e in u['cotrain_epochs'] for k in e['dev']})
    with open(os.path.join(out_dir, 'R29_DIAG_EXPORT_EPOCHS.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['unit', 'cell', 'arm', 'epoch', 'peer', 'lr', 'is_best', 'seconds', 'peak_mem_gb']
                   + ['split_scored_by_this_peer_' + k for k in ('n_pred_clean', 'precision', 'recall')]
                   + ['dev_' + k for k in dkeys] + ['train_' + k for k in tkeys])
        for uid, u in units.items():
            c, a = uid.split('__')
            for e in u['cotrain_epochs']:
                for p in ('A', 'B'):
                    sp = e.get('split_' + p, {})
                    w.writerow([uid, c, a, e['epoch'], p, e.get('lr', NA), e.get('is_best', NA), e.get('seconds', NA), e.get('peak_mem_gb', NA)]
                               + [sp.get(k, NA) for k in ('n_pred_clean', 'precision', 'recall')]
                               + [e['dev'].get(k, NA) for k in dkeys] + [e['train'][p].get(k, NA) for k in tkeys])

    # markdown
    L = ['# R29 已记录诊断导出（R30 计划 §6.1）', '',
         '只导出 R29 原代码已写入的记录（R29_WARMUP_DIAG.json、EPOCHS.json）；未记录的字段为 NA，不从末轮状态推造。'
         '完整逐 epoch、逐 peer 数值见 R29_DIAG_EXPORT.json 与 R29_DIAG_EXPORT_EPOCHS.csv；下表 co-train 部分只列出 epoch '
         + ', '.join(map(str, MD_EPOCHS)) + '。', '',
         '字段说明：划分 precision/recall 为 split_quality（该 peer 打分的 GMM 划分对注入噪声真值，事后诊断，不进入训练）；'
         'peer A 用 peer B 打分划分的 labeled loader 训练，反之亦然。梯度范数为报告用诊断批的均值。'
         '计数器类字段（跳过B<2、noisy步数、无候选数）缺失表示原代码未写入该键（defaultdict 只在事件发生时建键），按规则记 NA。', '']
    L += ['## 表一 单元与硬件', '']
    rows = []
    for uid, u in units.items():
        env = u['run_env'][-1] if u['run_env'] else {}
        best = max(u['cotrain_epochs'], key=lambda e: e['dev']['rsum'])
        c = u['cost']
        rows.append([uid, f(env.get('gpu')), f(env.get('host')), f(env.get('job')), f(len(u['attempts']) if isinstance(u['attempts'], list) else NA),
                     f(len(u['warmup_epochs'])), f(len(u['cotrain_epochs'])), f(g(u['done'], 'best_rsum'), 2), f(best['epoch']),
                     f(c['gpu_hours_recorded'], 3), f(c['warmup_seconds_sum'], 0), f(c['cotrain_seconds_sum'], 0)])
    L += [md_table(['单元', 'GPU', '节点', '作业', '尝试数', 'warmup轮', 'co-train轮', 'best rSum', 'best epoch（raw）',
                    'GPU小时（driver墙钟）', 'warmup秒合计', 'co-train秒合计'], rows), '']

    L += ['## 表二 warmup 诊断（R29_WARMUP_DIAG.json 全部条目）与 warmup DEV', '']
    rows = []
    for uid, u in units.items():
        wd = u['warmup_diag']
        for we in u['warmup_epochs']:
            ep = we['epoch']
            for p in ('A', 'B'):
                r = g(wd, f'{ep}_{p}')
                rows.append([uid, str(ep), p] + [f(g(r, k), count=k in COUNT_KEYS) for k, _ in WARM_COLS]
                            + [f(we['dev'].get('peer0_rsum' if p == 'A' else 'peer1_rsum', NA), 2), f(we['dev'].get('rsum', NA), 2)])
    L += [md_table(['单元', 'warmup轮', 'peer'] + [h for _, h in WARM_COLS] + ['该peer DEV rSum', '平均DEV rSum'], rows), '']

    L += ['## 表三 co-train 逐 epoch 记录（逐 peer）', '']
    rows = []
    for uid, u in units.items():
        for e in u['cotrain_epochs']:
            if e['epoch'] not in MD_EPOCHS:
                continue
            for p in ('A', 'B'):
                sp, tr = e.get('split_' + p, {}), e['train'][p]
                rows.append([uid, str(e['epoch']), p, f(e.get('lr'), 6), f(g(sp, 'n_pred_clean'), count=True), f(g(sp, 'precision')), f(g(sp, 'recall'))]
                            + [f(tr.get(k, NA), count=k in COUNT_KEYS) for k, _ in TRAIN_COLS]
                            + [f(e['dev'].get('peer0_rsum' if p == 'A' else 'peer1_rsum', NA), 2), f(e['dev'].get('rsum', NA), 2), f(e.get('is_best', NA))])
    L += [md_table(['单元', 'epoch', 'peer', 'lr', '该peer打分划分选中数', '划分precision', '划分recall'] + [h for _, h in TRAIN_COLS]
                   + ['该peer DEV rSum', '平均DEV rSum', 'is_best'], rows), '']
    L += ['## 表四 保存的完整状态中的 FIFO 队列量（只有 raw epoch 10/20/39 存有完整状态；其余 epoch 为 NA）', '']
    rows = []
    for uid, u in units.items():
        if not isinstance(u['queue_at_saved_states'], list):
            continue
        for r in sorted(u['queue_at_saved_states'], key=lambda r: r['raw_epoch']):
            for p in ('A', 'B'):
                q = r[p]
                if not isinstance(q, dict):
                    rows.append([uid, str(r['raw_epoch']), str(r['completed_cotrain_epochs']), p] + [NA] * 6); continue
                rows.append([uid, str(r['raw_epoch']), str(r['completed_cotrain_epochs']), p, str(q['capacity']), str(q['filled']), f(q['is_full']),
                             str(q['id_bearing_slots']), str(q['distinct_captions_in_queue']), str(q['write_epoch_min_max'])])
    L += [md_table(['单元', 'raw epoch', '完成co-train轮', 'peer', '容量', '已填', '已满', '带ID槽', '不同caption数', '写入epoch范围'], rows), '']
    L += ['注：EPOCHS 中的 epoch 为 raw 编号（0 起），完成轮数 = raw + 1；state_epoch10.pt 为 raw 10（完成 11 轮）。成本只含 driver 墙钟（PEAK_*.json），排队/拷贝未计入。', '']
    open(os.path.join(out_dir, 'R29_DIAG_EXPORT.md'), 'w').write('\n'.join(L) + '\n')
    print('units', len(units), 'train keys', len(tkeys))


if __name__ == '__main__':
    main()
