"""R27-S / R27-D / R27-O number tables (full DEV only, no TEST, no interpretation). Finished units only; missing = NA.
Reads each run's EPOCHS.json + DONE.json (40 co-train epochs, DONE best == max rsum), exactly as summarize_r27.py.
Writes RESULTS_NORMALIZED.json (package input format), COMPARISONS.json (package tools/analyze_results.py, unchanged),
SDO_SUMMARY.json and SDO_TABLES.md (per-cell endpoints + paired differences where both sides are finished).
Usage: summarize_sdo.py"""
import json, os, glob, sys, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = '/home/infres/yinwang/CS_QMI/r27d_server/VCS_Retrieval_R27D_Server_20261030'
ROOTS = {'r27': '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/runs',
         'r27d': '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_directional_20261030/runs',
         'r27o': '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_oracle_20261030/runs',
         'r27o3': '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_oracle3_20261031/runs',
         'r27cs': '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_candsrc_20261031/runs'}
R25 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups'
CELLS = ['f40_s1001', 'f60_s1001']
ARMS = [('A0_native', None), ('a1_iface', 'r27'), ('a3_vcs', 'r27'),
        ('s1_balanced_dual', 'r27'), ('s2_single_free_vcs', 'r27'), ('s3_single_cap_delta', 'r27'), ('s4_single_cap_vcs', 'r27'),
        ('s5_single_cap_bce', 'r27'), ('s6_single_cap_snce', 'r27'),
        ('d1_vcs_i2t', 'r27d'), ('d2_vcs_t2i', 'r27d'), ('d3_snce_i2t', 'r27d'), ('d4_snce_t2i', 'r27d'),
        ('o1_rescue_native_t2i', 'r27o'), ('o2_rescue_vcs_t2i', 'r27o'), ('o3_true_target_t2i', 'r27o3'),
        ('bn_cross_t2i', 'r27cs'), ('cn_mix_t2i', 'r27cs')]
KEYS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
PAIRS = [('s1_balanced_dual', 'a1_iface'), ('s2_single_free_vcs', 'a1_iface'), ('s3_single_cap_delta', 'a1_iface'), ('s4_single_cap_vcs', 'a1_iface'),
         ('s5_single_cap_bce', 'a1_iface'), ('s6_single_cap_snce', 'a1_iface'), ('s2_single_free_vcs', 'a3_vcs'), ('s4_single_cap_vcs', 'a3_vcs'),
         ('s4_single_cap_vcs', 's5_single_cap_bce'), ('s4_single_cap_vcs', 's6_single_cap_snce'), ('s4_single_cap_vcs', 's3_single_cap_delta'),
         ('d1_vcs_i2t', 'a1_iface'), ('d2_vcs_t2i', 'a1_iface'), ('d3_snce_i2t', 'a1_iface'), ('d4_snce_t2i', 'a1_iface'),
         ('d1_vcs_i2t', 'd3_snce_i2t'), ('d2_vcs_t2i', 'd4_snce_t2i'), ('s4_single_cap_vcs', 'd1_vcs_i2t'), ('s4_single_cap_vcs', 'd2_vcs_t2i'),
         ('s6_single_cap_snce', 'd3_snce_i2t'), ('s6_single_cap_snce', 'd4_snce_t2i'),
         ('o1_rescue_native_t2i', 'a1_iface'), ('o2_rescue_vcs_t2i', 'd2_vcs_t2i'), ('o2_rescue_vcs_t2i', 'o1_rescue_native_t2i'),
         ('o3_true_target_t2i', 'o1_rescue_native_t2i'), ('o3_true_target_t2i', 'a1_iface'), ('o3_true_target_t2i', 'o2_rescue_vcs_t2i'),
         ('bn_cross_t2i', 'a1_iface'), ('cn_mix_t2i', 'a1_iface'), ('bn_cross_t2i', 'cn_mix_t2i')]
METRICS = ['best', 'last5', 'i2t_r1', 't2i_r1']


def run(d):
    done = os.path.join(d, 'DONE.json')
    if not os.path.exists(done):
        return None
    ep = json.load(open(os.path.join(d, 'EPOCHS.json')))['cotrain']
    rs = [e['dev']['rsum'] for e in ep]
    assert len(ep) == 40 and [e['epoch'] for e in ep] == list(range(40)), d
    assert abs(max(rs) - json.load(open(done))['best_rsum']) < 1e-6, d
    b = int(np.argmax(rs))
    peaks = sorted(glob.glob(os.path.join(d, 'PEAK_*.json')))
    att = open(os.path.join(d, 'ATTEMPTS')).read().strip() if os.path.exists(os.path.join(d, 'ATTEMPTS')) else None
    return {'best': max(rs), 'epoch': b, 'last': rs[-1], 'last5': float(np.mean(rs[-5:])), **{k: ep[b]['dev'][k] for k in KEYS},
            'attempts': att, 'gpu': [json.load(open(p)).get('gpu') for p in peaks],
            'wall_h_sum_recorded': sum(json.load(open(p)).get('wall_seconds', 0) for p in peaks) / 3600}


res, missing = {}, []
for c in CELLS:
    res[c] = {}
    for a, root in ARMS:
        r = run(os.path.join(R25, c, 'H0')) if root is None else run(os.path.join(ROOTS[root], c, a))
        if r is None:
            missing.append(f'{c}/{a}')
        else:
            res[c][a] = r
pkg_arms = {'a1_iface', 's4_single_cap_vcs', 's6_single_cap_snce', 'd1_vcs_i2t', 'd2_vcs_t2i', 'd3_snce_i2t', 'd4_snce_t2i'}
norm = {'runs': {c: {a: {k: v for k, v in r.items() if k in ['best', 'epoch', 'last', 'last5'] + KEYS} for a, r in res[c].items() if a in pkg_arms} for c in CELLS}}
json.dump(norm, open(os.path.join(HERE, 'RESULTS_NORMALIZED.json'), 'w'), indent=1)
spec = importlib.util.spec_from_file_location('r27d_analyze', os.path.join(PKG, 'tools/analyze_results.py'))
AR = importlib.util.module_from_spec(spec); spec.loader.exec_module(AR)
pkg_out = AR.analyze(norm['runs'])
json.dump(pkg_out, open(os.path.join(HERE, 'COMPARISONS.json'), 'w'), indent=1)
pairs = {c: {f'{x}-{y}': ({m: res[c][x][m] - res[c][y][m] for m in METRICS} if x in res[c] and y in res[c] else None) for x, y in PAIRS} for c in CELLS}
inter = {c: pkg_out[c]['interaction'] for c in CELLS}
json.dump({'scope': 'Flickr30K seed1001, full DEV (1,014 img / 5,070 cap), best rSum over 40 co-train epochs; A0 = R25 H0; '
                    'R27-O (O1/O2: true image inserted into t2i candidates; O3: true image re-encoded as the t2i encoder target) are oracle diagnostics, not main-table results; '
                    'bn_cross_t2i / cn_mix_t2i = A1 host with t2i candidates from direct cross-modal search / fixed 3+2 mix (whole-interface change, unique IDs); no TEST',
           'missing': missing, 'runs': res, 'pairs': pairs, 'interaction_package': inter},
          open(os.path.join(HERE, 'SDO_SUMMARY.json'), 'w'), indent=1)

f = lambda v: 'NA' if v is None else f'{v:+.2f}'
L = ['# R27-S / R27-D / R27-O 数字表（自动生成，full DEV；未读 TEST；只列数字，不含解读）', '',
     'Flickr30K × seed1001。A0 = R25 原生 H0；A1/A3 为 R27 已完成单元；R27-O 为显式使用真对应的诊断（O1/O2：t2i 候选救援；O3：t2i 编码器目标直接取当前 peer 重编码的真图），不进入主结果表。bn_cross_t2i / cn_mix_t2i：A1 宿主，t2i 候选改为直接跨模态检索 / 固定 3+2 混合（唯一 ID，整体接口改动）。', '',
     f'缺失单元（未完成，记 NA，不以其他臂替代）：{", ".join(missing) if missing else "无"}', '']
for c in CELLS:
    L += [f'## {c}：各臂终点', '', '| 臂 | best rSum | best epoch | last | last5 | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | attempts |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a, _ in ARMS:
        r = res[c].get(a)
        if r is None:
            L.append(f'| {a} | NA |' + ' |' * 10 + ' |')
        else:
            L.append(f"| {a} | {r['best']:.2f} | {r['epoch']} | {r['last']:.2f} | {r['last5']:.2f} | " + ' | '.join(f'{r[k]:.2f}' for k in KEYS) + f" | {r['attempts'] or ''} |")
    L += ['', f'## {c}：配对差（单 seed）', '', '| 比较 | best rSum | last5 | i2t R@1 | t2i R@1 |', '|---|---|---|---|---|']
    for k, v in pairs[c].items():
        L.append(f'| {k} | ' + (' | '.join(f(v[m]) for m in METRICS) if v else 'NA | NA | NA | NA') + ' |')
    L += ['', f'## {c}：2×2 交互（包 tools/analyze_results.py；I_V = S4−D1−D2+A1，I_N = S6−D3−D4+A1）', '', '| 交互 | best rSum | last5 | i2t R@1 | t2i R@1 |', '|---|---|---|---|---|']
    for lab, d in inter[c].items():
        L.append(f'| I_{lab} | ' + ' | '.join(f(d[m]) for m in METRICS) + ' |')
    L.append('')
open(os.path.join(HERE, 'SDO_TABLES.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
