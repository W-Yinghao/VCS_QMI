"""Summarise runs/*/ROWS.jsonl (DEV) -> RESULTS.json + RESULTS_CN.txt. No new computation on images."""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
THR = {'comir': (9.0, 42.0), 'mre': (16.68, 9.0, 42.0)}
LABEL = {'MF_VCS_M1': 'MF_VCS_M1 (VCS)', 'MLP_VCS_perm8': 'MLP_VCS_perm8 (VCS)', 'InfoNCE_host[J_perm1]': 'InfoNCE_host',
         'CoMIR_native': 'CoMIR + our searcher', 'CoMIR_SIFT': 'CoMIR + SIFT (paper best pipeline)', 'MIND_2D_adapted': 'MIND + our searcher',
         'MI_native': 'soft MI + our searcher', 'MI_SimpleITK': 'Mattes MI (SimpleITK, registered config)', 'MI_SimpleITK_tuned': 'Mattes MI (SimpleITK, CAL-tuned, amendment 1)'}


def load():
    rows = []
    for d in sorted((HERE / 'runs').glob('*')):
        if (d / 'COMPLETE.json').exists() and json.loads((d / 'COMPLETE.json').read_text()).get('role', 'DEV') == 'DEV':
            rows += [json.loads(l) for l in open(d / 'ROWS.jsonl')]
    return rows


def key(r):
    return (r['method'] + ('' if r.get('norm') in ('patch', 'raw', None) else f"[norm={r['norm']}]"))


def stats(errs, thr):
    e = np.asarray(errs, float)
    return dict(n=len(e), **{f's<{t:g}': float(np.mean(e < t)) for t in thr}, median=float(np.median(e)), p95=float(np.percentile(e, 95)))


def core_rates(rows, t):
    by = defaultdict(list)
    for r in rows: by[r['core']].append(r['err_full'] < t)
    return {c: float(np.mean(v)) for c, v in by.items()}


def paired(a_rows, b_rows, t, B=10000, seed=0):
    a, b = core_rates(a_rows, t), core_rates(b_rows, t); cores = sorted(set(a) & set(b))
    d = np.array([a[c] - b[c] for c in cores]); g = np.random.default_rng(seed)
    boot = d[g.integers(0, len(d), size=(B, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], cores=len(cores),
                positive_cores=int((d > 0).sum()), negative_cores=int((d < 0).sum()))


def main():
    rows = load(); out = dict(protocols={}, note='DEV32 development evidence; BF reference / transformed SHG moving; see PROTOCOL.json')
    for proto in ('comir', 'mre'):
        P = [r for r in rows if r['protocol'] == proto]; thr = THR[proto]
        head = (lambda r: r['k'] == 0) if proto == 'comir' else (lambda r: r['k'] < 4)
        by = defaultdict(list)
        for r in P: by[(key(r), r['seed'])].append(r)
        res = {}
        for (m, s), rs in sorted(by.items()):
            res.setdefault(m, {})[str(s)] = dict(all=stats([r['err_full'] for r in rs], thr), published_design=stats([r['err_full'] for r in rs if head(r)], thr),
                                                 **({'levels': {str(L): stats([r['err_full'] for r in rs if r['level'] == L], thr) for L in (1, 2, 3, 4)}} if proto == 'mre' else {}))
        means = {}
        for m, d in res.items():
            seeds = sorted(d); means[m] = {k: float(np.mean([d[s]['all'][k] for s in seeds])) for k in d[seeds[0]]['all'] if k != 'n'}
            means[m]['seeds'] = len(seeds)
            means[m]['published_design'] = {k: float(np.mean([d[s]['published_design'][k] for s in seeds])) for k in d[seeds[0]]['published_design'] if k != 'n'}
        pooled = defaultdict(list)
        for r in P: pooled[key(r)].append(r)
        t0 = thr[0]; refs = [m for m in ('CoMIR_SIFT', 'CoMIR_native', 'MI_native', 'MI_SimpleITK_tuned', 'MI_SimpleITK') if m in pooled]
        pairs = {m: {ref: paired(pooled[m], pooled[ref], t0) for ref in refs if ref != m and not (m in refs and refs.index(m) > refs.index(ref))} for m in pooled}
        if 'MF_VCS_M1' in pooled and 'InfoNCE_host[J_perm1]' in pooled:
            pairs['VCS_vs_InfoNCE'] = {'MF_VCS_M1 − InfoNCE_host': paired(pooled['MF_VCS_M1'], pooled['InfoNCE_host[J_perm1]'], t0),
                                      'MLP_VCS_perm8 − InfoNCE_host': paired(pooled.get('MLP_VCS_perm8', []), pooled['InfoNCE_host[J_perm1]'], t0) if 'MLP_VCS_perm8' in pooled else None}
        out['protocols'][proto] = dict(thresholds=thr, per_seed=res, seed_means=means, paired_core_level_primary_threshold=pairs)
    cost = []
    for d in sorted((HERE / 'runs').glob('*')):
        c = json.loads((d / 'COMPLETE.json').read_text()) if (d / 'COMPLETE.json').exists() else None
        if c: cost.append(dict(run=d.name, role=c.get('role'), seconds=c.get('seconds'), gpu=c.get('gpu', 'CPU')))
    out['cost'] = dict(runs=cost, gpu_hours=sum(c['seconds'] for c in cost if c['gpu'] != 'CPU') / 3600,
                       cpu_job_hours=sum(c['seconds'] for c in cost if c['gpu'] == 'CPU') / 3600,
                       note='in-job seconds from COMPLETE.json; scheduler/startup overhead not included')
    out['literature_TEST134_other_problems'] = dict(CoMIR_paper_Table2_BF_to_SHG='CoMIR+SIFT 56.0/73.1 %, MS alpha-AMD 53.7/67.2 %, MI 5/22 % (<9/<42 px; CoMIR fixed patches)',
                                                   Lu2022_histological_lt2pct='CoMIR+alpha-AMD 81.3 %, MI 56.5 %, MIND 29.1 % (their 4-level patches)')
    (HERE / 'RESULTS.json').write_text(json.dumps(out, indent=1) + '\n')
    L = ['R23 任务 B：CoMIR / MultiRegEval 已发表协议在 DEV32 上的复现（开发证据；BF 参照、SHG 变换；未读 TEST）', '']
    for proto, name in (('comir', 'CoMIR 协议（±30°、x/y ±100 px、834 patch；成功 <9 / <42 px）'), ('mre', 'MultiRegEval 协议（4 级，θ≤20°、t≤80 px；成功 <2% = 16.68 px）')):
        d = out['protocols'].get(proto)
        if not d: continue
        L.append(name)
        for m, v in sorted(d['seed_means'].items(), key=lambda kv: -kv[1][f's<{d["thresholds"][0]:g}']):
            cols = ' / '.join(f"{v[f's<{t:g}']:.3f}" for t in d['thresholds'])
            pd_ = ' / '.join(f"{v['published_design'][f's<{t:g}']:.3f}" for t in d['thresholds'])
            base = m.split('[')[0]; lab = LABEL.get(m) or LABEL.get(base, base) + m[len(base):]
            L.append(f"  {lab}（{v['seeds']} seed）：全部 320 问题 {cols}；中位 {v['median']:.2f} px；p95 {v['p95']:.1f}；已发表设计子集 {pd_}")
            if proto == 'mre':
                lv = [float(np.mean([d['per_seed'][m][s_]['levels'][str(Lv)]['s<16.68'] for s_ in d['per_seed'][m]])) for Lv in (1, 2, 3, 4)]
                L.append('      分级 <16.68 px（L1–L4）：' + ' / '.join(f'{x:.3f}' for x in lv))
        L.append('  配对（core 级，主阈值，均值 [95% bootstrap]）：')
        for m, prs in d['paired_core_level_primary_threshold'].items():
            for ref, s in (prs or {}).items():
                if s: L.append((f"    {ref}: " if m == 'VCS_vs_InfoNCE' else f"    {m} − {ref}: ") + f"{s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]，正/负 core {s['positive_cores']}/{s['negative_cores']}")
        L.append('')
    L.append(f"成本：GPU {out['cost']['gpu_hours']:.2f} h（作业内计时），CPU 作业 {out['cost']['cpu_job_hours']:.2f} h。")
    L.append('文献（TEST134、不同问题，仅作量级参照）：CoMIR 论文 BF→SHG CoMIR+SIFT 56.0/73.1%、MI 5/22%；Lu 2022 <2%：CoMIR+α-AMD 81.3%、MI 56.5%、MIND 29.1%。')
    L.append('限制：DEV 是复用的开发数据（V10 checkpoint 在宿主 DEV 协议上开发过）；已发表数字在 TEST134、不同的固定变换上，不能直接比较。')
    L.append('  CoMIR 是官方代码在我们 FIT32 上以 1/4 尺度（208 px 补丁）重训的模型；CoMIR+SIFT 用 OpenCV SIFT（上采样与 RANSAC 阈值在 CAL 上选），论文用 834 px 全分辨率 CoMIR 与 Fiji SIFT，因此这里的 CoMIR+SIFT 显著低估论文流程，不能当作论文结果的复现。')
    L.append('  方向只做 BF 参照、SHG 变换；我们的搜索器平移被限制在 ±28（512 帧）= ±112 全分辨率 px，覆盖 ±100 px 协议但不是无界搜索。')
    (HERE / 'RESULTS_CN.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
