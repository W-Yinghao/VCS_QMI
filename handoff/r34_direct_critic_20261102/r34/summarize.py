"""R34 numbers-only summary: RESULTS.json + RESULTS.md from runs/*/RESULT.json (missing units = NA)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, read_json, write_json, utc

K6 = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
S = {'D': 'D_dual_cosine', 'X': 'X_joint_mlp'}; O = {'J': 'J', 'BCE': 'BCE', 'NCE': 'InfoNCE'}


def f(x, n=2):
    return 'NA' if x is None else f'{x:.{n}f}'


def main():
    units = read_json(ROOT / 'configs' / 'UNITS.json')
    R = {u: read_json(OUT / 'runs' / u / 'RESULT.json') for u in units if (OUT / 'runs' / u / 'DONE.json').exists()}
    L = [f'# R34 DIRECT_CRITIC_RETRIEVAL_v1 results (numbers only)', f'generated {utc()}; {len(R)}/{len(units)} units complete; full DEV 1014 img / 5070 cap; fixed step 2000; TEST not read.', '']
    L += ['## 1. Fixed step-2000 full-DEV retrieval (upstream IN2R i2t/t2i, peer-mean g)', '',
          '| cond | unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δ rSum vs H | rSum tie-opt | rSum tie-pess |', '|' + '---|' * 12]
    summ = {'units': {}, 'paired': {}, 'native': {}}
    for cond in ('n00', 'n60'):
        nat = None
        for s in S:
            for o in O:
                u = f'direct_{cond}_s1001_{S[s]}_{O[o]}'; r = R.get(u)
                if r is None:
                    L.append(f'| {cond} | {u} |' + ' NA |' * 10); continue
                m = r['dev_retrieval_step2000']; tb = r['dev_retrieval_tie_bounded']; nat = r['native_H_dev']
                L.append(f'| {cond} | {S[s]}-{O[o]} | ' + ' | '.join(f(m[k]) for k in K6) + f" | {f(r['diff_vs_native']['rsum'])} | {f(tb['rsum_optimistic'])} | {f(tb['rsum_pessimistic'])} |")
                summ['units'][u] = dict(retrieval=m, diff_vs_native=r['diff_vs_native'], tie_bounded=tb)
        if nat:
            L.append(f'| {cond} | frozen H native (zero training) | ' + ' | '.join(f(nat[k]) for k in K6) + ' | 0 | | |')
            summ['native'][cond] = nat
    L += ['', '## 2. Within-structure paired differences (same init/batches), step 2000', '', '| cond | structure | J−BCE rSum | J−NCE rSum | J−BCE i2t R1 | J−BCE t2i R1 | J−NCE i2t R1 | J−NCE t2i R1 |', '|' + '---|' * 8]
    for cond in ('n00', 'n60'):
        for s in S:
            g = {o: R.get(f'direct_{cond}_s1001_{S[s]}_{O[o]}') for o in O}
            def dd(a, b, k):
                return None if g[a] is None or g[b] is None else g[a]['dev_retrieval_step2000'][k] - g[b]['dev_retrieval_step2000'][k]
            row = [dd('J', 'BCE', 'rsum'), dd('J', 'NCE', 'rsum'), dd('J', 'BCE', 'i2t_r1'), dd('J', 'BCE', 't2i_r1'), dd('J', 'NCE', 'i2t_r1'), dd('J', 'NCE', 't2i_r1')]
            L.append(f'| {cond} | {S[s]} | ' + ' | '.join(f(x) for x in row) + ' |')
            summ['paired'][f'{cond}_{s}'] = dict(zip(('J-BCE_rsum', 'J-NCE_rsum', 'J-BCE_i2t_r1', 'J-BCE_t2i_r1', 'J-NCE_i2t_r1', 'J-NCE_t2i_r1'), row))
            if all(g.values()):
                ids = {(x['identity']['init_state_sha256'], x['identity']['batch_index_sha256']) for x in g.values()}
                summ['paired'][f'{cond}_{s}']['identical_init_and_batches'] = len(ids) == 1
    L += ['', '## 3. J readouts at step 2000 (mean of per-peer J; record U-statistic) and T saturation (DEV, peer A/B)', '',
          '| unit | FIT-obs J | DEV-obs J | DEV-clean J | FIT−DEVobs | J(peer-mean g) DEV-clean | frac abs T≥0.999 A/B | frac abs T==1 A/B |', '|' + '---|' * 8]
    for u, r in R.items():
        j = r['J_step2000']; sat = r['T_saturation_dev']
        L.append(f"| {u} | {f(j['fit_observed']['J_mean_of_peers'],4)} | {f(j['dev_observed']['J_mean_of_peers'],4)} | {f(j['dev_clean']['J_mean_of_peers'],4)} | {f(r['J_fit_minus_devobs'],4)} | {f(j['dev_clean']['J_of_peer_mean_score'],4)} | "
                 f"{f(sat['peerA']['frac_absT_ge_0p999'],4)}/{f(sat['peerB']['frac_absT_ge_0p999'],4)} | {f(sat['peerA']['frac_absT_eq_1_fp32'],4)}/{f(sat['peerB']['frac_absT_eq_1_fp32'],4)} |")
    L += ['', '## 4. Curves (every 250 updates): FIT-obs J / DEV-obs J / DEV-clean J / train loss (window mean) / DEV rSum (trajectory only)', '']
    for u, r in R.items():
        L.append(f'**{u}**'); L.append(''); L.append('| step | FIT-obs J | DEV-obs J | DEV-clean J | loss | rSum |'); L.append('|---|---|---|---|---|---|')
        for c in r['curves']:
            L.append(f"| {c['step']} | {f(c['fit_observed']['J_mean_of_peers'],4)} | {f(c['dev_observed']['J_mean_of_peers'],4)} | {f(c['dev_clean']['J_mean_of_peers'],4)} | "
                     f"{f(c.get('train_window', {}).get('loss_mean'),4)} | {f(c.get('dev_retrieval', {}).get('rsum'))} |")
        L.append('')
    L += ['## 5. Mechanism (DEV clean annotation; native = frozen H cosine)', '',
          '| unit | t2i rescued/lost R1 | i2t rescued/lost R1 | t2i win vs native hardest (critic / native) | i2t win vs native hardest | t2i pair-acc native hard10 (critic / native) | i2t pair-acc hard10 | t2i pair-acc random | top-100 cov t2i/i2t | top-100 rerank rSum-parts t2i R1 / i2t R1 |', '|' + '---|' * 10]
    for u, r in R.items():
        m = r['mechanism']; c, n = m['critic'], m['native']; t = m['top100']
        L.append(f"| {u} | {m['t2i_rank_change']['rescued_r1']}/{m['t2i_rank_change']['lost_r1']} | {m['i2t_rank_change']['rescued_r1']}/{m['i2t_rank_change']['lost_r1']} | "
                 f"{f(c['t2i_win_vs_native_hardest'],4)} / {f(n['t2i_win_vs_native_hardest'],4)} | {f(c['i2t_win_vs_native_hardest'],4)} / {f(n['i2t_win_vs_native_hardest'],4)} | "
                 f"{f(c['t2i_pairacc_native_hard10'],4)} / {f(n['t2i_pairacc_native_hard10'],4)} | {f(c['i2t_pairacc_native_hard10'],4)} / {f(n['i2t_pairacc_native_hard10'],4)} | "
                 f"{f(c['t2i_pairacc_random'],4)} / {f(n['t2i_pairacc_random'],4)} | {f(t['coverage_t2i'],4)}/{f(t['coverage_i2t'],4)} | {f(t['critic_rerank']['t2i_r1'])} / {f(t['critic_rerank']['i2t_r1'])} |")
    L += ['', 'Score means (critic g; native in cosine units, not comparable in scale):', '', '| unit | mean g pos | mean g all pairs | mean g native-hardest t2i | mean g native-hardest i2t | margin vs native hardest t2i mean/median |', '|---|---|---|---|---|---|']
    for u, r in R.items():
        c = r['mechanism']['critic']
        L.append(f"| {u} | {f(c['score_mean_pos'],4)} | {f(c['score_mean_all_pairs'],4)} | {f(c['score_mean_native_hardest_t2i'],4)} | {f(c['score_mean_native_hardest_i2t'],4)} | {f(c['margin_vs_native_hardest_t2i']['mean'],4)}/{f(c['margin_vs_native_hardest_t2i']['median'],4)} |")
    L += ['', '## 6. Cost', '', '| unit | GPU | params | train s (2000 upd) | unit wall s | DEV full scoring s (2 peers) | pairs/s | peak GPU MiB |', '|---|---|---|---|---|---|---|---|']
    for u, r in R.items():
        c = r['cost']
        L.append(f"| {u} | {r['env']['gpu']} ({r['env']['host']}, job {r['env']['job']}) | {c['params']} | {f(c['train_seconds_2000_updates'],1)} | {f(c['unit_wall_seconds'],1)} | {f(c['dev_full_scoring_seconds_both_peers'],3)} | {c['dev_pairs_scored_per_second']:.3g} | {c['peak_gpu_memory_bytes']/2**20:.0f} |")
    fails = sorted(str(p.relative_to(OUT)) for p in (OUT / 'runs').glob('*/FAILED_*.json'))
    L += ['', '## 7. Failure records', ''] + ([f'- {x}' for x in fails] or ['- none'])
    summ['failures'] = fails; summ['n_complete'] = len(R)
    write_json(ROOT / 'RESULTS.json', summ)
    (ROOT / 'RESULTS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
