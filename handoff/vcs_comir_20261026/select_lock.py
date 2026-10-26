"""Pre-registered CAL selection (PROTOCOL.json cal_selection): per family, then best critic per objective. Writes LOCK.json and
LINES_FINAL_BF.txt / LINES_FINAL_EPFL.txt. Reads CAL summaries only (no DEV)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/vcs_comir_20261026/runs')
FAMILIES = ('nce_mse', 'nce_cos', 'vcs_exp', 'vcs_cos', 'bce_exp', 'bce_cos')
OBJECTIVE = {'nce': ('nce_mse', 'nce_cos'), 'vcs': ('vcs_exp', 'vcs_cos'), 'bce': ('bce_exp', 'bce_cos')}
LINE = ('sbatch --partition=RTX6000PRO,L40S,H100 --job-name=r24_comir_{ds}_{name}_s{seed} --time=02:00:00 '
        '/home/infres/yinwang/CS_QMI/handoff/vcs_comir_20261026/gpu.sbatch final --ds {ds} --name {name} --seed {seed}')


def cal_grid():
    rows = []
    for obj in ('nce_mse', 'nce_cos', 'vcs_exp', 'bce_exp'):
        rows += [dict(obj=obj, tau=tau, lr=0.01) for tau in (0.1, 0.5, 2.0)]
    for obj in ('vcs_cos', 'bce_cos'):
        rows += [dict(obj=obj, tau=None, lr=lr) for lr in (0.01, 0.003)]
    return rows


def cfg_name(c):
    return c['obj'] + (f"_t{c['tau']}" if c['tau'] is not None else '') + f"_lr{c['lr']}"


def key(entry):
    m = entry['cal']
    return (-m['success42'], -m['success9'], m['median_err'], entry['order'])


def main():
    assert not (HERE / 'LOCK.json').exists(), 'LOCK.json already written'
    res, missing = {}, []
    for order, c in enumerate(cal_grid()):
        p = OUT / f"bf_cal_{cfg_name(c)}_s1017" / 'COMPLETE.json'
        if not p.exists():
            missing.append(cfg_name(c)); continue
        d = json.loads(p.read_text())
        assert d['status'] == 'COMPLETE' and 'DEV' not in d and d['identity']['config'] == c
        res[cfg_name(c)] = dict(config=c, order=order, cal=d['CAL'])
    if missing:
        print('WAITING', len(res), '/ 16; missing', missing); return
    selected = {}
    for fam in FAMILIES:
        name, entry = min(((n, e) for n, e in res.items() if e['config']['obj'] == fam), key=lambda kv: key(kv[1]))
        selected[fam] = dict(name=name, **entry)
    best = {o: min((selected[f] for f in fams), key=key)['name'] for o, fams in OBJECTIVE.items()}
    finals = sorted({selected[f]['name'] for f in FAMILIES} | {'nce_mse_t0.5_lr0.01'})
    lock = dict(rule='per family lexicographic (CAL success42 desc, success9 desc, median asc, grid order); best critic per objective by '
                     'the same rule over its two selected configs; nce_mse_t0.5_lr0.01 always added as the V10 CoMIR_native reproduction',
                selected=selected, best_critic=best, finals=finals, cal_all=res)
    (HERE / 'LOCK.json').write_text(json.dumps(lock, indent=1) + '\n')
    for ds in ('bf', 'epfl'):
        lines = [LINE.format(ds=ds, name=n, seed=s) for n in finals for s in (1001, 1002, 1003)]
        (HERE / f'LINES_FINAL_{ds.upper()}.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({f: selected[f]['name'] for f in FAMILIES}), json.dumps(best))


if __name__ == '__main__':
    main()
