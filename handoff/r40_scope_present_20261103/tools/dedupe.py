"""R40 de-duplication: none of the 27 R40 cells may repeat an existing unit (R37 / R37C / R37S critic / R38 / R39 / R39X configs, runs on disk,
live queue). Cell key = (cond, structure, supervision, updated modalities, objective, steps, schedule, init_seed, sampler_seed, n_queries).
Also lists the reused read-only references (R38 N-FROZEN, R39 N-JOINT, R39 ORACLE-FROZEN) per R40 unit. Writes DEDUPE.json.
Usage: python3 -I -B tools/dedupe.py"""
import glob, json, subprocess, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
H = Path('/home/infres/yinwang/CS_QMI/handoff')
SOURCES = {'R37': H / 'r37_confirm_20261102/configs', 'R37C': H / 'r37c_cc152k_20261102/configs', 'R37S': H / 'r37s_cc152k_r30h_20261102/configs',
           'R38': H / 'r38_direction_20261102/configs', 'R39+R39X': H / 'r39_complete_joint_20261102/configs'}
MOD = {'frozen': (), 'joint_global': ('img', 'txt')}


def key(c):
    mods = tuple(c['student_modalities']) if 'student_modalities' in c else MOD[c.get('encoder', 'frozen')]
    return (c.get('cond'), c.get('structure'), c.get('supervision'), mods, c.get('objective'), c.get('steps'), c.get('optimizer', {}).get('schedule', 'constant'),
            c.get('init_seed'), json.dumps(c.get('sampler_seed')), c.get('n_queries'))


def main():
    existing = {}
    for name, d in SOURCES.items():
        for f in sorted(glob.glob(str(d / '*.json'))):
            if f.endswith('UNITS.json') or 'UNITS_' in f:
                continue
            c = json.load(open(f))
            if not isinstance(c, dict) or 'unit_id' not in c:
                continue
            existing.setdefault(key(c), []).append(dict(line=name, unit=c['unit_id']))
    r40 = json.load(open(W / 'configs/UNITS.json'))
    reuse = json.load(open(W / 'REUSE_REFERENCES.json'))['by_unit']
    rows, dups = [], []
    try:
        q = subprocess.run(['squeue', '-u', 'yinwang', '-h', '-o', '%j'], capture_output=True, text=True, timeout=60).stdout.split()
    except Exception as e:   # noqa
        q = None
    for u in r40:
        c = json.load(open(W / 'configs' / f'{u}.json'))
        k = key(c); hit = existing.get(k, [])
        near = [e for kk, v in existing.items() for e in v if kk[0] == k[0] and kk[2] == k[2] and kk[3] == k[3] and kk[4] == k[4] and kk != k]
        if hit:
            dups.append(u)
        rows.append(dict(unit=u, key=list(k), duplicate_of=hit, same_cond_supervision_scope_objective_other_recipe=near, reused_references=reuse[u],
                         out_dir_exists=any(Path(p).exists() for p in (f'/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/runs/{u}',
                                                                        f'/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/oracle_runs/{u}'))))
    out = dict(rule='cell key = (cond, structure, supervision, updated modalities, objective, steps, schedule, init_seed, sampler_seed, n_queries) over all R37/R37C/R37S/R38/R39/R39X configs',
               n_r40=len(r40), n_existing_configs=sum(len(v) for v in existing.values()), duplicates=dups, ALL_UNIQUE=not dups,
               queue_r40_jobs=[j for j in (q or []) if j.startswith('r40_')] if q is not None else 'squeue unavailable',
               notes=['R37 r37_factor_*_N_PRESENT_ONLY_G_* (Flickr60 s1001, constant schedule) differs in schedule: not an exact reuse (package section 4); listed under same_cond_* when matching',
                      'R39X changes the parent / noise level (other conds); R40 keeps the R39 parents n60 / cc'],
               units=rows)
    (W / 'DEDUPE.json').write_text(json.dumps(out, indent=1))
    print('DEDUPE ALL_UNIQUE', out['ALL_UNIQUE'], 'existing', out['n_existing_configs'], 'dups', dups)


if __name__ == '__main__':
    main()
