"""R50 production patch (protocol FIT_FIXED_CENTER_V1; package VCS_Layout_Audit_R50_20261104). Applied once to a byte copy of an R48 tree (r48f -> r50f,
r48c -> r50c); anchors asserted (each exactly once). Edits:
  common.py      R50 block appended (outputs on home: /projects had 301 GB free = the 300 GB floor; R50 FIT-fixed conds <base>_lfF + JOINT parents).
  train_unit.py  import r50ext; after the step-0 and the final readout (phase 1) run r50ext.layout_checks (in-unit checks P/B/C/D/E/F/G, forked RNG).
  r50ext.py      new (tools/r50_src/r50ext.py, byte copy).
  r50f only: oracle_truth.py = byte copy of r42f/oracle_truth.py; train_oracle.py = r42f/train_oracle.py with the diagnostics written per job and per cond
                 (ORACLE_DIAGNOSTICS_R50_<job>.json: no shared file -> oracle jobs may run in parallel; noise file of the unit's cond instead of n60) and the
                 full-FIT-pool oracle exposure (true partner in the natural top-32, N-null rows that the rule relabels) computed in the oracle process.
Usage: python3 tools/r50_patch.py <tree_dir> <f|c>"""
import shutil, sys
from pathlib import Path
T = Path(sys.argv[1]); KIND = sys.argv[2]; W = Path(__file__).resolve().parents[1]
assert KIND in ('f', 'c')


def patch(fn, reps):
    s = (T / fn).read_text()
    for a, b in reps:
        assert s.count(a) == 1, (fn, a[:80], s.count(a)); s = s.replace(a, b)
    (T / fn).write_text(s)


BASES = "('n60', 'f80in')" if KIND == 'f' else "('cc',)"
COMMON = f'''

# ----------------------------------------------------------------------------- R50 (FIT_FIXED_CENTER_V1; tree = byte copy of the R48 tree + this block + tools/r50_patch.py edits)
# One fixed FIT mean of the raw parent encoder output per peer / modality (the R47 G FIT means: images = unique FIT images, captions = FIT caption records);
# the frozen-H cache, the critic inputs and the JOINT student use normalize(e - mu_FIT) on FIT and DEV (DEV only applies the FIT statistic). Caches / prep /
# JOINT prep: tools/r50_build.py (FIT cache files = hard links of the R47 G FIT cache; DEV recomputed). Run outputs on home (/projects at the 300 GB floor).
PROTOCOL_ID = 'FIT_FIXED_CENTER_V1'
R50_OUT = Path('/home/infres/yinwang/CS_QMI/outputs_v9_r50')
RUNS = R50_OUT / 'runs_r50'
ORACLE_RUNS = R50_OUT / 'oracle_runs_r50'
FROZEN_FILE = 'PROTOCOL_FROZEN_R50.json'
FROZEN_FILES_R50 = ('PROTOCOL_FROZEN_R50_A3.json', 'PROTOCOL_FROZEN_R50.json')   # amendment A3 freeze first: re-run units bind to the A3 code; check_frozen takes the first file listing all units
FROZEN_FILES_R41 = FROZEN_FILES_R50   # name used by train_unit.check_frozen
R50_PARENTS = OUT / 'r50_parents'
R50_BASES = {BASES}
for _b in R50_BASES:
    CONDS[f'{{_b}}_lfF'] = dict(CONDS[f'{{_b}}_lfG'], cache_dir=str(R50_PARENTS / 'cache' / _b), prep_dir=str(R50_PARENTS / 'prep' / _b), prep_key=_b, lf_base=_b,
                               lf_variant='F', protocol_id=PROTOCOL_ID)
    JOINT_PARENTS[f'{{_b}}_lfF'] = dict(JOINT_PARENTS[f'{{_b}}_lfG'], prep_dir=R50_PARENTS / 'joint' / f'{{_b}}_F')
'''
with open(T / 'common.py', 'a') as f:
    f.write(COMMON)
patch('train_unit.py', [
    ("import r40ext as R40\n", "import r40ext as R40\nimport r50ext as R50   # R50: in-unit layout checks (FIT_FIXED_CENTER_V1)\n"),
    ("""                assert rec['step0_native_check']['passed'], f"step-0 does not reproduce native H: {rec['step0_native_check']}"
""",
     """                assert rec['step0_native_check']['passed'], f"step-0 does not reproduce native H: {rec['step0_native_check']}"
            if rk is not None and cfg.get('phase', 1) == 1 and step in (0, steps):   # R50: in-unit layout checks P/B/C/D/E/F/G (forked RNG; state digest asserted)
                rec['r50_layout_check'] = R50.layout_checks(ctx, heads, stud, cfg, out, step, g_pairs_lists)
"""),
])
shutil.copyfile(W / 'tools' / 'r50_src' / 'r50ext.py', T / 'r50ext.py')
if KIND == 'f':
    shutil.copyfile(W / 'r42f' / 'oracle_truth.py', T / 'oracle_truth.py')
    shutil.copyfile(W / 'r42f' / 'train_oracle.py', T / 'train_oracle.py')
    patch('train_oracle.py', [
        ("""    truth = oracle_truth.truth_map()
    obs = np.load(CONDS['n60']['noise_file']).astype(np.int64)
    ORACLE_RUNS.mkdir(parents=True, exist_ok=True)
    write_json(ORACLE_RUNS / 'ORACLE_DIAGNOSTICS.json', dict(utc=utc(), observed_mismatch_fraction_full_pool=float((obs != truth).mean()),
               note='evaluator statistic computed in the oracle process only'))
""",
         """    truth = oracle_truth.truth_map()
    ORACLE_RUNS.mkdir(parents=True, exist_ok=True)
    # R50: per job / per cond diagnostics (no shared file between parallel oracle jobs); exposure of the ORACLE_MISSING rule over the full FIT pool under
    # the N query distribution (record draws: i2t query = observed image of the record, t2i query = the record's caption)
    from common import ROOT, read_json
    _units = sys.argv[sys.argv.index('--units') + 1].split(',') if '--units' in sys.argv else []
    _diag = {}
    for _u in _units:
        _cond = read_json(ROOT / 'configs' / f'{_u}.json')['cond']
        if _cond in _diag:
            continue
        _c = CONDS[_cond]; obs = np.load(_c['noise_file']).astype(np.int64)
        _z = np.load(Path(_c['prep_dir']) / f"candidates_{_c['prep_key']}.npz"); ci = _z['i2t_top32'].astype(np.int64); ct = _z['t2i_top32'].astype(np.int64)
        _img = np.arange(len(ci))
        pres_i = (obs[ci] == _img[:, None]).any(1); true_i = (truth[ci] == _img[:, None]).any(1)
        pres_t = (ct == obs[:, None]).any(1); true_t = (ct == truth[:, None]).any(1)
        acc_i = (~pres_i & true_i)[obs]; acc_t = ~pres_t & true_t
        _diag[_cond] = dict(noise_file=_c['noise_file'], observed_mismatch_fraction_full_pool=float((obs != truth).mean()),
                            true_partner_in_top32=dict(i2t=float(true_i[obs].mean()), t2i=float(true_t.mean()), i2t_per_image=float(true_i.mean())),
                            observed_partner_in_top32=dict(i2t=float(pres_i[obs].mean()), t2i=float(pres_t.mean())),
                            null_rows=dict(i2t=float((~pres_i)[obs].mean()), t2i=float((~pres_t).mean())),
                            relabelled_rows=dict(i2t=float(acc_i.mean()), t2i=float(acc_t.mean())),
                            relabelled_fraction_of_null_rows=dict(i2t=float(acc_i.sum() / max(1, (~pres_i)[obs].sum())), t2i=float(acc_t.sum() / max(1, (~pres_t).sum()))),
                            mass_added_per_relabelled_row=oracle_truth.REAL_MASS,
                            expected_added_true_mass_per_row=dict(i2t=float(oracle_truth.REAL_MASS * acc_i.mean()), t2i=float(oracle_truth.REAL_MASS * acc_t.mean())))
    write_json(ORACLE_RUNS / f"ORACLE_DIAGNOSTICS_R50_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(utc=utc(), units=_units, conds=_diag,
               note='evaluator statistics computed in the oracle process only (R50: per job file; full FIT pool, record-induced query distribution)'))
"""),
    ])
print('R50 patched', T, KIND)
