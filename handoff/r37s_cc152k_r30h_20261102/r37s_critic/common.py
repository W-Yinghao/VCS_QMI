"""R37S critic tree r37s_critic = byte copy of R38 r38c (CC152K, directional head + cosine schedule; superset of R37C), R37S edits
in common.py only: ROOT/OUT -> the R37S line dir / outputs, CODE_NATIVE -> code_r37s (the code the R37S parent was trained with;
its native vse/data/evaluation/vocab/co_train/r25_co_train/model are byte-identical to the R31 code used by R37C/R38), and a single
condition 'cc_r37s' = the R37S CC152K R30-style H parent (warmup 5 + H0 co-train 40, A1, K5, i2t/t2i on, seed 1001, DEV-best).
This parent is a DIFFERENT SERIES from the R31 H0_default parents of R37C/R38: never mixed in one table with R31-parent results.
Development experiment: no TEST file is read in this line. All R30-R38 files are read-only and never written.
(R38 docstring follows.)
R38 (CC152K code tree r38c = byte copy of R37C r37c; R38 edits: ROOT/OUT, three R31 parents s1001/s1002/s1003, per-parent
prep dirs; s1001 cache/prep are the R37C artifacts, reused read-only). R37C CC152K: shared paths, constants, hashing, TEST-file guard, isolation guard. Copied from R37 common.py
(paths/conditions/dataset constants changed). Separate line R37C (CC152K block of R37), development experiment:
no TEST file is read in this line. All R30-R37 files are read-only and never written.

CC152K identity (native IN2R data.py, re-derived and asserted in build_cache.py): train_ims.npy has one feature row per
caption row (150000 = 150000), so the native PrecompDataset sets im_div = 1 and t2i_index = arange(150000): record k =
image row k <-> caption row k (one caption per image). DEV 1000 images x 1000 captions, per_captions = 1. The observed
relation is this native identity (real, uninjected noise; no truth exists). Flickr's 5-captions-per-image / record//5
rule is NOT used anywhere (PER_CAP = 1 and the identity map are explicit)."""
from __future__ import annotations
import hashlib, importlib.abc, json, os, re, sys, time
from pathlib import Path

ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37s_cc152k_r30h_20261102/critic')
R37C_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102')
R37C_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102')
RUNS = OUT / 'runs'
ORACLE_RUNS = OUT / 'oracle_runs'   # no oracle units; kept so the trainer-role guard still refuses this tree
PREP = OUT / 'prep'                 # R37S: prep dir of the single R37S parent (locks, calibration, top-32 tables, B-TRUST)
R37_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102')
R31_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101')
CODE_NATIVE = ROOT / 'code_r37s'    # R37S: the code the parent was trained with (native modules byte-identical to R31 code)
DATA_NAME = 'cc152k_precomp'
DATA_SRC = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data') / DATA_NAME
VOCAB = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab')
R31_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101/runs')
PARENT_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37s_cc152k_r30h_20261102/runs')
PARENT_DIR = PARENT_RUNS / 'r37s_cc_s1001_H0'
PACKAGE = Path('/home/infres/yinwang/CS_QMI/r38_plan/VCS_R37_Review_R38_Plan_20261102')

N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP = 150000, 150000, 1000, 1000
PER_CAP = 1          # captions per image (FIT and DEV) in CC152K
PEERS = ('A', 'B')
CKPT_STEPS_ABC = (0, 250, 500, 1000, 2000, 4000, 8000)

# single condition: the R37S R30-style H parent, seed 1001, CC152K
SERIES = 'R37S R30-style H adaptation on CC152K (warmup 5 + H0 co-train 40, A1, K5, i2t/t2i on, DEV-best); NOT the R31 IN2R-default series'
PARENT_RECIPE = dict(warmup_epoch=5, k=5, i2t=True, t2i=True, arm='a1_iface', num_epochs=40)
CONDS = {
    'cc_r37s': dict(dataset='CC152K', noise_percent=None, noise='real (uninjected); observed relation = native identity', parent_seed=1001,
                    h_unit='r37s_cc_s1001_H0', parent_dir=str(PARENT_DIR), run_json='RUN_1082196.json', noise_ratio=0.0, series=SERIES,
                    cache_dir=str(OUT / 'cache' / 'cc_r37s'), prep_dir=str(PREP / 'cc_r37s'), prep_key='cc_r37s', trust_file='trust_cc_r37s.npz'),
}


def sha256(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# --------------------------------------------------------------------------- TEST guard (as R34)
_TEST_RE = re.compile(r'^test(all|5k)?_')
GUARD = {'installed': False, 'blocked_attempts': []}


def _is_test_data(path) -> bool:
    if isinstance(path, (bytes, bytearray)):
        path = path.decode(errors='ignore')
    if not isinstance(path, str):
        return False
    p = os.path.abspath(path)
    return bool(_TEST_RE.match(os.path.basename(p))) and '_precomp' in os.path.dirname(p)


_FORBIDDEN_PREFIXES = []   # role-dependent (install_guards): trainer -> oracle_runs + audit; oracle -> audit


def _hook(event, args):
    if event == 'open' and args:
        if _is_test_data(args[0]):
            GUARD['blocked_attempts'].append(str(args[0]))
            raise PermissionError(f'R38 TEST guard: refusing to open {args[0]}')
        a0 = args[0]
        if isinstance(a0, (bytes, bytearray)):
            a0 = a0.decode(errors='ignore')
        if isinstance(a0, str) and _FORBIDDEN_PREFIXES:
            ap = os.path.abspath(a0)
            for pre in _FORBIDDEN_PREFIXES:
                if ap == pre or ap.startswith(pre + os.sep):
                    GUARD.setdefault('isolation_refusals', []).append(ap)
                    raise PermissionError(f'R38 isolation guard ({GUARD.get("role")}): refusing to open {ap}')


# --------------------------------------------------------------------------- oracle isolation
ORACLE_MODULE = 'oracle_truth'
ORACLE_ENV = 'R38_ORACLE_PROCESS'


EVAL_ONLY_MODULES = ('audit_export', 'audit_candidate_lists')


class _OracleBlocker(importlib.abc.MetaPathFinder):
    """Refuses to import the oracle-truth module in any process not started by train_oracle.py, and the
    evaluator-only audit modules in every trainer/oracle process."""
    def find_spec(self, name, path=None, target=None):
        leaf = name.split('.')[-1]
        if leaf == ORACLE_MODULE and os.environ.get(ORACLE_ENV) != '1':
            GUARD.setdefault('oracle_import_refusals', []).append(name)
            raise ImportError(f'R38 oracle isolation: {name} may only be imported by the oracle runner process')
        if leaf in EVAL_ONLY_MODULES and GUARD.get('role') in ('trainer', 'oracle'):
            GUARD.setdefault('oracle_import_refusals', []).append(name)
            raise ImportError(f'R38 isolation: evaluator-only module {name} refused in a {GUARD.get("role")} process')
        return None


def install_guards(role='trainer'):
    """role: trainer (T/N/V/M-RANDOM: refuses opening OUT/oracle_runs and OUT/audit), oracle (refuses OUT/audit),
    audit / gate (TEST guard only)."""
    if not GUARD['installed']:
        GUARD['role'] = role
        if role == 'trainer':
            _FORBIDDEN_PREFIXES[:] = [str(ORACLE_RUNS), str(OUT / 'audit')]
        elif role == 'oracle':
            _FORBIDDEN_PREFIXES[:] = [str(OUT / 'audit')]
        sys.addaudithook(_hook)
        sys.meta_path.insert(0, _OracleBlocker())
        GUARD['installed'] = True
    return GUARD


def assert_no_oracle_loaded():
    bad = [m for m in sys.modules if m.split('.')[-1] == ORACLE_MODULE]
    if bad:
        raise RuntimeError(f'R38 oracle isolation violated: {bad} loaded in a non-oracle process')


def guard_record():
    return {'installed': GUARD['installed'], 'rule': 'basename ^test(all|5k)?_ inside *_precomp dir -> PermissionError',
            'blocked_attempts': list(GUARD['blocked_attempts']),
            'oracle_import_refusals': list(GUARD.get('oracle_import_refusals', [])),
            'role': GUARD.get('role'), 'forbidden_prefixes': list(_FORBIDDEN_PREFIXES), 'isolation_refusals': list(GUARD.get('isolation_refusals', [])),
            'oracle_process': os.environ.get(ORACLE_ENV) == '1'}


def state_hash(*modules):
    h = hashlib.sha256()
    for m in modules:
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
