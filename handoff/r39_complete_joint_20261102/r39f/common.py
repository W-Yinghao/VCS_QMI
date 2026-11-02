"""R39 Flickr tree r39f = byte copy of R38 r38f (R39 edits: ROOT/OUT/R38_OUT/R39_PREP paths, oracle env name, R39 dataset/joint constants at the end). R38 (Flickr code tree r38f = byte copy of R37 r37; R38 edits: ROOT/OUT, PACKAGE, CONDS = the R36 parent n60
(R30-H f60 s1001: R34 cache, R35 prep, read-only) + (user amendment) the R37 parents n60s1002/n60s1003 (R37 cache/prep, read-only)). R37 CONFIRM: shared paths, constants, hashing, TEST-file guard, isolation guard. Copied from R36 common.py
(paths/conditions changed); R34 caches, R35 prep and all R30-R36 files are read-only and never written."""
from __future__ import annotations
import hashlib, importlib.abc, json, os, re, sys, time
from pathlib import Path

ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
R38_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
R38_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102')
R39_PREP = OUT / 'prep'   # R39 prep: completion tables (teacher cross-check), JOINT centering vectors (+ CC152K tokens)
RUNS = OUT / 'runs'
ORACLE_RUNS = OUT / 'oracle_runs'   # R37 has no oracle units; kept so the trainer-role guard still refuses this tree
R36_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r36_supervision_20261102')
R35_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102')
R35_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102')
R35_PREP = R35_OUT / 'prep'   # R35 prep (read-only): record locks + tokens (all parents), and the s1001 parent's calibration/tables/trust
PREP = R35_PREP               # alias kept for the copied R35/R36 code paths that read locks/tokens
R37_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102')
R37_PREP = R37_OUT / 'prep'   # R37 prep of parents s1002/s1003 (calibration a0/b0, top-32 tables, B-TRUST subset), read-only
R34_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102')
R34_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102')
CODE_R30 = Path('/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/code_r30v11')
DATA_SRC = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp')
VOCAB = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab')
R30_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101/runs')
PACKAGE = Path('/home/infres/yinwang/CS_QMI/r39_design/VCS_R38_Review_R39_Design_20261102')   # R39 package (read-only)

N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP = 29000, 145000, 1014, 5070
PEERS = ('A', 'B')
CKPT_STEPS_ABC = (0, 250, 500, 1000, 2000, 4000, 8000)
CKPT_STEPS_D = (0, 1000, 2000, 4000)

_N60 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.6.npy'
_N60_SHA = 'd81939538fb43272bcb7aae6e408ded86c224ba9175896c3d786cbe895d42cd1'
# R37 conditions = bound Flickr60 H parents of the R30 H series (a1_iface arm, objective H, 5-epoch H warmup boundary
# -> 40 co-train epochs, DEV-best model_best). 'n60' is the R36 parent (R34 cache, R35 prep reused read-only).
# All three parents trained with the same observed map file (R30 used one 60% map for every seed).
CONDS = {
    'n60': dict(noise_percent=60, parent_seed=1001, h_unit='r30_f60_s1001_H0', run_json='RUN_1080232.json',
                noise_file=_N60, noise_sha256=_N60_SHA, noise_ratio=0.6,
                cache_dir=str(R34_OUT / 'cache' / 'n60'), prep_dir=str(R35_PREP), prep_key='n60', dev_lock='n60', trust_file='trust_n60.npz'),
    # R38 user amendment: the two R37 parents (R37 CONDS entries, cache paths made absolute to the R37 output tree; read-only)
    'n60s1002': dict(noise_percent=60, parent_seed=1002, h_unit='r30v11_f60_s1002_HH', run_json='RUN_1080479.json',
                     noise_file=_N60, noise_sha256=_N60_SHA, noise_ratio=0.6,
                     cache_dir=str(R37_OUT / 'cache' / 'n60s1002'), prep_dir=str(R37_PREP), prep_key='n60s1002', dev_lock='n60', trust_file='trust_n60s1002.npz'),
    'n60s1003': dict(noise_percent=60, parent_seed=1003, h_unit='r30v11_f60_s1003_HH', run_json='RUN_1080481.json',
                     noise_file=_N60, noise_sha256=_N60_SHA, noise_ratio=0.6,
                     cache_dir=str(R37_OUT / 'cache' / 'n60s1003'), prep_dir=str(R37_PREP), prep_key='n60s1003', dev_lock='n60', trust_file='trust_n60s1003.npz'),
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
            raise PermissionError(f'R39 TEST guard: refusing to open {args[0]}')
        a0 = args[0]
        if isinstance(a0, (bytes, bytearray)):
            a0 = a0.decode(errors='ignore')
        if isinstance(a0, str) and _FORBIDDEN_PREFIXES:
            ap = os.path.abspath(a0)
            for pre in _FORBIDDEN_PREFIXES:
                if ap == pre or ap.startswith(pre + os.sep):
                    GUARD.setdefault('isolation_refusals', []).append(ap)
                    raise PermissionError(f'R39 isolation guard ({GUARD.get("role")}): refusing to open {ap}')


# --------------------------------------------------------------------------- oracle isolation
ORACLE_MODULE = 'oracle_truth'
ORACLE_ENV = 'R39_ORACLE_PROCESS'


EVAL_ONLY_MODULES = ('audit_export', 'audit_candidate_lists')


class _OracleBlocker(importlib.abc.MetaPathFinder):
    """Refuses to import the oracle-truth module in any process not started by train_oracle.py, and the
    evaluator-only audit modules in every trainer/oracle process."""
    def find_spec(self, name, path=None, target=None):
        leaf = name.split('.')[-1]
        if leaf == ORACLE_MODULE and os.environ.get(ORACLE_ENV) != '1':
            GUARD.setdefault('oracle_import_refusals', []).append(name)
            raise ImportError(f'R39 oracle isolation: {name} may only be imported by the oracle runner process')
        if leaf in EVAL_ONLY_MODULES and GUARD.get('role') in ('trainer', 'oracle'):
            GUARD.setdefault('oracle_import_refusals', []).append(name)
            raise ImportError(f'R39 isolation: evaluator-only module {name} refused in a {GUARD.get("role")} process')
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
        raise RuntimeError(f'R39 oracle isolation violated: {bad} loaded in a non-oracle process')


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


# ----------------------------------------------------------------------------- R39 constants
PER_CAP = 5                              # Flickr30K: 5 captions per image; FIT caption record k is in native row k, image j in rows 5j..5j+4
TARGET_COND = 'n60'                      # R30-H f60 s1001 (development parent)
TEACHER_CONDS = ('n60s1002', 'n60s1003')  # same series (R30-H f60), fixed by the package (never chosen on DEV)
NATIVE_CODE = CODE_R30                   # parent training code (vse.py: EncoderImage / EncoderText / GPO / MLP)
PARENT_CKPT = R30_RUNS / 'r30_f60_s1001_H0' / 'model_best.pth.tar'
TOKENS_FILE = R35_PREP / 'tokens.npz'    # R35 prep: data.process_caption(vocab, caption, drop=False) (native eval tokenisation)
RAW_FIT_IMS = DATA_SRC / 'train_ims.npy'; RAW_DEV_IMS = DATA_SRC / 'dev_ims.npy'
DEV_IMS_ROW_STRIDE = 5                   # DEV image k = dev_ims row 5k (dev_ims.npy has one row per caption)
RAW_FIT_ON_GPU = True                    # 29000x36x2048 fp32 = 8.5 GB
DS_KEY = 'f60'
