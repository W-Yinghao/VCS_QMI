"""R41-F tree r41c = byte copy of R40 r40c (R41-F edits: RUNS/ORACLE_RUNS -> runs_r41 / oracle_runs_r41, freeze file PROTOCOL_FROZEN_R41F.json, trainer guard also refuses the R40 oracle dirs). R40 tree r40c = byte copy of R39 r39c (R40 edits: ROOT/OUT/RUNS/ORACLE_RUNS to the R40 dirs, R39_PREP pinned to the R39 output tree (read-only), trainer guard also refuses the R39 oracle/audit outputs, R40 constants at the end). R39 CC152K tree r39c = byte copy of R38 r38c (R39 edits: ROOT/OUT/R38_OUT/R39_PREP paths, R38 cc_s1002/1003 cache/prep paths made absolute to the R38 output tree (read-only), oracle env name, R39 dataset/joint constants at the end). R38 (CC152K code tree r38c = byte copy of R37C r37c; R38 edits: ROOT/OUT, three R31 parents s1001/s1002/s1003, per-parent
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

ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103')
R39_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')   # R39 (read-only)
R38_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R39_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')   # R39 outputs (read-only)
R38_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102')
R39_PREP = R39_OUT / 'prep'   # R40: the R39 prep is reused read-only   # R39 prep: completion tables (teacher cross-check), JOINT centering vectors (+ CC152K tokens)
R37C_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102')
R37C_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102')
RUNS = OUT / 'runs'
ORACLE_RUNS = OUT / 'oracle_runs'   # no oracle units; kept so the trainer-role guard still refuses this tree
PREP = R38_OUT / 'prep'             # R38 (read-only): per-parent prep dirs OUT/prep/<cond> for the new parents (s1001: R37C prep, read-only). R37C prep: record locks (rebuilt for 150000 records), calibration, top-32 tables, B-TRUST
R37_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102')
R31_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101')
CODE_NATIVE = R31_ROOT / 'code'     # byte copy of R25 IN2R code (author IN2R-code 909e5329); the parent was trained with it
DATA_NAME = 'cc152k_precomp'
DATA_SRC = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data') / DATA_NAME
VOCAB = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab')
R31_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101/runs')
PARENT_DIR = R31_RUNS / 'cc_s1001' / 'H0_default'
PACKAGE = Path('/home/infres/yinwang/CS_QMI/r39_design/VCS_R38_Review_R39_Design_20261102')   # R39 package (read-only)

N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP = 150000, 150000, 1000, 1000
PER_CAP = 1          # captions per image (FIT and DEV) in CC152K
PEERS = ('A', 'B')
CKPT_STEPS_ABC = (0, 250, 500, 1000, 2000, 4000, 8000)

# single condition: the R31 native IN2R parent (author run.py defaults: warmup 1, K 1, i2t/t2i off), seed 1001, CC152K
CONDS = {
    'cc': dict(dataset='CC152K', noise_percent=None, noise='real (uninjected); observed relation = native identity', parent_seed=1001,
               h_unit='r31_cc_s1001_H0_default', parent_dir=str(PARENT_DIR), run_json='RUN_1080782.json', noise_ratio=0.0,
               cache_dir=str(R37C_OUT / 'cache' / 'cc'), prep_dir=str(R37C_OUT / 'prep'), prep_key='cc', trust_file='trust_cc.npz'),
    'cc_s1002': dict(dataset='CC152K', noise_percent=None, noise='real (uninjected); observed relation = native identity', parent_seed=1002,
                     h_unit='r31_cc_s1002_H0_default', parent_dir=str(R31_RUNS / 'cc_s1002' / 'H0_default'), run_json='RUN_1080783.json', noise_ratio=0.0,
                     cache_dir=str(R38_OUT / 'cache' / 'cc_s1002'), prep_dir=str(PREP / 'cc_s1002'), prep_key='cc_s1002', trust_file='trust_cc_s1002.npz'),
    'cc_s1003': dict(dataset='CC152K', noise_percent=None, noise='real (uninjected); observed relation = native identity', parent_seed=1003,
                     h_unit='r31_cc_s1003_H0_default', parent_dir=str(R31_RUNS / 'cc_s1003' / 'H0_default'), run_json='RUN_1080784.json', noise_ratio=0.0,
                     cache_dir=str(R38_OUT / 'cache' / 'cc_s1003'), prep_dir=str(PREP / 'cc_s1003'), prep_key='cc_s1003', trust_file='trust_cc_s1003.npz'),
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
            _FORBIDDEN_PREFIXES[:] = [str(ORACLE_RUNS), str(OUT / 'oracle_runs'), str(OUT / 'oracle_runs_r41'), str(OUT / 'oracle_runs_r42'), str(OUT / 'oracle_runs_r43'), str(OUT / 'oracle_runs_r44'), str(OUT / 'oracle_runs_r45'), str(OUT / 'oracle_runs_r46'), str(OUT / 'oracle_runs_r47'), str(OUT / 'audit'), str(R39_OUT / 'oracle_runs'), str(R39_OUT / 'audit'), str(OUT / 'diag7' / 'oracle'), str(OUT / 'r46_parents' / 'truth_eval')]   # R40-R47 oracle / evaluator-only outputs
        elif role == 'oracle':
            _FORBIDDEN_PREFIXES[:] = [str(OUT / 'audit'), str(R39_OUT / 'audit')]
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
TARGET_COND = 'cc'                       # R31 H0_default s1001 (development parent)
TEACHER_CONDS = ('cc_s1002', 'cc_s1003')  # same series (R31 H0_default), fixed by the package (never chosen on DEV)
NATIVE_CODE = CODE_NATIVE                # parent training code (vse.py: EncoderImage / EncoderText / GPO / MLP)
PARENT_CKPT = PARENT_DIR / 'model_best.pth.tar'
TOKENS_FILE = R39_PREP / 'cc' / 'tokens_cc.npz'   # R39 prep: data.process_caption(vocab, caption, drop=False) (native eval tokenisation)
RAW_FIT_IMS = DATA_SRC / 'train_ims.npy'; RAW_DEV_IMS = DATA_SRC / 'dev_ims.npy'
DEV_IMS_ROW_STRIDE = 1                   # DEV image k = dev_ims row k (per_captions 1)
RAW_FIT_ON_GPU = False                   # 150000x36x2048 fp32 = 44 GB: host RAM, per-step gather of the unique batch images
DS_KEY = 'cc'


# ----------------------------------------------------------------------------- R40 constants (update scope x incomplete-correspondence supervision)
R40_PACKAGE = Path('/home/infres/yinwang/CS_QMI/r40_plan/VCS_R39_Review_R40_Plan_20261103')   # R40 design package (read-only)
FROZEN_FILE = 'PROTOCOL_FROZEN.json'      # R40 protocol (ROOT/PROTOCOL_FROZEN.json); the R39/R39X freeze files are never read for binding
ADAPTATIONS = {'frozen': (), 'image_only': ('img',), 'text_only': ('txt',), 'both_joint': ('img', 'txt')}


# ----------------------------------------------------------------------------- R41-F (CC152K encoder step-size variant; amendment to R41)
RUNS = OUT / 'runs_r41'
ORACLE_RUNS = OUT / 'oracle_runs_r41'   # no oracle units in R41-F
FROZEN_FILE = 'PROTOCOL_FROZEN_R41F.json'
FROZEN_FILES_R41 = ('PROTOCOL_FROZEN_R41F.json',)


# ----------------------------------------------------------------------------- R47 (JOINT with layout-free centring: G = fixed global parent means, S = own batch means)
RUNS = OUT / 'runs_r47'
ORACLE_RUNS = OUT / 'oracle_runs_r47'   # no oracle units in R47
FROZEN_FILE = 'PROTOCOL_FROZEN_R47.json'
FROZEN_FILES_R47 = ('PROTOCOL_FROZEN_R47.json',)
FROZEN_FILES_R41 = FROZEN_FILES_R47   # name used by train_unit.check_frozen
R47_PARENTS = OUT / 'r47_parents'
# layout-free parent cache per base parent (tools/r47_build_lf.py): parent encoder outputs centred by ONE mean per split / modality / peer over all items
# (no sequential-batch grouping); prep (calibration, natural top-32) rebuilt on it with the R35 rules; JOINT prep per variant (G: global means; S: own batch means).
LF_BASES = ('cc',)
JOINT_PARENTS = {'cc': dict(ckpt=PARENT_CKPT, prep_dir=R39_PREP / 'cc', code=CODE_NATIVE)}   # R47: per-cond JOINT parents (r42f r39ext); 'cc' = the R39 JOINT prep (read-only)
for _b in LF_BASES:
    for _v in ('G', 'S'):
        CONDS[f'{_b}_lf{_v}'] = dict(CONDS[_b], cache_dir=str(R47_PARENTS / 'cache' / _b), prep_dir=str(R47_PARENTS / 'prep' / _b), prep_key=_b, lf_base=_b, lf_variant=_v)
        JOINT_PARENTS[f'{_b}_lf{_v}'] = dict(JOINT_PARENTS[_b], prep_dir=R47_PARENTS / 'joint' / f'{_b}_{_v}')


# ----------------------------------------------------------------------------- R48 (N-frozen trained on the R47 layout-free G parent cache; tree = byte copy of the R47 tree + this block)
RUNS = OUT / 'runs_r48'
ORACLE_RUNS = OUT / 'oracle_runs_r48'   # no oracle units in R48
FROZEN_FILE = 'PROTOCOL_FROZEN_R48.json'
FROZEN_FILES_R48 = ('PROTOCOL_FROZEN_R48.json',)
FROZEN_FILES_R41 = FROZEN_FILES_R48   # name used by train_unit.check_frozen
