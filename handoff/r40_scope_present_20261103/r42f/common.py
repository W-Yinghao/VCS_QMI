"""R42 tree r42f = byte copy of R41-G r41g (R42 edits at the end: runs_r42 / oracle_runs_r42 / r42_parents, PROTOCOL_FROZEN_R42.json, seven new IN2R-native Flickr parents (f20 s1001-3, f40 s1002-3, f80 s1002-3) and their JOINT parents; trainer guard also refuses the R41 oracle dir). R41-G tree r41g = byte copy of R41 r41f with FROZEN_FILES_R41 -> (PROTOCOL_FROZEN_R41G.json,) (amendment; r41f stays bound to the R41 part 1/2 freeze). R41 tree r41f = byte copy of R40 r40f (R41 edits: RUNS/ORACLE_RUNS/FROZEN_FILE -> runs_r41 / oracle_runs_r41 / PROTOCOL_FROZEN_R41*.json, trainer guard also refuses the R40 oracle dirs, R41 conditions (R39X n00, IN2R-native Flickr parents f40in / f80in) and per-cond JOINT parents at the end). R40 tree r40f = byte copy of R39 r39f (R40 edits: ROOT/OUT/RUNS/ORACLE_RUNS to the R40 dirs, R39_PREP pinned to the R39 output tree (read-only), trainer guard also refuses the R39 oracle/audit outputs, R40 constants at the end). R39 Flickr tree r39f = byte copy of R38 r38f (R39 edits: ROOT/OUT/R38_OUT/R39_PREP paths, oracle env name, R39 dataset/joint constants at the end). R38 (Flickr code tree r38f = byte copy of R37 r37; R38 edits: ROOT/OUT, PACKAGE, CONDS = the R36 parent n60
(R30-H f60 s1001: R34 cache, R35 prep, read-only) + (user amendment) the R37 parents n60s1002/n60s1003 (R37 cache/prep, read-only)). R37 CONFIRM: shared paths, constants, hashing, TEST-file guard, isolation guard. Copied from R36 common.py
(paths/conditions changed); R34 caches, R35 prep and all R30-R36 files are read-only and never written."""
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
            _FORBIDDEN_PREFIXES[:] = [str(ORACLE_RUNS), str(OUT / 'oracle_runs'), str(OUT / 'oracle_runs_r41'), str(OUT / 'audit'), str(R39_OUT / 'oracle_runs'), str(R39_OUT / 'audit'), str(OUT / 'diag7' / 'oracle')]   # R40: + R39 oracle/audit outputs; R41: + R40 oracle_runs
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


# ----------------------------------------------------------------------------- R40 constants (update scope x incomplete-correspondence supervision)
R40_PACKAGE = Path('/home/infres/yinwang/CS_QMI/r40_plan/VCS_R39_Review_R40_Plan_20261103')   # R40 design package (read-only)
FROZEN_FILE = 'PROTOCOL_FROZEN.json'      # R40 protocol (ROOT/PROTOCOL_FROZEN.json); the R39/R39X freeze files are never read for binding
ADAPTATIONS = {'frozen': (), 'image_only': ('img',), 'text_only': ('txt',), 'both_joint': ('img', 'txt')}


# ----------------------------------------------------------------------------- R41 (budget / variance / parent-recipe / oracle / image_only extension)
RUNS = OUT / 'runs_r41'
ORACLE_RUNS = OUT / 'oracle_runs_r41'
FROZEN_FILE = 'PROTOCOL_FROZEN_R41.json'
FROZEN_FILES_R41 = ('PROTOCOL_FROZEN_R41G.json',)   # R41-G amendment freeze (r41g tree)   # part 1 (A/B/D/E), part 2 (C: new IN2R-native parents, frozen after their prep)
R41_PARENTS = OUT / 'r41_parents'            # caches / prep / JOINT prep of the new IN2R-native Flickr parents (written once by tools/r41_build_parent.py)
CODE_IN2R = Path('/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101/code')   # R25 IN2R code copy (R31; R33 code_h0 is byte-identical in vse/data/evaluation)
R39_RUNS_R39X = R39_OUT / 'runs_r39x'
# R39X capability parent (copied from the R39X tree r39xf): Flickr 0% R30-H f00 s1001 (R34 cache, R35 prep; no B-TRUST file)
CONDS['n00'] = dict(noise_percent=0, parent_seed=1001, h_unit='r30_f00_s1001_H0', run_json='RUN_1080229.json',
                    noise_file='/projects/EEG-foundation-model/yinghao/outputs_v9/r29_listwise_20261101/noise/f30k_precomp_0.0.npy',
                    noise_sha256='311f1f81f192c8ea213cb734bd7270917179193f249e42e846ba48901b8f5153', noise_ratio=0.0,
                    cache_dir=str(R34_OUT / 'cache' / 'n00'), prep_dir=str(R35_PREP), prep_key='n00', dev_lock='n00', trust_file=None)
# R41 C: IN2R-native Flickr parents (same family as the CC31 parent; author run.py defaults for f40, R25 code configuration for f80).
# DEV-observed lock 'n00' (identity) is loaded but unused by the N / JOINT route (registered); no B-TRUST (N route only).
CONDS['f40in'] = dict(noise_percent=40, parent_seed=1001, h_unit='r31_f40_s1001_H0_default', run_json='RUN_1080763.json',
                      parent_dir='/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101/runs/f40_s1001/H0_default',
                      noise_file='/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy',
                      noise_sha256='4926687360301d9052d348c2a75776915e4827bfb2d9c06dcf728dfa704bf739', noise_ratio=0.4,
                      cache_dir=str(R41_PARENTS / 'cache' / 'f40in'), prep_dir=str(R41_PARENTS / 'prep'), prep_key='f40in', dev_lock='n00', trust_file=None,
                      code=str(CODE_IN2R))
CONDS['f80in'] = dict(noise_percent=80, parent_seed=1001, h_unit='r33_f80_s1001_H0', run_json='RUN_1081030.json',
                      parent_dir='/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/runs/f80_s1001/H0',
                      noise_file='/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/noise/f30k_precomp_0.8.npy',
                      noise_sha256='d4de36b7e05cd254f910c110adeb2a72e2a4aa606924d1c049128ba8ae7a68dc', noise_ratio=0.8,
                      cache_dir=str(R41_PARENTS / 'cache' / 'f80in'), prep_dir=str(R41_PARENTS / 'prep'), prep_key='f80in', dev_lock='n00', trust_file=None,
                      code=str(CODE_IN2R))
JOINT_PARENTS = {   # per-cond JOINT parent: checkpoint, JOINT prep dir (centring vectors + manifest), native code dir
    'n60': dict(ckpt=PARENT_CKPT, prep_dir=R39_PREP / 'f60', code=CODE_R30),
    'n60s1002': dict(ckpt=R30_RUNS / 'r30v11_f60_s1002_HH' / 'model_best.pth.tar', prep_dir=R39_PREP / 'f60_s1002', code=CODE_R30),
    'n60s1003': dict(ckpt=R30_RUNS / 'r30v11_f60_s1003_HH' / 'model_best.pth.tar', prep_dir=R39_PREP / 'f60_s1003', code=CODE_R30),
    'n00': dict(ckpt=R30_RUNS / 'r30_f00_s1001_H0' / 'model_best.pth.tar', prep_dir=R39_PREP / 'f00_s1001', code=CODE_R30),
    'f40in': dict(ckpt=Path(CONDS['f40in']['parent_dir']) / 'model_best.pth.tar', prep_dir=R41_PARENTS / 'joint' / 'f40in', code=CODE_IN2R),
    'f80in': dict(ckpt=Path(CONDS['f80in']['parent_dir']) / 'model_best.pth.tar', prep_dir=R41_PARENTS / 'joint' / 'f80in', code=CODE_IN2R),
}


# ----------------------------------------------------------------------------- R42 (noise dose-response of JOINT on IN2R-native Flickr parents)
RUNS = OUT / 'runs_r42'
ORACLE_RUNS = OUT / 'oracle_runs_r42'
FROZEN_FILE = 'PROTOCOL_FROZEN_R42.json'
FROZEN_FILES_R42 = ('PROTOCOL_FROZEN_R42.json', 'PROTOCOL_FROZEN_R42B.json')   # part 1 (42 units) + part 2 (f40 s1003 cells, frozen after that parent's build)
FROZEN_FILES_R41 = FROZEN_FILES_R42   # name used by train_unit.check_frozen
R42_PARENTS = OUT / 'r42_parents'
_R31 = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101/runs')
_R33 = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/runs')
_NOISE = {20: ('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/noise_index/f30k_precomp_0.2.npy', 'cdd8b019e69dc1341316fb089a1d9e2c32ca14a62cdd48d72c6af49ab0120f50'),
          40: (CONDS['f40in']['noise_file'], CONDS['f40in']['noise_sha256']), 80: (CONDS['f80in']['noise_file'], CONDS['f80in']['noise_sha256'])}
for _key, _pct, _seed, _pdir, _run, _unit in (
        ('f20in_s1001', 20, 1001, _R31 / 'f20_s1001' / 'H0_codecfg', 'RUN_1080760.json', 'r31_f20_s1001_H0_codecfg'),
        ('f20in_s1002', 20, 1002, _R31 / 'f20_s1002' / 'H0_codecfg', 'RUN_1080761.json', 'r31_f20_s1002_H0_codecfg'),
        ('f20in_s1003', 20, 1003, _R31 / 'f20_s1003' / 'H0_codecfg', 'RUN_1080762.json', 'r31_f20_s1003_H0_codecfg'),
        ('f40in_s1002', 40, 1002, _R31 / 'f40_s1002' / 'H0_default', 'RUN_1080764.json', 'r31_f40_s1002_H0_default'),
        ('f40in_s1003', 40, 1003, _R31 / 'f40_s1003' / 'H0_default', 'RUN_1080765.json', 'r31_f40_s1003_H0_default'),
        ('f80in_s1002', 80, 1002, _R33 / 'f80_s1002' / 'H0', 'RUN_1081032.json', 'r33_f80_s1002_H0'),
        ('f80in_s1003', 80, 1003, _R33 / 'f80_s1003' / 'H0', 'RUN_1081034.json', 'r33_f80_s1003_H0')):
    CONDS[_key] = dict(noise_percent=_pct, parent_seed=_seed, h_unit=_unit, run_json=_run, parent_dir=str(_pdir), noise_file=_NOISE[_pct][0], noise_sha256=_NOISE[_pct][1],
                       noise_ratio=_pct / 100, cache_dir=str(R42_PARENTS / 'cache' / _key), prep_dir=str(R42_PARENTS / 'prep' / _key), prep_key=_key, dev_lock='n00', trust_file=None,
                       code=str(CODE_IN2R))
    JOINT_PARENTS[_key] = dict(ckpt=_pdir / 'model_best.pth.tar', prep_dir=R42_PARENTS / 'joint' / _key, code=CODE_IN2R)
