"""R34 DIRECT_CRITIC_RETRIEVAL_v1: shared paths, hashing, TEST-file guard."""
from __future__ import annotations
import hashlib, json, os, re, sys, time
from pathlib import Path

ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102')
CODE_R30 = Path('/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/code_r30v11')
DATA_SRC = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp')
VOCAB = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab')
R30_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101/runs')
PACKAGE = Path('/home/infres/yinwang/CS_QMI/r30_r32_review/VCS_R30_R32_Independent_Review_20261102')

N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP = 29000, 145000, 1014, 5070

CONDS = {
    'n00': dict(noise_percent=0, h_unit='r30_f00_s1001_H0', run_json='RUN_1080229.json',
                noise_file='/projects/EEG-foundation-model/yinghao/outputs_v9/r29_listwise_20261101/noise/f30k_precomp_0.0.npy',
                noise_sha256='311f1f81f192c8ea213cb734bd7270917179193f249e42e846ba48901b8f5153', noise_ratio=0.0),
    'n60': dict(noise_percent=60, h_unit='r30_f60_s1001_H0', run_json='RUN_1080232.json',
                noise_file='/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.6.npy',
                noise_sha256='d81939538fb43272bcb7aae6e408ded86c224ba9175896c3d786cbe895d42cd1', noise_ratio=0.6),
}
PEERS = ('A', 'B')


def sha256(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# --------------------------------------------------------------------------- TEST guard
# Blocks opening of any dataset split file named test* (incl. its all/5k variants) inside a *_precomp
# directory (any copy, staged or not). Functional: an attempt raises PermissionError and is counted.
_TEST_RE = re.compile(r'^test(all|5k)?_')
GUARD = {'installed': False, 'blocked_attempts': []}


def _is_test_data(path) -> bool:
    if isinstance(path, (bytes, bytearray)):
        path = path.decode(errors='ignore')
    if not isinstance(path, str):
        return False
    p = os.path.abspath(path)
    return bool(_TEST_RE.match(os.path.basename(p))) and '_precomp' in os.path.dirname(p)


def _hook(event, args):
    if event == 'open' and args and _is_test_data(args[0]):
        GUARD['blocked_attempts'].append(str(args[0]))
        raise PermissionError(f'R34 TEST guard: refusing to open {args[0]}')


def install_test_guard():
    if not GUARD['installed']:
        sys.addaudithook(_hook)
        GUARD['installed'] = True
    return GUARD


def guard_record():
    return {'installed': GUARD['installed'], 'rule': 'basename ^test(all|5k)?_ inside *_precomp dir -> PermissionError',
            'blocked_attempts': list(GUARD['blocked_attempts'])}
