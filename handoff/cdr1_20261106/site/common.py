"""CDR1 site constants and small helpers (shared by all site stages)."""
import hashlib, json, os, stat, sys, time
from pathlib import Path

H = Path('/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106')
PKG = H / 'package'
sys.path.insert(0, str(PKG / 'src'))
OUT = Path(os.environ.get('CDR1_TEST_OUT') or '/projects/EEG-foundation-model/yinghao/outputs_cdr1/CDR1_20261106')
TEST_MODE = bool(os.environ.get('CDR1_TEST_OUT'))
DATA = OUT / 'data'
BANKS = OUT / 'curator_only' / 'banks'
PY = '/home/infres/yinwang/CS_QMI/env_nr1/bin/python'
PROTOCOL_ID = 'CDR1_CONTAMINATION_CALIBRATION_SELECTIVE_RETRIEVAL_V1'
FROZEN = H / 'PROTOCOL_FROZEN_CDR1.json'
DATASETS = ('Flickr30K', 'COCO')
SEEDS = (1001, 2002, 3003)
M_SIZES = (32, 128, 512)
PIS = (.25, .5, .75)
TAUS = (.90, .95)
LAMBDAS = (.5, .1)
RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
EPSILONS = ('none', '0', '0.05', '0.1')
NULL_FAR = (.05, .10)
CC_NAME = 'CC152K_RECOVERED_20261104'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float) + '\n')
    os.replace(tmp, p)


def load(p):
    return json.loads(Path(p).read_text())


def group_dirs(ds, seed):
    return dict(data=DATA / ds / f'd{seed}', models=OUT / 'models' / ds / f'd{seed}',
                read=OUT / 'readouts' / ds / f'd{seed}', d1=OUT / 'd1' / ds / f'd{seed}',
                oracle=OUT / 'oracle' / ds / f'd{seed}', rho=OUT / 'rho' / ds / f'd{seed}',
                receipts=OUT / 'receipts' / ds / f'd{seed}')


def lock_truth(ds_dir, open_=False):
    """Directory-permission separation for evaluator_only/ (fit/predict run with it closed)."""
    ev = Path(ds_dir) / 'evaluator_only'
    if ev.exists():
        os.chmod(ev, stat.S_IRWXU if open_ else 0)


def check_frozen():
    if TEST_MODE:
        return {}
    fz = load(FROZEN)
    for rel, h in fz['site_code_sha256'].items():
        assert sha(H / rel) == h, f'site code changed after freeze: {rel}'
    for rel, h in fz['package_code_sha256'].items():
        assert sha(PKG / rel) == h, f'package code changed: {rel}'
    return fz


class Timer:
    def __init__(self): self.t = {}
    def run(self, name, fn, *a, **k):
        t0 = time.time(); r = fn(*a, **k); self.t[name] = self.t.get(name, 0) + time.time() - t0; return r
