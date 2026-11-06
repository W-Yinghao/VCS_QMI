"""CR6 common paths/constants. Every older tree is imported READ-ONLY (no file in CR1-CR5/NR4 is modified)."""
from __future__ import annotations
import contextlib, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path

TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr6_20261105')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr6/CR6_20261105')
CR1_TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104'); CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
CR2_TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr2_20261104')
CR3_TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr3_20261104'); CR3 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104')
CR4_TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr4_20261104')
CR5_TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr5_20261105'); CR5 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr5/CR5_20261105')
CR5P_TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr5p_20261105'); CR5P = Path('/projects/EEG-foundation-model/yinghao/outputs_cr5p/CR5P1105')
NR1_CODE = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/code/src')
NR2_CODE = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code/src')
NR3_CODE = Path('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/code')
NR4_CODE = Path('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/code')
NR4_RAW = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/raw')
PY = '/home/infres/yinwang/CS_QMI/env_nr1/bin/python'
PROTOCOL_ID = 'CR6_SOURCE_REPLICATE_AND_STUDENT_SCOPE_V1'
EVIDENCE = '05754e6a0e55447a77d776d70a2aefad264b3167'
FROZEN = TREE/'PROTOCOL_FROZEN_CR6.json'
SAVES = (0, 250, 500, 1000, 2000, 4000, 8000)
CONDS = ('Flickr60', 'CC152K_RECOVERED_20261104')
GRAPHS6 = ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1')
EXPERTS = ('E01', 'E02', 'E03', 'E12', 'E13', 'E23')
DATASET = {'Flickr60': 'flickr', 'CC152K_RECOVERED_20261104': 'cc'}
NR1_HOME = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104')
CCD = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104')

for p in (CR1_TREE/'code'/'src', CR1_TREE/'code'/'site', NR1_CODE, NR2_CODE, NR3_CODE, NR4_CODE, CR4_TREE/'code'/'pkg', TREE/'code'/'site'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): h.update(c)
    return h.hexdigest()


def read_json(p): return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str)+'\n'); tmp.replace(p)


def utc(): return time.strftime('%FT%TZ', time.gmtime())


def bank_path(c): return CR1/'banks'/c/'fit_bank.npz'
def prep_dir(c): return CR1/c/'prepared'


def dev_paths(c):
    if c.startswith('CC152K'): return CCD/'cache'/'dev', CCD/'evaluator_only'/'dev_positives.json'
    return NR1_HOME/'cache'/'dev', NR1_HOME/'evaluator_only'/'dev_positives.json'


def gpu_env():
    import torch
    drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.split('\n')[0].strip()
    return {'gpu_name': torch.cuda.get_device_name(), 'driver': drv, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version()}


def job(): return os.environ.get('SLURM_JOB_ID'), socket.gethostname()


def frozen_check():
    """Formal CR6 commands refuse to run unless every frozen code/data hash still matches."""
    if not FROZEN.exists(): raise RuntimeError('PROTOCOL_FROZEN_CR6.json missing: formal units cannot run')
    fz = read_json(FROZEN)
    for rel, h in fz['code_sha256'].items():
        p = Path(rel) if rel.startswith('/') else TREE/rel
        if sha256(p) != h: raise RuntimeError(f'frozen code changed {rel}')
    for p, h in fz['data_sha256'].items():
        if sha256(p) != h: raise RuntimeError(f'frozen data changed {p}')
    return fz


class StopReplay(Exception):
    pass


@contextlib.contextmanager
def stop_after(n):
    """Short-trajectory replay with the PRODUCTION schedule: run the unchanged CR1 optim_loop (T_max = full steps) and stop before
    update n+1 (after step n has been saved). Patches the cr1_prod.optim_loop attribute only inside the block."""
    import cr1_prod as C1
    orig = C1.optim_loop

    def loop(model, step_fn, out, steps, device, rng, binding, log_extra=None):
        c = [0]

        def sf(m):
            c[0] += 1
            if c[0] > n: raise StopReplay()
            return step_fn(m)
        return orig(model, sf, out, steps, device, rng, binding, log_extra)
    C1.optim_loop = loop
    try:
        yield
    except StopReplay:
        pass
    finally:
        C1.optim_loop = orig
