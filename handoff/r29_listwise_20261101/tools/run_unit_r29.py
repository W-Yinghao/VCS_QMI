#!/usr/bin/env python3
"""R29 unit runner (listwise-competition VCS main alignment; copy of the R28 runner pattern). Prints the exact driver
command; executes only with --execute and a passed R29 GPU gate whose code sha256s match code_r29 and whose protocol is
frozen. Never submits SLURM jobs. Each unit starts from the common untrained seed-determined init (no prefix); resume only
from this unit's own state_last.pt (exclusive lock + R29_IDENTITY.json; the state's own opt must name the same arm, seed,
noise ratio and noise file). A state from another arm/cell or an unidentified state is refused."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, shlex, subprocess, sys

ARMS = ('h_native', 'n_infonce', 'p_pointwise_vcs', 'l_listwise_vcs', 'b_listwise_bce')
CELLS = {'f00_s1001': (0.0, 1001), 'f60_s1001': (0.6, 1001), 'f60_s1002': (0.6, 1002), 'f60_s1003': (0.6, 1003)}
AUTH = [(c, a) for c in ('f00_s1001', 'f60_s1001') for a in ARMS] + \
       [(c, a) for c in ('f60_s1002', 'f60_s1003') for a in ('h_native', 'n_infonce', 'l_listwise_vcs', 'b_listwise_bce')]
UNITS = {f'{c}__{a}': dict(unit_id=f'{c}__{a}', cell=c, arm=a, noise_ratio=CELLS[c][0], seed=CELLS[c][1],
                           warmup_epoch=5, num_epochs=40, k=5, init='common_untrained') for c, a in AUTH}
assert len(UNITS) == 18


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity_for(unit, cells, code_dir):
    if unit not in UNITS:
        raise SystemExit('Unit is not in the authorized R29 18-unit matrix.')
    u = UNITS[unit]
    c = json.loads(Path(cells).read_text())[u['cell']]
    if (c['data_name'], int(c['seed']), float(c['noise_ratio'])) != ('f30k_precomp', u['seed'], u['noise_ratio']):
        raise SystemExit('CELLS identity mismatch.')
    code = Path(code_dir).resolve()
    im = json.loads((code / 'INSTALL_MANIFEST.json').read_text())
    for f, h in im['files_sha256'].items():
        if digest(code / f) != h:
            raise SystemExit('Installed source changed: ' + f)
    noise = Path(c['noise_file']).resolve()
    if not noise.is_file():
        raise SystemExit('Missing noise file: ' + str(noise))
    if u['noise_ratio'] == 0.0:  # identity mapping: must equal data.py's own t2i_index (data.py does not load the file at ratio 0)
        import numpy as np
        a = np.load(noise)
        if not (a.dtype == np.int64 and np.array_equal(a, np.arange(a.shape[0]) // 5)):
            raise SystemExit('0% noise file is not the identity mapping.')
    pf = code.parent / 'PROTOCOL_FROZEN.json'
    ident = {'unit': u, 'noise_file': str(noise), 'noise_sha256': digest(noise), 'source_files_sha256': im['files_sha256'],
             'protocol_frozen_sha256': digest(pf) if pf.is_file() else None}
    return u, c, im, noise, ident


def check_state(path, u, noise):
    import torch
    st = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    o = st['opt']
    want = dict(r29_arm=u['arm'], seed=u['seed'], noise_ratio=u['noise_ratio'], noise_file=str(noise), num_epochs=u['num_epochs'],
                warmup_epoch=u['warmup_epoch'], k=u['k'], max_iters=0)
    bad = {k: (o.get(k), v) for k, v in want.items() if o.get(k) != v}
    if bad:
        raise SystemExit(f'Cannot resume: state_last.pt belongs to another unit/config {bad}')
    return int(st['epoch'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('unit', 'cells', 'code-dir', 'data-path', 'output-root'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--python', default=sys.executable); p.add_argument('--execute', action='store_true')
    p.add_argument('--gate-report'); a = p.parse_args()
    u, c, im, noise, identity = identity_for(a.unit, a.cells, a.code_dir)
    code = Path(a.code_dir).resolve()
    out = Path(a.output_root).resolve() / u['cell'] / u['arm']
    identity_file = out / 'R29_IDENTITY.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise SystemExit('Cannot resume: identity mismatch.')
    last = out / 'state_last.pt'
    if last.exists() and not identity_file.exists():
        raise SystemExit('Unidentified state in output directory; refuse resume.')
    cmd = [a.python, '-u', '-B', str(code / 'r29_run.py'), '--r29_arm', u['arm'],
           '--data_path', a.data_path, '--data_name', c['data_name'], '--noise_file', str(noise),
           '--noise_ratio', str(u['noise_ratio']), '--seed', str(u['seed']), '--k', '5', '--num_epochs', '40',
           '--warmup_epoch', '5', '--prefix_epoch', '-1', '--output_dir', str(out)]
    if last.exists():
        cmd += ['--resume', str(last)]
    print(shlex.join(cmd), flush=True)
    if not a.execute:
        return
    if not a.gate_report:
        raise SystemExit('--gate-report required for execution.')
    gate = json.loads(Path(a.gate_report).read_text())
    pf = code.parent / 'PROTOCOL_FROZEN.json'
    if not (gate.get('all_pass') is True and gate.get('code_files_sha256') == im['files_sha256'] and gate.get('config_frozen') is True
            and pf.is_file() and gate.get('protocol_frozen_sha256') == digest(pf)):
        raise SystemExit('GPU gate missing PASS, not frozen, or does not match this source snapshot / protocol.')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.running.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Same unit already running.')
        if (out / 'DONE.json').exists():
            print('Already complete; no duplicate run.'); return
        resumed_epoch = check_state(last, u, noise) if last.exists() else None
        if not identity_file.exists():
            identity_file.write_text(json.dumps(identity, indent=2) + '\n')
        with (out / 'R29_ATTEMPTS.jsonl').open('a') as f:
            f.write(json.dumps({'job': os.environ.get('SLURM_JOB_ID'), 'command': cmd, 'resume_epoch': resumed_epoch}) + '\n')
        result = subprocess.run(cmd, cwd=code)
        if result.returncode:
            raise SystemExit(result.returncode)
        if not (out / 'DONE.json').exists():
            raise SystemExit('Driver exited without DONE; inspect, do not mark complete.')


if __name__ == '__main__':
    main()
