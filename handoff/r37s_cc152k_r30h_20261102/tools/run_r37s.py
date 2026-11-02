#!/usr/bin/env python3
"""R37S runner: CC152K seed1001 R30-style H parent = R30 v1.1 recipe, two stages in the R30 layout (same driver code
r30_run.py / r30_co_train.py of code_r37s = code_r30v11 + the registered CC152K data adaptation, R37S_ADAPTATION.json):
  stage parent : 5 native-H warmup epochs from the seed-determined untrained init -> parents/wp_cc_s1001_H_e5/parent_state.pt
                 (read-only) + PARENT.json (R30 boundary; built once, never overwritten)
  stage unit   : H0 continuation (objective H, scale native, A1 noisy interface, K 5, i2t+t2i on), 40 co-train epochs,
                 mode fork from the parent (sha256 + fingerprint) unless this unit's own state_last.pt exists (mode resume,
                 R37S_IDENTITY.json must match). DEV-best rule (PROTOCOL.json): model_best.pth.tar = co-train epoch with the
                 strictly highest full-DEV rSum (first maximum), native r25_co_train rule; warmup epochs never eligible.
Checks before every stage: installed code == INSTALL_MANIFEST.json, PROTOCOL.json frozen and its sha256 recorded,
hardware block (GPU_BLOCK.json: the GPU name of the first attempt; every later attempt must run on the same GPU type).
Prints the command; runs only with --execute. Never submits SLURM jobs and never reads TEST."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, shlex, subprocess, sys, time

PARENT_ID, UNIT_ID = 'wp_cc_s1001_H_e5', 'r37s_cc_s1001_H0'
DATA_NAME, SEED, WARMUP_EPOCH, K, EPOCHS, MILESTONE = 'cc152k_precomp', 1001, 5, 5, 40, 10
NOISE_FILE = 'none'   # CC152K: real observed pairs, no injected noise (r25_run forces noise_ratio 0)


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def checks(a):
    code = Path(a.code_dir).resolve()
    im = json.loads((code / 'INSTALL_MANIFEST.json').read_text())
    for f, h in im['files_sha256'].items():
        if digest(code / f) != h:
            raise SystemExit('Installed source changed: ' + f)
    if sorted(im['files_sha256']) != sorted(p.name for p in code.glob('*.py')):
        raise SystemExit('Installed source set differs from the manifest.')
    root = Path(a.root).resolve()
    pr = json.loads((root / 'PROTOCOL.json').read_text())
    if pr.get('frozen') is not True or pr.get('code_files_sha256') != im['files_sha256']:
        raise SystemExit('PROTOCOL.json not frozen or bound to another code snapshot.')
    return root, code, im, digest(root / 'PROTOCOL.json')


def gpu_block(out_base, python):
    # device name as torch reports it (the driver's --r30_require_gpu compares the same string); separate process, so this
    # runner holds no CUDA context during training
    name = subprocess.run([python, '-c', 'import torch; print(torch.cuda.get_device_name(0))'], check=True,
                          capture_output=True, text=True).stdout.strip()
    f = Path(out_base) / 'GPU_BLOCK.json'
    if f.exists():
        want = json.loads(f.read_text())['gpu']
        if want != name:
            raise SystemExit(f'GPU {name} != hardware block {want} of the first attempt')
    else:
        f.write_text(json.dumps({'gpu': name, 'job': os.environ.get('SLURM_JOB_ID'), 'host': os.uname().nodename,
                                 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=1) + '\n')
    return name


def base_args(a, out, num_epochs):
    return ['--data_path', a.data_path, '--data_name', DATA_NAME, '--noise_file', NOISE_FILE, '--noise_ratio', '0.0',
            '--seed', str(SEED), '--k', str(K), '--num_epochs', str(num_epochs), '--warmup_epoch', str(WARMUP_EPOCH),
            '--prefix_epoch', '-1', '--output_dir', str(out)]


def run_locked(out, cmd, cwd, record):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.running.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Same stage already running.')
        if record():
            return
        r = subprocess.run(cmd, cwd=cwd)
        if r.returncode:
            raise SystemExit(r.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('parent', 'unit'))
    for n in ('root', 'code-dir', 'data-path', 'out-base'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    root, code, im, psha = checks(a)
    job = os.environ.get('SLURM_JOB_ID')
    gpu = gpu_block(a.out_base, a.python) if a.execute else ''
    pdir = Path(a.out_base) / 'parents' / PARENT_ID

    if a.stage == 'parent':
        cmd = [a.python, '-u', '-B', str(code / 'r30_run.py'), '--r30_stage', 'warmup', '--r30_parent_id', PARENT_ID,
               '--r30_source_warmup', 'H', '--r30_require_gpu', gpu] + base_args(a, pdir, EPOCHS)
        print(shlex.join(cmd), flush=True)
        if not a.execute:
            return

        def rec():
            if (pdir / 'PARENT.json').exists():
                print('Parent already built; no duplicate warmup.'); return True
            if (pdir / 'parent_state.pt').exists():
                raise SystemExit('parent_state.pt without PARENT.json; inspect, do not overwrite.')
            with (pdir / 'R37S_ATTEMPTS.jsonl').open('a') as f:
                f.write(json.dumps({'job': job, 'command': cmd, 'protocol_sha256': psha}) + '\n')
            return False
        run_locked(pdir, cmd, code, rec)
        if not (pdir / 'PARENT.json').exists():
            raise SystemExit('Warmup exited without PARENT.json.')
        return

    # ---- unit (H0 continuation)
    prec = json.loads((pdir / 'PARENT.json').read_text())
    if digest(prec['state_path']) != prec['sha256'] or prec['code_sha256'] != im['files_sha256']:
        raise SystemExit('Parent file sha256 / code identity differs from PARENT.json or this snapshot.')
    if prec['max_iters'] != 0 or len(prec['warmup_dev']) != WARMUP_EPOCH or prec['gpu'] != gpu:
        raise SystemExit('Parent is not a full 5-epoch warmup on this hardware block.')
    out = Path(a.out_base) / 'runs' / UNIT_ID
    identity = {'unit_id': UNIT_ID, 'parent_id': PARENT_ID, 'parent_state': prec['state_path'], 'parent_sha256': prec['sha256'],
                'parent_fingerprint': prec['training_state_fingerprint'], 'objective': 'H', 'scale': 'native', 'co_train_epochs': EPOCHS,
                'source_files_sha256': im['files_sha256'], 'protocol_sha256': psha, 'block_gpu': gpu, 'milestone': MILESTONE,
                'data_name': DATA_NAME, 'seed': SEED, 'k': K}
    idf = out / 'R37S_IDENTITY.json'
    if idf.exists() and json.loads(idf.read_text()) != identity:
        raise SystemExit('Cannot resume: identity mismatch.')
    last = out / 'state_last.pt'
    if last.exists() and not idf.exists():
        raise SystemExit('Unidentified state in output directory; refuse resume.')
    if (out / f'state_completed{MILESTONE}.pt').exists() and not (out / f'E{MILESTONE}_RECORD.json').exists():
        raise SystemExit('Completed-epoch checkpoint without its record (interrupted milestone); inspect.')
    mode = 'resume' if last.exists() else 'fork'
    cmd = [a.python, '-u', '-B', str(code / 'r30_run.py'), '--r30_stage', 'unit', '--r30_unit_id', UNIT_ID,
           '--r30_source_warmup', 'H', '--r30_objective', 'H', '--r30_scale', 'native', '--r30_mode', mode,
           '--r30_parent', prec['state_path'], '--r30_parent_sha', prec['sha256'], '--r30_parent_id', PARENT_ID,
           '--r30_parent_fingerprint', prec['training_state_fingerprint'], '--r30_milestone', str(MILESTONE),
           '--r30_require_gpu', gpu] + base_args(a, out, EPOCHS)
    if mode == 'resume':
        cmd += ['--resume', str(last)]
    print(shlex.join(cmd), flush=True)
    if not a.execute:
        return

    def rec():
        if (out / 'DONE.json').exists():
            print('Already complete; no duplicate run.'); return True
        ep = None
        if last.exists():
            import torch
            st = torch.load(last, map_location='cpu', mmap=True, weights_only=False)
            o = st['opt']
            want = dict(r30_stage='unit', r30_unit_id=UNIT_ID, r30_parent_sha=prec['sha256'], r30_parent_fingerprint=prec['training_state_fingerprint'],
                        r30_objective='H', r30_scale='native', seed=SEED, noise_ratio=0, num_epochs=EPOCHS, warmup_epoch=WARMUP_EPOCH,
                        k=K, max_iters=0, r30_milestone=MILESTONE, data_name=DATA_NAME)
            bad = {k: (o.get(k), v) for k, v in want.items() if o.get(k) != v}
            if bad:
                raise SystemExit(f'Cannot resume: state_last.pt belongs to another unit/config {bad}')
            ep = int(st['epoch'])
        if not idf.exists():
            idf.write_text(json.dumps(identity, indent=2) + '\n')
        with (out / 'R37S_ATTEMPTS.jsonl').open('a') as f:
            f.write(json.dumps({'job': job, 'mode': mode, 'command': cmd, 'resume_epoch': ep}) + '\n')
        return False
    run_locked(out, cmd, code, rec)
    if not (out / 'DONE.json').exists():
        raise SystemExit('Driver exited without DONE; inspect, do not mark complete.')


if __name__ == '__main__':
    main()
