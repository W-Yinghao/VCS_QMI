#!/usr/bin/env python3
"""R30 v1.1 runner for the formal stages (parent state / calibration / branch unit). Prints the exact driver command;
executes only with --execute. Never submits SLURM jobs. Every stage requires: installed code == INSTALL_MANIFEST.json,
v1.1 PROTOCOL_FROZEN.json frozen, GPU_GATE.json all_pass with the same code sha256s and protocol sha256, the package
MATRIX.csv / WARMUP_PARENTS.csv byte-identical to the v1.1 package manifest, and the hardware block GPU of the cell
(CELLS_R30.json block_gpu; the driver refuses any other device).
  parent : one of the 7 WARMUP_PARENTS rows -> <out>/parents/<parent_id>; built once, never overwritten.
  calib  : a matching H parent registered for calibration (s1001; A1 adds s1002/s1003), bound in PARENT_BINDINGS.json, file
           sha256 re-checked -> <out>/calib/<parent_id>/.
  unit   : one MATRIX_A1.csv unit (package MATRIX.csv + frozen amendment A1) whose plan_A1 action is RUN40 / RUN10 /
           RUN10_CONTROL (REUSE rows are refused);
           mode fork from the bound parent (sha256 + fingerprint) unless this unit's own state_last.pt exists (mode resume,
           exclusive lock + R30_IDENTITY.json; the state's own opt must name the same unit / parent / objective / scale /
           coefficient / budget). fit_static units need the parent's coefficient in COEFFICIENTS.json (same parent sha,
           protocol, code; status OK)."""
from pathlib import Path
import argparse, csv, fcntl, hashlib, json, os, shlex, subprocess, sys

WARMUP_EPOCH, K, N_BATCHES, MILESTONE = 5, 5, 64, 10
COEF_KEY = {'L': 'Ls', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}
RUN_ACTIONS = ('RUN40', 'RUN10', 'RUN10_CONTROL')


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def common_checks(a):
    code = Path(a.code_dir).resolve()
    im = json.loads((code / 'INSTALL_MANIFEST.json').read_text())
    for f, h in im['files_sha256'].items():
        if digest(code / f) != h:
            raise SystemExit('Installed source changed: ' + f)
    if sorted(im['files_sha256']) != sorted(p.name for p in code.glob('*.py')):
        raise SystemExit('Installed source set differs from the manifest.')
    root = Path(a.root).resolve()
    pf = root / 'PROTOCOL_FROZEN.json'
    pr = json.loads(pf.read_text()) if pf.is_file() else {}
    if not (pr.get('frozen') is True and pr.get('version') == 'R30-v1.1-merged-20261101'):
        raise SystemExit('v1.1 PROTOCOL_FROZEN.json missing or not frozen.')
    psha = digest(pf)
    if not Path(a.gate_report).is_file():
        raise SystemExit('No GPU gate report: run the v1.1 gate first.')
    gate = json.loads(Path(a.gate_report).read_text())
    if not (gate.get('all_pass') is True and gate.get('code_files_sha256') == im['files_sha256'] and gate.get('config_frozen') is True
            and gate.get('protocol_frozen_sha256') == psha):
        raise SystemExit('GPU gate missing PASS, not frozen, or does not match this source snapshot / protocol.')
    man = json.loads((root / 'v11_protocol' / 'MANIFEST_SHA256.json').read_text())['files']
    for name, rel in (('MATRIX.csv', 'MATRIX.csv'), ('WARMUP_PARENTS.csv', 'configs/WARMUP_PARENTS.csv'), ('PROTOCOL.json', 'configs/PROTOCOL.json')):
        if digest(root / 'v11_protocol' / name) != man[rel]:
            raise SystemExit('Protocol copy differs from the v1.1 package: ' + name)
    af = root / 'PROTOCOL_AMENDMENT_A1.json'
    am = json.loads(af.read_text()) if af.is_file() else {}
    asha = digest(af) if af.is_file() else None
    if not (am.get('frozen') is True and pr.get('amendment_a1_sha256') == asha and gate.get('amendment_a1_sha256') == asha
            and digest(root / am['matrix_a1']['file']) == am['matrix_a1']['sha256']):
        raise SystemExit('Amendment A1 missing, not frozen, not bound in the protocol / gate, or MATRIX_A1.csv changed.')
    return root, code, im, psha, am


def cell_for(a, noise_percent, seed):
    cid = f'f{int(noise_percent):02d}_s{int(seed)}'
    c = json.loads(Path(a.cells).read_text())[cid]
    if (c['data_name'], int(c['noise_percent']), int(c['seed'])) != ('f30k_precomp', int(noise_percent), int(seed)):
        raise SystemExit('CELLS identity mismatch.')
    noise = Path(c['noise_file']).resolve()
    if not noise.is_file():
        raise SystemExit('Missing noise file: ' + str(noise))
    if c['noise_ratio'] == 0.0:  # identity mapping (data.py itself uses arange(N)//5 at ratio 0 and does not load the file)
        import numpy as np
        x = np.load(noise)
        if not (x.dtype == np.int64 and np.array_equal(x, np.arange(x.shape[0]) // 5)):
            raise SystemExit('0% noise file is not the identity mapping.')
    return cid, c, noise


def base_args(a, c, noise, out, num_epochs):
    return ['--data_path', a.data_path, '--data_name', c['data_name'], '--noise_file', str(noise),
            '--noise_ratio', str(c['noise_ratio']), '--seed', str(c['seed']), '--k', str(K), '--num_epochs', str(num_epochs),
            '--warmup_epoch', str(WARMUP_EPOCH), '--prefix_epoch', '-1', '--output_dir', str(out)]


def binding_for(a, pid, im):
    b = {p['parent_id']: p for p in json.loads(Path(a.bindings).read_text())['parents']}.get(pid)
    if not b or b.get('status') != 'BOUND':
        raise SystemExit(f'Parent {pid} is not bound.')
    rec = json.loads((Path(b['state_path']).parent / 'PARENT.json').read_text())
    if digest(b['state_path']) != b['boundary_sha256'] or rec['sha256'] != b['boundary_sha256']:
        raise SystemExit('Parent file sha256 differs from its binding.')
    if rec['training_state_fingerprint'] != b['training_state_fingerprint'] or rec['code_sha256'] != im['files_sha256']:
        raise SystemExit('Parent fingerprint / code identity differs from the binding or this (gated) snapshot.')
    if rec['max_iters'] != 0 or len(rec['warmup_dev']) != WARMUP_EPOCH:
        raise SystemExit('Parent is not a full 5-epoch warmup.')
    return b


def run_locked(out, cmd, cwd, record):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.running.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Same stage/unit already running.')
        if record():
            return
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode:
            raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('parent', 'calib', 'unit'))
    p.add_argument('--parent'); p.add_argument('--unit')
    for n in ('root', 'cells', 'code-dir', 'data-path', 'out-base', 'gate-report'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--bindings'); p.add_argument('--plan'); p.add_argument('--coefficients')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    root, code, im, psha, am = common_checks(a)
    job = os.environ.get('SLURM_JOB_ID')
    parents = {r['parent_id']: r for r in rows(root / 'v11_protocol' / 'WARMUP_PARENTS.csv')}

    if a.stage == 'parent':
        pr = parents.get(a.parent)
        if pr is None:
            raise SystemExit('Not one of the 7 registered parents.')
        cid, c, noise = cell_for(a, pr['noise_percent'], pr['seed'])
        out = Path(a.out_base) / 'parents' / a.parent
        cmd = [a.python, '-u', '-B', str(code / 'r30_run.py'), '--r30_stage', 'warmup', '--r30_parent_id', a.parent,
               '--r30_source_warmup', pr['warmup_source'], '--r30_require_gpu', c['block_gpu']] + base_args(a, c, noise, out, 40)
        print(shlex.join(cmd), flush=True)
        if not a.execute:
            return

        def rec():
            if (out / 'PARENT.json').exists():
                print('Parent already built; no duplicate warmup.'); return True
            if (out / 'parent_state.pt').exists():
                raise SystemExit('parent_state.pt without PARENT.json; inspect, do not overwrite.')
            with (out / 'R30_ATTEMPTS.jsonl').open('a') as f:
                f.write(json.dumps({'job': job, 'command': cmd}) + '\n')
            return False
        run_locked(out, cmd, code, rec)
        if not (out / 'PARENT.json').exists():
            raise SystemExit('Warmup exited without PARENT.json.')
        return

    if a.stage == 'calib':
        pr = parents.get(a.parent)
        if pr is None or pr['warmup_source'] != 'H' or a.parent not in am['changes']['calibration_parents']:
            raise SystemExit('Calibration only on the registered matching H parents.')
        cid, c, noise = cell_for(a, pr['noise_percent'], pr['seed'])
        b = binding_for(a, a.parent, im)
        out = Path(a.out_base) / 'calib' / a.parent
        js = out / f'COEFFICIENTS_{a.parent}.json'
        cmd = [a.python, '-u', '-B', str(code / 'r30_calibrate.py'), '--r30_parent', b['state_path'], '--r30_parent_sha', b['boundary_sha256'],
               '--r30_parent_id', a.parent, '--r30_parent_fingerprint', b['training_state_fingerprint'], '--n_batches', str(N_BATCHES),
               '--out', str(js), '--r30_require_gpu', c['block_gpu'], '--protocol_sha256', psha] + base_args(a, c, noise, out, 40)
        print(shlex.join(cmd), flush=True)
        if not a.execute:
            return

        def rec():
            if js.exists():
                raise SystemExit('Coefficients already written for this parent; the calibration runs once.')
            with (out / 'R30_ATTEMPTS.jsonl').open('a') as f:
                f.write(json.dumps({'job': job, 'command': cmd}) + '\n')
            return False
        run_locked(out, cmd, code, rec)
        return

    # ---- unit
    if digest(a.plan) != am['plan_a1']['sha256']:
        raise SystemExit('Execution plan differs from the frozen plan_A1.')
    m = {r['unit_id']: r for r in rows(root / am['matrix_a1']['file'])}.get(a.unit)
    plan = {r['unit_id']: r for r in rows(a.plan)}.get(a.unit)
    if m is None or plan is None:
        raise SystemExit('Unit is not in the v1.1 matrix / execution plan.')
    if plan['action'] not in RUN_ACTIONS:
        raise SystemExit(f"Plan action {plan['action']}: not a run.")
    if any(plan[k] != m[k] for k in m):
        raise SystemExit('Execution plan row differs from MATRIX.csv.')
    cid, c, noise = cell_for(a, m['noise_percent'], m['seed'])
    b = binding_for(a, m['parent_id'], im)
    obj, scale, epochs = m['continuation_objective'], m['scale_rule'], int(m['co_train_epochs'])
    if (plan['action'] == 'RUN40') != (epochs == 40):
        raise SystemExit('Plan action / budget mismatch.')
    coef, calrec = None, None
    if scale == 'fit_static':
        e = (json.loads(Path(a.coefficients).read_text()) if a.coefficients else {}).get('parents', {}).get(m['parent_id'])
        if not e:
            raise SystemExit('No coefficients for this parent.')
        key = COEF_KEY[obj]
        if not (e['parent_sha256'] == b['boundary_sha256'] and e['protocol_frozen_sha256'] == psha and e['code_sha256'] == im['files_sha256']
                and e['all_ok'] is True and e['coefficients'][key]['status'] == 'OK' and digest(e['source_file']) == e['source_sha256']):
            raise SystemExit('Coefficient record does not match parent / protocol / code, or is not OK.')
        coef, calrec = float(e['coefficients'][key]['c']), e['source_sha256']
    elif scale == '1':
        coef = 1.0
    out = Path(a.out_base) / 'runs' / m['output_subdir']
    identity = {'unit': m, 'action': plan['action'], 'parent_id': m['parent_id'], 'parent_state': b['state_path'],
                'parent_sha256': b['boundary_sha256'], 'parent_fingerprint': b['training_state_fingerprint'], 'coefficient': coef,
                'calibration_record_sha256': calrec, 'noise_file': str(noise), 'noise_sha256': digest(noise),
                'source_files_sha256': im['files_sha256'], 'protocol_frozen_sha256': psha, 'block_gpu': c['block_gpu'], 'milestone': MILESTONE}
    identity_file = out / 'R30_IDENTITY.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise SystemExit('Cannot resume: identity mismatch.')
    last = out / 'state_last.pt'
    if last.exists() and not identity_file.exists():
        raise SystemExit('Unidentified state in output directory; refuse resume.')
    if (out / f'state_completed{MILESTONE}.pt').exists() and not (out / f'E{MILESTONE}_RECORD.json').exists():
        raise SystemExit('Completed-epoch checkpoint without its record (interrupted milestone); inspect.')
    mode = 'resume' if last.exists() else 'fork'
    cmd = [a.python, '-u', '-B', str(code / 'r30_run.py'), '--r30_stage', 'unit', '--r30_unit_id', a.unit,
           '--r30_source_warmup', m['warmup_source'], '--r30_objective', obj, '--r30_scale', scale, '--r30_mode', mode,
           '--r30_parent', b['state_path'], '--r30_parent_sha', b['boundary_sha256'], '--r30_parent_id', m['parent_id'],
           '--r30_parent_fingerprint', b['training_state_fingerprint'], '--r30_milestone', str(MILESTONE), '--r30_require_gpu', c['block_gpu']] + \
          (['--r30_coef', repr(coef)] if coef is not None else []) + base_args(a, c, noise, out, epochs)
    if mode == 'resume':
        cmd += ['--resume', str(last)]
    print(shlex.join(cmd), flush=True)
    if not a.execute:
        return

    def rec():
        if (out / 'DONE.json').exists():
            print('Already complete; no duplicate run.'); return True
        resumed_epoch = None
        if last.exists():
            import torch
            st = torch.load(last, map_location='cpu', mmap=True, weights_only=False)
            o = st['opt']
            want = dict(r30_stage='unit', r30_unit_id=a.unit, r30_parent_sha=b['boundary_sha256'], r30_parent_fingerprint=b['training_state_fingerprint'],
                        r30_objective=obj, r30_scale=scale, r30_coef=coef, seed=int(c['seed']), noise_ratio=c['noise_ratio'], noise_file=str(noise),
                        num_epochs=epochs, warmup_epoch=WARMUP_EPOCH, k=K, max_iters=0, r30_milestone=MILESTONE)
            bad = {k: (o.get(k), v) for k, v in want.items() if o.get(k) != v}
            if bad:
                raise SystemExit(f'Cannot resume: state_last.pt belongs to another unit/config {bad}')
            resumed_epoch = int(st['epoch'])
        if not identity_file.exists():
            identity_file.write_text(json.dumps(identity, indent=2) + '\n')
        with (out / 'R30_ATTEMPTS.jsonl').open('a') as f:
            f.write(json.dumps({'job': job, 'mode': mode, 'command': cmd, 'resume_epoch': resumed_epoch}) + '\n')
        return False
    run_locked(out, cmd, code, rec)
    if not (out / 'DONE.json').exists():
        raise SystemExit('Driver exited without DONE; inspect, do not mark complete.')


if __name__ == '__main__':
    main()
