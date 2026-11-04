#!/usr/bin/env python3
"""Freeze NR2 before formal units: code / config / cache manifests / calibration / observed / proposal tables / env."""
import hashlib, json, os, sys, time
from pathlib import Path
N = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104'); O1 = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104')
O2 = Path('/home/infres/yinwang/CS_QMI/outputs_nr2/NR2_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
chk = json.loads((N/'checks/REAL_CHECKS.json').read_text())
if not chk['all_pass']: sys.exit('real checks not ALL_PASS')
out = N/'PROTOCOL_FROZEN_NR2.json'
if out.exists(): sys.exit('already frozen')
files = sorted([*(N/'code/src').rglob('*.py'), *[N/'code/site'/f for f in ('train_unit_nr1.py', 'train_unit_nr2.py', 'fit_readout.py',
                 'fit_strata_evalonly.py', 'build_tables.py', 'check_frozen.py', 'make_configs.py', 'real_checks.py')],
                 N/'sbatch/unit.sbatch', N/'sbatch/common_env.sh',
                O1/'cache/fit/manifest.json', O1/'cache/dev/manifest.json', O1/'evaluator_only/dev_positives.json',
                *[O1/f'calibration/{c}.json' for c in ('Flickr00', 'Flickr60', 'Flickr80')],
                *[O1/f'data/observed_{c}.tsv' for c in ('Flickr00', 'Flickr60', 'Flickr80')],
                O2/'proposal/base_top100_fit.npz', O2/'proposal/TABLES_MANIFEST.json', N/'package/PACKAGE_MANIFEST.json',
                N/'package/configs/MATRIX.csv', N/'package/reference/proposal_reference.py'])
fz = dict(protocol_id='NR2_20261104', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          authorisation=dict(user_quote='查看这个包，准备，并提交相关实验', package='VCS_NR1_Review_NR2_Server_20261104.zip sha 8c2fafc1…, PACKAGE_MANIFEST 18/18',
                             evidence_snapshot='81e8fee19852a823309cc7dc24933a21c50e10ba'),
          constraints=['36 units per package MATRIX.csv only', 'encoder frozen, architecture G only', 'TEST never read',
                       'DEV only in separate evaluator process, scores hashed before truth', 'deployment ranking uses s only',
                       'CONFIRM = NR1 seed-1001 G recipe with seed changed only', 'U is a new sampler run (no NR1 reuse)',
                       'primary endpoint step 8000; no early stopping / cancellation on interim DEV'],
          model=dict(id='google/siglip2-base-patch16-224', revision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'),
          real_checks_sha256=sha(N/'checks/REAL_CHECKS.json'), cpu_tests='package 18/18, production 7/7 (job 1086418)',
          hashes={str(p): sha(p) for p in files},
          units={p.stem: sha(p) for p in sorted((N/'locked_configs').glob('*.json'))})
assert len(fz['units']) == 36
out.write_text(json.dumps(fz, indent=2, ensure_ascii=False) + '\n'); os.chmod(out, 0o444); print('frozen', len(fz['hashes']), 'files', len(fz['units']), 'units')
