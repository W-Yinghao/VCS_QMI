"""Freeze the files the R24 runs read (package code + the imported host modules)."""
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; BASE = HERE.parents[1]
FILES = ['handoff/vcs_comir_20261026/run.py', 'handoff/vcs_comir_20261026/gpu.sbatch', 'handoff/vcs_comir_20261026/PROTOCOL.json',
         'ssl_pilot_v10_r22c/v10/r22_pmi/run_unit.py', 'ssl_pilot_v10_r22c/v10/r22_pmi/pmi_energy.py',
         'ssl_pilot_v10_r22c/src/v10/supplement/comir.py', 'ssl_pilot_v10_r22c/src/v10/supplement/registration.py',
         'ssl_pilot_v10_r22c/src/v10/supplement/checkpoints.py', 'ssl_pilot_v10_r22c/src/v10/adapter.py',
         'ssl_pilot_v10_r22c/src/v10/epfl/host.py', 'ssl_pilot_v10_r22c/src/v10/epfl/guard.py', 'ssl_pilot_v10_r22c/src/v10/epfl/production.py',
         'ssl_pilot_v10_r22c/src/v9/reg/r6_reg_p2.py', 'ssl_pilot_v10_r22c/src/v9/reg/r6_reg_pilot.py',
         'handoff/v10_bf1_20261017/PROTOCOL.json']
sha = lambda p: hashlib.sha256((BASE / p).read_bytes()).hexdigest()
(HERE / 'FREEZE.json').write_text(json.dumps({'files': {f: sha(f) for f in FILES}}, indent=1) + '\n')
print(hashlib.sha256((HERE / 'FREEZE.json').read_bytes()).hexdigest())
