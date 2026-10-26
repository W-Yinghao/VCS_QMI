"""Write FREEZE.json: sha256 of every file the R23 runs read (code, protocol, frozen host/training sources, unit record)."""
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; BASE = HERE.parents[1]
FILES = ['handoff/r23_bf_local_20261026/run.py', 'handoff/r23_bf_local_20261026/gpu.sbatch', 'handoff/r23_bf_local_20261026/PROTOCOL.json',
         'handoff/r22_geometry_long_v2_20261024/run.py', 'handoff/r22_geometry_train_20261023/train.py',
         'handoff/r22_geometry_train_20261023/runs/bf_vcs_exact_1001/COMPLETE.json',
         'handoff/r22_c3_codex_20261023/units/r22_pmi3_bf_MLPp1_s1001.json',
         'ssl_pilot_v10_r22c/v10/r22_pmi/run_unit.py', 'ssl_pilot_v10_r22c/v10/r22_pmi/pmi_energy.py',
         'ssl_pilot_v10_r22c/src/v10/adapter.py', 'ssl_pilot_v10_r22c/src/v10/supplement/checkpoints.py',
         'ssl_pilot_v10_r22c/src/v9/reg/r6_reg_p2.py', 'ssl_pilot_v10_r22c/src/v9/reg/r6_reg_pilot.py',
         'handoff/r22_geometry_long_v2_20261024/runs/bf_vcs_exact_1001/COMPLETE.json',
         'handoff/r22_geometry_long_v2_20261024/runs/bf_nce_1001/COMPLETE.json']
sha = lambda p: hashlib.sha256((BASE / p).read_bytes()).hexdigest()
(HERE / 'FREEZE.json').write_text(json.dumps(dict(files={p: sha(p) for p in FILES}), indent=1) + '\n')
print('FREEZE', hashlib.sha256((HERE / 'FREEZE.json').read_bytes()).hexdigest())
