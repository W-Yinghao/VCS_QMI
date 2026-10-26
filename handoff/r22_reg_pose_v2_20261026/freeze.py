"""Write FREEZE.json: sha256 of every file the v2 runs read (own files, v1 runner/constants, v1's frozen dependencies)."""
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; BASE = HERE.parents[1]
own = [f'handoff/{HERE.name}/{n}' for n in ('run.py', 'gpu.sbatch', 'CONSTANTS.json', 'PROTOCOL.json')]
v1 = ['handoff/r22_reg_pose_20261025/run.py', 'handoff/r22_reg_pose_20261025/CONSTANTS.json']
deps = [p for p in json.loads((BASE / 'handoff/r22_reg_pose_20261025/FREEZE.json').read_text())['files'] if not p.startswith('handoff/r22_reg_pose_20261025/')]
sha = lambda p: hashlib.sha256((BASE / p).read_bytes()).hexdigest()
v1f = json.loads((BASE / 'handoff/r22_reg_pose_20261025/FREEZE.json').read_text())['files']
assert all(sha(p) == v1f[p] for p in deps + v1[:1] + v1[1:]), 'v1 dependencies changed since the v1 freeze'
(HERE / 'FREEZE.json').write_text(json.dumps(dict(files={p: sha(p) for p in own + v1 + deps}), indent=1) + '\n')
print(len(own + v1 + deps), 'files frozen')
