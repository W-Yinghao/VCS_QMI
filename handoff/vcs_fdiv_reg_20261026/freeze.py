import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; BASE = HERE.parents[1]
files = ["run.py", "fdiv.py", "gpu.sbatch", "cpu.sbatch", "PROTOCOL.json"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
json.dump({"files": {str((HERE / f).relative_to(BASE)): sha(HERE / f) for f in files}}, open(HERE / "FREEZE.json", "w"), indent=1)
print(json.load(open(HERE / "FREEZE.json")))
