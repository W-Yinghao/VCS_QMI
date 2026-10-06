"""Print {config file: resolved config_hash or ERROR} for every configs/*.yaml, using whichever vcs_ssl is first on sys.path.
Run twice (working tree vs a HEAD worktree via PYTHONPATH) and diff to show that new optional fields leave existing hashes byte-identical.
    python scripts/p138_config_hashes.py --src <path to src> --out hashes.json"""
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--configs", default="configs"); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.src)
from vcs_ssl.config import load_config  # noqa: E402
out = {}
for p in sorted(Path(a.configs).glob("*.yaml")):
    try:
        out[p.name] = load_config(p, require_dirs=False)["_meta"]["config_hash"]
    except Exception as e:  # noqa: BLE001
        out[p.name] = f"ERROR {type(e).__name__}: {str(e)[:80]}"
Path(a.out).write_text(json.dumps(out, indent=1))
print(f"{len(out)} configs, {sum(not v.startswith('ERROR') for v in out.values())} loaded")
