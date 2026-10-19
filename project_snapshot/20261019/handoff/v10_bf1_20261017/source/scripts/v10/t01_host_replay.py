"""V10 T01 host replay (CPU): runs the UNCHANGED V9 host trainer G.cmd_train for a few steps, exactly as the host runs it.

This process never imports v10.adapter (whose V9-writer tripwires would replace G.cmd_train), and V9_REG_ROOT must point under
outputs_v10 BEFORE the host import, so the host writes only <root>/units/<uid>/run/{ckpt_<steps>.pt, train.json} (it reads the
R6 FIT cache and manifest itself, read-only).  CUDA_VISIBLE_DEVICES must be empty: the host trainer uses CUDA whenever visible.
Called by scripts/v10/t01_validate.py part host_replay; prints one JSON line (host code hash, root, files, thread count).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("V10_OUT_ROOT", "/projects/EEG-foundation-model/yinghao/outputs_v10")).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="replay root (becomes the host's V9_ROOT); must be under outputs_v10")
    ap.add_argument("--uid", required=True)
    ap.add_argument("--seed", type=int, default=901)
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args()
    root = a.root.resolve()
    if OUT_ROOT not in root.parents:
        raise SystemExit(f"refused: {root} is not under {OUT_ROOT}")
    if os.environ.get("V9_REG_ROOT") is None or Path(os.environ["V9_REG_ROOT"]).resolve() != root:
        raise SystemExit("refused: V9_REG_ROOT must equal --root before the host import")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("refused: CUDA_VISIBLE_DEVICES must be empty (the host trainer uses CUDA whenever it is visible)")
    if not 1 <= a.steps <= 10:
        raise SystemExit("refused: 1..10 steps only (a replay, not a training run)")
    if "v10.adapter" in sys.modules:
        raise SystemExit("refused: v10.adapter is imported in this process")
    sys.path.insert(0, str(REPO / "src"))
    from v9.reg import g1 as G                                   # the unchanged host (imports r6_reg_pilot / r6_reg_p2)
    if Path(G.V9_ROOT).resolve() != root:
        raise SystemExit(f"refused: host V9_ROOT {G.V9_ROOT} != {root}")
    files = (Path(G.__file__).resolve(), Path(G.P.__file__).resolve(), Path(G.Q.__file__).resolve())
    host_hash = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]
    rc = G.cmd_train(a.uid, "vcs", 1.0, 0.0, a.seed, steps=a.steps)
    run = G.unit_root(a.uid) / "run"
    import torch
    print(json.dumps({"rc": rc, "host_code_hash": host_hash, "v9_root": str(Path(G.V9_ROOT).resolve()), "run": str(run),
                      "files": sorted(p.name for p in run.iterdir()), "v10_adapter_imported": "v10.adapter" in sys.modules,
                      "cmd_train_is_tripwire": bool(getattr(G.cmd_train, "v10_tripwire", False)), "torch_threads": torch.get_num_threads(),
                      "cuda_available": torch.cuda.is_available()}), flush=True)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
