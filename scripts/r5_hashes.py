"""r5 intake / R5-30 registration hashes: pretrained SigLIP 2 B/16 weight files (HF cache), image-role assignment per dataset, and the VL1-21
proposal files (FIT / CAL / DEV) per dataset.  Writes reports/R5/r5_hashes.json.
    VL_DATASET=refcocog python scripts/r5_hashes.py   (run once per dataset; results merged)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl3_finetune import RG, SIGLIP_REPOS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "R5" / "r5_hashes.json"; PROP = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import snapshot_download
    ds = RG.DATASET.replace("+", "plus"); res = json.load(open(OUT)) if OUT.exists() else {}
    snap = Path(snapshot_download(SIGLIP_REPOS["siglip2_b16"], local_files_only=True))
    res["siglip2_b16_weights"] = {"repo": SIGLIP_REPOS["siglip2_b16"], "snapshot": snap.name,
                                  "files": {p.name: sha(p) for p in sorted(snap.iterdir()) if p.suffix in (".safetensors", ".bin", ".json", ".model")}}
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    roles = sorted((str(i), str(r)) for i, r in role.items())
    res.setdefault("datasets", {})[ds] = {"roles_sha256": hashlib.sha256(json.dumps(roles).encode()).hexdigest(),
                                         "role_counts": {r: sum(1 for _, x in roles if x == r) for r in sorted({x for _, x in roles})},
                                         "proposals_sha256": {r: sha(PROP / f"{ds}_{r}.pt") for r in ("FIT", "CAL", "DEV")}}
    OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res["datasets"][ds], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
