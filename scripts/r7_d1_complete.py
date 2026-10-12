"""r7 R7-30 D1-10 completeness check before the CAL readout: every shard's done-record present with present == expected, 0 duplicates,
0 skipped images; per-role question totals equal the frozen manifest; manifest sha256 equals the frozen value.  Reads done-records and
the manifest only (no teacher probabilities).
    python scripts/r7_d1_complete.py --teacher-dir T --manifest-dir D --frozen-sha256 SHA
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher-dir", required=True); ap.add_argument("--manifest-dir", required=True); ap.add_argument("--frozen-sha256", required=True)
    a = ap.parse_args(); td, md = Path(a.teacher_dir), Path(a.manifest_dir)
    plan = json.loads((md / "d1_shards.json").read_text()); meta = json.loads((md / "d1_manifest_meta.json").read_text())
    sha = hashlib.sha256((md / "d1_manifest.jsonl").read_bytes()).hexdigest()
    rows = [json.loads(x) for x in (md / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]; per_role = {}
    for r in rows:
        per_role[r["role"]] = per_role.get(r["role"], 0) + 1
    shards, bad, tot = {}, [], {}
    for s in plan["shards"]:
        f = td / f"{s}.done.json"
        if not f.exists():
            bad.append(f"{s}: missing done-record"); continue
        d = json.loads(f.read_text()); shards[s] = {k: d[k] for k in ("expected", "present", "duplicates", "gpu")}; shards[s]["skipped"] = len(d["skipped_images"])
        if d["present"] != d["expected"] or d["duplicates"] or d["skipped_images"]:
            bad.append(f"{s}: {shards[s]}")
        tot[s.split("-")[0]] = tot.get(s.split("-")[0], 0) + d["present"]
    res = {"shards": len(plan["shards"]), "done_records": len(shards), "problems": bad, "present_per_role": tot, "manifest_per_role": per_role,
           "manifest_sha256": sha, "frozen_sha256": a.frozen_sha256, "sha256_unchanged": sha == a.frozen_sha256 == meta["manifest_sha256"],
           "frozen_flag": meta.get("frozen"), "gpus": sorted({v["gpu"] for v in shards.values()})}
    res["complete"] = (not bad and res["sha256_unchanged"] and all(tot.get(k) == per_role[k] for k in per_role) and res["frozen_flag"] is True)
    print(json.dumps(res, indent=1)); (td / "d1_10_completeness.json").write_text(json.dumps(res, indent=1))
    return 0 if res["complete"] else 7


if __name__ == "__main__":
    sys.exit(main())
