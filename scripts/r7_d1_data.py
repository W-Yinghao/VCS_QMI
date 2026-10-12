"""r7 R7-30 D1-10 data roles: GQA train-balanced binary (yes / no) questions, grouped by image into FIT / CAL / DEV (no val, no testdev).
  manifest : questions file (official train_balanced_questions.json, or questions1.2.zip, or an lmms-lab parquet dir) -> manifest + meta.
             Eligible images carry >= 1 binary question; images sorted by sha256('r7-30/d1-role/<imageId>') and the first FIT+CAL+DEV of them
             assigned in a 5:1:1 interleave (positions 0-4 FIT, 5 CAL, 6 DEV); per image up to K binary questions in
             sha256('r7-30/d1-q/<qid>') order.  Optional LLaVA-1.5 mix665k file: QA pairs LLaVA-1.5 trained on are excluded before selection
             (every GQA train image was in its instruction mix, so image exposure remains and is disclosed either way).
  shards   : fixed shard plan (real image ids per shard; CAL first, then FIT, then DEV) and GPU bundles of shards.
  feed     : feed-lines file for slurm/feed_normal.sh (CAP 27): one sbatch line per bundle, H100 / L40S only (the SparseVLM / LLaVA env has no
             Blackwell kernels), node51 / node52 / node60 excluded, no --export.
  zipcheck : every manifest image present in GQA images.zip (central directory only).
  freeze   : mark the manifest frozen after the coordinator's go (the GPU teacher refuses an unfrozen manifest or a failing probe).
    python scripts/r7_d1_data.py manifest --questions PATH --out DIR [--fit 10000 --cal 2000 --dev 2000 --per-image 4] [--mix665k PATH]
    python scripts/r7_d1_data.py shards --out DIR [--images-per-shard 250 --shards-per-bundle 4]
    python scripts/r7_d1_data.py feed --out DIR --lines slurm/r7_d1_10_feed_lines.txt
    python scripts/r7_d1_data.py zipcheck --out DIR --zip /projects/EEG-foundation-model/yinghao/data/gqa/zips/images.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

SHORT = "\nAnswer the question using a single word or phrase."
ROLE_CYCLE = ("FIT", "FIT", "FIT", "FIT", "FIT", "CAL", "DEV")


def h(tag, key):
    return hashlib.sha256(f"r7-30/{tag}/{key}".encode()).hexdigest()


def sha256_file(p: Path, chunk=1 << 24):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            d.update(b)
    return d.hexdigest()


def norm_q(q: str) -> str:
    q = q.replace("<image>", "").replace(SHORT.strip(), "")
    return re.sub(r"[^a-z0-9 ]+", "", q.lower()).strip()


def read_questions(path: Path):
    """Rows {qid, image_id, question, answer, structural, semantic, detailed} from the official JSON (plain or inside questions1.2.zip) or
    an lmms-lab instructions parquet directory."""
    def from_official(d):
        for qid, r in d.items():
            t = r.get("types") or {}
            yield {"qid": str(qid), "image_id": str(r["imageId"]), "question": r["question"], "answer": r["answer"],
                   "structural": t.get("structural"), "semantic": t.get("semantic"), "detailed": t.get("detailed")}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("train_balanced_questions.json"))
            with z.open(name) as f:
                yield from from_official(json.load(f))
    elif path.suffix == ".json":
        yield from from_official(json.loads(path.read_text()))
    else:
        import pyarrow.parquet as pq
        for f in sorted(path.glob("*.parquet")):
            for b in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["id", "imageId", "question", "answer", "types"]):
                for r in b.to_pylist():
                    t = r.get("types") or {}
                    yield {"qid": str(r["id"]), "image_id": str(r["imageId"]), "question": r["question"], "answer": r["answer"],
                           "structural": t.get("structural"), "semantic": t.get("semantic"), "detailed": t.get("detailed")}


def llava_seen_pairs(path: Path):
    """(image id, normalised question) pairs of the GQA part of llava_v1_5_mix665k.json."""
    seen = set()
    for e in json.loads(path.read_text()):
        img = str(e.get("image", ""))
        if not img.startswith("gqa/"):
            continue
        iid = Path(img).stem
        for t in e.get("conversations", []):
            if t.get("from") == "human":
                seen.add((iid, norm_q(t.get("value", ""))))
    return seen


def build_manifest(rows, n_fit=10000, n_cal=2000, n_dev=2000, per_image=4, seen=None):
    by_img, n_rows, n_bin, n_seen = {}, 0, 0, 0
    for r in rows:
        n_rows += 1
        a = str(r["answer"]).strip().lower()
        if a not in ("yes", "no"):
            continue
        n_bin += 1
        if seen is not None and (r["image_id"], norm_q(r["question"])) in seen:
            n_seen += 1
            continue
        r = dict(r, answer=a, C=1 if a == "yes" else -1, llava_qa_seen=(False if seen is not None else "unknown"))
        by_img.setdefault(r["image_id"], []).append(r)
    order = sorted(by_img, key=lambda i: h("d1-role", i)); need = n_fit + n_cal + n_dev
    if len(order) < need:
        raise ValueError(f"only {len(order)} eligible images for {need} requested")
    quota = {"FIT": n_fit, "CAL": n_cal, "DEV": n_dev}; count = {k: 0 for k in quota}; out = []; pos = 0
    for img in order:
        role = ROLE_CYCLE[pos % len(ROLE_CYCLE)]; pos += 1
        if count[role] >= quota[role]:
            role = next((k for k in ("FIT", "CAL", "DEV") if count[k] < quota[k]), None)
            if role is None:
                break
        count[role] += 1
        for r in sorted(by_img[img], key=lambda r: h("d1-q", r["qid"]))[:per_image]:
            out.append(dict(r, role=role))
    fit = [r for r in out if r["role"] == "FIT"]; pi_yes = sum(r["C"] == 1 for r in fit) / len(fit)
    meta = {"rows_read": n_rows, "binary_rows": n_bin, "excluded_llava_seen_qa": n_seen if seen is not None else None,
            "eligible_images": len(order), "per_image": per_image, "role_rule": "sha256('r7-30/d1-role/<imageId>') order, 5:1:1 interleave",
            "question_rule": "sha256('r7-30/d1-q/<qid>') order within image",
            "class_weights_from_FIT": {"pi_yes": pi_yes, "w_yes": 0.5 / pi_yes, "w_no": 0.5 / (1 - pi_yes)},
            "counts": {k: {"images": len({r["image_id"] for r in out if r["role"] == k}), "questions": sum(r["role"] == k for r in out),
                           "yes": sum(r["role"] == k and r["C"] == 1 for r in out)} for k in quota}}
    return out, meta


def cmd_manifest(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); src = Path(a.questions); t0 = time.time()
    seen = llava_seen_pairs(Path(a.mix665k)) if a.mix665k else None
    rows, meta = build_manifest(read_questions(src), a.fit, a.cal, a.dev, a.per_image, seen)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r7" / "package_r7_20261011" / "reference")); import vcs_vl_reference as REF  # noqa: E402
    REF.assert_group_disjoint({k: [r["image_id"] for r in rows if r["role"] == k] for k in ("FIT", "CAL", "DEV")})
    meta.update({"questions_source": str(src), "questions_sha256": sha256_file(src) if src.is_file() else "parquet-dir",
                 "mix665k": str(a.mix665k) if a.mix665k else None, "mix665k_sha256": sha256_file(Path(a.mix665k)) if a.mix665k else None,
                 "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)})
    (out / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta["manifest_sha256"] = sha256_file(out / "d1_manifest.jsonl")
    (out / "d1_manifest_meta.json").write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))
    return 0


def load_manifest(out: Path):
    return [json.loads(x) for x in (out / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]


def plan_shards(rows, images_per_shard=250, shards_per_bundle=4):
    shards, bundles = {}, {}
    for role in ("CAL", "FIT", "DEV"):                                                 # CAL first: the D1-10 readout needs it earliest
        imgs = sorted({r["image_id"] for r in rows if r["role"] == role}, key=lambda i: h("d1-role", i))
        for j in range(0, len(imgs), images_per_shard):
            shards[f"{role}-{j // images_per_shard:03d}"] = imgs[j:j + images_per_shard]
    ids = list(shards)
    for b in range(0, len(ids), shards_per_bundle):
        bundles[f"b{b // shards_per_bundle:03d}"] = ids[b:b + shards_per_bundle]
    return {"images_per_shard": images_per_shard, "shards_per_bundle": shards_per_bundle, "shards": shards, "bundles": bundles,
            "questions_per_shard": {s: sum(r["image_id"] in set(v) for r in rows) for s, v in shards.items()}}


def cmd_shards(a):
    out = Path(a.out); plan = plan_shards(load_manifest(out), a.images_per_shard, a.shards_per_bundle)
    (out / "d1_shards.json").write_text(json.dumps(plan, indent=1))
    print(f"[r7_d1_data] {len(plan['shards'])} shards in {len(plan['bundles'])} bundles; questions {sum(plan['questions_per_shard'].values())}")
    return 0


def feed_lines(plan, wrapper="slurm/r7_d1_gpu.sbatch"):
    return [f"sbatch --parsable --job-name=r7d1t_{b} --partition=H100,L40S --exclude=node51,node52,node60 {wrapper} bundle {b}" for b in plan["bundles"]]


def cmd_feed(a):
    plan = json.loads((Path(a.out) / "d1_shards.json").read_text()); lines = feed_lines(plan)
    Path(a.lines).write_text("# R7-30 D1-10 teacher bundles; submit with: CAP=27 nohup setsid bash slurm/feed_normal.sh " + a.lines + " >/dev/null 2>&1 &\n"
                             + "\n".join(lines) + "\n")
    print(f"[r7_d1_data] {len(lines)} feed lines -> {a.lines} (not submitted)")
    return 0


def cmd_freeze(a):
    """Mark the manifest frozen (only after the coordinator's go on the protocol); records who / when and re-checks the sha256."""
    out = Path(a.out); meta = json.loads((out / "d1_manifest_meta.json").read_text())
    if sha256_file(out / "d1_manifest.jsonl") != meta["manifest_sha256"]:
        raise SystemExit("manifest changed since it was built")
    meta.update({"frozen": True, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "protocol": a.protocol, "go_reference": a.go})
    (out / "d1_manifest_meta.json").write_text(json.dumps(meta, indent=1)); print(f"[r7_d1_data] frozen: {out} ({a.protocol}; go: {a.go})")
    return 0


def cmd_zipcheck(a):
    rows = load_manifest(Path(a.out)); need = {r["image_id"] for r in rows}
    with zipfile.ZipFile(a.zip) as z:
        names = {Path(n).stem for n in z.namelist() if n.lower().endswith(".jpg")}
    missing = sorted(need - names)
    res = {"zip": a.zip, "zip_images": len(names), "manifest_images": len(need), "missing": len(missing), "missing_examples": missing[:10]}
    (Path(a.out) / "d1_zipcheck.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0 if not missing else 3


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--questions", required=True); m.add_argument("--out", required=True)
    m.add_argument("--fit", type=int, default=10000); m.add_argument("--cal", type=int, default=2000); m.add_argument("--dev", type=int, default=2000)
    m.add_argument("--per-image", type=int, default=4); m.add_argument("--mix665k", default=None)
    s = sub.add_parser("shards"); s.add_argument("--out", required=True); s.add_argument("--images-per-shard", type=int, default=250)
    s.add_argument("--shards-per-bundle", type=int, default=4)
    f = sub.add_parser("feed"); f.add_argument("--out", required=True); f.add_argument("--lines", required=True)
    z = sub.add_parser("zipcheck"); z.add_argument("--out", required=True); z.add_argument("--zip", required=True)
    fr = sub.add_parser("freeze"); fr.add_argument("--out", required=True); fr.add_argument("--protocol", required=True); fr.add_argument("--go", required=True)
    a = ap.parse_args()
    return {"manifest": cmd_manifest, "shards": cmd_shards, "feed": cmd_feed, "zipcheck": cmd_zipcheck, "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
