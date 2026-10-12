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
TRAIN_MEMBER = "train_balanced_questions.json"                                        # allow-listed GQA question files (train only)
TRAIN_ALL_RE = re.compile(r"^train_all_questions_\d+\.json$")
CLOSED_NAMES = ("testdev", "test_", "challenge", "submission")
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


def _official_rows(d):
    for qid, r in d.items():
        t = r.get("types") or {}
        yield {"qid": str(qid), "image_id": str(r["imageId"]), "question": r["question"], "answer": r["answer"],
               "structural": t.get("structural"), "semantic": t.get("semantic"), "detailed": t.get("detailed")}


def read_questions(path: Path, source="balanced"):
    """Rows {qid, image_id, question, answer, structural, semantic, detailed}.  source='balanced': train_balanced_questions.json (plain,
    inside questions1.2.zip, or an lmms-lab parquet dir).  source='all': the ten train_all_questions_<n>.json files (a directory, or the
    members of questions1.2.zip), opened one by one by exact allow-listed name; test / testdev / challenge / submission are never opened."""
    if source == "all":
        if any(part.lower() == "closed" for part in path.parts):
            raise SystemExit(f"refusing a closed source: {path}")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                names = sorted(n for n in z.namelist() if TRAIN_ALL_RE.match(Path(n).name))
                if not names:
                    raise SystemExit(f"no train_all_questions_<n>.json member in {path}")
                for n in names:
                    with z.open(n) as f:
                        d = json.load(f)
                    yield from _official_rows(d); del d
        else:
            files = sorted(f for f in path.iterdir() if TRAIN_ALL_RE.match(f.name))
            if not files:
                raise SystemExit(f"no train_all_questions_<n>.json in {path}")
            for f in files:
                d = json.loads(f.read_text()); yield from _official_rows(d); del d
        return
    from_official = _official_rows
    if any(part.lower() in ("closed",) for part in path.parts) or any(t in path.name.lower() for t in CLOSED_NAMES):
        raise SystemExit(f"refusing a closed / test-type questions source: {path}")
    if path.suffix == ".zip":                                                          # open ONE allow-listed member by exact name;
        with zipfile.ZipFile(path) as z:                                              # test / testdev / challenge / submission stay unopened
            names = [n for n in z.namelist() if Path(n).name == TRAIN_MEMBER]
            if len(names) != 1:
                raise SystemExit(f"expected exactly one {TRAIN_MEMBER} in {path}, found {names}")
            with z.open(names[0]) as f:
                yield from from_official(json.load(f))
    elif path.suffix == ".json":
        if path.name != TRAIN_MEMBER:
            raise SystemExit(f"only {TRAIN_MEMBER} may be read, got {path.name}")
        yield from from_official(json.loads(path.read_text()))
    else:
        import pyarrow.parquet as pq
        for f in sorted(path.glob("*.parquet")):
            for b in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["id", "imageId", "question", "answer", "types"]):
                for r in b.to_pylist():
                    t = r.get("types") or {}
                    yield {"qid": str(r["id"]), "image_id": str(r["imageId"]), "question": r["question"], "answer": r["answer"],
                           "structural": t.get("structural"), "semantic": t.get("semantic"), "detailed": t.get("detailed")}


MATCH_RULE = ("(imageId, normalised question text): imageId = file stem of the mix665k 'image' path gqa/images/<imageId>.jpg; "
              "normalisation = drop '<image>' and the LLaVA short-answer suffix, lowercase, keep [a-z0-9 ] only, strip")


def llava_seen_pairs(path: Path, with_stats=False):
    """(image id, normalised question) pairs of every human turn in the GQA part of llava_v1_5_mix665k.json."""
    seen, conv, turns = set(), 0, 0
    for e in json.loads(path.read_text()):
        img = str(e.get("image", ""))
        if not img.startswith("gqa/"):
            continue
        iid = Path(img).stem; conv += 1
        for t in e.get("conversations", []):
            if t.get("from") == "human":
                seen.add((iid, norm_q(t.get("value", "")))); turns += 1
    stats = {"mix665k_gqa_conversations": conv, "mix665k_gqa_human_turns": turns, "distinct_seen_pairs": len(seen),
             "distinct_seen_images": len({i for i, _ in seen})}
    return (seen, stats) if with_stats else seen


BALANCE_RULE = ("per role, images in role order; inside an image the unseen binary questions are split into YES / NO lists (each in "
                "sha256('r7-30/d1-q/<qid>') order) and taken alternately, starting with the class the role is short of so far (YES on a tie), "
                "up to K per image, the other class continuing when one runs out; then, while the role's pi_yes lies outside [0.48, 0.52], the "
                "latest-selected question of the over-represented class is removed from the last image (in role order) that keeps >= 1 question")


def pick_alternating(yes, no, k, start_yes):
    out, iy, ino, take_yes = [], 0, 0, start_yes
    while len(out) < k and (iy < len(yes) or ino < len(no)):
        if (take_yes and iy < len(yes)) or ino >= len(no):
            out.append(yes[iy]); iy += 1
        else:
            out.append(no[ino]); ino += 1
        take_yes = not take_yes
    return out


def trim_to_band(per_img, lo=0.48, hi=0.50 + 0.02):
    """per_img: list of (image_id, [rows in selection order]); removes latest-selected rows of the over-represented class, last image first."""
    n_trim = 0
    while True:
        n = sum(len(v) for _, v in per_img); y = sum(r["C"] == 1 for _, v in per_img for r in v)
        if n == 0 or lo <= y / n <= hi:
            return n_trim
        over = 1 if y / n > hi else -1; done = False
        for _, v in reversed(per_img):
            if len(v) >= 2:
                for j in range(len(v) - 1, -1, -1):
                    if v[j]["C"] == over:
                        v.pop(j); n_trim += 1; done = True; break
            if done:
                break
        if not done:
            return n_trim


def build_manifest(rows, n_fit=10000, n_cal=2000, n_dev=2000, per_image=4, seen=None, balance=False):
    by_img, n_rows, n_bin, n_seen, all_pairs, seen_imgs = {}, 0, 0, 0, set(), set()
    for r in rows:
        n_rows += 1
        if seen is not None:
            all_pairs.add((r["image_id"], norm_q(r["question"])))
        a = str(r["answer"]).strip().lower()
        if a not in ("yes", "no"):
            continue
        n_bin += 1
        if seen is not None and (r["image_id"], norm_q(r["question"])) in seen:
            n_seen += 1; seen_imgs.add(r["image_id"])
            continue
        r = dict(r, answer=a, C=1 if a == "yes" else -1, llava_qa_seen=(False if seen is not None else "unknown"))
        by_img.setdefault(r["image_id"], []).append(r)
    order = sorted(by_img, key=lambda i: h("d1-role", i)); need = n_fit + n_cal + n_dev
    if len(order) < need:
        raise ValueError(f"only {len(order)} eligible images for {need} requested")
    quota = {"FIT": n_fit, "CAL": n_cal, "DEV": n_dev}; count = {k: 0 for k in quota}; out = []; pos = 0
    per_role = {k: [] for k in quota}
    for img in order:
        role = ROLE_CYCLE[pos % len(ROLE_CYCLE)]; pos += 1
        if count[role] >= quota[role]:
            role = next((k for k in ("FIT", "CAL", "DEV") if count[k] < quota[k]), None)
            if role is None:
                break
        count[role] += 1
        cands = sorted(by_img[img], key=lambda r: h("d1-q", r["qid"]))
        if balance:
            sel = per_role[role]; y = sum(r["C"] == 1 for _, v in sel for r in v); n = sum(len(v) for _, v in sel)
            pick = pick_alternating([r for r in cands if r["C"] == 1], [r for r in cands if r["C"] == -1], per_image, start_yes=(n - y) >= y)
        else:
            pick = cands[:per_image]
        per_role[role].append((img, [dict(r, role=role) for r in pick]))
    bal = {}
    for role, sel in per_role.items():
        n0 = sum(len(v) for _, v in sel); y0 = sum(r["C"] == 1 for _, v in sel for r in v)
        trimmed = trim_to_band(sel) if balance else 0; n1 = sum(len(v) for _, v in sel); y1 = sum(r["C"] == 1 for _, v in sel for r in v)
        bal[role] = {"pi_yes_before_trim": y0 / n0 if n0 else None, "trimmed": trimmed, "pi_yes": y1 / n1 if n1 else None}
    by_role_img = {img: (role, v) for role, sel in per_role.items() for img, v in sel}
    out = [r for img in order if img in by_role_img for r in by_role_img[img][1]]
    fit = [r for r in out if r["role"] == "FIT"]; pi_yes = sum(r["C"] == 1 for r in fit) / len(fit)
    meta = {"rows_read": n_rows, "binary_rows": n_bin, "excluded_llava_seen_qa": n_seen if seen is not None else None,
            "llava_exclusion": None if seen is None else {
                "excluded_binary_pairs": n_seen, "images_with_excluded_binary_pairs": len(seen_imgs),
                "seen_pairs_matching_a_train_balanced_question": len(seen & all_pairs), "match_rule": MATCH_RULE},
            "eligible_images": len(order), "per_image": per_image, "role_rule": "sha256('r7-30/d1-role/<imageId>') order, 5:1:1 interleave",
            "balance_rule": BALANCE_RULE if balance else None, "balance": bal,
            "question_rule": "sha256('r7-30/d1-q/<qid>') order within image",
            "class_weights_from_FIT": {"pi_yes": pi_yes, "w_yes": 0.5 / pi_yes, "w_no": 0.5 / (1 - pi_yes)},
            "counts": {k: {"images": len({r["image_id"] for r in out if r["role"] == k}), "questions": sum(r["role"] == k for r in out),
                           "yes": sum(r["role"] == k and r["C"] == 1 for r in out)} for k in quota}}
    return out, meta


def cmd_manifest(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); src = Path(a.questions); t0 = time.time()
    seen, seen_stats = llava_seen_pairs(Path(a.mix665k), with_stats=True) if a.mix665k else (None, None)
    rows, meta = build_manifest(read_questions(src, a.source), a.fit, a.cal, a.dev, a.per_image, seen, balance=a.balance)
    if seen_stats:
        meta["llava_exclusion"].update(seen_stats)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r7" / "package_r7_20261011" / "reference")); import vcs_vl_reference as REF  # noqa: E402
    REF.assert_group_disjoint({k: [r["image_id"] for r in rows if r["role"] == k] for k in ("FIT", "CAL", "DEV")})
    files = sorted(f for f in src.iterdir() if TRAIN_ALL_RE.match(f.name)) if (src.is_dir() and a.source == "all") else []
    meta.update({"questions_source": str(src), "question_set": "train_all (unbalanced)" if a.source == "all" else "train_balanced",
                 "questions_member": (TRAIN_MEMBER if a.source == "balanced" else "train_all_questions_<n>.json") if src.suffix in (".zip", ".json") or files else "lmms parquet",
                 "questions_sha256": sha256_file(src) if src.is_file() else {f.name: sha256_file(f) for f in files} if files else "parquet-dir",
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


def cmd_mixstats(a):
    """Exclusion statistics: mix665k GQA conversations / human turns / distinct seen pairs and images; binary pairs of the question source;
    how many are LLaVA-seen; 5 seeded random examples of unmatched seen pairs (if any) and of unseen binary pairs (if any)."""
    import random
    seen, st = llava_seen_pairs(Path(a.mix665k), with_stats=True); pairs, binp = set(), set()
    for r in read_questions(Path(a.questions), a.source):
        k = (r["image_id"], norm_q(r["question"])); pairs.add(k)
        if str(r["answer"]).strip().lower() in ("yes", "no"):
            binp.add(k)
    rnd = random.Random(20261012); unmatched = sorted(seen - pairs); unseen_bin = sorted(binp - seen)
    res = {**st, "question_source": a.questions, "question_set": a.source, "match_rule": MATCH_RULE, "source_distinct_pairs": len(pairs),
           "source_binary_pairs": len(binp), "seen_pairs_matching_source": len(seen & pairs), "binary_pairs_seen": len(binp & seen),
           "fraction_binary_seen": len(binp & seen) / max(len(binp), 1), "unmatched_seen_pairs": len(unmatched),
           "unmatched_examples": rnd.sample(unmatched, min(5, len(unmatched))), "unseen_binary_pairs": len(unseen_bin),
           "unseen_binary_examples": rnd.sample(unseen_bin, min(5, len(unseen_bin))), "images_with_unseen_binary": len({i for i, _ in unseen_bin})}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"mixstats_{a.source}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0


def cmd_freeze(a):
    """Mark the manifest frozen (only after the coordinator's go on the protocol); records who / when and re-checks the sha256."""
    out = Path(a.out); meta = json.loads((out / "d1_manifest_meta.json").read_text())
    if sha256_file(out / "d1_manifest.jsonl") != meta["manifest_sha256"]:
        raise SystemExit("manifest changed since it was built")
    if getattr(a, "mix665k", None):                                                   # disclosure: manifest images inside LLaVA-1.5's GQA mix
        imgs = {Path(str(e.get("image", ""))).stem for e in json.loads(Path(a.mix665k).read_text()) if str(e.get("image", "")).startswith("gqa/")}
        rows = load_manifest(out)
        meta["image_exposure"] = {role: {"images": len({r["image_id"] for r in rows if r["role"] == role}),
                                         "in_llava_gqa_mix": len({r["image_id"] for r in rows if r["role"] == role} & imgs)} for role in ("FIT", "CAL", "DEV")}
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
    m.add_argument("--source", choices=["balanced", "all"], default="balanced"); m.add_argument("--balance", action="store_true")
    ms = sub.add_parser("mixstats"); ms.add_argument("--questions", required=True); ms.add_argument("--source", choices=["balanced", "all"], default="balanced")
    ms.add_argument("--mix665k", required=True); ms.add_argument("--out", required=True)
    s = sub.add_parser("shards"); s.add_argument("--out", required=True); s.add_argument("--images-per-shard", type=int, default=250)
    s.add_argument("--shards-per-bundle", type=int, default=4)
    f = sub.add_parser("feed"); f.add_argument("--out", required=True); f.add_argument("--lines", required=True)
    z = sub.add_parser("zipcheck"); z.add_argument("--out", required=True); z.add_argument("--zip", required=True)
    fr = sub.add_parser("freeze"); fr.add_argument("--out", required=True); fr.add_argument("--protocol", required=True); fr.add_argument("--go", required=True)
    fr.add_argument("--mix665k", default=None)
    a = ap.parse_args()
    return {"manifest": cmd_manifest, "shards": cmd_shards, "feed": cmd_feed, "zipcheck": cmd_zipcheck, "freeze": cmd_freeze, "mixstats": cmd_mixstats}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
