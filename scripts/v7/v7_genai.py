"""V7 GenAI-Rank inputs without any human label.  The Rank release ships its prompts only inside human_rating.json (closed).  Here the 800
Rank prompt ids (1..1611, a numbering different from GenAI-Bench's Index 0..1599) are mapped to GenAI-Bench prompts through the IMAGES: the
GenAI-Bench parquet carries, per prompt, one DALLE_3 and one SDXL_Base image; a Rank candidate that is the same image (sha256, else dHash
distance <= 4) ties the ids.  Columns read from GenAI-Bench: Index, Prompt, Tags and the two generators' image columns (HumanRatings never).
Skill tags (genai_skills.json by Rank id; Tags by Index) are an independent consistency check.
    python -I scripts/v7/v7_genai.py rank-inputs --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_data as VD  # noqa: E402

RANK = Path("/projects/EEG-foundation-model/yinghao/data/genai_rank"); BENCH = Path("/projects/EEG-foundation-model/yinghao/data/genai_bench/data")
GENS = ("DALLE_3", "SDXL_Base"); N_CAND = 9


def image_bytes(cell):
    """A parquet image cell -> bytes (HF Image feature: struct {bytes, path})."""
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell


def cmd_rank_inputs(a):
    import pyarrow.parquet as pq
    from PIL import Image
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rep = {"generators": {}}
    cands = []
    for g in GENS:
        z = zipfile.ZipFile(RANK / f"{g}.zip"); per = defaultdict(dict); extras = 0
        for n in z.namelist():
            m = re.fullmatch(rf"{g}/(\d+)_(\d+)\.jpeg", n)
            if not m:
                continue
            pid, k = int(m.group(1)), int(m.group(2))
            if 1 <= k <= N_CAND:
                per[pid][k] = n
            else:
                extras += 1
        incomplete = sorted(p for p, v in per.items() if set(v) != set(range(1, N_CAND + 1)))
        for pid in sorted(per):
            for k in range(1, N_CAND + 1):
                n = per[pid].get(k)
                if n is None:
                    continue
                data = z.read(n); im = Image.open(io.BytesIO(data)); im.load()
                cands.append({"prompt_id": f"{pid:05d}", "generator_id": g, "image_id": n, "candidate_index": k - 1, "image_sha256": hashlib.sha256(data).hexdigest(),
                              "dhash": VD.dhash64(im), "width": im.size[0], "height": im.size[1]})
        rep["generators"][g] = {"prompts": len(per), "candidates": sum(len(v) for v in per.values()), "extra_files_not_candidates": extras, "incomplete_pools": incomplete}
    files = sorted(BENCH.glob("*.parquet")); schema = pq.read_schema(files[0]).names
    label_like = [c for c in schema if re.search(r"score|rating|human|vote|label|rank|annot", c, re.I)]; cols = ["Index", "Prompt", "Tags", *GENS]
    assert not set(cols) & set(label_like), "refusing to read a label-like column"
    bench = {}
    for f in files:
        t = pq.read_table(f, columns=cols).to_pydict()
        for i in range(len(t["Index"])):
            row = {"index": int(t["Index"][i]), "prompt": t["Prompt"][i], "tags": t["Tags"][i], "img": {}}
            for g in GENS:
                b = image_bytes(t[g][i])
                if b:
                    im = Image.open(io.BytesIO(b)); im.load(); row["img"][g] = (hashlib.sha256(b).hexdigest(), VD.dhash64(im))
            bench[row["index"]] = row
    by_sha = {(g, v[0]): i for i, r in bench.items() for g, v in r["img"].items()}; votes = defaultdict(Counter)
    for c in cands:
        hit = by_sha.get((c["generator_id"], c["image_sha256"]))
        if hit is not None:
            votes[c["prompt_id"]][(hit, "sha256")] += 1; continue
        for i, r in bench.items():
            v = r["img"].get(c["generator_id"])
            if v and bin(v[1] ^ c["dhash"]).count("1") <= VD.DHASH_MAX_HAMMING:
                votes[c["prompt_id"]][(i, "dhash")] += 1
    mapping, conflicts = {}, {}
    for pid, cnt in votes.items():
        idx = Counter()
        for (i, how), n in cnt.items():
            idx[i] += n
        if len(idx) == 1:
            mapping[pid] = next(iter(idx))
        else:
            conflicts[pid] = {str(k): v for k, v in idx.items()}
    rank_ids = sorted({c["prompt_id"] for c in cands}); unmapped = [p for p in rank_ids if p not in mapping]
    inv = Counter(mapping.values()); many_to_one = {str(i): n for i, n in inv.items() if n > 1}
    skills = json.loads((RANK / "genai_skills.json").read_text()); rank_sk = defaultdict(set)
    for sk, ids in skills.items():
        for i in ids:
            rank_sk[f"{int(i):05d}"].add(sk.lower())
    agree = disagree = 0
    for pid, i in mapping.items():
        tags = bench[i]["tags"]; text = json.dumps(tags).lower()
        if all(sk in text for sk in rank_sk.get(pid, set()) if sk not in ("basic", "advanced", "universal")):
            agree += 1
        else:
            disagree += 1
    rep.update({"rank_prompt_ids": len(rank_ids), "mapped": len(mapping), "unmapped": unmapped[:50], "n_unmapped": len(unmapped), "conflicts": conflicts,
                "many_to_one": many_to_one, "skill_tag_check": {"consistent": agree, "inconsistent": disagree}, "bench_rows": len(bench),
                "bench_columns_read": cols, "label_like_columns_not_read": label_like, "seconds": time.time() - t0})
    with open(out / "genai_rank_inputs.jsonl", "w") as fh:
        for c in cands:
            i = mapping.get(c["prompt_id"]); row = {k: v for k, v in c.items() if k != "dhash"}
            row.update({"bench_index": i, "prompt": bench[i]["prompt"] if i is not None else None})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "genai_rank_inputs_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(json.dumps({k: rep[k] for k in ("generators", "rank_prompt_ids", "mapped", "n_unmapped", "many_to_one", "skill_tag_check")}, indent=1))
    return 0 if not unmapped and not conflicts and not many_to_one else 5


def cmd_seal_rank(a):
    """ONLY with the owner's approval (--approved-by records it): split the closed human_rating.json into (i) the prompt text per Rank id -> an
    input file, and (ii) the ratings -> a sealed labels file (mode 0400) in closed/.  Nothing about the ratings is printed: counts and sha256 only.
    The candidate-structure checks (both generators, images 1..9) run on the file without reporting any value."""
    import os
    import stat
    if not a.approved_by:
        raise SystemExit("refusing: the sealed split of human_rating.json needs the owner's approval (--approved-by '<who, when>')")
    src = RANK / "closed" / "human_rating.json"; raw = src.read_bytes(); d = json.loads(raw); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prompts, sealed, problems = [], [], Counter()
    for pid, rec in d.items():
        if str(rec.get("id")) != str(pid):
            problems["id_mismatch"] += 1
        prompts.append({"prompt_id": f"{int(pid):05d}", "prompt": rec["prompt"]})
        for g in GENS:
            imgs = rec.get("models", {}).get(g, {})
            if sorted(int(k) for k in imgs) != list(range(1, N_CAND + 1)):
                problems[f"{g}_not_1_to_9"] += 1
            for k, ratings in imgs.items():
                if 1 <= int(k) <= N_CAND:
                    sealed.append({"prompt_id": f"{int(pid):05d}", "generator_id": g, "image_id": f"{g}/{int(pid):05d}_{int(k):02d}.jpeg", "candidate_index": int(k) - 1,
                                   "human_score": float(sum(ratings) / len(ratings)), "ratings": ratings})
    pf = out / "genai_rank_prompts.jsonl"; pf.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prompts) + "\n")
    lf = RANK / "closed" / "genai_rank_labels_sealed.jsonl"
    if lf.exists():
        os.chmod(lf, stat.S_IRUSR | stat.S_IWUSR)
    lf.write_text("\n".join(json.dumps(r) for r in sealed) + "\n"); os.chmod(lf, stat.S_IRUSR)
    inputs = [json.loads(l) for l in open(out / "genai_rank_inputs.jsonl")]; by = {r["prompt_id"]: r["prompt"] for r in prompts}
    for r in inputs:
        r["prompt"] = by.get(r["prompt_id"]); r["prompt_source"] = "human_rating.json prompt field (sealed split, approved)"
    (out / "genai_rank_inputs.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in inputs) + "\n")
    rep = {"approved_by": a.approved_by, "source_sha256": hashlib.sha256(raw).hexdigest(), "prompts": len(prompts), "sealed_rows": len(sealed),
           "sealed_sha256": hashlib.sha256(lf.read_bytes()).hexdigest(), "prompts_sha256": hashlib.sha256(pf.read_bytes()).hexdigest(),
           "candidates_without_prompt": sum(1 for r in inputs if not r["prompt"]), "structure_problems": dict(problems)}
    (out / "genai_rank_seal_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0 if not problems and not rep["candidates_without_prompt"] else 6


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank-inputs"); r.add_argument("--out", required=True)
    sl = sub.add_parser("seal-rank"); sl.add_argument("--out", required=True); sl.add_argument("--approved-by", default=None)
    a = ap.parse_args(argv)
    return cmd_rank_inputs(a) if a.cmd == "rank-inputs" else cmd_seal_rank(a)


if __name__ == "__main__":
    sys.exit(main())
