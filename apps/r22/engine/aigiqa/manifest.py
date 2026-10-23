"""Immutable, prompt-grouped annotation preparation; no target labels in inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

INPUT_FIELDS = ["sample_id", "dataset", "image", "image_sha256", "prompt", "prompt_group", "generator", "split"]
LABEL_FIELDS = ["sample_id", "quality_mos", "alignment_mos"]
SOURCE_COMMIT = "4108d1d6c26c5494ddac7bb71b11ecdf67203d95"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_prompt(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompt must be a nonempty string")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    text = " ".join(text.split())
    if not text:
        raise ValueError("Empty prompt after canonicalization")
    return text


def safe_image_path(root, relative):
    relative = str(relative).replace("\\", "/")
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or not relative or ":" in p.parts[0]:
        raise ValueError(f"Image path must be relative and remain under image_root: {relative!r}")
    root = Path(root).resolve()
    result = (root / p).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Image symlink escapes image_root: {relative!r}")
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _adapt(row, dataset):
    if dataset == "AGIQA-3k":
        image, prompt = row["name"], row["prompt"]
        quality, alignment = row["mos_quality"], row["mos_align"]
        # The complete filename prefix is retained; no claim that it identifies
        # a held-out generator. Use an explicit mapping before generator OOD.
        generator = row.get("generator", image.split("_")[0])
    elif dataset == "AIGCIQA2023":
        image, prompt = row["model"] + "/" + row["image_name"], row["prompt"]
        quality, alignment, generator = row["mosz1"], row["mosz3"], row["model"]
    elif dataset == "PKU-AIGIQA-T2I":
        if row.get("Image_prompt", "").strip():
            raise ValueError("PKU adapter accepts T2I only; Image_prompt must be empty")
        image, prompt = row["Generated_image"], row["Text_prompt"]
        quality, alignment = row["MOS_q"], row["MOS_c"]
        generator = row.get("generator", image.split("_")[0])
    elif dataset == "canonical":
        image, prompt = row["image"], row["prompt"]
        quality, alignment = row["quality_mos"], row["alignment_mos"]
        generator = row.get("generator", "unknown")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    quality, alignment = float(quality), float(alignment)
    if not all(math.isfinite(s) and 0.0 <= s <= 5.0 for s in [quality, alignment]):
        raise ValueError("Expected raw MOS on declared [0, 5] scale; do not infer a scale from TEST")
    return dict(image=image, prompt=prompt, quality_mos=quality, alignment_mos=alignment, generator=generator)


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def prepare(source_csv, image_root, output, dataset, seed=20261019, verify_images=True,
            fit_fraction=0.7, cal_fraction=0.15, near_threshold=0.97):
    """Prepare once; duplicate images and conservative near-prompts share a split.

    Near merging affects split assignment only, never positive-pair construction.
    Canonical prompt equality defines candidate selection groups. split_group can
    combine distinct prompts to prevent very similar prompts crossing splits.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a prepared dataset: {output}")
    if not (0 < fit_fraction < 1 and 0 < cal_fraction < 1 and fit_fraction + cal_fraction < 1):
        raise ValueError("FIT/CAL fractions invalid")
    rows = [_adapt(r, dataset) for r in read_csv(source_csv)]
    if not rows:
        raise ValueError("Empty annotation file")
    prompts = sorted({canonical_prompt(r["prompt"]) for r in rows})
    uf, merges = UnionFind(prompts), []
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0,1]")
    for i, a in enumerate(prompts):
        aa = set(a.split())
        for b in prompts[i + 1:]:
            bb = set(b.split())
            if len(a) >= 20 and len(b) >= 20 and len(aa & bb) / max(len(aa | bb), 1) >= .90:
                sim = SequenceMatcher(None, a, b, autojunk=False).ratio()
                if sim >= near_threshold:
                    uf.union(a, b)
                    merges.append({"prompt_a": a, "prompt_b": b, "reason": "near_prompt", "similarity": sim})
    seen_images, seen_ids = {}, set()
    effective_dataset = dataset if dataset != "canonical" else "synthetic_or_custom"
    for row in rows:
        norm = canonical_prompt(row["prompt"])
        path = safe_image_path(image_root, row["image"])
        if verify_images and not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        checksum = sha256_file(path) if verify_images else "NOT_VERIFIED"
        if verify_images and checksum in seen_images:
            other = seen_images[checksum]
            uf.union(norm, other)
            if norm != other:
                merges.append({"prompt_a": norm, "prompt_b": other, "reason": "same_image_bytes", "similarity": 1.0})
        if verify_images:
            seen_images[checksum] = norm
        sid = hashlib.sha256((effective_dataset + "\0" + row["image"]).encode()).hexdigest()[:24]
        if sid in seen_ids:
            raise ValueError(f"Repeated image path / sample id in source: {row['image']}")
        seen_ids.add(sid)
        row.update(sample_id=sid, dataset=effective_dataset, image_sha256=checksum,
                   prompt_group=hashlib.sha256(norm.encode()).hexdigest()[:24], _norm=norm)
    groups = sorted({uf.find(p) for p in prompts}, key=lambda p: hashlib.sha256(f"{seed}\0{p}".encode()).hexdigest())
    if len(groups) < 6:
        raise ValueError("At least six independent prompt/image groups are required")
    n_fit = min(len(groups) - 4, max(2, int(len(groups) * fit_fraction)))
    n_cal = min(len(groups) - n_fit - 2, max(2, int(len(groups) * cal_fraction)))
    split_of = {g: "fit" if i < n_fit else "cal" if i < n_fit + n_cal else "test" for i, g in enumerate(groups)}
    for row in rows:
        row["split"] = split_of[uf.find(row["_norm"])]
    output.mkdir(parents=True, exist_ok=True)
    for split in ["fit", "cal"]:
        write_csv(output / f"{split}.csv", [r for r in rows if r["split"] == split], INPUT_FIELDS + LABEL_FIELDS[1:])
    test = [r for r in rows if r["split"] == "test"]
    write_csv(output / "test_inputs.csv", test, INPUT_FIELDS)
    write_csv(output / "all_inputs.csv", rows, INPUT_FIELDS)
    write_csv(output / "sealed" / "test_labels.csv", test, LABEL_FIELDS)
    write_csv(output / "sealed" / "all_labels.csv", rows, LABEL_FIELDS)
    write_csv(output / "group_merges.csv", merges, ["prompt_a", "prompt_b", "reason", "similarity"])
    info = {
        "schema": "vcs_aigiqa_manifest_v1", "dataset": effective_dataset,
        "source_csv_sha256": sha256_file(source_csv), "image_root": str(Path(image_root).resolve()),
        "image_bytes_verified": verify_images, "source_reference_commit": SOURCE_COMMIT,
        "split_seed": seed, "fit_fraction": fit_fraction, "cal_fraction": cal_fraction,
        "near_threshold": near_threshold, "near_merge_rule": "NFKC/case/punctuation + char similarity>=threshold and token Jaccard>=0.9",
        "candidate_group_rule": "exact canonical prompt; near merges only affect split membership",
        "score_min": 0, "score_max": 5,
        "split_group_counts": {s: sum(v == s for v in split_of.values()) for s in ["fit", "cal", "test"]},
        "image_counts": {s: sum(r["split"] == s for r in rows) for s in ["fit", "cal", "test"]},
        "unique_image_byte_hashes": len(seen_images) if verify_images else None,
        "duplicate_image_byte_rows": len(rows) - len(seen_images) if verify_images else None,
        "input_and_label_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*.csv")},
        "roles": {"fit": "training", "cal": "hyperparameters/checkpoint", "test": "post-freeze evaluation only"},
        "warning": "sealed is a workflow boundary, not filesystem access control. Train does not open sealed files."
    }
    write_json(output / "manifest.json", info)
    return info


def verify_prepared(data_dir, require_images=True):
    root = Path(data_dir)
    info = json.loads((root / "manifest.json").read_text())
    if require_images and not info["image_bytes_verified"]:
        raise ValueError("Prepared manifest did not verify image bytes; formal training is blocked")
    # Deliberately verify training inputs only: no TEST label file is opened here.
    for name in ["fit.csv", "cal.csv"]:
        expected = info["input_and_label_hashes"][name]
        if sha256_file(root / name) != expected:
            raise ValueError(f"Frozen manifest was modified: {name}")
    fit, cal = read_csv(root / "fit.csv"), read_csv(root / "cal.csv")
    if {r["sample_id"] for r in fit} & {r["sample_id"] for r in cal}:
        raise ValueError("FIT/CAL sample overlap")
    if {r["prompt_group"] for r in fit} & {r["prompt_group"] for r in cal}:
        raise ValueError("FIT/CAL prompt overlap")
    if require_images and {r["image_sha256"] for r in fit} & {r["image_sha256"] for r in cal}:
        raise ValueError("FIT/CAL image-byte overlap")
    return info


def cross_dataset_audit(source_dir, target_inputs):
    """Read target metadata only; reject exact/near prompts and duplicate images.

    Reports overlaps so they can be declared replication or excluded via a frozen
    input selection before target labels are opened. No automatic label-based fix.
    """
    source = read_csv(Path(source_dir) / "fit.csv") + read_csv(Path(source_dir) / "cal.csv")
    target = read_csv(target_inputs)
    sp = {canonical_prompt(r["prompt"]) for r in source}
    sh = {r["image_sha256"] for r in source if r["image_sha256"] != "NOT_VERIFIED"}
    overlaps = []
    for row in target:
        p = canonical_prompt(row["prompt"])
        reason = []
        if p in sp:
            reason.append("same_prompt")
        elif any(len(p) >= 20 and len(q) >= 20 and len(set(p.split()) & set(q.split())) / max(len(set(p.split()) | set(q.split())), 1) >= .90 and SequenceMatcher(None, p, q, autojunk=False).ratio() >= .97 for q in sp):
            reason.append("near_prompt")
        if row.get("image_sha256") in sh:
            reason.append("same_image_bytes")
        if reason:
            overlaps.append({"sample_id": row["sample_id"], "reason": ";".join(reason)})
    return {"source_manifest_sha256": sha256_file(Path(source_dir) / "manifest.json"),
            "target_inputs_sha256": sha256_file(target_inputs), "overlap_count": len(overlaps), "overlaps": overlaps}
