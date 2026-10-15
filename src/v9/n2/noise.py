"""V9-N2 mechanism mismatch (package §05.2): pairs are (caption, image).  U pairs receive a design fraction rho of mismatches by shuffling the
images of the selected U pairs among themselves, donors only from U, so A stays clean.  This follows the official NCR / L2RM construction
(`PrecompDataset`: a random subset of captions, their image indices permuted among the subset) restricted to U; the mapping therefore differs from
the official noise index (which draws donors from the whole train split).  The official index is used only for the A-table native reproduction.

Outputs: an OBSERVED pair table (pair_id, caption_id, observed_image_id, role) that critic and student loaders read, and a sealed EVAL-ONLY file
(pair_id, original_image_id, mismatched) for the offline scorer.  The manifest records design_rho, realized_rho (fraction of U pairs whose observed
image differs from the original; a permutation can map a pair back to its own image, so realized <= design), per-image flags, the mapping hash and
the caption-group handling (every caption of an image is its own pair; mismatch is drawn per pair).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EVAL_ONLY_SUFFIX = ".eval_only.json"


def make_pair_table(caption_image: dict, roles_of_image: dict) -> list[dict]:
    """caption_image: {caption_id: image_id} (every caption is one pair).  Returns rows sorted by caption id with the image's role."""
    rows = []
    for cap in sorted(caption_image, key=lambda c: (str(c))):
        img = str(caption_image[cap])
        if img not in roles_of_image:
            raise ValueError(f"image {img} of caption {cap} has no role")
        rows.append({"pair_id": f"p{cap}", "caption_id": str(cap), "image_id": img, "role": roles_of_image[img]})
    return rows


def inject_mismatch(pairs: list[dict], design_rho: float, seed: int) -> tuple[list[dict], dict, dict]:
    """Return (observed_rows, eval_only_labels, manifest).  Only role 'U' pairs can be mismatched; donors are the images of the selected U pairs."""
    if not 0 <= design_rho < 1:
        raise ValueError("design_rho must be in [0, 1)")
    u_idx = [k for k, r in enumerate(pairs) if r["role"] == "U"]
    rng = np.random.default_rng([int(seed), 9_004, int(round(design_rho * 1000))])
    n_sel = int(round(design_rho * len(u_idx)))
    sel = sorted(rng.choice(len(u_idx), size=n_sel, replace=False).tolist()) if n_sel else []
    sel_rows = [u_idx[s] for s in sel]
    donors = [pairs[k]["image_id"] for k in sel_rows]
    perm = rng.permutation(len(donors)) if donors else np.array([], dtype=int)
    observed, labels = [], {}
    for k, r in enumerate(pairs):
        obs = dict(r); obs.pop("image_id"); obs["observed_image_id"] = r["image_id"]; observed.append(obs)
    for j, k in enumerate(sel_rows):
        observed[k]["observed_image_id"] = donors[int(perm[j])]
    n_mis = 0
    for k, r in enumerate(pairs):
        mis = observed[k]["observed_image_id"] != r["image_id"]; n_mis += int(mis)
        labels[r["pair_id"]] = {"original_image_id": r["image_id"], "mismatched": bool(mis)}
    img_flag = {}
    for k, r in enumerate(pairs):
        if r["role"] == "U":
            img_flag[r["image_id"]] = img_flag.get(r["image_id"], False) or labels[r["pair_id"]]["mismatched"]
    mapping_hash = hashlib.sha256("\n".join(f"{o['pair_id']}\t{o['observed_image_id']}" for o in observed).encode()).hexdigest()
    manifest = {"design_rho": design_rho, "seed": int(seed), "n_pairs": len(pairs), "n_U_pairs": len(u_idx), "n_selected": n_sel,
                "n_mismatched": n_mis, "realized_rho": (n_mis / len(u_idx)) if u_idx else 0.0, "n_A_mismatched": 0,
                "donor_policy": "images of the selected U pairs permuted among themselves (NCR construction restricted to U); A untouched",
                "caption_group_handling": "each caption is one pair; mismatch drawn per pair; all captions of an image share its role",
                "images_with_any_mismatch": int(sum(img_flag.values())), "mapping_sha256": mapping_hash,
                "official_noise_index_used": False}
    return observed, labels, manifest


def write_noise(observed: list[dict], labels: dict, manifest: dict, out_dir: str | Path, tag: str) -> dict:
    """Observed table + manifest go to out_dir; the labels go to out_dir/eval_only/<tag>.eval_only.json (scorer only)."""
    out = Path(out_dir); (out / "eval_only").mkdir(parents=True, exist_ok=True)
    obs_path = out / f"{tag}.observed_pairs.json"; obs_path.write_text(json.dumps(observed, indent=0))
    lab_path = out / "eval_only" / f"{tag}{EVAL_ONLY_SUFFIX}"; lab_path.write_text(json.dumps(labels, indent=0))
    man = dict(manifest); man["observed_sha256"] = hashlib.sha256(obs_path.read_bytes()).hexdigest(); man["eval_only_sha256"] = hashlib.sha256(lab_path.read_bytes()).hexdigest()
    (out / f"{tag}.noise_manifest.json").write_text(json.dumps(man, indent=1))
    return {"observed": str(obs_path), "eval_only": str(lab_path), "manifest": man}


def load_observed(path: str | Path) -> list[dict]:
    """The only reader the critic / student side may use.  Refuses eval-only files by construction (different schema, different directory)."""
    p = Path(path)
    if EVAL_ONLY_SUFFIX in p.name or "eval_only" in p.parts:
        raise ValueError("refusing to load an eval-only label file into the training side")
    rows = json.loads(p.read_text())
    for r in rows:
        if "original_image_id" in r or "mismatched" in r:
            raise ValueError("observed table carries labels; refusing")
    return rows
