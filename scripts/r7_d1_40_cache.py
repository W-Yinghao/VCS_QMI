"""r7 R7-40b feature cache (Qwen env, any of RTX6000PRO / H100 / L40S): the selector's cheap inputs, computed once from pixels and question
text only (no label, no teacher output); GQA images from the extracted folder only (no zip fallback: archives deleted 2026-10-12).
  - visual: merged Qwen2.5-VL visual tokens (official processor + vision tower + merger, bf16) -> PCA 3584 -> 512 fitted on FIT images only
    (all 10,000 FIT images, a seeded subsample of 64 tokens per image (seed sha256('r7-40/d1-pca/<imageId>')), streaming fp64 covariance,
    eigh) -> fp16 features of every token, all images of a role concatenated: feats_<role>.npy [Σ N_i, 512] + images_<role>.json (image ids,
    offsets, N_i, merged grid rows × cols)
  - question: Qwen tokenizer ids of the raw question (no special tokens) per qid; the frozen input-embedding table (bf16 [152064, 3584])
Modes:  d1_40  --manifest-dir D --out C     (one load: PCA on FIT, FIT / CAL features, FIT / CAL question ids; DEV is encoded at R7-40c)
        encode --manifest-dir D --out C --role DEV --allow-dev   (R7-40c only)
        smoke  --fixture F --out C           (R7-20 GQA fixture binary items: PCA on the fixture images; smoke only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_vl_common as V  # noqa: E402

CHUNK = 16


def token_subsample40(image_id: str, n_tok: int, k: int = 64) -> np.ndarray:
    """A fixed, seeded subsample of k of the image's n_tok merged tokens (the frozen D1-20 PCA rule, Qwen stream)."""
    hh = int(hashlib.sha256(f"r7-40/d1-pca/{image_id}".encode()).hexdigest()[:16], 16)
    return np.sort(np.random.default_rng([20261012, hh]).choice(n_tok, size=k, replace=False))


def chunks(ids, opener, size=CHUNK):
    for c in range(0, len(ids), size):
        part = ids[c:c + size]; yield part, [opener(i) for i in part]


def fit_pca40(host, ids, opener, dim=512, k=64):
    """Streaming fp64 covariance of the seeded token subsample of every image in ids, then eigh; top-dim basis."""
    torch = host.torch; D = int(host.core.embed.weight.shape[1])                       # 3584 (LLM input width = merger output width)
    S = torch.zeros(D, D, dtype=torch.float64, device="cuda"); m = torch.zeros(D, dtype=torch.float64, device="cuda"); n = 0
    for part, pil in chunks(ids, opener):
        enc = host.encode(pil)
        sel = torch.cat([f[torch.as_tensor(token_subsample40(iid, f.shape[0], k), device=f.device)] for iid, (f, _, _) in zip(part, enc)]).double()
        S += sel.T @ sel; m += sel.sum(0); n += sel.shape[0]
    mean = m / n; cov = S / n - torch.outer(mean, mean); evals, evecs = torch.linalg.eigh(cov)
    basis = evecs[:, -dim:].flip(1); explained = float(evals[-dim:].sum() / evals.clamp(min=0).sum())
    return {"mean": mean.float().cpu(), "basis": basis.float().cpu(), "explained_variance": explained, "tokens": int(n)}


def encode_role40(host, pca, ids, opener):
    """fp16 PCA features of every token of every image (concatenated) + per-image N and grid."""
    torch = host.torch; mean, basis = pca["mean"].cuda(), pca["basis"].cuda(); feats, ns, grids = [], [], []
    for part, pil in chunks(ids, opener):
        for f, grid, _ in host.encode(pil):
            feats.append(((f.float() - mean) @ basis).half().cpu().numpy()); ns.append(int(f.shape[0])); grids.append([int(grid[0]), int(grid[1])])
    offsets = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(np.int64) if ns else np.zeros(0, np.int64)
    return (np.concatenate(feats) if feats else np.zeros((0, basis.shape[1]), np.float16)), {"images": list(ids), "offsets": offsets.tolist(), "n": ns, "grid": grids}


def save_questions(host, rows, out: Path):
    torch = host.torch
    with open(out / "question_ids.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"qid": r["qid"], "ids": host.tok(r["question"], add_special_tokens=False).input_ids}) + "\n")
    torch.save(host.core.embed.weight.detach().cpu().clone(), out / "embed_table.pt")                  # bf16, as the model


def images_of(manifest_dir: Path, role: str):
    rows = [json.loads(x) for x in (manifest_dir / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
    seen, out = set(), []
    for r in rows:
        if r["role"] == role and r["image_id"] not in seen:
            seen.add(r["image_id"]); out.append(r["image_id"])
    return out, rows


def write_role(out: Path, role: str, feats, index):
    np.save(out / f"feats_{role}.npy", feats); (out / f"images_{role}.json").write_text(json.dumps(index))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["d1_40", "encode", "smoke"]); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-dir"); ap.add_argument("--role", choices=["FIT", "CAL", "DEV"]); ap.add_argument("--allow-dev", action="store_true")
    ap.add_argument("--max-images", type=int, default=10000); ap.add_argument("--fixture"); ap.add_argument("--pca-dim", type=int, default=512)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import r7_d1_40_host as QH
    import r7_d1_40_teacher as TE40                                                     # extracted-folder image source (no zip fallback)
    host = QH.QwenPrefill(out); torch = host.torch; t0 = time.time(); rec = {"mode": a.mode, "role": a.role}
    if a.mode == "smoke":
        from PIL import Image
        rows = [json.loads(x) for x in Path(a.fixture).read_text().splitlines()]; rows = [r for r in rows if r.get("binary")]
        files = {r["image_id"]: r["image_file"] for r in rows}; ids = sorted(files); opener = lambda i: Image.open(files[i]).convert("RGB")  # noqa: E731
        pca = fit_pca40(host, ids, opener, a.pca_dim); torch.save(pca, out / "pca.pt")
        feats, index = encode_role40(host, pca, ids, opener); write_role(out, "PROBE", feats, index); save_questions(host, rows, out)
        rec.update({"explained_variance": pca["explained_variance"], "tokens": pca["tokens"], "images": len(ids)})
    else:
        md = Path(a.manifest_dir); meta = json.loads((md / "d1_manifest_meta.json").read_text())
        if meta.get("frozen") is not True or hashlib.sha256((md / "d1_manifest.jsonl").read_bytes()).hexdigest() != meta.get("manifest_sha256"):
            raise SystemExit("manifest not frozen or changed")
        src = TE40.ExtractedImageSource(); opener = src.open
        if a.mode == "d1_40":                                                          # one load: PCA on FIT, FIT / CAL features and questions
            fit_ids, rows = images_of(md, "FIT"); t1 = time.time()
            pca = fit_pca40(host, fit_ids[: a.max_images], opener, a.pca_dim)
            pca.update({"fit_images": min(len(fit_ids), a.max_images), "tokens_per_image": 64, "manifest_sha256": meta["manifest_sha256"],
                        "host": "Qwen2.5-VL-7B-Instruct", "seconds": time.time() - t1}); torch.save(pca, out / "pca.pt")
            rec.update({"explained_variance": pca["explained_variance"], "pca_tokens": pca["tokens"], "pca_seconds": pca["seconds"]})
            for role in ("FIT", "CAL"):                                                # DEV is encoded at R7-40c, not here
                ri, _ = images_of(md, role); t1 = time.time(); feats, index = encode_role40(host, pca, ri, opener); write_role(out, role, feats, index)
                rec[f"{role}_images"], rec[f"{role}_tokens"], rec[f"{role}_seconds"] = len(ri), int(feats.shape[0]), time.time() - t1
                rec[f"{role}_grid_max"] = [max(g[0] for g in index["grid"]), max(g[1] for g in index["grid"])]
            save_questions(host, [r for r in rows if r["role"] in ("FIT", "CAL")], out)
        else:
            if a.role == "DEV" and not a.allow_dev:
                raise SystemExit("DEV features are encoded at R7-40c only (--allow-dev)")
            pca = torch.load(out / "pca.pt", weights_only=False); ri, _ = images_of(md, a.role)
            feats, index = encode_role40(host, pca, ri, opener); write_role(out, a.role, feats, index)
    rec.update({"seconds": time.time() - t0, "load_seconds": host.load_seconds, "host": host.info, "env": V.env_info(), "peak_mem_gb": V.cuda_peak_gb(torch)})
    (out / f"cache_{a.mode}{'_' + a.role if a.role else ''}.json").write_text(V.jdump(rec, indent=1)); print(V.jdump(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
