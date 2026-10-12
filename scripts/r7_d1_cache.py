"""r7 R7-30 D1-20 feature cache (SparseVLM / LLaVA env, H100 / L40S): the selector's cheap inputs, computed once.
  - visual: LLaVA-1.5 projected visual tokens (vision tower + mm_projector, official fp32 tower path) -> PCA 4096 -> 512 (fitted on FIT
    images only: streaming covariance, eigh) -> fp16 [n_img, 576, 512] per role (.npy) + image index
  - question: LLaVA tokenizer ids of the question text (no special tokens) per qid; the frozen input-embedding table (fp16 [V, 4096])
Modes:  fit_pca  --manifest-dir D --out C [--max-images 10000]    (FIT images only; 64 seeded tokens per image)
        encode   --manifest-dir D --out C --role FIT|CAL|DEV         (pixels only; no label, no teacher output is read)
        questions --manifest-dir D --out C                           (token ids for every manifest question + the embedding table)
        smoke    --fixture F --out C                                 (R7-20 GQA fixture: PCA fitted on the fixture images; smoke only)
        d1_20    --manifest-dir D --out C                            (one load: PCA on FIT, FIT / CAL features, FIT / CAL question ids)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_vl_common as V  # noqa: E402


def host(out):
    import r7_d1_teacher as TE
    return TE.LlavaPrefill(Path(out)), TE.ImageSource()


def images_of(manifest_dir: Path, role: str):
    rows = [json.loads(x) for x in (manifest_dir / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
    seen, out = set(), []
    for r in rows:
        if r["role"] == role and r["image_id"] not in seen:
            seen.add(r["image_id"]); out.append(r["image_id"])
    return out, rows


def token_subsample(image_id: str, n_tok=576, k=64):
    """Coordinator decision 2026-10-12: a fixed, seeded subsample of 64 visual tokens per FIT image (seed from sha256 of the image id)."""
    import hashlib
    hh = int(hashlib.sha256(f"r7-30/d1-pca/{image_id}".encode()).hexdigest()[:16], 16)
    return np.sort(np.random.default_rng([20261012, hh]).choice(n_tok, size=k, replace=False))


def fit_pca(h, items, dim=512, chunk=16, k=64):
    """items: iterable of (image_id, PIL image).  Streaming fp64 covariance over the seeded token subsample, then eigh."""
    torch = h.torch; S = torch.zeros(4096, 4096, dtype=torch.float64, device="cuda"); m = torch.zeros(4096, dtype=torch.float64, device="cuda"); n = 0
    buf = []

    def add(batch):
        nonlocal S, m, n
        f = h.encode([im for _, im in batch])
        sel = torch.cat([f[j, torch.as_tensor(token_subsample(iid, f.shape[1], k), device=f.device)] for j, (iid, _) in enumerate(batch)]).double()
        S += sel.T @ sel; m += sel.sum(0); n += sel.shape[0]
    for item in items:
        buf.append(item)
        if len(buf) == chunk:
            add(buf); buf = []
    if buf:
        add(buf)
    mean = m / n; cov = S / n - torch.outer(mean, mean); evals, evecs = torch.linalg.eigh(cov)
    basis = evecs[:, -dim:].flip(1); explained = float(evals[-dim:].sum() / evals.clamp(min=0).sum())
    return {"mean": mean.float().cpu(), "basis": basis.float().cpu(), "explained_variance": explained, "tokens": int(n)}


def encode_role(h, pca, pil_list, chunk=16):
    torch = h.torch; mean, basis = pca["mean"].cuda(), pca["basis"].cuda(); out = []
    for c in range(0, len(pil_list), chunk):
        f = h.encode(pil_list[c:c + chunk]).float(); out.append(((f - mean) @ basis).half().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 576, basis.shape[1]), np.float16)


def save_questions(h, rows, out: Path):
    torch = h.torch
    with open(out / "question_ids.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"qid": r["qid"], "ids": h.tok(r["question"], add_special_tokens=False).input_ids}) + "\n")
    torch.save(h.embed.weight.detach().half().cpu(), out / "embed_table.pt")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit_pca", "encode", "questions", "smoke", "d1_20"]); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-dir"); ap.add_argument("--role", choices=["FIT", "CAL", "DEV"]); ap.add_argument("--max-images", type=int, default=10000)
    ap.add_argument("--fixture"); ap.add_argument("--pca-dim", type=int, default=512)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); h, src = host(out); torch = h.torch; t0 = time.time()
    if a.mode == "smoke":
        from PIL import Image
        rows = [json.loads(x) for x in Path(a.fixture).read_text().splitlines()]; rows = [r for r in rows if r.get("binary")]
        ids = sorted({r["image_id"] for r in rows}); pil = [Image.open(next(r["image_file"] for r in rows if r["image_id"] == i)).convert("RGB") for i in ids]
        pca = fit_pca(h, zip(ids, pil), a.pca_dim); torch.save(pca, out / "pca.pt")
        np.save(out / "feats_PROBE.npy", encode_role(h, pca, pil)); (out / "images_PROBE.json").write_text(json.dumps(ids))
        save_questions(h, rows, out)
    else:
        md = Path(a.manifest_dir); meta = json.loads((md / "d1_manifest_meta.json").read_text())
        if meta.get("frozen") is not True:
            raise SystemExit("manifest not frozen")
        if a.mode == "d1_20":                                                          # one load: PCA on FIT, FIT / CAL features, FIT / CAL questions
            imgs, rows = images_of(md, "FIT")
            pca = fit_pca(h, ((i, src.open(i)) for i in imgs[: a.max_images]), a.pca_dim)
            pca.update({"fit_images": min(len(imgs), a.max_images), "tokens_per_image": 64, "manifest_sha256": meta["manifest_sha256"]}); torch.save(pca, out / "pca.pt")
            for role in ("FIT", "CAL"):                                                # DEV is encoded at D1-30, not here
                ri, _ = images_of(md, role); np.save(out / f"feats_{role}.npy", encode_role(h, pca, [src.open(i) for i in ri]))
                (out / f"images_{role}.json").write_text(json.dumps(ri))
            save_questions(h, [r for r in rows if r["role"] in ("FIT", "CAL")], out)
        elif a.mode == "fit_pca":
            imgs, _ = images_of(md, "FIT"); imgs = imgs[: a.max_images]                 # FIT only; never CAL / DEV
            pca = fit_pca(h, ((i, src.open(i)) for i in imgs), a.pca_dim)
            pca.update({"fit_images": len(imgs), "tokens_per_image": 64, "manifest_sha256": meta["manifest_sha256"]})
            torch.save(pca, out / "pca.pt")
        elif a.mode == "encode":
            pca = torch.load(out / "pca.pt"); imgs, _ = images_of(md, a.role)
            np.save(out / f"feats_{a.role}.npy", encode_role(h, pca, [src.open(i) for i in imgs])); (out / f"images_{a.role}.json").write_text(json.dumps(imgs))
        else:
            _, rows = images_of(md, "FIT"); save_questions(h, [r for r in rows if r["role"] in ("FIT", "CAL")], out)
    rec = {"mode": a.mode, "role": a.role, "seconds": time.time() - t0, "env": V.env_info()}
    (out / f"cache_{a.mode}{'_' + a.role if a.role else ''}.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
