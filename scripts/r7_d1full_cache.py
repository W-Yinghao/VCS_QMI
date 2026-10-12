"""r7 R7-30 D1-FULL feature cache: the frozen D1-20 cache format for FIT_FULL + CAL (LLaVA env, H100 / L40S).  DEV is not cached: the frozen
D1-20 design never encodes DEV here (D1-30 encodes its evaluation images inside the runner).  Files in --out, identical in format to
outputs/R7_D1/cache: pca.pt {mean, basis, explained_variance, tokens, fit_images, tokens_per_image, manifest_sha256}; feats_{FIT,CAL}.npy fp16
[n_img, 576, 512] in role order; images_{FIT,CAL}.json; question_ids.jsonl (FIT + CAL rows in manifest order); embed_table.pt (fp16 [32000, 4096]).
PCA basis — coordinator decision, recorded in cache_d1_full.json:
  fitfull (default) : the frozen fit_pca (64 seeded tokens per image, streaming fp64 covariance, eigh, 512 components) on ALL FIT_FULL images
  frozen            : the frozen D1-20 basis outputs/R7_D1/cache/pca.pt (fitted on the 10,000 frozen FIT images), copied byte for byte
Encoding: the frozen encode_role arithmetic ((f - mean) @ basis -> fp16) per chunk of 16 images in role order, streamed to disk (npy header + chunk
bytes; the file equals np.save of the frozen in-memory result, unit-tested) because FIT_FULL (~41 GB) does not fit the frozen in-memory path.
Images through the strict source (missing file -> MissingImage, no zip), read ahead by a thread pool (order preserved).  Each step is resumable
(a finished file is kept; partial files are written as .tmp and renamed); the marker cache_d1_full.json is written last.
    python scripts/r7_d1full_cache.py --manifest-dir DF/data --out DF/cache [--pca-basis fitfull|frozen]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_d1_cache as CA  # noqa: E402   (frozen D1-20 cache: token_subsample, fit_pca, save_questions; read-only use)
import r7_vl_common as V  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
D1, DF = OUT / "R7_D1", OUT / "R7_D1_FULL"
CHUNK = 16                                                                              # frozen encode_role / fit_pca chunk
DIM = 512


def sha256_file(p: Path):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            d.update(b)
    return d.hexdigest()


def prefetch(ids, opener, workers=8, ahead=64):
    """(id, PIL image) in input order; files opened by a thread pool `ahead` items in advance (a MissingImage propagates)."""
    from concurrent.futures import ThreadPoolExecutor
    it = iter(ids)
    with ThreadPoolExecutor(workers) as ex:
        q = collections.deque((i, ex.submit(opener, i)) for i in itertools.islice(it, ahead))
        while q:
            i, f = q.popleft()
            for j in itertools.islice(it, 1):
                q.append((j, ex.submit(opener, j)))
            yield i, f.result()


class NpyStream:
    """Write an .npy file chunk by chunk (format 1.0 header of the final shape, then C-order bytes): the bytes equal np.save(final array)."""

    def __init__(self, path: Path, shape, dtype=np.float16):
        self.path, self.shape, self.dtype, self.n = Path(path), tuple(int(x) for x in shape), np.dtype(dtype), 0
        self.tmp = self.path.with_name(self.path.name + ".tmp"); self.fh = open(self.tmp, "wb")
        np.lib.format.write_array_header_1_0(self.fh, {"descr": np.lib.format.dtype_to_descr(self.dtype), "fortran_order": False, "shape": self.shape})

    def write(self, arr):
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        if arr.shape[1:] != self.shape[1:] or self.n + arr.shape[0] > self.shape[0]:
            raise ValueError(f"chunk {arr.shape} does not fit {self.shape} at row {self.n}")
        self.fh.write(arr.tobytes(order="C")); self.n += arr.shape[0]

    def close(self):
        if self.n != self.shape[0]:
            raise ValueError(f"wrote {self.n} of {self.shape[0]} rows")
        self.fh.flush(); os.fsync(self.fh.fileno()); self.fh.close(); os.replace(self.tmp, self.path)


def npy_shape(path: Path):
    try:
        return tuple(np.load(path, mmap_mode="r").shape)
    except Exception:
        return None


def encode_chunk(h, mean, basis, pils):
    """The frozen encode_role arithmetic for one chunk."""
    f = h.encode(pils).float()
    return ((f - mean) @ basis).half().cpu().numpy()


def encode_stream(h, pca, ids, opener, path: Path, chunk=CHUNK, device="cuda"):
    mean, basis = pca["mean"].to(device), pca["basis"].to(device); w = NpyStream(path, (len(ids), 576, basis.shape[1])); buf = []
    for _, im in prefetch(ids, opener):
        buf.append(im)
        if len(buf) == chunk:
            w.write(encode_chunk(h, mean, basis, buf)); buf = []
    if buf:
        w.write(encode_chunk(h, mean, basis, buf))
    w.close()


def images_of(rows, role):
    return list(dict.fromkeys(r["image_id"] for r in rows if r["role"] == role))


def run(manifest_dir: Path, out: Path, basis="fitfull", frozen_pca=D1 / "cache" / "pca.pt", host=None, src=None, roles=("FIT", "CAL"), device="cuda"):
    meta = json.loads((manifest_dir / "d1_manifest_meta.json").read_text())
    if meta.get("frozen") is not True or sha256_file(manifest_dir / "d1_manifest.jsonl") != meta["manifest_sha256"]:
        raise SystemExit("D1-FULL manifest not frozen or changed")
    rows = [json.loads(x) for x in (manifest_dir / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
    if any(r["role"] not in ("FIT", "CAL", "DEV") for r in rows) or "DEV" in roles:
        raise SystemExit("roles: FIT and CAL only")
    out.mkdir(parents=True, exist_ok=True); mk = out / "cache_d1_full.json"
    if mk.exists():
        print(f"[r7_d1full_cache] complete: {mk}"); return json.loads(mk.read_text())
    if host is None:
        import r7_d1_teacher as TE
        host = TE.LlavaPrefill(out)
    if src is None:
        import r7_d1full_teacher as FT
        src = FT.StrictImageSource()
    torch = host.torch; rec = {"basis": basis, "seconds": {}, "manifest_sha256": meta["manifest_sha256"]}; t0 = time.time()
    fit_imgs = images_of(rows, "FIT")
    if not (out / "pca.pt").exists():
        if basis == "frozen":
            tmp = out / "pca.pt.tmp"; shutil.copyfile(frozen_pca, tmp)
            if sha256_file(tmp) != sha256_file(frozen_pca):
                raise SystemExit("copy of the frozen PCA basis differs")
            os.replace(tmp, out / "pca.pt")
        else:
            pca = CA.fit_pca(host, prefetch(fit_imgs, src.open), DIM)
            pca.update({"fit_images": len(fit_imgs), "tokens_per_image": 64, "manifest_sha256": meta["manifest_sha256"]})
            torch.save(pca, out / "pca.pt.tmp"); os.replace(out / "pca.pt.tmp", out / "pca.pt")
        rec["seconds"]["pca"] = time.time() - t0
    pca = torch.load(out / "pca.pt", map_location="cpu")
    same = (sha256_file(out / "pca.pt") == sha256_file(frozen_pca)) if basis == "frozen" else (pca.get("fit_images") == len(fit_imgs) and pca.get("manifest_sha256") == meta["manifest_sha256"])
    if not same:                                                                        # never mix bases across a resume
        raise SystemExit(f"{out / 'pca.pt'} does not match the requested basis '{basis}'; remove the cache dir to rebuild")
    rec.update({"pca_sha256": sha256_file(out / "pca.pt"), "explained_variance": pca.get("explained_variance"), "pca_fit_images": pca.get("fit_images"),
                "pca_tokens": pca.get("tokens")})
    if basis == "frozen":
        rec["frozen_pca_sha256"] = sha256_file(frozen_pca); rec["pca_equals_frozen"] = rec["pca_sha256"] == rec["frozen_pca_sha256"]
    for role in roles:
        ids = images_of(rows, role); f = out / f"feats_{role}.npy"; t1 = time.time()
        if not (npy_shape(f) == (len(ids), 576, pca["basis"].shape[1]) and (out / f"images_{role}.json").exists()
                and json.loads((out / f"images_{role}.json").read_text()) == ids):
            encode_stream(host, pca, ids, src.open, f, device=device); (out / f"images_{role}.json").write_text(json.dumps(ids))
            rec["seconds"][f"encode_{role}"] = time.time() - t1; rec["seconds"][f"per_image_{role}"] = (time.time() - t1) / max(len(ids), 1)
        rec[f"feats_{role}_shape"] = list(npy_shape(f))
    qrows = [r for r in rows if r["role"] in ("FIT", "CAL")]; qf = out / "question_ids.jsonl"
    n_lines = sum(1 for x in qf.read_text().splitlines() if x.strip()) if qf.exists() else -1
    if n_lines != len(qrows) or not (out / "embed_table.pt").exists():
        qt = out / "questions.tmp"; qt.mkdir(exist_ok=True); CA.save_questions(host, qrows, qt)        # frozen writer into a temp dir, then atomic moves
        os.replace(qt / "embed_table.pt", out / "embed_table.pt"); os.replace(qt / "question_ids.jsonl", qf); qt.rmdir()
    lens = [len(json.loads(x)["ids"]) for x in qf.read_text().splitlines() if x.strip()]
    rec.update({"questions": len(lens), "max_question_tokens": max(lens), "mean_question_tokens": float(np.mean(lens)), "seconds_total": time.time() - t0,
                "env": V.env_info(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    tmp = out / "cache_d1_full.json.tmp"; tmp.write_text(V.jdump(rec, indent=1)); os.replace(tmp, mk); print(V.jdump(rec, indent=1))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest-dir", default=str(DF / "data")); ap.add_argument("--out", default=str(DF / "cache"))
    ap.add_argument("--pca-basis", choices=["fitfull", "frozen"], default="fitfull"); ap.add_argument("--frozen-pca", default=str(D1 / "cache" / "pca.pt"))
    a = ap.parse_args(argv)
    run(Path(a.manifest_dir), Path(a.out), a.pca_basis, Path(a.frozen_pca))
    return 0


if __name__ == "__main__":
    sys.exit(main())
