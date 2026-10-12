"""r7 R7-30 D1-FULL teacher: LLaVA-1.5-7B with the frozen D1-10 code and settings (scripts/r7_d1_teacher.py: full pass b, 3 mechanisms × 4
nested budgets a, text-only pass; first-token yes / no probabilities) on the FIT_FULL shards that the frozen D1-10 run did not cover.
  bundle <id> : preflight (every image of the bundle present in the extracted GQA folder), then the frozen cmd_bundle with the frozen settings
                (16 images per chunk; prefill batches of <= 16,384 tokens / 64 sequences — the defaults every D1-10 bundle used) and a STRICT image
                source: a missing file raises MissingImage, a BaseException, so the frozen loop's reason-coded skip ('except Exception') cannot
                swallow it, and nothing ever falls back to a zip (the archives were deleted 2026-10-12).  A bundle owns its shards' files (single
                writer, execution note E1).  Launch gates of the frozen code: frozen D1-FULL manifest (sha256) and a passing probe report.  The
                completion marker bundle_<id>.ok.json is written only when every shard of the bundle is complete with no skipped image.
  replay      : GPU check behind the reuse of the frozen D1-10 FIT records: the first chunk (16 images) of frozen FIT shards recomputed with
                the frozen run_items and settings, compared with the stored records (same GPU type: identical qsim orderings and
                max |ΔP| <= 1e-3 required, bit identity reported; other GPU type: reported only)
    python scripts/r7_d1full_teacher.py bundle f000 [--manifest-dir DF/data --out DF/teacher --probe-report DF/smoke/probe/probe_report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_d1_teacher as TE  # noqa: E402   (frozen D1-10 teacher; read-only use)
import r7_vl_common as V  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
D1, DF = OUT / "R7_D1", OUT / "R7_D1_FULL"
IMAGES = TE.GQA_DIR / "images"
SETTINGS = {"images_per_chunk": 16, "max_tokens": 16384, "max_seqs": 64}          # = r7_d1_teacher.py bundle defaults (checked by the reuse check)
PROBE = DF / "smoke" / "probe" / "probe_report.json"
REPLAY_TOL = 1e-3


class MissingImage(BaseException):
    """A GQA image absent from the extracted folder.  Derives from BaseException so that the frozen bundle loop, which turns any Exception into
    a reason-coded skip, cannot swallow it: the unit stops loudly instead of finishing a shard with silently skipped images."""


class StrictImageSource:
    """GQA image by id from the extracted folder only (never images.zip)."""

    def __init__(self, folder=IMAGES):
        self.folder = Path(folder)

    def exists(self, image_id: str) -> bool:
        return (self.folder / f"{image_id}.jpg").is_file()

    def open(self, image_id: str):
        from PIL import Image
        f = self.folder / f"{image_id}.jpg"
        if not f.is_file():
            raise MissingImage(f"missing GQA image {f} (extracted folder only; no zip fallback)")
        return Image.open(f).convert("RGB")


def preflight(src, image_ids):
    miss = [i for i in image_ids if not src.exists(i)]
    if miss:
        raise SystemExit(f"preflight: {len(miss)} images missing from {getattr(src, 'folder', '?')}, e.g. {miss[:5]}")


def cmd_bundle(a, host_factory=None, image_source=None):
    root, out = Path(a.manifest_dir), Path(a.out); plan = json.loads((root / "d1_shards.json").read_text())
    if a.bundle not in plan["bundles"]:
        raise SystemExit(f"no bundle {a.bundle} in {root / 'd1_shards.json'}")
    shards = plan["bundles"][a.bundle]; reused = set(plan.get("d1full", {}).get("reused_shards", []))
    if any(not s.startswith("FIT-") for s in shards) or reused & set(shards):
        raise SystemExit(f"bundle {a.bundle}: D1-FULL bundles hold FIT shards that are not reused only ({shards})")
    if not a.probe_report:
        raise SystemExit("a probe report is required (launch gate)")
    src = image_source or StrictImageSource(); preflight(src, [i for s in shards for i in plan["shards"][s]]); t0 = time.time()
    TE.cmd_bundle(Namespace(bundle=a.bundle, manifest_dir=str(root), out=str(out), probe_report=a.probe_report, **SETTINGS),
                  host_factory=host_factory, image_source=src)
    st = []
    for s in shards:
        d = json.loads((out / f"{s}.done.json").read_text())
        st.append({"shard": s, "expected": plan["questions_per_shard"][s], "present": d["present"], "duplicates": d["duplicates"],
                   "skipped": len(d["skipped_images"]), "new": d["new_this_session"], "seconds": d["seconds"], "gpu": d["gpu"]})
    bad = [x for x in st if x["present"] != x["expected"] or x["duplicates"] or x["skipped"]]
    if bad:
        raise SystemExit(f"bundle {a.bundle} incomplete: {bad}")
    rec = {"bundle": a.bundle, "shards": st, "questions": sum(x["present"] for x in st), "new_questions": sum(x["new"] for x in st),
           "seconds_wall": time.time() - t0, "settings": SETTINGS, "manifest_sha256": json.loads((root / "d1_manifest_meta.json").read_text())["manifest_sha256"],
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = out / f"bundle_{a.bundle}.ok.json.tmp"; tmp.write_text(V.jdump(rec, indent=1)); tmp.replace(out / f"bundle_{a.bundle}.ok.json")
    print(f"[r7_d1full_teacher] bundle {a.bundle}: {rec['questions']} questions ({rec['new_questions']} new) in {rec['seconds_wall']:.0f}s", flush=True)
    return 0


def T(pair):
    y, n = pair
    return 2.0 * y / max(y + n, 1e-12) - 1.0


def compare(new, old):
    """new: run_items record; old: the stored JSON record.  Bit identity after the JSON round trip, qsim ordering identity, max |ΔP| / |ΔT|."""
    rt = json.loads(V.jdump(new)); keys = [("full",), ("text",)] + [("masked", m, str(k)) for m in ("random", "grid", "qsim") for k in (72, 144, 288, 432)]
    def get(r, k):
        return r[k[0]] if len(k) == 1 else r[k[0]][k[1]][k[2]]
    dp = max(max(abs(x - y) for x, y in zip(get(rt, k), get(old, k))) for k in keys); dt = max(abs(T(get(rt, k)) - T(get(old, k))) for k in keys)
    return {"bit_identical": rt == old, "qsim_identical": rt["qsim_order"] == old["qsim_order"], "max_abs_dP": dp, "max_abs_dT": dt}


def replay(host, shards, src, frozen_teacher=D1 / "teacher", frozen_data=D1 / "data", n_images=16):
    """Recompute the first chunk of each frozen FIT shard (the chunk the D1-10 bundle computed first: images_per_chunk = 16, one session) and
    compare with the stored records.  pass = >= 1 shard made on this GPU type and, on every such shard, identical qsim orderings and
    max |ΔP| <= REPLAY_TOL."""
    if any(not s.startswith("FIT-") for s in shards):
        raise SystemExit("replay reads frozen FIT shards only")
    plan = json.loads((frozen_data / "d1_shards.json").read_text()); man = {}
    for x in (frozen_data / "d1_manifest.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x); man.setdefault(r["image_id"], []).append(r)
    torch = host.torch; gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"; res = {"gpu": gpu, "shards": {}}
    for s in shards:
        done = json.loads((frozen_teacher / f"{s}.done.json").read_text()); stored = {r["qid"]: r for r in V.read_jsonl(frozen_teacher / f"{s}.jsonl")}
        if done.get("new_this_session") != done.get("present"):
            res["shards"][s] = {"skipped": "shard was resumed in D1-10 (first-chunk composition unknown)"}; continue
        imgs = plan["shards"][s][:n_images]; items = [q for i in imgs for q in man[i]]; pil = {i: src.open(i) for i in imgs}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time(); recs = TE.run_items(host, items, pil, SETTINGS["max_tokens"], SETTINGS["max_seqs"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec = time.time() - t0; cmp = [compare(r, stored[r["qid"]]) for r in recs]
        res["shards"][s] = {"gpu_then": done.get("gpu"), "same_gpu": done.get("gpu") == gpu, "items": len(recs), "images": len(imgs),
                            "bit_identical": sum(c["bit_identical"] for c in cmp), "qsim_identical": sum(c["qsim_identical"] for c in cmp),
                            "max_abs_dP": max(c["max_abs_dP"] for c in cmp), "max_abs_dT": max(c["max_abs_dT"] for c in cmp),
                            "seconds": sec, "s_per_question": sec / max(len(recs), 1)}
    same = {s: v for s, v in res["shards"].items() if v.get("same_gpu")}; cross = {s: v for s, v in res["shards"].items() if v.get("same_gpu") is False}
    res.update({"same_gpu_shards": sorted(same), "cross_gpu_shards": sorted(cross),
                "bit_identical_same_gpu": bool(same) and all(v["bit_identical"] == v["items"] for v in same.values()),
                "max_abs_dT_same_gpu": max((v["max_abs_dT"] for v in same.values()), default=None),
                "max_abs_dP_same_gpu": max((v["max_abs_dP"] for v in same.values()), default=None),
                "max_abs_dT_cross_gpu": max((v["max_abs_dT"] for v in cross.values()), default=None),
                "max_abs_dP_cross_gpu": max((v["max_abs_dP"] for v in cross.values()), default=None), "tolerance_dP": REPLAY_TOL})
    res["pass"] = bool(same) and all(v["qsim_identical"] == v["items"] and v["max_abs_dP"] <= REPLAY_TOL for v in same.values())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle"); b.add_argument("bundle"); b.add_argument("--manifest-dir", default=str(DF / "data")); b.add_argument("--out", default=str(DF / "teacher"))
    b.add_argument("--probe-report", default=str(PROBE))
    a = ap.parse_args(argv)
    return cmd_bundle(a)


if __name__ == "__main__":
    sys.exit(main())
