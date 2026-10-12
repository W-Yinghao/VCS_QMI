"""r5 final evaluation — the ONE fixed script of the one-shot official-test evaluation (owner approval 2026-10-12 10:16 UTC; proposal
reports/R5/final_evaluation_proposal.yaml incl. revision_20261012; protocol reports/R5/R5_FINAL_EVAL_PROTOCOL_DRAFT_20261012.yaml).

Every pre-declared row = one CAL-selected checkpoint (method x seed x dataset) on one official split, ranked over the frozen VL1-21 clean detector's
proposals exactly as in the reported DEV evaluation (top-100, clipped, < 1 px dropped; uniform models rank by cosine, R4-VL2 nu models by
2 a cos + log s, fused rows by 2 a cos + lambda_CAL log s with the R5-10 F1 lambda chosen on CAL).  Sub-commands (all write under --root):
  plan       CPU  rows, checkpoints (+ sha256), critic slopes, lambda_CAL, evaluation sets, detector / backbone hashes -> <root>/plan.json
  build      CPU  opens each annotation source ONCE: inputs WITHOUT labels (image id / path / size, query id, text) and a sealed label file
                  (ground-truth box per query, >= 2-referred-objects flag per image); logs sha256 of sources, inputs, labels and counts only
  proposals  GPU  the frozen VL1-21 detector on every evaluation image (sorted image ids, batches of 8: the DEV proposal settings)
  rank       GPU  one shard of checkpoints: top-1 proposal per query for every row -> single-writer prediction files (sealed directory)
  manifest   CPU  completeness of every prediction file + sha256 manifest; then the run STOPS until <root>/ALLOW_SCORING exists
  score      CPU  the ONLY step that opens the labels: query-level Acc@0.5 over all expressions (primary; uncovered = error), the development
                  image-macro metric and the >= 2-referred subset in separate tables, every contrast of the proposal, Holm on VQ - VCS
  make-dev-refer / integrity   development only (refused in test mode): a REFER-format file with RefCOCOg DEV content, and the comparison of a
                  DEV-mode score with the recorded run values
  rows-yaml  CPU  the plan's frozen row list (checkpoints with sha256, rows, lambda_CAL, missing rows) as a YAML fragment for the protocol
Test annotation files are opened only by `build` of a plan made with --mode test; every other path that names a test split, lies under a closed/
directory or holds several splits (refs(umd).p, scene_inputs.pkl, archives) is refused.  Run with `python -I`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pickle
import re
import socket
import subprocess
import sys
import tarfile
import time
import types
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for _p in (str(REPO / "src"), str(HERE)):          # python -I adds neither the script directory nor PYTHONPATH
    if _p not in sys.path:
        sys.path.insert(0, _p)
from vcs_vl import refcocog as RG  # noqa: E402  (module import reads no data)

OUTPUTS = Path("/home/infres/yinwang/CS_QMI/outputs"); ROOT_DEFAULT = OUTPUTS / "R5_FINAL"
MODELS = Path("/projects/EEG-foundation-model/yinghao/models"); DATA = Path("/projects/EEG-foundation-model/yinghao/datasets")
REFER_DIR = DATA / "refcocog_umd" / "refcocog"; MDETR_DIR = DATA / "mdetr_annotations"
DETECTOR = MODELS / "vl1_21_detector" / "final.pt"; PROP_DIR = OUTPUTS / "VL1_21" / "proposals"; META_DIR = OUTPUTS / "R4_VL0"
FUSION_JSON = REPO / "reports" / "R5" / "fusion_F0_F1.json"; R5_HASHES = REPO / "reports" / "R5" / "r5_hashes.json"
INTAKE_HASHES = REPO / "reports" / "R4" / "intake_hashes.json"
BACKBONE, SEEDS, IOU_THR, TOPK, DET_BS, CROP_CHUNK, BOOT_B, ALPHA = "siglip2_b16", (0, 1, 2), 0.5, 100, 8, 256, 1000, 0.05
MDETR_NAME = {"refcocog": "refcocog", "refcoco": "refcoco", "refcocoplus": "refcoco+"}
RECORDED_ARCHIVE_SHA256 = {"refcocog": (RG.ARCHIVE["sha256"], "reports/VL1/VL1_DATA_AUDIT.json revision.sha256; src/vcs_vl/refcocog.py ARCHIVE (download 2026-10-08)"),
                           "mdetr": ("ed7bf62535aef48b6b6f44f2ab32fea33c467bd1a94e8fb304408a23baaae11a",
                                     "reports/DATA_LICENCE_REGISTER_20261012.md V4 (archive hashed 2026-10-12, members listed from tar headers, nothing extracted)")}
# method -> (checkpoint dir, objective tag, run-output dir, reference law); checkpoints <dir>/<ds>_siglip2_b16_<tag>_s<seed>_calAcc.pt
METHODS = {"VCS": ("r4_vl1", "vcs", "R4_VL1", "uniform"), "JS": ("r4_vl1", "js", "R4_VL1", "uniform"), "E": ("r4_vl1", "softmax", "R4_VL1", "uniform"),
           "VNU": ("r4_vl2", "vcs_nuscore", "R4_VL2", "score"), "JNU": ("r4_vl2", "js_nuscore", "R4_VL2", "score"),
           "D": ("r5_dist", "distribution_softmax", "R5_DIST", "uniform"), "VQ": ("r5_qnorm", "vcs_qnorm", "R5_QNORM", "uniform"),
           "JQ": ("r5_qnorm", "js_qnorm", "R5_QNORM", "uniform")}
METHOD_LABEL = {"VCS": "VCS (R4-VL1 / R5-X1, uniform nu)", "JS": "matched JS (R4-VL1 / R5-X1)", "E": "set-event softmax E (R4-VL1 / R5-X1)",
                "VNU": "VCS(nu), native rank 2a cos + log s (R4-VL2)", "JNU": "JS(nu), native rank (R4-VL2)", "D": "distribution softmax D (R5-30 / R5-QX1)",
                "VQ": "VCS, query-normalized critic VQ (R5-QN / R5-QX1)", "JQ": "JS, query-normalized critic JQ (R5-QN / R5-QX1)"}
FUSION_KEY = {"VCS": "r4vl1:vcs", "JS": "r4vl1:js", "E": "r4vl1:softmax", "VNU": "r4vl2:vcs", "JNU": "r4vl2:js"}   # R5-10 F1 lambda_CAL (fusion_F0_F1.json)
for _m in list(FUSION_KEY):
    METHOD_LABEL[_m + "+F"] = METHOD_LABEL[_m].split(" (")[0] + " + lambda_CAL log s (R5-10 F1)"
CONTRASTS = (("primary", "VQ", "VCS"), ("estimation", "VQ", "D"), ("estimation", "VQ", "E"), ("estimation", "VQ", "JQ"),
             ("secondary", "VNU", "E"), ("secondary", "VCS+F", "E+F"), ("secondary", "VNU", "JNU"), ("secondary", "VCS", "E"), ("secondary", "D", "E"))
METRICS = ("query_all", "image_macro_ge2", "query_ge2")
METRIC_LABEL = {"query_all": "query-level Acc@0.5 over ALL official expressions (primary; uncovered / no proposal = error)",
                "image_macro_ge2": "development image-macro Acc@0.5 on images with >= 2 referred objects (the DEV definition)",
                "query_ge2": "query-level Acc@0.5 on the >= 2-referred-objects subset"}
TEST_SETS = (("refcocog", "test"), ("refcocoplus", "testA"), ("refcocoplus", "testB"), ("refcoco", "testA"), ("refcoco", "testB"))
# split sizes already recorded by the data audits (reports/VL1/VL1_DATA_AUDIT.json split_counts.TEST_CLOSED; reports/VL2/VL2_DATA_AUDIT_*.json
# splits): images / referred objects / expressions / images with >= 2 referred objects.  build stops BEFORE writing labels on any mismatch.
EXPECTED = {("refcocog", "test"): (2600, 5023, 9602, 1580), ("refcoco", "testA"): (750, 1975, 5657, 715), ("refcoco", "testB"): (750, 1810, 5095, 691),
            ("refcocoplus", "testA"): (750, 1975, 5726, 715), ("refcocoplus", "testB"): (750, 1798, 4889, 689),
            ("refcocoplus", "DEV"): (1699, None, None, 1587), ("refcoco", "DEV"): (1699, None, None, 1580), ("refcocog", "DEV"): (1356, None, 6442, 1356)}
EXIT_CHECK = 3                                     # deliberate stop on a failed check: the driver does NOT retry it


# ----------------------------------------------------------------------------------------------------------------- small utilities
def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(block), b""):
            h.update(blk)
    return h.hexdigest()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_atomic(path: Path, data: bytes, mode: int | None = None) -> str:
    """Single-writer atomic write (tmp name carries host / pid / job id), optional chmod; returns the sha256 of the bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}.{os.environ.get('SLURM_JOB_ID', 'nojob')}")
    if path.exists() and mode is not None:
        os.chmod(path, 0o600)                       # a retried unit replaces its own sealed file
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)
    if mode is not None:
        os.chmod(path, mode)
    return sha256_bytes(data)


def code_hashes() -> dict:
    files = [Path(__file__).resolve(), HERE / "vl3_finetune.py", HERE / "vl1_21_detector.py", REPO / "src" / "vcs_vl" / "refcocog.py",
             REPO / "src" / "vcs_vl" / "pairlaw.py"]                                            # everything the stages import from the repo
    return {str(p.relative_to(REPO)): sha256_file(p) for p in files if p.exists()}


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:                                                                              # noqa: BLE001 (record only)
        return None


def git_dirty(paths) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *paths], capture_output=True, text=True, timeout=30).stdout
        return [ln for ln in out.splitlines() if ln.strip()]
    except Exception:                                                                              # noqa: BLE001
        return ["git status failed"]


def stage_record(root: Path, stage: str, extra: dict) -> Path:
    rec = {"stage": stage, "utc_end": utc(), "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git_head": git_head(),
           "dirty_code_files": git_dirty(["scripts/r5_final_eval.py", "slurm/r5_final_eval_cpu.sbatch", "slurm/r5_final_eval_gpu.sbatch",
                                          "slurm/r5_final_eval_submit.sh"]),
           "code_sha256": code_hashes(), **extra}
    p = root / "logs" / f"{stage}.json"; write_atomic(p, json.dumps(rec, indent=1, sort_keys=True).encode()); return p


def jload(path):
    with open(path) as fh:
        return json.load(fh)


def log(msg: str) -> None:
    print(f"[r5_final_eval {utc()}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------- test-file guard
def protected_paths() -> set[Path]:
    return {REFER_DIR / "refs(umd).p", REFER_DIR / "refs(google).p", REFER_DIR / "instances.json", DATA / "refcocog_umd" / "refcocog.zip",
            DATA / "mdetr_annotations_dl" / "mdetr_annotations.tar.gz", DATA / "refcoco_unc" / "scene_inputs.pkl", DATA / "refcocoplus_unc" / "scene_inputs.pkl"}


def is_protected(path) -> bool:
    """True for official test files, anything under a closed/ directory, and the multi-split annotation files / caches / archives."""
    p = Path(os.path.abspath(path))
    return p in protected_paths() or "closed" in p.parts or re.search(r"test", p.name, re.IGNORECASE) is not None


def read_source(path, official: bool) -> tuple[bytes, str]:
    """The only reader of annotation sources: reads the bytes once and hashes them; protected files only in an official (test-mode) build."""
    if is_protected(path) and not official:
        raise PermissionError(f"refusing to open a protected / test file outside the official build: {path}")
    with open(path, "rb") as fh:
        data = fh.read()
    return data, sha256_bytes(data)


# ----------------------------------------------------------------------------------------------------------------- plan
def ckpt_path(method: str, ds: str, seed: int) -> Path:
    d, tag, _, _ = METHODS[method]; return MODELS / d / f"{ds}_{BACKBONE}_{tag}_s{seed}_calAcc.pt"


def run_json_path(method: str, ds: str, seed: int) -> Path:
    _, tag, o, _ = METHODS[method]; return OUTPUTS / o / f"{ds}_{BACKBONE}_{tag}_s{seed}.json"


def lambda_cal(method: str, ds: str, seed: int, fusion: dict) -> float | None:
    try:
        return float(fusion["F1"][ds]["models"][FUSION_KEY[method]][str(seed)]["lambda_cal"])
    except KeyError:
        return None


def eval_set_specs(mode: str, datasets, root: Path, max_images: int | None, dev_proposals: str = "regenerated") -> dict:
    out = {}
    if mode == "test":
        for ds, sp in TEST_SETS:
            if ds not in datasets:
                continue
            es = f"{ds}_{sp}"
            if ds == "refcocog":
                src = {"refs": str(REFER_DIR / "refs(umd).p"), "instances": str(REFER_DIR / "instances.json")}; fmt = "refer"
                arch = {"path": str(DATA / "refcocog_umd" / "refcocog.zip"), "members": {"refs": "refcocog/refs(umd).p", "instances": "refcocog/instances.json"},
                        "recorded_sha256": RECORDED_ARCHIVE_SHA256["refcocog"][0], "recorded_in": RECORDED_ARCHIVE_SHA256["refcocog"][1]}
            else:
                src = {"mdetr": str(MDETR_DIR / f"finetune_{MDETR_NAME[ds]}_{sp}.json")}; fmt = "mdetr"
                arch = {"path": str(DATA / "mdetr_annotations_dl" / "mdetr_annotations.tar.gz"), "members": {"mdetr": f"finetune_{MDETR_NAME[ds]}_{sp}.json"},
                        "recorded_sha256": RECORDED_ARCHIVE_SHA256["mdetr"][0], "recorded_in": RECORDED_ARCHIVE_SHA256["mdetr"][1]}
            n_img, n_ref, n_expr, n_ge2 = EXPECTED[(ds, sp)]
            out[es] = {"dataset": ds, "split": sp, "format": fmt, "sources": src, "archive_check": arch, "official": True,
                       "expected": {"images": n_img, "referred": n_ref, "expressions": n_expr, "images_ge2": n_ge2},
                       "dev_echo": str(META_DIR / "meta_refcocog.pt") if ds == "refcocog" else None}
    else:
        for ds in datasets:
            es = f"{ds}_DEV"
            if ds == "refcocog":                     # REFER-format file with DEV content only (make-dev-refer); split label "test" = same selection path
                src = {"refs": str(root / "dev_refer" / "refs(umd).p"), "instances": str(root / "dev_refer" / "instances.json")}; fmt, sp, filt = "refer", "test", None
            else:                                    # the official MDETR train file (same format as testA / testB), DEV images = VL1-21 DEV proposal keys
                src = {"mdetr": str(MDETR_DIR / f"finetune_{MDETR_NAME[ds]}_train.json")}; fmt, sp, filt = "mdetr", "train", str(PROP_DIR / f"{ds}_DEV.pt")
            n_img, n_ref, n_expr, n_ge2 = EXPECTED[(ds, "DEV")]
            out[es] = {"dataset": ds, "split": sp, "format": fmt, "sources": src, "archive_check": None, "official": False, "image_filter": filt,
                       "max_images": max_images, "expected": None if max_images else {"images": n_img, "referred": n_ref, "expressions": n_expr, "images_ge2": n_ge2},
                       "meta_echo": str(META_DIR / f"meta_{ds}.pt"), "compare_proposals": str(PROP_DIR / f"{ds}_DEV.pt"),
                       "detector_universe": str(PROP_DIR / f"{ds}_DEV.pt") if mode == "dev" else None,   # dev: the VL1-21 DEV batch composition;
                                                                                                                # smoke: the evaluation images (the official path)
                       "use_proposals": dev_proposals}                                 # stored: the VL1-21 DEV proposals the reported numbers used
    return out


def cmd_plan(a) -> int:
    root = Path(a.root); fusion = jload(FUSION_JSON); evs = eval_set_specs(a.mode, a.datasets, root, a.max_images, a.dev_proposals)
    plan = {"protocol": "reports/R5/R5_FINAL_EVAL_PROTOCOL_DRAFT_20261012.yaml", "proposal": "reports/R5/final_evaluation_proposal.yaml (revision_20261012)",
            "mode": a.mode, "created_utc": utc(), "root": str(root), "eval_sets": evs, "models": {}, "rows": [], "missing": [], "shards": {},
            "contrasts": [list(c) for c in CONTRASTS], "metrics": list(METRICS), "statistics": {
                "seed_paired_t": "per-seed differences (common seeds), mean, sd, 95 % t interval (df = n - 1), two-sided p", "bootstrap": {
                    "unit": "image (all expressions of a resampled image come with it)", "B": BOOT_B, "interval": "percentile 2.5 / 97.5",
                    "pairing": "seed-averaged per-image differences", "index_matrix": "one per evaluation set, shared by every row and contrast",
                    "seed": "numpy SeedSequence([20261012, 77, evaluation-set index])"},
                "holm": {"family": "primary contrast VQ - VCS on query_all over the evaluation sets", "alpha": ALPHA, "test": "seed-paired t, two-sided"}},
            "ranking": {"proposals": f"VL1-21 detector top-{TOPK} (score >= 0.05, per-class NMS) in detector order, clipped to the image, < 1 px dropped",
                        "cos": "argmax_r cos(r, w) (uniform-nu models; = argmax of the trained critic, a > 0; VQ / JQ normalizer is constant per query)",
                        "score_fusion": "argmax_r 2 a cos(r, w) + lambda log s_r (a = the checkpoint's CAL-selected critic slope; nu models native at lambda = 1)",
                        "ties": "first maximum in detector order", "no_proposal": "query predicted None = error"},
            "detector": {"path": str(DETECTOR), "batch": DET_BS, "order": "evaluation images sorted by image id", "partition": "RTX6000PRO (as the reported DEV / CAL proposals)"},
            "backbone": {"name": BACKBONE, "hf_repo": "google/siglip2-base-patch16-224", "note": "config / tokenizer / processor from the frozen HF snapshot; weights from each checkpoint"},
            "lambda_source": str(FUSION_JSON.relative_to(REPO))}
    if not a.no_hash:
        plan["detector"]["sha256"] = sha256_file(DETECTOR)
        rec = jload(INTAKE_HASHES)["files"].get(str(DETECTOR), {}).get("sha256") if INTAKE_HASHES.exists() else None
        plan["detector"]["recorded_sha256"] = rec; plan["detector"]["matches_record"] = rec == plan["detector"]["sha256"]
        snap = next((MODELS / "hf" / "models--google--siglip2-base-patch16-224" / "snapshots").iterdir())
        files = {p.name: sha256_file(p) for p in sorted(snap.iterdir()) if p.suffix in (".safetensors", ".bin", ".json", ".model")}
        recb = jload(R5_HASHES).get("siglip2_b16_weights", {}) if R5_HASHES.exists() else {}
        rf = recb.get("files", {})
        plan["backbone"].update({"snapshot": snap.name, "files_sha256": files, "vs_r5_hashes": {
            "snapshot_matches": recb.get("snapshot") == snap.name, "recorded_files": len(rf), "recorded_files_matching": sum(files.get(k) == v for k, v in rf.items()),
            "files_not_in_record": sorted(set(files) - set(rf))}})
    for method in a.methods:
        law = METHODS[method][3]
        for ds in a.datasets:
            for seed in a.seeds:
                cp, rj = ckpt_path(method, ds, seed), run_json_path(method, ds, seed)
                if not cp.exists() or not rj.exists():
                    plan["missing"].append({"method": method, "dataset": ds, "seed": seed, "checkpoint": str(cp), "exists": cp.exists(), "run_json_exists": rj.exists(),
                                            "reason": "never trained (no checkpoint)" if not cp.exists() else "run record missing"}); continue
                r = jload(rj); mid = f"{method}_{ds}_s{seed}"
                m = {"method": method, "dataset": ds, "seed": seed, "path": str(cp), "bytes": cp.stat().st_size, "run_json": str(rj), "law": law,
                     "critic_a": float(r["selected"]["critic_a_b"][0]), "critic_b": float(r["selected"]["critic_a_b"][1]),
                     "cal_selected_epoch": int(r["selected"]["cal_acc50_epoch"]), "objective": r.get("objective"), "loss_impl": r.get("loss_impl")}
                if not a.no_hash:
                    m["sha256"] = sha256_file(cp)
                shard = {"family": f"{ds}_{METHODS[method][0]}", "dataset": ds, "all": "all"}[a.shard_by]
                plan["models"][mid] = m; plan["shards"].setdefault(shard, []).append(mid)
                for es, spec in evs.items():
                    if spec["dataset"] != ds:
                        continue
                    base = {"kind": "cos"} if law == "uniform" else {"kind": "score_fusion", "lambda": 1.0, "native": True}
                    plan["rows"].append({"row": f"{method}|{es}|s{seed}", "method": method, "model": mid, "eval_set": es, "seed": seed, "rank": base})
                    if method not in FUSION_KEY and law == "uniform":
                        plan["missing"].append({"method": method + "+F", "dataset": ds, "seed": seed, "reason": "no lambda_CAL (not part of R5-10 F1)"})
                    if method in FUSION_KEY:
                        lam = lambda_cal(method, ds, seed, fusion)
                        if lam is None:
                            plan["missing"].append({"method": method + "+F", "dataset": ds, "seed": seed, "reason": "no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)"})
                        else:
                            plan["rows"].append({"row": f"{method}+F|{es}|s{seed}", "method": method + "+F", "model": mid, "eval_set": es, "seed": seed,
                                                 "rank": {"kind": "score_fusion", "lambda": lam, "native": law == "score" and lam == 1.0,
                                                          "label": "native" if (law == "score" and lam == 1.0) else ("posthoc_task_calibration" if law == "score" else "test_time_fusion")}})
    seen = set()
    for m in plan["missing"]:                                                                          # dedupe (one entry per eval set otherwise)
        seen.add(json.dumps(m, sort_keys=True))
    plan["missing"] = [json.loads(x) for x in sorted(seen)]
    out = Path(a.out) if a.out else root / "plan.json"
    sha = write_atomic(out, json.dumps(plan, indent=1, sort_keys=True, ensure_ascii=False).encode())
    if out.resolve() == (root / "plan.json").resolve():                                              # read by slurm/r5_final_eval_submit.sh
        write_atomic(root / "shards.txt", ("\n".join(sorted(plan["shards"])) + "\n").encode())
    log(f"plan ({a.mode}) -> {out} sha256 {sha}: {len(evs)} evaluation sets, {len(plan['models'])} checkpoints, {len(plan['rows'])} rows, "
        f"{len(plan['shards'])} shards, {len(plan['missing'])} missing entries")
    return 0


def load_plan(root: Path, plan_path=None) -> tuple[dict, str]:
    p = Path(plan_path) if plan_path else root / "plan.json"; b = p.read_bytes(); return json.loads(b), sha256_bytes(b)


# ----------------------------------------------------------------------------------------------------------------- build
def xyxy(b) -> list[float]:
    return [float(b[0]), float(b[1]), float(b[0]) + float(b[2]), float(b[1]) + float(b[3])]


def assemble(es: str, per_img: dict, sizes: dict, problems: dict) -> tuple[list, dict, dict]:
    """per_img: image id -> [(target ann id, file order, sentence id, text, bbox xywh)].  Queries ordered as the DEV records (referred objects by
    annotation id, expressions in file order).  Returns (inputs without labels, labels, counts)."""
    images, lab_q, lab_i = [], {}, {}; c = {"images": 0, "referred": 0, "expressions": 0, "images_ge2": 0, "images_missing_on_disk": 0,
                                          "images_inconsistent_size": 0, "degenerate_referred_boxes": 0, "duplicate_query_ids": 0, **problems}
    for iid in sorted(per_img):
        rows = sorted(per_img[iid], key=lambda t: (t[0], t[1])); sz = sizes[iid]
        if len(sz) != 1:
            c["images_inconsistent_size"] += 1
        W, H = sorted(sz)[0]; path = RG.coco_path(iid)
        if path is None:
            c["images_missing_on_disk"] += 1
        refd = {}
        for ann, _, _, _, box in rows:
            refd.setdefault(ann, box)
        nondeg = sum(RG.clip_box(tuple(b), W, H) is not None for b in refd.values()); c["degenerate_referred_boxes"] += len(refd) - nondeg
        ge2 = nondeg >= 2; qs = []
        for ann, _, sid, text, box in rows:
            qid = f"{es}:{sid}"
            if qid in lab_q:
                c["duplicate_query_ids"] += 1; continue
            lab_q[qid] = xyxy(box); qs.append({"qid": qid, "text": text})
        images.append({"image_id": int(iid), "path": path, "width": int(W), "height": int(H), "queries": qs})
        lab_i[str(iid)] = {"ge2": bool(ge2)}
        c["images"] += 1; c["referred"] += len(refd); c["expressions"] += len(qs); c["images_ge2"] += int(ge2)
    return images, {"eval_set": es, "iou_threshold": IOU_THR, "queries": lab_q, "images": lab_i}, c


def parse_refer(es: str, spec: dict, official: bool, cache: dict) -> tuple:
    """REFER release (RefCOCOg UMD): refs(umd).p (all splits in one pickle) + instances.json; selects refs of spec['split']; raw sentences (the
    text form every RefCOCOg model was trained and DEV-evaluated on)."""
    rb, rsha = read_source(spec["sources"]["refs"], official); ib, isha = read_source(spec["sources"]["instances"], official)
    refs = pickle.loads(rb); inst = json.loads(ib); imgs = {im["id"]: im for im in inst["images"]}; anns = {x["id"]: x for x in inst["annotations"]}
    split_of: dict = {}
    for r in refs:
        split_of.setdefault(r["image_id"], set()).add(r["split"])
    per_img, sizes, probs, order = {}, {}, {"ref_ann_missing": 0, "image_record_missing": 0, "mixed_split_images": 0}, 0
    for r in refs:
        if r["split"] != spec["split"]:
            continue
        x = anns.get(r["ann_id"]); im = imgs.get(r["image_id"])
        if x is None or im is None:
            probs["ref_ann_missing" if x is None else "image_record_missing"] += len(r["sentences"]); continue
        for s in r["sentences"]:
            per_img.setdefault(r["image_id"], []).append((int(r["ann_id"]), order, int(s["sent_id"]), s["raw"], x["bbox"])); order += 1
        sizes.setdefault(r["image_id"], set()).add((int(im["width"]), int(im["height"])))
    probs["mixed_split_images"] = sum(len(split_of[i]) != 1 for i in per_img)
    if spec.get("max_images") and not official:                                               # smoke subsets (development only)
        ids = sorted(per_img)[: spec["max_images"]]; per_img = {i: per_img[i] for i in ids}
    images, labels, counts = assemble(es, per_img, sizes, probs)
    srcs = {"refs": {"path": spec["sources"]["refs"], "sha256": rsha, "bytes": len(rb)}, "instances": {"path": spec["sources"]["instances"], "sha256": isha, "bytes": len(ib)}}
    echo = refer_dev_echo(refs, anns, imgs, split_of, spec["dev_echo"]) if spec.get("dev_echo") else None
    return images, labels, counts, srcs, echo


def refer_dev_echo(refs, anns, imgs, split_of, meta_path) -> dict:
    """Test-mode RefCOCOg only, at the single opening of refs(umd).p: rebuild the DEV records (RG.dev_roles on the train split; >= 2 referred
    objects; raw text; xyxy target box) and compare them with the R4-VL0 DEV meta every reported RefCOCOg DEV number was computed from."""
    import torch
    objs = [types.SimpleNamespace(image_id=i, umd_split=next(iter(sp))) for i, sp in sorted(split_of.items()) if len(sp) == 1]
    role = RG.dev_roles(objs); per: dict = {}
    for r in refs:
        if role.get(r["image_id"]) != "DEV" or r["ann_id"] not in anns:
            continue
        for s in r["sentences"]:
            per.setdefault(r["image_id"], []).append((int(r["ann_id"]), int(s["sent_id"]), s["raw"], xyxy(anns[r["ann_id"]]["bbox"])))
    meta = torch.load(meta_path, weights_only=False); n = ok = 0; bad_img = 0
    for m in meta:
        rows = {sid: (t, b) for _, sid, t, b in per.get(m["image_id"], [])}
        im = imgs.get(m["image_id"], {}); W, H = im.get("width", 0), im.get("height", 0)
        nondeg = len({ann for ann, _, _, b in per.get(m["image_id"], []) if RG.clip_box((b[0], b[1], b[2] - b[0], b[3] - b[1]), W, H) is not None})
        bad_img += nondeg < 2
        for sid, t, k in zip(m["sent_id"], m["texts"], m["target"]):
            n += 1; got = rows.get(int(sid))
            ok += got is not None and got[0] == t and bool(np.allclose(got[1], m["ref_boxes"][k], atol=1e-6))
    n_dev_ge2 = sum(1 for i, rows in per.items() if len({x[0] for x in rows}) >= 2)
    return {"meta": str(meta_path), "meta_images": len(meta), "dev_images_ge2_rebuilt": n_dev_ge2, "meta_expressions": n, "matching_expressions": ok,
            "meta_images_not_ge2_in_rebuild": bad_img, "pass": ok == n and n > 0 and n_dev_ge2 == len(meta) and bad_img == 0}


def parse_mdetr(es: str, spec: dict, official: bool, cache: dict) -> tuple:
    """MDETR pre-processed annotations (RefCOCO / RefCOCO+; one entry per expression: COCO image id, target COCO annotation id, caption = the
    text form the RefCOCO / RefCOCO+ models were trained and DEV-evaluated on, target bbox, image size)."""
    b, sha = read_source(spec["sources"]["mdetr"], official); d = json.loads(b)
    anns, probs = {}, {"entries_without_annotation": 0, "duplicate_annotation_entries": 0}
    for x in d["annotations"]:
        probs["duplicate_annotation_entries"] += x["image_id"] in anns; anns[x["image_id"]] = x
    keep = None
    if spec.get("image_filter"):
        import torch
        keep = {int(k) for k in torch.load(spec["image_filter"], map_location="cpu", weights_only=False).keys()}
    per_img, sizes = {}, {}
    for order, im in enumerate(d["images"]):
        iid = int(im["original_id"])
        if keep is not None and iid not in keep:
            continue
        x = anns.get(im["id"])
        if x is None:
            probs["entries_without_annotation"] += 1; continue
        per_img.setdefault(iid, []).append((int(x["original_id"]), order, int(im["id"]), im["caption"], x["bbox"]))
        sizes.setdefault(iid, set()).add((int(im["width"]), int(im["height"])))
    if spec.get("max_images"):
        ids = sorted(per_img)[: spec["max_images"]]; per_img = {i: per_img[i] for i in ids}
    images, labels, counts = assemble(es, per_img, sizes, probs)
    return images, labels, counts, {"mdetr": {"path": spec["sources"]["mdetr"], "sha256": sha, "bytes": len(b)}}, None


def meta_echo(images: list, labels: dict, meta_path: str) -> dict:
    """Development: the built records of the >= 2-referred images against the R4-VL0 DEV meta (texts and target boxes by sentence id)."""
    import torch
    meta = torch.load(meta_path, weights_only=False); es = labels["eval_set"]; by = {im["image_id"]: im for im in images}; n = ok = 0; flag_ok = 0
    for m in meta:
        im = by.get(m["image_id"])
        if im is None:
            n += len(m["texts"]); continue
        flag_ok += labels["images"][str(m["image_id"])]["ge2"]; txt = {q["qid"]: q["text"] for q in im["queries"]}
        for sid, t, k in zip(m["sent_id"], m["texts"], m["target"]):
            n += 1; qid = f"{es}:{int(sid)}"
            ok += qid in txt and txt[qid] == t and bool(np.allclose(labels["queries"][qid], m["ref_boxes"][k], atol=1e-6))
    n_ge2 = sum(v["ge2"] for v in labels["images"].values())
    return {"meta": meta_path, "meta_images": len(meta), "meta_images_present": sum(m["image_id"] in by for m in meta), "meta_images_flagged_ge2": flag_ok,
            "built_images_ge2": n_ge2, "meta_expressions": n, "matching_expressions": ok}


def archive_member_hashes(path: str, wanted: set) -> dict:
    """Read the archive once (hash it) and hash only the members whose base name is wanted (one decompression pass)."""
    b, sha = read_source(path, True); got: dict = {}
    if path.endswith(".zip"):
        with zipfile.ZipFile(io_bytes(b)) as z:
            for n in z.namelist():
                if os.path.basename(n) in wanted:
                    got.setdefault(os.path.basename(n), []).append(sha256_bytes(z.read(n)))
    else:
        with tarfile.open(fileobj=io_bytes(b), mode="r:*") as t:
            for mi in t:
                if mi.isfile() and os.path.basename(mi.name) in wanted:
                    got.setdefault(os.path.basename(mi.name), []).append(sha256_bytes(t.extractfile(mi).read()))
    return {"sha256": sha, "bytes": len(b), "members": got}


def archive_check(spec: dict, srcs: dict, cache: dict, wanted: dict) -> dict | None:
    """Official build only: the release archive's sha256 (against a recorded one where it exists) and its members used by this build compared
    with the opened files."""
    ac = spec.get("archive_check")
    if not ac:
        return None
    p = ac["path"]
    if p not in cache:
        cache[p] = archive_member_hashes(p, wanted[p]) if Path(p).exists() else None
    arc = cache[p]
    if arc is None:
        return {"archive": p, "present": False, "pass": None}
    res = {"archive": p, "present": True, "sha256": arc["sha256"], "archive_bytes": arc["bytes"], "recorded_sha256": ac.get("recorded_sha256"),
           "recorded_in": ac.get("recorded_in"), "members": {}}
    ok = ac.get("recorded_sha256") is None or ac["recorded_sha256"] == arc["sha256"]; res["archive_matches_record"] = None if ac.get("recorded_sha256") is None else ok
    for key, member in ac["members"].items():
        got = arc["members"].get(os.path.basename(member), []); same = srcs[key]["sha256"] in got
        res["members"][key] = {"member": member, "found": len(got), "identical_to_opened_file": same}; ok = ok and same
    res["pass"] = bool(ok); return res


def io_bytes(b: bytes):
    import io
    return io.BytesIO(b)


def cmd_build(a) -> int:
    root = Path(a.root); plan, plan_sha = load_plan(root, a.plan); official = plan["mode"] == "test"; t0 = time.time()
    if (root / "logs" / "BUILD_DONE").exists():
        log("build already done (logs/BUILD_DONE); nothing to do"); return 0
    built, cache, report, failed = {}, {}, {}, []
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with open(root / "logs" / "build_attempts.log", "a") as fh:                              # every opening of the sources is logged
        fh.write(f"{utc()} build attempt: job {os.environ.get('SLURM_JOB_ID')} host {socket.gethostname()} plan {plan_sha} official {official}\n")
    for es, spec in plan["eval_sets"].items():
        if bool(spec.get("official")) != official:
            raise SystemExit(f"{es}: evaluation-set mode does not match the plan mode")
        images, labels, counts, srcs, echo = (parse_refer if spec["format"] == "refer" else parse_mdetr)(es, spec, official, cache)
        checks = {}
        if spec.get("expected"):
            for k, v in spec["expected"].items():
                if v is not None:
                    checks[f"expected_{k}"] = {"expected": v, "got": counts[k], "pass": counts[k] == v}
        for k in ("images_missing_on_disk", "images_inconsistent_size", "duplicate_query_ids", "ref_ann_missing", "image_record_missing",
                  "entries_without_annotation", "duplicate_annotation_entries"):
            if k in counts:
                checks[k] = {"expected": 0, "got": counts[k], "pass": counts[k] == 0}
        if echo is not None:
            checks["dev_echo_refs_umd"] = {**echo}
        if spec.get("meta_echo") and not spec.get("max_images"):
            me = meta_echo(images, labels, spec["meta_echo"]); me["pass"] = (me["matching_expressions"] == me["meta_expressions"] > 0 and
                                                                             me["meta_images_flagged_ge2"] == me["meta_images"] == me["built_images_ge2"])
            checks["meta_echo"] = me
        if official:
            wanted: dict = {}
            for sp2 in plan["eval_sets"].values():
                if sp2.get("archive_check"):
                    wanted.setdefault(sp2["archive_check"]["path"], set()).update(os.path.basename(x) for x in sp2["archive_check"]["members"].values())
            checks["archive"] = archive_check(spec, srcs, cache, wanted) or {"pass": None}
        bad = [k for k, v in checks.items() if v.get("pass") is False]; failed += [f"{es}:{k}" for k in bad]
        built[es] = (images, labels); report[es] = {"sources": srcs, "counts": counts, "checks": checks}
        log(f"{es}: {counts['images']} images, {counts['referred']} referred objects, {counts['expressions']} expressions, {counts['images_ge2']} images "
            f"with >= 2 referred; sources " + ", ".join(f"{k} sha256 {v['sha256']}" for k, v in srcs.items()) + f"; checks failed: {bad or 'none'}")
    if failed:
        rec = stage_record(root, "build_FAILED", {"plan_sha256": plan_sha, "official": official, "failed_checks": failed, "eval_sets": report, "seconds": time.time() - t0})
        log(f"STOP: checks failed {failed}; NO input or label file written; see {rec}"); return EXIT_CHECK
    for es, (images, labels) in built.items():
        ish = write_atomic(root / "inputs" / f"{es}.json", canon({"eval_set": es, "images": images}))
        (root / "sealed" / "labels").mkdir(parents=True, exist_ok=True); os.chmod(root / "sealed" / "labels", 0o700)
        lsh = write_atomic(root / "sealed" / "labels" / f"{es}.json", canon(labels), mode=0o400)
        report[es]["inputs"] = {"path": str(root / "inputs" / f"{es}.json"), "sha256": ish}; report[es]["labels"] = {"path": str(root / "sealed" / "labels" / f"{es}.json"), "sha256": lsh}
        log(f"{es}: inputs sha256 {ish}; sealed labels sha256 {lsh}")
    rec = stage_record(root, "build", {"plan_sha256": plan_sha, "official": official, "eval_sets": report, "seconds": time.time() - t0})
    write_atomic(root / "logs" / "BUILD_DONE", f"{utc()} {sha256_file(rec)}\n".encode()); log(f"build done in {time.time() - t0:.0f} s -> {rec}")
    return 0


# ----------------------------------------------------------------------------------------------------------------- proposals
def load_inputs(root: Path, es: str, expect_sha: str | None = None) -> dict:
    b = (root / "inputs" / f"{es}.json").read_bytes()
    if expect_sha and sha256_bytes(b) != expect_sha:
        raise SystemExit(f"inputs {es}: sha256 differs from the build record")
    return json.loads(b)


def build_log(root: Path) -> dict:
    return jload(root / "logs" / "build.json")


def cmd_proposals(a) -> int:
    import torch
    from PIL import Image
    from vl1_21_detector import build_model, detect
    root = Path(a.root); plan, plan_sha = load_plan(root, a.plan); bl = build_log(root); t0 = time.time()
    if not (root / "logs" / "BUILD_DONE").exists():
        raise SystemExit("build not done")
    if (root / "proposals" / "PROPOSALS_DONE").exists():
        log("proposals already done"); return 0
    det_sha = sha256_file(DETECTOR)
    if plan["detector"].get("sha256") and det_sha != plan["detector"]["sha256"]:
        log(f"STOP: detector sha256 {det_sha} differs from the plan"); return EXIT_CHECK
    dev = torch.device("cuda"); model = build_model(); model.load_state_dict(torch.load(DETECTOR, map_location="cpu", weights_only=False)["model"]); model = model.to(dev).eval()
    gpu = torch.cuda.get_device_name(0); rep = {}
    import io
    for es, spec in plan["eval_sets"].items():
        inp = load_inputs(root, es, bl["eval_sets"][es]["inputs"]["sha256"]); ims = [im for im in inp["images"] if im["path"]]; t1 = time.time()
        size_mismatch = 0
        for im in ims:
            with Image.open(im["path"]) as pim:
                size_mismatch += (pim.width, pim.height) != (im["width"], im["height"])
        if spec.get("detector_universe") and not spec.get("official"):                    # development: the VL1-21 DEV batch composition
            uni = [int(k) for k in torch.load(spec["detector_universe"], map_location="cpu", weights_only=False).keys()]
            uni_ids, uni_paths = sorted(uni), [RG.coco_path(i) for i in sorted(uni)]
        else:                                                                                # official: the split's images, sorted by image id
            uni_ids, uni_paths = [int(im["image_id"]) for im in ims], [im["path"] for im in ims]
        dd = detect(model, uni_paths, dev, bs=DET_BS); allp = dict(zip(uni_ids, dd)); props = {int(im["image_id"]): allp[int(im["image_id"])] for im in ims}
        secs = time.time() - t1
        r = {"images": len(ims), "detector_batch_universe": len(uni_ids), "images_without_path": len(inp["images"]) - len(ims),
             "mean_detections": float(np.mean([len(x) for x in props.values()])) if props else 0.0, "images_zero_detections": int(sum(len(x) == 0 for x in props.values())),
             "image_size_mismatch": int(size_mismatch), "seconds": secs, "images_per_second": len(uni_ids) / max(1e-9, secs)}
        use = props
        if spec.get("compare_proposals") and not spec.get("official"):                     # development: against the stored VL1-21 DEV proposals
            old = torch.load(spec["compare_proposals"], map_location="cpu", weights_only=False); r["vs_stored_dev_proposals"] = compare_proposal_sets(props, old)
            if spec.get("use_proposals") == "stored":
                buf = io.BytesIO(); torch.save(props, buf); r["regenerated"] = {"path": str(root / "proposals" / f"{es}.regenerated.pt"),
                                                                              "sha256": write_atomic(root / "proposals" / f"{es}.regenerated.pt", buf.getvalue())}
                use = {iid: old[iid] for iid in props}; r["ranked_proposals"] = f"stored VL1-21 DEV proposals ({spec['compare_proposals']})"
        buf = io.BytesIO(); torch.save(use, buf); sha = write_atomic(root / "proposals" / f"{es}.pt", buf.getvalue()); r.update({"path": str(root / "proposals" / f"{es}.pt"), "sha256": sha})
        rep[es] = r; log(f"{es}: {len(ims)} images (detector batches over {len(uni_ids)}), {r['mean_detections']:.1f} detections / image, {r['images_per_second']:.1f} images/s "
                         f"on {gpu}; sha256 {sha}" + (f"; vs stored DEV proposals: {r['vs_stored_dev_proposals']}" if "vs_stored_dev_proposals" in r else ""))
    rec = stage_record(root, "proposals", {"plan_sha256": plan_sha, "gpu": gpu, "detector_sha256": det_sha, "eval_sets": rep, "seconds": time.time() - t0})
    write_atomic(root / "proposals" / "PROPOSALS_DONE", f"{utc()} {sha256_file(rec)}\n".encode()); log(f"proposals done in {time.time() - t0:.0f} s")
    return 0


def compare_proposal_sets(new: dict, old: dict) -> dict:
    """Development: regenerated vs stored detector outputs per image — bit identity, and an order-insensitive match (same class, IoU >= 0.99)."""
    def iou1(a, B):
        x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1]); x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None); return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)
    st = {"bit_identical": 0, "same_count": 0, "top1_same": 0, "matched": [], "score_diff": []}
    for iid, x in new.items():
        y = old.get(iid)
        if y is None:
            continue
        st["bit_identical"] += int(tuple(x.shape) == tuple(y.shape) and bool((x == y).all())); x, y = x.numpy(), y.numpy(); st["same_count"] += int(len(x) == len(y))
        st["top1_same"] += int(len(x) > 0 and len(y) > 0 and x[0, 5] == y[0, 5] and iou1(x[0], y[:1])[0] >= 0.99)
        m = [float(iou1(r, y[y[:, 5] == r[5]]).max()) if (y[:, 5] == r[5]).any() else 0.0 for r in x]; st["matched"].append(float(np.mean(np.array(m) >= 0.99)) if m else 1.0)
        k = min(len(x), len(y)); st["score_diff"].append(float(np.abs(np.sort(x[:, 4])[::-1][:k] - np.sort(y[:, 4])[::-1][:k]).max()) if k else 0.0)
    n = len(st["matched"])
    return {"images": n, "bit_identical": st["bit_identical"], "same_count": st["same_count"], "top1_same": st["top1_same"],
            "mean_share_boxes_matched_iou99": float(np.mean(st["matched"])) if n else None, "median_max_sorted_score_diff": float(np.median(st["score_diff"])) if n else None,
            "max_sorted_score_diff": float(np.max(st["score_diff"])) if n else None}


# ----------------------------------------------------------------------------------------------------------------- rank
def clean_proposals(P, W: int, H: int) -> tuple[list, list]:
    """The DEV candidate rule (scripts/r4_vl1_finetune.py build): top-100 in detector order, clipped to the image, < 1 px dropped."""
    boxes, det = [], []
    for x1, y1, x2, y2, sc, _ in P[:TOPK].tolist():
        b = RG.clip_box((x1, y1, x2 - x1, y2 - y1), W, H)
        if b is not None:
            boxes.append(b); det.append(sc)
    return boxes, det


def rank_rows(cos: np.ndarray, det: np.ndarray, a: float, rules: dict) -> dict:
    """cos [n_p, n_q] (float32 cosines as float64); returns rule name -> argmax proposal index per query (first maximum)."""
    out = {}
    for name, r in rules.items():
        if r["kind"] == "cos":
            S = cos
        elif r["kind"] == "score_fusion":
            if not (det > 0).all():
                raise ValueError("non-positive detector score")
            S = 2.0 * a * cos + float(r["lambda"]) * np.log(det)[:, None]
        else:
            raise ValueError(r["kind"])
        out[name] = S.argmax(0)
    return out


class _Crops:
    def __init__(self, recs, bb):
        self.recs, self.bb = recs, bb

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        r = self.recs[i]
        if not r["boxes"]:
            return i, None, self.bb.tokenize(r["texts"])
        with Image.open(r["path"]) as im:
            im = im.convert("RGB"); crops = torch.stack([self.bb.preprocess(im.crop(tuple(b))) for b in r["boxes"]])
        return i, crops, self.bb.tokenize(r["texts"])


def cmd_rank(a) -> int:
    import torch
    import torch.nn.functional as F
    from vl3_finetune import Backbone
    root = Path(a.root); plan, plan_sha = load_plan(root, a.plan); bl = build_log(root); t0 = time.time()
    if not (root / "proposals" / "PROPOSALS_DONE").exists():
        raise SystemExit("proposals not done")
    pl = jload(root / "logs" / "proposals.json"); mids = plan["shards"][a.shard]; pred_dir = root / "sealed" / "predictions"
    marker = pred_dir / "shards" / f"{a.shard}.done"
    if marker.exists():
        log(f"shard {a.shard} already done"); return 0
    dev = torch.device("cuda"); gpu = torch.cuda.get_device_name(0); bb = Backbone(BACKBONE).to(dev); bb.eval()
    cache = {}

    def records(es):
        if es not in cache:
            inp = load_inputs(root, es, bl["eval_sets"][es]["inputs"]["sha256"]); pp = root / "proposals" / f"{es}.pt"
            if sha256_file(pp) != pl["eval_sets"][es]["sha256"]:
                raise SystemExit(f"proposals {es}: sha256 differs from the proposals record")
            props = torch.load(pp, map_location="cpu", weights_only=False); recs = []
            for im in inp["images"]:
                P = props.get(int(im["image_id"])); boxes, det = clean_proposals(P.numpy(), im["width"], im["height"]) if (P is not None and im["path"]) else ([], [])
                recs.append({"image_id": im["image_id"], "path": im["path"], "boxes": boxes, "det": np.asarray(det, np.float64),
                             "qids": [q["qid"] for q in im["queries"]], "texts": [q["text"] for q in im["queries"]]})
            cache[es] = recs
        return cache[es]

    done, timing = {}, []
    for mid in mids:
        m = plan["models"][mid]; t1 = time.time()
        sha = sha256_file(m["path"])
        if m.get("sha256") and sha != m["sha256"]:
            log(f"STOP: {mid} checkpoint sha256 {sha} differs from the plan {m['sha256']}"); return EXIT_CHECK
        ck = torch.load(m["path"], map_location="cpu", weights_only=False); bb.load_state_dict({k: v.float() for k, v in ck["backbone"].items()}); bb.eval()
        a_ck = float(ck["critic_a_b"][0])
        if abs(a_ck - m["critic_a"]) > 1e-6 or int(ck["epoch"]) != m["cal_selected_epoch"]:
            log(f"STOP: {mid} checkpoint critic a / epoch ({a_ck}, {ck['epoch']}) differ from the run record ({m['critic_a']}, {m['cal_selected_epoch']})"); return EXIT_CHECK
        for es in [e for e, s in plan["eval_sets"].items() if s["dataset"] == m["dataset"]]:
            rows = {r["row"]: r for r in plan["rows"] if r["model"] == mid and r["eval_set"] == es}; recs = records(es); t2 = time.time()
            preds = {rn: {} for rn in rows}; n_crops = 0
            dl = torch.utils.data.DataLoader(_Crops(recs, bb), batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=lambda b: b[0])
            with torch.no_grad():
                for i, crops, toks in dl:
                    r = recs[i]
                    if crops is None:
                        for rn in rows:
                            for q in r["qids"]:
                                preds[rn][q] = None
                        continue
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        U = torch.cat([bb.enc_img(crops[k:k + CROP_CHUNK].to(dev, non_blocking=True)) for k in range(0, len(crops), CROP_CHUNK)])
                        V = bb.enc_txt(toks.to(dev, non_blocking=True))
                    C = (F.normalize(U.float(), dim=-1) @ F.normalize(V.float(), dim=-1).T).cpu().numpy().astype(np.float64); n_crops += len(crops)
                    idx = rank_rows(C, r["det"], m["critic_a"], {rn: rr["rank"] for rn, rr in rows.items()})
                    for rn in rows:
                        for j, q in enumerate(r["qids"]):
                            k = int(idx[rn][j]); preds[rn][q] = [k] + [float(v) for v in r["boxes"][k]]
            doc = {"model": mid, "model_sha256": sha, "eval_set": es, "critic_a": m["critic_a"], "rows": {rn: {"rank": rows[rn]["rank"], "predictions": preds[rn]} for rn in rows},
                   "n_queries": sum(len(r["qids"]) for r in recs), "n_queries_without_proposal": sum(len(r["qids"]) for r in recs if not r["boxes"])}
            fsha = write_atomic(pred_dir / f"pred__{mid}__{es}.json", canon(doc), mode=0o444); done[f"pred__{mid}__{es}.json"] = fsha
            timing.append({"model": mid, "eval_set": es, "images": len(recs), "crops": n_crops, "seconds": time.time() - t2})
            log(f"{mid} on {es}: {len(rows)} rows, {doc['n_queries']} queries, {n_crops} crops in {time.time() - t2:.0f} s ({gpu})")
        log(f"{mid}: done in {time.time() - t1:.0f} s")
    rec = stage_record(root, f"rank_{a.shard}", {"plan_sha256": plan_sha, "gpu": gpu, "shard": a.shard, "files_sha256": done, "timing": timing, "seconds": time.time() - t0})
    write_atomic(marker, canon({"files": done, "record": str(rec)})); log(f"shard {a.shard} done in {time.time() - t0:.0f} s")
    return 0


# ----------------------------------------------------------------------------------------------------------------- manifest
def expected_pred_files(plan: dict) -> dict:
    out = {}
    for r in plan["rows"]:
        out.setdefault(f"pred__{r['model']}__{r['eval_set']}.json", set()).add(r["row"])
    return out


def stage_code_hashes(root: Path) -> dict:
    out = {}
    for p in sorted((root / "logs").glob("*.json")):
        if p.name.endswith("_FAILED.json"):
            continue
        out[p.stem] = jload(p).get("code_sha256", {}).get("scripts/r5_final_eval.py")
    return out


def cmd_manifest(a) -> int:
    root = Path(a.root); plan, plan_sha = load_plan(root, a.plan); bl = build_log(root); pred_dir = root / "sealed" / "predictions"
    missing_shards = [s for s in plan["shards"] if not (pred_dir / "shards" / f"{s}.done").exists()]
    if missing_shards:
        log(f"not complete: shards without marker {missing_shards}"); return 1
    qids = {es: [q["qid"] for im in load_inputs(root, es, bl["eval_sets"][es]["inputs"]["sha256"])["images"] for q in im["queries"]] for es in plan["eval_sets"]}
    lines, problems = [], []
    for fn, rows in sorted(expected_pred_files(plan).items()):
        p = pred_dir / fn
        if not p.exists():
            problems.append(f"missing {fn}"); continue
        b = p.read_bytes(); d = json.loads(b); es = d["eval_set"]
        if set(d["rows"]) != rows:
            problems.append(f"{fn}: rows {sorted(set(d['rows']) ^ rows)} differ from the plan")
        for rn, rr in d["rows"].items():
            if set(rr["predictions"]) != set(qids[es]):
                problems.append(f"{fn}: row {rn} covers {len(rr['predictions'])} of {len(qids[es])} queries")
        lines.append(f"{sha256_bytes(b)}  {p.relative_to(root)}")
    codes = stage_code_hashes(root); consistent = len(set(codes.values())) == 1
    if not consistent and not a.allow_code_change:
        problems.append(f"scripts/r5_final_eval.py differs between stages: {codes}")
    if problems:
        rec = stage_record(root, "manifest_FAILED", {"plan_sha256": plan_sha, "problems": problems}); log(f"STOP: {len(problems)} problems, e.g. {problems[:5]}; {rec}")
        return EXIT_CHECK
    txt = ("\n".join(lines) + "\n").encode(); msha = write_atomic(pred_dir / "MANIFEST.sha256", txt, mode=0o444)
    side = {"plan_sha256": plan_sha, "manifest_sha256": msha, "n_prediction_files": len(lines), "n_rows": len(plan["rows"]),
            "build_record_sha256": sha256_file(root / "logs" / "build.json"), "proposals_record_sha256": sha256_file(root / "logs" / "proposals.json"),
            "inputs_sha256": {es: v["inputs"]["sha256"] for es, v in bl["eval_sets"].items()}, "labels_sha256": {es: v["labels"]["sha256"] for es, v in bl["eval_sets"].items()},
            "proposals_sha256": {es: v["sha256"] for es, v in jload(root / "logs" / "proposals.json")["eval_sets"].items()},
            "script_sha256_by_stage": codes, "code_consistent_across_stages": consistent}
    write_atomic(pred_dir / "manifest.json", json.dumps(side, indent=1, sort_keys=True).encode(), mode=0o444)
    stage_record(root, "manifest", side); write_atomic(pred_dir / "MANIFEST_DONE", f"{utc()} {msha}\n".encode())
    log(f"manifest: {len(lines)} prediction files, MANIFEST.sha256 sha256 {msha}.  Commit {pred_dir / 'MANIFEST.sha256'} and {pred_dir / 'manifest.json'}, "
        f"then create {root / 'ALLOW_SCORING'} (line 'manifest_sha256={msha}') to allow scoring.")
    return 0


# ----------------------------------------------------------------------------------------------------------------- score
def iou_pairs(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Row-wise IoU of xyxy boxes (continuous coordinates, as the DEV evaluation: scripts/vl1_22_tableC.py iou_xyxy)."""
    ix = np.clip(np.minimum(P[:, 2], G[:, 2]) - np.maximum(P[:, 0], G[:, 0]), 0, None); iy = np.clip(np.minimum(P[:, 3], G[:, 3]) - np.maximum(P[:, 1], G[:, 1]), 0, None)
    inter = ix * iy; union = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1]) + (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def t_stats(diffs) -> dict:
    from scipy import stats
    x = np.asarray(diffs, float); n = len(x); m = float(x.mean()) if n else float("nan")
    if n < 2:
        return {"mean": m, "sd": None, "ci95": None, "p_two_sided": None, "n": n}
    sd = float(x.std(ddof=1)); h = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    p = (0.0 if m != 0 else 1.0) if sd == 0 else float(2 * stats.t.sf(abs(m / (sd / math.sqrt(n))), n - 1))
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "p_two_sided": p, "n": n}


def holm(pvals: dict, alpha: float = ALPHA) -> dict:
    """Holm step-down; returns key -> (adjusted p, reject)."""
    items = sorted((p, k) for k, p in pvals.items() if p is not None); m = len(items); out, run = {}, 0.0
    for j, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p)); out[k] = (run, run < alpha)
    return out


def boot_matrix(n_img: int, B: int, es_index: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([20261012, 77, es_index])).integers(0, n_img, (B, n_img))


def boot_ratio_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> list | None:
    nn = num[idx].sum(1); dd = den[idx].sum(1); ok = dd > 0
    if not ok.any():
        return None
    v = nn[ok] / dd[ok]; return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def reading(ci) -> str:
    if ci is None:
        return "no interval"
    return "contains 0" if ci[0] <= 0 <= ci[1] else ("above 0" if ci[0] > 0 else "below 0")


def es_arrays(inp: dict, labels: dict) -> dict:
    qids, img, n_q, ge2 = [], [], [], []
    for k, im in enumerate(inp["images"]):
        for q in im["queries"]:
            qids.append(q["qid"]); img.append(k)
        n_q.append(len(im["queries"])); ge2.append(bool(labels["images"][str(im["image_id"])]["ge2"]))
    G = np.array([labels["queries"][q] for q in qids], float).reshape(-1, 4)
    return {"qids": qids, "img": np.array(img, np.int64), "n_q": np.array(n_q, float), "ge2": np.array(ge2, bool), "G": G}


def hits_of(preds: dict, A: dict) -> np.ndarray:
    P = np.array([[np.nan] * 4 if preds[q] is None else preds[q][1:5] for q in A["qids"]], float).reshape(-1, 4); has = ~np.isnan(P[:, 0])
    h = np.zeros(len(A["qids"]), bool); h[has] = iou_pairs(P[has], A["G"][has]) >= IOU_THR; return h


def metrics_of(h: np.ndarray, A: dict) -> dict:
    per_img = np.bincount(A["img"], weights=h.astype(float), minlength=len(A["n_q"])); g = A["ge2"]; gq = g[A["img"]]
    return {"query_all": float(h.mean()) if len(h) else float("nan"),
            "image_macro_ge2": float((per_img[g] / A["n_q"][g]).mean()) if g.any() else float("nan"),
            "query_ge2": float(h[gq].mean()) if gq.any() else float("nan")}


def cmd_score(a) -> int:
    import torch
    root = Path(a.root); plan, plan_sha = load_plan(root, a.plan); pred_dir = root / "sealed" / "predictions"; gate = root / "ALLOW_SCORING"; t0 = time.time()
    out = Path(a.out) if a.out else root / "score"
    if (out / "SCORE_DONE").exists():
        log(f"STOP: {out} already holds a score; a deterministic re-verification of the frozen predictions writes to another --out"); return EXIT_CHECK
    if not gate.exists():
        log(f"STOP: {gate} does not exist (created by the coordinator only after committing the prediction hashes)"); return EXIT_CHECK
    gtxt = gate.read_text(); want = re.search(r"manifest_sha256=([0-9a-f]{64})", gtxt); mb = (pred_dir / "MANIFEST.sha256").read_bytes(); msha = sha256_bytes(mb)
    if want and want.group(1) != msha:
        log(f"STOP: MANIFEST.sha256 sha256 {msha} differs from the gate's {want.group(1)}"); return EXIT_CHECK
    bad = []
    for ln in mb.decode().splitlines():
        h, rel = ln.split("  ", 1)
        if sha256_file(root / rel) != h:
            bad.append(rel)
    if bad:
        log(f"STOP: prediction files changed after the manifest: {bad[:5]}"); return EXIT_CHECK
    bl = build_log(root); pl = jload(root / "logs" / "proposals.json"); res = {"plan_sha256": plan_sha, "manifest_sha256": msha, "gate": gtxt.strip(),
                                                                                   "mode": plan["mode"], "eval_sets": {}, "rows": {}, "methods": {}, "contrasts": [],
                                                                                   "missing_in_plan": plan["missing"], "statistics": plan["statistics"]}
    rows_by_es: dict = {}
    for r in plan["rows"]:
        rows_by_es.setdefault(r["eval_set"], []).append(r)
    for ei, es in enumerate(plan["eval_sets"]):
        inp = load_inputs(root, es, bl["eval_sets"][es]["inputs"]["sha256"]); lp = root / "sealed" / "labels" / f"{es}.json"; lb = lp.read_bytes()
        if sha256_bytes(lb) != bl["eval_sets"][es]["labels"]["sha256"]:
            log(f"STOP: labels {es} differ from the build record"); return EXIT_CHECK
        labels = json.loads(lb); A = es_arrays(inp, labels)
        props = torch.load(root / "proposals" / f"{es}.pt", map_location="cpu", weights_only=False)
        if sha256_file(root / "proposals" / f"{es}.pt") != pl["eval_sets"][es]["sha256"]:
            log(f"STOP: proposals {es} differ from the record"); return EXIT_CHECK
        cover, no_prop, k = np.zeros(len(A["qids"]), bool), 0, 0
        for im in inp["images"]:
            P = props.get(int(im["image_id"])); boxes, _ = clean_proposals(P.numpy(), im["width"], im["height"]) if (P is not None and im["path"]) else ([], [])
            B = np.array(boxes, float).reshape(-1, 4)
            for _ in im["queries"]:
                if len(B):
                    cover[k] = bool((iou_pairs(B, np.repeat(A["G"][k][None], len(B), 0)) >= IOU_THR).any())
                else:
                    no_prop += 1
                k += 1
        gq = A["ge2"][A["img"]]
        res["eval_sets"][es] = {"dataset": plan["eval_sets"][es]["dataset"], "split": plan["eval_sets"][es]["split"], "images": int(len(A["n_q"])), "queries": len(A["qids"]),
                                "images_ge2": int(A["ge2"].sum()), "queries_ge2": int(gq.sum()), "queries_without_any_proposal": int(no_prop),
                                "ceiling_any_valid_proposal_query_all": float(cover.mean()), "ceiling_any_valid_proposal_query_ge2": float(cover[gq].mean()) if gq.any() else None}
        H: dict = {}; pcache: dict = {}
        for r in rows_by_es.get(es, []):
            fn = f"pred__{r['model']}__{es}.json"
            if fn not in pcache:
                pcache.clear(); pcache[fn] = jload(pred_dir / fn)
            d = pcache[fn]; h = hits_of(d["rows"][r["row"]]["predictions"], A)
            res["rows"][r["row"]] = {"method": r["method"], "seed": r["seed"], "eval_set": es, "rank": r["rank"], **metrics_of(h, A),
                                     "queries_without_prediction": int(sum(v is None for v in d["rows"][r["row"]]["predictions"].values()))}; H[(r["method"], r["seed"])] = h
        meth = {}
        for (mt, s), h in sorted(H.items()):
            meth.setdefault(mt, {})[s] = h
        res["methods"][es] = {}
        for mt, by in meth.items():
            vals = {s: metrics_of(h, A) for s, h in by.items()}
            res["methods"][es][mt] = {"label": METHOD_LABEL.get(mt, mt), "seeds": sorted(by), **{
                mk: {"per_seed": {str(s): vals[s][mk] for s in sorted(by)}, **{k2: v for k2, v in t_stats([vals[s][mk] for s in sorted(by)]).items() if k2 in ("mean", "sd", "n")}}
                for mk in METRICS}}
        idx = boot_matrix(len(A["n_q"]), plan["statistics"]["bootstrap"]["B"], ei); g = A["ge2"].astype(float)
        for fam, x, y in [tuple(c) for c in plan["contrasts"]]:
            if x not in meth or y not in meth:
                continue
            ss = sorted(set(meth[x]) & set(meth[y]))
            if not ss:
                continue
            dq = np.mean([np.bincount(A["img"], weights=meth[x][s].astype(float) - meth[y][s].astype(float), minlength=len(A["n_q"])) for s in ss], 0)
            for mk in METRICS:
                diffs = [metrics_of(meth[x][s], A)[mk] - metrics_of(meth[y][s], A)[mk] for s in ss]; t = t_stats(diffs)
                num, den = {"query_all": (dq, A["n_q"]), "image_macro_ge2": (dq / np.maximum(A["n_q"], 1) * g, g), "query_ge2": (dq * g, A["n_q"] * g)}[mk]
                ci_b = boot_ratio_ci(num, den, idx)
                res["contrasts"].append({"family": fam, "contrast": f"{x} - {y}", "x": x, "y": y, "eval_set": es, "metric": mk, "seeds": ss, "per_seed_diff": diffs,
                                         "mean": t["mean"], "sd": t["sd"], "t_ci95": t["ci95"], "p_two_sided": t["p_two_sided"], "boot_ci95": ci_b,
                                         "reading_t": reading(t["ci95"]), "reading_boot": reading(ci_b)})
    prim = {c["eval_set"]: c["p_two_sided"] for c in res["contrasts"] if c["family"] == "primary" and c["metric"] == "query_all"}
    adj = holm(prim, plan["statistics"]["holm"]["alpha"])
    for c in res["contrasts"]:
        if c["family"] == "primary" and c["metric"] == "query_all":
            pa = adj.get(c["eval_set"]); c["holm_p_adj"], c["holm_reject"] = (pa if pa else (None, None))
            c["decision"] = (None if pa is None else ("VQ improves VCS" if (pa[1] and c["mean"] > 0) else ("VQ lowers VCS" if pa[1] else "no detectable difference (Holm)")))
    res["holm"] = {"family": plan["statistics"]["holm"]["family"], "alpha": plan["statistics"]["holm"]["alpha"], "m": len(prim), "seconds": time.time() - t0}
    write_atomic(out / "results.json", json.dumps(res, indent=1, sort_keys=True).encode()); (out / "tables.md").write_text(tables_md(res, plan))
    rec = stage_record(root, "score" if out == root / "score" else f"score_{out.name}", {"plan_sha256": plan_sha, "manifest_sha256": msha, "out": str(out),
                                                                                       "results_sha256": sha256_file(out / "results.json"), "seconds": time.time() - t0})
    write_atomic(out / "SCORE_DONE", f"{utc()} {sha256_file(rec)}\n".encode()); log(f"score done in {time.time() - t0:.0f} s -> {out}")
    return 0


def tables_md(res: dict, plan: dict) -> str:
    pc = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.2f}"
    pci = lambda ci: "—" if not ci else f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"
    L = [f"# r5 final evaluation — results ({res['mode']} mode)", "", f"plan sha256 `{res['plan_sha256']}`; prediction manifest sha256 `{res['manifest_sha256']}`.", "",
         "Evaluation sets (counts; ceiling = queries with >= 1 proposal at IoU >= 0.5):", "", "| set | images | queries | images >= 2 referred | queries >= 2 | no proposal | ceiling all / >= 2 |", "|---|---|---|---|---|---|---|"]
    for es, e in res["eval_sets"].items():
        L.append(f"| {es} | {e['images']} | {e['queries']} | {e['images_ge2']} | {e['queries_ge2']} | {e['queries_without_any_proposal']} | {pc(e['ceiling_any_valid_proposal_query_all'])} / {pc(e['ceiling_any_valid_proposal_query_ge2'])} |")
    L += ["", f"## Primary confirmation — VQ − VCS, {METRIC_LABEL['query_all']}; Holm over the evaluation sets", "",
          "| set | VQ − VCS | seed-paired 95 % t | image bootstrap 95 % | p (two-sided) | Holm p | decision |", "|---|---|---|---|---|---|---|"]
    for c in res["contrasts"]:
        if c["family"] == "primary" and c["metric"] == "query_all":
            pv = "" if c["p_two_sided"] is None else format(c["p_two_sided"], ".4f"); hp = "" if c.get("holm_p_adj") is None else format(c["holm_p_adj"], ".4f")
            L.append(f"| {c['eval_set']} | {100 * c['mean']:+.2f} | {pci(c['t_ci95'])} | {pci(c['boot_ci95'])} | {pv} | {hp} | {c.get('decision') or '—'} |")
    for ti, mk in enumerate(METRICS, 1):
        L += ["", f"## Table {ti} — {METRIC_LABEL[mk]}", "", "| set | method | seed 0 | seed 1 | seed 2 | mean ± sd |", "|---|---|---|---|---|---|"]
        for es, ms in res["methods"].items():
            for mt, v in ms.items():
                ps = [pc(v[mk]["per_seed"].get(str(s))) if str(s) in v[mk]["per_seed"] else "—" for s in SEEDS]
                L.append(f"| {es} | {mt} | " + " | ".join(ps) + f" | {pc(v[mk]['mean'])}" + (f" ± {pc(v[mk]['sd'])}" if v[mk].get("sd") is not None else "") + " |")
        L += ["", f"Contrasts on this metric (paired by seed; t = 95 % seed-paired t interval, df = n - 1; boot = 95 % image bootstrap, B = {plan['statistics']['bootstrap']['B']})", "",
              "| family | contrast | set | mean | t interval | boot interval | p (two-sided) | Holm p | reading (t / boot) |", "|---|---|---|---|---|---|---|---|---|"]
        for c in res["contrasts"]:
            if c["metric"] == mk:
                hp = "" if c.get("holm_p_adj") is None else f"{c['holm_p_adj']:.4f}" + (" *" if c.get("holm_reject") else "")
                pv = "" if c["p_two_sided"] is None else f"{c['p_two_sided']:.4f}"
                L.append(f"| {c['family']} | {c['contrast']} | {c['eval_set']} | {100 * c['mean']:+.2f} | {pci(c['t_ci95'])} | {pci(c['boot_ci95'])} | "
                         f"{pv} | {hp} | {c['reading_t']} / {c['reading_boot']} |")
    L += ["", f"Primary family: {res['holm']['family']}; Holm at alpha = {res['holm']['alpha']} over m = {res['holm']['m']} evaluation sets (* = rejected).", ""]
    if res["missing_in_plan"]:
        L += ["Rows of the proposal without a checkpoint / lambda_CAL (listed, never filled):", ""] + [f"- {m['method']} {m['dataset']} seed {m['seed']}: {m['reason']}" for m in res["missing_in_plan"]] + [""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------- development only
def cmd_make_dev_refer(a) -> int:
    """A REFER-format pair (refs(umd).p, instances.json) holding ONLY RefCOCOg DEV content (the R4-VL0 DEV meta: eligible DEV images, raw texts,
    sentence ids, referred boxes), split label 'test' so that the official selection path is exercised.  Never touches the real release files."""
    import torch
    root = Path(a.root); meta = torch.load(META_DIR / "meta_refcocog.pt", weights_only=False); refs, images, anns = [], [], []
    for m in meta:
        images.append({"id": int(m["image_id"]), "width": int(m["width"]), "height": int(m["height"]), "file_name": Path(m["path"]).name})
        for oi, ann in enumerate(m["ref_ann"]):
            b = [float(v) for v in m["ref_boxes"][oi]]; bbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            anns.append({"id": int(ann), "image_id": int(m["image_id"]), "bbox": bbox, "category_id": int(m["ref_cat"][oi]), "area": bbox[2] * bbox[3], "iscrowd": 0})
            sents = [{"sent_id": int(sid), "raw": t, "sent": t.lower(), "tokens": t.lower().split()} for sid, t, k in zip(m["sent_id"], m["texts"], m["target"]) if int(k) == oi]
            refs.append({"ref_id": len(refs), "ann_id": int(ann), "image_id": int(m["image_id"]), "split": "test", "category_id": int(m["ref_cat"][oi]),
                         "sent_ids": [x["sent_id"] for x in sents], "sentences": sents, "file_name": Path(m["path"]).name})
    d = root / "dev_refer"; d.mkdir(parents=True, exist_ok=True)
    s1 = write_atomic(d / "refs(umd).p", pickle.dumps(refs)); s2 = write_atomic(d / "instances.json", json.dumps({"images": images, "annotations": anns, "categories": []}).encode())
    write_atomic(d / "README.txt", ("Development file: RefCOCOg DEV content only (outputs/R4_VL0/meta_refcocog.pt), REFER format, split label 'test' used only to "
                                    "exercise the official selection code path.  Not the release file.\n").encode())
    log(f"dev REFER files: {len(images)} images, {len(refs)} refs, {sum(len(r['sentences']) for r in refs)} sentences; sha256 {s1} / {s2}"); return 0


def cmd_integrity(a) -> int:
    """Development: a DEV-mode score against the recorded per-run DEV values (same eligible images: >= 2 referred objects) — the image-macro
    and query-level Acc@0.5 of each run's CAL-selected state (scripts/r4_vl1_finetune.py) and the R5-10 F1 fused values (fusion_F0_F1.json);
    tolerance 0.3 point (the R4-VL0 G2 gate: fp16 reload, bf16 batch composition, regenerated proposals)."""
    root = Path(a.root); plan, _ = load_plan(root, a.plan)
    if plan["mode"] == "test":
        raise SystemExit("integrity is a development check; refused in test mode")
    res = jload(root / "score" / "results.json"); fusion = jload(FUSION_JSON); out, per_method = [], {}
    for rn, r in sorted(res["rows"].items()):
        mt, s, es = r["method"], r["seed"], r["eval_set"]; ds = plan["eval_sets"][es]["dataset"]
        if mt.endswith("+F"):
            base = mt[:-2]; rec = fusion["F1"][ds]["models"][FUSION_KEY[base]][str(s)]; ref_m, ref_q = rec["dev_at_cal_lambda"], None
            src = f"{FUSION_JSON.relative_to(REPO)} F1.{ds}.{FUSION_KEY[base]}.{s}.dev_at_cal_lambda (lambda_CAL {rec['lambda_cal']})"
        else:
            rj = jload(run_json_path(mt, ds, s))["dev_at_selected"]["acc50"]; ref_m, ref_q = rj["image_macro"], rj["query"]
            src = f"{run_json_path(mt, ds, s).relative_to(OUTPUTS.parent)} dev_at_selected.acc50"
        dm, dq = r["image_macro_ge2"] - ref_m, (None if ref_q is None else r["query_ge2"] - ref_q)
        ok = abs(dm) <= 0.003 and (dq is None or abs(dq) <= 0.003)
        out.append({"row": rn, "recorded_image_macro": ref_m, "new_image_macro_ge2": r["image_macro_ge2"], "diff_image_macro": dm, "recorded_query": ref_q,
                    "new_query_ge2": r["query_ge2"], "diff_query": dq, "pass": ok, "recorded_source": src})
        per_method.setdefault((es, mt), []).append((ref_m, r["image_macro_ge2"]))
    means = [{"eval_set": es, "method": mt, "n_seeds": len(v), "recorded_mean": float(np.mean([x for x, _ in v])), "new_mean": float(np.mean([y for _, y in v])),
              "diff": float(np.mean([y - x for x, y in v]))} for (es, mt), v in sorted(per_method.items())]
    bl = build_log(root); pl = jload(root / "logs" / "proposals.json")
    rep = {"rows": out, "method_means": means, "all_rows_pass": all(o["pass"] for o in out), "tolerance": 0.003,
           "build_checks": {es: v["checks"] for es, v in bl["eval_sets"].items()}, "proposal_checks": {es: v.get("vs_stored_dev_proposals") for es, v in pl["eval_sets"].items()}}
    write_atomic(root / "integrity.json", json.dumps(rep, indent=1, sort_keys=True).encode())
    L = ["# r5 final evaluation — DEV integrity check", "", f"All rows within 0.3 point: **{rep['all_rows_pass']}** ({sum(o['pass'] for o in out)} / {len(out)}).", "",
         "| row | recorded image-macro | new | diff | recorded query | new | diff | pass |", "|---|---|---|---|---|---|---|---|"]
    for o in out:
        L.append(f"| {o['row']} | {100 * o['recorded_image_macro']:.2f} | {100 * o['new_image_macro_ge2']:.2f} | {100 * o['diff_image_macro']:+.2f} | "
                 + ("— | — | — |" if o["recorded_query"] is None else f"{100 * o['recorded_query']:.2f} | {100 * o['new_query_ge2']:.2f} | {100 * o['diff_query']:+.2f} |") + f" {o['pass']} |")
    L += ["", "| set | method | seeds | recorded mean | new mean | diff |", "|---|---|---|---|---|---|"]
    L += [f"| {m['eval_set']} | {m['method']} | {m['n_seeds']} | {100 * m['recorded_mean']:.2f} | {100 * m['new_mean']:.2f} | {100 * m['diff']:+.2f} |" for m in means]
    (root / "integrity.md").write_text("\n".join(L) + "\n"); print("\n".join(L), flush=True)
    return 0 if rep["all_rows_pass"] else 1


def cmd_rows_yaml(a) -> int:
    """The frozen row list of a plan as a YAML fragment for the protocol: every checkpoint (method, dataset, seed, path, sha256, CAL-selected epoch,
    critic slope), every row (evaluation set, rank rule with lambda_CAL), detector / backbone hashes and the rows without checkpoint / lambda."""
    plan, sha = load_plan(Path(a.root), a.plan); q = lambda v: json.dumps(v, ensure_ascii=False)
    order = {m: i for i, m in enumerate(METHODS)}; ds_order = {"refcocog": 0, "refcocoplus": 1, "refcoco": 2}
    L = [f"plan_file: {q(str(Path(a.plan) if a.plan else Path(a.root) / 'plan.json'))}", f"plan_sha256: {sha}", f"plan_mode: {plan['mode']}",
         f"detector: {{path: {q(plan['detector']['path'])}, sha256: {plan['detector'].get('sha256')}, recorded_sha256: {plan['detector'].get('recorded_sha256')}, "
         f"matches_record: {str(plan['detector'].get('matches_record')).lower()}}}",
         f"backbone: {{hf_repo: {plan['backbone']['hf_repo']}, snapshot: {plan['backbone'].get('snapshot')}, files_sha256: {q(plan['backbone'].get('files_sha256'))}, "
         f"vs_r5_hashes: {q(plan['backbone'].get('vs_r5_hashes'))}}}", "evaluation_sets:"]
    for es, v in plan["eval_sets"].items():
        L.append(f"  - {{id: {es}, dataset: {v['dataset']}, split: {v['split']}, format: {v['format']}, sources: {q(v['sources'])}, expected_counts: {q(v.get('expected'))}, "
                 f"archive: {q(None if not v.get('archive_check') else {k: v['archive_check'][k] for k in ('path', 'recorded_sha256', 'recorded_in')})}}}")
    L.append(f"checkpoints:   # {len(plan['models'])} CAL-selected checkpoints (selection rule: CAL all-proposal image-macro Acc@0.5, ties earliest epoch)")
    for mid, m in sorted(plan["models"].items(), key=lambda kv: (ds_order[kv[1]["dataset"]], order[kv[1]["method"]], kv[1]["seed"])):
        L.append(f"  - {{id: {mid}, method: {m['method']}, dataset: {m['dataset']}, seed: {m['seed']}, cal_selected_epoch: {m['cal_selected_epoch']}, "
                 f"critic_a: {m['critic_a']:.6f}, path: {q(m['path'])}, sha256: {m.get('sha256')}}}")
    L.append(f"rows:   # {len(plan['rows'])} rows = checkpoint x official split x rank rule")
    for r in sorted(plan["rows"], key=lambda r: (r["eval_set"], r["method"], r["seed"])):
        L.append(f"  - {{row: {q(r['row'])}, model: {r['model']}, rank: {q(r['rank'])}}}")
    L.append("rows_without_checkpoint_or_lambda:")
    L += [f"  - {{method: {q(m['method'])}, dataset: {m['dataset']}, seed: {m['seed']}, reason: {q(m['reason'])}}}" for m in plan["missing"]]
    txt = "\n".join(L) + "\n"
    if a.out:
        write_atomic(Path(a.out), txt.encode()); log(f"rows YAML -> {a.out}")
    else:
        print(txt)
    return 0


# ----------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", default=str(ROOT_DEFAULT)); p.add_argument("--plan", default=None, help="default <root>/plan.json"); return p
    p = common(sub.add_parser("plan")); p.add_argument("--mode", required=True, choices=["test", "dev", "smoke"])
    p.add_argument("--datasets", nargs="+", default=["refcocog", "refcocoplus", "refcoco"], choices=["refcocog", "refcocoplus", "refcoco"])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS)); p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--max-images", type=int, default=None); p.add_argument("--no-hash", action="store_true"); p.add_argument("--out", default=None)
    p.add_argument("--shard-by", default="family", choices=["family", "dataset", "all"], help="rank job grouping (family = dataset x checkpoint directory)")
    p.add_argument("--dev-proposals", default="regenerated", choices=["regenerated", "stored"], help="dev / smoke only: rank the regenerated or the stored VL1-21 DEV proposals")
    common(sub.add_parser("build")); common(sub.add_parser("proposals"))
    p = common(sub.add_parser("rank")); p.add_argument("--shard", required=True); p.add_argument("--workers", type=int, default=8)
    p = common(sub.add_parser("manifest")); p.add_argument("--allow-code-change", action="store_true", help="implementation-error path only (recorded)")
    p = common(sub.add_parser("score")); p.add_argument("--out", default=None, help="default <root>/score; another directory = deterministic re-verification")
    common(sub.add_parser("make-dev-refer")); common(sub.add_parser("integrity"))
    p = common(sub.add_parser("rows-yaml")); p.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "plan" and a.mode == "smoke" and not a.max_images:
        a.max_images = 40
    if a.cmd == "plan" and a.mode != "test" and Path(a.root).resolve() == ROOT_DEFAULT.resolve():
        raise SystemExit("development plans must use a root other than the official one")
    if a.cmd == "plan" and a.mode == "test" and Path(a.root).resolve() != ROOT_DEFAULT.resolve():
        log(f"note: test-mode root {a.root} differs from the official {ROOT_DEFAULT}")
    return {"plan": cmd_plan, "build": cmd_build, "proposals": cmd_proposals, "rank": cmd_rank, "manifest": cmd_manifest, "score": cmd_score,
            "make-dev-refer": cmd_make_dev_refer, "integrity": cmd_integrity, "rows-yaml": cmd_rows_yaml}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
