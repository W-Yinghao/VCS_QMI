"""r7 R7-50 — one-shot official-test evaluation (owner approval 2026-10-12 10:16 UTC: reports/OWNER_DECISIONS_20261012.md T1 / 七;
reports/R7/R7_50_60_OUTLINE_DRAFT.md §A; protocol reports/R7/R7_50_PROTOCOL_*.yaml).  GQA testdev-balanced (12,578 q) and ChartQA test (2,500 q);
TextVQA test is not read.  Models, prompts, evaluators and statistics are the frozen D1-30 code, imported read-only (never edited):
r7_d1_30_run (quality runner), r7_d1_30_sparsevlm (official SparseVLM rows, E3 counter check), r7_d1_30_plan (configuration grid, partition
rules), r7_d1_30_score (official scores, group bootstrap, interpolation, contrasts, Holm), r7_d1_30_data / r7_d1_data (prompts, ChartQA type rule,
image extraction, GQA question reader), r7_vl_common.  This module adds the sealed-test plumbing only.
Phases (rows of all three declared in the protocol before any read; each phase is hash-committed, then scored once; test results never choose rows):
  L  LLaVA-1.5 rows with the D1-20 selectors (10k FIT images): full, uniform grid, S1 / S2 / S3 × seeds 0-2 × 4 budgets, S1 dynamic × 3 operating
     points, 8 SparseVLM rows — after D1-30 stage 1
  F  the same selector rows (S1 / S2 / S3 × seeds 0-2 × 4 budgets + S1 dynamic) with the D1-FULL selectors (retrained on all 62,852 eligible GQA
     train images, same CAL / DEV / recipe) — after D1-FULL; full / grid / SparseVLM baseline rows are phase L's (shared, not regenerated)
  Q  Qwen2.5-VL-7B rows — after R7-40 (its addendum and the R7-40c runner; refused here until then)
Sub-commands:
  build      CPU  opens each approved test file ONCE (size and sha256 of the bytes read checked against reports/R7/R7_DATA_CLOSED_SHA256.csv before
                  anything is parsed); inputs WITHOUT answers in the D1-30 set format (<root>/sealed/inputs/<bench>.jsonl), ChartQA images extracted
                  into the sealed input dir, a separate sealed label file per benchmark (<root>/sealed/labels/), counts / hashes / §A.3b overlap
                  counts only.  Closed sources need the frozen R7-50 protocol ('status: frozen', top-level 'r7_50_freeze_<yyyymmdd>:').  GQA images
                  only from the extracted folder (missing / undecodable -> STOP, no archive fallback).  Shared by every phase.
  configs / lines / prepare     the phase's rows from the frozen D1-30 planner grid (r7_d1_30_plan.write_configs on the phase's lr_choice.json),
                  selector hashes (configs_<P>/rows.json) and GPU unit lines (D1-30 partition rules and shard counts) with completion markers
  integrity-prepare / integrity-run / integrity-compare     full-scale integrity gate (owner directive 2026-10-12): the R7-50 path on the FULL
                  GQA DEV and ChartQA val sets with D1-30's own question ids, reproducing D1-30's sharding / chunking on the GPU class recorded in
                  each reference shard's completion marker; every answer must equal D1-30's, item by item (full, grid-144, SparseVLM-192 and S1
                  seed 0 at 25 % for phase L); any mismatch stops the driver before a test file is opened
  run / sparsevlm GPU   the D1-30 runners on the sealed inputs -> <root>/sealed/pred_<P>/<bench>/<cfg>@s<i>of<n>.jsonl (single writer, E1)
  manifest   CPU  coverage of every row (labels not opened), sha256 of every prediction file and of the profiling-workload timing files
  score      CPU  the only step that opens the sealed labels; refuses unless <root>/ALLOW_SCORING_<P> names the manifest's sha256 (production root: and
                  the commit holding a copy of it) and every prediction / timing file still matches; latency copied from the frozen D1-30 timing
                  ('measured on the pre-fixed profiling workload'); results JSON + markdown
Bench names gqa_testdev / chartqa_test; their D1-30 counterparts gqa_dev / chartqa_val supply the frozen generation length (MAX_NEW 128 / 16), the
configuration grid, the partition rule and the profiling-workload latency.
    python -I scripts/r7_50_test.py prepare --root R --protocol P            python -I scripts/r7_50_test.py run --root R --bench gqa_testdev --shard 0/6
    python -I scripts/r7_50_test.py manifest --root R --phase L              python -I scripts/r7_50_test.py score --root R --phase L
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_data as DD  # noqa: E402   (prompt suffixes, ChartQA type rule, image extraction; read-only use)
import r7_d1_30_plan as PL  # noqa: E402   (frozen configuration grid, partition rules, shard counts)
import r7_d1_30_run as RUN  # noqa: E402   (quality runner, MAX_NEW, unit names)
import r7_d1_30_score as SC  # noqa: E402   (official scores, bootstrap, interpolation, contrasts, Holm)
import r7_d1_30_sparsevlm as SV  # noqa: E402   (official SparseVLM rows)
import r7_d1_data as DA  # noqa: E402   (GQA official-row reader, sha256)
import r7_vl_common as V  # noqa: E402   (tolerant JSONL reader for prediction files, as the D1-30 runner)

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R7_50")
DATA = Path("/projects/EEG-foundation-model/yinghao/data")
D1 = RUN.D1
D30 = D1 / "d1_30"
D30_PRED, D30_SETS = D30 / "pred", D30 / "sets"
D30_LOG = Path("/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_30_submit.log")
GQA_TEST = DATA / "gqa" / "closed" / "testdev_balanced_questions.json"
CHARTQA_TEST = DATA / "chartqa" / "closed" / "data" / "test-00000-of-00001.parquet"
CHARTQA_VAL = DATA / "chartqa" / "data" / "val-00000-of-00001.parquet"
GQA_IMAGES = DATA / "gqa" / "images"
CLOSED_CSV = REPO / "reports" / "R7" / "R7_DATA_CLOSED_SHA256.csv"
MIX665K = DATA / "llava_mix665k" / "llava_v1_5_mix665k.json"           # LLaVA-1.5 instruction mix (A.3b known-overlap check)
SELECTORS = D1 / "selectors"
PCA = D1 / "cache" / "pca.pt"
LR_CHOICE = SELECTORS / "lr_choice.json"
TIMING_DIR = D30 / "timing"
D30_CONFIGS = D30 / "configs"
PHASES = ("L", "F", "Q")
BENCHES = ("gqa_testdev", "chartqa_test")
D30_BENCH = {"gqa_testdev": "gqa_dev", "chartqa_test": "chartqa_val"}
EXPECTED_ITEMS = {"gqa_testdev": 12578, "chartqa_test": 2500}          # §A.1
QUALITY_SHARDS = {"gqa_testdev": 6, "chartqa_test": 2}                 # as D1-30 (gqa_dev 6, chartqa_val 2)
SV_SHARDS = {"gqa_testdev": 3, "chartqa_test": 1}                      # as D1-30 (gqa_dev 3, chartqa_val 1)
N_ROWS = {"L": (50, 8), "F": (45, 0)}                                  # (pipeline rows generated in the phase, SparseVLM rows)
BASELINE_FROM_L = ("full", "static:grid:72", "static:grid:144", "static:grid:288", "static:grid:432")   # + the 8 SparseVLM rows (phase F)
FREEZE_RE = re.compile(r"^r7_50_freeze_\d{8}\s*:", re.M)
STATUS_RE = re.compile(r"^status:\s*frozen\b", re.M)
TEST_NAME_RE = re.compile(r"(test|challenge|submission)", re.I)
CHART_COLUMNS = ("imgname", "query", "label", "type", "image")
R750_FILES = ("scripts/r7_50_test.py", "slurm/r7_50_cpu.sbatch", "slurm/r7_50_gpu.sbatch", "slurm/r7_50_submit.sh")
GPU_PIN = {"L40S": "--partition=L40S --exclude=node51,node52,node60", "H100": "--partition=H100 --exclude=node51,node52,node60"}
SV_AGREEMENT_MIN = 0.98      # coordinator decision 2026-10-12, fixed before any rehearsal number: SparseVLM gate rows need >= 98.0 % identical
                              # answers (documented run-to-run agreement 99.1 %), identical kept counts and the E3 counter within tolerance; every
                              # deterministic row (full, grid-144, S1 seed 0 at 144) must be identical on every item.  Same rule for phase F's gate.
GATE_RULE = ("deterministic rows (full, grid-144, S1 seed 0 at 144): answer and kept count identical on every item; SparseVLM-192: answer "
             f"agreement >= {SV_AGREEMENT_MIN:.3f}, kept counts identical, E3 counter within tolerance; inputs identical on every item; anything else "
             "-> STOP (coordinator decision 2026-10-12, fixed before any rehearsal number)")
UNIT_TIME, INTEGRITY_TIME = "04:00:00", "03:00:00"   # measured: <= 1.4 h per generation unit, <= 0.5 h per integrity unit on L40S (backfill-friendly)
SV_DEFAULT_CLASS = "L40S"     # every D1-30 SparseVLM shard ran on L40S (slurm_logs/r7d30v_*: node39 / node50 / node57); markers after E3 carry gpu None
LATENCY_LABEL = ("latency measured on the pre-fixed profiling workload: the frozen D1-30 timing results of the mapped validation benchmark "
                 "(gqa_dev for GQA testdev-balanced, chartqa_val for ChartQA test; NVIDIA H100 NVL on nodesumo01, 8 forced decode steps, "
                 "B* = 4 primary).  Not re-measured on test and not the test sets' actual mean end-to-end latency (it depends on input shapes, "
                 "generation lengths and batch composition)")
EXPOSURE = [
    "Relative to this project: the test splits were never used for selector training, hyperparameters or any selection; every row, "
    "checkpoint, operating point and statistic was fixed before the read.",
    "Relative to the hosts' public training data: LLaVA-1.5's instruction mix (llava_v1_5_mix665k.json) includes GQA train-balanced, Visual "
    "Genome, COCO and TextCaps.  GQA's test splits extend Visual Genome with new images whose scene graphs are hidden (Hudson & Manning 2019), "
    "so test-dev images need not be Visual Genome images: the build counts the test-dev image ids by prefix ('n'-prefixed vs numeric Visual "
    "Genome ids) and the test-dev images and (image, question) pairs present in the mix (sources.<bench>.exposure_check; the measured sentence "
    "is exposure_statement_measured).  ChartQA is not part of LLaVA-1.5's published instruction mix.",
    "Undisclosed upstream pre-training exposure (CLIP ViT-L/14-336, Vicuna) is not guaranteed either way.",
]


# ------------------------------------------------------------------------------------------------------------------------------ io
def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p) -> str:
    return DA.sha256_file(Path(p))


def read_jsonl_strict(p):
    """Every non-blank line must parse (inputs / labels are written whole; a torn line is an error, not a skip)."""
    out = []
    for i, line in enumerate(Path(p).read_text().splitlines()):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{p}: line {i + 1} does not parse ({e.msg})")
    return out


def write_atomic(p, text: str, mode=None):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, p)


def jsonl(rows) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def git_state():
    """HEAD and whether the R7-50 files differ from it (recorded; the driver's gate 0 enforces a clean, committed state)."""
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *R750_FILES], capture_output=True, text=True, timeout=60).stdout
        return {"head": head or None, "r7_50_files_dirty": bool(dirty.strip())}
    except Exception as e:                                                             # recorded, never fatal
        return {"head": None, "error": type(e).__name__}


class Layout:
    """Shared by every phase: <root>/sealed/{inputs,labels} (built once, by the one read of each test file).  Per phase P: <root>/sealed/pred_<P>,
    <root>/configs_<P>, <root>/lines_<P>.txt, <root>/manifest_<P>/, <root>/ALLOW_SCORING_<P>, <root>/results_<P>, <root>/integrity_<P> (gate)."""

    def __init__(self, root, phase="L"):
        if phase not in PHASES:
            raise SystemExit(f"unknown phase {phase!r} (L, F, Q)")
        self.root, self.phase = Path(root), phase; s = self.root / "sealed"
        self.inputs, self.labels, self.pred = s / "inputs", s / "labels", s / f"pred_{phase}"
        self.configs, self.lines, self.results = self.root / f"configs_{phase}", self.root / f"lines_{phase}.txt", self.root / f"results_{phase}"
        self.summary = self.inputs / "build_summary.json"; md = self.root / f"manifest_{phase}"
        self.manifest, self.manifest_sha = md / "predictions_manifest.json", md / "predictions_manifest.sha256"
        self.allow, self.integrity = self.root / f"ALLOW_SCORING_{phase}", self.root / f"integrity_{phase}"

    def bench_summary(self, b):
        return self.inputs / f"build_{b}.json"

    def inputs_file(self, b):
        return self.inputs / f"{b}.jsonl"

    def labels_file(self, b):
        return self.labels / f"{b}.labels.jsonl"

    def images_file(self, b):
        return self.inputs / f"images_{b}.json"

    def configs_file(self, b):
        return self.configs / f"configs_{b}.txt"

    def sv_rows_file(self, b):
        return self.configs / f"sparsevlm_rows_{b}.txt"


def load_summary(L: Layout):
    if not L.summary.is_file():
        raise SystemExit(f"{L.summary} missing: inputs not built")
    s = json.loads(L.summary.read_text())
    if s.get("ok") is not True:
        raise SystemExit(f"{L.summary}: build not ok")
    return s


def walk_files(d: Path):
    """Regular files under d (symlinks and the host's hub_links symlink farm excluded; never follows links)."""
    out = []
    for dp, dn, fn in os.walk(d, followlinks=False):
        dn[:] = sorted(x for x in dn if x != "hub_links")
        for f in fn:
            p = Path(dp) / f
            if not p.is_symlink() and p.is_file():
                out.append(p)
    return sorted(out)


# ------------------------------------------------------------------------------------------------------------------------------ build
def recorded(src, data_root=DATA, csv_path=CLOSED_CSV):
    """Download-time record of a closed file (path relative to the data root, bytes, sha256); None when absent."""
    try:
        rel = Path(src).relative_to(Path(data_root)).as_posix()
    except ValueError:
        return None
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["path"] == rel:
                return {"path": rel, "bytes": int(row["bytes"]), "sha256": row["sha256"], "record_file": str(csv_path)}
    return None


def is_closed(p) -> bool:
    return any(part.lower() == "closed" for part in Path(p).parts)


def require_freeze(protocol):
    if not protocol or not Path(protocol).is_file():
        raise SystemExit("a closed (official-test) source needs --protocol <the frozen R7-50 protocol>; nothing was opened")
    t = Path(protocol).read_text(); m = FREEZE_RE.search(t)
    if not (m and STATUS_RE.search(t)):
        raise SystemExit(f"{protocol} is not a frozen R7-50 protocol (needs 'status: frozen' and a top-level 'r7_50_freeze_<yyyymmdd>:' key); "
                         "the test files stay unopened")
    return {"protocol": str(protocol), "sha256": sha(protocol), "freeze_key": m.group(0).rstrip(": \t")}


def read_once(src, rec):
    """One open() of the source; with a record, the size is checked before the open and the sha256 of the bytes before any parsing."""
    if rec is not None:
        size = os.stat(src).st_size
        if size != rec["bytes"]:
            raise SystemExit(f"{src}: {size} bytes on disk, {rec['bytes']} recorded — not opened")
    with open(src, "rb") as fh:
        data = fh.read()
    digest = hashlib.sha256(data).hexdigest()
    if rec is not None and digest != rec["sha256"]:
        raise SystemExit(f"{src}: sha256 of the bytes read {digest} != recorded {rec['sha256']} — not parsed")
    return data, digest


def gqa_rows(data: bytes, bench, images_dir):
    """GQA balanced questions file ({qid: {imageId, question, answer, types, ...}}) -> (inputs, labels); the D1-30 GQA prompt."""
    d = json.loads(data)
    if not isinstance(d, dict):
        raise SystemExit("GQA questions file is not a {question id: question} object")
    items, labels = [], []
    for r in DA._official_rows(d):
        items.append({"bench": bench, "qid": r["qid"], "image_id": r["image_id"], "image_file": str(Path(images_dir) / f"{r['image_id']}.jpg"),
                      "question": r["question"], "prompt": r["question"] + DD.SHORT})
        labels.append({"qid": r["qid"], "answer": r["answer"], "structural": r["structural"], "semantic": r["semantic"], "detailed": r["detailed"]})
    return items, labels


def gqa_images(items):
    """Every referenced image must exist in the extracted folder and decode; sha256 per image.  Missing -> STOP (no archive fallback)."""
    from PIL import Image
    files = sorted({it["image_file"] for it in items}); missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(files)} GQA images missing from the extracted folder {Path(files[0]).parent} "
                         f"(first: {[Path(f).stem for f in missing[:10]]}); no fallback to any archive")
    man, bad = {}, []
    for f in files:
        raw = Path(f).read_bytes()
        try:
            im = Image.open(io.BytesIO(raw)); fmt = im.format; im.verify()
        except Exception:
            bad.append(Path(f).stem); continue
        man[Path(f).stem] = {"file": f, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "format": fmt}
    if bad:
        raise SystemExit(f"{len(bad)} GQA images do not decode (first: {bad[:10]})")
    return man


def chartqa_rows(data: bytes, bench, inputs_dir, qid_prefix=None):
    """ChartQA parquet (imgname, query, label, type, image) -> (inputs, labels, images, extra).  The frozen D1-30 rule (r7_d1_30_data.chartqa_items:
    'type' must name exactly one of human / augmented; prompt = query + the pinned lmms-eval suffix) with qid <bench>_<row index>; images
    written once per chart name (original bytes, r7_d1_30_data.write_image); a chart name carrying two different images stops the build."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(pa.BufferReader(data)); sch = pf.schema_arrow
    missing = [c for c in CHART_COLUMNS if c not in sch.names]
    if missing:
        raise SystemExit(f"ChartQA parquet lacks columns {missing} (columns {sch.names})")
    for c in ("imgname", "query", "label", "type"):
        if not (pa.types.is_string(sch.field(c).type) or pa.types.is_large_string(sch.field(c).type)):
            raise SystemExit(f"ChartQA column {c} has type {sch.field(c).type}; the frozen evaluator compares strings")
    rows, paths, img_sha, conflicts = [], {}, {}, set()
    for b in pf.iter_batches(batch_size=64, columns=list(CHART_COLUMNS)):
        for r in b.to_pylist():
            img = r.pop("image"); raw = img["bytes"] if isinstance(img, dict) else img; h = hashlib.sha256(raw).hexdigest()
            if img_sha.setdefault(r["imgname"], h) != h:
                conflicts.add(r["imgname"])
            DD.write_image(bench, r["imgname"], raw, Path(inputs_dir), paths); rows.append(r)
    if conflicts:
        raise SystemExit(f"{len(conflicts)} chart names carry different image bytes (first {sorted(conflicts)[:5]}): one image per name would mis-assign them")
    stale = [n for n, p in paths.items() if sha(p) != img_sha[n]]
    if stale:
        raise SystemExit(f"{len(stale)} extracted chart images differ from the parquet bytes (stale files in {inputs_dir}?)")
    its = DD.chartqa_items(rows); items, labels = [], []
    for i, it in enumerate(its):
        if it["qid"] != f"chartqa_val_{i:05d}":
            raise SystemExit("unexpected ChartQA item order")
        qid = f"{qid_prefix or bench}_{i:05d}"
        items.append({"bench": bench, "qid": qid, "image_id": it["image_id"], "image_file": paths[it["image_id"]], "question": it["question"], "prompt": it["prompt"]})
        labels.append({"qid": qid, "answer": it["answer"], "type": it["type"], "split": it["split"]})
    man = {n: {"file": paths[n], "bytes": Path(paths[n]).stat().st_size, "sha256": img_sha[n]} for n in sorted(paths)}
    extra = {"schema": {f.name: str(f.type) for f in sch}, "parquet_rows": pf.metadata.num_rows,
             "split_counts": {k: sum(lab["split"] == k for lab in labels) for k in ("human", "augmented")}}
    return items, labels, man, extra


def mix_index(path):
    """LLaVA-1.5 instruction mix: image stems under gqa/ and vg/, the (image id, normalised question) pairs of its GQA human turns (the frozen D1
    match rule of r7_d1_data.llava_seen_pairs) and the number of entries whose image path names a chart source."""
    gqa, vg, seen, chart = set(), set(), set(), 0
    for e in json.loads(Path(path).read_text()):
        img = str(e.get("image", "")); stem = Path(img).stem
        if img.startswith("gqa/"):
            gqa.add(stem)
            for turn in e.get("conversations", []):
                if turn.get("from") == "human":
                    seen.add((stem, DA.norm_q(turn.get("value", ""))))
        elif img.startswith("vg/"):
            vg.add(stem)
        chart += "chart" in img.lower()
    return {"file": str(path), "sha256": sha(path), "gqa_images": gqa, "vg_images": vg, "gqa_pairs": seen, "chart_entries": chart}


def exposure(bench, items, mix):
    """§A.3b known-overlap check (counts only): GQA — test images among the mix's gqa/ and vg/ image stems, test (image, normalised question)
    pairs among the mix's GQA human turns, numeric (Visual Genome-style) image ids; ChartQA — mix entries with a chart image path."""
    if mix is None:
        return None
    base = {"mix_file": mix["file"], "mix_sha256": mix["sha256"]}
    if bench.startswith("gqa"):
        ids = {it["image_id"] for it in items}
        n_pref = sum(i.startswith("n") for i in ids); num = sum(i.isdigit() for i in ids)
        return dict(base, images=len(ids), numeric_image_ids=num, n_prefixed_image_ids=n_pref, other_image_ids=len(ids) - num - n_pref,
                    images_in_mix_gqa_paths=len(ids & mix["gqa_images"]),
                    images_in_mix_vg_paths=len(ids & mix["vg_images"]), images_in_mix_any=len(ids & (mix["gqa_images"] | mix["vg_images"])),
                    questions=len(items), questions_seen_in_mix_gqa_turns=sum((it["image_id"], DA.norm_q(it["question"])) in mix["gqa_pairs"] for it in items),
                    rule="image id = file stem of the mix's gqa/ and vg/ image paths; question pairs by the frozen D1 match rule (r7_d1_data.MATCH_RULE)")
    return dict(base, mix_entries_with_chart_image_paths=mix["chart_entries"], note="ChartQA is not part of LLaVA-1.5's published instruction mix")


def build_bench(L: Layout, bench, src, closed, images_dir, data_root, csv_path, expected, mix=None, chartqa_qid_prefix=None):
    rec = recorded(src, data_root, csv_path) if closed else None
    if closed and rec is None:
        raise SystemExit(f"{src}: no download-time sha256 record in {csv_path}; not opened")
    if not closed and TEST_NAME_RE.search(Path(src).name):
        raise SystemExit(f"{src}: a test-type file name outside closed/ is refused in development mode; not opened")
    if closed and bench.startswith("gqa") and Path(images_dir) != GQA_IMAGES:
        raise SystemExit(f"official GQA images come only from {GQA_IMAGES}")
    t0 = time.time(); data, digest = read_once(src, rec); n_bytes = len(data)
    if bench.startswith("gqa"):
        items, labels = gqa_rows(data, bench, images_dir); del data; img = gqa_images(items); extra = {}
    else:
        items, labels, img, extra = chartqa_rows(data, bench, L.inputs, chartqa_qid_prefix); del data
    n_exp = expected.get(bench) if closed else None
    if n_exp is not None and len(items) != n_exp:
        raise SystemExit(f"{bench}: {len(items)} items read, {n_exp} expected (§A.1); no input or label file written")
    qids = [it["qid"] for it in items]
    if len(set(qids)) != len(qids):
        raise SystemExit(f"{bench}: duplicate question ids")
    empty = sum(not isinstance(lab["answer"], str) or not lab["answer"].strip() for lab in labels)
    write_atomic(L.inputs_file(bench), jsonl(items)); write_atomic(L.labels_file(bench), jsonl(labels), mode=0o400)
    write_atomic(L.images_file(bench), json.dumps(img, indent=1, sort_keys=True))
    s = {"bench": bench, "mode": "official test (closed source, one read)" if closed else "development (non-test source; not a test split)",
         "source": str(src), "recorded": rec, "bytes_read": n_bytes, "sha256": digest, "sha256_matches_record": None if rec is None else True,
         "items": len(items), "groups": len({it["image_id"] for it in items}), "images": len(img), "labels_empty_answers": empty,
         "inputs_file": str(L.inputs_file(bench)), "inputs_sha256": sha(L.inputs_file(bench)), "inputs_fields": sorted(items[0]) if items else [],
         "labels_file": str(L.labels_file(bench)), "labels_sha256": sha(L.labels_file(bench)), "images_manifest_sha256": sha(L.images_file(bench)),
         "built_utc": utc(), "seconds": round(time.time() - t0, 1), "exposure_check": exposure(bench, items, mix), **extra}
    write_atomic(L.bench_summary(bench), json.dumps(s, indent=1))
    return s


def build(root, gqa_src=GQA_TEST, chartqa_src=CHARTQA_TEST, protocol=None, images_dir=GQA_IMAGES, data_root=DATA, csv_path=CLOSED_CSV,
          expected=EXPECTED_ITEMS, mix665k=None, chartqa_qid_prefix=None):
    """Sealed inputs + labels for both benchmarks (one read per source; a benchmark already built in an earlier, failed attempt is kept).
    mix665k: LLaVA-1.5's instruction mix for the §A.3b overlap counts — required (and loaded before any test file is opened) for closed sources.
    chartqa_qid_prefix: development only (the smoke's like-for-like comparison with D1-30's chartqa_val ids); the official ids are
    chartqa_test_<row index>, and qid-seeded masks (the uniform grid) are seeded by them."""
    L = Layout(root)
    if L.summary.exists():
        raise SystemExit(f"{L.summary} exists: the build is one-shot")
    srcs = {"gqa_testdev": Path(gqa_src), "chartqa_test": Path(chartqa_src)}; closed = {b: is_closed(p) for b, p in srcs.items()}
    if chartqa_qid_prefix not in (None, "chartqa_test") and any(closed.values()):
        raise SystemExit("the official ChartQA ids are chartqa_test_<row index>; no prefix override for closed sources")
    freeze = require_freeze(protocol) if any(closed.values()) else None
    if any(closed.values()) and not (mix665k and Path(mix665k).is_file()):
        raise SystemExit(f"the official build needs LLaVA-1.5's instruction mix for the §A.3b overlap check ({mix665k}); nothing was opened")
    mix = mix_index(mix665k) if mix665k else None
    L.inputs.mkdir(parents=True, exist_ok=True); L.labels.mkdir(parents=True, exist_ok=True)
    out = {"started_utc": utc(), "root": str(L.root), "freeze": freeze, "git": git_state(), "benches": {}, "labels_printed": False}
    for b, src in srcs.items():
        if L.bench_summary(b).exists():
            prev = json.loads(L.bench_summary(b).read_text())
            if prev["source"] != str(src):
                raise SystemExit(f"{b}: an earlier attempt built {prev['source']}, now {src}")
            out["benches"][b] = dict(prev, kept_from_earlier_attempt=True); continue
        out["benches"][b] = build_bench(L, b, src, closed[b], images_dir, data_root, csv_path, expected, mix, chartqa_qid_prefix)
    out["ok"] = True; out["finished_utc"] = utc(); write_atomic(L.summary, json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ rows
def main_units(lr_choice):
    """(choice, main units {objective: [3 tags]}) from a D1-20 / D1-FULL lr_choice.json (frozen D1-30 naming)."""
    choice = json.loads(Path(lr_choice).read_text()); main, _ = RUN.unit_names(choice)
    return choice, main


def a2_rows(grid, main):
    """§A.2 pipeline rows from the frozen D1-30 grid: full, uniform grid, S1 / S2 / S3 fixed budgets, S1 dynamic (order kept)."""
    units = {u for o in RUN.OBJECTIVES for u in main[o]}; s1 = set(main["vcs"]); out = []
    for c in grid:
        p = c.split(":")
        if c == "full" or c.startswith("static:grid:") or (p[0] == "sel" and p[1] in units) or (p[0] == "dyn" and p[1] in s1):
            out.append(c)
    return out


def phase_rows(grid, main, phase):
    """Rows GENERATED in a phase: L = the 50 §A.2 pipeline rows; F = their 45 selector rows (sel / dyn) with the D1-FULL units."""
    rows = a2_rows(grid, main)
    return rows if phase == "L" else [r for r in rows if r.startswith(("sel:", "dyn:"))]


def sparsevlm_rows(bench, rows=None):
    return [SV.row_tag(T, o) for T, o in SV.rows_for(bench, list(rows or (SV.ANCHORS + SV.ADAPTED)))]


def row_meta(c, main):
    inv = {u: (o, s) for o in RUN.OBJECTIVES for s, u in enumerate(main[o])}; p = c.split(":")
    if c == "full":
        return {"family": "full", "tokens": RUN.N_VIS}
    if p[0] == "static":
        return {"family": "uniform_grid", "tokens": int(p[2])}
    if p[0] in ("sel", "dyn"):
        o, s = inv[p[1]]; base = {"objective": o, "name": SC.NAMES[o], "seed": s, "unit": p[1]}
        return dict(base, family=SC.NAMES[o], tokens=int(p[2])) if p[0] == "sel" else dict(base, family=SC.NAMES[o] + "_dyn", operating_point=float(p[2]))
    if p[0] == "sparsevlm":
        return {"family": "SparseVLM", "tokens": int(p[1]), "label": p[2]}
    raise ValueError(c)


def read_rows(p):
    return [c.strip() for c in Path(p).read_text().splitlines() if c.strip() and not c.startswith("#")]


def unit_record(selectors_dir, main):
    """sha256 of every selector unit (.pt / .json), its CAL-selected step and dynamic taus; refuses an incomplete unit."""
    out = {}
    for o in RUN.OBJECTIVES:
        for s, u in enumerate(main[o]):
            pt, js = Path(selectors_dir) / f"{u}.pt", Path(selectors_dir) / f"{u}.json"
            if not (pt.is_file() and js.is_file()):
                raise SystemExit(f"selector unit {u} incomplete in {selectors_dir}")
            lg = json.loads(js.read_text())
            if "dynamic" not in lg:
                raise SystemExit(f"{u}: no CAL dynamic taus in {js}")
            out[u] = {"objective": o, "name": SC.NAMES[o], "seed": s, "pt_sha256": sha(pt), "json_sha256": sha(js), "selected_step": lg.get("selected_step"),
                      "selected_cal_spearman": lg.get("selected_cal_spearman"), "dynamic_tau": {f: v.get("tau") for f, v in lg["dynamic"].items()}}
    return out


def configs(root, phase="L", lr_choice=LR_CHOICE, selectors_dir=SELECTORS, pca=PCA, d30_configs=D30_CONFIGS):
    """The phase's rows: the frozen D1-30 planner's grid on the phase's lr_choice.json, filtered to §A.2 (phase F: the selector rows; its full /
    grid / SparseVLM baseline rows are phase L's).  Records selectors_dir / pca / unit hashes in configs_<P>/rows.json (the runner reads them)."""
    if phase == "Q":
        raise SystemExit("phase Q (Qwen2.5-VL) needs its protocol addendum and the R7-40c runner; its rows are declared in the protocol only")
    L = Layout(root, phase); choice, main = main_units(lr_choice); n_pipe, n_sv = N_ROWS[phase]
    plan = L.configs / "d1_30_plan"; PL.write_configs(choice, plan)                  # the frozen D1-30 planner, verbatim
    rec = {"phase": phase, "created_utc": utc(), "lr_choice": {"file": str(lr_choice), "sha256": sha(lr_choice), "chosen_lr": {o: choice[o]["chosen_lr"] for o in RUN.OBJECTIVES}},
           "planner": "scripts/r7_d1_30_plan.py write_configs (frozen D1-30 grid), filtered to the R7-50 §A.2 rows", "benches": {},
           "selectors_dir": str(selectors_dir), "pca": {"file": str(pca), "sha256": sha(pca)}, "units": unit_record(selectors_dir, main)}
    for b in BENCHES:
        rows = phase_rows(read_rows(plan / f"configs_{D30_BENCH[b]}.txt"), main, phase); sv = sparsevlm_rows(b) if phase == "L" else []
        if len(rows) != n_pipe or len(sv) != n_sv:
            raise SystemExit(f"{b}: {len(rows)} pipeline + {len(sv)} SparseVLM rows, expected {n_pipe} + {n_sv} (phase {phase})")
        live = Path(d30_configs) / f"configs_{D30_BENCH[b]}.txt" if (d30_configs and phase == "L") else None; live_rec = None
        if live is not None and live.exists():                                          # the D1-30 driver's own file (written after D1-20)
            miss = [r for r in rows if r not in set(read_rows(live))]
            if miss:
                raise SystemExit(f"{b}: rows absent from the D1-30 driver's {live}: {miss[:5]}")
            live_rec = {"file": str(live), "sha256": sha(live)}
        write_atomic(L.configs_file(b), "\n".join(rows) + "\n"); write_atomic(L.sv_rows_file(b), "".join(r + "\n" for r in sv))
        rec["benches"][b] = {"d1_30_counterpart": D30_BENCH[b], "pipeline_rows": len(rows), "sparsevlm_rows": len(sv), "d1_30_driver_configs": live_rec,
                             "configs_sha256": sha(L.configs_file(b)), "sparsevlm_rows_sha256": sha(L.sv_rows_file(b)),
                             "rows": {c: row_meta(c, main) for c in rows + sv},
                             "baseline_rows_from_phase_L": (list(BASELINE_FROM_L) + sparsevlm_rows(b)) if phase == "F" else []}
    write_atomic(L.configs / "rows.json", json.dumps(rec, indent=1))
    return {b: {"pipeline_rows": v["pipeline_rows"], "sparsevlm_rows": v["sparsevlm_rows"]} for b, v in rec["benches"].items()}


def lines(root, phase="L"):
    """GPU unit lines of a phase: quality shards (GQA 6, ChartQA 2) and, phase L only, SparseVLM shards (GQA 3, ChartQA 1); GQA-only and
    ChartQA-only units with D1-30's partition rules; each line carries its completion marker."""
    if phase == "Q":
        raise SystemExit("phase Q has no lines here (R7-40c runner)")
    L = Layout(root, phase); out = []
    for b in BENCHES:
        n = QUALITY_SHARDS[b]
        for i in range(n):
            out.append((f"sbatch --parsable --job-name=r7_50{phase.lower()}q_{b}_{i} {PL.QUALITY_GPU[D30_BENCH[b]]} --time={UNIT_TIME} slurm/r7_50_gpu.sbatch run {b} {i}/{n} {L.root} {phase}",
                        L.pred / b / f"done_shard{i}of{n}.json"))
        if phase == "L":
            n = SV_SHARDS[b]
            for i in range(n):
                out.append((f"sbatch --parsable --job-name=r7_50lv_{b}_{i} {PL.GPU} --time={UNIT_TIME} slurm/r7_50_gpu.sbatch sparsevlm {b} {i}/{n} {L.root}",
                            L.pred / b / f"sparsevlm_run_{b}_shard{i}of{n}.json"))
    return out


def write_lines(root, phase="L", path=None):
    L = Layout(root, phase); path = Path(path or L.lines); units = lines(root, phase)
    write_atomic(path, f"# R7-50 phase {phase} generation units: <sbatch line> #MARK <completion marker>\n" + "".join(f"{l} #MARK {m}\n" for l, m in units))
    return len(units)


# ------------------------------------------------------------------------------------------------------------------------------ generation
def guard_generation(L: Layout, benches, configs_file=None):
    load_summary(L)
    if L.manifest.exists():
        raise SystemExit(f"{L.manifest} exists: phase {L.phase} predictions are sealed, no generation after hashing")
    for b in benches:
        cf = Path(configs_file) if configs_file else L.configs_file(b)
        if not cf.is_file():
            raise SystemExit(f"{cf} missing (run 'configs')")


def extend_max_new():
    """The test benchmarks inherit the frozen D1-30 generation length of their counterparts (r7_d1_30_run.MAX_NEW: GQA 128, ChartQA 16)."""
    for b, src in D30_BENCH.items():
        RUN.MAX_NEW.setdefault(b, RUN.MAX_NEW[src])


def run_quality(root, bench, shard, phase="L", limit=0, selectors_dir=None, pca=None, host=None, bank=None, configs_file=None):
    """The D1-30 quality runner on the sealed inputs, into <root>/sealed/pred_<P>/<bench>.  Selectors / PCA default to the ones recorded (and
    hashed) in configs_<P>/rows.json, so a phase cannot silently load other checkpoints."""
    L = Layout(root, phase); guard_generation(L, [bench], configs_file); extend_max_new()
    rec = json.loads((L.configs / "rows.json").read_text()) if (L.configs / "rows.json").is_file() else {}
    selectors_dir = selectors_dir or rec.get("selectors_dir") or SELECTORS; pca = pca or (rec.get("pca") or {}).get("file") or PCA
    return RUN.run(Namespace(sets_dir=str(L.inputs), bench=bench, configs_file=str(configs_file or L.configs_file(bench)), shard=shard, out=str(L.pred),
                             chunk=32, batch=32, selectors_dir=str(selectors_dir), pca=str(pca), limit=limit), host=host, bank=bank)


def run_sparsevlm(root, benches, shard, limit=0, rows=None, phase="L"):
    if phase != "L":
        raise SystemExit("SparseVLM rows are generated in phase L only")
    L = Layout(root, phase); guard_generation(L, benches); extend_max_new()
    for b in benches:
        if sparsevlm_rows(b, rows) != read_rows(L.sv_rows_file(b)):
            raise SystemExit(f"{b}: requested SparseVLM rows differ from {L.sv_rows_file(b)}")
    argv = ["--sets-dir", str(L.inputs), "--bench", *benches, "--shard", shard, "--out", str(L.pred)]
    argv += (["--limit", str(limit)] if limit else []) + (["--rows", *[str(r) for r in rows]] if rows else [])
    return SV.main(argv)


# ------------------------------------------------------------------------------------------------------------------------------ integrity gate
def gpu_class(name):
    if name and "L40S" in name:
        return "L40S"
    if name and "H100" in name:
        return "H100"
    raise SystemExit(f"unexpected or missing GPU class {name!r} in a reference marker")


def integrity_groups(phase, main, groups=None):
    """Rows of the full-scale gate.  Phase L: A = full + static:grid:144 (D1-30's early shard 0/1 run), B = S1 seed 0 at 25 % (D1-30's stage-1
    quality shards), SV = sparsevlm:192:official.  Phase F: B only, with the D1-FULL S1 seed-0 unit."""
    g = {"A": ["full", "static:grid:144"], "B": [f"sel:{main['vcs'][0]}:144"], "SV": ["sparsevlm:192:official"]} if phase == "L" else {"B": [f"sel:{main['vcs'][0]}:144"]}
    return {k: v for k, v in g.items() if groups is None or k in groups}


def integrity_prepare(root, phase="L", lr_choice=LR_CHOICE, selectors_dir=None, pca=None, ref_pred=D30_PRED, ref_sets=D30_SETS, ref_log=D30_LOG,
                      groups=None, chartqa_src=CHARTQA_VAL, images_dir=GQA_IMAGES):
    """Gate step 1 (CPU): the R7-50 build path (development mode) on the FULL GQA DEV set (D1-30's items re-expressed in the GQA balanced format,
    answers withheld) and the FULL ChartQA val parquet with D1-30's ids (chartqa_val_<row>); row groups with D1-30's sharding; GPU-pinned unit
    lines (each reference shard's GPU class from its completion marker; SparseVLM: L40S, where every D1-30 SparseVLM shard ran)."""
    L = Layout(root, phase); base = L.integrity; src = base / "sources"; src.mkdir(parents=True, exist_ok=True); _, main = main_units(lr_choice)
    rec = json.loads((L.configs / "rows.json").read_text()) if (L.configs / "rows.json").is_file() else {}
    selectors_dir = str(selectors_dir or rec.get("selectors_dir") or SELECTORS); pca = str(pca or (rec.get("pca") or {}).get("file") or PCA)
    IB = Layout(base, "L"); G = integrity_groups(phase, main, groups)
    if not IB.summary.exists():
        dev = read_jsonl_strict(Path(ref_sets) / "gqa_dev.jsonl"); gq = src / "gqa_dev_d1_30_questions.json"
        gq.write_text(json.dumps({it["qid"]: {"imageId": it["image_id"], "question": it["question"], "answer": "withheld", "types": {}} for it in dev}))
        build(base, gq, chartqa_src, images_dir=images_dir, chartqa_qid_prefix="chartqa_val")
    for b in BENCHES:
        for g in ("A", "B"):
            if g in G:
                write_atomic(IB.configs / f"group_{g}_{b}.txt", "\n".join(G[g]) + "\n")
        write_atomic(IB.configs_file(b), "".join(r + "\n" for g in ("A", "B") for r in G.get(g, [])))   # the guard's file; runs use the group files
        write_atomic(IB.sv_rows_file(b), "".join(r + "\n" for r in G.get("SV", [])))
    retried = [ln.split(" without ")[-1].strip() for ln in (Path(ref_log).read_text().splitlines() if Path(ref_log).is_file() else []) if "RETRY" in ln]
    units = {}
    for b in BENCHES:
        rd = Path(ref_pred) / D30_BENCH[b]
        if "A" in G:
            m = rd / "done_shard0of1.json"; c = gpu_class(json.loads(m.read_text())["env"].get("gpu"))
            units.setdefault(f"pipe_{c}", {"class": c, "tasks": []})["tasks"].append({"bench": b, "group": "A", "shard": "0/1", "reference_marker": str(m)})
        if "B" in G:
            n = PL.SHARDS[D30_BENCH[b]]
            for i in range(n):
                m = rd / f"done_shard{i}of{n}.json"
                if not m.is_file():
                    raise SystemExit(f"reference shard marker missing: {m} (the gate needs the reference's full-scale outputs)")
                c = gpu_class(json.loads(m.read_text())["env"].get("gpu"))
                units.setdefault(f"pipe_{c}", {"class": c, "tasks": []})["tasks"].append({"bench": b, "group": "B", "shard": f"{i}/{n}", "reference_marker": str(m),
                                                                                         "reference_retried": str(m) in retried})
        if "SV" in G:
            units.setdefault(f"sv_{SV_DEFAULT_CLASS}", {"class": SV_DEFAULT_CLASS, "tasks": []})["tasks"].append({"bench": b, "group": "SV", "shard": "0/1"})
    plan = {"phase": phase, "created_utc": utc(), "groups": G, "units": units, "selectors_dir": selectors_dir, "pca": pca, "lr_choice": str(lr_choice),
            "reference_pred": str(ref_pred), "reference_sets": str(ref_sets), "retried_reference_markers": retried,
            "rule": GATE_RULE}
    write_atomic(base / "plan.json", json.dumps(plan, indent=1))
    write_atomic(base / "lines.txt", f"# R7-50 phase {phase} full-scale integrity units: <sbatch line> #MARK <marker>\n" + "".join(
        f"sbatch --parsable --job-name=r7_50i{phase.lower()}_{name} {GPU_PIN[u['class']]} --time={INTEGRITY_TIME} slurm/r7_50_gpu.sbatch integrity {L.root} {phase} {name} "
        f"#MARK {base / f'done_{name}.json'}\n" for name, u in sorted(units.items())))
    return {name: len(u["tasks"]) for name, u in units.items()}


def integrity_run(root, phase, unit):
    """Gate step 2 (GPU): one unit = the tasks pinned to one GPU class, one host load; the reference's sharding / chunking is reproduced."""
    L = Layout(root, phase); base = L.integrity; plan = json.loads((base / "plan.json").read_text()); u = plan["units"][unit]; IB = Layout(base, "L"); t0 = time.time()
    import torch
    got = gpu_class(torch.cuda.get_device_name(0))
    if got != u["class"]:
        raise SystemExit(f"unit {unit} is pinned to {u['class']}, running on {got}")
    if unit.startswith("sv_"):
        run_sparsevlm(base, sorted({t["bench"] for t in u["tasks"]}), "0/1", rows=[192])
    else:
        import r7_d1_teacher as TE
        host = TE.LlavaPrefill(IB.pred)
        for t in u["tasks"]:
            run_quality(base, t["bench"], t["shard"], "L", configs_file=IB.configs / f"group_{t['group']}_{t['bench']}.txt", selectors_dir=plan["selectors_dir"],
                        pca=plan["pca"], host=host)
    write_atomic(base / f"done_{unit}.json", json.dumps({"tasks": u["tasks"], "class": got, "seconds": time.time() - t0, "utc": utc()}, indent=1))
    return 0


def integrity_compare(root, phase="L"):
    """Gate step 3 (CPU): inputs (question, prompt, image) and every answer / kept count of every gate row vs the reference, item by item.
    Prediction strings are compared for equality only: no label is opened, nothing is scored, no prediction is printed."""
    L = Layout(root, phase); base = L.integrity; plan = json.loads((base / "plan.json").read_text()); IB = Layout(base, "L")
    res = {"phase": phase, "compared_utc": utc(), "rule": GATE_RULE, "plan_rule_text": plan.get("rule"), "sv_agreement_min": SV_AGREEMENT_MIN,
           "retried_reference_markers": plan["retried_reference_markers"], "benches": {}}; ok = True
    for b in BENCHES:
        d30b = D30_BENCH[b]; mine = {it["qid"]: it for it in read_jsonl_strict(IB.inputs_file(b))}
        ref = {it["qid"]: it for it in read_jsonl_strict(Path(plan["reference_sets"]) / f"{d30b}.jsonl")}
        r = {"items": len(ref), "same_qid_set": set(mine) == set(ref)}
        same = {"question": 0, "prompt": 0, "image": 0}
        for q, o in ref.items():
            it = mine.get(q)
            if it is None:
                continue
            same["question"] += it["question"] == o["question"]; same["prompt"] += it["prompt"] == o["prompt"]
            same["image"] += it["image_id"] == o["image_id"] and (it["image_file"] == o["image_file"] if b.startswith("gqa") else sha(it["image_file"]) == sha(o["image_file"]))
        r["inputs_identical"] = same; ok &= r["same_qid_set"] and all(v == len(ref) for v in same.values())
        for g, rows in plan["groups"].items():
            for cfg in rows:
                got = {x["qid"]: x for p in RUN.cfg_paths(IB.pred / b, cfg) for x in V.read_jsonl(p)}
                old = {x["qid"]: x for p in RUN.cfg_paths(Path(plan["reference_pred"]) / d30b, cfg) for x in V.read_jsonl(p)}
                diff = [q for q in ref if q not in got or q not in old or got[q]["pred"] != old[q]["pred"] or got[q].get("kept") != old[q].get("kept")]
                cover = all(q in got and q in old for q in ref); kept_same = all(got[q].get("kept") == old[q].get("kept") for q in ref if q in got and q in old)
                x = {"group": g, "items": len(ref), "answered_here": sum(q in got for q in ref), "answered_reference": sum(q in old for q in ref),
                     "identical": len(ref) - len(diff), "agreement": (len(ref) - len(diff)) / len(ref) if ref else None, "kept_identical": kept_same,
                     "mismatched_qids_first_20": diff[:20]}
                if cfg.startswith("sparsevlm:"):                                        # >= 98.0 % answers, kept identical, E3 counters in tolerance
                    cnt = [row for f in sorted((IB.pred / b).glob(f"sparsevlm_run_{b}_shard*.json")) for row in json.loads(f.read_text()).get("rows", [])
                           if row.get("config") == cfg]
                    x["counter_records"] = [{k: row.get(k) for k in ("official_forwards", "predicted_equal_tokens", "equal_visual_tokens", "counter_deviation",
                                                                     "counter_check")} for row in cnt]
                    x["rule"] = f"agreement >= {SV_AGREEMENT_MIN:.3f}, kept identical, E3 counter within tolerance"
                    x["pass"] = bool(cover and kept_same and x["agreement"] is not None and x["agreement"] >= SV_AGREEMENT_MIN and cnt
                                     and all(row.get("counter_check") is True for row in cnt))
                else:
                    x["rule"] = "identical answer and kept count on every item"; x["pass"] = not diff
                r[cfg] = x; ok &= x["pass"]
        res["benches"][b] = r
    res["pass"] = bool(ok); write_atomic(base / "integrity.json", json.dumps(res, indent=1))
    print(json.dumps({b: {c: (v["identical"], v["items"]) for c, v in r.items() if isinstance(v, dict) and "identical" in v} for b, r in res["benches"].items()}))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------------------ manifest
def timing_files(timing_dir, tbench):
    return {name: Path(timing_dir) / tbench / name for name in (f"timing_{tbench}.json", f"sparsevlm_timing_{tbench}.json")}


def manifest(root, phase="L", timing_dir=TIMING_DIR, timing_map=None, extra_timing_dir=None):
    """Coverage and integrity of every row generated in the phase (prediction files read for question ids only; labels never opened), sha256 of
    every prediction file of the phase and of the profiling-workload timing files the score will use.  Refuses partial coverage."""
    L = Layout(root, phase); tmap = dict(D30_BENCH, **(timing_map or {})); summ = load_summary(L)
    if L.manifest.exists():
        raise SystemExit(f"{L.manifest} exists: phase {phase} predictions already sealed")
    if phase == "F" and not extra_timing_dir:
        raise SystemExit("phase F needs D1-FULL's own timing run (D1-30 procedure on nodesumo01; coordinator decision F3): --extra-timing-dir")
    out = {"phase": phase, "created_utc": utc(), "root": str(L.root), "git": git_state(), "build_summary_sha256": sha(L.summary), "inputs": {},
           "labels_not_opened": {}, "rows": {}, "coverage": {}, "sparsevlm_markers": {}, "profiling_timing": {}, "rows_json_sha256": sha(L.configs / "rows.json"),
           "integrity_gate": json.loads((L.integrity / "integrity.json").read_text()).get("pass") if (L.integrity / "integrity.json").is_file() else None}
    for b in BENCHES:
        bs = summ["benches"][b]
        if sha(L.inputs_file(b)) != bs["inputs_sha256"]:
            raise SystemExit(f"{L.inputs_file(b)} changed since the build")
        qids = [it["qid"] for it in read_jsonl_strict(L.inputs_file(b))]; exp = set(qids)
        cfg, sv = read_rows(L.configs_file(b)), read_rows(L.sv_rows_file(b)); cov = {}
        for c in cfg + sv:
            recs, n = SC.load_config(L.pred / b, c); have = set(recs)
            if have != exp:
                raise SystemExit(f"coverage: {b} {c}: {len(have & exp)} / {len(exp)} items, {len(have - exp)} unexpected; integrity {n}")
            cov[c] = {"answers": len(recs), **n}
        out["coverage"][b] = cov
        out["sparsevlm_markers"][b] = {f.name: {"rows": [{k: r.get(k) for k in ("config", "official_forwards", "predicted_equal_tokens", "equal_visual_tokens",
                                                                               "counter_deviation", "counter_check", "deviation_is_float32_accumulation")}
                                                         for r in json.loads(f.read_text()).get("rows", [])],
                                                "rows_without_counter_record": json.loads(f.read_text()).get("rows_without_counter_record", [])}
                                       for f in sorted((L.pred / b).glob(f"sparsevlm_run_{b}_shard*.json"))}
        out["inputs"][b] = {"file": str(L.inputs_file(b)), "sha256": bs["inputs_sha256"], "items": len(qids)}
        out["labels_not_opened"][b] = {"file": str(L.labels_file(b)), "sha256_recorded_at_build": bs["labels_sha256"]}
        out["rows"][b] = {"configs_sha256": sha(L.configs_file(b)), "sparsevlm_rows_sha256": sha(L.sv_rows_file(b)), "n_rows": len(cfg) + len(sv)}
        tf = timing_files(timing_dir, tmap[b]); missing = [str(p) for p in tf.values() if not p.is_file()]
        if missing:
            raise SystemExit(f"profiling-workload timing files missing: {missing}")
        out["profiling_timing"][b] = {"d1_30_bench": tmap[b], "files": {k: {"path": str(p), "sha256": sha(p)} for k, p in tf.items()}}
        if extra_timing_dir:                                                            # phase F: D1-FULL's timing run (its dynamic rows)
            ef = Path(extra_timing_dir) / tmap[b] / f"timing_{tmap[b]}.json"
            if not ef.is_file():
                raise SystemExit(f"extra (D1-FULL) timing file missing: {ef}")
            out["profiling_timing"][b]["extra_files"] = {ef.name: {"path": str(ef), "sha256": sha(ef)}}
    files = walk_files(L.pred)
    out["files"] = [{"path": str(p.relative_to(L.root)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]; out["n_files"] = len(files)
    write_atomic(L.manifest, json.dumps(out, indent=1, sort_keys=True) + "\n"); digest = sha(L.manifest)
    write_atomic(L.manifest_sha, f"{digest}  predictions_manifest.json\n")
    return digest, out


# ------------------------------------------------------------------------------------------------------------------------------ score
def committed_copy(kv, digest):
    """True when ALLOW_SCORING's commit=<git sha> holds manifest_copy=<repo path> with exactly the manifest's bytes (git show, sha256)."""
    c, path = kv.get("commit"), kv.get("manifest_copy")
    if not c or not path:
        return False
    try:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{path}"], capture_output=True, timeout=120)
    except Exception:
        return False
    return r.returncode == 0 and hashlib.sha256(r.stdout).hexdigest() == digest


def verify_manifest(L: Layout):
    """Every prediction file of the phase still matches its manifest (none added, removed or changed)."""
    if not L.manifest.is_file():
        raise SystemExit(f"{L.manifest} missing (phase {L.phase} predictions not sealed)")
    man = json.loads(L.manifest.read_text()); listed = {f["path"]: f["sha256"] for f in man["files"]}
    now = {str(p.relative_to(L.root)): p for p in walk_files(L.pred)}
    if set(now) != set(listed):
        raise SystemExit(f"phase {L.phase} prediction files changed since the manifest: {sorted(set(now) - set(listed))[:3]} new, "
                         f"{sorted(set(listed) - set(now))[:3]} gone")
    bad = [r for r, p in now.items() if sha(p) != listed[r]]
    if bad:
        raise SystemExit(f"{len(bad)} phase {L.phase} prediction files differ from the manifest (first {bad[:3]}); labels stay sealed")
    return man


def check_gate(L: Layout, require_commit=None):
    """ALLOW_SCORING_<P> (created by the coordinator after committing the hashes) must carry the manifest's sha256 — for the production root also
    commit=<git sha> and manifest_copy=<repo path> whose committed bytes hash to it — and every prediction file must still match the manifest."""
    if not L.allow.is_file():
        raise SystemExit(f"{L.allow} absent: the coordinator creates it only after committing the prediction hashes; labels stay sealed")
    if not L.manifest.is_file():
        raise SystemExit(f"{L.manifest} missing")
    kv = {}
    for line in L.allow.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1); kv[k.strip()] = v.strip()
    have = sha(L.manifest)
    if kv.get("manifest_sha256") != have:
        raise SystemExit(f"{L.allow} names manifest_sha256={kv.get('manifest_sha256')!r}, the manifest's sha256 is {have}; labels stay sealed")
    if require_commit is None:
        require_commit = L.root.resolve() == ROOT.resolve()
    if require_commit and not committed_copy(kv, have):
        raise SystemExit(f"{L.allow}: commit={kv.get('commit')!r} / manifest_copy={kv.get('manifest_copy')!r} do not hold the manifest in git "
                         f"(git show <commit>:<path> must hash to {have}); labels stay sealed")
    return verify_manifest(L), kv, have


def labelled_items(L: Layout, bench, summ):
    """Sealed inputs joined with the sealed labels (both verified against the build's sha256)."""
    bs = summ["benches"][bench]
    if sha(L.inputs_file(bench)) != bs["inputs_sha256"] or sha(L.labels_file(bench)) != bs["labels_sha256"]:
        raise SystemExit(f"{bench}: inputs or labels changed since the build")
    items = read_jsonl_strict(L.inputs_file(bench)); lab = {r["qid"]: r for r in read_jsonl_strict(L.labels_file(bench))}
    if set(lab) != {it["qid"] for it in items} or len(lab) != len(items):
        raise SystemExit(f"{bench}: label and input question ids differ")
    return [dict(it, **{k: v for k, v in lab[it["qid"]].items() if k != "qid"}) for it in items]


class TestBench(SC.Bench):
    """r7_d1_30_score.Bench with the items (sealed inputs + labels) in memory, each row read from its phase's prediction dir (sources: cfg -> dir),
    and the latencies of the mapped D1-30 benchmark's frozen timing files (profiling workload; optional extra timing files, e.g. a D1-FULL timing run,
    for rows the D1-30 files lack).  Scores, coverage guard, bootstrap means, latency lookup (selector rows -> the representative 10k S1 seed-0
    unit: same architecture and cost), curves and interpolation are the frozen D1-30 code."""

    def __init__(self, bench, items, sources, timing_dir, timing_bench, reps, rep_unit, extra_timing=()):
        self.bench, self.items, self.rep = bench, items, rep_unit
        self.qids = [it["qid"] for it in items]; self.boot = SC.Boot([it["image_id"] for it in items], reps)
        self.S, self.K, self._bm, self.integrity, self.source = {}, {}, {}, {}, {}
        for cfg, pred_dir in sources.items():
            recs, self.integrity[cfg] = SC.load_config(Path(pred_dir) / bench, cfg); preds = {q: r["pred"] for q, r in recs.items()}
            if set(preds) != set(self.qids):
                raise SystemExit(f"coverage guard: {bench} {cfg}: {len(set(preds) & set(self.qids))} / {len(self.qids)} items, "
                                 f"{len(set(preds) - set(self.qids))} unexpected; integrity {self.integrity[cfg]}")
            sc = SC.item_scores(bench, items, preds); kept = {q: r.get("kept", RUN.N_VIS) for q, r in recs.items()}
            self.S[cfg] = np.array([sc[q] for q in self.qids], float); self.K[cfg] = np.array([kept[q] for q in self.qids], float); self.source[cfg] = str(pred_dir)
        self.T, self.Tci, self.Trow, self.timing_src = {}, {}, {}, {}
        for name, p in list(timing_files(timing_dir, timing_bench).items()) + [(Path(x).name, Path(x)) for x in extra_timing]:
            j = json.loads(Path(p).read_text()); self.timing_src[name] = {"path": str(p), "sha256": sha(p), "gpu": j.get("gpu"), "host": j.get("host")}
            for r in j["rows"]:
                k = (r["config"], r["B"])
                if k not in self.T:                                                      # the frozen D1-30 files take precedence
                    self.T[k] = r["per_item_ms_median"]; self.Tci[k] = r.get("per_item_ms_median_ci95"); self.Trow[k] = r

    def subset(self, mask, reps):
        """The same rows restricted to a subset of items, with its own group bootstrap (secondary subset contrasts)."""
        sb = object.__new__(TestBench); m = np.asarray(mask, bool)
        sb.bench, sb.rep, sb.items = self.bench, self.rep, [it for it, k in zip(self.items, m) if k]
        sb.qids = [it["qid"] for it in sb.items]; sb.boot = SC.Boot([it["image_id"] for it in sb.items], reps)
        sb.S = {c: x[m] for c, x in self.S.items()}; sb.K = {c: x[m] for c, x in self.K.items()}; sb._bm = {}; sb.integrity = self.integrity
        sb.T, sb.Tci, sb.Trow, sb.timing_src, sb.source = self.T, self.Tci, self.Trow, self.timing_src, self.source
        return sb


def families(main):
    F = {SC.NAMES[o]: [[f"sel:{u}:{k}" for k in SC.BUDGETS] for u in main[o]] for o in RUN.OBJECTIVES}
    F["S1_dyn"] = [[f"dyn:{u}:{f}" for f in RUN.FRACTIONS] for u in main["vcs"]]
    F["grid"] = [[f"static:grid:{k}" for k in SC.BUDGETS]]
    return F


def lat_key(bn, c):
    return f"sel:{bn.rep}:{c.split(':')[2]}" if c.startswith("sel:") else c


def required_latencies(bn, main, sv_rows, need_dyn=True):
    """Every latency the endpoints use must exist in the frozen timing files (else STOP rather than 'not reached' by omission).  need_dyn=False
    only for phase F without a D1-FULL timing file: its S1 dynamic rows then carry no latency (reported as such)."""
    need = [("full", SC.PRIMARY_B), ("full", 1)] + [(f"static:grid:{k}", B) for k in SC.BUDGETS for B in (SC.PRIMARY_B, 1)]
    need += [(f"sel:{bn.rep}:{k}", B) for k in SC.BUDGETS for B in (SC.PRIMARY_B, 1)] + [(c, 1) for c in sv_rows]
    need += [(f"dyn:{u}:{f}", SC.PRIMARY_B) for u in main["vcs"] for f in RUN.FRACTIONS] if need_dyn else []
    missing = [k for k in need if k not in bn.T]
    if missing:
        raise SystemExit(f"{bn.bench}: profiling-workload latencies missing for {missing[:6]} ({len(missing)} in total)")


def contrasts(bn, main, sv_rows, F):
    """§A.3 contrasts at the D1-30 latency points (B* = 4): S1 − S2, S1 − S3 (primary), S1 − SparseVLM budget-matched (primary, D30-1);
    S1 − uniform grid and S1 − SparseVLM latency-matched at B = 1 (secondary).  Formulas of r7_d1_30_score.score_bench."""
    C = {}
    for L, k in SC.LPOINTS.items():
        Lms = bn.lat(f"static:grid:{k}", SC.PRIMARY_B); s1 = None if Lms is None else SC.seed_mean_at(bn, F["S1"], Lms, SC.PRIMARY_B)
        for other, role in (("S2", "primary"), ("S3", "primary"), ("grid", "secondary")):
            o = None if Lms is None else SC.seed_mean_at(bn, F[other], Lms, SC.PRIMARY_B); key = f"S1-{other}@{L}"
            if s1 is None or o is None:
                C[key] = {"not_reached": True, "role": role}; continue
            seed_d = [a - b for a, b in zip(s1[2], o[2])] if len(o[2]) == len(s1[2]) else None
            C[key] = dict(SC.contrast(s1[0] - o[0], s1[1] - o[1], seed_d), per_seed=SC.per_seed(s1[2], s1[3], o[2], o[3]), role=role,
                          matched=f"{L} = uniform-grid {k}-token per-item latency at B = {SC.PRIMARY_B} on the profiling workload ({Lms:.2f} ms)")
        sv = f"sparsevlm:{k}:adapted"
        if sv in sv_rows:
            s1k = [f"sel:{u}:{k}" for u in main["vcs"]]; d = np.mean([bn.S[c] for c in s1k], 0) - bn.S[sv]
            C[f"S1-SparseVLM@{L}[budget]"] = dict(SC.contrast(d.mean(), bn.boot.means(d), [bn.q(c) - bn.q(sv) for c in s1k]),
                                                  per_seed=SC.per_seed([bn.q(c) for c in s1k], [bn.bm(c) for c in s1k], [bn.q(sv)], [bn.bm(sv)]),
                                                  matched=f"equal-token budget {k} (SparseVLM row {sv})", role="primary")
            L1 = bn.lat(f"static:grid:{k}", 1); s1b1 = None if L1 is None else SC.seed_mean_at(bn, F["S1"], L1, 1)
            cv = bn.curve([c for c in sv_rows if not c.endswith(":ocr")], 1); svq = None if L1 is None else bn.at(cv, L1); key = f"S1-SparseVLM@{L}[latency_b1]"
            if s1b1 is None or svq is None:
                C[key] = {"not_reached": True, "role": "secondary"}
            else:
                svr = bn.at(cv, L1, rep=True)
                C[key] = dict(SC.contrast(s1b1[0] - svq, s1b1[1] - svr, [x - svq for x in s1b1[2]]), per_seed=SC.per_seed(s1b1[2], s1b1[3], [svq], [svr]),
                              role="secondary", matched=f"both interpolated at the uniform-grid {k}-token latency at B = 1 ({L1:.2f} ms)")
    return C


def a3_s1(bn, main):
    """Secondary (D1-30 A3, S1 only): dynamic − fixed at the dynamic row's realised mean kept tokens ON TEST (fixed curve interpolated in tokens)."""
    C = {}
    for f in RUN.FRACTIONS:
        pt, bs = [], []
        for u in main["vcs"]:
            dc = f"dyn:{u}:{f}"; m = float(bn.K[dc].mean()); ks = list(SC.BUDGETS) + [RUN.N_VIS]
            cfgs = [f"sel:{u}:{k}" for k in SC.BUDGETS] + ["full"]; i, j, w = SC.bracket(ks, m); a, b = cfgs[i], cfgs[j]
            pt.append(bn.q(dc) - ((1 - w) * bn.q(a) + w * bn.q(b))); bs.append(bn.bm(dc) - ((1 - w) * bn.bm(a) + w * bn.bm(b)))
        C[f"A3:S1_dyn-fixed@f{f}"] = dict(SC.contrast(np.mean(pt), np.mean(bs, 0), pt), role="secondary")
    return C


def score_bench(bn: TestBench, main, sv_rows, reps, with_subsets=True):
    F = families(main); R = {"n_items": len(bn.items), "n_groups": bn.boot.G}
    tot = {k: sum(v[k] for v in bn.integrity.values()) for k in ("dropped_lines", "blank_lines", "duplicate_qids")}
    R["integrity"] = {"totals": tot, "configs_with_issues": {c: v for c, v in bn.integrity.items() if v["dropped_lines"] or v["blank_lines"] or v["duplicate_qids"]},
                      "files_per_config_max": max(len(v["files"]) for v in bn.integrity.values()),
                      "rule": "first complete record per qid wins; unparseable / incomplete lines dropped and counted (D1-30 E1)"}
    Bs = sorted({B for (_, B) in bn.T}); full = bn.q("full")
    R["rows"] = {c: {"quality": 100 * bn.q(c), "ci95": [100 * x for x in SC.ci(bn.bm(c))], "retention_rel": bn.q(c) / full if full else None,
                     "retention_abs_drop": 100 * (full - bn.q(c)), "mean_visual_tokens": float(bn.K[c].mean()),
                     "latency_profiling_workload_ms": {f"B{B}": {"median": bn.T[(lat_key(bn, c), B)], "ci95": bn.Tci.get((lat_key(bn, c), B))}
                                                       for B in Bs if (lat_key(bn, c), B) in bn.T}} for c in bn.S}
    R["latency_points_ms"] = {str(B): {L: bn.lat(f"static:grid:{k}", B) for L, k in SC.LPOINTS.items()} for B in Bs}
    R["at_latency"] = {}
    for B in Bs:
        R["at_latency"][str(B)] = {}
        for name, seeds in F.items():
            R["at_latency"][str(B)][name] = {}
            for L, k in SC.LPOINTS.items():
                Lms = bn.lat(f"static:grid:{k}", B); r = None if Lms is None else SC.seed_mean_at(bn, seeds, Lms, B)
                R["at_latency"][str(B)][name][L] = None if r is None else {"mean": 100 * r[0], "ci95": [100 * x for x in SC.ci(r[1])], "seed_sd": 100 * float(np.std(r[2]))}
        if B == 1:
            cv = bn.curve([c for c in sv_rows if not c.endswith(":ocr")], 1); R["at_latency"]["1"]["SparseVLM"] = {}
            for L, k in SC.LPOINTS.items():
                L1 = bn.lat(f"static:grid:{k}", 1); v = None if L1 is None else bn.at(cv, L1)
                R["at_latency"]["1"]["SparseVLM"][L] = None if v is None else {"mean": 100 * v}
    R["contrasts"] = dict(contrasts(bn, main, sv_rows, F), **a3_s1(bn, main))
    target = full - 0.01; R["within_1pt_of_full_ms_B4"] = {}
    for name, seeds in F.items():
        vals = []
        for c in seeds:
            cv = bn.curve(c, SC.PRIMARY_B); vals.append(None if cv is None else SC.first_within([p[0] for p in cv], [bn.q(p[1]) for p in cv], target))
        R["within_1pt_of_full_ms_B4"][name] = vals
    wrong = bn.S["full"] < 0.5; right = ~wrong
    R["preserve_correct"] = {"threshold": "score < 0.5 = wrong", "full_wrong": int(wrong.sum())}
    for c, x in bn.S.items():
        if c != "full":
            R["preserve_correct"][c] = {"kept_error": float(np.mean(x[wrong] < 0.5)) if wrong.any() else None,
                                        "corrected": float(np.mean(x[wrong] >= 0.5)) if wrong.any() else None,
                                        "broke_correct": float(np.mean(x[right] < 0.5)) if right.any() else None}
    R["dynamic_rows_on_test"] = {}
    for c in (c for c in bn.S if c.startswith("dyn:")):
        K = bn.K[c]; prof = bn.Trow.get((c, SC.PRIMARY_B), {})
        R["dynamic_rows_on_test"][c] = {"operating_point": float(c.split(":")[2]), "kept_tokens_share_test": {str(k): float(np.mean(K == k)) for k in RUN.GRID},
                                        "mean_kept_tokens_test": float(K.mean()), "sd_kept_tokens_test": float(K.std()), "mean_kept_fraction_test": float(K.mean() / RUN.N_VIS),
                                        "kept_tokens_share_profiling_workload_B4": prof.get("budget_share"),
                                        "mean_kept_tokens_profiling_workload_B4": prof.get("mean_visual_tokens")}
    if with_subsets:
        R["subsets"] = subsets(bn, main, sv_rows, reps)
    return R


def subsets(bn: TestBench, main, sv_rows, reps):
    """GQA structural types (quality per row); ChartQA human / augmented separately (quality per row and the §A.3 contrasts on the subset with
    its own chart bootstrap — secondary, not in the Holm family) plus the mean of the two splits.  The combined benchmark is primary."""
    out = {}
    if bn.bench.startswith("gqa"):
        lab = np.array([str(it.get("structural")) for it in bn.items])
        for v in sorted(set(lab)):
            m = lab == v; out[f"structural:{v}"] = {"n": int(m.sum()), "quality": {c: 100 * float(x[m].mean()) for c, x in bn.S.items()}}
    elif bn.bench.startswith("chartqa"):
        lab = np.array([it.get("split") for it in bn.items])
        for v in ("human", "augmented"):
            m = lab == v
            if not m.any():
                continue
            sb = bn.subset(m, reps)
            out[v] = {"n": int(m.sum()), "n_charts": sb.boot.G, "quality": {c: 100 * float(x[m].mean()) for c, x in bn.S.items()},
                      "ci95": {c: [100 * y for y in SC.ci(sb.bm(c))] for c in bn.S},
                      "contrasts_secondary": dict(contrasts(sb, main, sv_rows, families(main)), **a3_s1(sb, main))}
            for k in out[v]["contrasts_secondary"].values():
                k["role"] = "secondary (subset)"
        if "human" in out and "augmented" in out:
            out["mean_of_splits"] = {c: 0.5 * (out["human"]["quality"][c] + out["augmented"]["quality"][c]) for c in bn.S}
    return out


def exposure_sentence(summ):
    """§A.3b measured sentences from the build's overlap counts (image ids and question pairs only; no label)."""
    out = []; g = (summ["benches"].get("gqa_testdev") or {}).get("exposure_check"); c = (summ["benches"].get("chartqa_test") or {}).get("exposure_check")
    if g:
        out.append(f"GQA test-dev: {g.get('n_prefixed_image_ids')} of {g['images']} images have 'n'-prefixed ids (not Visual Genome ids) and "
                   f"{g['numeric_image_ids']} numeric Visual Genome ids; {g['images_in_mix_any']} appear in LLaVA-1.5's instruction mix by image id "
                   f"({g['images_in_mix_gqa_paths']} under gqa/, {g['images_in_mix_vg_paths']} under vg/); {g['questions_seen_in_mix_gqa_turns']} of "
                   f"{g['questions']} (image, normalised question) pairs appear among its GQA turns.")
    if c:
        out.append(f"ChartQA test: {c['mix_entries_with_chart_image_paths']} entries of LLaVA-1.5's instruction mix have chart image paths.")
    return out


def score(root, phase="L", lr_choice=None, timing_dir=TIMING_DIR, timing_map=None, reps=2000, out_dir=None, require_commit=None, extra_timing_dir=None):
    """Score one phase once.  Phase L: its 58 rows.  Phase F: its 45 D1-FULL selector rows + phase L's full / grid / SparseVLM baseline rows (phase
    L's sealed predictions, re-verified against manifest_L); latency of every selector row = the 10k representative S1 seed-0 unit's (same
    architecture); D1-FULL dynamic rows take their latencies from D1-FULL's own timing run (extra_timing_dir, D1-30 layout; required, F3)."""
    if phase == "Q":
        raise SystemExit("phase Q (Qwen2.5-VL) is scored by its addendum's code")
    L, LL = Layout(root, phase), Layout(root, "L"); man, kv, man_sha = check_gate(L, require_commit); tmap = dict(D30_BENCH, **(timing_map or {}))
    summ = load_summary(L)
    if phase == "F" and not extra_timing_dir:
        raise SystemExit("phase F scoring needs D1-FULL's own timing run (--extra-timing-dir; coordinator decision F3)")
    extra = {}
    for b in BENCHES:                                                                   # latency inputs exactly as hashed in the manifest
        for name, p in timing_files(timing_dir, tmap[b]).items():
            if man["profiling_timing"][b]["files"].get(name, {}).get("sha256") != sha(p):
                raise SystemExit(f"{p} differs from the manifest")
        extra[b] = []
        if extra_timing_dir:
            ef = Path(extra_timing_dir) / tmap[b] / f"timing_{tmap[b]}.json"
            if (man["profiling_timing"][b].get("extra_files") or {}).get(ef.name, {}).get("sha256") != (sha(ef) if ef.is_file() else None):
                raise SystemExit(f"{ef} is not the extra timing file sealed in the manifest")
            extra[b] = [ef]
    rec = json.loads((L.configs / "rows.json").read_text()); lr_choice = lr_choice or rec["lr_choice"]["file"]; _, main = main_units(lr_choice)
    if phase == "F":
        verify_manifest(LL)                                                             # the shared baseline rows are phase L's sealed predictions
        rep = main_units(json.loads((LL.configs / "rows.json").read_text())["lr_choice"]["file"])[1]["vcs"][0]
    else:
        rep = main["vcs"][0]
    res = {"title": f"R7-50 official-test evaluation, phase {phase} ({'LLaVA-1.5-7B, D1-20 selectors' if phase == 'L' else 'LLaVA-1.5-7B, D1-FULL selectors'})",
           "phase": phase, "scored_utc": utc(), "root": str(L.root), "manifest_sha256": man_sha, "allow_scoring": kv, "git": git_state(),
           "freeze": summ.get("freeze"), "representative_latency_unit": rep, "reps": reps, "latency_label": LATENCY_LABEL, "exposure_statement": EXPOSURE,
           "exposure_statement_measured": exposure_sentence(summ),
           "benches": {}, "sources": {b: {k: summ["benches"][b].get(k) for k in ("mode", "source", "sha256", "recorded", "items", "groups", "split_counts",
                                                                            "exposure_check")} for b in BENCHES},
           "notes": ["primary: S1 - S2 and S1 - S3 at L25 / L50 / L75 (B* = 4) and S1 - SparseVLM budget-matched at 144 / 288 / 432 equal tokens (D30-1); "
                     "Holm across these primary contrasts of both benchmarks at each latency point, within this phase",
                     "no latency claim against SparseVLM at B* = 4 (official code batch-1 only); S1 - SparseVLM latency-matched at B = 1 is secondary",
                     "ChartQA human / augmented: quality and contrasts per split are secondary (unadjusted); the combined benchmark is primary",
                     "no equivalence / non-inferiority claim (no margin set in advance)"]}
    if phase == "F":
        res["notes"].append("phase F: full / uniform-grid / SparseVLM rows are phase L's predictions (sealed under manifest_L), not regenerated")
    for b in BENCHES:
        cfg, sv = read_rows(L.configs_file(b)), read_rows(LL.sv_rows_file(b))
        if len(cfg) != N_ROWS[phase][0] or cfg != phase_rows(cfg, main, phase) or sv != sparsevlm_rows(b):
            raise SystemExit(f"{b}: configuration files do not match the §A.2 rows of phase {phase} for {lr_choice}")
        sources = {c: L.pred for c in cfg}
        sources.update({c: (L.pred if phase == "L" else LL.pred) for c in ([] if phase == "L" else list(BASELINE_FROM_L)) + sv})
        bn = TestBench(b, labelled_items(L, b, summ), sources, timing_dir, tmap[b], reps, rep, extra[b])
        required_latencies(bn, main, sv, need_dyn=True)
        r = score_bench(bn, main, sv, reps); r["profiling_timing"] = {"d1_30_bench": tmap[b], "files": bn.timing_src}
        r["row_sources"] = {c: ("phase L" if (phase == "F" and src == LL.pred) else f"phase {phase}") for c, src in sources.items()}
        r["sparsevlm_counters"] = SC.sparsevlm_counters(LL.pred / b, b); res["benches"][b] = r
    for Lp in SC.LPOINTS:                                                               # Holm over the primary contrasts at each latency point
        keys = [(b, k) for b in res["benches"] for k, v in res["benches"][b]["contrasts"].items() if f"@{Lp}" in k and v.get("role") == "primary" and "p" in v]
        if keys:
            for (b, k), pv in zip(keys, SC.holm([res["benches"][b]["contrasts"][k]["p"] for b, k in keys])):
                res["benches"][b]["contrasts"][k]["p_holm"] = pv; res["benches"][b]["contrasts"][k]["holm_family_size"] = len(keys)
    od = Path(out_dir) if out_dir else L.results
    if (od / "r7_50_results.json").exists():
        raise SystemExit(f"{od / 'r7_50_results.json'} exists; a re-verification writes to a new --out directory (A.4 case b, recorded)")
    write_atomic(od / "r7_50_results.json", json.dumps(res, indent=1, default=float)); write_atomic(od / "r7_50_results.md", markdown(res))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ markdown
def _f(x, d=1):
    return "—" if x is None else f"{x:.{d}f}"


def _ci(c):
    return "—" if not c else f"[{c[0]:.1f}, {c[1]:.1f}]"


def markdown(res) -> str:
    L = [f"# {res['title']}", "", f"Scored {res['scored_utc']}; predictions manifest sha256 `{res['manifest_sha256']}` (committed before scoring); "
         f"freeze {json.dumps(res.get('freeze'))}.", "", f"**Latency.** {res['latency_label']}.", "", "**Exposure (graded).**"]
    L += [f"- {x}" for x in res["exposure_statement"]] + [f"- measured: {x}" for x in res.get("exposure_statement_measured", [])]
    L += [""] + [f"- {x}" for x in res["notes"]] + ["", "## Sources", ""]
    for b, s in res["sources"].items():
        L.append(f"- {b}: {s['mode']}; {s['items']} items, {s['groups']} groups; sha256 {s['sha256']}" + (f"; splits {s['split_counts']}" if s.get("split_counts") else ""))
        if s.get("exposure_check"):
            L.append(f"  - overlap check: {json.dumps({k: v for k, v in s['exposure_check'].items() if k not in ('mix_file', 'mix_sha256', 'rule')})}")
    L += ["", "## Primary contrasts (points; B* = 4 latency points from the profiling workload; Holm per latency point)", "",
          "| bench | contrast | Δ | 95 % CI | p | p_holm | seed Δ | basis |", "|---|---|---|---|---|---|---|---|"]
    for b, r in res["benches"].items():
        for k, v in r["contrasts"].items():
            if v.get("role") == "primary":
                if v.get("not_reached"):
                    L.append(f"| {b} | {k} | not reached | | | | | |"); continue
                sd = ", ".join("%+.1f" % x for x in v.get("seed_deltas") or [])
                L.append(f"| {b} | {k} | {v['delta']:+.2f} | {_ci(v['ci95'])} | {v['p']:.4f} | {_f(v.get('p_holm'), 4)} | {sd} | {v.get('matched', '')} |")
    for b, r in res["benches"].items():
        L += ["", f"## {b}: fixed rows (n = {r['n_items']}, {r['n_groups']} groups)", ""]
        sub = r.get("subsets", {}); ch = b.startswith("chartqa") and "human" in sub
        head = "| row | visual tokens | quality [95 % CI] |" + (" human | augmented |" if ch else "") + " retention (rel.) | latency B=4 / B=1, ms (profiling workload) |"
        L += [head, "|" + "---|" * (head.count("|") - 1)]
        for c, v in r["rows"].items():
            lat = v["latency_profiling_workload_ms"]; l4, l1 = lat.get("B4", {}).get("median"), lat.get("B1", {}).get("median")
            L.append(f"| {c} | {v['mean_visual_tokens']:.1f} | {v['quality']:.2f} {_ci(v['ci95'])} |" + (f" {sub['human']['quality'][c]:.2f} | {sub['augmented']['quality'][c]:.2f} |" if ch else "")
                     + f" {_f(v['retention_rel'], 3)} | {_f(l4)} / {_f(l1)} |")
        L += ["", "Quality at the latency points (seed mean; B = batch size of the profiling workload):", "", "| B | family | L25 | L50 | L75 |", "|---|---|---|---|---|"]
        for B, fam in r["at_latency"].items():
            for name, pts in fam.items():
                L.append(f"| {B} | {name} | " + " | ".join("not reached" if pts[p] is None else f"{pts[p]['mean']:.2f}" for p in ("L25", "L50", "L75")) + " |")
        L += ["", "S1 dynamic rows: kept tokens on test vs the profiling workload (B = 4):", "", "| row | mean kept (test) | share 72/144/288/432/576 (test) | mean kept (profiling) |", "|---|---|---|---|"]
        for c, v in r["dynamic_rows_on_test"].items():
            share = " / ".join("%.2f" % v["kept_tokens_share_test"][str(k)] for k in RUN.GRID)
            L.append(f"| {c} | {v['mean_kept_tokens_test']:.1f} | {share} | {_f(v['mean_kept_tokens_profiling_workload_B4'])} |")
        L += ["", "Secondary contrasts (unadjusted):", "", "| contrast | Δ | 95 % CI | p |", "|---|---|---|---|"]
        for k, v in r["contrasts"].items():
            if v.get("role") != "primary":
                L.append(f"| {k} | not reached | | |" if v.get("not_reached") else f"| {k} | {v['delta']:+.2f} | {_ci(v['ci95'])} | {v['p']:.4f} |")
        for split in ("human", "augmented"):
            if split in sub:
                L += ["", f"ChartQA {split} (n = {sub[split]['n']}; secondary):", "", "| contrast | Δ | 95 % CI | p |", "|---|---|---|---|"]
                for k, v in sub[split]["contrasts_secondary"].items():
                    L.append(f"| {k} | not reached | | |" if v.get("not_reached") else f"| {k} | {v['delta']:+.2f} | {_ci(v['ci95'])} | {v['p']:.4f} |")
        bad = [c for c in r["sparsevlm_counters"] if c.get("within_tolerance") is not True]
        L += ["", f"Integrity: {json.dumps(r['integrity']['totals'])}; files per row ≤ {r['integrity']['files_per_config_max']}; SparseVLM counter records "
              f"{len(r['sparsevlm_counters'])}, outside tolerance / unmeasured {len(bad)}."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------------------ cli
def tmap_arg(pairs):
    out = {}
    for p in pairs or []:
        b, _, t = p.partition("=")
        if b not in BENCHES or not t:
            raise SystemExit(f"--timing-bench expects <bench>=<d1-30 timing bench>, got {p!r}")
        out[b] = t
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        p = sub.add_parser(name); p.add_argument("--root", default=str(ROOT))
        for k, v in kw.items():
            p.add_argument(f"--{k.replace('_', '-')}", **v)
        return p
    ph = {"phase": {"default": "L", "choices": PHASES}}
    src = {"gqa": {"default": str(GQA_TEST)}, "chartqa": {"default": str(CHARTQA_TEST)}, "protocol": {"default": None}, "gqa_images": {"default": str(GQA_IMAGES)},
           "mix665k": {"default": str(MIX665K)}}
    sel = {"lr_choice": {"default": str(LR_CHOICE)}, "selectors_dir": {"default": str(SELECTORS)}, "pca": {"default": str(PCA)}}
    tim = {"timing_dir": {"default": str(TIMING_DIR)}, "timing_bench": {"nargs": "*", "default": []}}
    add("build", **src)
    add("units", lr_choice=dict(required=True))
    add("configs", d30_configs=dict(default=str(D30_CONFIGS)), **ph, **sel)
    add("lines", lines=dict(default=None), **ph)
    add("prepare", d30_configs=dict(default=str(D30_CONFIGS)), **src, **sel)
    add("run", bench=dict(required=True, choices=BENCHES), shard=dict(default="0/1"), limit=dict(type=int, default=0), **ph)
    add("sparsevlm", bench=dict(required=True, nargs="+", choices=BENCHES), shard=dict(default="0/1"), limit=dict(type=int, default=0), rows=dict(type=int, nargs="+"))
    add("integrity-prepare", lr_choice=dict(default=str(LR_CHOICE)), selectors_dir=dict(default=None), pca=dict(default=None), ref_pred=dict(default=str(D30_PRED)),
        ref_sets=dict(default=str(D30_SETS)), ref_log=dict(default=str(D30_LOG)), groups=dict(nargs="+", default=None), **ph)
    add("integrity-run", unit=dict(required=True), **ph)
    add("integrity-compare", **ph)
    add("manifest", extra_timing_dir=dict(default=None), **ph, **tim)
    add("score", lr_choice=dict(default=None), reps=dict(type=int, default=2000), out=dict(default=None), extra_timing_dir=dict(default=None), **ph, **tim)
    a = ap.parse_args(argv)
    if a.cmd == "units":                                                                # the 9 main unit tags of an lr_choice.json (driver gates)
        print(" ".join(u for o in RUN.OBJECTIVES for u in main_units(a.lr_choice)[1][o])); return 0
    if a.cmd in ("build", "prepare"):
        s = build(a.root, a.gqa, a.chartqa, a.protocol, a.gqa_images, mix665k=None if a.mix665k in ("", "none") else a.mix665k)
        print(json.dumps({b: {k: v.get(k) for k in ("mode", "source", "sha256", "sha256_matches_record", "items", "groups", "images", "split_counts",
                                                    "inputs_sha256", "labels_sha256", "seconds", "exposure_check")} for b, v in s["benches"].items()}, indent=1))
        if a.cmd == "build":
            return 0
    if a.cmd in ("configs", "prepare"):
        d30c = None if a.d30_configs in ("", "none") else a.d30_configs
        print(json.dumps(configs(a.root, getattr(a, "phase", "L"), a.lr_choice, a.selectors_dir, a.pca, d30c)))
        if a.cmd == "configs":
            return 0
    if a.cmd in ("lines", "prepare"):
        phase = getattr(a, "phase", "L"); print(f"{write_lines(a.root, phase, getattr(a, 'lines', None))} units -> {getattr(a, 'lines', None) or Layout(a.root, phase).lines}")
        return 0
    if a.cmd == "run":
        return run_quality(a.root, a.bench, a.shard, a.phase, a.limit)
    if a.cmd == "sparsevlm":
        return run_sparsevlm(a.root, a.bench, a.shard, a.limit, a.rows)
    if a.cmd == "integrity-prepare":
        print(json.dumps(integrity_prepare(a.root, a.phase, a.lr_choice, a.selectors_dir, a.pca, a.ref_pred, a.ref_sets, a.ref_log, a.groups))); return 0
    if a.cmd == "integrity-run":
        return integrity_run(a.root, a.phase, a.unit)
    if a.cmd == "integrity-compare":
        return integrity_compare(a.root, a.phase)
    if a.cmd == "manifest":
        digest, m = manifest(a.root, a.phase, a.timing_dir, tmap_arg(a.timing_bench), a.extra_timing_dir)
        print(json.dumps({"phase": a.phase, "manifest_sha256": digest, "n_files": m["n_files"], "rows": {b: v["n_rows"] for b, v in m["rows"].items()},
                          "items": {b: v["items"] for b, v in m["inputs"].items()}}, indent=1)); return 0
    r = score(a.root, a.phase, a.lr_choice, a.timing_dir, tmap_arg(a.timing_bench), a.reps, a.out, extra_timing_dir=a.extra_timing_dir)
    print(json.dumps({b: {"n_items": v["n_items"], "rows": len(v["rows"]), "primary_contrasts": sum(c.get("role") == "primary" for c in v["contrasts"].values())}
                      for b, v in r["benches"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
