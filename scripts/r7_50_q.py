"""r7 R7-50 phase Q — Qwen2.5-VL-7B-Instruct rows on the official test splits (GQA testdev-balanced 12,578 q, ChartQA test 2,500 q) under the
frozen addendum reports/R7/R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml (status frozen_20261013; decisions Q1–Q8) of the frozen R7-50 protocol
reports/R7/R7_50_PROTOCOL_FROZEN_20261012.yaml (phases, sealed inputs / labels, manifest -> commit -> ALLOW_SCORING -> score once).  Reused
read-only, by import: the R7-40c quality runner (r7_d1_40c_run 'run': host, prompts (Q5), selection, generation, D1-30 sharding, chunks of 32
sorted by merged-token count, single-writer shard files, markers; the GPU wrapper executes it with this module's verified arguments), its
planner (r7_d1_40c_plan: grammar, config_list, unit tags, MAX_NEW incl. gqa_testdev / chartqa_test, shard counts, partition rule), its scorer
(r7_d1_40c_score: Bench40, score_bench, freeze key), its B* reader (r7_d1_40c_timing.read_b_star) and phase L's sealed-test plumbing
(r7_50_test: Layout with the reserved phase-Q paths, the development-mode build the gate uses, ALLOW_SCORING / committed-manifest gate, sealed
labels).  No test file is read here: phase Q runs on the sealed inputs phase L built (Q3, 'no second read of a test file').
Sub-commands:
  configs      CPU  configs_Q: the 41 rows of Q1 per benchmark (R7-40c's core list without its 9 dynamic rows), unit tags from R7-40b's
                    lr_choice.json, sha256 of the 9 units (.pt / .json), the PCA and lr_choice.json -> configs_Q/rows.json; every sealed-input
                    image's merged grid within the selectors' 64 × 64 tables (R7-40c stage-0 rule; else STOP for a decision)
  lines        CPU  lines_Q: GQA testdev 6 / ChartQA test 2 quality units (R7-40c's sharding) on L40S (R7-40c's quality class), one marker each
  runner-args  GPU node, before the runner: the unit's guard; prints the arguments of 'r7_d1_40c_run.py run' (--sets-dir <root>/sealed/inputs,
                    --out <root>/sealed/pred_Q; or the integrity root) that slurm/r7_50_q_gpu.sbatch then executes
  integrity-prepare / integrity-compare     the full-scale integrity gate (Q3 / Q6): the phase-Q path on the FULL GQA DEV (7,986) and ChartQA
                    val (1,920) sets with R7-40c's question ids, sharding (GQA 6, ChartQA 2) and chunking, one unit per R7-40c reference shard on
                    the GPU class of its marker; full, static:grid:0.25 and S1 seed 0 at 0.25 must give R7-40c's answer and kept count on every
                    item, the inputs (question, prompt, image file / bytes) must be identical on every item; anything else STOPs before generation
  manifest     CPU  coverage of the 41 rows (labels not opened), shard markers vs configs_Q / sealed inputs / selector hashes, sha256 of every
                    prediction file, rows.json, the build summary, and R7-40c's timing bundle files, its marker and the R7-40a B* report (Q2 / Q7)
  score        CPU  the only phase-Q step that opens the sealed labels: refuses unless ALLOW_SCORING_Q names the manifest (production root: and its
                    committed copy) and every prediction / timing file still matches; R7-40c's scorer on the labelled items, Holm over 6 per point
                    (Q4), the profiling-workload latency of the mapped validation benchmark (Q2), primary basis = R7-40c's (B* from the R7-40a report)
    python -I scripts/r7_50_q.py configs --root R          python -I scripts/r7_50_q.py integrity-prepare --root R          python -I scripts/r7_50_q.py score --root R
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_50_test as T  # noqa: E402   (phase L: Layout, development-mode build, sealed labels, ALLOW_SCORING gate, io helpers; read-only use)
import r7_d1_30_score as SC  # noqa: E402   (E1 loader, group bootstrap, Holm)
import r7_d1_40c_plan as P40  # noqa: E402   (grammar, config_list, unit tags, MAX_NEW, shard counts, partition rule)
import r7_d1_40c_run as RN  # noqa: E402   (the R7-40c quality runner: defaults, image-grid summary)
import r7_d1_40c_score as S40  # noqa: E402   (Bench40, score_bench, freeze key)
import r7_d1_40c_timing as TMG  # noqa: E402   (B* reader, bundle marker name)
import r7_vl_common as V  # noqa: E402

PHASE = "Q"
ROOT = T.ROOT
R40, R40C = P40.R40, P40.R40C
LR_CHOICE = R40 / "selectors" / "lr_choice.json"
SELECTORS, PCA = RN.SELECTORS, RN.PCA                                     # R7-40b's CAL-selected states and PCA (the runner's defaults)
R40C_PRED, R40C_CONFIGS, R40C_TIMING = R40C / "pred", R40C / "configs", R40C / "timing"
TIMING_REPORT = P40.TIMING_REPORT                                         # R7-40a reference timing report (B*, D40a-7; reused by the bundle, decision 14)
D30_SETS = P40.D30_SETS                                                   # the sets R7-40c ran on (its own question ids)
R40C_LOG = Path("/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_40c_submit.log")
ADDENDUM = T.REPO / "reports" / "R7" / "R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml"
PROTOCOL_40C = S40.PROTOCOL                                               # carries R7-40c's freeze key (r7_40c_staged_freeze)
BENCHES = T.BENCHES
VAL = T.D30_BENCH                                                         # gqa_testdev <- gqa_dev, chartqa_test <- chartqa_val (Q2 latency map)
SHARDS = {b: P40.QUALITY_SHARDS[v] for b, v in VAL.items()}              # GQA 6, ChartQA 2 (R7-40c's sharding of the mapped sets)
N_ROWS = 41                                                               # Q1
QPART = "L40S"                                                            # R7-40c's quality class (OPEN-R740C-7); Q8: no node-pinned unit
UNIT_TIME = {"gqa_testdev": "03:00:00", "chartqa_test": "04:00:00"}       # ≈ 3 × the cost line's per-shard estimate (see units_and_cost)
INTEGRITY_TIME = "00:45:00"                                               # ≈ 3 × 12-15 min (3 rows × one reference shard + one ≈ 7-min host load)
GATE_F = "0.25"
STATUS_RE = re.compile(r"^status:\s*(frozen\S*)", re.M)
SAFE = re.compile(r"^[A-Za-z0-9_./:=+@-]+$")                              # runner arguments go through the wrapper's word splitting
GATE_RULE = ("Q3 / Q6 (coordinator, 2026-10-13): full, static:grid:0.25 and S1 seed 0 at 0.25 (R7-40c's own unit) give R7-40c's answer and kept "
             "count on every item of the FULL GQA DEV (7,986) and ChartQA val (1,920) sets, with R7-40c's question ids, sharding (GQA 6, ChartQA 2) "
             "and chunking, each unit on the GPU class recorded in its R7-40c reference marker (env.gpu); inputs (question, prompt, image file / "
             "bytes) identical on every item; anything else -> STOP before any test generation.  Residual risk (reported, not waived): a reference "
             "shard resumed after a crash has different chunk compositions after its resume point")
LATENCY_LABEL = ("latency measured on the pre-fixed profiling workload: R7-40c's timing results of the mapped validation benchmark (gqa_dev for GQA "
                 "testdev-balanced, chartqa_val for ChartQA test; NVIDIA H100 NVL on nodesumo01, 8 forced decode steps, one nodesumo01 bundle job); "
                 "every fixed-fraction selector row takes the representative S1 seed-0 unit's latency.  Not re-measured on test and not the test "
                 "sets' actual mean end-to-end latency (addendum Q2 / Q7)")
# OPEN: the addendum keeps the project-side sentence 'unchanged' (phase L's: '... fixed before the read').  Phase Q's 41 rows were declared in the
# frozen protocol (Q1) before phase L's read, but the addendum (Holm over 6, basis, latency mapping) was frozen after that read (git 45b3e02, before
# any label was opened and any Qwen test generation); the coordinator may want to state that order in the report.
EXPOSURE = [T.EXPOSURE[0],
            "Relative to the host's training data: Qwen2.5-VL's training data is undisclosed, so exposure to GQA / ChartQA train (and possibly val / "
            "test) cannot be excluded; stated as such (addendum analysis.exposure)."]
sha = T.sha


def fsha(p):
    """sha256 of a file, None when it is missing (a guard refuses cleanly instead of raising)."""
    return sha(p) if p and Path(p).is_file() else None


# ------------------------------------------------------------------------------------------------------------------------------ rows
def q_rows(main):
    """Q1: R7-40c's core list (r7_d1_40c_plan.config_list) without its 9 dynamic rows, order kept: full, static:grid × 4 fractions, S1 / S2 / S3 ×
    seeds 0-2 × 4 fractions = 41 (no dynamic, random, qsim or SparseVLM row; 0.125 = the stress point, never a primary point)."""
    return [c for c in P40.config_list(main, "core") if not c.startswith("dyn:")]


def row_meta(c, main):
    kind, a, f = P40.parse(c)
    if kind == "full":
        return {"family": "full", "fraction": 1.0}
    if kind == "static":
        return {"family": "uniform_grid", "fraction": f, "stress_point": f == 0.125}
    o = P40.objective_of(a)
    return {"family": P40.NAMES[o], "objective": o, "seed": main[o].index(a), "unit": a, "fraction": f, "stress_point": f == 0.125}


def require_addendum(path):
    t = Path(path).read_text() if Path(path).is_file() else ""; m = STATUS_RE.search(t)
    if not m:
        raise SystemExit(f"{path} is not the frozen phase-Q addendum (a line 'status: frozen…'): phase Q stays unconfigured")
    return {"file": str(path), "sha256": sha(path), "status": m.group(1)}


def unit_record(selectors_dir, main):
    """sha256 of the 9 R7-40b units at their CAL-chosen rates (.pt = the CAL-selected state the runner loads, .json = its R7-40b log) and their
    CAL-selected step; refuses an incomplete unit (nothing selected on test)."""
    out = {}
    for o in P40.OBJECTIVES:
        for s, u in enumerate(main[o]):
            pt, js = Path(selectors_dir) / f"{u}.pt", Path(selectors_dir) / f"{u}.json"
            if not (pt.is_file() and js.is_file()):
                raise SystemExit(f"R7-40b unit {u} incomplete in {selectors_dir}")
            lg = json.loads(js.read_text())
            out[u] = {"objective": o, "name": P40.NAMES[o], "seed": s, "pt_sha256": sha(pt), "json_sha256": sha(js), "selected_step": lg.get("selected_step"),
                      "selected_cal_spearman": lg.get("selected_cal_spearman")}
    return out


def configs(root, lr_choice=LR_CHOICE, selectors_dir=SELECTORS, pca=PCA, r40c_configs=R40C_CONFIGS, addendum=ADDENDUM):
    """Stage 0 (CPU; no test file is read; labels never opened).  configs_Q/configs_<bench>.txt = the 41 rows of Q1 (cross-checked as a subset of
    R7-40c's own configuration files, as phase L against D1-30's), configs_Q/rows.json = addendum, lr_choice, selectors_dir, PCA and the 9 units
    with sha256 (the guard and the runner use exactly these).  Image headers of the sealed inputs (no pixel decoded): a missing image or a merged
    grid beyond the selectors' 64 × 64 tables stops phase Q for a decision (configs_Q/grid_check.json).  Fixed once generation has started."""
    L = T.Layout(root, PHASE); summ = T.load_summary(L); add = require_addendum(addendum)
    if L.pred.exists() and T.walk_files(L.pred):
        raise SystemExit(f"{L.pred} holds predictions: configs_Q is fixed once phase-Q generation has started")
    for b in BENCHES:
        if sha(L.inputs_file(b)) != summ["benches"][b]["inputs_sha256"]:
            raise SystemExit(f"{L.inputs_file(b)} changed since the build")
    if not Path(pca).is_file() or not Path(lr_choice).is_file():
        raise SystemExit(f"R7-40b PCA {pca} or lr_choice {lr_choice} missing")
    grid = RN.grid_summary(L.inputs, BENCHES)
    bad = {b: {k: g[k] for k in ("missing_images", "missing_examples", "images_over_64x64", "grid_max_rows", "grid_max_cols")}
           for b, g in grid.items() if g["missing_images"] or g["images_over_64x64"]}
    if bad:
        T.write_atomic(L.configs / "grid_check.json", json.dumps({"utc": T.utc(), "benches": grid, "pass": False}, indent=1))
        raise SystemExit(f"sealed-input images missing or with a merged grid beyond the selectors' 64 × 64 tables: {bad}; phase Q stops for a "
                         "decision (addendum host_and_generation.images)")
    choice = json.loads(Path(lr_choice).read_text()); main = P40.unit_names(choice); rows = q_rows(main)
    if len(rows) != N_ROWS:
        raise SystemExit(f"{len(rows)} rows, expected {N_ROWS} (Q1)")
    rec = {"phase": PHASE, "created_utc": T.utc(), "addendum": add, "planner": "scripts/r7_d1_40c_plan.py config_list (core) without the 9 dynamic rows (Q1)",
           "lr_choice": {"file": str(lr_choice), "sha256": sha(lr_choice), "chosen_lr": {o: choice[o]["chosen_lr"] for o in P40.OBJECTIVES}},
           "selectors_dir": str(selectors_dir), "pca": {"file": str(pca), "sha256": sha(pca)}, "units": unit_record(selectors_dir, main),
           "representative_unit": main["vcs"][0], "grid_check": grid, "benches": {}}
    for b in BENCHES:
        live = Path(r40c_configs) / f"configs_{VAL[b]}.txt" if r40c_configs else None; live_rec = None
        if live is not None and live.is_file():                                         # R7-40c's own configuration file
            miss = [r for r in rows if r not in set(P40.read_configs(live))]
            if miss:
                raise SystemExit(f"{b}: rows absent from R7-40c's {live}: {miss[:5]}")
            live_rec = {"file": str(live), "sha256": sha(live)}
        T.write_atomic(L.configs_file(b), "\n".join(rows) + "\n")
        rec["benches"][b] = {"r7_40c_counterpart": VAL[b], "rows": len(rows), "configs_sha256": sha(L.configs_file(b)), "r7_40c_configs": live_rec,
                             "max_new_tokens": P40.MAX_NEW[b], "shards": SHARDS[b], "row_meta": {c: row_meta(c, main) for c in rows}}
    T.write_atomic(L.configs / "rows.json", json.dumps(rec, indent=1))
    return {b: v["rows"] for b, v in rec["benches"].items()}


def lines(root):
    """lines_Q: GQA testdev 6 / ChartQA test 2 quality units (R7-40c's sharding of gqa_dev / chartqa_val), all 41 rows of a shard in one job
    (paired rows share the GPU), on L40S with node51 / node60 excluded (R7-40c's partition rule); each line carries the runner's shard marker."""
    L = T.Layout(root, PHASE); out = []
    for b in BENCHES:
        n = SHARDS[b]
        for i in range(n):
            out.append((f"sbatch --parsable --job-name=r7_50qq_{b}_{i} {P40.gpu(QPART)} --time={UNIT_TIME[b]} slurm/r7_50_q_gpu.sbatch run {b} {i}/{n} {L.root}",
                        L.pred / b / f"done_shard{i}of{n}.json"))
    return out


def write_lines(root, path=None):
    L = T.Layout(root, PHASE); path = Path(path or L.lines); units = lines(root)
    T.write_atomic(path, "# R7-50 phase Q generation units: <sbatch line> #MARK <completion marker>\n" + "".join(f"{l} #MARK {m}\n" for l, m in units))
    return len(units)


# ------------------------------------------------------------------------------------------------------------------------------ guards (GPU node)
def shard_of(bench, shard):
    try:
        i, n = (int(x) for x in shard.split("/"))
    except ValueError:
        raise SystemExit(f"bad shard {shard!r}") from None
    if bench not in BENCHES or n != SHARDS[bench] or not 0 <= i < n:
        raise SystemExit(f"{bench} shard {shard}: phase Q uses R7-40c's sharding {SHARDS}")
    return i, n


def check_selectors(selectors_dir, pca, units, pca_sha):
    """The selector / PCA files the runner will load must equal the recorded sha256 (a phase cannot silently load other checkpoints)."""
    bad = [u for u, r in units.items() if fsha(Path(selectors_dir) / f"{u}.pt") != r["pt_sha256"] or fsha(Path(selectors_dir) / f"{u}.json") != r["json_sha256"]]
    if fsha(pca) != pca_sha:
        bad.append("pca")
    if bad:
        raise SystemExit(f"selector / PCA files differ from their recorded sha256 ({bad[:3]}): not run")


def runner_argv(sets_dir, bench, configs_file, shard, out, selectors_dir, pca):
    """Arguments of 'r7_d1_40c_run.py run' (the R7-40c quality runner, unchanged; chunk / batch 32 = its defaults, as R7-40c's own units)."""
    argv = ["run", "--sets-dir", str(sets_dir), "--bench", bench, "--configs-file", str(configs_file), "--shard", shard, "--out", str(out),
            "--selectors-dir", str(selectors_dir), "--pca", str(pca)]
    bad = [x for x in argv if not SAFE.match(x)]
    if bad:
        raise SystemExit(f"runner arguments with unsafe characters {bad[:3]}")
    return argv


def gate_passed(L):
    p = L.integrity / "integrity.json"
    return p.is_file() and json.loads(p.read_text()).get("pass") is True


def generation_args(root, bench, shard):
    """Guard of one test generation unit (GPU node, right before the runner): sealed inputs built and unchanged since the build, the full-scale
    integrity gate passed (Q3 / Q6: no test generation before it), predictions not yet sealed (no generation after manifest_Q), configs_Q
    unedited and equal to the 41 rows of Q1, selector / PCA files equal to configs_Q/rows.json.  -> runner arguments: --sets-dir
    <root>/sealed/inputs, --out <root>/sealed/pred_Q."""
    L = T.Layout(root, PHASE); shard_of(bench, shard); T.guard_generation(L, [bench])
    if not gate_passed(L):
        raise SystemExit(f"{L.integrity / 'integrity.json'} does not say pass: no phase-Q test generation before the full-scale integrity gate (Q3 / Q6)")
    summ = T.load_summary(L)
    if sha(L.inputs_file(bench)) != summ["benches"][bench]["inputs_sha256"]:
        raise SystemExit(f"{L.inputs_file(bench)} changed since the build")
    rec = json.loads((L.configs / "rows.json").read_text())
    if fsha(rec["lr_choice"]["file"]) != rec["lr_choice"]["sha256"]:
        raise SystemExit(f"{rec['lr_choice']['file']} missing or changed since configs_Q")
    main = P40.unit_names(json.loads(Path(rec["lr_choice"]["file"]).read_text()))
    if sha(L.configs_file(bench)) != rec["benches"][bench]["configs_sha256"] or P40.read_configs(L.configs_file(bench)) != q_rows(main):
        raise SystemExit(f"{L.configs_file(bench)} differs from configs_Q/rows.json or from the 41 rows of Q1")
    check_selectors(rec["selectors_dir"], rec["pca"]["file"], rec["units"], rec["pca"]["sha256"])
    return runner_argv(L.inputs, bench, L.configs_file(bench), shard, L.pred, rec["selectors_dir"], rec["pca"]["file"])


def device_class(name):
    return T.gpu_class(name) if name else None


def integrity_args(root, bench, shard, device_name=None):
    """Guard of one integrity unit (GPU node, right before the runner): the plan's unit for (bench, shard), this GPU's class = the class recorded in
    the R7-40c reference marker (Q6), the gate inputs built, the gate rows, the S1 seed-0 unit and PCA equal to the plan's sha256.  -> runner
    arguments with the integrity root (--sets-dir <root>/integrity_Q/sealed/inputs, --out <root>/integrity_Q/sealed/pred_Q)."""
    L = T.Layout(root, PHASE); base = L.integrity; IB = T.Layout(base, PHASE); i, n = shard_of(bench, shard)
    plan = json.loads((base / "plan.json").read_text()); u = plan["units"].get(f"{bench}_{i}of{n}")
    if u is None:
        raise SystemExit(f"no integrity unit {bench} {shard} in {base / 'plan.json'}")
    if device_name is None:
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if not device_name:
        raise SystemExit("no CUDA device: an integrity unit runs on its reference's GPU class")
    if device_class(device_name) != u["class"]:
        raise SystemExit(f"integrity unit {bench} {shard} is pinned to {u['class']} (its R7-40c reference marker), running on {device_name}")
    T.load_summary(IB)
    if P40.read_configs(IB.configs_file(bench)) != plan["rows"]:
        raise SystemExit(f"{IB.configs_file(bench)} differs from the gate rows {plan['rows']}")
    check_selectors(plan["selectors_dir"], plan["pca"]["file"], plan["units_sha256"], plan["pca"]["sha256"])
    return runner_argv(IB.inputs, bench, IB.configs_file(bench), shard, IB.pred, plan["selectors_dir"], plan["pca"]["file"])


# ------------------------------------------------------------------------------------------------------------------------------ integrity gate
def retried_markers(log):
    """Reference markers the R7-40c driver resubmitted (its 'RETRY … without <marker>' lines)."""
    return [ln.split(" without ")[-1].strip() for ln in (Path(log).read_text().splitlines() if Path(log).is_file() else []) if "RETRY" in ln]


def resumed(marker):
    """True when the marker's own session answered fewer than items × configurations (an earlier session wrote part of the shard, so chunk
    compositions after its resume point differ); None when the marker does not say (written by a session that had nothing left to answer)."""
    a = marker.get("answers_this_session")
    return None if a is None else bool(a < (marker.get("items") or 0) * (marker.get("configs") or 0))


def integrity_prepare(root, lr_choice=LR_CHOICE, selectors_dir=SELECTORS, pca=PCA, ref_pred=R40C_PRED, ref_sets=D30_SETS, ref_log=R40C_LOG,
                      chartqa_src=T.CHARTQA_VAL, images_dir=T.GQA_IMAGES):
    """Gate step 1 (CPU; non-test data only): the R7-50 build path in development mode on the FULL GQA DEV set (R7-40c's = D1-30's items
    re-expressed in the GQA balanced format, answers withheld) and the FULL ChartQA val parquet with R7-40c's ids (chartqa_val_<row>), into
    <root>/integrity_Q/sealed/inputs (as phase L's gate), then the plan (integrity_plan)."""
    L = T.Layout(root, PHASE); base = L.integrity; IB = T.Layout(base, PHASE)
    if not IB.summary.exists():
        src = base / "sources"; src.mkdir(parents=True, exist_ok=True); dev = T.read_jsonl_strict(Path(ref_sets) / "gqa_dev.jsonl")
        gq = src / "gqa_dev_r7_40c_questions.json"
        gq.write_text(json.dumps({it["qid"]: {"imageId": it["image_id"], "question": it["question"], "answer": "withheld", "types": {}} for it in dev}))
        T.build(base, gq, chartqa_src, images_dir=images_dir, chartqa_qid_prefix="chartqa_val")
    return integrity_plan(root, lr_choice, selectors_dir, pca, ref_pred, ref_sets, ref_log)


def integrity_plan(root, lr_choice=LR_CHOICE, selectors_dir=SELECTORS, pca=PCA, ref_pred=R40C_PRED, ref_sets=D30_SETS, ref_log=R40C_LOG):
    """Gate plan: the 3 gate rows per benchmark (configs file of the integrity root), one unit per R7-40c reference shard (GQA 6, ChartQA 2; the
    addendum's 8 host loads), each pinned to the GPU class of its reference marker (env.gpu; a class, not a node: Q8), with the reference's
    retry (driver log) and resume (marker) status; plan.json + lines.txt in <root>/integrity_Q."""
    # OPEN: units_and_cost says 'GQA 6 / ChartQA 2 shards × 3 rows, grouped by GPU class' and costs '8 loads'; phase L grouped every task of a
    # class into one unit.  Read here as one unit (one host load, a fresh process as R7-40c's own shard job) per reference shard, pinned to its
    # class: the 8 loads of the cost line and R7-40c's unit structure.
    L = T.Layout(root, PHASE); base = L.integrity; IB = T.Layout(base, PHASE); T.load_summary(IB)
    main = P40.unit_names(json.loads(Path(lr_choice).read_text())); s1 = main["vcs"][0]
    rows = ["full", f"static:grid:{GATE_F}", f"sel:{s1}:{GATE_F}"]
    for b in BENCHES:
        T.write_atomic(IB.configs_file(b), "\n".join(rows) + "\n")
    retried = retried_markers(ref_log); units = {}
    for b in BENCHES:
        rd, n = Path(ref_pred) / VAL[b], SHARDS[b]
        for i in range(n):
            m = rd / f"done_shard{i}of{n}.json"
            if not m.is_file():
                raise SystemExit(f"R7-40c reference marker missing: {m} (the gate needs R7-40c's full-scale outputs)")
            mk = json.loads(m.read_text())
            units[f"{b}_{i}of{n}"] = {"bench": b, "shard": f"{i}/{n}", "class": T.gpu_class((mk.get("env") or {}).get("gpu")), "reference_marker": str(m),
                                     "reference_retried": str(m) in retried, "reference_resumed": resumed(mk), "reference_items": mk.get("items"),
                                     "reference_configs": mk.get("configs"), "reference_sets_file_sha256": mk.get("sets_file_sha256"),
                                     "marker": str(IB.pred / b / f"done_shard{i}of{n}.json")}
    sp, sj = Path(selectors_dir) / f"{s1}.pt", Path(selectors_dir) / f"{s1}.json"
    if not (sp.is_file() and sj.is_file() and Path(pca).is_file()):
        raise SystemExit(f"S1 seed-0 unit {s1} or the PCA missing in {selectors_dir} / {pca}")
    plan = {"phase": PHASE, "created_utc": T.utc(), "rule": GATE_RULE, "rows": rows, "units": units, "lr_choice": {"file": str(lr_choice), "sha256": sha(lr_choice)},
            "selectors_dir": str(selectors_dir), "pca": {"file": str(pca), "sha256": sha(pca)}, "units_sha256": {s1: {"pt_sha256": sha(sp), "json_sha256": sha(sj)}},
            "reference_pred": str(ref_pred), "reference_sets": str(ref_sets), "reference_log": str(ref_log), "retried_reference_markers": retried}
    T.write_atomic(base / "plan.json", json.dumps(plan, indent=1))
    T.write_atomic(base / "lines.txt", "# R7-50 phase Q full-scale integrity units (one per R7-40c reference shard, pinned to its GPU class): <sbatch line> #MARK <marker>\n" + "".join(
        f"sbatch --parsable --job-name=r7_50iq_{u['bench']}_{u['shard'].split('/')[0]} {P40.gpu(u['class'])} --time={INTEGRITY_TIME} slurm/r7_50_q_gpu.sbatch "
        f"integrity {L.root} {u['bench']} {u['shard']} #MARK {u['marker']}\n" for u in units.values()))
    return {k: u["class"] for k, u in units.items()}


def integrity_compare(root):
    """Gate step 3 (CPU): inputs (question, prompt, image) of every item, then every gate row's answer and kept count against R7-40c's, item by
    item (an item missing on either side is a mismatch); R7-40c's reference sets must be the files its markers recorded; every unit's marker must
    show its pinned GPU class, the gate configs and the gate inputs.  Prediction strings are compared for equality only: no label is opened,
    nothing is scored, no prediction is printed or recorded.  integrity.json 'pass' (top level only) -> the driver's gate."""
    L = T.Layout(root, PHASE); base = L.integrity; IB = T.Layout(base, PHASE); plan = json.loads((base / "plan.json").read_text())
    res = {"phase": PHASE, "compared_utc": T.utc(), "rule": GATE_RULE, "plan_rule_text": plan.get("rule"), "rows": plan["rows"],
           "retried_reference_markers": plan["retried_reference_markers"], "benches": {}, "units": {}}; ok = True
    for b in BENCHES:
        v = VAL[b]; rf = Path(plan["reference_sets"]) / f"{v}.jsonl"; mine = {it["qid"]: it for it in T.read_jsonl_strict(IB.inputs_file(b))}
        ref = {it["qid"]: it for it in T.read_jsonl_strict(rf)}; r = {"items": len(ref), "same_qid_set": set(mine) == set(ref)}
        same = {"question": 0, "prompt": 0, "image": 0}
        for q, o in ref.items():
            it = mine.get(q)
            if it is None:
                continue
            same["question"] += it["question"] == o["question"]; same["prompt"] += it["prompt"] == o["prompt"]
            same["image"] += it["image_id"] == o["image_id"] and (it["image_file"] == o["image_file"] if b.startswith("gqa") else sha(it["image_file"]) == sha(o["image_file"]))
        # OPEN: 'inputs identical' is checked against R7-40c's inputs, i.e. the sets file every R7-40c reference marker recorded (sets_file_sha256);
        # a reference sets file that differs from what R7-40c read fails the gate (stricter than phase L, which did not check the markers' hash).
        rs = sha(rf); recorded = sorted({u["reference_sets_file_sha256"] for u in plan["units"].values() if u["bench"] == b}, key=str)
        r.update(inputs_identical=same, reference_sets_sha256=rs, reference_sets_match_r7_40c_markers=recorded == [rs])
        ok &= r["same_qid_set"] and all(x == len(ref) for x in same.values()) and r["reference_sets_match_r7_40c_markers"]
        for cfg in plan["rows"]:
            got, ng = SC.load_config(IB.pred / b, cfg); old, no = SC.load_config(Path(plan["reference_pred"]) / v, cfg)
            here = [q for q in ref if q not in got]; there = [q for q in ref if q not in old]; both = [q for q in ref if q in got and q in old]
            ans = {q for q in both if got[q]["pred"] != old[q]["pred"]}; kept = {q for q in both if got[q].get("kept") != old[q].get("kept")}
            bad = set(here) | set(there) | ans | kept; first = [q for q in ref if q in bad][:20]
            x = {"items": len(ref), "answered_here": len(ref) - len(here), "answered_reference": len(ref) - len(there), "identical": len(ref) - len(bad),
                 "answer_differs": len(ans), "kept_differs": len(kept), "missing_here": len(here), "missing_reference": len(there),
                 "n_vis_identical": sum(got[q].get("n_vis") == old[q].get("n_vis") for q in both), "mismatched_qids_first_20": first,
                 "files_here": ng["files"], "files_reference": no["files"], "lines_dropped_or_duplicate": {"here": ng["dropped_lines"] + ng["duplicate_qids"],
                                                                                                        "reference": no["dropped_lines"] + no["duplicate_qids"]},
                 "rule": "identical answer and kept count on every item", "ok": not bad}
            r[cfg] = x; ok &= x["ok"]
        res["benches"][b] = r
    for k, u in plan["units"].items():
        m = Path(u["marker"]); y = {"bench": u["bench"], "shard": u["shard"], "class": u["class"], "reference_retried": u["reference_retried"],
                                    "reference_resumed": u["reference_resumed"], "marker_present": m.is_file()}
        if m.is_file():
            mk = json.loads(m.read_text()); g = (mk.get("env") or {}).get("gpu")
            try:
                c = device_class(g)
            except SystemExit:
                c = None
            y.update(gpu=g, class_here=c, resumed_here=resumed(mk), configs_ok=mk.get("configs_sha256") == sha(IB.configs_file(u["bench"])),
                     inputs_ok=mk.get("sets_file_sha256") == sha(IB.inputs_file(u["bench"])))
            y["ok"] = c == u["class"] and y["configs_ok"] and y["inputs_ok"]
        else:
            y["ok"] = False
        res["units"][k] = y; ok &= y["ok"]
    res["pass"] = bool(ok); T.write_atomic(base / "integrity.json", json.dumps(res, indent=1))
    print(json.dumps({"pass": res["pass"], **{b: {c: (x["identical"], x["items"]) for c, x in r.items() if isinstance(x, dict) and "identical" in x}
                                             for b, r in res["benches"].items()}}))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------------------ manifest
def timing_files(timing_dir=R40C_TIMING, report=TIMING_REPORT):
    """Q2 / Q7: the files phase Q's latency and basis rest on — R7-40c's timing bundle marker, every file its parts list (the two mapped
    benchmarks' timing files required, the others hashed too) and the R7-40a B* report the scorer reads, which the bundle must list as its
    reused B* part (BSTAR=report, decision 14).  Every sha256 the marker records must still hold.  {name: path}."""
    # OPEN: the addendum's implementation note counts 'the six R7-40c timing / timing-report files'; Q2 / Q7 name the mapped timing files and the
    # bundle marker.  Hashed here: the marker, the R7-40a B* report and every file the marker lists (gqa_dev / textvqa_val / chartqa_val) = 5.
    td = Path(timing_dir); mark = td / TMG.BUNDLE_MARK; need = {mark.name: mark, Path(report).name: Path(report)}
    need.update({f"timing_{v}.json": td / v / f"timing_{v}.json" for v in VAL.values()})
    missing = [str(p) for p in need.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"R7-40c timing / timing-report files missing: {missing}")
    parts = {p.get("part"): p for p in json.loads(mark.read_text()).get("parts", [])}; out = dict(need)
    for name, want in [("b_star", Path(report))] + [(v, td / v / f"timing_{v}.json") for v in VAL.values()]:
        f = (parts.get(name) or {}).get("file")
        if not f or Path(f).resolve() != want.resolve():
            raise SystemExit(f"{mark}: part {name!r} does not list {want} (lists {f!r}); phase Q's latency must come from this bundle")
    for p in parts.values():
        f = p.get("file")
        if f:
            f = Path(f)
            if not f.is_file() or (p.get("sha256") and sha(f) != p["sha256"]):
                raise SystemExit(f"{f} is missing or differs from the sha256 recorded in {mark}")
            out.setdefault(f.name, f)
    return out


def shard_markers(L, b, rec, inputs_sha):
    """Every shard marker of a benchmark: written by the R7-40c runner on these sealed inputs with configs_Q and (when the marker's session loaded
    them) the recorded selector / PCA files.  A marker rewritten by a session with nothing left to answer carries no selector record."""
    out, n, cfs = {}, SHARDS[b], rec["benches"][b]["configs_sha256"]
    for i in range(n):
        m = L.pred / b / f"done_shard{i}of{n}.json"
        if not m.is_file():
            raise SystemExit(f"shard marker missing: {m}")
        mk = json.loads(m.read_text()); sf = mk.get("selector_files")
        if mk.get("configs_sha256") != cfs or mk.get("sets_file_sha256") != inputs_sha:
            raise SystemExit(f"{m}: written with other configs or inputs than configs_Q / the sealed inputs")
        if sf is not None:
            bad = [u for u, r in rec["units"].items() if (sf.get(u) or {}).get("pt_sha256") != r["pt_sha256"] or (sf.get(u) or {}).get("json_sha256") != r["json_sha256"]]
            if bad or (sf.get("pca") or {}).get("sha256") != rec["pca"]["sha256"]:
                raise SystemExit(f"{m}: selector / PCA files differ from configs_Q/rows.json ({bad[:3]})")
        out[m.name] = {"items": mk.get("items"), "configs": mk.get("configs"), "gpu": (mk.get("env") or {}).get("gpu"), "seconds": mk.get("seconds"),
                       "answers_this_session": mk.get("answers_this_session"), "resumed": resumed(mk), "selector_record": sf is not None, "utc": mk.get("utc")}
    return out


def manifest(root, timing_dir=R40C_TIMING, timing_report=TIMING_REPORT):
    """Stage 2 (CPU): coverage of the 41 rows of each benchmark (prediction files read for question ids only; labels never opened; first complete
    record per qid, E1), the shard markers, sha256 of every phase-Q prediction file, rows.json, the build summary and the R7-40c timing files
    (timing_files) -> manifest_Q/predictions_manifest.json + .sha256; generation refuses to run once it exists.  Refuses partial coverage."""
    L = T.Layout(root, PHASE); summ = T.load_summary(L)
    if L.manifest.exists():
        raise SystemExit(f"{L.manifest} exists: phase Q predictions already sealed")
    rec = json.loads((L.configs / "rows.json").read_text()); tf = timing_files(timing_dir, timing_report); ig = L.integrity / "integrity.json"
    out = {"phase": PHASE, "created_utc": T.utc(), "root": str(L.root), "git": T.git_state(), "build_summary_sha256": sha(L.summary),
           "rows_json_sha256": sha(L.configs / "rows.json"), "addendum": rec.get("addendum"), "inputs": {}, "labels_not_opened": {}, "rows": {}, "coverage": {},
           "shard_markers": {}, "integrity_gate": json.loads(ig.read_text()).get("pass") if ig.is_file() else None,
           "integrity_json_sha256": sha(ig) if ig.is_file() else None,
           "profiling_timing": {"map": dict(VAL), "files": {k: {"path": str(p), "sha256": sha(p)} for k, p in tf.items()}}}
    main = P40.unit_names(json.loads(Path(rec["lr_choice"]["file"]).read_text()))
    for b in BENCHES:
        bs = summ["benches"][b]
        if sha(L.inputs_file(b)) != bs["inputs_sha256"]:
            raise SystemExit(f"{L.inputs_file(b)} changed since the build")
        exp = {it["qid"] for it in T.read_jsonl_strict(L.inputs_file(b))}; cfg = P40.read_configs(L.configs_file(b))
        if sha(L.configs_file(b)) != rec["benches"][b]["configs_sha256"] or cfg != q_rows(main):
            raise SystemExit(f"{L.configs_file(b)} differs from configs_Q/rows.json or from the 41 rows of Q1")
        cov = {}
        for c in cfg:
            recs, n = SC.load_config(L.pred / b, c); have = set(recs)
            if have != exp:
                raise SystemExit(f"coverage: {b} {c}: {len(have & exp)} / {len(exp)} items, {len(have - exp)} unexpected; integrity {n}")
            cov[c] = {"answers": len(recs), **n}
        out["coverage"][b] = cov; out["shard_markers"][b] = shard_markers(L, b, rec, bs["inputs_sha256"])
        out["inputs"][b] = {"file": str(L.inputs_file(b)), "sha256": bs["inputs_sha256"], "items": len(exp)}
        out["labels_not_opened"][b] = {"file": str(L.labels_file(b)), "sha256_recorded_at_build": bs["labels_sha256"]}
        out["rows"][b] = {"configs_sha256": sha(L.configs_file(b)), "n_rows": len(cfg)}
    files = T.walk_files(L.pred)
    out["files"] = [{"path": str(p.relative_to(L.root)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]; out["n_files"] = len(files)
    T.write_atomic(L.manifest, json.dumps(out, indent=1, sort_keys=True) + "\n"); digest = sha(L.manifest)
    T.write_atomic(L.manifest_sha, f"{digest}  predictions_manifest.json\n")
    return digest, out


# ------------------------------------------------------------------------------------------------------------------------------ score
class QBench(S40.Bench40):
    """r7_d1_40c_score.Bench40 with the items in memory (sealed inputs joined with the sealed labels; nothing written) and the latency rows of the
    mapped validation benchmark's R7-40c timing file (profiling workload, Q2).  Coverage guard, per-item scores, token counts, bootstrap, latency
    lookup (sel rows -> the representative S1 seed-0 unit) and every statistic are the frozen R7-40c / D1-30 code."""

    def __init__(self, bench, items, pred_dir, timing_file, expected, reps, rep_unit):
        self.bench, self.items, self.rep = bench, items, rep_unit
        self.qids = [it["qid"] for it in items]; self.boot = SC.Boot([it["image_id"] for it in items], reps)
        self.S, self.K, self.N, self.F, self.X, self.Xi, self.P, self._bm, self.integrity = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for cfg in expected:
            recs, self.integrity[cfg] = SC.load_config(Path(pred_dir) / bench, cfg); preds = {q: r["pred"] for q, r in recs.items()}
            if set(preds) != set(self.qids):
                raise SystemExit(f"coverage guard: {bench} {cfg}: {len(set(preds) & set(self.qids))} / {len(self.qids)} items, "
                                 f"{len(set(preds) - set(self.qids))} unexpected; integrity {self.integrity[cfg]}")
            sc = SC.item_scores(bench, items, preds)
            self.S[cfg] = np.array([sc[q] for q in self.qids]); self.K[cfg] = np.array([recs[q]["kept"] for q in self.qids], float)
            self.N[cfg] = np.array([recs[q]["n_vis"] for q in self.qids], float); self.F[cfg] = np.array([recs[q]["f"] for q in self.qids], float)
            self.Xi[cfg] = np.array([bool(recs[q].get("hit_max")) for q in self.qids]); self.X[cfg] = int(self.Xi[cfg].sum()); self.P[cfg] = [preds[q] for q in self.qids]
        j = json.loads(Path(timing_file).read_text()); self.timing_meta = {k: j.get(k) for k in ("gpu", "host", "b_star", "primary_batch", "pools", "allow_any_gpu")}
        self.T, self.Tci = {}, {}
        for r in j["rows"]:
            self.T[(r["config"], r["B"])] = r["per_item_ms_median"]; self.Tci[(r["config"], r["B"])] = r.get("per_item_ms_median_ci95")

    def subset(self, mask, reps):
        """The same rows on a subset of items with its own group bootstrap (secondary split contrasts, as phase L)."""
        sb = object.__new__(QBench); m = np.asarray(mask, bool)
        sb.bench, sb.rep, sb.items = self.bench, self.rep, [it for it, k in zip(self.items, m) if k]
        sb.qids = [it["qid"] for it in sb.items]; sb.boot = SC.Boot([it["image_id"] for it in sb.items], reps); sb._bm = {}
        sb.S, sb.K, sb.N, sb.F, sb.Xi = ({c: x[m] for c, x in d.items()} for d in (self.S, self.K, self.N, self.F, self.Xi))
        sb.X = {c: int(x.sum()) for c, x in sb.Xi.items()}; sb.P = {c: [p for p, k in zip(x, m) if k] for c, x in self.P.items()}
        sb.integrity, sb.T, sb.Tci, sb.timing_meta = self.integrity, self.T, self.Tci, self.timing_meta
        return sb


def splits(bn, main, b_star, basis, reps):
    """ChartQA human / augmented (secondary, D50-3 and the R7-50 secondary list, as phase L): quality per row with CIs and the contrasts on each
    split with its own chart bootstrap (unadjusted), plus the mean of the two splits; the pooled 2,500 items are primary."""
    # OPEN: the addendum's secondary list reads 'every R7-40c secondary ... ChartQA human / augmented' (R7-40c: quality per row per split, kept in
    # score_bench's 'subsets'); the R7-50 protocol's endpoints.secondary adds CIs and per-split contrasts with their own chart bootstrap, as phase L.
    lab = np.array([it.get("split") for it in bn.items]); out = {}
    for v in ("human", "augmented"):
        m = lab == v
        if not m.any():
            continue
        sb = bn.subset(m, reps); Rs = S40.score_bench(sb, main, b_star, basis)
        out[v] = {"n": int(m.sum()), "n_charts": sb.boot.G, "quality": {c: x["quality"] for c, x in Rs["fixed_fraction"].items()},
                  "ci95": {c: x["ci95"] for c, x in Rs["fixed_fraction"].items()},
                  "contrasts_secondary": {k: dict(x, role="secondary (subset)") for k, x in Rs["contrasts"].items()}}
    if "human" in out and "augmented" in out:
        out["mean_of_splits"] = {c: 0.5 * (out["human"]["quality"][c] + out["augmented"]["quality"][c]) for c in out["human"]["quality"]}
    return out


def score(root, timing_dir=R40C_TIMING, timing_report=TIMING_REPORT, reps=2000, out_dir=None, require_commit=None, protocol_40c=PROTOCOL_40C):
    """Score phase Q once.  Gates: ALLOW_SCORING_Q with the manifest's sha256 (production root: commit= / manifest_copy= whose committed bytes hash
    to it; r7_50_test.check_gate), every prediction file as in the manifest, rows.json and the timing files as sealed, R7-40c's freeze key.  Basis
    as R7-40c's freeze records it (Q2): Qwen's B* from the R7-40a report (r7_d1_40c_timing.read_b_star; must equal the timing files' b_star) ->
    latency points L25 / L50 / L75 at B*, else fixed fractions 0.25 / 0.5 / 0.75 and no acceleration claim.  r7_d1_40c_score.score_bench per
    benchmark (no anchor row on test), Holm over the primary contrasts of both benchmarks at each point (6: Q4)."""
    L = T.Layout(root, PHASE); man, kv, man_sha = T.check_gate(L, require_commit); S40.require_freeze(protocol_40c)
    summ = T.load_summary(L); od = Path(out_dir) if out_dir else L.results
    if (od / "r7_50_results.json").exists():                                           # refused before any label is opened
        raise SystemExit(f"{od / 'r7_50_results.json'} exists; a re-verification writes to a new --out directory (A.4 case b, recorded)")
    if sha(L.summary) != man["build_summary_sha256"] or sha(L.configs / "rows.json") != man["rows_json_sha256"]:
        raise SystemExit("build summary or configs_Q/rows.json differ from the manifest")
    tf = timing_files(timing_dir, timing_report); sealed = man["profiling_timing"]["files"]
    if set(tf) != set(sealed) or any(sealed[k]["path"] != str(p) or sealed[k]["sha256"] != sha(p) for k, p in tf.items()):
        raise SystemExit("the R7-40c timing / timing-report files differ from the ones sealed in manifest_Q")
    b_star, brec = TMG.read_b_star(Path(timing_report)); basis = "latency" if b_star else "fraction"; pts = S40.LPOINTS if basis == "latency" else S40.FPOINTS
    rec = json.loads((L.configs / "rows.json").read_text())
    if fsha(rec["lr_choice"]["file"]) != rec["lr_choice"]["sha256"]:
        raise SystemExit(f"{rec['lr_choice']['file']} missing or changed since configs_Q")
    main = P40.unit_names(json.loads(Path(rec["lr_choice"]["file"]).read_text())); rep = main["vcs"][0]
    res = {"title": "R7-50 official-test evaluation, phase Q (Qwen2.5-VL-7B-Instruct, R7-40b selectors)", "phase": PHASE, "host": "Qwen2.5-VL-7B-Instruct",
           "scored_utc": T.utc(), "root": str(L.root), "manifest_sha256": man_sha, "allow_scoring": kv, "git": T.git_state(), "addendum": rec.get("addendum"),
           "freeze": summ.get("freeze"), "r7_40c_freeze": {"protocol": str(protocol_40c), "key": S40.FREEZE_KEY}, "representative_latency_unit": rep,
           "b_star": b_star, "b_star_source": brec, "primary_basis": basis, "primary_points": list(pts), "reps": reps, "latency_label": LATENCY_LABEL,
           "exposure_statement": EXPOSURE, "benches": {},
           "sources": {b: {k: summ["benches"][b].get(k) for k in ("mode", "source", "sha256", "recorded", "items", "groups", "split_counts")} for b in BENCHES},
           "notes": ["primary family (Q4, D50-2): S1 − S2, S1 − S3 and S1 − uniform grid × GQA testdev-balanced and ChartQA test = 6 contrasts per point, "
                     "Holm per point within phase Q; the replication reading across hosts rests on S1 − S2 and S1 − S3 (R7-40c decision 5)",
                     "points = R7-40c's basis: L25 / L50 / L75 at Qwen's B* when it has one, otherwise quality at fixed fractions 0.25 / 0.5 / 0.75 and no "
                     "acceleration claim (OPEN-R740-8); the 0.125 rows are the stress point (reported, never a primary point)",
                     "cross-host reading descriptive only: phase L (LLaVA) and phase Q (Qwen) side by side; no pooling, no cross-host test",
                     "ChartQA: all 2,500 items pooled are primary; human / augmented secondary (D50-3, unadjusted)",
                     "no equivalence / non-inferiority claim (no margin set in advance); no anchor row on test (R7-40c's batch-1 official anchor is a "
                     "validation-set diagnostic)"]}
    for b in BENCHES:
        cfg = P40.read_configs(L.configs_file(b))
        if cfg != q_rows(main):
            raise SystemExit(f"{b}: configuration file does not hold the 41 rows of Q1 for {rec['lr_choice']['file']}")
        tfile = tf[f"timing_{VAL[b]}.json"]; bn = QBench(b, T.labelled_items(L, b, summ), L.pred, tfile, cfg, reps, rep)
        if bn.timing_meta.get("b_star") != b_star:
            raise SystemExit(f"{b}: timing file B* {bn.timing_meta.get('b_star')} != the R7-40a report's B* {b_star}")
        if basis == "latency" and not bn.T:
            raise SystemExit(f"{b}: Qwen has a B* but {tfile} has no timing rows: the primary endpoint cannot be computed")
        R = S40.score_bench(bn, main, b_star, basis); Bs = sorted({B for (_, B) in bn.T})
        R["row_latency_profiling_workload_ms"] = {c: {f"B{B}": {"median": bn.lat(c, B), "ci95": bn.Tci.get((T.lat_key(bn, c), B))} for B in Bs if bn.lat(c, B) is not None}
                                                  for c in bn.S}
        R["profiling_timing"] = {"validation_bench": VAL[b], "file": str(tfile), "sha256": sha(tfile), "meta": bn.timing_meta}
        if b.startswith("chartqa"):
            R["splits_secondary"] = splits(bn, main, b_star, basis, reps)
        res["benches"][b] = R
    for P in pts:                                                                       # Holm over the primary family at each point (Q4)
        keys = [(b, k) for b in res["benches"] for k, v in res["benches"][b]["contrasts"].items() if k.endswith(f"@{P}") and v.get("role") == "primary" and "p" in v]
        for (b, k), pv in zip(keys, SC.holm([res["benches"][b]["contrasts"][k]["p"] for b, k in keys]) if keys else []):
            res["benches"][b]["contrasts"][k]["p_holm"] = pv; res["benches"][b]["contrasts"][k]["holm_family_size"] = len(keys)
    if (od / "r7_50_results.json").exists():
        raise SystemExit(f"{od / 'r7_50_results.json'} exists; a re-verification writes to a new --out directory (A.4 case b, recorded)")
    T.write_atomic(od / "r7_50_results.json", V.jdump(res, indent=1)); T.write_atomic(od / "r7_50_results.md", markdown(res))
    return res


def markdown(res) -> str:
    f, ci = T._f, T._ci; add = res.get("addendum") or {}
    L = [f"# {res['title']}", "", f"Scored {res['scored_utc']}; predictions manifest sha256 `{res['manifest_sha256']}` (committed before scoring); addendum "
         f"{add.get('status')} (sha256 {str(add.get('sha256'))[:12]}…).", "",
         f"**Primary basis.** {res['primary_basis']} — Qwen's B* = {res['b_star']} (R7-40a report {res['b_star_source'].get('file')}); points "
         f"{', '.join(res['primary_points'])}; Holm per point over the phase-Q family.", "", f"**Latency.** {res['latency_label']}.", "", "**Exposure.**"]
    L += [f"- {x}" for x in res["exposure_statement"]] + [""] + [f"- {x}" for x in res["notes"]]
    L += ["", "## Primary contrasts (points)", "", "| bench | contrast | Δ | 95 % CI | p | p_holm | family | seed Δ |", "|---|---|---|---|---|---|---|---|"]
    for b, r in res["benches"].items():
        for k, v in r["contrasts"].items():
            if v.get("role") == "primary":
                if v.get("not_reached"):
                    L.append(f"| {b} | {k} | not reached | | | | | |"); continue
                sd = ", ".join("%+.1f" % x for x in v.get("seed_deltas") or [])
                L.append(f"| {b} | {k} | {v['delta']:+.2f} | {ci(v['ci95'])} | {v['p']:.4f} | {f(v.get('p_holm'), 4)} | {v.get('holm_family_size', '')} | {sd} |")
    for b, r in res["benches"].items():
        sp = r.get("splits_secondary") or {}; ch = "human" in sp and "augmented" in sp; ff = r["fixed_fraction"]
        L += ["", f"## {b}: rows (n = {r['n_items']}, {r['n_groups']} groups; full = {ff['full']['quality']:.2f})", ""]
        head = ("| row | visual tokens | kept fraction | quality [95 % CI] |" + (" human | augmented |" if ch else "") +
                " retention (rel.) | cut at max_new | latency B=4 / B=1, ms (profiling workload) |")
        L += [head, "|" + "---|" * (head.count("|") - 1)]
        for c, v in ff.items():
            lat = r["row_latency_profiling_workload_ms"].get(c, {}); l4, l1 = (lat.get("B4") or {}).get("median"), (lat.get("B1") or {}).get("median")
            L.append(f"| {c} | {v['mean_visual_tokens']:.1f} | {v['mean_kept_fraction']:.3f} | {v['quality']:.2f} {ci(v['ci95'])} |"
                     + (f" {sp['human']['quality'][c]:.2f} | {sp['augmented']['quality'][c]:.2f} |" if ch else "")
                     + f" {f(v['retention_rel'], 3)} | {v['answers_cut_at_max_new_tokens']} | {f(l4)} / {f(l1)} |")
        L += ["", "Quality at the latency points (seed mean; B = batch size of the profiling workload):", "", "| B | family | L25 | L50 | L75 |", "|---|---|---|---|---|"]
        for B, fam in r["at_latency"].items():
            for name, p in fam.items():
                L.append(f"| {B} | {name} | " + " | ".join("not reached" if p[x] is None else f"{p[x]['mean']:.2f}" for x in ("L25", "L50", "L75")) + " |")
        L += ["", "Secondary contrasts (unadjusted):", "", "| contrast | Δ | 95 % CI | p |", "|---|---|---|---|"]
        for k, v in r["contrasts"].items():
            if v.get("role") != "primary":
                L.append(f"| {k} | not reached | | |" if v.get("not_reached") else f"| {k} | {v['delta']:+.2f} | {ci(v['ci95'])} | {v['p']:.4f} |")
        for s in ("human", "augmented"):
            if s in sp:
                L += ["", f"ChartQA {s} (n = {sp[s]['n']}; secondary, own chart bootstrap):", "", "| contrast | Δ | 95 % CI | p |", "|---|---|---|---|"]
                for k, v in sp[s]["contrasts_secondary"].items():
                    L.append(f"| {k} | not reached | | |" if v.get("not_reached") else f"| {k} | {v['delta']:+.2f} | {ci(v['ci95'])} | {v['p']:.4f} |")
        L += ["", f"Integrity: {json.dumps(r['integrity']['totals'])}; files per row ≤ {r['integrity']['files_per_config_max']}."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------------------ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        p = sub.add_parser(name); p.add_argument("--root", default=str(ROOT))
        for k, v in kw.items():
            p.add_argument(f"--{k.replace('_', '-')}", **v)
        return p
    sel = {"lr_choice": {"default": str(LR_CHOICE)}, "selectors_dir": {"default": str(SELECTORS)}, "pca": {"default": str(PCA)}}
    tim = {"timing_dir": {"default": str(R40C_TIMING)}, "timing_report": {"default": str(TIMING_REPORT)}}
    add("configs", r40c_configs=dict(default=str(R40C_CONFIGS)), addendum=dict(default=str(ADDENDUM)), **sel)
    add("lines", lines=dict(default=None))
    add("runner-args", bench=dict(required=True, choices=BENCHES), shard=dict(required=True), integrity=dict(action="store_true"))
    add("integrity-prepare", ref_pred=dict(default=str(R40C_PRED)), ref_sets=dict(default=str(D30_SETS)), ref_log=dict(default=str(R40C_LOG)), **sel)
    add("integrity-compare")
    add("manifest", **tim)
    add("score", protocol_40c=dict(default=str(PROTOCOL_40C)), reps=dict(type=int, default=2000), out=dict(default=None), **tim)
    a = ap.parse_args(argv)
    if a.cmd == "configs":
        print(json.dumps(configs(a.root, a.lr_choice, a.selectors_dir, a.pca, None if a.r40c_configs in ("", "none") else a.r40c_configs, a.addendum))); return 0
    if a.cmd == "lines":
        print(f"{write_lines(a.root, a.lines)} units -> {a.lines or T.Layout(a.root, PHASE).lines}"); return 0
    if a.cmd == "runner-args":                                                          # stdout = the runner's arguments only
        print(" ".join(integrity_args(a.root, a.bench, a.shard) if a.integrity else generation_args(a.root, a.bench, a.shard))); return 0
    if a.cmd == "integrity-prepare":
        print(json.dumps(integrity_prepare(a.root, a.lr_choice, a.selectors_dir, a.pca, a.ref_pred, a.ref_sets, a.ref_log))); return 0
    if a.cmd == "integrity-compare":
        return integrity_compare(a.root)
    if a.cmd == "manifest":
        digest, m = manifest(a.root, a.timing_dir, a.timing_report)
        print(json.dumps({"phase": PHASE, "manifest_sha256": digest, "n_files": m["n_files"], "rows": {b: v["n_rows"] for b, v in m["rows"].items()},
                          "items": {b: v["items"] for b, v in m["inputs"].items()}, "timing_files": sorted(m["profiling_timing"]["files"])}, indent=1)); return 0
    r = score(a.root, a.timing_dir, a.timing_report, a.reps, a.out, protocol_40c=a.protocol_40c)
    print(json.dumps({"basis": r["primary_basis"], **{b: {"n_items": v["n_items"], "rows": len(v["fixed_fraction"]),
                                                          "primary_contrasts": sum(c.get("role") == "primary" for c in v["contrasts"].values())}
                                                      for b, v in r["benches"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
