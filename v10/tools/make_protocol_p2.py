#!/usr/bin/env python3
"""V10 protocol P2 generator: owner decision V10-phase1-decisions-20261017, decision section 10 step 1 (protocol landing).

Writes (the only files written):
  v10/protocol/V10_PROTOCOL_P2_20261017.json  versioned machine-readable protocol, status LANDED_NOT_FROZEN (the batch-1 freeze is
                                              decision step 3, after the relative-M1 code passes all T01 parts under its new identity)
  v10/protocol/T03_UNITS.csv                  T03 unit table: 252 CAL candidates + 600 main + 120 extra algorithm seeds + 240 capacity
                                              = 1212 grid rows, plus the smoke rows (block "smoke", outside the 1212)

Derivation.  Every value list is parsed from the decision text v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md (regexes anchored to
its sentences, tables and numbered lists; a changed sentence aborts the run) or recomputed from the phase-1 records, all read as git objects at the
phase-1 results commit named by the decision (11746dd): the v10/T00 manifest (sha256 pinned by MANIFEST_SHA256.txt), v10/T01
{TORCH_VALIDATION, GEOMETRY_GRADIENT_CHECK}.json, v10/T02/{BUDGET, T02_TIMING_cuda}.json, the package (configs/V10_PROTOCOL.json,
V10_DESIGN_CN.md, synthetic/src/v10_synthetic.py by ast) and the phase-1 code (adapter / backend defaults, the REG-P3 audit).  Two
more git objects at fixed commits: the phase-1 report at the reviewed HEAD of decision section 1 (cross-check of the T02 lead-step
figures of decision 2.2) and src/v10/adapter.py at STEP2_CODE_COMMIT (the decision step-2 code: the landing choice of the fp64
diagnostic trigger).  The working tree is read only for the three decision documents (required committed and unmodified), so later
code, result or manifest commits cannot change this output.  The result is then compared
field by field with the owner's decision contract v10/decisions/V10_PROTOCOL_DECISIONS_20261017.json.  Any mismatch aborts before
anything is written; the checks that ran are listed in the output ("derivation_checks", all PASS by construction).
Hand-written: English glosses; the mapping tables between decision wording and machine keys (MAP_* below); the implementation
choices of this landing, listed under "landing_choices" and each labelled as not an owner decision.
Reproducible: no wall-clock time, no live query, no randomness; two runs give the same bytes.  --check rebuilds both outputs in
memory and compares them with the files on disk (exit 1 on any difference) without writing.

    /home/infres/yinwang/CS_QMI/env/bin/python v10/tools/make_protocol_p2.py [--check]
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELF_REL = Path(__file__).resolve().relative_to(REPO).as_posix()
DEC_MD = "v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md"
DEC_SUMMARY = "v10/decisions/V10_PHASE1_DECISIONS_20261017_SUMMARY_CN.md"
DEC_JSON = "v10/decisions/V10_PROTOCOL_DECISIONS_20261017.json"
MANIFEST = "v10/T00/V10_DATA_MANIFEST.csv"
MANIFEST_SHA = "v10/T00/MANIFEST_SHA256.txt"
MANIFEST_SUMMARY = "v10/T00/MANIFEST_BUILD_SUMMARY.json"
TV = "v10/T01/TORCH_VALIDATION.json"
GG = "v10/T01/GEOMETRY_GRADIENT_CHECK.json"
BUDGET = "v10/T02/BUDGET.json"
T02C = "v10/T02/T02_TIMING_cuda.json"
PKG = "v10/package_v10_server_v1_20261016"
PKG_PROTO = f"{PKG}/configs/V10_PROTOCOL.json"
PKG_DESIGN = f"{PKG}/V10_DESIGN_CN.md"
PKG_SYN = f"{PKG}/synthetic/src/v10_synthetic.py"
PHASE1_REPORT = "v10/V10_PHASE1_STAGE_REPORT_20261017.md"
OUT_JSON = "v10/protocol/V10_PROTOCOL_P2_20261017.json"
OUT_UNITS = "v10/protocol/T03_UNITS.csv"
# git objects read at the phase-1 results commit (the decision's "最近的测量结果提交"; checked below)
CODE_AT_RESULTS = ("src/v10/adapter.py", "src/v10/mf_vcs.py", "src/v9/reg/r6_reg_p3.py")
# decision step-2 code (relative M1 rule; review fixes) at a fixed commit: read as a git object, only for the landing choice of the
# fp64 diagnostic trigger (later code commits cannot change this output)
STEP2_CODE_COMMIT = "278abe4556e21c52d1ab974a2d4a948d1cf417a3"
# package text superseded for T03 only (decision 9; quoted verbatim, checked in the design)
PKG_T03_MODE = "U 保留为受控 iid 估计实验和命名清楚的目标敏感性分析，不能悄悄替换 V 来提高结果。"
PKG_SEED_RULE = "主训练seed固定为1001/1002/1003；调参seed为1017。"
PKG_T03_RETAINED = ("包内首先实现balanced two-block与diagonal-mixture的有限可观察表：m=8/32，rho=0/.25/.75；p/q/eta/S可以精确计算。"
                    "FIT/CAL/EVAL分别使用独立随机流，truth单独保存在评估目录，训练器不读取。")
# decision wording of the items frozen before the primary runs (section 10 step 3 and section 5) -> contract
# rollout.freeze_before_primary_runs
MAP_FREEZE = {"manifest": "manifest", "候选顺序": "candidate order", "成本上限": "cost ceilings", "数据": "manifest", "预处理": "preprocessing",
              "seed": "seeds and RNG stream policy", "几何": "geometry", "CAL": "CAL rules", "精度": "precision"}
UNIT_COLUMNS = ("unit_id", "block", "family", "m", "rho", "method", "K", "lr", "cap", "data_root_seed", "algo_seed", "steps",
                "ckpt_every", "selection_rule", "depends_on")

# ------------------------------------------------------------------------------------------------ hand-written maps
# decision wording -> canonical method names (package / BUDGET names; the decision tables use these short forms)
MAP_T03_METHODS = {"MF perm1": "MF_VCS_perm1", "M0": "MF_VCS_M0", "M1": "MF_VCS_M1", "MLP perm1": "MLP_VCS_perm1",
                   "MLP perm8": "MLP_VCS_perm8"}
# decision section 5 table "后续方法" -> package method names (V10_PROTOCOL.json learned / nonlearned) and the contract's
# rollout.remaining_required entries
MAP_LATER = {
    "MLP_VCS_perm8": (["MLP_VCS_perm8"], ["MLP_VCS_perm8 registration"]),
    "InfoNCE_host[J_perm1]": (["InfoNCE_host"], ["InfoNCE_host[J_perm1] adapter"]),
    "MI、NMI": (["MI_native", "NMI_native"], ["MI_native adapter", "NMI_native adapter"]),
    "CoMIR_native": (["CoMIR_native"], ["CoMIR_native adapter and full cost"]),
    "RuLSIF / 核精确矩": (["RuLSIF_kernel"], ["RuLSIF exact moments"]),
    "MIND_2D_adapted": (["MIND_2D_adapted"], ["MIND_2D_adapted"]),
}
# decision section 2.3 per-batch log items -> contract m1.log_fields (the first three are literal names in the text)
MAP_M1_LOG = OrderedDict([("gap_ref_zero", "gap_ref_zero"), ("gap_start", "gap_start"), ("gap_final", "gap_final"),
                          ("相对参考 gap", "gap_relative_to_zero"), ("实际阈值", "tolerance_effective"),
                          ("迭代数", "iterations"), ("停止原因", "stop_reason"), ("HVP 次数", "hvp_calls"),
                          ("矩遍历次数", "moments_passes"), ("W 更新量", "W_update"),
                          ("encoder 梯度抽查", "encoder_gradient_diagnostic")])
M1_LOG_EXTRA = {"converged": "solver flag; true only for gap_tolerance or zero_reference_gap (contract m1.log_fields)"}
MAP_M1_ACCEPT = OrderedDict([
    ("零头正常启动", "zero-head start: from W = 0 with Delta_b != 0 the solver iterates (relative gap 1 > 0.01) and W, J and the "
                    "encoder objective gradient become nonzero"),
    ("warm start 与零头基准不同", "warm start: gap_start = G_b(W_start) is logged separately and the threshold uses G_b(0) of the same "
                               "batch, not gap_start"),
    ("零参考值", "zero reference: Delta_b exactly 0 returns W = 0, stop_reason zero_reference_gap, no division by zero, no zero "
              "tolerance passed to the solver"),
    ("达到上限仍正确返回", "cap reached: returns W and the actual gap with converged false and stop_reason iteration_budget_exhausted "
                      "(require_converged=False)"),
    ("计数正确", "counts: hvp_calls and moments_passes equal the actual Hessian-vector products and moment passes (including any added "
              "for the reference)"),
])
# decision section 3 table row label -> environment record keys (src/v10/adapter.py environment(), same names in T01 / T02 JSON)
MAP_PRECISION = OrderedDict([
    ("PyTorch / CUDA / cuDNN", ("torch", "cuda_runtime", "cudnn")),
    ("GPU", ("gpu_name", "gpu_capability")),
    ("float32 matmul precision", ("float32_matmul_precision",)),
    ("cuDNN convolution fp32 precision", ("cudnn_conv_fp32_precision",)),
    ("原始 cuda_matmul_fp32_precision 字段", ("cuda_matmul_fp32_precision",)),
    ("deterministic algorithms", ("deterministic_algorithms",)),
    ("cuDNN benchmark / deterministic", ("cudnn_benchmark", "cudnn_deterministic")),
    ("CUBLAS_WORKSPACE_CONFIG", ("cublas_workspace_config",)),
])
# environment record key -> contract precision.observed key
MAP_PRECISION_CONTRACT = {"torch": "torch_version", "cuda_runtime": "cuda_version", "cudnn": "cudnn_version", "gpu_name": "gpu_name",
                          "gpu_capability": "compute_capability", "float32_matmul_precision": "float32_matmul_precision",
                          "cudnn_conv_fp32_precision": "cudnn_conv_fp32_precision",
                          "cuda_matmul_fp32_precision": "cuda_matmul_fp32_precision_raw",
                          "deterministic_algorithms": "deterministic_algorithms", "cudnn_benchmark": "cudnn_benchmark",
                          "cudnn_deterministic": "cudnn_deterministic", "cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG"}
# decision section 7 table rows -> T01 check names (adapter_cpu / adapter_cuda) and contract max_abs_errors keys
MAP_GATE = OrderedDict([
    ("fp32 全图 vs 同输入 fp64", ("MLP_dense_full_float32_grads_vs_float64_value_same_inputs", "full_fp32_vs_fp64", False)),
    ("fp32 分块 checkpoint vs 同输入 fp64", ("MLP_dense_blocked_checkpointed_float32_grads_vs_float64_value_same_inputs",
                                           "blocked_fp32_vs_fp64", True)),
    ("fp32 分块 vs fp32 全图", ("MLP_dense_blocked_checkpointed_vs_full_grads_float32", "blocked_fp32_vs_full_fp32", False)),
    ("fp64 分块 vs fp64 全图", ("MLP_dense_blocked_checkpointed_vs_full_grads_float64", None, True)),
])
# decision section 8.1 CAL selection items -> machine keys
MAP_BF_SELECTION = ["cal_success_rate_lt_{t42}_native_px:desc", "cal_success_rate_lt_{t9}_native_px:desc",
                    "cal_median_corner_error_native_px:asc", "checkpoint_step:asc", "candidate_ordinal_within_family:asc"]
# decision section 6.2 EPFL items -> contract data.EPFL.activation_gates (1-based); item 3's "G2/P3 对齐及预处理重放" and item 5's
# host checks together are the contract's gate 5, item 5's full timing is gate 6
MAP_EPFL_GATES = {1: [1], 2: [2], 3: [3, 5], 4: [4], 5: [5, 6]}
EN_EPFL = {1: "read and verify, on the V10 path, hash, decoding, shape, modality order and value range of all 477 pairs",
           2: "replay the registered rules: 33 alignment exclusions, 24 size exclusions, 420 retained windows; no re-selection by V10 "
              "method results",
           3: "3 representatives per category in frozen ID order (9 categories x 3 overlays) plus every flagged exception; replay the "
              "G2/P3 alignment and preprocessing",
           4: "uniformly scaled thumbnail review of cross-category / cross-shape near-duplicate candidates with the originals or metadata; "
              "similarity alone is not duplication; freeze a new group record first if a split changes",
           5: "pass the V10 data gate, warp, geometry gradient, pixel units, effective support and frozen-checkpoint checks, then measure "
              "the full EPFL cost; the old proxy cost is never relabelled as measured"}
EN_STEPS = {1: "materialise the decision as a versioned protocol and configuration; correct the report wording on the gradient reference "
               "and the EPFL threshold; keep the package and the old failures; update candidate counts, the T03 unit table and the data "
               "sensitivity rule",
            2: "implement the relative zero-head M1 rule and its logs plus targeted checks; rerun and merge all T01 parts under the new "
               "code hash; then update the affected M1 T02 rows and BUDGET",
            3: "freeze the batch-1 common protocol, manifest, candidate order and cost ceilings; run BF CAL/DEV for the four families; "
               "write T03 in parallel (generation checks and smoke first, then the frozen grid)",
            4: "complete the baseline adapters, their T01/T02 and frozen attachments; run them under the same common protocol; record "
               "successes, failures, over-budget parts and full cumulative cost",
            5: "activate EPFL separately after the image-level and host gates; enable BF TEST_REPEAT only after the complete V10 method "
               "selection is frozen; label every current result development or repeat benchmark"}

CHECKS: list = []


# ------------------------------------------------------------------------------------------------ helpers
def die(msg: str):
    raise SystemExit(f"[make_protocol_p2] check failed: {msg}")


def check(name: str, cond, detail: str = ""):
    if not cond:
        die(f"{name}" + (f": {detail}" if detail else ""))
    row = {"check": name, "status": "PASS"}
    if detail:
        row["detail"] = detail
    CHECKS.append(row)


def same(name: str, a, b, show: bool = True):
    check(name, a == b, (f"{a!r}" if show else "") if a == b else f"{a!r} != {b!r}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_bytes(rel: str) -> bytes:
    return (REPO / rel).read_bytes()


def read_text(rel: str) -> str:
    return read_bytes(rel).decode("utf-8")


def load_json(rel: str):
    return json.loads(read_text(rel))


def git(*args) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if p.returncode != 0:
        die(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def git_blob(commit: str, path: str) -> bytes:
    p = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    if p.returncode != 0:
        die(f"git cat-file {commit}:{path}")
    return p.stdout


def one(pattern: str, text: str, what: str, flags=re.S):
    ms = list(re.finditer(pattern, text, flags))
    check(f"decision text: {what}", len(ms) == 1, f"{len(ms)} matches for {pattern!r}" if len(ms) != 1 else "")
    return ms[0]


def section(md: str, num: str) -> str:
    """text of '## <num> ...' (or '### <num> ...' when num contains a dot) up to the next heading of the same or a higher level"""
    lvl = "###" if "." in num else "##"
    stop = r"^### |^## " if lvl == "###" else r"^## "
    m = re.search(rf"^{lvl} {re.escape(num)} [^\n]*\n(.*?)(?={stop}|\Z)", md, re.M | re.S)
    if m is None:
        die(f"decision text: section {num} not found")
    return m.group(1)


def md_table(text: str, first_header: str) -> list[list[str]]:
    """rows (cells, stripped) of the markdown table whose header row starts with first_header; exactly one such table"""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("| ") and ln.split("|")[1].strip() == first_header]
    if len(starts) != 1:
        die(f"decision text: {len(starts)} tables with header {first_header!r}")
    i = starts[0] + 2
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    return rows


def numbered(text: str) -> list[str]:
    items = re.findall(r"^(\d+)\. (.+)$", text, re.M)
    check("decision text: numbered list is 1..n", [int(n) for n, _ in items] == list(range(1, len(items) + 1)), f"{len(items)} items")
    return [t.strip() for _, t in items]


def bare(s: str) -> str:
    """a list item without its closing punctuation (；。)"""
    return s.rstrip("；。;.")


def cn_split(s: str) -> list[str]:
    return [x.strip() for x in re.split(r"、|，|, ", s) if x.strip()]


def agrees(x: float, shown: str) -> bool:
    """x rounds to the decimal string shown (half a unit of its last digit), e.g. 0.009221 vs '0.0092' or 3.1758e-5 vs '3.176e-5'"""
    d = Decimal(shown)
    q = Decimal(1).scaleb(d.as_tuple().exponent)
    return abs(Decimal(repr(float(x))) - d) <= q / 2 * Decimal("1.000001")


def lr_text(x: float) -> str:
    return {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}[x]


def gfmt(x) -> str:
    return format(float(x), "g")


# ------------------------------------------------------------------------------------------------ inputs
RECORDS = (MANIFEST, MANIFEST_SHA, MANIFEST_SUMMARY, TV, GG, BUDGET, T02C, PKG_PROTO, PKG_DESIGN, PKG_SYN, f"{PKG}/MANIFEST_SHA256.json")


def load_inputs() -> dict:
    check("decision documents committed and unmodified", git("status", "--porcelain", "--", "v10/decisions").strip() == "")
    for rel in (DEC_MD, DEC_SUMMARY, DEC_JSON):
        check(f"{rel} tracked", git("ls-files", "--error-unmatch", rel).strip() == rel)
    md = read_text(DEC_MD)
    res = one(r"最近的测量结果提交为 `([0-9a-f]{40})`", section(md, "1"), "results commit").group(1)
    rec = {rel: git_blob(res, rel) for rel in RECORDS}
    js = lambda rel: json.loads(rec[rel].decode("utf-8"))  # noqa: E731
    I = {"md": md, "summary": read_text(DEC_SUMMARY), "contract": load_json(DEC_JSON), "results_commit": res, "tv": js(TV), "gg": js(GG),
         "budget": js(BUDGET), "t02c": js(T02C), "pkg_proto": js(PKG_PROTO), "design": rec[PKG_DESIGN].decode("utf-8"), "mbs": js(MANIFEST_SUMMARY),
         "syn_src": rec[PKG_SYN].decode("utf-8"), "pkg_manifest": js(f"{PKG}/MANIFEST_SHA256.json")}
    sha_txt = {}
    for ln in rec[MANIFEST_SHA].decode("utf-8").splitlines():
        if ln.strip():
            h, name = ln.split(None, 1)
            sha_txt[name.strip()] = h
    check("manifest sha256 = v10/T00/MANIFEST_SHA256.txt", sha256_bytes(rec[MANIFEST]) == sha_txt["V10_DATA_MANIFEST.csv"],
          sha_txt["V10_DATA_MANIFEST.csv"][:16])
    I["manifest"] = list(csv.DictReader(io.StringIO(rec[MANIFEST].decode("utf-8"), newline="")))
    check("BUDGET source = T02_TIMING_cuda.json", I["budget"]["source"]["sha256"] == sha256_bytes(rec[T02C]))
    for rel in (PKG_PROTO, PKG_DESIGN, PKG_SYN):
        same(f"{rel} = package MANIFEST_SHA256.json", sha256_bytes(rec[rel]), I["pkg_manifest"][rel[len(PKG) + 1:]], show=False)
    I["inputs"] = OrderedDict([(rel, sha256_bytes(read_bytes(rel))) for rel in (DEC_MD, DEC_SUMMARY, DEC_JSON)]
                              + [(f"{res}:{rel}", sha256_bytes(rec[rel])) for rel in RECORDS])
    return I


# ------------------------------------------------------------------------------------------------ sections
def build_identity(I) -> dict:
    md, C, tv = I["md"], I["contract"], I["tv"]
    s1 = section(md, "1")
    head = one(r"HEAD 为 `([0-9a-f]{40})`", s1, "reviewed HEAD").group(1)
    res = one(r"最近的测量结果提交为 `([0-9a-f]{40})`", s1, "results commit").group(1)
    rows = {r[0]: r[1] for r in md_table(s1, "核对项")}
    code = one(r"`([0-9a-f]{16})`", rows["测量代码 hash"], "code hash").group(1)
    host = one(r"`([0-9a-f]{16})`", rows["宿主 hash"], "host hash").group(1)
    backend = one(r"`([0-9a-f]{64})`", rows["数值后端 SHA256"], "backend sha256").group(1)
    npkg = int(one(r"(\d+) 个文件", rows["原包清单"], "package manifest files").group(1))
    t01 = [int(x) for x in one(r"(\d+) 项；(\d+) 个必需项全部 PASS；(\d+) 个诊断项中 (\d+) PASS、(\d+) FAIL", rows["T01"], "T01 counts").groups()]
    cur = tv["provenance"]["current"]
    same("phase-1 code hash: decision = T01 provenance", code, cur["code_hash_src_scripts_v10"])
    same("phase-1 host hash: decision = T01 provenance", host, cur["host_code_hash"])
    same("phase-1 backend sha256: decision = T01 provenance", backend, cur["mf_vcs_sha256"], show=False)
    gg = I["gg"]
    parts = {**tv["parts"], **gg["parts"]}
    tot = sum(p["n_checks"] for p in parts.values())
    req = sum(p["n_required"] for p in parts.values())
    dpass = sum(1 for p in parts.values() for x in p["checks"] if not x.get("required") and x["status"] == "PASS")
    dfail = sum(1 for p in parts.values() for x in p["checks"] if not x.get("required") and x["status"] != "PASS")
    rfail = sum(1 for p in parts.values() for x in p["checks"] if x.get("required") and x["status"] != "PASS")
    same("T01 counts (total, required PASS, diagnostic, diagnostic PASS, diagnostic FAIL): decision = T01 JSON", t01,
         [tot, req, tot - req, dpass, dfail])
    same("T01 required failures = 0", rfail, 0)
    rp = C["reviewed_provenance"]
    same("contract reviewed head", rp["head"], head, show=False)
    same("contract results commit", rp["measured_results_commit"], res, show=False)
    same("contract code / host hash / backend", [rp["code_hash"], rp["host_code_hash"], rp["backend_sha256"]], [code, host, backend], show=False)
    same("contract package manifest files", rp["package_manifest_matching_files"], npkg)
    same("contract T01 counts", [rp["T01"][k] for k in ("total", "required_pass", "diagnostic_pass", "diagnostic_fail")],
         [t01[0], t01[1], t01[3], t01[4]])
    same("package manifest file count", len(I["pkg_manifest"]), npkg)
    same("results commit: decision section 1 = the commit the records were read at", res, I["results_commit"], show=False)
    # the results commit and the reviewed head exist and are ancestors of HEAD
    for c_ in (res, head):
        check(f"git: {c_[:7]} is an ancestor of HEAD",
              subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", c_, "HEAD"]).returncode == 0)
    return {"reviewed_head": head, "results_commit": res, "code_hash": code, "host_code_hash": host, "backend_sha256": backend,
            "t01": {"total": t01[0], "required": t01[1], "required_pass": t01[1], "diagnostic": t01[2], "diagnostic_pass": t01[3],
                    "diagnostic_fail": t01[4]},
            "code_hash_rule": "sha256 over '<sha256>  <path>' lines of all tracked files of src/v10 + scripts/v10 (paths sorted), first 16 "
                              "hex (src/v10/adapter.py code_hash; job header code=); job pin adds slurm/v10",
            "revalidation_rule": "the relative-M1 change creates a new code identity: rerun and merge every T01 part under it "
                                 "(t01_validate.py merge provenance rule); phase-1 PASS does not carry over; T02 remeasures only the "
                                 "affected M1 configurations and the budget derived from them (decision 2.3, 10)"}


def build_supersedes(I) -> list:
    P, C = I["pkg_proto"], I["contract"]
    rows = [
        {"contract_clause": "M1 primary absolute tolerance", "decision_section": "2",
         "package": {"head.M1_gap_absolute_target": P["head"]["M1_gap_absolute_target"]},
         "p2": "m1_rule: relative zero-head gap, tau_b = 0.01 * G_b(0); strict absolute mode kept for numerical audit only"},
        {"contract_clause": "InfoNCE registration energy label", "decision_section": "4",
         "package": {"registration.learned_methods[InfoNCE_host]": "InfoNCE_host (design section 5: 原生目标/候选归一化)"},
         "p2": "infonce: main-table name InfoNCE_host[J_perm1] with the common J readout"},
        {"contract_clause": "MF K x lr primary tuning grid", "decision_section": "8.1",
         "package": {"head.width_candidates": P["head"]["width_candidates"],
                     "head.encoder_learning_rate": P["head"]["encoder_learning_rate"]},
         "p2": "bf_batch1: K16 fixed, one learning rate tied across all optimiser-learned modules, 18 candidates; K8/K32 only as a named "
               "capacity extension"},
        {"contract_clause": "T03 target mode and repetition grid", "decision_section": "9",
         "package": {"design 2.1 (target mode)": PKG_T03_MODE, "design 7 (seed rule, as applied to T03)": PKG_SEED_RULE},
         "scope": "T03 only.  registration.training_seeds / registration.tuning_seed of the package (natural registration, "
                  f"{P['registration']['training_seeds']} / {P['registration']['tuning_seed']}) stay in force for BF batch 1 (decision "
                  "8.1; bf_batch1.seeds); design 6.3 (conditions, independent FIT / CAL / EVAL streams, truth kept from the trainer, "
                  "oracle-free CAL selection) is retained (decision 9.2)",
         "package_retained_design_6_3": PKG_T03_RETAINED,
         "p2": "t03: U primary, uniform derangements, 10 independent data suites + conditional algorithm seeds, 1212 trainings"},
        {"contract_clause": "fp32 dense-MLP gradient gate role", "decision_section": "7",
         "package": {"design 6.1": "大规模fp32消减错误超限 -> 显式更高精度积累并把成本计入"},
         "p2": "t01_gates: a8f4c5d confirmed (fp32 blocked/checkpointed vs fp64 gradient reference required; full-graph rows diagnostic)"},
        {"contract_clause": "data-role activation details", "decision_section": "6",
         "package": {"registration.dataset_activation": P["registration"]["dataset_activation"],
                     "registration.confirm_policy": P["registration"]["confirm_policy"]},
         "p2": "data_roles / tma_colocation_sensitivity / epfl_activation"},
    ]
    same("contract supersedes clauses = P2 supersedes rows", [r["contract_clause"] for r in rows], C["supersedes"]["clauses"])
    for what, text in (("design 2.1 U sentence", PKG_T03_MODE), ("design 7 seed sentence", PKG_SEED_RULE),
                       ("design 6.3 T03 sentences", PKG_T03_RETAINED)):
        check(f"package {what} quoted verbatim (supersedes: T03)", text in I["design"])
    same("package natural-registration seeds = contract BF batch-1 seeds (not superseded)",
         [P["registration"]["tuning_seed"], P["registration"]["training_seeds"]],
         [C["natural_registration"]["tuning_algorithm_seed"], C["natural_registration"]["final_algorithm_seeds"]])
    same("package protocol M1 absolute target (superseded)", P["head"]["M1_gap_absolute_target"], 1e-4)
    check("package design section 5 InfoNCE row names 原生目标/候选归一化", "| InfoNCE宿主实现 | 相同局部观测与对应数据 | 原生目标/候选归一化 |" in I["design"])
    return rows


def build_m1(I) -> dict:
    md, C = I["md"], I["contract"]["m1"]
    s21, s22, s23 = section(md, "2.1"), section(md, "2.2"), section(md, "2.3")
    rel = float(one(r"\\tau_b=([0-9.]+)\\,g_\{\\mathrm\{ref\},b\}", s21, "tau_b factor").group(1))
    one(r"g_\{\\mathrm\{ref\},b\}=G_b\(0\)=\\\|\\Delta_b\\\|_1", s21, "g_ref = G_b(0) = ||Delta_b||_1")
    latex = " ".join(one(r"\\\[\n(.*?)\n\\\]", s21, "section 2.1 display math").group(1).split())
    one(r"\\Delta_b=\\nabla F_b\(0\)", s21, "Delta_b = grad F_b(0)")
    caps = [int(x) for x in cn_split(one(r"CAL 候选上限为 \*\*([^*]+)\*\*", s21, "M1 caps").group(1))]
    rc = one(r"主训练保持 `require_converged=(False|True)`", s21, "require_converged").group(1) == "True"
    one(r"这里的基准是\*\*当前批次 W=0 的 gap\*\*，不是当前 warm start 的 gap", s21, "reference = current-batch W=0 gap")
    one(r"不要用随机非零初始化、M0 预训练或绝对容差下限改变这项决定。数值审计保留原有严格绝对容差模式。", s21, "forbidden alternatives")
    logline = one(r"逐批记录 (`gap_ref_zero`、`gap_start`、`gap_final`、相对参考 gap、实际阈值、迭代数、停止原因、HVP 次数和矩遍历次数)。"
                  r"保留 (W 更新量)与 (encoder 梯度抽查)。", s23, "per-batch log items")
    items = [x.strip("`") for x in re.split(r"、|和", logline.group(1))] + [logline.group(2).strip(), logline.group(3).strip()]
    same("M1 log items: decision text = MAP_M1_LOG keys", items, list(MAP_M1_LOG.keys()))
    log_fields = list(MAP_M1_LOG.values()) + list(M1_LOG_EXTRA)
    same("M1 log fields: P2 = contract (as sets)", sorted(log_fields), sorted(C["log_fields"]))
    acc = cn_split(one(r"新增有针对性的验收：([^。]+)。", s23, "targeted acceptance list").group(1))
    same("M1 targeted acceptance: decision text = MAP_M1_ACCEPT keys", acc, list(MAP_M1_ACCEPT.keys()))
    one(r"新代码下重跑全部 T01 part，重新合并 provenance；T02 只重测受影响的 M1 配置和由此改变的预算。", s23, "revalidation sentence")
    one(r"若精确计算的 \\\(\\Delta_b=0\\\)，明确返回 W=0 和 `zero_reference_gap`", s23, "zero_reference_gap rule")
    one(r"从同一个已构建的 `QuadraticHead\.delta` 取得参考值并求解", s23, "reference from QuadraticHead.delta")
    for k, v in (("relative_tolerance", rel), ("max_iterations_CAL", caps), ("require_converged", rc), ("reference_is_warm_start_gap", False),
                 ("initial_W", "zeros"), ("absolute_floor", False), ("strict_absolute_mode_retained_for_numerical_audit", True),
                 ("final_FIT_refit", False), ("encoder_W_checkpoint_pair_indivisible", True), ("new_code_requires_all_T01_parts", True),
                 ("reuse_same_quadratic_snapshot_for_reference_and_solve", True), ("ridge", 0)):
        same(f"contract m1.{k}", C[k], v)
    same("contract m1 zero_reference status", C["zero_reference_behavior"]["status"], "zero_reference_gap")
    same("contract m1 zero_reference: no zero tolerance to the solver", C["zero_reference_behavior"]["do_not_pass_zero_tolerance_to_existing_solver"], True)
    # phase-1 code facts (git objects at the results commit)
    res = I["results_commit"]
    ad = git_blob(res, "src/v10/adapter.py").decode()
    mf = git_blob(res, "src/v10/mf_vcs.py").decode()
    tol0 = float(one(r"^M1_PROTOCOL_TOLERANCE = ([0-9.e+-]+)", ad, "phase-1 adapter M1_PROTOCOL_TOLERANCE", re.M).group(1))
    same("phase-1 M1 absolute tolerance = package head.M1_gap_absolute_target", tol0, I["pkg_proto"]["head"]["M1_gap_absolute_target"])
    check("phase-1 backend solver rejects a non-positive tolerance", 'raise ValueError("tolerance must be finite and positive")' in mf)
    check("phase-1 backend: QuadraticHead.delta is the W-gradient at W = 0", "self.delta = scenes[0].a.new_zeros(self.shape)" in mf
          and "return (self.delta * w).sum() - 0.5 * (w * hw).sum(), self.delta - hw" in mf)
    st = re.findall(r'"(gap_tolerance|iteration_budget_exhausted)"', mf)
    same("phase-1 backend solver status strings", sorted(set(st)), ["gap_tolerance", "iteration_budget_exhausted"])
    # evidence quoted by decision 2.2, recomputed from T01
    tvp = {k: {x["name"]: x for x in I["tv"]["parts"][k]["checks"]} for k in ("adapter_cpu", "adapter_cuda")}
    a = tvp["adapter_cpu"]["MF_M1_protocol_tolerance_from_W0_moves_W_and_trains_encoders"]
    cb = tvp["adapter_cpu"]["MF_M1_cap_binding_step_logs_gap_iterations_hvp"]
    cw = tvp["adapter_cpu"]["MF_M1_cap_binding_W_is_inner_solution_nonzero_in_box"]
    eg = tvp["adapter_cpu"]["MF_M1_encoder_grad_nonzero_and_params_change"]
    g0 = one(r"真实宿主初始化的零头 gap 为约 \\\(([\d.]+)\\times10\^\{(-\d+)\}\\\)", s22, "host-init zero-head gap")
    g0s = f"{g0.group(1)}e{g0.group(2)}"
    wmax = one(r"强制 30 次迭代后 W 的最大绝对值约 ([\d.]+)", s22, "forced-30 |W|max").group(1)
    egm = one(r"encoder 梯度范数约 \\\(([\d.]+)\\times10\^\{(-\d+)\}\\\)", s22, "forced-30 encoder gradient norm")
    g30 = one(r"降至 \\\(([\d.]+)\\times10\^\{(-\d+)\}\\\)，下降约 ([\d.]+)%", s22, "forced-30 gap and decrease")
    drop = 100 * (cb["m1_gap_initial"] - cb["m1_gap"]) / cb["m1_gap_initial"]
    lead = one(r"T02 旧规则的前 (\d+) 步为零次迭代，第 (\d+) 步", s22, "T02 degenerate lead")
    ev = {"gap_ref_zero_host_init_cpu": a["m1_gap_initial_equals_Delta_l1"],
          "gap_ref_zero_host_init_cuda": tvp["adapter_cuda"]["MF_M1_protocol_tolerance_from_W0_moves_W_and_trains_encoders"]["m1_gap_initial_equals_Delta_l1"],
          "old_absolute_tolerance": a["m1_tolerance"], "old_rule_iterations": a["m1_iterations"], "old_rule_w_absmax": a["w_absmax"],
          "old_rule_encoder_grad_norm": a["encoder_grad_norm"], "forced30_gap_start": cb["m1_gap_initial"], "forced30_gap_final": cb["m1_gap"],
          "forced30_gap_decrease_percent": drop, "forced30_w_absmax": cw["w_absmax"],
          "forced30_encoder_grad_norm_postclip": eg["encoder_grad_norm_postclip"], "source": f"{TV} at {res[:7]}"}
    head = one(r"HEAD 为 `([0-9a-f]{40})`", section(md, "1"), "reviewed HEAD (phase-1 report evidence)").group(1)
    rep_bytes = git_blob(head, PHASE1_REPORT)
    rep = rep_bytes.decode("utf-8")
    I["inputs"][f"{head}:{PHASE1_REPORT}"] = sha256_bytes(rep_bytes)
    rr = one(r"逐步文件 `train_steps_per_step\.json` sha256 ([0-9a-f]{16})…）：前 (\d+) 步（恰为全部 \d+ 个 warmup 步，不在计时窗口内）"
             r"全部 0 次迭代、W = 0、J = 0，[^；]*；第 (\d+) 步（index (\d+)）", rep, "phase-1 report 6.1: T02 record-row lead steps")
    same("decision 2.2 T02 lead steps = phase-1 report 6.1 (reviewed HEAD)", [int(lead.group(1)), int(lead.group(2))],
         [int(rr.group(2)), int(rr.group(3))])
    same("phase-1 report 6.1: first active index = step - 1", int(rr.group(4)), int(rr.group(3)) - 1)
    ev_text = {"t02_record_row_degenerate_lead_steps": int(lead.group(1)), "t02_record_row_first_active_step_1based": int(lead.group(2)),
               "per_step_record": "outputs_v10/T02/runs/cuda_1043695/train_steps_per_step.json (outside the repository)",
               "per_step_record_sha256_prefix": rr.group(1),
               "source": f"decision 2.2 text; the same two figures in the phase-1 report section 6.1 at the reviewed HEAD {head[:7]} "
                         "(rendered there from the per-step record, whose sha256 that report pins only as the 16-hex prefix "
                         f"{rr.group(1)}...; the full hash is in no committed record); checked equal here"}
    check("decision 2.2 host-init gap agrees with T01", agrees(ev["gap_ref_zero_host_init_cpu"], g0s), f"{ev['gap_ref_zero_host_init_cpu']!r} ~ {g0s}")
    check("decision 2.2 forced-30 |W|max agrees with T01", agrees(cw["w_absmax"], wmax), f"{cw['w_absmax']!r} ~ {wmax}")
    check("decision 2.2 encoder gradient norm agrees with T01", agrees(eg["encoder_grad_norm_postclip"], f"{egm.group(1)}e{egm.group(2)}"))
    check("decision 2.2 forced-30 final gap agrees with T01", agrees(cb["m1_gap"], f"{g30.group(1)}e{g30.group(2)}"))
    check("decision 2.2 gap decrease percent agrees with T01", agrees(drop, g30.group(3)), f"{drop!r} ~ {g30.group(3)}")
    check("T01 old-rule row: 0 iterations, W = 0, encoder gradient 0", a["m1_iterations"] == 0 and a["w_absmax"] == 0.0 and a["encoder_grad_norm"] == 0.0)
    same("decision 2.2 old tolerance = T01 row tolerance", a["m1_tolerance"], tol0)
    same("decision 2.2 T02 lead: first active step = lead + 1", int(lead.group(2)), int(lead.group(1)) + 1)
    return {
        "decision_section": "2", "report_option": "1A",
        "objective": "F_b(W): head objective of the current batch, on the current fixed features, with the same negative reference",
        "definitions": {"Delta_b": "grad_W F_b(0) = QuadraticHead.delta of the batch's own snapshot",
                        "gap_ref_zero": "G_b(0) = ||Delta_b||_1 (box FW gap at W = 0)",
                        "tau_b": f"{rel!r} * gap_ref_zero", "latex_decision_2_1": latex},
        "relative_tolerance": rel,
        "stop_rule": "stop with stop_reason gap_tolerance when G_b(W) <= tau_b; otherwise iterate to the cap (stop_reason "
                     "iteration_budget_exhausted, converged false, actual gap kept: approximate head)",
        "max_iterations_CAL": caps,
        "require_converged_training": rc,
        "reference": {"is": "G_b(0) of the current batch (W = 0)", "is_not": "the gap at the warm start",
                      "W_first_batch": "zeros", "W_later_batches": "warm start from the previous training batch's W",
                      "warm_start_gap": "logged separately as gap_start"},
        "forbidden_alternatives": ["random nonzero W initialisation", "M0 pretraining of W", "absolute tolerance floor"],
        "strict_absolute_mode": "kept for the numerical audit only (absolute tolerance, require_converged=True); never the training rule",
        "zero_reference_gap": {"condition": "Delta_b exactly zero (every entry == 0)", "W": "zeros", "stop_reason": "zero_reference_gap",
                               "converged": True, "gap_relative_to_zero": None,
                               "rules": ["no division by zero", "no zero tolerance passed to the solver (the phase-1 backend rejects "
                                         "tolerance <= 0)", "W = 0 is the maximiser of this negative-semidefinite quadratic empirical "
                                         "objective; it is not population independence"]},
        "near_fp32_resolution": "keep a same-input fp64 numerical diagnostic for references near fp32 resolution; no global switch of "
                                "training to fp64 without an actual failure",
        "reference_source": "the same constructed QuadraticHead.delta is used for the reference and the solve (no extra moment pass for "
                            "the reference); any added moment pass or HVP updates the counters and the cost",
        "log_fields_per_batch": [{"field": v, "decision_item": k} for k, v in MAP_M1_LOG.items()]
                                + [{"field": k, "note": v} for k, v in M1_LOG_EXTRA.items()],
        "log_field_rules": {"tolerance_effective": "tau_b passed to the solver (null for zero_reference_gap)",
                            "gap_relative_to_zero": "gap_final / gap_ref_zero (null for zero_reference_gap)",
                            "stop_reason": ["gap_tolerance", "iteration_budget_exhausted", "zero_reference_gap"],
                            "gap_scope": "current fixed-feature empirical head objective before the outer feature update of this step"},
        "targeted_acceptance_new": [{"decision_item": k, "check": v, "status": "NOT_RUN"} for k, v in MAP_M1_ACCEPT.items()],
        "revalidation": {"t01": "rerun every T01 part under the new code hash and re-merge provenance", "t02": "remeasure only the "
                         "affected M1 configurations and the budget they change", "status": "NOT_RUN"},
        "checkpoint": "encoder, feature maps and W saved as one indivisible checkpoint; no final FIT refit in the main protocol; "
                      "registration keeps the model fixed and keeps geometry gradients",
        "interpretation": "the gap certifies only the sub-optimality of the fixed-feature empirical head problem; it bounds neither the "
                          "encoder gradient error nor the population S error",
        "phase1_code": {"commit": res, "M1Config.tolerance_absolute": tol0, "solver_status_strings": ["gap_tolerance", "iteration_budget_exhausted"]},
        "evidence_phase1": ev,
        "evidence_t02_record_row": ev_text,
        "evidence_scope": "the relative rule removes the batch-noise-dependent start; the forced-30 decrease (about "
                          f"{round(drop, 2)}%) does not show that tau_b is reachable within 100 iterations; the T02 cap-bound gap "
                          "medians of the phase-1 report are absolute gaps, not per-batch relative zero-head gaps",
    }


def build_precision(I) -> dict:
    md, C = I["md"], I["contract"]["precision"]
    s3 = section(md, "3")
    rows = {r[0]: r[1] for r in md_table(s3, "项目")}
    env = I["t02c"]["environment"]
    cuda_parts = {k: p["provenance"] for k, p in {**I["tv"]["parts"], **I["gg"]["parts"]}.items() if k.endswith("_cuda")}
    observed = OrderedDict()
    for label, keys in MAP_PRECISION.items():
        cell = rows[label]
        if label == "PyTorch / CUDA / cuDNN":
            vals = [x.strip() for x in cell.split("/")]
            vals[2] = int(vals[2])
        elif label == "GPU":
            m = one(r"^(.+)，compute capability ([\d.]+)$", cell, "GPU row", 0)
            vals = [m.group(1), m.group(2)]
        elif label == "cuDNN benchmark / deterministic":
            vals = [x.strip() == "true" for x in cell.split("/")]
        elif label == "deterministic algorithms":
            vals = [cell == "true"]
        elif label == "原始 cuda_matmul_fp32_precision 字段":
            vals = [one(r"字符串 `([^`]+)`；按原值保存，不改写为 ieee，也不把该字符串当设置值", cell, "raw cuda matmul field", 0).group(1)]
        else:
            vals = [one(r"^`([^`]+)`$", cell, f"{label} cell", 0).group(1)]
        for k, v in zip(keys, vals):
            observed[k] = v
            same(f"precision {k}: decision = T02_TIMING_cuda.json environment", env[k], v)
            for part, pv in sorted(cuda_parts.items()):
                same(f"precision {k}: decision = T01 {part} provenance", pv[k], v)
            same(f"precision {k}: decision = contract", C["observed"][MAP_PRECISION_CONTRACT[k]], v)
    ov = one(r"下一次必须记录 `([A-Z0-9_]+)` 和 `([A-Z0-9_]+)`，不得假定为空", rows["两项 TF32 环境覆盖变量"], "override variables", 0).groups()
    same("override variables: decision = contract", list(ov), C["previously_unrecorded_must_capture"])
    for v in ov:
        check(f"phase-1 records lack {v} (decision: 旧记录未覆盖)", v not in env and all(v not in pv for pv in cuda_parts.values()))
    one(r"按 2\.9 的同一套 API 管理并在全部导入完成后记录有效值，不混用新旧 TF32 开关。", s3, "API rule")
    res = I["results_commit"]
    ad = git_blob(res, "src/v10/adapter.py").decode()
    fn = one(r"def environment\(device=None\) -> dict:\n(.*?)\n    return env\n", ad, "phase-1 environment()").group(1)
    rec_keys = re.findall(r'[{,]\s*"([a-z0-9_]+)":', fn)
    check("phase-1 environment() records every decision-table key", all(k in rec_keys for k in observed), f"{len(rec_keys)} keys")
    same("contract: do not assume unrecorded env unset", C["do_not_assume_unrecorded_env_unset"], True)
    same("contract: avoid mixing TF32 APIs", C["avoid_mixing_old_and_new_tf32_apis"], True)
    return {
        "decision_section": "3", "report_option": "2A",
        "policy": "every Torch method of the main tables runs under this one recorded effective configuration; the protocol pins the "
                  "actual versions and switches, never 'torch default'",
        "observed_phase1": observed,
        "observed_sources": [T02C + " environment", "T01 CUDA part provenance in TORCH_VALIDATION.json / GEOMETRY_GRADIENT_CHECK.json (" + ", ".join(sorted(cuda_parts)) + ")"],
        "record_contract": {
            "when": "after all imports, in every job, before the first measured step",
            "keys": rec_keys + list(ov),
            "existing_keys_source": f"src/v10/adapter.py environment() at {res[:7]}",
            "override_env_vars": {"names": list(ov), "phase1_recorded": False,
                                  "rule": "record the raw value, or JSON null when the variable is unset; the key must be present; "
                                          "never assume unset"},
            "raw_field_rule": "cuda_matmul_fp32_precision is stored as the raw string (phase 1: 'none'); it is not rewritten to 'ieee' "
                              "and not used as a setting value",
            "api_rule": "manage precision through one torch 2.9 API set and record effective values; do not mix the old and new TF32 "
                        "switches; 'highest' matmul precision does not imply IEEE convolutions",
        },
        "diagnostic_only_variants": ["IEEE cuDNN convolution (named numerical diagnostic)", "fp64 numerical reference"],
        "native_systems": "an external system's extra precision constraint is named and costed with the method",
        "no_extra_grid": "no complete scientific grid is opened for IEEE / fp64 in this batch",
        "evidence_scope": "the geometry diagnostic shows lower numerical error with IEEE convolutions in that comparison; it does not show "
                          "that TF32 caused the V9 registration non-reproducibility or that it changes task accuracy",
    }


def build_infonce(I) -> dict:
    md, C = I["md"], I["contract"]["infonce"]
    s4 = section(md, "4")
    name = one(r"主表命名为 \*\*`([^`]+)`\*\*", s4, "InfoNCE name").group(1)
    m = one(r"表注必须写明：这是“([^”]+)”，与原包中“([^”]+)”的表述不同。", s4, "InfoNCE table note")
    same("InfoNCE name: decision = contract", name, C["main_table_name"])
    check("package design names 原生目标/候选归一化 (decision quotes it with spaces)", m.group(2).replace(" ", "") in I["design"])
    one(r"配准使用 `J\(tanh f\)`，普通 perm1，保持同场景参考；一次配准内固定负排列、搜索支撑与比较规则。", s4, "InfoNCE registration")
    one(r"不能将已测 perm1 成本移作原生能量成本。", s4, "native cost rule")
    return {
        "decision_section": "4", "report_option": "3A", "main_table_name": name,
        "training": "the V9 host InfoNCE objective", "registration_energy": "J(tanh f), ordinary perm1, within-scene reference; negative "
        "permutation, search support and comparison rule fixed within one registration",
        "checkpoint": "the same complete frozen checkpoint; no critic or W retraining at test time",
        "table_note_cn": f"{name}：{m.group(1)}；与原包中“{m.group(2)}”的表述不同。",
        "table_note_en": C["table_disclosure"],
        "scope": "answers the effect of the training objective under the common readout; does not represent native InfoNCE "
                 "registration systems",
        "native_variant": {"name": "a separately named variant (not defined yet)", "status": C["native_InfoNCE_registration_variant"]["status"],
                           "blocks_batch1": False, "requires": C["native_InfoNCE_registration_variant"]["requires"],
                           "cost_rule": "the measured perm1 cost is never moved to the native energy"},
    }


def build_baselines(I) -> dict:
    md, C = I["md"], I["contract"]["rollout"]
    s5 = section(md, "5")
    batch1 = cn_split(one(r"第一批为 \*\*([^*]+)\*\*", s5, "batch-1 families").group(1))
    same("batch 1: decision = contract", batch1, C["first_natural_families"])
    B = I["budget"]["per_unit_projection"]
    for mth in batch1:
        same(f"batch 1 {mth}: phase-1 BUDGET status", B[mth]["status"], "measured")
    later = []
    contract_items = []
    pkg_methods = I["pkg_proto"]["registration"]["learned_methods"] + I["pkg_proto"]["registration"]["nonlearned_methods"]
    for label, req in md_table(s5, "后续方法"):
        check(f"later baseline {label!r} mapped", label in MAP_LATER)
        names, citems = MAP_LATER[label]
        contract_items += citems
        for n in names:
            check(f"later baseline {n} is a package method", n in pkg_methods)
            st = B.get(n, {}).get("status", "not_in_budget")
            later.append({"method": n, "decision_label": label, "requirement_cn": req,
                          "phase1": {"MLP_VCS_perm8": "training implemented and timed (T02 train step); registration energy not implemented"}.get(
                              n, "not implemented in V10"),
                          "phase1_budget_status": st, "t01": "NOT_RUN", "t02": "NOT_RUN", "frozen_attachment": "NOT_RUN"})
    same("later baselines: decision table = contract remaining_required", contract_items, C["remaining_required"])
    one(r"每个新适配器先做对应 T01 核对与 T02 完整计时，再按冻结的数据、预处理、seed、几何、CAL 和精度规则补跑。", s5, "adapter gate")
    one(r"必需基线补齐之前不发布方法优劣、强基线优势或新颖性比较结论。", s5, "no comparative conclusions")
    one(r"完整方法集合和方法内选择全部固定以后才能启用 TEST 重复块。", s5, "TEST repeat activation")
    same("contract no comparative conclusions", C["no_comparative_conclusions_until_required_baselines_complete"], True)
    start_cn = one(r"(完成新 M1 验收和本文件所列协议落地后，可以开展 BF 的 CAL 与 DEV 工作；其余基线同步实施。)", s5, "batch-1 start").group(1)
    step3_cn = one(r"(冻结首批公共协议、[^；]+；开展 BF 的四族 CAL/DEV。)", section(md, "10"), "step 3: freeze then BF CAL/DEV").group(1)
    check("contract server_next_steps[2]: freeze first, then BF CAL/DEV",
          I["contract"]["server_next_steps"][2].startswith("Freeze first-batch common protocol; run BF CAL/DEV"))
    return {
        "decision_section": "5", "report_option": "4B",
        "batch1": batch1,
        "batch1_start_condition": "after the new M1 acceptance (all T01 parts under the new code hash), this protocol landing and the "
                                  "decision step-3 freeze of bf_batch1.freeze_before_primary_runs (contract rollout.freeze_before_primary_runs); "
                                  "then BF CAL and DEV work may start while the other baselines are implemented",
        "batch1_start_condition_sources": ["decision 5: " + start_cn, "decision 10 step 3: " + step3_cn,
                                           "contract server_next_steps[2]: " + I["contract"]["server_next_steps"][2]],
        "later": later,
        "every_new_adapter": ["targeted T01 checks", "T02 complete timing", "frozen attachment (implementation note and predefined "
                              "candidates within the cost ceiling)", "rerun under the frozen data, preprocessing, seed, geometry, CAL and "
                              "precision rules"],
        "conclusion_rule": "progress, failures and actual cost may be reported per method; no method ranking, strong-baseline advantage "
                           "or novelty conclusion before the required baselines are complete",
        "after_global_freeze": "new methods may add their own implementation notes and predefined candidates; the common rules never "
                               "change because of batch-1 DEV results; a material bug in a common implementation bumps the protocol "
                               "version and triggers matched reruns of the affected methods",
        "rulsif_v9_rff_perm1_is_not_the_exact_moment_baseline": True,
        "rulsif_t03_followup": "RuLSIF exact moments also enter T03 as the quadratic-risk comparison once implemented and checked",
        "calibration_extensions": {"methods": I["pkg_proto"]["registration"]["calibrated_extension_methods"],
                                   "rule": "registered separately; not blocking batch 1"},
        "test_repeat_requires": "complete method set and every within-method selection fixed",
    }


def manifest_sets(I) -> dict:
    rows = [r for r in I["manifest"] if r["dataset"] == "BF_SHG"]
    pos = lambda r: r["record_id"].split("_", 1)[1]  # noqa: E731
    fitcal = [r for r in rows if r["inclusion"] == "include" and r["role"] in ("FIT", "CAL")]
    dev = [r for r in rows if r["inclusion"] == "include" and r["role"] == "DEV" and r["original_split"] != "test"]
    test = [r for r in rows if r["original_split"] == "test"]
    return {"rows": rows, "pos": pos, "fitcal": fitcal, "dev": dev, "test": test,
            "roles": {k: sum(1 for r in rows if r["inclusion"] == "include" and r["role"] == k and r["original_split"] != "test")
                      for k in ("FIT", "CAL", "DEV")}}


def build_data_roles(I) -> dict:
    md, C = I["md"], I["contract"]["data"]
    s61 = section(md, "6.1")
    M = manifest_sets(I)
    fit, cal, dev = (int(x) for x in one(r"当前维持 \*\*FIT(\d+)、CAL(\d+)、DEV(\d+)\*\*", s61, "roles").groups())
    ntest = int(one(r"(\d+) 个官方 TEST core 已由 G1 读过，当前仍 HOLD", s61, "TEST count").group(1))
    nrep = int(one(r"启用单独的 TEST_REPEAT(\d+) 表", s61, "TEST_REPEAT").group(1))
    npool = int(one(r"不合并成 (\d+) 个独立确认样本", s61, "no pooling").group(1))
    one(r"当前没有独立 CONFIRM；这不阻止开发和重复基准工作。", s61, "CONFIRM 0")
    same("roles FIT/CAL/DEV: decision = manifest (BF_SHG include, non-test)", [fit, cal, dev], [M["roles"]["FIT"], M["roles"]["CAL"], M["roles"]["DEV"]])
    same("roles: decision = contract", [fit, cal, dev, 0], [C["roles"][k] for k in ("FIT", "CAL", "DEV", "CONFIRM")])
    same("TEST: decision = manifest original_split=test", [ntest, nrep], [len(M["test"]), len(M["test"])])
    check("TEST rows are HOLD in the manifest (role DEV, inclusion exclude)", all(r["role"] == "DEV" and r["inclusion"] == "exclude" for r in M["test"]))
    same("DEV + TEST = the forbidden pooled count", npool, dev + ntest)
    same("contract TEST_REPEAT core count / status", [C["TEST_REPEAT"]["core_count"], C["TEST_REPEAT"]["current_status"]], [ntest, "HOLD_EXCLUDED"])
    check("manifest has no CONFIRM role", not any(r["role"] == "CONFIRM" for r in I["manifest"]))
    same("phase-1 manifest summary roles", I["mbs"]["bf_shg"]["roles"], {"CAL": cal, "FIT": fit, "DEV": dev})
    return {
        "decision_section": "6.1", "report_option": "5A now, 5B after the freeze",
        "BF_SHG": {"FIT": fit, "CAL": cal, "DEV": dev, "DEV_role": "development main table (DEV32)",
                   "TEST": {"cores": ntest, "status": "HOLD", "this_workflow": "never read",
                            "prior_exposure": C["TEST_REPEAT"]["prior_exposure"],
                            "activation": f"TEST_REPEAT{nrep}: a separate repeated-benchmark table, only after the complete V10 freeze "
                                          "(common protocol, complete method set and every within-method selection), with a new "
                                          "manifest version",
                            "pooling": f"DEV and TEST_REPEAT are never pooled into {npool} independent confirmation samples"},
                   "CONFIRM": 0, "CONFIRM_note": "no independent CONFIRM source; development and repeat-benchmark work are not blocked; "
                                                 "every current result is labelled development or repeat benchmark"},
        "EPFL_RGB_NIR": {"status": I["contract"]["data"]["EPFL"]["current_status"], "this_workflow": "not read (see epfl_activation)"},
        "statistics": {"unit": "core", "paired_across_methods": True, "perturbations": "the 10 perturbations of a core are not independent "
                       "tissue samples", "sensitivity": "also report a conservative resampling over TMA position groups (see "
                       "tma_colocation_sensitivity)",
                       "no_wholesale_deletion": "DEV co-located with the old TEST is not FIT/CAL leakage; the DEV cores co-located with "
                                                "any other role are not deleted wholesale",
                       "if_same_tissue_confirmed": "the next protocol version defines a formal group split"},
        "manifest": {"path": MANIFEST, "commit": I["results_commit"], "sha256": I["inputs"][f"{I['results_commit']}:{MANIFEST}"]},
    }


def build_tma(I) -> dict:
    md, C = I["md"], I["contract"]["data"]["BF_sensitivity"]
    s61 = section(md, "6.1")
    M = manifest_sets(I)
    pos = M["pos"]
    one(r"按 record_id 的下划线后部分作为 TMA 位置，排除与当前 FIT/CAL 同位的评估 core。", s61, "position rule")
    same("contract position rule string", C["position_rule"], "record_id.split('_', 1)[1]")
    same("contract FIT/CAL filter string", C["fitcal_filter"], "dataset=BF_SHG AND inclusion=include AND role in {FIT,CAL}")
    same("contract DEV filter string", C["DEV_filter"], "dataset=BF_SHG AND inclusion=include AND role=DEV AND original_split!=test")
    same("contract TEST filter string", C["TEST_REPEAT_metadata_filter"], "dataset=BF_SHG AND original_split=test; does not activate image access")
    fcpos = sorted({pos(r) for r in M["fitcal"]})
    same("FIT/CAL position count: manifest = contract", len(fcpos), C["fitcal_position_count"])
    table = {r[0]: r for r in md_table(s61, "评估块")}
    out = {}
    for block, rows_, key in (("DEV", M["dev"], "DEV"), ("TEST_REPEAT", M["test"], "TEST_REPEAT，未来激活")):
        full = sorted(r["record_id"] for r in rows_)
        excl = sorted(r["record_id"] for r in rows_ if pos(r) in set(fcpos))
        keep = sorted(r["record_id"] for r in rows_ if pos(r) not in set(fcpos))
        groups_full = defaultdict(list)
        for r in sorted(rows_, key=lambda r: r["record_id"]):
            groups_full[pos(r)].append(r["record_id"])
        groups_keep = {p: ids for p, ids in groups_full.items() if p not in set(fcpos)}
        tf = [int(x) for x in re.findall(r"\d+", table[key][1])]
        ts = [int(x) for x in re.findall(r"\d+", table[key][2])]
        same(f"{block} full cores / position groups: decision table = manifest", tf, [len(full), len(groups_full)])
        same(f"{block} sensitivity cores / position groups: decision table = manifest", ts, [len(keep), len(groups_keep)])
        cb = C[block]
        same(f"{block}: contract counts", [cb["full_core_count"], cb["full_position_group_count"], cb["sensitivity_core_count"],
                                           cb["sensitivity_position_group_count"]], tf + ts)
        same(f"{block}: contract excluded ids = manifest recomputation", cb["excluded_fitcal_same_position_ids"], excl, show=False)
        out[block] = {"full_cores": len(full), "full_position_groups": len(groups_full), "sensitivity_cores": len(keep),
                      "sensitivity_position_groups": len(groups_keep), "excluded_ids": excl, "retained_ids": keep, "full_ids": full,
                      "position_groups_full": OrderedDict(sorted(groups_full.items())),
                      "position_groups_retained": OrderedDict(sorted(groups_keep.items()))}
    m = one(r"DEV(\d+) 排除的 (\d+) 个 core：`([^`]+)`", s61, "DEV24 excluded ids")
    ids = [x.strip() for x in m.group(3).split(",")]
    same("DEV24 excluded ids: decision text = manifest recomputation", ids, out["DEV"]["excluded_ids"])
    same("DEV24 counts in the sentence", [int(m.group(1)), int(m.group(2))], [out["DEV"]["sensitivity_cores"], len(out["DEV"]["excluded_ids"])])
    ntex = int(one(r"TEST 的 (\d+) 个排除 ID 和可复现规则见配套 JSON", s61, "TEST excluded count").group(1))
    same("TEST excluded count: decision text = manifest recomputation", ntex, len(out["TEST_REPEAT"]["excluded_ids"]))
    # DEV co-located with any other BF role (the report's 24): kept, only FIT/CAL co-location is excluded
    other = {pos(r) for r in M["rows"] if r not in M["dev"]}
    dev_any = sorted(r["record_id"] for r in M["dev"] if pos(r) in other)
    one(r"不能把所有 (\d+) 个“与其他角色同位”的 DEV 一概删除", s61, "DEV co-located with other roles")
    same("DEV co-located with any other role: decision = manifest", int(one(r"不能把所有 (\d+) 个", s61, "24 DEV").group(1)), len(dev_any))
    return {
        "decision_section": "6.1", "status": "PREREGISTERED_SENSITIVITY",
        "position_rule": "position = record_id.split('_', 1)[1] (the part after the first underscore; slide prefix 1B / 2B before it)",
        "hypothesis_status": C["relation_status"],
        "rule": "exclude an evaluation core whose position occurs among the currently included FIT or CAL cores; recompute; original "
                "manifest roles unchanged",
        "filters": {"fitcal": C["fitcal_filter"], "DEV": C["DEV_filter"], "TEST_REPEAT": C["TEST_REPEAT_metadata_filter"]},
        "fitcal_cores": len(M["fitcal"]), "fitcal_positions": fcpos,
        "DEV": out["DEV"], "TEST_REPEAT": out["TEST_REPEAT"],
        "dev_colocated_with_any_other_role": {"count": len(dev_any), "ids": dev_any,
                                              "rule": "not deleted: only FIT/CAL co-location defines the sensitivity block"},
        "assertions": ["DEV 32/32 -> 24/24", "TEST 134/97 -> 112/75", "DEV excluded: " + ", ".join(out["DEV"]["excluded_ids"]),
                       f"TEST excluded: {len(out['TEST_REPEAT']['excluded_ids'])} ids"],
        "test_access": "metadata only (manifest ids); TEST images stay HOLD and are never read in this workflow",
    }


def build_epfl(I) -> dict:
    md, C = I["md"], I["contract"]["data"]["EPFL"]
    s62 = section(md, "6.2")
    items = numbered(s62)
    same("EPFL activation items: count", len(items), 5)
    gates = C["activation_gates"]
    same("EPFL contract gates covered once by the mapping", sorted({g for v in MAP_EPFL_GATES.values() for g in v}), list(range(1, len(gates) + 1)))
    ep = [r for r in I["manifest"] if r["dataset"] == "EPFL_RGB_NIR"]
    n_align = sum(1 for r in ep if r["inclusion"] == "exclude" and r["exclusion_reason"].startswith("alignment:"))
    n_size = sum(1 for r in ep if r["inclusion"] == "exclude" and r["exclusion_reason"].startswith("size:"))
    n_keep = sum(1 for r in ep if r["inclusion"] == "include")
    m1 = one(r"核对 (\d+) 对图像", items[0], "EPFL item 1 pairs", 0)
    m2 = one(r"复现 (\d+) 个对齐排除、(\d+) 个尺寸排除及 (\d+) 个保留窗口", items[1], "EPFL item 2 counts", 0)
    m3 = one(r"共 (\d+) 类 × (\d+) 个 overlay", items[2], "EPFL item 3 overlays", 0)
    same("EPFL counts: decision = manifest", [int(m1.group(1))] + [int(x) for x in m2.groups()], [len(ep), n_align, n_size, n_keep])
    same("EPFL counts: decision = contract", [len(ep), n_align, n_size, n_keep],
         [C["pair_count"], C["fixed_alignment_exclusions"], C["fixed_size_exclusions"], C["retained_windows"]])
    same("EPFL categories: decision = manifest summary", int(m3.group(1)), I["mbs"]["epfl"]["categories"])
    thr = float(one(r"([\d.]+) native px 严格阈值可用于已知施加扰动的恢复误差", s62, "strict threshold").group(1))
    same("EPFL strict threshold: decision = package = contract", [thr, thr], [I["pkg_proto"]["registration"]["threshold_native_px"]["EPFL_RGB_NIR"][1],
                                                                               C["strict_threshold_native_px"]])
    one(r"元数据级检查是必要条件；激活前还要完成：", s62, "metadata necessary, not sufficient")
    same("contract metadata-only audit sufficient", C["metadata_only_alignment_audit_sufficient"], False)
    wording = one(r"报告中的“约 2 native px 残差”需要收窄：(.+?)\[EPFL P3 检查\]", s62, "EPFL wording correction").group(1).strip()
    p3 = git_blob(I["results_commit"], "src/v9/reg/r6_reg_p3.py").decode()
    k = one(r"^ALIGN_SEARCH, ALIGN_MARGIN, ALIGN_EXCLUDE, THUMB_DOWN, DUP_THR = (\d+), (\d+), (\d+), (\d+), ([\d.]+)", p3, "REG-P3 constants", re.M)
    rule = one(r"the check peaks >= (\d+) half-resolution px \(>= (\d+) native px\) from zero shift", p3, "REG-P3 exclusion rule")
    check("REG-P3 alignment_check at half resolution (2 x 2 area mean)", "def alignment_check(" in p3 and "resolution (2 × 2 area mean)" in p3)
    search, excl = int(k.group(1)), int(k.group(3))
    same("REG-P3 exclusion threshold: constant = docstring", int(rule.group(1)), excl)
    factor = int(rule.group(2)) // excl
    axis = excl - 1
    check("decision 6.2: retained best shift within ±1 px per axis (half resolution)", "每轴 ±1 像素" in wording and axis == 1)
    check("decision 6.2: diagonal native distance 2*sqrt(2)", "2\\sqrt2" in wording and factor * axis == 2)
    return {
        "decision_section": "6.2", "status": C["current_status"],
        "metadata_level_checks": "necessary (T00 done) but not sufficient",
        "items": [{"item": i + 1, "text_cn": t, "en": EN_EPFL[i + 1], "contract_gates": [gates[g - 1] for g in MAP_EPFL_GATES[i + 1]],
                   "status": "NOT_RUN"} for i, t in enumerate(items)],
        "counts_manifest": {"pairs": len(ep), "alignment_exclusions": n_align, "size_exclusions": n_size, "retained_windows": n_keep,
                            "categories": I["mbs"]["epfl"]["categories"]},
        "strict_threshold_native_px": thr,
        "reg_p3_alignment_check": {"source": f"{I['results_commit']}:src/v9/reg/r6_reg_p3.py alignment_check",
                                   "statistic": "integer shift maximising the correlation of RGB-luminance and NIR gradient magnitudes at half "
                                                "resolution (2 x 2 area mean)", "search_half_res_px": search,
                                   "exclude_if_max_abs_shift_ge_half_res_px": excl, "native_px_per_half_res_px": factor,
                                   "passing_best_shift_per_axis_half_res_px": axis, "passing_best_shift_per_axis_native_px": axis * factor,
                                   "passing_diagonal_native_px_max": axis * factor * 2 ** 0.5,
                                   "meaning": "an integer statistic of the correlation peak, not an upper bound on the true alignment error"},
        "threshold_interpretation_cn": wording,
        "threshold_interpretation_en": C["strict_threshold_interpretation"],
        "proxy_cost_is_not_measured_cost": C["proxy_cost_is_not_measured_cost"],
        "this_workflow": "no EPFL read",
    }


def build_gates(I) -> dict:
    md, C = I["md"], I["contract"]["fp32_gradient_gate"]
    s7 = section(md, "7")
    commit = one(r"确认 `([0-9a-f]{7})` 的有限范围修改，容差保持不变", s7, "confirmed commit").group(1)
    same("confirmed commit: decision = contract", commit, C["confirmed_commit"])
    check("confirmed commit exists in git", git("log", "-1", "--format=%s", commit).strip().startswith("V10 T01: fp32 dense-MLP gate"))
    tvp = {k: {x["name"]: x for x in I["tv"]["parts"][k]["checks"]} for k in ("adapter_cpu", "adapter_cuda")}
    rows = []
    for r in md_table(s7, "项目"):
        label, cpu, cuda, role = r
        check(f"gate row {label!r} mapped", label in MAP_GATE)
        name, ckey, required = MAP_GATE[label]
        xc, xg = tvp["adapter_cpu"][name], tvp["adapter_cuda"][name]
        same(f"gate {name}: required flag (CPU, CUDA) = decision role", [bool(xc.get("required")), bool(xg.get("required"))], [required, required])
        if ckey:
            check(f"gate {name}: CPU max abs error agrees", agrees(xc["max_abs_err"], cpu), f"{xc['max_abs_err']!r} ~ {cpu}")
            check(f"gate {name}: CUDA max abs error agrees", agrees(xg["max_abs_err"], cuda), f"{xg['max_abs_err']!r} ~ {cuda}")
            same(f"gate {name}: contract CPU value", C["max_abs_errors"]["cpu"][ckey], float(cpu))
            same(f"gate {name}: contract CUDA value", C["max_abs_errors"]["cuda"][ckey], float(cuda))
        else:
            check(f"gate {name}: PASS on CPU and CUDA (decision: 通过既有门限)", xc["status"] == "PASS" and xg["status"] == "PASS")
        if "CPU FAIL 保留" in role:
            same(f"gate {name}: CPU status FAIL kept", xc["status"], "FAIL")
        rows.append({"comparison_cn": label, "t01_check": name, "role": "required" if required else "diagnostic",
                     "cpu": {"status": xc["status"], "max_abs_err": xc["max_abs_err"]},
                     "cuda": {"status": xg["status"], "max_abs_err": xg["max_abs_err"]}, "decision_role_cn": role})
    tol = one(r"fp32 仍为 atol=([0-9e.-]+)、rtol=([0-9e.-]+)；fp64 仍为 atol=([0-9e.-]+)、rtol=([0-9e.-]+)。", s7, "tolerances").groups()
    td = I["tv"]["tolerances_design"]
    same("gate tolerances: decision = T01 tolerances_design", [float(x) for x in tol],
         [td["float32"]["atol"], td["float32"]["rtol"], td["float64"]["atol"], td["float64"]["rtol"]])
    same("gate tolerances: contract", [float(x) for x in tol], [C["tolerances_unchanged"]["fp32"]["atol"], C["tolerances_unchanged"]["fp32"]["rtol"],
                                                                 C["tolerances_unchanged"]["fp64"]["atol"], C["tolerances_unchanged"]["fp64"]["rtol"]])
    m = one(r"比较包含 du、dv 两项输入描述符梯度及 (\d+) 项 critic 参数梯度，共 (\d+) 个张量、(\d+) 个元素。", s7, "gate tensors")
    g = tvp["adapter_cpu"]["MLP_dense_blocked_checkpointed_float32_grads_vs_float64_value_same_inputs"]
    tens = [t["tensor"] for t in g["per_tensor"]]
    same("gate tensors (critic params, tensors, elements): decision = T01", [int(x) for x in m.groups()],
         [len([t for t in tens if t.startswith("critic.")]), g["n_tensors"], g["numel"]])
    same("gate tensors: contract", [int(x) for x in m.groups()], [C["gradient_tensors"]["critic_parameter_tensor_count"],
                                                                   C["gradient_tensors"]["total_tensor_count"], C["gradient_tensors"]["total_element_count"]])
    same("gate input descriptor tensors", tens[:2], ["du", "dv"])
    corr = one(r"报告第 8 节的“([^”]+)”应改为\*\*([^*]+)\*\*", s7, "report section 8 wording").groups()
    return {"decision_section": "7", "confirmed_commit": commit, "scope": C["scope"], "tested_path": C["tested_path"],
            "reference": "fp64 gradients recomputed from the same fp32-rounded inputs and the same fp32-rounded weights",
            "compared_tensors": tens, "n_tensors": g["n_tensors"], "n_elements": g["numel"], "rows": rows,
            "tolerances_unchanged": {"float32": {"atol": float(tol[0]), "rtol": float(tol[1])}, "float64": {"atol": float(tol[2]), "rtol": float(tol[3])}},
            "also_required": ["objective values blocked vs full at both dtypes", "every other mathematical, gradient and geometry gate (no downgrade)"],
            "cpu_full_graph_fp32": "known out-of-bound path, unused by T04/T05; old failure and new diagnostic kept; if it ever enters formal "
                                   "runs it becomes required and must pass, or uses explicit higher-precision accumulation with measured cost "
                                   "and revalidation",
            "report_wording": {"section": 8, "old": corr[0], "new_decision_text": corr[1],
                               "new_report_correction": "同输入和同权重的 fp64 梯度参考"}}


def budget_projection(I) -> dict:
    """old L40S timing x new candidate counts, by the BUDGET.json formula (train_steps x step time + registrations x s/registration)"""
    B = I["budget"]
    c = B["protocol_counts"]
    n_cal = c["checkpoints_per_unit"] * c["bf_cal_cores"] * c["perturbations_per_image"]
    n_unit = n_cal + c["bf_dev_cores"] * c["perturbations_per_image"]
    same("BUDGET registrations per BF unit", n_unit, B["registrations_per_unit"]["BF_SHG"])
    steps = c["train_steps_per_unit"]
    ms = B["measured"]["train_step_ms_median"]
    pu = B["per_unit_projection"]

    def cand(train_ms, s_reg):
        return (steps * train_ms / 1000 + n_cal * s_reg) / 3600

    def unit(train_ms, s_reg):
        return (steps * train_ms / 1000 + n_unit * s_reg) / 3600
    s_m1 = pu["MF_VCS_M1"]["s_per_registration_median"]
    tune = OrderedDict()
    for fam in ("MF_VCS_perm1", "MF_VCS_M0"):
        tune[fam] = 3 * cand(pu[fam]["train_step_ms_median"], pu[fam]["s_per_registration_median"])
    tune["MF_VCS_M1"] = 3 * sum(cand(ms[f"MF_M1_K16_it{cap}_capbound"], s_m1) for cap in (10, 30, 100))
    tune["MLP_VCS_perm1"] = 3 * cand(pu["MLP_VCS_perm1"]["train_step_ms_median"], pu["MLP_VCS_perm1"]["s_per_registration_median"])
    final = OrderedDict((fam, unit(pu[fam]["train_step_ms_median"], pu[fam]["s_per_registration_median"]))
                        for fam in ("MF_VCS_perm1", "MF_VCS_M0", "MLP_VCS_perm1"))
    m1_unit = OrderedDict((str(cap), unit(ms[f"MF_M1_K16_it{cap}_capbound"], s_m1)) for cap in (10, 30, 100))
    totals = OrderedDict((cap, sum(tune.values()) + 3 * sum(final.values()) + 3 * v) for cap, v in m1_unit.items())
    return {"tuning": tune, "tuning_total": sum(tune.values()), "final_unit": final, "final_unit_M1_by_cap": m1_unit,
            "totals_by_final_M1_cap": totals, "n_reg_cal_per_candidate": n_cal, "n_reg_per_final_unit": n_unit, "steps": steps}


def build_bf(I) -> dict:
    md, C = I["md"], I["contract"]["natural_registration"]
    s81, s82 = section(md, "8.1"), section(md, "8.2")
    K, temp, ridge, wwd = one(r"本批 MF 主表固定 \*\*K=(\d+)、temperature=([\d.]+)、ridge=([\d.]+)、W weight decay=([\d.]+)\*\*", s81, "fixed MF").groups()
    K, temp, ridge, wwd = int(K), float(temp), float(ridge), float(wwd)
    opt = one(r"AdamW，非 W 参数 weight decay=([0-9e.-]+)，clip_norm=([\d.]+)，(\d+) 场景 × (\d+) 位置，crop=(\d+)；", s81, "inherited optimiser").groups()
    nwd, clip, nsc, npos, crop = float(opt[0]), float(opt[1]), int(opt[2]), int(opt[3]), int(opt[4])
    one(r"所有由优化器学习的模块共享同一个候选学习率；M1 的 W 由 solver 求解，不给它额外学习率。", s81, "tied lr")
    one(r"M0/perm1 的 W 按既定规则参与 clip 后投影。", s81, "W clip then projection")
    table = md_table(s81, "方法")
    methods, lrs, cand_counts = [], None, {}
    for r in table:
        if r[0] == "合计":
            total = int(r[3].strip("*"))
            continue
        methods.append(r[0])
        if r[1] != "同上":
            lrs = [float(x) for x in cn_split(r[1])]
        caps = [int(x) for x in re.findall(r"\d+", r[2])] if r[2] != "不适用" else None
        check(f"BF {r[0]}: cap column", (caps is None) == (r[0] != "MF_VCS_M1") and (caps is None or "完整笛卡尔积" in r[2]))
        cand_counts[r[0]] = (int(r[3]), caps)
    same("BF batch-1 methods: section 8.1 table = section 5 batch 1", methods, cn_split(one(r"第一批为 \*\*([^*]+)\*\*", section(md, "5"), "batch 1").group(1)))
    for mth, (n, caps) in cand_counts.items():
        same(f"BF {mth}: candidate count = |lr| x |caps|", n, len(lrs) * (len(caps) if caps else 1))
    same("BF total candidates", total, sum(n for n, _ in cand_counts.values()))
    tseed, fseeds = one(r"调参 seed=(\d+)；正式算法 seeds=([\d、]+)。", s81, "seeds").groups()
    tseed, fseeds = int(tseed), [int(x) for x in cn_split(fseeds)]
    steps, every, nck = (int(x) for x in one(r"每次 (\d+) 步，每 (\d+) 步一个 checkpoint，共 (\d+) 个。", s81, "steps").groups())
    same("BF checkpoints = steps / every", nck, steps // every)
    cal_c, cal_p, dev_c, dev_p = (int(x) for x in one(r"CAL 为 (\d+) core×(\d+) 扰动；DEV 为 (\d+) core×(\d+) 扰动。", s81, "CAL/DEV").groups())
    sel = numbered(s81)
    t42 = int(one(r"success <(\d+) native px 较高", sel[0], "threshold 1", 0).group(1))
    t9 = int(one(r"success <(\d+) native px 较高", sel[1], "threshold 2", 0).group(1))
    same("BF selection items", [bare(x) for x in sel[2:]], ["median corner error 较低", "更早的 checkpoint", "冻结的候选顺序"])
    same("BF thresholds: decision = package", [float(t42), float(t9)], I["pkg_proto"]["registration"]["threshold_native_px"]["BF_SHG"])
    order_caps = [int(x) for x in cn_split(one(r"候选顺序为学习率升序；M1 在每个学习率内按 cap ([\d、]+)。", s81, "candidate order").group(1))]
    same("BF candidate order caps = table caps", order_caps, cand_counts["MF_VCS_M1"][1])
    same("BF lr ascending", lrs, sorted(lrs))
    # cross-checks: contract, package, phase-1 records
    for k, v in (("MF_K", K), ("MF_temperature", temp), ("ridge", ridge), ("W_weight_decay", wwd), ("non_W_weight_decay", nwd),
                 ("clip_norm", clip), ("scenes_per_step", nsc), ("positions_per_scene", npos), ("crop", crop), ("learning_rate_candidates", lrs),
                 ("training_steps", steps), ("checkpoint_interval", every), ("checkpoints_per_fit", nck), ("tuning_algorithm_seed", tseed),
                 ("final_algorithm_seeds", fseeds), ("optimizer", "AdamW"), ("tie_learning_rates_across_optimizer_learned_modules", True)):
        same(f"contract natural_registration.{k}", C[k], v)
    same("contract candidate counts", C["candidate_counts"], {**{m_: n for m_, (n, _) in cand_counts.items()}, "total": total})
    same("contract CAL / DEV sizes", [C["CAL"]["cores"], C["CAL"]["perturbations_per_core"], C["DEV"]["cores"], C["DEV"]["perturbations_per_core"]],
         [cal_c, cal_p, dev_c, dev_p])
    same("contract CAL selection length", len(C["CAL"]["selection_lexicographic"]), len(sel))
    pp = I["pkg_proto"]
    same("package lr candidates", pp["head"]["feature_head_learning_rate_candidates"], lrs)
    same("package M1 caps", pp["head"]["M1_max_iterations_candidates"], order_caps)
    same("package seeds", [pp["registration"]["tuning_seed"], pp["registration"]["training_seeds"]], [tseed, fseeds])
    bc = I["budget"]["protocol_counts"]
    same("BUDGET protocol counts", [bc["train_steps_per_unit"], bc["checkpoints_per_unit"], bc["bf_cal_cores"], bc["bf_dev_cores"],
                                    bc["perturbations_per_image"], bc["tuning_seed"], bc["training_seeds"]],
         [steps, nck, cal_c, dev_c, cal_p, tseed, fseeds])
    same("manifest CAL / DEV core counts", [manifest_sets(I)["roles"]["CAL"], manifest_sets(I)["roles"]["DEV"]], [cal_c, dev_c])
    note = I["t02c"]["parts"]["train"]["configs"]["MF_perm1_K16"]["notes"]
    check("phase-1 T02 train note: 16 x 128 host batch, AdamW + clip 1.0",
          f"host batch ({nsc} FIT crops x {npos} positions)" in note and f"AdamW + clip {clip}" in note)
    ad = git_blob(I["results_commit"], "src/v10/adapter.py").decode()
    tc = one(r"class TrainConfig:\n(.*?)\n\n\n", ad, "phase-1 TrainConfig").group(1)
    dflt = dict(re.findall(r"^    (\w+): [^=\n]+= ([^\s#]+)", tc, re.M))
    same("phase-1 TrainConfig K / temperature / weight_decay / clip_norm / n_pos / crop / w_init",
         [int(dflt["K"]), float(dflt["temperature"]), float(dflt["weight_decay"]), float(dflt["clip_norm"]), int(dflt["n_pos"]), int(dflt["crop"]),
          dflt["w_init"]], [K, temp, nwd, clip, npos, crop, '"zeros"'])
    # budget (decision 8.2)
    mx_c, mx_h = (int(x) for x in one(r"每族调参上限为\*\*最多 (\d+) 个候选、最多 (\d+) 个 L40S 已分配 GPU 小时\*\*", s82, "per-family cap").groups())
    same("contract per-family cap", [C["tuning_budget_per_family"]["max_candidates"], C["tuning_budget_per_family"]["max_L40S_allocated_GPU_hours"]], [mx_c, mx_h])
    check("every family's candidate count <= the per-family cap", all(n <= mx_c for n, _ in cand_counts.values()))
    proj = budget_projection(I)
    btab = md_table(s82, "调参部分")
    shown = {r[0]: r[1].strip("*") for r in btab}
    keys = [("MF perm1，3 格", "MF_VCS_perm1"), ("M0，3 格", "MF_VCS_M0"), ("M1，3 lr × 3 cap", "MF_VCS_M1"), ("MLP perm1，3 格", "MLP_VCS_perm1")]
    for lab, fam in keys:
        check(f"budget 8.2 {fam}: recomputed agrees with the decision table", agrees(proj["tuning"][fam], shown[lab]), f"{proj['tuning'][fam]!r} ~ {shown[lab]}")
        same(f"budget 8.2 {fam}: contract", C["old_measured_cost_projection_only"]["tuning_GPU_hours"][fam], float(shown[lab]))
    check("budget 8.2 tuning total agrees (sum of the unrounded parts)", agrees(proj["tuning_total"], shown["调参合计"]), f"{proj['tuning_total']!r} ~ {shown['调参合计']}")
    m9 = one(r"当前 capbound 计时对 M1 九格的投影约为 ([\d.]+) 小时", s82, "M1 nine-cell projection").group(1)
    check("budget 8.2 M1 nine cells agree", agrees(proj["tuning"]["MF_VCS_M1"], m9))
    check("M1 nine-cell projection within the 8 GPU-h cap", proj["tuning"]["MF_VCS_M1"] < mx_h)
    caps_t = one(r"若最终 M1 cap 为 (\d+) / (\d+) / (\d+)，总计分别约 \*\*([\d.]+) / ([\d.]+) / ([\d.]+) GPU-h\*\*", s82, "totals by cap").groups()
    for cap, val in zip(caps_t[:3], caps_t[3:]):
        check(f"budget 8.2 total with final M1 cap {cap} agrees", agrees(proj["totals_by_final_M1_cap"][cap], val), f"{proj['totals_by_final_M1_cap'][cap]!r} ~ {val}")
        same(f"budget 8.2 total cap {cap}: contract", C["old_measured_cost_projection_only"]["tuning_plus_3_final_seeds_per_family_GPU_hours_by_selected_M1_cap"][cap], float(val))
    old = one(r"报告旧的 ([\d.]+) 或 ([\d.]+) 小时不能继续作为此网格的总预算", s82, "old totals").groups()
    g = I["budget"]["grid_projection_measured_methods"]
    check("old total 21.089 = phase-1 CAL tuning + BF 4 families x 3 seeds", agrees(g["CAL_tuning_gpu_h_BF"] + g["BF_SHG_gpu_h"], old[0]))
    check("old total 29.78 = phase-1 BUDGET total", agrees(g["total_gpu_h"], old[1]))
    one(r"相同步数主表称为 same-data / fixed-steps。", s82, "fixed-steps name")
    one(r"不能用超预算训练完成后的最佳 checkpoint 回填较早预算。", s82, "no backfill")
    # candidates and formal units
    cands, ordinal = [], 0
    for mth in methods:
        fo = 0
        for lr in lrs:
            for cap in (cand_counts[mth][1] or [None]):
                ordinal += 1
                fo += 1
                cid = f"BF1_{mth}_lr{lr_text(lr)}" + (f"_cap{cap}" if cap else "")
                cands.append(OrderedDict([("candidate_id", cid), ("ordinal", ordinal), ("family_ordinal", fo), ("method", mth),
                                          ("lr", lr), ("m1_cap", cap), ("K", K if mth.startswith("MF_") else None),
                                          ("temperature", temp if mth.startswith("MF_") else None),
                                          ("ridge", ridge if mth.startswith("MF_") else None),
                                          ("W_weight_decay", wwd if mth.startswith("MF_") else None),
                                          ("tuning_seed", tseed), ("steps", steps), ("checkpoint_every", every), ("checkpoints", nck),
                                          ("status", "NOT_RUN")]))
    same("BF candidates enumerated", len(cands), total)
    # items frozen before the primary runs (decision 10 step 3 and section 5) = contract rollout.freeze_before_primary_runs
    fr3 = one(r"冻结首批公共协议、(.+?)；开展 BF 的四族 CAL/DEV", section(md, "10"), "step 3: frozen items").group(1)
    fr5 = one(r"再按冻结的(.+?)规则补跑。", section(md, "5"), "section 5: frozen rules").group(1)
    items = [x.strip() for x in re.split(r"、|和", fr3) if x.strip()] + [x.strip() for x in re.split(r"、|和", fr5) if x.strip()]
    for it in items:
        check(f"freeze item {it!r}: mapped (MAP_FREEZE)", it in MAP_FREEZE)
    cfreeze = I["contract"]["rollout"]["freeze_before_primary_runs"]
    same("freeze before primary runs: decision 10 step 3 + section 5 = contract rollout.freeze_before_primary_runs (as sets)",
         sorted({MAP_FREEZE[x] for x in items}), sorted(cfreeze))
    same("contract freeze list without repeats", len(set(cfreeze)), len(cfreeze))
    freeze = [OrderedDict([("item", it), ("decision_text", [x for x in items if MAP_FREEZE[x] == it]), ("status", "NOT_RUN")])
              for it in cfreeze]
    n_dev_reg = proj["n_reg_per_final_unit"] - proj["n_reg_cal_per_candidate"]
    same("formal-unit DEV registrations = DEV cores x perturbations", n_dev_reg, dev_c * dev_p)
    formal = [OrderedDict([("unit_id", f"BF1_{mth}_s{s}"), ("method", mth), ("algo_seed", s), ("hyperparameters", "the family's CAL-selected "
                           "candidate (tuning seed 1017)"), ("checkpoint_selection", "decision 8.1 items 1-4 over this seed's 12 checkpoints on CAL"),
                           ("dev", f"{dev_c} cores x {dev_p} perturbations at the selected checkpoint"), ("status", "NOT_RUN")])
              for mth in methods for s in fseeds]
    return {
        "decision_section": "8", "status": "LANDED_NOT_FROZEN (freeze = decision step 3)",
        "freeze_before_primary_runs": freeze,
        "freeze_rule": "decision step 3 freezes these items (with the batch-1 common protocol) before any BF CAL / DEV primary run; "
                       "items are NOT_RUN until then (baselines.batch1_start_condition)",
        "fixed": {"MF_K": K, "MF_temperature": temp, "ridge": ridge, "W_weight_decay": wwd,
                  "architecture": "the audited host encoders and MLP critic unchanged; MF head = host encoders + per-side feature map + W"},
        "learning_rate": {"candidates": lrs, "tied": "one candidate learning rate shared by every optimiser-learned module (MF perm1 / M0: "
                          "encoders, feature maps and W; MF M1: encoders and feature maps, W solved with no learning rate; MLP: encoders and "
                          "critic)", "phase1_code_delta": "phase-1 TrainConfig kept encoder_lr = 1e-3 (host) apart from head_lr; P2 ties them"},
        "inherited": {"optimizer": "AdamW", "non_W_weight_decay": nwd, "clip_norm": clip, "scenes_per_step": nsc, "positions_per_scene": npos,
                      "crop": crop, "W_perm1_M0": "W enters the clip, then is projected to the box after every step"},
        "methods": methods,
        "candidate_counts": {**{m_: n for m_, (n, _) in cand_counts.items()}, "total": total},
        "M1_caps": order_caps,
        "candidate_order": "within a family: learning rate ascending; MF_VCS_M1: cap ascending within each learning rate (decision 8.1); "
                           "families listed in the decision 8.1 table order",
        "candidates": cands,
        "seeds": {"tuning": tseed, "formal": fseeds},
        "training": {"steps": steps, "checkpoint_every": every, "checkpoints": nck},
        "CAL": {"cores": cal_c, "perturbations_per_core": cal_p, "registrations_per_candidate": proj["n_reg_cal_per_candidate"]},
        "DEV": {"cores": dev_c, "perturbations_per_core": dev_p},
        "selection": {"scope": "candidate and checkpoint per family on CAL only (tuning seed); formal seeds select their checkpoint on CAL "
                               "with items 1-4", "lexicographic": [x.format(t42=t42, t9=t9) for x in MAP_BF_SELECTION],
                      "lexicographic_cn": [bare(x) for x in sel], "thresholds_native_px": [t42, t9]},
        "formal_units": formal,
        "budget": {"per_family_cap": {"max_candidates": mx_c, "max_L40S_allocated_GPU_hours": mx_h, "stopping": "first limit reached",
                                      "includes": ["training", "checkpoints", "CAL registration", "in-job overhead"],
                                      "queue_wait": "reported separately", "report": "actual candidates and actual total cost per family; "
                                      "no artificial exhaustion of the budget", "at_limit": "remaining parts marked incomplete; no "
                                      "outcome-dependent extension"},
                   "engineering_cap_note": "8 GPU-h is an engineering ceiling, not a measured commitment of the relative rule; new native "
                                           "baselines are timed first and enter the frozen attachment with candidates bounded by this "
                                           "ceiling, or are disclosed as resource-limited",
                   "old_timing_projection_GPU_h": {"label": "phase-1 L40S timing (cap-bound M1) x the new candidate counts; conditional "
                                                   "projection only", "formula": f"candidate = ({proj['steps']} steps x step time + "
                                                   f"{proj['n_reg_cal_per_candidate']} CAL registrations x s/registration) / 3600; formal unit "
                                                   f"= ({proj['steps']} steps x step time + {proj['n_reg_per_final_unit']} registrations "
                                                   f"({proj['n_reg_cal_per_candidate']} CAL + {n_dev_reg} DEV) x s/registration) / 3600",
                                                   "tuning": proj["tuning"], "tuning_total": proj["tuning_total"],
                                                   "formal_unit": proj["final_unit"], "formal_unit_MF_VCS_M1_by_cap": proj["final_unit_M1_by_cap"],
                                                   "tuning_plus_3_formal_seeds_by_final_M1_cap": proj["totals_by_final_M1_cap"],
                                                   "excludes": C["old_measured_cost_projection_only"]["excludes"],
                                                   "replace_after": "the relative-M1 measurement (decision step 2) rewrites BUDGET",
                                                   "not_valid_totals": [float(old[0]), float(old[1])]}},
        "views": {"same_data_fixed_steps": "main table: same data, every candidate trained for the fixed step count",
                  "same_total_cost": "actual cumulative cost and the checkpoints produced within the budget, selected only on CAL; never "
                                     "backfilled with the best checkpoint of an over-budget run", "note": "an equal ceiling is not equal "
                                     "spending"},
        "capacity_extension": "K8 / K32 on natural images only as a separately named capacity extension, never mixed into the main table "
                              "and never 27 unbudgeted cells",
        "scheduling": "parallel scheduling allowed; completion, over-budget and failures are never handled selectively by DEV results; the "
                      "common searcher, negative permutation, geometry support, random streams and checkpoint rules stay identical",
    }


def synthetic_defaults(I) -> dict:
    tree = ast.parse(I["syn_src"])
    fam = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "FAMILIES" for t in node.targets):
            fam = list(ast.literal_eval(node.value).keys())
    d = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            kw = {k.arg: k.value for k in node.keywords}
            if "default" in kw:
                name = ast.literal_eval(kw["dest"]) if "dest" in kw else node.args[0].value.lstrip("-").replace("-", "_")
                try:
                    d[name] = ast.literal_eval(kw["default"])
                except ValueError:
                    d[name] = ast.unparse(kw["default"])
    d["families"] = fam
    return d


def case_id(fam: str, m: int, rho: float) -> str:
    """package v10_synthetic.case_id"""
    return f"{fam}__m{m:03d}__rho{format(rho, '.17g').replace('.', 'p')}"


def build_t03(I) -> tuple[dict, list]:
    md, C = I["md"], I["contract"]["T03"]
    s91, s92, s93, s94, s95 = (section(md, x) for x in ("9.1", "9.2", "9.3", "9.4", "9.5"))
    mode = one(r"T03 主模式为 \*\*(\w)\*\*", s91, "T03 mode").group(1)
    same("T03 mode: decision = contract", mode.lower(), C["primary_q_mode"])
    u_formula = one(r"(\\widehat Q_U=[^\n]+)", s91, "U reference formula").group(1)
    one(r"所有 perm1/perm8 对照也使用\*\*均匀无固定点排列\*\*", s91, "uniform derangements")
    one(r"小 n 穷举验证所有无固定点排列的平均值和梯度等于完整 U。", s91, "small-n enumeration gate")
    one(r"V 仅作\*\*同一个冻结 critic 上的 U/V 评估诊断\*\*，不另开第二套 V 训练网格。", s91, "V diagnostic only")
    T = {r[0]: r[1] for r in md_table(s92, "项目")}
    cm = one(r"^([a-z_]+)、([a-z_]+) × m=(\d+)、(\d+) × rho=([\d.]+)、([\d.]+)、([\d.]+)，共 (\d+) 条件$", T["数据条件"], "conditions", 0).groups()
    fams, ms, rhos, ncond = list(cm[:2]), [int(cm[2]), int(cm[3])], [float(x) for x in cm[4:7]], int(cm[7])
    conds = [(f, m, r) for f in fams for m in ms for r in rhos]
    same("T03 condition count", len(conds), ncond)
    sizes = [int(x) for x in one(r"^(\d+) / (\d+) / (\d+) 个独立基本观测$", T["每条件 FIT / CAL / EVAL"], "sizes", 0).groups()]
    syn = synthetic_defaults(I)
    same("T03 grid: decision = package generator defaults", [fams, ms, rhos, sizes],
         [syn["families"], syn["m"], syn["rho"], [syn["n_fit"], syn["n_cal"], syn["n_eval"]]])
    same("T03 conditions: contract", [C["conditions"]["families"], C["conditions"]["m"], C["conditions"]["rho"], C["conditions"]["count"]],
         [fams, ms, rhos, ncond])
    same("T03 sizes: contract", [C["observations_per_condition"][k] for k in ("FIT", "CAL", "EVAL")], sizes)
    mfk = one(r"复用 SimplexBilinearCritic\(m,m,K,K\)，每侧 Linear\+softmax，主 K(\d+)，temperature=([\d.]+)，W=0", T["输入与 MF"], "MF row", 0).groups()
    same("T03 MF: contract", [C["MF"]["main_K"], C["MF"]["temperature"], C["MF"]["initial_W"]], [int(mfk[0]), float(mfk[1]), "zeros"])
    mlp = one(r"Linear\(2m,(\d+)\)→GELU→Linear\((\d+),(\d+)\)→GELU→Linear\((\d+),1\)，T=tanh f；perm1/8 同结构同初始化", T["MLP"], "MLP row", 0).groups()
    same("T03 MLP: contract", [C["MLP"]["hidden_dims"], C["MLP"]["hidden_activation"], C["MLP"]["output_dim"], C["MLP"]["critic_transform"]],
         [[int(mlp[0]), int(mlp[2])], "GELU", 1, "tanh"])
    fam_labels = cn_split(one(r"^(.+)，共 (\d+) 族$", T["主方法"], "T03 families", 0).group(1))
    methods = [MAP_T03_METHODS[x] for x in fam_labels]
    same("T03 families: decision = contract", methods, C["main_families"])
    tr = one(r"FIT 全批，(\d+) 步，每 (\d+) 步 checkpoint；AdamW；非 W 参数 decay=([0-9e.-]+)、W decay=([\d.]+)、ridge=([\d.]+)、clip=([\d.]+)", T["训练与优化"], "T03 training", 0).groups()
    steps, every = int(tr[0]), int(tr[1])
    same("T03 training: contract", [C["training"]["steps"], C["training"]["checkpoint_interval"], C["training"]["checkpoints_per_fit"],
                                    C["optimizer"]["non_W_weight_decay"], C["optimizer"]["W_weight_decay"], C["optimizer"]["ridge"], C["optimizer"]["clip_norm"]],
         [steps, every, steps // every, float(tr[2]), float(tr[3]), float(tr[4]), float(tr[5])])
    lrm = one(r"^([0-9e、-]+)，学习模块共享候选；M1 同时调 cap=([\d、]+)$", T["学习率"], "T03 lr", 0).groups()
    lrs, caps = [float(x) for x in cn_split(lrm[0])], [int(x) for x in cn_split(lrm[1])]
    same("T03 lr / caps: contract", [C["optimizer"]["learning_rates"], C["M1_caps"]], [lrs, caps])
    cc = one(r"每条件 ([\d+]+)=(\d+)；(\d+) 条件共 \*\*(\d+) 次 CAL 训练\*\*", T["CAL 候选数"], "T03 CAL count", 0).groups()
    per = [int(x) for x in cc[0].split("+")]
    exp_per = [len(lrs) * (len(caps) if mth == "MF_VCS_M1" else 1) for mth in methods]
    same("T03 CAL candidates per family = |lr| x |caps|", per, exp_per)
    same("T03 CAL candidates per condition / total", [int(cc[1]), int(cc[2]), int(cc[3])], [sum(per), ncond, sum(per) * ncond])
    same("T03 CAL: contract", [C["CAL"]["candidates_per_condition"][m_] for m_ in methods] + [C["CAL"]["candidates_per_condition"]["total"], C["CAL"]["fit_count"]],
         per + [sum(per), sum(per) * ncond])
    ts = one(r"独立数据 root seed=(\d+)，算法 seed=(\d+)", T["超参数调参套"], "tuning suite", 0).groups()
    tune_root, tune_algo = int(ts[0]), int(ts[1])
    same("T03 tuning suite: contract", [C["CAL"]["independent_data_root_seed"], C["CAL"]["algorithm_seed"]], [tune_root, tune_algo])
    one(r"只最大化独立 CAL 的完整经验 U-J；并列取更早 checkpoint，再取上述冻结候选顺序", T["CAL 选模"], "T03 selection", 0)
    one(r"truth 仅交 evaluator；训练和 CAL 选择只读样本", T["真实量访问"], "truth access", 0)
    roots = one(r"root seeds 为 \*\*(\d+) 至 (\d+)\*\*", s93, "data root seeds").groups()
    data_roots = list(range(int(roots[0]), int(roots[1]) + 1))
    algo_main = int(one(r"每套均有独立 FIT/CAL/EVAL，算法 seed 固定 (\d+)。", s93, "main algo seed").group(1))
    ex = one(r"第一套数据 (\d+) 另加算法 seeds (\d+)、(\d+)，", s93, "extra algorithm seeds").groups()
    ex_root, ex_algos = int(ex[0]), [int(ex[1]), int(ex[2])]
    same("T03 10 data suites", len(data_roots), 10)
    same("T03 extra seeds use the first suite", ex_root, data_roots[0])
    same("T03 repetitions: contract", [C["main_repetitions"]["data_root_seeds"], C["main_repetitions"]["algorithm_seed"],
                                       C["conditional_algorithm_repetitions"]["data_root_seed"], C["conditional_algorithm_repetitions"]["additional_algorithm_seeds"]],
         [data_roots, algo_main, ex_root, ex_algos])
    check("T03 tuning suite root is distinct from the formal roots", tune_root not in data_roots)
    capm = one(r"容量诊断限定在两个 family 的 \*\*m(\d+)、rho=([\d.]+)/([\d.]+)\*\*，共 (\d+) 个条件。三个 MF 族增加 K(\d+) 和 K(\d+)，", s94, "capacity").groups()
    cap_conds = [(f, int(capm[0]), float(r)) for f in fams for r in (capm[1], capm[2])]
    same("capacity conditions count", len(cap_conds), int(capm[3]))
    cap_K = [int(capm[4]), int(capm[5])]
    cap_methods = [m_ for m_ in methods if m_.startswith("MF_")]
    capst = one(r"同为 (\d+) 步、每 (\d+) 步 checkpoint，按该重复的 CAL 样本 U-J 选 checkpoint，不访问 oracle。使用相同 (\d+) 个 data seed 和算法 seed(\d+)，", s94, "capacity training").groups()
    same("capacity training = main training; same data seeds and algo seed", [int(capst[0]), int(capst[1]), int(capst[2]), int(capst[3])],
         [steps, every, len(data_roots), algo_main])
    cd = I["contract"]["T03"]["capacity_diagnostic"]
    same("capacity: contract", [[(x["family"], x["m"], float(x["rho"])) for x in cd["conditions"]], cd["families"], cd["additional_K"],
                                cd["data_root_seeds"], cd["algorithm_seed"], cd["training_steps"], cd["checkpoint_interval"]],
         [cap_conds, cap_methods, cap_K, data_roots, algo_main, steps, every])
    counts_tab = {r[0]: r for r in md_table(s94, "部分")}
    n_main = len(conds) * len(methods) * len(data_roots)
    n_extra = len(conds) * len(methods) * len(ex_algos)
    n_cap = len(cap_conds) * len(cap_methods) * len(cap_K) * len(data_roots)
    n_cal = sum(per) * ncond
    for key, val in (("主实验", n_main), ("固定首套数据的额外算法重复", n_extra), ("容量诊断", n_cap), ("正式与诊断合计", n_main + n_extra + n_cap),
                     ("独立 CAL 候选", n_cal), ("合计，不含 smoke 或后补基线", n_main + n_extra + n_cap + n_cal)):
        same(f"T03 count table {key}", int(counts_tab[key][2].strip("*")), val)
    fc = C["fit_counts"]
    same("T03 counts: contract", [fc["main"], fc["conditional_algorithm_extra"], fc["capacity"], fc["final_and_diagnostic"], fc["CAL"],
                                  fc["total_excluding_smoke_and_followup_baselines"]],
         [n_main, n_extra, n_cap, n_main + n_extra + n_cap, n_cal, n_main + n_extra + n_cap + n_cal])
    one(r"先完成一个条件的五族 smoke 和实际全训练成本测量，再按此冻结网格分批实施。", s94, "smoke first")
    one(r"有限状态下 evaluator 可以枚举最多 m²=1024 个单元", s95, "m^2 <= 1024")
    same("max m^2 = 1024", max(ms) ** 2, 1024)
    sat = " ".join(one(r"(\\hat p_\{xy\}=N_\{xy\}/n,.*?T\^\*_\{xy\}=\\frac\{\\hat p_\{xy\}-\\hat q_\{xy\}\}\{\\hat p_\{xy\}\+\\hat q_\{xy\}\}),", s95,
                       "saturated table").group(1).split())
    one(r"主 EVAL U-J 的负数保留，不截成零。", s95, "negative J kept")
    one(r"非 S 输出的 InfoNCE、DV、NWJ 等不按 S 真值计算 RMSE。", s95, "no S-RMSE for non-S outputs")
    # ---------------- units
    units = build_t03_units(conds, methods, lrs, caps, tune_root, tune_algo, data_roots, algo_main, ex_algos, cap_conds, cap_methods, cap_K,
                            steps, every, int(mfk[0]))
    nb = defaultdict(int)
    for u in units:
        nb[u["block"]] += 1
    same("T03 units per block", dict(nb), {"smoke": len(SMOKE_PLAN(methods, caps)), "CAL": n_cal, "main": n_main, "algo_extra": n_extra, "capacity": n_cap})
    same("T03 grid rows (excluding smoke)", sum(v for k, v in nb.items() if k != "smoke"), n_main + n_extra + n_cap + n_cal)
    same("T03 unit ids unique", len({u["unit_id"] for u in units}), len(units))
    ids = {u["unit_id"] for u in units}
    check("T03 depends_on references existing unit ids", all(d in ids for u in units for d in u["depends_on"].split(";") if d))
    t03 = {
        "decision_section": "9", "status": "LANDED_NOT_FROZEN (T03 code not written; smoke first)",
        "target": {"primary_q_mode": mode, "population": "the package generator's Q = P_X P_Y, S and eta (observed integer variables)",
                   "empirical_U_reference": u_formula, "MF_exact_moments": "U mode",
                   "negatives_perm1_perm8": {"sampler": "uniform derangement: rejection of ordinary uniform permutations with a fixed point, "
                                             "or a validated equivalent uniform algorithm", "forbidden": ["plain randperm (its mean is V)",
                                             "fixed-point repair without a distribution check", "a fixed cyclic shift"],
                                             "gate": "small-n exhaustive enumeration: the mean value and gradient over all derangements equal "
                                                     "the complete U objective", "status": "NOT_RUN"},
                   "V": "only as a U/V evaluation diagnostic on the same frozen critic and data; no V training grid",
                   "natural_registration": "unchanged: within-scene V with ordinary permutations (fixed points allowed)",
                   "caveat": "for a fixed T independent of the evaluation sample the V Q-term carries a 1/n joint-distribution part; the "
                             "finite-sample formula does not make the trained and selected maximum J an unbiased estimate of S"},
        "conditions": [OrderedDict([("case_id", case_id(f, m, r)), ("family", f), ("m", m), ("rho", r)]) for f, m, r in conds],
        "observations_per_condition": {"FIT": sizes[0], "CAL": sizes[1], "EVAL": sizes[2]},
        "generator": {"path": PKG_SYN, "sha256": I["inputs"][f"{I['results_commit']}:{PKG_SYN}"], "rule": "one complete suite (all 12 conditions, FIT/CAL/EVAL) per data "
                      "root seed via generate_suite(root_seed=...); record generator and sample hashes; truth only for the evaluator"},
        "input": "integer observations as one-hot",
        "MF": {"implementation": "SimplexBilinearCritic(m, m, K, K): Linear + softmax per side", "K_main": int(mfk[0]), "temperature": float(mfk[1]),
               "W_initial": "zeros", "init": "current PyTorch default Linear initialisation with a frozen RNG order; the three MF methods share "
               "the corresponding initial state; no method gets a separately searched initialisation"},
        "MLP": {"architecture": f"concat(one-hot x, one-hot y) -> Linear(2m,{mlp[0]}) -> GELU -> Linear({mlp[1]},{mlp[2]}) -> GELU -> "
                                f"Linear({mlp[3]},1); T = tanh f", "perm1_perm8": "same structure and same initial state"},
        "methods": methods,
        "training": {"batch": "full FIT batch", "steps": steps, "checkpoint_every": every, "checkpoints": steps // every, "optimizer": "AdamW",
                     "non_W_weight_decay": float(tr[2]), "W_weight_decay": float(tr[3]), "ridge": float(tr[4]), "clip_norm": float(tr[5]),
                     "learning_rates": lrs, "lr_tied": "one candidate shared by all learned modules", "M1_caps": caps,
                     "M1_rule": "m1_rule of this protocol (relative zero-head gap)"},
        "tuning_suite": {"data_root_seed": tune_root, "algo_seed": tune_algo, "candidates_per_condition": dict(zip(methods, per)),
                         "candidates_total": n_cal, "candidate_order": "learning rate ascending; MF_VCS_M1 cap ascending within each "
                         "learning rate; ordinal within (condition, method)",
                         "selection": "per (condition, method): maximise the complete empirical U-J on the tuning suite's independent CAL "
                                      "samples over candidates x checkpoints; ties -> earlier checkpoint -> frozen candidate ordinal; the "
                                      "selected lr / cap are frozen for every formal replicate of that (condition, method)",
                         "truth_access": "none: training and CAL selection read samples only (load_samples); truth only via the "
                                         "evaluator (load_evaluation_truth)"},
        "replicates": {"main": {"data_root_seeds": data_roots, "algo_seed": algo_main, "count": n_main},
                       "conditional_algorithm": {"data_root_seed": ex_root, "algo_seeds": ex_algos, "count": n_extra,
                                                 "rule": "own table; never pooled with the 10 data replicates into 12 identically "
                                                         "distributed replicates"},
                       "checkpoint_selection": "each replicate selects its checkpoint only from its own CAL samples (complete empirical "
                                               "U-J; ties -> earlier checkpoint)",
                       "pairing_unit": "the data replicate (bias, variance, RMSE with finite-replicate uncertainty)",
                       "rng": "separate data, model-initialisation and permutation streams; every method on a data root seed sees the "
                              "same observations"},
        "capacity": {"conditions": [OrderedDict([("case_id", case_id(f, m, r)), ("family", f), ("m", m), ("rho", r)]) for f, m, r in cap_conds],
                     "methods": cap_methods, "K": cap_K, "hyperparameters": "inherit the K16 tuning-suite selection of the same "
                     "(condition, method)", "data_root_seeds": data_roots, "algo_seed": algo_main, "count": n_cap,
                     "interpretation": "capacity response under fixed training rules (includes optimisation effects); not an "
                                       "independently optimised capacity curve"},
        "counts": OrderedDict([("main", n_main), ("conditional_algorithm_extra", n_extra), ("capacity", n_cap),
                               ("final_and_diagnostic", n_main + n_extra + n_cap), ("CAL", n_cal),
                               ("total_excluding_smoke_and_followup_baselines", n_main + n_extra + n_cap + n_cal)]),
        "smoke": {"rule": "one condition, all five families, full 2000-step trainings with the actual full training cost measured, before "
                          "any grid unit; the phase-1 head-only estimate is not a measured budget",
                  "plan": "landing_choices[T03 smoke] (not an owner decision)",
                  "prerequisites": ["T03 code and unit tests", "suite generation + package synthetic/tests/check_synthetic.py",
                                    "uniform-derangement enumeration gate"],
                  "over_budget": "if the measured resources exceed the original order of magnitude, record the new budget and the "
                                 "completable scope first; never drop seeds, pick conditions or change the grid afterwards"},
        "metrics": {"error_decomposition": ["fixed-feature head optimisation gap", "population approximation error of the learned critic "
                                            "(E_M(T-eta)^2 = S - J_pop(T), evaluator enumerates at most m^2 = 1024 cells)",
                                            "independent EVAL empirical-J sampling error", "conditional permutation variance on the same "
                                            "base data"], "negative_EVAL_UJ": "kept, never truncated at zero",
                    "saturated_FIT_table": {"formulas": sat, "zero_denominator": "T* = 0", "role": "empirical optimum of the saturated "
                                            "FIT-table function class; no truth in training; not a trained MF feature",
                                            "caveat": "an MLP can also aggregate by cell counts in this discrete setting; no general "
                                                      "MF-vs-MLP compute-advantage inference"},
                    "rulsif": "RuLSIF exact moments join the quadratic-risk comparison after implementation and checks",
                    "non_S_outputs": "InfoNCE, DV, NWJ etc. get no S-truth RMSE",
                    "cost": "full cost of every unit"},
        "units_file": {"path": OUT_UNITS, "columns": list(UNIT_COLUMNS), "rows": len(units), "rows_by_block": dict(nb),
                       "id_scheme": "T03_<SMK|CAL|MAIN|ALGO|CAP>_<n>, numbered in the frozen order of each block",
                       "order": {"smoke": "method order, MF_VCS_M1 cap ascending", "CAL": "condition -> method -> lr ascending -> cap ascending",
                                 "main": "data root seed -> condition -> method", "algo_extra": "algo seed -> condition -> method",
                                 "capacity": "data root seed -> condition -> method -> K"},
                       "conventions": {"lr": "1e-4 / 3e-4 / 1e-3 for CAL and smoke rows; CAL = the CAL-selected value of the same "
                                       "(condition, method); CAL_K16 = the K16 CAL selection inherited by a capacity row",
                                       "cap": "MF_VCS_M1 only (10 / 30 / 100, CAL or CAL_K16); NA otherwise", "K": "MF only; NA for MLP",
                                       "selection_rule": SELECTION_RULES,
                                       "depends_on": "';'-joined unit ids that must be complete first (CAL rows: every smoke row; formal "
                                                     "rows: the CAL candidates of the same condition and method)"}},
    }
    return t03, units


SELECTION_RULES = OrderedDict([
    ("SMOKE_NONE", "smoke: implementation checks and full-cost measurement; outputs never enter any selection"),
    ("CAL_UJ_SELECT", "tuning suite: candidate x checkpoint selection per (condition, method) by the complete empirical U-J on CAL; ties "
                      "-> earlier checkpoint -> frozen candidate ordinal"),
    ("OWN_CAL_UJ_CKPT", "formal replicate: hyperparameters fixed from the tuning suite; checkpoint selected only on this replicate's own "
                        "CAL samples by the complete empirical U-J; ties -> earlier checkpoint"),
])
SMOKE_CHOICE = {"family": "diagonal_mixture", "m": 32, "rho": 0.75, "lr": 3e-4, "data_root_seed": 20261016, "algo_seed": 1017}


def SMOKE_PLAN(methods, caps):
    return [(m_, cap) for m_ in methods for cap in (caps if m_ == "MF_VCS_M1" else [None])]


def build_t03_units(conds, methods, lrs, caps, tune_root, tune_algo, data_roots, algo_main, ex_algos, cap_conds, cap_methods, cap_K,
                    steps, every, K_main) -> list:
    def row(uid, block, cond, method, K, lr, cap, root, algo, rule, dep):
        f, m, r = cond
        return OrderedDict([("unit_id", uid), ("block", block), ("family", f), ("m", str(m)), ("rho", gfmt(r)), ("method", method),
                            ("K", str(K) if method.startswith("MF_") else "NA"), ("lr", lr), ("cap", cap if method == "MF_VCS_M1" else "NA"),
                            ("data_root_seed", str(root)), ("algo_seed", str(algo)), ("steps", str(steps)), ("ckpt_every", str(every)),
                            ("selection_rule", rule), ("depends_on", dep)])
    units = []
    sc = (SMOKE_CHOICE["family"], SMOKE_CHOICE["m"], SMOKE_CHOICE["rho"])
    check("smoke condition is a grid condition", sc in conds)
    check("smoke lr is a candidate", SMOKE_CHOICE["lr"] in lrs)
    check("smoke data root is neither the tuning nor a formal root", SMOKE_CHOICE["data_root_seed"] not in [tune_root] + data_roots)
    for i, (mth, cap) in enumerate(SMOKE_PLAN(methods, caps), 1):
        units.append(row(f"T03_SMK_{i:03d}", "smoke", sc, mth, K_main, lr_text(SMOKE_CHOICE["lr"]), str(cap) if cap else "NA",
                         SMOKE_CHOICE["data_root_seed"], SMOKE_CHOICE["algo_seed"], "SMOKE_NONE", ""))
    smoke_ids = ";".join(u["unit_id"] for u in units)
    cal_of = defaultdict(list)
    n = 0
    for cond in conds:
        for mth in methods:
            for lr in lrs:
                for cap in (caps if mth == "MF_VCS_M1" else [None]):
                    n += 1
                    u = row(f"T03_CAL_{n:04d}", "CAL", cond, mth, K_main, lr_text(lr), str(cap) if cap else "NA", tune_root, tune_algo,
                            "CAL_UJ_SELECT", smoke_ids)
                    units.append(u)
                    cal_of[(cond, mth)].append(u["unit_id"])
    n = 0
    for root in data_roots:
        for cond in conds:
            for mth in methods:
                n += 1
                units.append(row(f"T03_MAIN_{n:04d}", "main", cond, mth, K_main, "CAL", "CAL", root, algo_main, "OWN_CAL_UJ_CKPT",
                                 ";".join(cal_of[(cond, mth)])))
    n = 0
    for algo in ex_algos:
        for cond in conds:
            for mth in methods:
                n += 1
                units.append(row(f"T03_ALGO_{n:04d}", "algo_extra", cond, mth, K_main, "CAL", "CAL", data_roots[0], algo, "OWN_CAL_UJ_CKPT",
                                 ";".join(cal_of[(cond, mth)])))
    n = 0
    for root in data_roots:
        for cond in cap_conds:
            for mth in cap_methods:
                for K in cap_K:
                    n += 1
                    units.append(row(f"T03_CAP_{n:04d}", "capacity", cond, mth, K, "CAL_K16", "CAL_K16", root, algo_main, "OWN_CAL_UJ_CKPT",
                                     ";".join(cal_of[(cond, mth)])))
    return units


def build_execution(I) -> list:
    md = I["md"]
    s10 = section(md, "10")
    steps = numbered(s10)
    same("execution order: 5 steps", len(steps), 5)
    par = one(r"(并行开展的 T03 与基线开发使用独立 worktree 或固定提交；BF 作业始终运行已经验收的冻结代码快照。[^\n]+)", s10,
              "parallel development rule").group(1)
    st = {1: "LANDED_NOT_FROZEN (this protocol and the phase-1 report corrections)"}
    return {"steps": [{"step": i + 1, "text_cn": t, "en": EN_STEPS[i + 1], "status": st.get(i + 1, "NOT_RUN")} for i, t in enumerate(steps)],
            "parallel_development_cn": par,
            "parallel_development_en": "T03 and baseline development run in an independent worktree or at a fixed commit; BF jobs always run "
                                       "the accepted frozen code snapshot; merged new code is revalidated under the provenance rule (no "
                                       "PASS carried over from an old code identity); T02 remeasures only the affected paths"}


def implementation_deltas(I) -> list:
    """code changes the decision implies relative to the phase-1 code at the results commit (for decision steps 2-3; all NOT_RUN)"""
    ad = git_blob(I["results_commit"], "src/v10/adapter.py").decode()
    check("phase-1 adapter keeps a separate encoder_lr", "encoder_lr: float = 1e-3" in ad and "head_lr: float = 1e-3" in ad)
    check("phase-1 adapter M1 absolute tolerance", "tolerance: float = M1_PROTOCOL_TOLERANCE" in ad)
    check("phase-1 environment() lacks the two TF32 override variables",
          "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" not in ad and "NVIDIA_TF32_OVERRIDE" not in ad)
    return [
        {"step": 2, "where": "src/v10 (M1Config / mf_objective)", "change": "relative zero-head M1 rule, zero_reference_gap, per-batch log "
         "fields and counters of m1_rule; strict absolute mode kept for the numerical audit", "phase1": "M1Config.tolerance = "
         "M1_PROTOCOL_TOLERANCE (absolute 1e-4)", "status": "NOT_RUN"},
        {"step": 2, "where": "scripts/v10/t01_validate.py", "change": "the five targeted M1 acceptance checks of m1_rule; then every T01 part "
         "rerun and merged under the new code hash", "status": "NOT_RUN"},
        {"step": 2, "where": "src/v10 environment()", "change": "record TORCH_ALLOW_TF32_CUBLAS_OVERRIDE and NVIDIA_TF32_OVERRIDE (raw value "
         "or null)", "phase1": "not recorded", "status": "NOT_RUN"},
        {"step": 2, "where": "scripts/v10/t02_timing.py + v10/T02/BUDGET.json", "change": "remeasure the affected relative-rule M1 "
         "configurations and rewrite BUDGET for the 18-candidate grid", "status": "NOT_RUN"},
        {"step": 3, "where": "src/v10 TrainConfig / Trainer", "change": "one candidate learning rate tied across every optimiser-learned "
         "module (bf_batch1.learning_rate)", "phase1": "encoder_lr 1e-3 (host) separate from head_lr", "status": "NOT_RUN"},
        {"step": 3, "where": "T03 code (independent worktree or fixed commit)", "change": "t03: U mode, uniform derangement sampler and its "
         "enumeration gate, one-hot MF / MLP critics, CAL selection, evaluator, smoke, units of T03_UNITS.csv", "status": "NOT_RUN"},
    ]


def landing_choices() -> list:
    return [
        {"topic": "T03 smoke", "choice": f"condition {case_id(SMOKE_CHOICE['family'], SMOKE_CHOICE['m'], SMOKE_CHOICE['rho'])}; the five "
         "families, MF_VCS_M1 at each cap 10 / 30 / 100 so that every cap's full training cost is measured (7 rows); lr 3e-4 (middle "
         f"candidate) for every smoke row; data root seed {SMOKE_CHOICE['data_root_seed']} (the package generator's default seed, neither "
         f"the tuning suite nor a formal suite); algo seed {SMOKE_CHOICE['algo_seed']}; outputs never enter any selection; not part of "
         "the 1212", "why": "the decision fixes 'one condition, five families, actual full training cost' but not the condition, lr or "
         "data seed; m = 32 with the full-rank diagonal_mixture structure and the largest dependence is the most demanding case for the "
         "training loop; a dedicated data seed keeps the tuning and formal suites untouched", "owner_decision": False},
        {"topic": "unit ids and orders", "choice": "T03_<SMK|CAL|MAIN|ALGO|CAP>_<n> and BF1_<method>_lr<lr>[_cap<cap>] / BF1_<method>_s<seed>, "
         "numbered in the frozen orders listed in t03.units_file.order", "owner_decision": False},
        {"topic": "T03_UNITS.csv conventions", "choice": "lr / cap = CAL (inherit the CAL selection of the same condition and method) or "
         "CAL_K16 (capacity rows); NA where not applicable; depends_on lists the gating unit ids", "owner_decision": False},
        {"topic": "BF budget accounting order", "choice": "within a family, candidates are launched and charged against the per-family cap "
         "in the frozen candidate order, so a budget cut is deterministic; parallel scheduling does not change the order of accounting",
         "owner_decision": False},
        {"topic": "M1 log vocabulary", "choice": "stop_reason uses the phase-1 solver statuses gap_tolerance / iteration_budget_exhausted plus "
         "zero_reference_gap; converged is true for zero_reference_gap (W = 0 is the exact maximiser); tolerance_effective and "
         "gap_relative_to_zero are null there", "owner_decision": False},
        {"topic": "override environment variables", "choice": "recorded as the raw string or JSON null when unset; the key is always "
         "present", "owner_decision": False},
    ]


def step2_landing_choices() -> list:
    """implementation choices of the decision step-2 code (not owner decisions), read as git objects at STEP2_CODE_COMMIT"""
    check(f"git: step-2 code commit {STEP2_CODE_COMMIT[:7]} is an ancestor of HEAD",
          subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", STEP2_CODE_COMMIT, "HEAD"]).returncode == 0)
    ad = git_blob(STEP2_CODE_COMMIT, "src/v10/adapter.py").decode("utf-8")
    m = re.search(r"^M1_FP64_DIAGNOSTIC_BELOW = ([0-9.e+-]+)", ad, re.M)
    check("step-2 adapter: fp64 diagnostic factor M1_FP64_DIAGNOSTIC_BELOW", m is not None)
    check("step-2 adapter: resolution unit = 2 eps(dtype)", "def m1_resolution_unit(dtype) -> float:" in ad
          and "    return 2.0 * torch.finfo(dtype).eps\n" in ad)
    check("step-2 adapter: the diagnostic's HVPs / moment pass enter the per-batch totals",
          '"m1_hvp_calls_total": int(hvp_calls) + diag_hvp' in ad and '"m1_moments_passes_total": int(moments_passes) + diag_mom' in ad)
    check("step-2 adapter: no diagnostic in float64", 'return {"triggered": False, "reason": "working dtype float64", **base}' in ad)
    below = float(m.group(1))
    return [{"topic": "M1 same-input fp64 diagnostic trigger",
             "choice": f"run the diagnostic on a batch when G_b(0) <= {gfmt(below)} x 2 eps(working dtype) (float32: "
                       f"{below * 2 * 2.0 ** -23:.4g}; never in float64); diagnostic only (the training head, W and gradients are not "
                       "touched); its HVPs and its one fp64 moment pass are logged per batch (m1_fp64_diag_*) and counted in "
                       "m1_hvp_calls_total / m1_moments_passes_total; its time is inside the measured training step",
             "where": f"src/v10/adapter.py M1_FP64_DIAGNOSTIC_BELOW and m1_resolution_unit at {STEP2_CODE_COMMIT[:7]}",
             "why": "decision 2.3 keeps a same-input fp64 diagnostic for references near fp32 resolution without fixing the threshold; "
                    "2 eps is the rounding scale of Delta_b and of the FW gap for simplex features with weights summing to 1, and the "
                    "factor keeps the diagnostic on while G_b(0) is within four orders of that scale (it runs at the host init, "
                    "G_b(0) about 3.2e-5)", "owner_decision": False}]


def build(I) -> tuple[str, str]:
    P = OrderedDict()
    P["schema"] = "vcs_qmi.v10.protocol.v2"
    P["protocol_id"] = "V10-P2-20261017"
    P["status"] = "LANDED_NOT_FROZEN"
    P["status_note"] = ("decision section 10 step 1 (protocol landing): the decision as a versioned machine-readable protocol; not a "
                        "runnable configuration.  The batch-1 freeze is step 3, after the relative-M1 code passes every T01 part under its "
                        "new code identity (step 2).  The package V10-server-handoff-v1 stays the historical baseline; this protocol "
                        "modifies it only where the decision says so (supersedes).")
    C = I["contract"]
    P["decision"] = OrderedDict([("id", C["decision_id"]), ("date", C["decision_date"]), ("status", C["status"]),
                                 ("text", {"path": DEC_MD, "sha256": I["inputs"][DEC_MD]}),
                                 ("summary", {"path": DEC_SUMMARY, "sha256": I["inputs"][DEC_SUMMARY]}),
                                 ("contract", {"path": DEC_JSON, "sha256": I["inputs"][DEC_JSON], "schema": C["schema"]}),
                                 ("report_accepted", C["report_accepted"]),
                                 ("options", one(r"五项主建议采用报告选项 ([^，]+)，以及数据角色的 A 立即生效、B 冻结后生效。", I["md"], "options").group(1)),
                                 ("phase1_report", PHASE1_REPORT)])
    same("decision id in the text", one(r"决定版本：([\w-]+)。", I["md"], "decision id").group(1), C["decision_id"])
    P["generator"] = {"path": SELF_REL, "command": f"/home/infres/yinwang/CS_QMI/env/bin/python {SELF_REL}", "check": "--check",
                      "reproducible": "no wall-clock time, no live query, no randomness: byte-identical across runs"}
    P["inputs"] = I["inputs"]
    P["inputs_code"] = OrderedDict((f"{I['results_commit']}:{p}", sha256_bytes(git_blob(I["results_commit"], p))) for p in CODE_AT_RESULTS)
    P["inputs_code"][f"{STEP2_CODE_COMMIT}:src/v10/adapter.py"] = sha256_bytes(git_blob(STEP2_CODE_COMMIT, "src/v10/adapter.py"))
    P["phase1_identity"] = build_identity(I)
    P["supersedes"] = build_supersedes(I)
    P["m1_rule"] = build_m1(I)
    P["precision_environment"] = build_precision(I)
    P["infonce"] = build_infonce(I)
    P["baselines"] = build_baselines(I)
    P["data_roles"] = build_data_roles(I)
    P["tma_colocation_sensitivity"] = build_tma(I)
    P["epfl_activation"] = build_epfl(I)
    P["t01_gates"] = build_gates(I)
    P["bf_batch1"] = build_bf(I)
    t03, units = build_t03(I)
    P["t03"] = t03
    P["execution_order"] = build_execution(I)
    P["implementation_deltas_vs_phase1_code"] = implementation_deltas(I)
    P["landing_choices"] = landing_choices() + step2_landing_choices()
    P["standing_constraints"] = {"CONFIRM": 0, "TEST134": "HOLD, never read", "EPFL": "not activated", "public_repo": "ids and rules only; "
                                 "no data, images, checkpoints or per-image labels"}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(UNIT_COLUMNS)
    for u in units:
        w.writerow([u[c_] for c_ in UNIT_COLUMNS])
    units_text = buf.getvalue()
    P["t03"]["units_file"]["sha256"] = sha256_bytes(units_text.encode("utf-8"))
    P["derivation_checks"] = {"n": len(CHECKS) + 1, "all": "PASS", "rows": CHECKS + [{"check": "every check above passed", "status": "PASS"}]}
    return json.dumps(P, indent=1, ensure_ascii=False) + "\n", units_text


def main() -> int:
    ap = argparse.ArgumentParser(description="V10 protocol P2 generator (writes only the two protocol outputs)")
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the files on disk; write nothing")
    a = ap.parse_args()
    I = load_inputs()
    pj, units = build(I)
    outs = {OUT_JSON: pj, OUT_UNITS: units}
    if a.check:
        bad = [p for p, t in outs.items() if not (REPO / p).exists() or (REPO / p).read_text(encoding="utf-8") != t]
        print(json.dumps({"check": "FAIL" if bad else "PASS", "differs": bad}))
        return 1 if bad else 0
    for p, t in outs.items():
        (REPO / p).parent.mkdir(parents=True, exist_ok=True)
        (REPO / p).write_text(t, encoding="utf-8")
    print(json.dumps({"protocol": OUT_JSON, "protocol_sha256": sha256_bytes(pj.encode())[:16], "units": OUT_UNITS,
                      "units_sha256": sha256_bytes(units.encode())[:16], "unit_rows": units.count("\n") - 1, "checks": len(CHECKS) + 1}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
