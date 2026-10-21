"""Build the frozen TEST_REPEAT134 records (login node / CPU; reads no TEST image or label).

    build_freeze.py           write OWNER_DECISION / MANIFEST / PROTOCOL / ACTIVATION to v10/test_repeat134/ and the bundle
    build_freeze.py --check   rebuild in memory and compare with both copies (exit 1 on a difference)

Every unit record comes from the verified loaders of v10/tools/phase2_report.py (identity chain, artifact hashes, exit
receipts, selection recomputation); each selected checkpoint is hashed again here.  TEST metadata only: the 134 core ids and
the 22 / 112 sensitivity split of protocol P2, the V10 data-manifest TEST rows and V9-G1's frozen member list (sha256 per
raw file).  No TEST file is opened.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402

P2R = C.REPO / "v10/tools/phase2_report.py"
V9_REPO = C.CS / "ssl_pilot_v9"
V9_CORES_REL, V9_CORES_COMMIT = "configs/V9_G1_TEST_CORES.csv", "f19865e2"
V9_TEST_ROOT = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/reg_test")
EXEC_ROOT = C.CS / "ssl_pilot_v10_test_repeat_exec_20261020"          # v1 execution tree (a85cc21b), untouched
EXEC_ROOT_V2 = C.CS / "ssl_pilot_v10_test_repeat_exec_v2_20261020"
V1_COMMIT = "a85cc21be97770c76f3930fbd913705367d7d9d9"
V1_FILES = {"protocol": "TEST_REPEAT134_PROTOCOL.json", "activation": "TEST_REPEAT134_ACTIVATION.json"}
EXEC_ROOT_V4 = C.CS / "ssl_pilot_v10_test_repeat_exec_v4_20261020"
V4_CODE_FILES = ("scripts/v10_test_repeat/tr_common.py", "scripts/v10_test_repeat/evaluate.py",
                 "scripts/v10_test_repeat/prepare_inputs.py", "slurm/v10_test_repeat/gpu.sbatch")     # v4's pinned code
V2_COMMIT = "ca9f517cd89a61190bf7315489d2515e02807c37"
V2_FILES = {"protocol": "TEST_REPEAT134_PROTOCOL_V2.json", "activation": "TEST_REPEAT134_ACTIVATION_V2.json",
            "amendment": "TEST_REPEAT134_GATE_AMENDMENT_V2.json", "delta": "TEST_REPEAT134_DELTA_REVIEW_V2.json"}
# the finished DEV re-evaluations (v1 / v2 gates) that existed when the v4 band was set
OBSERVED_GATE_JOBS_V4 = ("1051029", "1051030", "1051031", "1051151", "1051263", "1051266", "1051274", "1051277", "1051292")
DECISION_V4_TEXT = {
    "question_verbatim": "评估本身有噪声（同一 checkpoint 重评，<9 变动 0.6–2.2 个百分点），TEST_REPEAT134 怎么继续？哪个方案都先登记、后读 TEST。",
    "chosen_option_verbatim": "同作业配对，评一次 (Recommended)",
    "chosen_option_text_verbatim": "每个单元在同一个作业里先重评 DEV32、再评 TEST134，TEST 和这次 DEV 并列报告。门禁改为：checkpoint、输入、扰动和"
                                   "真实位姿能量必须逐位一致，成功次数只粗查，用来拦流程错误。重评数据披露为噪声下限。三个分区都可以跑，约 50 GPU 小时。",
    "options_not_chosen": [
        {"option": "确定性评估（启用确定性算法 / cuDNN 确定性后重评 DEV32 与 TEST）",
         "source": "paraphrase of the alternative prepared for this decision (only the chosen option was relayed verbatim)"},
        {"option": "重复评估取平均（多次重评以平均评估噪声）",
         "source": "paraphrase of the alternative prepared for this decision (only the chosen option was relayed verbatim)"}],
}
# The TEST-stage evaluator path and the L40S criterion: these definitions must be AST-identical to v1 (a85cc21b).
EVALUATOR_DEFS = {
    "scripts/v10_test_repeat/evaluate.py": ("import_path", "make_session", "init_body", "check_session_replica", "load_source",
                                            "environment", "verify_unit_files", "evaluate_unit", "pins"),
    "scripts/v10_test_repeat/tr_common.py": ("TestRepeatGuard", "Budget", "compare_rows", "metrics", "require", "require_slurm",
                                             "unit", "gate_spec", "receipt_ok", "input_lock", "verify_code", "sha", "read",
                                             "write", "lines", "lock", "append", "committed_bytes", "git")}
N_PERT, TEST_REG = 10, 1340
OVERHEAD_S = 180                     # imports, source / data-lock hashing, trainer rebuild with FIT inputs, checkpoint load

DECISION_TEXT = {
    "question_verbatim": "BF 上 V10 全部方法和 seed 的选择已经齐全，可以做 TEST_REPEAT134（读 held-out 的 TEST134）。现在开始吗？",
    "chosen_option_verbatim": "开始 (Recommended)",
    "chosen_option_text_verbatim": "先把所有方法和 seed 选中的 checkpoint 冻结成新 manifest，冻结完成前不读 TEST；然后在 TEST134 上评估。"
                                   "结果单独报告为重复测试，另附 112 个 core 的敏感性集，不当作独立确认。",
}
MEMORY_RULE = ("BF TEST134 曾在 V9-G1 读取，当前仍 HOLD。须完整 V10 方法集合、共同协议及所有方法内选择全部冻结后，创建新 manifest，"
               "以 TEST_REPEAT134 单独报告；不能与 DEV32 合并为独立 confirmation。排除 FIT/CAL 同 position 的未来 TEST_REPEAT "
               "敏感性集为112 cores；不能提前读取它来选方法。")


def load_phase2():
    spec = importlib.util.spec_from_file_location("phase2_report", P2R)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase2_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def family_key(fam: str) -> str:
    return fam.replace("[", "_").replace("]", "")


def ceil_minutes(seconds: float, step: int = 30, minimum: int = 60) -> int:
    return max(minimum, int(math.ceil(seconds / 60 / step)) * step)


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


# ------------------------------------------------------------------------------------------------------------ TEST metadata
def test_metadata(pr, R) -> dict:
    p2 = C.read(pr.REPO / "v10/protocol/V10_PROTOCOL_P2_20261017.json")
    C.require(C.sha(pr.REPO / "v10/protocol/V10_PROTOCOL_P2_20261017.json") == pr.PIN["protocol_p2"], "protocol P2 pin")
    t = p2["tma_colocation_sensitivity"]["TEST_REPEAT"]
    full, kept, excl = list(t["full_ids"]), list(t["retained_ids"]), list(t["excluded_ids"])
    C.require(len(full) == len(set(full)) == 134 and len(kept) == 112 and len(excl) == 22, "P2 TEST_REPEAT counts")
    C.require(sorted(kept) == sorted(set(full) - set(excl)), "P2 retained ids != full - excluded")
    positions = {c.split("_", 1)[1] for c in R["roles"]["FIT"] + R["roles"]["CAL"]}
    C.require(sorted(c for c in full if c.split("_", 1)[1] in positions) == sorted(excl), "22-core exclusion rule recomputation")
    gfull = {g: sorted(v) for g, v in t["position_groups_full"].items()}
    gkept = {g: sorted(v) for g, v in t["position_groups_retained"].items()}
    C.require(len(gfull) == 97 and len(gkept) == 75, "position group counts 97 / 75")
    regroup = {}
    for c in full:
        regroup.setdefault(c.split("_", 1)[1], []).append(c)
    C.require({g: sorted(v) for g, v in regroup.items()} == gfull, "position groups recomputation")
    # V10 data manifest TEST rows (metadata; inclusion=exclude / HOLD) and V9-G1's frozen member list
    man_path = pr.REPO / "v10/T00/V10_DATA_MANIFEST.csv"
    C.require(C.sha(man_path) == pr.PIN["manifest"], "V10 data manifest pin")
    rows = [r for r in csv.DictReader(open(man_path)) if r["dataset"] == "BF_SHG" and r["original_split"] == "test"]
    C.require(sorted(r["record_id"] for r in rows) == sorted(full)
              and all(r["inclusion"] == "exclude" and r["role"] == "DEV" for r in rows), "manifest TEST rows")
    v9_b = subprocess.run(["git", "show", f"{V9_CORES_COMMIT}:{V9_CORES_REL}"], cwd=V9_REPO, capture_output=True,
                          check=True).stdout
    C.require(v9_b == (V9_REPO / V9_CORES_REL).read_bytes(), "V9-G1 core list changed after its freeze commit")
    v9 = {r["core"]: r for r in csv.DictReader(io.StringIO(v9_b.decode()))}
    C.require(sorted(v9) == sorted(full), "V9-G1 core list != P2 TEST ids")
    raw = []
    for r in sorted(rows, key=lambda x: x["record_id"]):
        c, sv = r["record_id"], r["source_version"]
        fixed = sv.split("sha256_fixed=")[1].split(";")[0].strip()
        moving = sv.split("sha256_moving=")[1].split(" ")[0].strip()
        C.require((fixed, moving) == (v9[c]["bf_sha256"], v9[c]["shg_sha256"]), f"{c}: member sha256 differs between records")
        C.require(r["fixed_path"] == str(V9_TEST_ROOT / "raw" / f"{c}_BF.tif")
                  and r["moving_path"] == str(V9_TEST_ROOT / "raw" / f"{c}_SHG.tif"), f"{c}: raw path")
        raw.append({"core": c, "bf_member": v9[c]["bf_member"], "bf_sha256": fixed, "bf_bytes": int(v9[c]["bf_bytes"]),
                    "shg_member": v9[c]["shg_member"], "shg_sha256": moving, "shg_bytes": int(v9[c]["shg_bytes"])})
    cores = sorted(full)
    return {
        "cores": cores, "n_cores": 134, "cores_sha256": C.sha_bytes(C.canonical(cores)),
        "source": {"protocol_p2": {"path": "v10/protocol/V10_PROTOCOL_P2_20261017.json", "sha256": pr.PIN["protocol_p2"],
                                   "keys": "tma_colocation_sensitivity.TEST_REPEAT.full_ids / retained_ids / excluded_ids"},
                   "data_manifest": {"path": "v10/T00/V10_DATA_MANIFEST.csv", "sha256": pr.PIN["manifest"],
                                     "filter": p2["tma_colocation_sensitivity"]["filters"]["TEST_REPEAT"],
                                     "rows": "role DEV, inclusion exclude (HOLD), original_split test; roles unchanged"},
                   "v9_member_list": {"path": str(V9_REPO / V9_CORES_REL), "commit": V9_CORES_COMMIT,
                                      "sha256": C.sha_bytes(v9_b)}},
        "raw": raw, "raw_sha256": C.sha_bytes(C.canonical(raw)),
        "sensitivity112": {"rule": "exclude a TEST core whose TMA position (record_id suffix) occurs among the included FIT or CAL "
                                   "cores (decision 6.1; P2 tma_colocation_sensitivity)",
                           "cores": sorted(kept), "cores_sha256": C.sha_bytes(C.canonical(sorted(kept))),
                           "excluded": sorted(excl), "position_groups": gkept},
        "position_groups": gfull,
        "prior_exposure": p2["data_roles"]["BF_SHG"]["TEST"]["prior_exposure"],
    }


# ---------------------------------------------------------------------------------------------------------------- units
def code_bf1(pr, R) -> dict:
    frozen = R["bf1"]["frozen"]
    return {"kind": "bf1", "source_root": str(pr.BF1 / "source"), "source_commit": frozen["source_commit"],
            "freeze": {"path": str(pr.BF1 / "FROZEN.json"), "sha256": pr.PIN["bf1_frozen"]},
            "core_code_hash": "e9b4239450c4e2eb", "host_code_hash": "6c6f094684fbc50a",
            "runner": "scripts/v10_bf1/run.py", "runner_sha256": frozen["source_files"]["scripts/v10_bf1/run.py"],
            "evaluator": "scripts/v10_bf1/run.py: new_trainer, load_checkpoint, evaluate (unchanged)"}


def code_supp(release_path: Path, release_sha: str) -> dict:
    C.require(C.sha(release_path) == release_sha, f"{release_path}: release pin")
    rel = C.read(release_path)
    return {"kind": "supp", "source_root": rel["source_root"], "source_commit": rel["source_commit"],
            "release": {"path": str(release_path), "sha256": release_sha}, "code_sha256": rel["code_sha256"],
            "runner": "scripts/v10_supplement/run.py", "runner_sha256": rel["source_files"]["scripts/v10_supplement/run.py"],
            "registration_sha256": rel["source_files"]["src/v10/supplement/registration.py"],
            "evaluator": "scripts/v10_supplement/run.py: new_trainer, CP.load, evaluate (unchanged); "
                         "RegistrationSession rebound to TestRepeatSession (role gate only)"}


def unit_record(pr, u, family, seed, tier, row_label, code, selection) -> dict:
    d = Path(u["dir"])
    comp = u["complete"]
    sel = comp["selected"]
    ck = d / sel["checkpoint"]
    C.require(C.sha(ck) == sel["checkpoint_sha256"] == comp["artifacts"][sel["checkpoint"]], f"{d}: selected checkpoint")
    C.require(C.sha(d / "COMPLETE.json") == u["complete_sha"], f"{d}: COMPLETE")
    dev_rows = C.lines(d / "DEV.jsonl")
    C.require(C.sha(d / "DEV.jsonl") == comp["artifacts"]["DEV.jsonl"] and len(dev_rows) == 320, f"{d}: DEV rows")
    dev_ident = C.read(d / "DEV.identity.json")
    dep = comp["identity"].get("selection_dependency_sha256", comp["identity"].get("dependency"))
    C.require(dep == selection["sha256"], f"{d}: selection dependency differs from the selection record")
    C.require(C.metrics(dev_rows) == comp["DEV32"], f"{d}: DEV32 recomputation")
    gpu_type = C.GPU_TYPES[u["gpu"]]
    dev_s = sum(r.get("full_registration_seconds", r["seconds"]) for r in dev_rows)
    est = dev_s * TEST_REG / 320 + OVERHEAD_S
    return {
        "key": f"{family_key(family)}__{seed}", "family": family, "seed": seed, "tier": tier, "row_label": row_label,
        "unit_dir": str(d), "complete_sha256": u["complete_sha"], "identity": comp["identity"], "selection": selection,
        "checkpoint": {"path": sel["checkpoint"], "sha256": sel["checkpoint_sha256"], "step": sel["step"],
                       "cal_metrics": sel["metrics"]},
        "dev": {"rows_sha256": comp["artifacts"]["DEV.jsonl"], "cores": dev_ident["cores"], "DEV32": comp["DEV32"],
                "DEV24": comp["DEV24"], "registration_seconds": dev_s},
        "code": code,
        "hardware": {"gpu_name": u["gpu"], "gpu_type": gpu_type, "host": u["host"], "partition": u["partition"],
                     "job": u["job"]},
        "test_gpu_type": gpu_type,
        "estimate": {"test_seconds": est, "test_time_limit": hhmm(ceil_minutes(1.5 * est + 1800)),
                     "gate_seconds": dev_s + OVERHEAD_S, "gate_time_limit": hhmm(ceil_minutes(1.5 * (dev_s + OVERHEAD_S) + 1200))},
    }


def build_units(pr, R) -> list:
    units = []
    bf1_code = code_bf1(pr, R)
    for fam in pr.BF1_FAMILIES:
        f = R["fams"][fam]
        sel = {"path": str(pr.BF1 / "results/CAL" / fam / "SELECTION.json"), "sha256": pr.SELECTION_PIN[fam]}
        for seed in pr.FINAL_SEEDS:
            units.append(unit_record(pr, f["finals"][seed], fam, seed, "primary", "frozen", bf1_code, sel))
    for fam, bundle, idx in pr.SUPP:
        f = R["fams"][fam]
        froot = bundle / "families" / str(idx)
        code = code_supp(froot / "RELEASE.json", pr.RELEASE_PIN[fam])
        sel = {"path": str(froot / "results/CAL" / fam / "SELECTION.json"), "sha256": pr.SELECTION_PIN[fam]}
        label = "failed_adaptation" if fam == pr.FAILED_V1 else "frozen"
        for seed, u in sorted(f["finals"].items()):
            units.append(unit_record(pr, u, fam, seed, "primary", label, code, sel))
    rv2 = R["rv2"]
    C.require(rv2["state"] == "complete", "RuLSIF_exact_v2 finals are not complete")
    code = code_supp(pr.RV2 / "RELEASE_XHW.json", pr.PIN["rv2_release_xhw"])
    sel = {"path": str(pr.RV2 / "results/CAL" / pr.RV2_FAMILY / "SELECTION.json"), "sha256": pr.PIN["rv2_selection"]}
    for seed, u in sorted(rv2["finals"].items()):
        units.append(unit_record(pr, u, pr.RV2_FAMILY, seed, "primary", "post_gate_amendment_A1", code, sel))
    proto = R["rep"]["protocol"]
    for u in sorted(R["rep"]["units"], key=lambda x: (pr.ORDER.index(x["method"]), x["seed"])):
        m = proto["methods"][u["method"]]
        if m["kind"] == "bf1":
            C.require(m["freeze"]["sha256"] == pr.PIN["bf1_frozen"] and Path(m["source_root"]) == pr.BF1 / "source",
                      "replication BF1 freeze / source")
            code = bf1_code
        else:
            code = code_supp(Path(m["release"]["path"]), m["release"]["sha256"])
        sel = {"path": m["selection"]["path"], "sha256": m["selection"]["sha256"]}
        units.append(unit_record(pr, u, u["method"], u["seed"], "secondary_replication", "frozen", code, sel))
    C.require(len(units) == 42 and len({u["key"] for u in units}) == 42, "expected 42 unique frozen final units")
    return units


# ---------------------------------------------------------------------------------------------------------- the records
def build(pr) -> dict:
    R = pr.collect(verify_pt=False)
    test = test_metadata(pr, R)
    units = build_units(pr, R)
    families = list(dict.fromkeys(u["family"] for u in units))
    manifest = {
        "schema": "v10.test_repeat134.manifest.v1", "status": "FROZEN_BEFORE_ANY_TEST_READ",
        "id": "V10-TEST-REPEAT134-MANIFEST-20261020",
        "version_note": "new manifest version for the BF TEST_REPEAT134 repeated benchmark (decision 6.1; P2 data_roles); the "
                        "V10 data manifest and its roles are unchanged (TEST rows stay role DEV / inclusion exclude there)",
        "test": test, "families": families,
        "units": units,
        "unit_counts": {"total": len(units), "primary": sum(u["tier"] == "primary" for u in units),
                        "secondary_replication": sum(u["tier"] == "secondary_replication" for u in units)},
        "excluded_from_test_repeat": [
            {"what": "post-DEV sensitivity D1 (CoMIR_native epoch chosen on CAL)", "why": "post-DEV sensitivity; never a main row"},
            {"what": "post-DEV sensitivity D2 (MIND_2D_warp_then_describe)", "why": "named post-DEV variant; never a main row"},
            {"what": "post-DEV D3 cross-hardware gate reruns", "why": "validation only"},
            {"what": "K8/K32 capacity extension (18 units)", "why": "separate capacity table; not a main row"},
            {"what": "RuLSIF_exact_v2 post-hoc hardware replicate (hw_replicates, seed 1001 on L40S)", "why": "post-hoc replicate; not a main row"},
            {"what": "CAL candidate units (seed 1017)", "why": "selection inputs, not final rows"},
            {"what": "validation / GPU-gate / smoke units", "why": "excluded from science by their releases"},
            {"what": "EPFL", "why": "separate benchmark, never pooled with BF"}],
        "verification": {"generator": "v10/tools/phase2_report.py collect(verify_pt=False) at analysis commit eaec886d",
                         "generator_sha256": C.sha(P2R),
                         "checkpoint_snapshot": "v10/tools/phase2_report_checkpoint_check.json (985 BF checkpoints re-hashed, all match)",
                         "selected_checkpoints_rehashed_here": len(units)},
    }
    plan_gate, seen = [], set()
    for u in sorted(units, key=lambda x: (x["test_gpu_type"] != "L40S", x["test_gpu_type"], families.index(x["family"]), x["seed"])):
        pair = (u["family"], u["test_gpu_type"])
        if pair in seen:
            continue
        seen.add(pair)
        plan_gate.append({"gate_id": f"{u['test_gpu_type']}__{family_key(u['family'])}", "family": u["family"],
                          "gpu_type": u["test_gpu_type"], "unit": u["key"], "unit_recorded_gpu": u["hardware"]["gpu_name"],
                          "estimated_seconds": u["estimate"]["gate_seconds"], "time_limit": u["estimate"]["gate_time_limit"]})
    by_type = {}
    for u in units:
        by_type.setdefault(u["test_gpu_type"], 0.0)
        by_type[u["test_gpu_type"]] += u["estimate"]["test_seconds"] / 3600
    gate_by_type = {}
    for g in plan_gate:
        gate_by_type[g["gpu_type"]] = gate_by_type.get(g["gpu_type"], 0.0) + g["estimated_seconds"] / 3600
    precision = C.read(pr.BF1 / "PROTOCOL.json")["precision"]
    protocol = {
        "schema": "v10.test_repeat134.protocol.v1", "status": "FROZEN_BEFORE_ANY_TEST_READ",
        "id": "V10-TEST-REPEAT134-PROTOCOL-20261020",
        "rules": {
            "owner_decision": DECISION_TEXT,
            "decision_6_1": pr.quote(R["dec"], "6.1", "134 个官方 TEST core 已由 G1 读过，当前仍 HOLD；V10 完整协议冻结后，以新 manifest 版本启用单独的 TEST_REPEAT134 表。DEV 与 TEST_REPEAT 不合并成 166 个独立确认样本。"),
            "decision_5": pr.quote(R["dec"], "5", "完整方法集合和方法内选择全部固定以后才能启用 TEST 重复块。"),
            "decision_5_claims": pr.quote(R["dec"], "5", "必需基线补齐之前不发布方法优劣、强基线优势或新颖性比较结论。"),
            "protocol_p2_TEST": {k: v for k, v in C.read(pr.REPO / "v10/protocol/V10_PROTOCOL_P2_20261017.json")["data_roles"]["BF_SHG"]["TEST"].items()},
            "project_memory": MEMORY_RULE,
            "evaluator_only": "TEST_REPEAT134 rows and summaries are evaluator-only: never used for training, CAL selection, "
                              "checkpoint choice, grid / candidate / protocol changes or method re-selection",
        },
        "data": {"benchmark": "BF/SHG official test split (Zenodo 4550300 test.zip), 134 cores; prior exposure: "
                              + test["prior_exposure"], "cores_sha256": test["cores_sha256"],
                 "sensitivity112_sha256": test["sensitivity112"]["cores_sha256"], "unit": "core (one TMA core = one BF/SHG pair)",
                 "perturbations_per_core": N_PERT, "registrations_per_unit": TEST_REG},
        "perturbations": {
            "rule": "the frozen DEV rule, applied to the TEST core ids: Q.perturbations_p2(core) = REG-P1 translations "
                    "(seed sha256('R6-REG-P1|<core>')[:16], magnitude strata) with theta uniform on [-30, 30] deg "
                    "(seed sha256('R6-REG-P2|<core>')[:16]); per-registration position stream seed_of('R6-REG-P2|pos|<core>|<i>'); "
                    "MLP extra stream seed_of('V10|MLP-extra-registration-perms|<core>|<i>')",
            "code": {"r6_reg_p2.py": R["bf1"]["frozen"]["source_files"]["src/v9/reg/r6_reg_p2.py"],
                     "r6_reg_pilot.py": R["bf1"]["frozen"]["source_files"]["src/v9/reg/r6_reg_pilot.py"],
                     "host_code_hash": "6c6f094684fbc50a"},
            "computed": "only inside the TEST jobs (the guard gates each core like a read); every TEST row records (theta, t) and "
                        "all units must share them (analysis check)",
            "difference_from_V9_G1": "V9-G1's sealed-test read used new seed tags (V9-G1-TEST-20261015|P1 / |P2); TEST_REPEAT134 "
                                     "uses V10's DEV rule unchanged, so the perturbations are not V9-G1's"},
        "searcher": "unchanged R6-REG-P2 (Q.optimise_rigid: coarse grid, top 3 starts, 100 Adam steps), as every V10 DEV evaluation",
        "metrics": {"err_full": "corner error at full resolution (px), Q.corner_error_full", "success42": "mean(err_full < 42)",
                    "success9": "mean(err_full < 9)", "median": "median(err_full)",
                    "same_as": "DEV32 (BF1 common.metrics == supplement records.metrics)"},
        "sets": {"TEST134": "all 134 cores x 10 perturbations",
                 "TEST112": "112 cores without a FIT/CAL TMA position (sensitivity set; the 22 excluded ids are frozen)",
                 "position_groups": {"TEST134": 97, "TEST112": 75}},
        "statistics": {
            "primary_unit": "core; methods paired by core; the 10 perturbations of a core are not independent",
            "per_unit": "TEST134 and TEST112 metrics and per-core metrics of every frozen unit",
            "families": "primary seeds 1001-1003 summarised as seed means (hardware strata where mixed); secondary replication "
                        "seeds 1004/1005 reported per unit and by hardware, never pooled with the primary seeds",
            "paired_contrasts": [list(c) for c in pr.CONTRASTS],
            "paired_rule": "phase2_report.paired: core score = mean over the method's primary units of the per-core success "
                           "fraction (median error: median over cores of the per-core difference); RuLSIF_exact (v1) and "
                           "RuLSIF_exact_v2 (A1, mixed hardware) are not contrast arms",
            "bootstrap": {"B": pr.N_BOOT, "resampled": "cores with replacement; sensitivity: TMA position groups with replacement",
                          "seed": "int(sha256('v10-phase2|TEST_REPEAT134|<A>|<B>|<set>|<comma-joined core list>')[:16], 16), "
                                  "numpy.random.default_rng", "interval": "percentile 2.5 / 97.5", "tests": "none"},
            "dev_beside": "each unit's recorded DEV32 / DEV24 (and phase-2 DEV contrasts) printed beside, never pooled"},
        "reporting": {
            "label": "TEST_REPEAT134 (repeated benchmark; BF/SHG official test cores read once before by V9-G1; not an "
                     "independent confirmation)",
            "rules": ["reported as its own table, separate from DEV32; DEV32 / DEV24 shown beside it",
                      "DEV and TEST_REPEAT are never pooled (no 166-sample confirmation)",
                      "no independent-confirmation claim; prior exposure V9-G1 stated",
                      "TEST112 sensitivity reported beside TEST134",
                      "RuLSIF_exact (v1) carries its failed-adaptation label; RuLSIF_exact_v2 carries amendment A1's label",
                      "hardware recorded per unit; secondary replication by hardware",
                      "no method-superiority, strong-baseline or novelty claim while the required baselines are not "
                      "confirmed complete (decision 5)",
                      "intervals are descriptive; no significance tests"]},
        "precision": precision,
        "hardware": {"rule": "matched hardware: each unit is TEST-evaluated on the GPU type that produced its recorded DEV32 "
                             "(L40S / RTX6000PRO / H100), only after the DEV gate of (family, that GPU type) PASSed",
                     "owner_quotes": ["final 放开，CAL 留 L40S", "什么快什么来吧"], "exclude_nodes": C.EXCLUDE_NODES,
                     "cross_hardware_equivalence": "not claimed"},
        "dev_gate": {
            "criterion": "bitwise: re-evaluate the gate unit's frozen selected checkpoint on DEV32 (32 cores x 10) with the "
                         "TEST-stage evaluator on the GPU type that produced its record; every (core, perturbation) row present "
                         "once, the same field names, and every non-timing field equal (==); DEV32 metrics equal the recorded "
                         "COMPLETE DEV32; data read only FIT / CAL / DEV",
            "timing_fields_excluded": list(C.TIMING_FIELDS),
            "scope": "one gate per (family, GPU type) used by a TEST unit; a TEST unit runs only if its gate PASSed",
            "on_fail": "a FAIL is reported with its first mismatches; that family's units on that GPU type are not TEST-evaluated "
                       "under this protocol; no tolerance is substituted after the fact (a criterion change needs a new protocol "
                       "version and activation before any TEST read of those units); other gates and units are unaffected",
            "plan": plan_gate},
        "inputs": {"raw_root": str(V9_TEST_ROOT / "raw"), "raw_rule": "V9-G1 byte copies of the sealed archive members; sha256 "
                   "must equal the frozen member list", "v9_prepared_root": str(V9_TEST_ROOT),
                   "root": str(C.OUT / "inputs"), "builder": "REG-P1 cmd_prepare verbatim (V9-G1's prepare path: core discovery "
                   "and role assignment pointed at the test files, role TEST)",
                   "host_source_root": str(pr.BF1 / "source"), "host_code_hash": "6c6f094684fbc50a",
                   "lock": "INPUT_LOCK.json (sha256 per prepared file); PASS requires the arrays to equal V9-G1's prepared cache",
                   "precondition": "every L40S family gate PASS"},
        "jobs": {"resume": "a TEST job that stops (time limit, node fault) is resubmitted with the same line; the frozen evaluator "
                           "resumes from the completed rows of the same output, which are never replaced; COMPLETE is written once",
                 "order": ["gate (all plan entries)", "prepare (after every L40S gate PASS)",
                           "test (one job per unit, after its gate PASS and INPUT_LOCK PASS)"],
                 "sbatch": "slurm/v10_test_repeat/gpu.sbatch", "execution_root": str(EXEC_ROOT),
                 "outputs": {"gate": str(C.BUNDLE / "gate/<gate_id>/<job>/"), "inputs": str(C.OUT / "inputs"),
                             "test": str(C.BUNDLE / "results/TEST_REPEAT/<unit>/"), "exits": str(C.BUNDLE / "exits/<job>.json"),
                             "failures": str(C.BUNDLE / "failures/<job>.json")},
                 "test_units": [{"unit": u["key"], "gpu_type": u["test_gpu_type"], "time_limit": u["estimate"]["test_time_limit"],
                                 "estimated_seconds": u["estimate"]["test_seconds"]} for u in units]},
        "cost_estimate": {
            "basis": "recorded DEV registration seconds of the same unit x 1340 / 320, plus a fixed per-job overhead of "
                     f"{OVERHEAD_S} s; native allocated GPU-h on the matched GPU type; prepare 0.25 h",
            "test_gpu_h": by_type, "gate_gpu_h": gate_by_type, "prepare_gpu_h": 0.25,
            "total_gpu_h": sum(by_type.values()) + sum(gate_by_type.values()) + 0.25},
    }
    decision = {"schema": "v10.owner_decision.v1", "id": "V10-TEST-REPEAT134-GO-20261020", "date": "2026-10-20",
                "time_utc": "about 06:45", **DECISION_TEXT,
                "scope": "BF TEST_REPEAT134 of the complete V10 BF method set; EPFL is not affected",
                "relayed": "by the coordinating session to the TEST_REPEAT lead, who records it verbatim"}
    return {"decision": decision, "manifest": manifest, "protocol": protocol}


def activation(shas: dict, code_files: dict) -> dict:
    return {
        "schema": "v10.test_repeat134.activation.v1", "status": "ACTIVE_FOR_FROZEN_CHECKPOINT_EVALUATION_ONLY",
        "id": "V10-TEST-REPEAT134-ACTIVATION-20261020",
        "owner_decision": {"path": f"{C.RECORDS}/{C.DECISION}", "sha256": shas[C.DECISION]},
        "records": shas, "code": {"files": code_files, "execution_root": str(EXEC_ROOT)},
        "permits": "reading the 134 BF/SHG TEST cores (their raw members once in the prepare job, then the locked prepared "
                   "inputs) solely to evaluate the frozen selected checkpoints of the manifest with the frozen evaluators",
        "conditions": [
            "inside a Slurm GPU job submitted with V10_TR_EXPECT_ACTIVATION = sha256 of this file and V10_TR_EXPECT_COMMIT = the "
            "execution worktree commit; the records, the code files above and the bundle copies are byte-identical",
            "prepare: every planned L40S DEV gate PASS; raw member sha256 == the manifest; prepared arrays == V9-G1's prepared cache",
            "test: the (family, GPU type) DEV gate PASS; the job's GPU type == the unit's recorded DEV32 GPU type; INPUT_LOCK PASS; "
            "frozen checkpoint sha256 == the manifest",
            "results are written once and never overwritten; every TEST read is logged"],
        "forbidden": ["any TEST access outside these jobs (login node, CPU tests, interactive use)",
                      "training, CAL selection, checkpoint choice, grid / candidate / protocol changes from TEST content",
                      "pooling TEST_REPEAT with DEV or calling it an independent confirmation",
                      "re-running a unit to replace a result"],
        "supersedes": "TEST134 HOLD (protocol P2 data_roles.BF_SHG.TEST) for this evaluation only; the V10 data manifest is unchanged",
        "evaluator_only": True}


def render(pr) -> dict:
    recs = build(pr)
    out = {C.DECISION: C.dumps(recs["decision"]), C.MANIFEST: C.dumps(recs["manifest"]), C.PROTOCOL: C.dumps(recs["protocol"])}
    shas = {k: C.sha_bytes(v.encode()) for k, v in out.items()}
    code_files = {rel: C.sha(C.REPO / rel) for rel in C.CODE_FILES}
    out[C.ACTIVATION] = C.dumps(activation(shas, code_files))
    return out


# ------------------------------------------------------------------------------------------------- version 2 (gate amendment)
def v1_bytes(rel: str) -> bytes:
    return subprocess.run(["git", "show", f"{V1_COMMIT}:{rel}"], cwd=C.REPO, capture_output=True, check=True).stdout


def top_defs(source: str) -> dict:
    tree = ast.parse(source)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def json_diff(a, b, path="") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else str(k)
            out += ([{"path": p, "change": "added"}] if k not in a else [{"path": p, "change": "removed"}] if k not in b
                    else json_diff(a[k], b[k], p))
        return out
    return [] if a == b else [{"path": path, "change": "changed"}]


def evidence() -> dict:
    epfl_am_path = C.CS / "ssl_pilot_v10_epfl_release_hw2_20261020/v10/epfl/EPFL_XHW_GATE_AMENDMENT_20261020.json"
    am = C.read(epfl_am_path)
    C.require(C.sha(epfl_am_path) == "cd673b7cbd5b87bb974be3b797d8686b6efe3c0a59ff497d7be2707484bdc7c1", "EPFL amendment pin")
    rows = []
    for gp in sorted((C.HAND / "v10_epfl_production_hw2_20261020/xhw_gates").glob("*/gpu_gate/*/GATE.json")):
        g = C.read(gp)
        for name in sorted(am["rule"]["changed_rows"]):
            r = g["checkpoint_rows_measured"][name]
            rows.append({"job": g["environment"]["slurm_job_id"], "gpu": g["environment"]["gpu_name"], "row": name,
                         "gradient_bitwise": r["gradient"]["bitwise"], "gradient_max_rel": r["gradient"]["max_rel_err"],
                         "energy_bitwise": r["energy"]["bitwise"], "gate_status": g["status"], "record_sha256": C.sha(gp)})
    d3_path = C.HAND / "v10_sensitivity_20261019/D3/validation/runs/1048684/MLP_VCS_perm8/GATE.json"
    d3 = C.read(d3_path)["resume"]
    adapter = (C.HAND / "v10_bf1_20261017/source/src/v10/adapter.py").read_text()
    tol = re.search(r'^TOLERANCES = (\{.*?\})', adapter, re.M).group(1)
    cap = C.read(C.HAND / "v10_capacity_ext_20261019/hw_amendment_20261019/HARDWARE_AMENDMENT.json")["authority"]
    return {
        "owner_rules": [
            {"quote": cap["owner_choice_verbatim"].strip("「」"), "date": cap["decision_date"],
             "record": "handoff/v10_capacity_ext_20261019/hw_amendment_20261019/HARDWARE_AMENDMENT.json authority.owner_choice_verbatim",
             "sha256": C.sha(C.HAND / "v10_capacity_ext_20261019/hw_amendment_20261019/HARDWARE_AMENDMENT.json")},
            {"quote": am["owner"]["words_verbatim"], "date": am["date"],
             "record": f"{epfl_am_path} owner.words_verbatim", "sha256": C.sha(epfl_am_path)}],
        "precedents": {
            "T01_tolerance": {"float32_atol_rtol": tol, "source": "src/v10/adapter.py TOLERANCES (package V10_DESIGN 6.1), the "
                              "accepted V10 T01 numeric tolerance (BF1 frozen source)"},
            "EPFL_XHW_GATE_AMENDMENT_20261020": {"path": str(epfl_am_path), "sha256": C.sha(epfl_am_path), "label": am["label"],
                                                 "failed_gate": am["failed_gate"]["job"], "rows_judged_with_T01_tolerance":
                                                 sorted(am["rule"]["changed_rows"])}},
        "non_bitwise_observations": {
            "EPFL_amended_gates_checkpoint_rows": rows,
            "D3_1048684_MLP_VCS_perm8_RTX6000PRO_resume": {
                "record": str(d3_path), "sha256": C.sha(d3_path),
                "expected_second_grad_norm_preclip": d3["expected_second"]["grad_norm_preclip"],
                "resumed_second_grad_norm_preclip": d3["resumed_second"]["grad_norm_preclip"],
                "loss_bitwise": d3["expected_second"]["loss"] == d3["resumed_second"]["loss"]}},
    }


def calibration() -> dict:
    """The coordinator-cited cross-hardware evidence: RuLSIF_exact_v2 seed 1001, RTX run vs its post-hoc L40S rerun (DEV rows
    of both units; DEV data only)."""
    base = C.HAND / "v10_rulsif_v2_20261019"
    ag_path = base / "hw_replicates/AGREEMENT_seed_1001.json"
    ag = C.read(ag_path)
    prim = C.lines(Path(ag["primary"]["unit"]) / "DEV.jsonl")
    repl = C.lines(Path(ag["replicate"]["unit"]) / "DEV.jsonl")
    C.require(C.sha(Path(ag["primary"]["unit"]) / "COMPLETE.json") == ag["primary"]["complete_sha256"]
              and C.sha(Path(ag["replicate"]["unit"]) / "COMPLETE.json") == ag["replicate"]["complete_sha256"],
              "calibration units changed after their agreement record")
    d = C.gate_diagnostics(prim, repl, {"unit": "same_identity"}, {"unit": "same_identity"})
    return {"record": str(ag_path), "sha256": C.sha(ag_path), "kind": ag["label"]["kind"],
            "primary": {"job": ag["primary"]["job"], "gpu": ag["primary"]["gpu"]},
            "replicate": {"job": ag["replicate"]["job"], "gpu": ag["replicate"]["gpu"]},
            "same_identity": ag["same_identity"], "fitted_model_equal": ag["fit"]["model_sha256_equal"],
            "perturbation_truths_bitwise": d["perturbation_fields_equal"] or all(
                (a["theta"], a["t"]) == (b["theta"], b["t"]) for a, b in zip(sorted(prim, key=lambda r: (r["core"], r["perturbation"])),
                                                                             sorted(repl, key=lambda r: (r["core"], r["perturbation"])))),
            "DEV32_replicate_minus_primary": ag["DEV32"]["replicate_minus_primary"],
            "computed_here_from_the_two_DEV_records": {"abs_delta_err_full": d["abs_delta_err_full"], "flips": d["flips"],
                                                       "counts": d["counts"], "median": d["median"],
                                                       "verdict_under_v2": C.gate_verdict("RTX6000PRO", False, False, True, {
                                                           **d, "identity_equal": True, "perturbation_fields_equal": True})}}


def build_v2(pr) -> dict:
    v1p_b, v1a_b = (C.REPO / C.RECORDS / V1_FILES["protocol"]).read_bytes(), (C.REPO / C.RECORDS / V1_FILES["activation"]).read_bytes()
    C.require(v1p_b == v1_bytes(f"{C.RECORDS}/{V1_FILES['protocol']}") and v1a_b == v1_bytes(f"{C.RECORDS}/{V1_FILES['activation']}"),
              "v1 protocol / activation differ from a85cc21b")
    recs_v1 = build(pr)
    checked = ((C.DECISION, "decision"), (C.MANIFEST, "manifest"))
    if C.BUNDLE == C.HAND / "v10_test_repeat134_20261020":       # the v1 protocol embeds the bundle path
        checked += ((V1_FILES["protocol"], "protocol"),)
    for name, key in checked:
        C.require(C.dumps(recs_v1[key]) == (C.REPO / C.RECORDS / name).read_text() == v1_bytes(f"{C.RECORDS}/{name}").decode(),
                  f"{name}: rebuilt at HEAD differs from the v1 freeze (the unit set and v1 rules must be unchanged)")
    # pre-registration state: no RTX6000PRO / H100 gate record, no input lock and no TEST result exists yet
    gdir = C.BUNDLE / "gate"
    C.require(all(d.name.startswith("L40S__") for d in gdir.iterdir()) if gdir.exists() else True, "an RTX6000PRO / H100 gate record exists")
    C.require(not (C.BUNDLE / "INPUT_LOCK.json").exists() and not (C.BUNDLE / "results").exists(), "TEST inputs or results exist")
    v1p = json.loads(v1p_b)
    l40s = [g["gate_id"] for g in v1p["dev_gate"]["plan"] if g["gpu_type"] == "L40S"]
    p2 = copy.deepcopy(v1p)
    p2["schema"] = "v10.test_repeat134.protocol.v2"
    p2["id"] = "V10-TEST-REPEAT134-PROTOCOL-V2-20261020"
    p2["status"] = "FROZEN_BEFORE_ANY_RTX6000PRO_OR_H100_GATE_AND_BEFORE_ANY_TEST_READ"
    p2["version"] = {"number": 2, "supersedes": {"path": f"{C.RECORDS}/{V1_FILES['protocol']}", "sha256": C.sha_bytes(v1p_b),
                                                 "commit": V1_COMMIT},
                     "amendment": f"{C.RECORDS}/{C.AMENDMENT}", "delta_review": f"{C.RECORDS}/{C.DELTA}",
                     "change": "the DEV-gate criterion of the RTX6000PRO and H100 gates only (plus the version fields and the "
                               "execution root of the new frozen execution tree)"}
    tol = {"a": "every identity (evaluation identity file), configuration (trainer rebuild and checkpoint load), "
                "checkpoint-hash, input-hash (DATA_LOCK) and perturbation-truth field (theta, t per registration; also core, "
                "perturbation, role, checkpoint_sha256, n_grid) is bitwise identical, and every value is finite",
           "b": "|delta DEV32 success<42| <= 3/320 and |delta DEV32 success<9| <= 3/320",
           "c": "|delta DEV32 median err_full| <= 0.25 (full-resolution px)",
           "d": "the median over the 320 registrations of |delta err_full| <= 0.25",
           "thresholds": dict(C.TOLERANCE),
           "intent": "catch a wrong checkpoint, wrong inputs or wrong perturbations (they shift the aggregates and the typical "
                     "error by whole pixels) while admitting benign cross-hardware numerical differences, which move single "
                     "pose estimates but barely the aggregate metrics (calibration: the amendment record)"}
    p2["dev_gate"]["criterion_by_gpu_type"] = {
        "L40S": v1p["dev_gate"]["criterion"],
        "RTX6000PRO": "PASS if bitwise (the L40S criterion); otherwise PASS_TOLERANCE iff (a)-(d) all hold; otherwise FAIL",
        "H100": "PASS if bitwise (the L40S criterion); otherwise PASS_TOLERANCE iff (a)-(d) all hold; otherwise FAIL"}
    p2["dev_gate"]["tolerance_rtx6000pro_h100"] = tol
    p2["dev_gate"]["passing_status"] = {k: list(v) for k, v in C.PASSING.items()}
    p2["dev_gate"]["recorded_by_every_gate"] = ["per-field bitwise flags and the count of bitwise-equal rows",
                                                "q50, q90, q95, q99 and max of |delta err_full|, the count within 0.05 and the "
                                                "count identical", "success<42 / success<9 flip counts",
                                                "F_hat / F_true / theta_hat / t_hat deltas (max abs, max rel); registration rows "
                                                "carry no gradient", "DEV32 success counts and median, recorded and re-evaluated"]
    p2["dev_gate"]["accepted_v1_l40s_gate_records"] = {"activation_sha256": C.sha_bytes(v1a_b), "code_commit": V1_COMMIT,
                                                       "gate_ids": l40s}
    p2["jobs"]["execution_root"] = str(EXEC_ROOT_V2)
    am = {"schema": "v10.test_repeat134.gate_amendment.v2", "id": "V10-TEST-REPEAT134-GATE-AMENDMENT-V2-20261020",
          "label": "PRE-REGISTERED before any RTX6000PRO / H100 DEV gate and before any TEST read: fallback criterion for the "
                   "RTX6000PRO and H100 DEV gates only; L40S stays bitwise and unchanged",
          "state_when_registered": "the 12 L40S DEV gates were submitted under v1 (jobs 1051029-1051040, all pending, no gate "
                                   "record); no RTX6000PRO / H100 gate, no prepare and no TEST job had been submitted; no TEST "
                                   "data had been read",
          "reason": "fp32 values after reload are not bitwise on RTX PRO 6000 or H100 (T01 cross-hardware diagnostics, D3, the "
                    "EPFL gates); a bitwise DEV-reproduction gate would fail on that ground alone, before saying anything about "
                    "the evaluator",
          "revision_before_freeze": "an earlier draft of (b)-(d) (exact counts, median within 0.05, >= 99 % of rows within "
                                    "0.05) was replaced before anything was frozen or run: the calibration evidence shows a "
                                    "benign cross-hardware rerun with only a minority of registrations within 0.05",
          "calibration": calibration(),
          "criterion": {"RTX6000PRO_and_H100": p2["dev_gate"]["criterion_by_gpu_type"]["RTX6000PRO"], "tolerance": tol,
                        "fail": "FAIL keeps that family's units on that GPU type blocked (v1 on_fail rule)",
                        "L40S": "bitwise, unchanged; the v1 L40S gate records of a85cc21b are accepted (identical evaluator and criterion)"},
          "reporting": "every TEST unit carries the status of the gate it ran under (PASS or PASS_TOLERANCE); no cross-hardware "
                       "equivalence is claimed",
          **evidence()}
    return {"protocol": p2, "amendment": am, "v1_protocol": v1p, "v1_protocol_bytes": v1p_b, "v1_activation_bytes": v1a_b,
            "l40s": l40s}


def delta_review(b2: dict) -> dict:
    files = {}
    for rel in C.CODE_FILES + ("scripts/v10_test_repeat/analyze.py", "scripts/v10_test_repeat/build_freeze.py",
                                "scripts/v10_test_repeat/submit_lines.py", "scripts/v10_test_repeat/status.py"):
        new = (C.REPO / rel).read_bytes()
        try:
            old = v1_bytes(rel)
        except subprocess.CalledProcessError:
            old = None
        entry = {"v1_sha256": C.sha_bytes(old) if old is not None else None, "v2_sha256": C.sha_bytes(new)}
        if rel.endswith(".py") and old is not None:
            a, b = top_defs(old.decode()), top_defs(new.decode())
            entry.update(identical=sorted(k for k in a if b.get(k) == a[k]), changed=sorted(k for k in a if k in b and b[k] != a[k]),
                         added=sorted(set(b) - set(a)), removed=sorted(set(a) - set(b)))
        elif old is not None:
            entry["changed_lines"] = [{"v1": x, "v2": y} for x, y in zip(old.decode().splitlines(), new.decode().splitlines()) if x != y]
        files[rel] = entry
    for rel, names in EVALUATOR_DEFS.items():
        C.require(set(names) <= set(files[rel]["identical"]), f"{rel}: evaluator definitions changed: "
                  f"{sorted(set(names) - set(files[rel]['identical']))}")
    sb = files["slurm/v10_test_repeat/gpu.sbatch"]["changed_lines"]
    C.require(len(sb) == 1 and sb[0]["v2"].startswith("R=") and str(EXEC_ROOT_V2) in sb[0]["v2"], "sbatch changed beyond the root")
    pdiff = json_diff(b2["v1_protocol"], b2["protocol"])
    allowed = ("schema", "id", "status", "version", "dev_gate.criterion_by_gpu_type", "dev_gate.tolerance_rtx6000pro_h100",
               "dev_gate.passing_status", "dev_gate.recorded_by_every_gate", "dev_gate.accepted_v1_l40s_gate_records",
               "jobs.execution_root")
    C.require(all(d["path"] in allowed for d in pdiff), f"protocol changed outside the gate criterion: {pdiff}")
    return {"schema": "v10.test_repeat134.delta_review.v2", "v1_commit": V1_COMMIT,
            "protocol": {"v1_sha256": C.sha_bytes(b2["v1_protocol_bytes"]), "changed_paths": pdiff,
                         "unchanged": "every other path, including the unit set, perturbations, metrics, sets, statistics, "
                                      "reporting, precision, hardware rule, gate plan, inputs, job plan and cost estimate"},
            "manifest_and_decision": "byte-identical to v1 (rebuilt at this commit and compared)",
            "code": files,
            "evaluator_path_identical": {rel: list(names) for rel, names in EVALUATOR_DEFS.items()},
            "l40s_criterion": "identical: PASS iff compare_rows is a bitwise PASS, the re-evaluated DEV32 equals the recorded "
                              "DEV32 and the data roles read are within FIT / CAL / DEV (v1 expression, gate_verdict 'exact')",
            "conclusion": "only the RTX6000PRO / H100 gate criterion changed (with its records, the acceptance of v1 L40S gate "
                          "records, the version fields and the new execution root)"}


def activation_v2(shas: dict, code_files: dict, v1_activation_sha: str, l40s: list) -> dict:
    return {
        "schema": "v10.test_repeat134.activation.v2", "status": "ACTIVE_FOR_FROZEN_CHECKPOINT_EVALUATION_ONLY",
        "id": "V10-TEST-REPEAT134-ACTIVATION-V2-20261020",
        "supersedes": {"activation_sha256": v1_activation_sha, "commit": V1_COMMIT,
                       "for": "every stage not yet run under v1: the RTX6000PRO / H100 gates, prepare and TEST"},
        "owner_decision": {"path": f"{C.RECORDS}/{C.DECISION}", "sha256": shas[C.DECISION]},
        "records": shas, "code": {"files": code_files, "execution_root": str(EXEC_ROOT_V2)},
        "accepted_v1_l40s_gate_records": {"activation_sha256": v1_activation_sha, "code_commit": V1_COMMIT, "gate_ids": l40s,
                                          "basis": "identical L40S evaluator and bitwise criterion (delta review)"},
        "permits": "reading the 134 BF/SHG TEST cores (their raw members once in the prepare job, then the locked prepared "
                   "inputs) solely to evaluate the frozen selected checkpoints of the manifest with the frozen evaluators",
        "conditions": [
            "inside a Slurm GPU job submitted with V10_TR_EXPECT_ACTIVATION = sha256 of this file and V10_TR_EXPECT_COMMIT = the "
            "v2 execution worktree commit; the records, the code files above and the bundle copies are byte-identical",
            "prepare: every L40S DEV gate PASS (v1 records of a85cc21b accepted); raw member sha256 == the manifest; prepared "
            "arrays == V9-G1's prepared cache",
            "test: the (family, GPU type) DEV gate PASS (L40S) or PASS / PASS_TOLERANCE (RTX6000PRO, H100); the job's GPU type == "
            "the unit's recorded DEV32 GPU type; INPUT_LOCK PASS under this activation; frozen checkpoint sha256 == the manifest",
            "results are written once and never overwritten; every TEST read is logged"],
        "forbidden": ["any TEST access outside these jobs (login node, CPU tests, interactive use)",
                      "training, CAL selection, checkpoint choice, grid / candidate / protocol changes from TEST content",
                      "pooling TEST_REPEAT with DEV or calling it an independent confirmation",
                      "re-running a unit to replace a result"],
        "evaluator_only": True}


def render_v2(pr) -> dict:
    b2 = build_v2(pr)
    out = {C.PROTOCOL: C.dumps(b2["protocol"]), C.AMENDMENT: C.dumps(b2["amendment"]), C.DELTA: C.dumps(delta_review(b2))}
    shas = {C.DECISION: C.sha((C.REPO / C.RECORDS / C.DECISION)), C.MANIFEST: C.sha(C.REPO / C.RECORDS / C.MANIFEST),
            **{k: C.sha_bytes(v.encode()) for k, v in out.items()},
            C.V1_PROTOCOL: C.sha_bytes(b2["v1_protocol_bytes"]), C.V1_ACTIVATION: C.sha_bytes(b2["v1_activation_bytes"])}
    code_files = {rel: C.sha(C.REPO / rel) for rel in C.CODE_FILES}
    out[C.ACTIVATION] = C.dumps(activation_v2(shas, code_files, C.sha_bytes(b2["v1_activation_bytes"]), b2["l40s"]))
    return out


# ------------------------------------------------------------------------------------------- version 4 (same-job pairing)
def git_bytes(commit: str, rel: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=C.REPO, capture_output=True, check=True).stdout


def observed_pairs_v4(man: dict) -> list:
    out = []
    for job in OBSERVED_GATE_JOBS_V4:
        hits = sorted(C.BUNDLE.glob(f"gate/*/{job}/GATE.json"))
        C.require(len(hits) == 1, f"observed DEV re-evaluation {job} not found")
        g = C.read(hits[0])
        u = next(x for x in man["units"] if x["key"] == g["unit"])
        d = Path(u["unit_dir"])
        diag = C.gate_diagnostics(C.lines(d / "DEV.jsonl"), C.lines(hits[0].parent / "DEV.jsonl"),
                                  C.read(d / "DEV.identity.json"), C.read(hits[0].parent / "DEV.identity.json"))
        cr, cn = diag["counts"]["recorded"], diag["counts"]["new"]
        out.append({"gate_id": g["gate_id"], "unit": g["unit"], "job": job, "gate_status": g["status"],
                    "d_lt42": cn["lt42"] - cr["lt42"], "d_lt9": cn["lt9"] - cr["lt9"],
                    "d_median": diag["median"]["new"] - diag["median"]["recorded"], "flips": diag["flips"],
                    "abs_delta_err_full": diag["abs_delta_err_full"], "bitwise_rows": diag["bitwise_rows"],
                    "F_true_bitwise": diag["bitwise_fields"].get("F_true"),
                    "identity_and_truths_bitwise": diag["identity_equal"] and diag["perturbation_fields_equal"],
                    "record_sha256": C.sha(hits[0])})
    return out


def build_v4(pr) -> dict:
    for commit, files in ((V1_COMMIT, V1_FILES), (V2_COMMIT, V2_FILES)):
        for name in files.values():
            C.require((C.REPO / C.RECORDS / name).read_bytes() == git_bytes(commit, f"{C.RECORDS}/{name}"),
                      f"{name} differs from its freeze commit {commit[:8]}")
    R = pr.collect(verify_pt=False)
    recs_v1 = build(pr)
    checked = ((C.DECISION, "decision"), (C.MANIFEST, "manifest"))
    if C.BUNDLE == C.HAND / "v10_test_repeat134_20261020":
        checked += ((V1_FILES["protocol"], "protocol"),)
    for name, key in checked:
        C.require(C.dumps(recs_v1[key]) == (C.REPO / C.RECORDS / name).read_text(), f"{name}: rebuilt at HEAD differs")
    C.require(not (C.BUNDLE / "INPUT_LOCK.json").exists() and not (C.BUNDLE / "results").exists(),
              "TEST inputs or results exist: v4 must be frozen before any TEST read")
    man = recs_v1["manifest"]
    v2p_b = (C.REPO / C.RECORDS / V2_FILES["protocol"]).read_bytes()
    v2p = json.loads(v2p_b)
    plan = []
    for u in man["units"]:
        est = u["dev"]["registration_seconds"] / 320 * (TEST_REG + 320) + 2 * OVERHEAD_S
        plan.append({"unit": u["key"], "gpu_type": u["hardware"]["gpu_type"], "dev_pass": "same-job DEV32 re-evaluation",
                     "time_limit": hhmm(ceil_minutes(1.5 * est + 1800)), "estimated_seconds": est})
    by_type = {}
    for t in plan:
        by_type[t["gpu_type"]] = by_type.get(t["gpu_type"], 0.0) + t["estimated_seconds"] / 3600
    p2dev = C.read(pr.REPO / "v10/protocol/V10_PROTOCOL_P2_20261017.json")["tma_colocation_sensitivity"]["DEV"]
    dev24 = sorted(p2dev["retained_ids"])
    C.require(dev24 == sorted(R["roles"]["DEV24"]) and len(dev24) == 24, "DEV24 core list")
    pairs = observed_pairs_v4(man)
    C.require(all(p["F_true_bitwise"] and p["identity_and_truths_bitwise"] for p in pairs),
              "an observed re-evaluation is not bitwise in F_true / identity / truths")
    p4 = copy.deepcopy(v2p)
    p4["schema"] = "v10.test_repeat134.protocol.v4"
    p4["id"] = "V10-TEST-REPEAT134-PROTOCOL-V4-20261020"
    p4["status"] = "FROZEN_BEFORE_ANY_TEST_READ"
    p4["version"] = {"number": 4, "supersedes": {"path": f"{C.RECORDS}/{V2_FILES['protocol']}", "sha256": C.sha_bytes(v2p_b),
                                                 "commit": V2_COMMIT},
                     "void_draft": "v3 (branch v10/test-repeat134-v3-draft, e136bd6a) was never frozen and is void",
                     "owner_decision": f"{C.RECORDS}/{C.DECISION_V4}", "amendment": f"{C.RECORDS}/{C.V4_AMENDMENT}",
                     "delta_review": f"{C.RECORDS}/{C.V4_DELTA}",
                     "change": "same-job pairing: one job per unit re-evaluates DEV32, runs the in-job check and only then "
                               "reads TEST134; the DEV gates no longer license TEST and become evaluation-noise samples"}
    p4["in_job_check"] = {
        "order": "in one process on one GPU of the unit's recorded DEV32 GPU type: re-evaluate DEV32 (32 x 10) with the "
                 "TEST-stage evaluator, run this check, and only if it passes read and evaluate TEST134; otherwise the job "
                 "exits non-zero, reads no TEST and records BLOCKED",
        "a": "identity, configuration (trainer rebuild and checkpoint load), checkpoint sha256, input hashes (DATA_LOCK), "
             "perturbation truths theta / t and n_grid are bitwise equal to the recorded unit, and every value is finite",
        "e": "F_true is bitwise equal in all 320 rows (the frozen energy is reproduced)",
        "f": "coarse gross-error band only: |delta count<42| <= 12, |delta count<9| <= 12 (of 320), |delta median err_full| <= 0.5",
        "thresholds": dict(C.IN_JOB),
        "band_note": "the band was set with the v1 / v2 gate re-evaluations in view (DEV only; no TEST result exists); the "
                     "largest changes seen were " + f"|delta count<42| {max(abs(p['d_lt42']) for p in pairs)}, "
                     f"|delta count<9| {max(abs(p['d_lt9']) for p in pairs)}, |delta median| "
                     f"{max(abs(p['d_median']) for p in pairs):.3f} over {len(pairs)} re-evaluations"}
    p4["dev_gate"]["role_v4"] = "no longer licenses TEST; every gate record (finished or still running) is an evaluation-noise sample"
    p4["hardware"]["rule"] = ("each unit is evaluated (DEV re-evaluation and TEST, same job) on the GPU type that produced its "
                              "recorded DEV32: 28 L40S, 12 RTX6000PRO, 2 H100 (nodesumo01); --exclude=node51,node53,node60")
    p4["sets"]["DEV24_cores"] = dev24
    p4["statistics"]["dev_beside"] = ("TEST134 / TEST112 beside the same-job DEV32 re-evaluation (the paired reference) and "
                                      "beside the recorded DEV32; never pooled")
    p4["statistics"]["same_job_dev_contrasts"] = ("paired contrasts on the same-job DEV32 rows with key "
                                                  "'TEST_REPEAT134|<A>|<B>|DEV32_same_job' (same bootstrap rule)")
    p4["evaluation_noise"] = {
        "pairs": "every DEV re-evaluation of a frozen checkpoint against its recorded DEV32 rows: the v1 / v2 gate records "
                 "(any verdict, with an exit receipt) and the 42 in-job re-evaluations",
        "per_family_and_gpu_type": ["delta counts <42 / <9", "flips", "|delta err_full| quantiles", "bitwise-equal rows",
                                    "F_true bitwise"],
        "statement": "评估本身有噪声：同一冻结 checkpoint 在同一 GPU 型号上重评，单次配准结果和成功计数都会变动。小于这一噪声的方法差，"
                     "在 TEST 和 DEV 上都不可解释；这同样适用于 V10 的主 DEV 表。"}
    p4["reporting"]["rules"] = v2p["reporting"]["rules"] + [
        "TEST134 / TEST112 are reported beside the same-job DEV32 re-evaluation (paired reference) and beside the recorded DEV32",
        "an evaluation-noise section reports every DEV re-evaluation pair; method differences smaller than this noise are not "
        "interpretable, on DEV as well as TEST (including the main V10 DEV tables)",
        "BLOCKED units are reported with their failed criteria; TEST is never read for them"]
    p4["inputs"]["precondition"] = "none beyond the frozen v4 records (prepare may run as soon as v4 is frozen)"
    p4["jobs"]["execution_root"] = str(EXEC_ROOT_V4)
    p4["jobs"]["order"] = ["prepare (any GPU type)", "one job per unit after INPUT_LOCK PASS: DEV32 re-evaluation, in-job "
                           "check, TEST134 (on the unit's recorded DEV32 GPU type)"]
    p4["jobs"]["test_units"] = plan
    p4["cost_estimate"] = {"basis": "recorded DEV seconds of the unit on its own GPU type x (1340 TEST + 320 DEV) / 320 + "
                                    f"2 x {OVERHEAD_S} s trainer rebuilds; native allocated GPU-h; prepare 0.25 h",
                           "unit_jobs_gpu_h": by_type, "prepare_gpu_h": 0.25,
                           "total_gpu_h": sum(by_type.values()) + 0.25}
    dec4 = {"schema": "v10.owner_decision.v1", "id": "V10-TEST-REPEAT134-V4-DECISION-20261020", "date": "2026-10-20",
            **DECISION_V4_TEXT, "relayed": "by the coordinating session to the TEST_REPEAT lead, who records it verbatim"}
    ev = evidence()
    am = {"schema": "v10.test_repeat134.amendment.v4", "id": "V10-TEST-REPEAT134-AMENDMENT-V4-20261020",
          "label": "owner decision 2026-10-20 after the v1 / v2 DEV re-evaluations showed evaluation noise (DEV only; no TEST "
                   "result exists); pre-registered before any TEST read",
          "owner_decision": {"path": f"{C.RECORDS}/{C.DECISION_V4}", **DECISION_V4_TEXT},
          "observed_dev_reevaluations": pairs,
          "in_job_check": p4["in_job_check"], "evaluation_noise": p4["evaluation_noise"],
          "hardware": p4["hardware"]["rule"], "owner_rules": ev["owner_rules"],
          "unchanged": ["manifest and units", "perturbation rule, searcher, metrics, unit, TEST134 / TEST112, statistics",
                        "TEST guard, input preparation and INPUT_LOCK", "the evaluator definitions (AST-identical to v1)"],
          "v1_v2": "v1 and v2 records and execution trees are untouched; the v3 draft is void"}
    return {"protocol": p4, "amendment": am, "decision_v4": dec4, "v2_protocol": v2p, "v2_protocol_bytes": v2p_b}


def delta_review_v4(b4: dict) -> dict:
    files = {}
    for rel in V4_CODE_FILES + ("scripts/v10_test_repeat/analyze.py", "scripts/v10_test_repeat/build_freeze.py",
                                "scripts/v10_test_repeat/submit_lines.py", "scripts/v10_test_repeat/status.py"):
        new, old = (C.REPO / rel).read_bytes(), git_bytes(V2_COMMIT, rel)
        entry = {"v2_sha256": C.sha_bytes(old), "v4_sha256": C.sha_bytes(new)}
        if rel.endswith(".py"):
            a, b = top_defs(old.decode()), top_defs(new.decode())
            entry.update(identical=sorted(k for k in a if b.get(k) == a[k]), changed=sorted(k for k in a if k in b and b[k] != a[k]),
                         added=sorted(set(b) - set(a)), removed=sorted(set(a) - set(b)))
        else:
            entry["changed_lines"] = [{"v2": x, "v4": y} for x, y in zip(old.decode().splitlines(), new.decode().splitlines()) if x != y]
        files[rel] = entry
    for rel, names in EVALUATOR_DEFS.items():
        v1defs, now = top_defs(git_bytes(V1_COMMIT, rel).decode()), top_defs((C.REPO / rel).read_text())
        C.require(all(v1defs[n] == now[n] for n in names), f"{rel}: evaluator definitions differ from v1")
    sb = files["slurm/v10_test_repeat/gpu.sbatch"]["changed_lines"]
    C.require(len(sb) == 1 and sb[0]["v4"].startswith("R=") and str(EXEC_ROOT_V4) in sb[0]["v4"], "sbatch changed beyond the root")
    return {"schema": "v10.test_repeat134.delta_review.v4", "v2_commit": V2_COMMIT, "v1_commit": V1_COMMIT,
            "protocol": {"v2_sha256": C.sha_bytes(b4["v2_protocol_bytes"]),
                         "changed_paths": json_diff(b4["v2_protocol"], b4["protocol"])},
            "code": files, "evaluator_definitions_identical_to_v1": {rel: list(n) for rel, n in EVALUATOR_DEFS.items()},
            "conclusion": "same-job pairing: the unit job re-evaluates DEV32, checks (a) (e) (f) and only then reads TEST; "
                          "gates become noise samples; prepare no longer waits for gates; evaluators unchanged"}


def activation_v4(shas: dict, code_files: dict) -> dict:
    return {
        "schema": "v10.test_repeat134.activation.v4", "status": "ACTIVE_FOR_FROZEN_CHECKPOINT_EVALUATION_ONLY",
        "id": "V10-TEST-REPEAT134-ACTIVATION-V4-20261020",
        "supersedes": {"activation_sha256": shas[C.V2_ACTIVATION], "commit": V2_COMMIT, "for": "prepare and every unit job"},
        "owner_decisions": {"start": {"path": f"{C.RECORDS}/{C.DECISION}", "sha256": shas[C.DECISION]},
                            "v4": {"path": f"{C.RECORDS}/{C.DECISION_V4}", "sha256": shas[C.DECISION_V4]}},
        "records": shas, "code": {"files": code_files, "execution_root": str(EXEC_ROOT_V4)},
        "permits": "reading the 134 BF/SHG TEST cores (their raw members once in the prepare job, then the locked prepared "
                   "inputs) solely to evaluate the frozen selected checkpoints of the manifest with the frozen evaluators, "
                   "after the unit's same-job DEV32 re-evaluation passed the in-job check",
        "conditions": [
            "inside a Slurm GPU job submitted with V10_TR_EXPECT_ACTIVATION = sha256 of this file and V10_TR_EXPECT_COMMIT = the "
            "v4 execution worktree commit; the records, the code files above and the bundle copies are byte-identical",
            "prepare: raw member sha256 == the manifest; prepared arrays == V9-G1's prepared cache",
            "unit job: on the unit's recorded DEV32 GPU type; INPUT_LOCK PASS under this activation; frozen checkpoint sha256 "
            "== the manifest; DEV32 re-evaluated first and in-job check (a), (e), (f) passed before TEST is read",
            "results are written once and never overwritten; every TEST read is logged"],
        "forbidden": ["any TEST access outside these jobs (login node, CPU tests, interactive use)",
                      "training, CAL selection, checkpoint choice, grid / candidate / protocol changes from TEST content",
                      "pooling TEST_REPEAT with DEV or calling it an independent confirmation",
                      "re-running a unit to replace a result"],
        "evaluator_only": True}


def render_v4(pr) -> dict:
    b4 = build_v4(pr)
    out = {C.DECISION_V4: C.dumps(b4["decision_v4"]), C.V4_PROTOCOL: C.dumps(b4["protocol"]),
           C.V4_AMENDMENT: C.dumps(b4["amendment"]), C.V4_DELTA: C.dumps(delta_review_v4(b4))}
    shas = {**{k: C.sha_bytes(v.encode()) for k, v in out.items()},
            **{n: C.sha(C.REPO / C.RECORDS / n) for n in (C.DECISION, C.MANIFEST, C.V2_PROTOCOL, C.V2_ACTIVATION,
                                                          C.V2_AMENDMENT, C.V2_DELTA, C.V1_PROTOCOL, C.V1_ACTIVATION)}}
    out[C.V4_ACTIVATION] = C.dumps(activation_v4(shas, {rel: C.sha(C.REPO / rel) for rel in V4_CODE_FILES}))
    return out


# ================================================================================================================ v5
V4_COMMIT = "827099d95702204352b589e4a9d1801429f68080"
V4_FILES = {"decision": C.DECISION_V4, "protocol": C.V4_PROTOCOL, "amendment": C.V4_AMENDMENT, "delta": C.V4_DELTA,
            "activation": C.V4_ACTIVATION}
DIAG_COMMIT = "af7c0798c886886db18ce6973c5a0111bd97c305"           # the input diagnostic's execution commit
DIAG_AUTH_SHA = "fe4e6093dfbaf85ece802d620346ec66bc6e37972230e66dee373a0ba5ad660a"
V4_LOCK_SHA = "fe7e40a570f8d1ac9c36d4089bbe380782979aacf10143ab92b6727d3ea3faf6"
V4_PREPARE_JOB, DIAG_JOB = "1051404", "1051581"
EXEC_ROOT_V5 = C.CS / "ssl_pilot_v10_test_repeat_exec_v5_20261020"
INPUTS_V5 = C.OUT / "inputs_v5"
REBUILD_ORDER = ("default_1", "default_2", "no_X86_V4", "baseline_X86_V2")   # the diagnostic's rebuilds, in its order
DECISION_V5_TEXT = {
    "question_verbatim": "TEST_REPEAT 的输入锁卡在 1B_D8（另外 133 个 core 与 V9-G1 缓存逐位一致），最可能是 CPU 不同导致 numpy 走了不同的 "
                         "SIMD 路径。继续需要读 1B_D8 的 TEST 原始文件做诊断（不做评估），并冻结 v5。怎么处理？",
    "chosen_option_verbatim": "先诊断，按结果处理 (Recommended)",
    "chosen_option_text_verbatim": "诊断确认是 CPU 路径造成的，就按 V9-G1 的 CPU 路径重新准备输入，判定标准不变，预计 134/134 通过。不是的话，"
                                   "就采用原始哈希已核对的 V10 重建版 1B_D8，另加一个排除它的 TEST133 敏感性集。",
    "options_not_chosen": [{"option": "直接排除 1B_D8", "source": "option label as relayed (its text was not relayed)"},
                           {"option": "先停在这里", "source": "option label as relayed (its text was not relayed)"}],
    "decision_rule_as_relayed": {
        "fixed": "before the diagnostic result (the diagnostic job 1051581 was submitted after this decision)",
        "A": "if DIAG.json shows a CPU-dispatch artifact - one rebuild variant reproduces V9-G1's file bitwise, the raw members "
             "match, and V9-G1's file lies inside its prepare window - re-prepare into a new root with the numpy dispatch pinned "
             "to V9-G1's path; the criterion stays bitwise equality with V9-G1's cache; the sets and the 42 units are unchanged",
        "B1": "otherwise accept the raw-verified, deterministic V10 rebuild of 1B_D8, keep TEST134 as primary and add a "
              "pre-registered TEST133 sensitivity set (1B_D8 excluded); TEST112 is unaffected unless 1B_D8 is in it, in which "
              "case add the corresponding TEST111 sensitivity",
        "C": "if the rebuild is not deterministic or the raw files mismatch, stop and report: not covered by the owner's choice"}}
V5_RULE = {
    "reads": "summary of DIAG.json of the input diagnostic (the single diagnostic job of the core)",
    "C_stop": "any value of summary.raw_members_match is not true, or summary.rebuild_deterministic is not true",
    "A": "not C, summary.rebuilds_equal_to_v9_g1 is non-empty and summary.v9_cache_inside_v9_g1_prepare_window is true",
    "A_pin": "the first of " + ", ".join(REBUILD_ORDER) + " in summary.rebuilds_equal_to_v9_g1: its numpy float64 log1p and "
             "add kernels (the only CPU-dispatched kernels of REG-P1's builder); NPY_DISABLE_CPU_FEATURES disables X86_V4 when "
             "that log1p kernel is the baseline one and X86_V3 when that add kernel is; the build process checks both kernels "
             "before reading any raw member",
    "B1": "not C and not A"}


def decision_v5() -> dict:
    return {"schema": "v10.owner_decision.v1", "id": "V10-TEST-REPEAT134-V5-DECISION-20261020", "date": "2026-10-20",
            **DECISION_V5_TEXT,
            "relayed": "by the coordinating session to the TEST_REPEAT lead, who records it verbatim; this record and both v5 "
                       "code paths were committed before the v5 builder opened DIAG.json"}


def select_v5(diag: dict) -> dict:
    """The owner's rule, applied mechanically to the diagnostic summary."""
    s = diag["summary"]
    rm = s.get("raw_members_match") or {}
    cond = {"raw_members_match_all": bool(rm) and all(v is True for v in rm.values()),
            "rebuild_deterministic": s.get("rebuild_deterministic") is True,
            "a_rebuild_reproduces_v9_g1": bool(s.get("rebuilds_equal_to_v9_g1")),
            "v9_g1_file_inside_its_prepare_window": s.get("v9_cache_inside_v9_g1_prepare_window") is True}
    if not (cond["raw_members_match_all"] and cond["rebuild_deterministic"]):
        return {"variant": "C", "conditions": cond}
    if cond["a_rebuild_reproduces_v9_g1"] and cond["v9_g1_file_inside_its_prepare_window"]:
        return {"variant": "A", "conditions": cond,
                "reproducing_rebuild": next(t for t in REBUILD_ORDER if t in s["rebuilds_equal_to_v9_g1"])}
    return {"variant": "B1", "conditions": cond}


def dispatch_pin(diag: dict, tag: str) -> dict:
    run = diag["rebuilds"][tag]
    d = run["dispatch"]
    kernels = {"log1p_float64": d["log1p_float64"], "add_float64": d["add_float64"]}
    off = [f for f, k in (("X86_V3", "add_float64"), ("X86_V4", "log1p_float64")) if kernels[k].startswith("baseline")]
    return {**kernels, "NPY_DISABLE_CPU_FEATURES": " ".join(off), "reproduced_by": tag,
            "rebuild_setting": run["NPY_DISABLE_CPU_FEATURES"],
            "observed_on": {"host": d["host"], "cpu_model": d["cpu_model"], "numpy": d["numpy"],
                            "node_has_X86_V4": diag["summary"]["node_has_X86_V4"],
                            "X86_V3_enabled_in_that_rebuild": d["X86_V3"], "X86_V4_enabled_in_that_rebuild": d["X86_V4"]}}


def minus(cores, groups, drop) -> dict:
    keep = sorted(c for c in cores if c not in drop)
    g = {k: sorted(c for c in v if c not in drop) for k, v in sorted(groups.items())}
    return {"cores": keep, "cores_sha256": C.sha_bytes(C.canonical(keep)), "excluded": sorted(drop),
            "position_groups": {k: v for k, v in g.items() if v}}


def v5_evidence() -> dict:
    """The bundle-only files v5 rests on (pinned by sha256 in the activation and checked by every v5 job)."""
    auth_b = (C.BUNDLE / C.DIAG_AUTH).read_bytes()
    C.require(C.sha_bytes(auth_b) == DIAG_AUTH_SHA and auth_b == (C.REPO / C.RECORDS / C.DIAG_AUTH).read_bytes()
              == git_bytes(DIAG_COMMIT, f"{C.RECORDS}/{C.DIAG_AUTH}"), "the diagnostic authorization differs from af7c0798")
    auth = json.loads(auth_b)
    C.require(auth["cores"] == ["1B_D8"], "the diagnostic scope is not the one failed core")
    core = auth["cores"][0]
    runs = sorted(p.parent.name for p in (C.BUNDLE / "diagnostics" / core).glob("*/DIAG.json"))
    C.require(runs == [DIAG_JOB], f"expected exactly the diagnostic job {DIAG_JOB} for {core}, found {runs}")
    rel = {"v4_lock": C.V4_LOCK, "v4_prepare_receipt": f"exits/{V4_PREPARE_JOB}.json",
           "diagnostic": f"diagnostics/{core}/{DIAG_JOB}/DIAG.json", "diagnostic_receipt": f"diagnostics/exits/{DIAG_JOB}.json"}
    lock, rcp4 = C.read(C.BUNDLE / rel["v4_lock"]), C.read(C.BUNDLE / rel["v4_prepare_receipt"])
    diag, rcp = C.read(C.BUNDLE / rel["diagnostic"]), C.read(C.BUNDLE / rel["diagnostic_receipt"])
    v4act = C.sha(C.REPO / C.RECORDS / C.V4_ACTIVATION)
    C.require(C.sha(C.BUNDLE / rel["v4_lock"]) == V4_LOCK_SHA and lock["status"] == "FAIL_V9_CACHE_MISMATCH"
              and lock["job"] == V4_PREPARE_JOB and lock["activation_sha256"] == v4act
              and lock["v9_cache_comparison"]["differing"] == [core], "v4's INPUT_LOCK.json is not the recorded failure")
    C.require(rcp4["job"] == V4_PREPARE_JOB and rcp4["exit_code"] == 3 and rcp4["stage"] == "prepare", "v4 prepare receipt")
    C.require(rcp["job"] == DIAG_JOB and rcp["exit_code"] == 0 and rcp["stage"] == "diagnose" and rcp["target"] == core
              and rcp["diagnostic_authorization_sha256"] == DIAG_AUTH_SHA and rcp["commit"] == DIAG_COMMIT,
              "the diagnostic exit receipt")
    C.require(diag["core"] == core and diag["job"] == DIAG_JOB and diag["authorization_sha256"] == DIAG_AUTH_SHA
              and diag["activation_sha256"] == v4act and diag["code_commit"] == DIAG_COMMIT
              and diag["input_lock_sha256"] == V4_LOCK_SHA and diag["evaluated"] is False
              and diag["labels_or_results_read"] is False and diag["v10_rebuild_matches_lock"] is True,
              "DIAG.json is not the pinned diagnostic of the failed lock")
    v4file = next(f["sha256"] for f in lock["files"] if f["core"] == core)
    C.require(diag["provenance"]["v10_rebuild"]["sha256"] == v4file, "the diagnosed V10 file is not the locked one")
    return {"core": core, "lock": lock, "diag": diag, "receipt": rcp, "v4_file_sha256": v4file, "paths": rel,
            "shas": {r: C.sha(C.BUNDLE / r) for r in rel.values()}}


def protocol_v5(p4: dict, man: dict, sel: dict, ev: dict) -> dict:
    core, lock, diag, rcp = ev["core"], ev["lock"], ev["diag"], ev["receipt"]
    variant = sel["variant"]
    C.require(variant in ("A", "B1"), "case C: the rebuild is not deterministic or the raw members mismatch; v5 is not "
                                      "built (not covered by the owner's decision)")
    p5 = copy.deepcopy(p4)
    p5["schema"] = "v10.test_repeat134.protocol.v5"
    p5["id"] = "V10-TEST-REPEAT134-PROTOCOL-V5-20261020"
    p5["status"] = "FROZEN_BEFORE_ANY_TEST_EVALUATION"
    s = diag["summary"]
    inputs = {k: p4["inputs"][k] for k in ("raw_root", "raw_rule", "v9_prepared_root", "builder", "host_source_root",
                                          "host_code_hash")}
    inputs["v4_lock"] = {"path": str(C.BUNDLE / C.V4_LOCK), "sha256": ev["shas"][C.V4_LOCK], "job": lock["job"],
                         "status": lock["status"], "root": lock["root"], "differing": lock["v9_cache_comparison"]["differing"],
                         "kept": "as written; never rewritten"}
    inputs["diagnostic"] = {"job": rcp["job"], "partition": rcp["partition"], "node": rcp["nodelist"], "code_commit": DIAG_COMMIT,
                            "authorization": {"path": f"{C.RECORDS}/{C.DIAG_AUTH}", "sha256": DIAG_AUTH_SHA},
                            "result": {"path": str(C.BUNDLE / ev["paths"]["diagnostic"]), "sha256": ev["shas"][ev["paths"]["diagnostic"]]},
                            "summary": s}
    if variant == "A":
        pin = dispatch_pin(diag, sel["reproducing_rebuild"])
        inputs["root"] = str(INPUTS_V5)
        inputs["numpy_dispatch"] = pin
        inputs["lock"] = ("INPUT_LOCK_V5.json (sha256 per prepared file); PASS requires the arrays of all 134 cores to equal "
                          "V9-G1's prepared cache bitwise (criterion unchanged from v1 / v4)")
        inputs["job"] = {"stage": "slurm/v10_test_repeat/cpu.sbatch inputs", "partition": rcp["partition"],
                         "nodelist": rcp["nodelist"], "time_limit": "01:00:00",
                         "why": "the node on which the pinned kernels reproduced V9-G1's file; the build process also checks "
                                "the two kernels before reading any raw member"}
        statement = (f"A (CPU-dispatch artefact): REG-P1's frozen builder rebuilds all 134 cores into {INPUTS_V5} with numpy's "
                     f"float64 log1p / add kernels pinned to {pin['log1p_float64']} / {pin['add_float64']} "
                     f"(NPY_DISABLE_CPU_FEATURES='{pin['NPY_DISABLE_CPU_FEATURES']}'), the kernels of the diagnostic rebuild "
                     f"'{pin['reproduced_by']}' that reproduced V9-G1's {core} file bitwise; criterion unchanged (all 134 cores "
                     "bitwise equal to V9-G1's prepared cache); TEST134, TEST112 and the 42 units unchanged")
        limitation = (f"输入按 V9-G1 的 CPU 路径重建（numpy float64 log1p / add 内核固定为 {pin['log1p_float64']} / "
                      f"{pin['add_float64']}），134 个 core 与 V9-G1 预处理缓存逐位相同才放行；v4 的输入（{core} 不同）不使用。")
    else:
        exc = {core: {"file_sha256": ev["v4_file_sha256"],
                      "basis": "raw members equal every frozen inventory; two fresh-process rebuilds are bitwise identical; "
                               "the file is the v4 prepare job's output recorded in v4's INPUT_LOCK.json",
                      "differs_from_v9_g1": s["v10_vs_v9_differing_arrays"]}}
        inputs["root"] = lock["root"]
        inputs["accepted_exceptions"] = exc
        inputs["lock"] = ("INPUT_LOCK_V5.json (sha256 per prepared file); PASS requires: the 268 raw members' sha256 equal the "
                          "manifest; every prepared file of the v4 root equals its v4 lock entry; the cores whose arrays differ "
                          "from V9-G1's prepared cache are exactly the accepted exceptions, at their accepted sha256")
        inputs["job"] = {"stage": "slurm/v10_test_repeat/cpu.sbatch inputs", "partition": "cpu-high", "time_limit": "01:00:00"}
        t = man["test"]
        s133 = minus(t["cores"], t["position_groups"], {core})
        sens = {f"TEST{len(s133['cores'])}": {**s133, "base": "TEST134", "rule": f"TEST134 without {core} (accepted V10 rebuild input)"}}
        if core in t["sensitivity112"]["cores"]:
            s111 = minus(t["sensitivity112"]["cores"], t["sensitivity112"]["position_groups"], {core})
            sens[f"TEST{len(s111['cores'])}"] = {**s111, "base": "TEST112", "rule": f"TEST112 without {core}"}
        for name, x in sens.items():
            p5["sets"][name] = f"{len(x['cores'])} cores: {x['rule']} (pre-registered sensitivity set; reported beside {x['base']})"
            p5["sets"]["position_groups"][name] = len(x["position_groups"])
        p5["sets"]["sensitivity"] = sens
        names = " / ".join(sens)
        statement = (f"B1: the raw-verified, deterministic V10 rebuild in the v4 root is accepted for {core}; TEST134 stays "
                     f"primary; {names} ({core} excluded) are pre-registered sensitivity sets reported beside TEST134 / TEST112")
        limitation = (f"{core} 的输入是 V10 重建版（原始哈希已核对、两次重建逐位相同），与 V9-G1 预处理缓存在 "
                      + "、".join(s["v10_vs_v9_differing_arrays"]) + f" 上不同；与 V9-G1 的逐 core 比较对 {core} 不是同一输入；"
                      f"{names} 敏感性集排除 {core}，与 TEST134 / TEST112 并列报告，不替代。")
    inputs["v5"] = {"variant": variant, "selection": sel, "rule": V5_RULE, "statement": statement, "limitation": limitation}
    inputs["precondition"] = "none beyond the frozen v5 records (the inputs job may run as soon as v5 is frozen)"
    p5["inputs"] = inputs
    p5["version"] = {"number": 5, "supersedes": {"path": f"{C.RECORDS}/{C.V4_PROTOCOL}",
                                                 "sha256": C.sha(C.REPO / C.RECORDS / C.V4_PROTOCOL), "commit": V4_COMMIT},
                     "void_draft": p4["version"]["void_draft"], "owner_decision": f"{C.RECORDS}/{C.DECISION_V5}",
                     "amendment": f"{C.RECORDS}/{C.AMENDMENT}", "delta_review": f"{C.RECORDS}/{C.DELTA}",
                     "change": "v5 inputs after v4's INPUT_LOCK failure (variant " + variant + "); everything else as v4"}
    p5["jobs"]["execution_root"] = str(EXEC_ROOT_V5)
    p5["jobs"]["sbatch"] = "slurm/v10_test_repeat/gpu.sbatch (unit jobs); slurm/v10_test_repeat/cpu.sbatch (the inputs job)"
    p5["jobs"]["outputs"] = {**p4["jobs"]["outputs"], "inputs": inputs["root"], "input_lock": str(C.BUNDLE / C.LOCK)}
    p5["jobs"]["order"] = [f"inputs (CPU, cpu.sbatch inputs, variant {variant}): writes INPUT_LOCK_V5.json once",
                           "one job per unit after INPUT_LOCK_V5 PASS: DEV32 re-evaluation, in-job check, TEST134 (on the "
                           "unit's recorded DEV32 GPU type)"]
    p5["reporting"]["rules"] = p4["reporting"]["rules"] + [
        f"the TEST inputs follow v5 variant {variant}; v4's INPUT_LOCK failure and the input diagnostic are reported",
        *([f"{' / '.join(p5['sets']['sensitivity'])} are pre-registered sensitivity sets without {core}; reported beside "
           "TEST134 / TEST112, never instead of them"] if variant == "B1" else [])]
    est = p4["cost_estimate"]
    p5["cost_estimate"] = {"basis": est["basis"].replace("; prepare 0.25 h", "") + "; the v5 inputs job runs on CPU",
                           "unit_jobs_gpu_h": est["unit_jobs_gpu_h"], "prepare_gpu_h": 0.0, "inputs_cpu_job_h": 0.25,
                           "total_gpu_h": sum(est["unit_jobs_gpu_h"].values())}
    return p5


def build_v5() -> dict:
    for commit, files in ((V1_COMMIT, {**V1_FILES, "decision": C.DECISION, "manifest": C.MANIFEST}), (V2_COMMIT, V2_FILES),
                          (V4_COMMIT, V4_FILES)):
        for name in files.values():
            b = (C.REPO / C.RECORDS / name).read_bytes()
            C.require(b == git_bytes(commit, f"{C.RECORDS}/{name}") == (C.BUNDLE / name).read_bytes(),
                      f"{name} differs from its freeze commit {commit[:8]} or from its bundle copy")
    C.require(not (C.BUNDLE / C.LOCK).exists() and not (C.BUNDLE / "results").exists(),
              "v5 inputs or TEST results exist: v5 must be frozen before any TEST evaluation")
    ev = v5_evidence()
    sel = select_v5(ev["diag"])
    p4_b = (C.REPO / C.RECORDS / C.V4_PROTOCOL).read_bytes()
    p4 = json.loads(p4_b)
    man = C.read(C.REPO / C.RECORDS / C.MANIFEST)
    p5 = protocol_v5(p4, man, sel, ev)
    dec5 = decision_v5()
    am = {"schema": "v10.test_repeat134.amendment.v5", "id": "V10-TEST-REPEAT134-AMENDMENT-V5-20261020",
          "label": "owner decision 2026-10-20 after v4's INPUT_LOCK failure (1B_D8); the rule was fixed before the diagnostic "
                   "result and is applied mechanically; pre-registered before any TEST evaluation (no TEST result exists)",
          "owner_decision": {"path": f"{C.RECORDS}/{C.DECISION_V5}", **DECISION_V5_TEXT},
          "v4_input_lock": p5["inputs"]["v4_lock"], "diagnostic": p5["inputs"]["diagnostic"], "rule": V5_RULE,
          "selection": sel, "variant": sel["variant"], "inputs": {k: v for k, v in p5["inputs"].items() if k != "diagnostic"},
          "sets": {k: {kk: vv for kk, vv in v.items() if kk != "cores"} for k, v in p5["sets"].get("sensitivity", {}).items()},
          "unchanged": ["manifest and units", "perturbation rule, searcher, metrics, unit, TEST134 / TEST112, statistics",
                        "in-job check (a) (e) (f) and its thresholds", "hardware rule and the 42 unit jobs",
                        "the TEST guard and the evaluator definitions (AST-identical to v1)",
                        "REG-P1's builder (host_modules / prepare / compare_cache identical to v4)"],
          "v1_v2_v4": "v1, v2 and v4 records and execution trees, v4's INPUT_LOCK.json and the v4 input root are untouched; "
                      "the v3 draft stays void"}
    return {"protocol": p5, "amendment": am, "decision_v5": dec5, "v4_protocol": p4, "v4_protocol_bytes": p4_b,
            "selection": sel, "evidence_shas": ev["shas"]}


def delta_review_v5(b5: dict) -> dict:
    files = {}
    for rel in C.CODE_FILES + ("scripts/v10_test_repeat/analyze.py", "scripts/v10_test_repeat/build_freeze.py",
                                "scripts/v10_test_repeat/submit_lines.py", "scripts/v10_test_repeat/status.py"):
        new = (C.REPO / rel).read_bytes()
        try:
            old = git_bytes(V4_COMMIT, rel)
        except subprocess.CalledProcessError:
            files[rel] = {"v4_sha256": None, "v5_sha256": C.sha_bytes(new), "added": True}
            continue
        entry = {"v4_sha256": C.sha_bytes(old), "v5_sha256": C.sha_bytes(new)}
        if rel.endswith(".py"):
            a, b = top_defs(old.decode()), top_defs(new.decode())
            entry.update(identical=sorted(k for k in a if b.get(k) == a[k]), changed=sorted(k for k in a if k in b and b[k] != a[k]),
                         added=sorted(set(b) - set(a)), removed=sorted(set(a) - set(b)))
        else:
            entry["changed_lines"] = [{"v4": x, "v5": y} for x, y in zip(old.decode().splitlines(), new.decode().splitlines()) if x != y]
        files[rel] = entry
    for rel, names in EVALUATOR_DEFS.items():
        v1defs, now = top_defs(git_bytes(V1_COMMIT, rel).decode()), top_defs((C.REPO / rel).read_text())
        C.require(all(v1defs[n] == now[n] for n in names), f"{rel}: evaluator definitions differ from v1")
    C.require({"host_modules", "prepare", "compare_cache"} <= set(files["scripts/v10_test_repeat/prepare_inputs.py"]["identical"]),
              "REG-P1's builder path changed")
    sb = files["slurm/v10_test_repeat/gpu.sbatch"]["changed_lines"]
    C.require(len(sb) == 1 and sb[0]["v5"].startswith("R=") and str(EXEC_ROOT_V5) in sb[0]["v5"], "gpu.sbatch changed beyond the root")
    return {"schema": "v10.test_repeat134.delta_review.v5", "v4_commit": V4_COMMIT, "v1_commit": V1_COMMIT,
            "protocol": {"v4_sha256": C.sha_bytes(b5["v4_protocol_bytes"]),
                         "changed_paths": json_diff(b5["v4_protocol"], b5["protocol"])},
            "code": files, "evaluator_definitions_identical_to_v1": {rel: list(n) for rel, n in EVALUATOR_DEFS.items()},
            "conclusion": "v5 changes only the TEST inputs (variant " + b5["selection"]["variant"] + ": a CPU inputs job writes "
                          "INPUT_LOCK_V5.json; unit jobs read it through input_lock_v5) and, for B1, adds the pre-registered "
                          "sensitivity sets; the guard, evaluators, in-job check and unit plan are unchanged"}


def activation_v5(shas: dict, code_files: dict, evidence: dict, variant: str) -> dict:
    reads = ("rebuilding all 134 cores from their 268 raw members into the v5 root and comparing them with V9-G1's prepared "
             "cache" if variant == "A" else
             "hashing the 268 raw members and the v4-prepared inputs and comparing them with V9-G1's prepared cache")
    return {
        "schema": "v10.test_repeat134.activation.v5", "status": "ACTIVE_FOR_FROZEN_CHECKPOINT_EVALUATION_ONLY",
        "id": "V10-TEST-REPEAT134-ACTIVATION-V5-20261020",
        "supersedes": {"activation_sha256": shas[C.V4_ACTIVATION], "commit": V4_COMMIT, "for": "the inputs stage and every unit job"},
        "owner_decisions": {"start": {"path": f"{C.RECORDS}/{C.DECISION}", "sha256": shas[C.DECISION]},
                            "v4": {"path": f"{C.RECORDS}/{C.DECISION_V4}", "sha256": shas[C.DECISION_V4]},
                            "v5": {"path": f"{C.RECORDS}/{C.DECISION_V5}", "sha256": shas[C.DECISION_V5]}},
        "records": shas, "evidence": evidence, "code": {"files": code_files, "execution_root": str(EXEC_ROOT_V5)},
        "input_variant": variant,
        "permits": f"in the v5 inputs job, {reads} to write INPUT_LOCK_V5.json once; then reading the locked prepared inputs "
                   "solely to evaluate the frozen selected checkpoints of the manifest with the frozen evaluators, after the "
                   "unit's same-job DEV32 re-evaluation passed the in-job check",
        "conditions": [
            "inside a Slurm job submitted with V10_TR_EXPECT_ACTIVATION = sha256 of this file and V10_TR_EXPECT_COMMIT = the v5 "
            "execution worktree commit; the records, the evidence files, the code files above and the bundle copies are "
            "byte-identical",
            "inputs job (CPU): " + ("the build process runs under the pinned numpy kernels; all 134 cores bitwise equal "
                                    "V9-G1's prepared cache" if variant == "A" else
                                    "raw members, prepared files and the accepted exceptions as the protocol states"),
            "unit job: on the unit's recorded DEV32 GPU type; INPUT_LOCK_V5 PASS under this activation; frozen checkpoint "
            "sha256 == the manifest; DEV32 re-evaluated first and in-job check (a), (e), (f) passed before TEST is read",
            "results are written once and never overwritten; every TEST read is logged"],
        "forbidden": ["any TEST access outside these jobs (login node, CPU tests, interactive use)",
                      "training, CAL selection, checkpoint choice, grid / candidate / protocol changes from TEST content",
                      "pooling TEST_REPEAT with DEV or calling it an independent confirmation",
                      "re-running a unit to replace a result", "rewriting v4's INPUT_LOCK.json or the v4 input root"],
        "evaluator_only": True}


def render_v5() -> dict:
    b5 = build_v5()
    out = {C.DECISION_V5: C.dumps(b5["decision_v5"]), C.PROTOCOL: C.dumps(b5["protocol"]),
           C.AMENDMENT: C.dumps(b5["amendment"]), C.DELTA: C.dumps(delta_review_v5(b5))}
    shas = {**{k: C.sha_bytes(v.encode()) for k, v in out.items()},
            **{n: C.sha(C.REPO / C.RECORDS / n) for n in C.FROZEN_RECORDS if n not in out}}
    out[C.ACTIVATION] = C.dumps(activation_v5(shas, {rel: C.sha(C.REPO / rel) for rel in C.CODE_FILES}, b5["evidence_shas"],
                                              b5["selection"]["variant"]))
    return out


# ================================================================================================================ v6
V5_COMMIT = "5fdcb942a085580f88c8e038ed4ed92fed22cf99"
V5_ACTIVATION_SHA = "c724c56b5ec37edddddf52dc1ff4f30d1dd2197c6735c947d63db6d148f2d926"
V5_FILES = {"decision": C.DECISION_V5, "protocol": C.PROTOCOL, "amendment": C.AMENDMENT, "delta": C.DELTA,
            "activation": C.V5_ACTIVATION}
V5_INPUTS_JOB = "1051656"
EXEC_ROOT_V6 = C.CS / "ssl_pilot_v10_test_repeat_exec_v6_20261020"
V6_UNITS = {"CoMIR_native__1001": ("1051911", "1052059", "1052145"), "CoMIR_native__1002": ("1051925", "1052066", "1052146"),
            "CoMIR_native__1003": ("1051928", "1052070", "1052153")}      # unit -> its three failed v5 attempts
COMIR_ERROR = "CoMIR model namespace already belongs to another package"
DECISION_V6_TEXT = {
    "chosen_option_verbatim": "修复后补跑这 3 个单元 (Recommended)",
    "chosen_option_text_verbatim": "在任何 CoMIR TEST 读取之前冻结 v6：只修这个检查，允许路径在固定 CoMIR 目录下的命名空间子包；模型、checkpoint、"
                                   "评估器和判定标准都不变。这 3 个单元仍在同一作业里先重评 DEV 再评 TEST，在 L40S 上跑。",
    "also_said_verbatim": "控制一下harness程度，稍微有些过了",
    "question": "not relayed verbatim (the 3 CoMIR_native units whose v5 TEST-stage trainer was refused by the CoMIR source check)"}


def decision_v6() -> dict:
    return {"schema": "v10.owner_decision.v1", "id": "V10-TEST-REPEAT134-V6-DECISION-20261020", "date": "2026-10-20",
            **DECISION_V6_TEXT, "relayed": "by the coordinating session to the TEST_REPEAT lead, who records it verbatim"}


def comir_attempts(man: dict) -> dict:
    """From the records: each of the 9 v5 attempts failed at the TEST-stage trainer with the CoMIR namespace error, logged no
    TEST read and wrote no TEST row; the one DEV re-evaluation per unit equals the recorded DEV32 rows."""
    out = {}
    for key, jobs in V6_UNITS.items():
        d, u = C.BUNDLE / "results/TEST_REPEAT" / key, C.unit(man, key)
        C.require(not any((d / n).exists() for n in ("TEST_REPEAT.jsonl", "COMPLETE.json", "BLOCKED.json")), f"{key}: TEST output exists")
        C.require(sorted(p.stem[len("ENV_"):] for p in d.glob("ENV_*.json")) == sorted(jobs), f"{key}: attempt jobs differ")
        files = {}
        for j in jobs:
            reads, fail, rcp = d / f"READS_INCOMPLETE_{j}.json", C.BUNDLE / "failures" / f"{j}.json", C.BUNDLE / "exits" / f"{j}.json"
            C.require(C.read(reads)["test_guard"] == [], f"{key} {j}: a TEST read was logged")
            C.require(COMIR_ERROR in C.read(fail)["exception"], f"{key} {j}: another failure")
            r = C.read(rcp)
            C.require(r["exit_code"] != 0 and r["activation_sha256"] == V5_ACTIVATION_SHA, f"{key} {j}: receipt")
            files.update({str(x.relative_to(C.BUNDLE)): C.sha(x) for x in (reads, fail, rcp)})
        cmp = C.compare_rows(C.lines(Path(u["unit_dir"]) / "DEV.jsonl"), C.lines(d / "DEV_REEVAL.jsonl"))
        C.require(cmp["status"] == "PASS" and cmp["n_mismatches"] == 0, f"{key}: DEV re-evaluation differs from the record")
        out[key] = {"v5_jobs": list(jobs), "test_guard_reads": 0, "test_rows_written": False,
                    "dev_reevaluation_vs_recorded": {k: cmp[k] for k in ("status", "rows_recorded", "rows_new", "n_mismatches")},
                    "dev_rows_job": sorted({r["job"] for r in C.lines(d / "DEV_REEVAL.jsonl")}), "records_sha256": files}
    return out


def build_v6() -> dict:
    for commit, files in ((V1_COMMIT, {**V1_FILES, "decision": C.DECISION, "manifest": C.MANIFEST}), (V2_COMMIT, V2_FILES),
                          (V4_COMMIT, V4_FILES), (V5_COMMIT, V5_FILES), (DIAG_COMMIT, {"auth": C.DIAG_AUTH})):
        for name in files.values():
            b = (C.REPO / C.RECORDS / name).read_bytes()
            C.require(b == git_bytes(commit, f"{C.RECORDS}/{name}") == (C.BUNDLE / name).read_bytes(),
                      f"{name} differs from its freeze commit {commit[:8]} or from its bundle copy")
    C.require(C.sha(C.BUNDLE / C.V5_ACTIVATION) == V5_ACTIVATION_SHA, "the v5 activation is not c724c56b")
    C.require(not (C.BUNDLE / C.RESULTS_V6).exists(), "v6 outputs exist: v6 must be frozen before any CoMIR TEST read")
    lock = C.read(C.BUNDLE / C.LOCK)
    C.require(lock["status"] == "PASS" and lock["activation_sha256"] == V5_ACTIVATION_SHA and lock["job"] == V5_INPUTS_JOB
              and C.read(C.BUNDLE / "exits" / f"{V5_INPUTS_JOB}.json")["exit_code"] == 0, "INPUT_LOCK_V5 is not the v5 PASS")
    for rel, names in EVALUATOR_DEFS.items():
        v1defs, now = top_defs(git_bytes(V1_COMMIT, rel).decode()), top_defs((C.REPO / rel).read_text())
        C.require(all(v1defs[n] == now[n] for n in names), f"{rel}: evaluator definitions differ from v1")
    code = {}
    for rel in C.CODE_FILES:
        old, new = git_bytes(V5_COMMIT, rel).decode(), (C.REPO / rel).read_text()
        if rel.endswith(".py"):
            a, b = top_defs(old), top_defs(new)
            code[rel] = {"changed": sorted(k for k in a if k in b and b[k] != a[k]), "added": sorted(set(b) - set(a)),
                         "removed": sorted(set(a) - set(b))}
        else:
            code[rel] = {"changed_lines": [{"v5": x, "v6": y} for x, y in zip(old.splitlines(), new.splitlines()) if x != y]}
    C.require(code["slurm/v10_test_repeat/cpu.sbatch"]["changed_lines"] == [] and code["scripts/v10_test_repeat/prepare_inputs.py"]
              == {"changed": [], "added": [], "removed": []}, "v6 changes the inputs stage")
    man = C.read(C.REPO / C.RECORDS / C.MANIFEST)
    attempts = comir_attempts(man)
    am = {"schema": "v10.test_repeat134.amendment.v6", "id": "V10-TEST-REPEAT134-AMENDMENT-V6-20261020",
          "label": "owner decision 2026-10-20 after the 3 CoMIR_native units failed in v5 before any CoMIR TEST read; scope: "
                   "those 3 units only; criteria unchanged",
          "owner_decision": {"path": f"{C.RECORDS}/{C.DECISION_V6}", **DECISION_V6_TEXT},
          "failure": {"error": COMIR_ERROR, "where": "evaluate.stage_test -> evaluate_unit -> run.new_trainer -> "
                      "CoMIRTrainer.__init__ -> verify_official_source (ssl_pilot_v10_native_recovery_20261019/src/v10/"
                      "supplement/comir.py:51), at the TEST-stage trainer",
                      "cause": "the DEV re-evaluation's trainer imported the official CoMIR models package; models/layers has "
                               "no __init__.py, so models.layers is a namespace package whose __file__ is None, which the frozen "
                               "check refuses; a second CoMIR trainer in one process (v4 same-job pairing) therefore always failed",
                      "attempts": attempts},
          "change": "TEST_REPEAT execution layer only (evaluate.comir_check_v6, installed for CoMIR units after the DEV "
                    "re-evaluation and before the TEST-stage trainer): while the frozen verify_official_source runs, the "
                    "models.* modules without __file__ whose every __path__ entry resolves inside the pinned CoMIR root are "
                    "hidden from sys.modules; every other condition is the frozen function itself (the frozen "
                    "native-recovery tree is untouched)",
          "scope": {"units": list(V6_UNITS), "gpu": "L40S (each unit's recorded DEV32 GPU type), v5 time limits",
                    "outputs": f"{C.BUNDLE / C.RESULTS_V6}/<unit>: a fresh output, so that the DEV32 re-evaluation and TEST "
                               "run in the same job (the v5 attempt directories are kept as written)",
                    "inputs": f"INPUT_LOCK_V5.json (job {V5_INPUTS_JOB}, v5 activation), unchanged"},
          "code": code,
          "unchanged": ["model, checkpoint, evaluator, perturbations, metrics and criteria (in-job check (a) (e) (f))",
                        "the v5 records, inputs and the other 39 units (they stay under v5)"]}
    return {"decision_v6": decision_v6(), "amendment_v6": am}


def activation_v6(shas: dict, code_files: dict, evidence: dict) -> dict:
    return {
        "schema": "v10.test_repeat134.activation.v6", "status": "ACTIVE_FOR_THE_3_CoMIR_NATIVE_UNITS_ONLY",
        "id": "V10-TEST-REPEAT134-ACTIVATION-V6-20261020",
        "supersedes": {"activation_sha256": V5_ACTIVATION_SHA, "commit": V5_COMMIT,
                       "for": "the 3 CoMIR_native units only; the inputs and the other 39 units stay under v5"},
        "owner_decisions": {n: {"path": f"{C.RECORDS}/{f}", "sha256": shas[f]} for n, f in
                            (("start", C.DECISION), ("v4", C.DECISION_V4), ("v5", C.DECISION_V5), ("v6", C.DECISION_V6))},
        "records": shas, "evidence": evidence, "code": {"files": code_files, "execution_root": str(EXEC_ROOT_V6)},
        "v6": {"units": list(V6_UNITS), "input_lock_activation_sha256": V5_ACTIVATION_SHA,
               "outputs": f"{C.BUNDLE / C.RESULTS_V6}/<unit>"},
        "permits": "for the 3 CoMIR_native units only: reading the v5-locked prepared TEST inputs solely to evaluate the unit's "
                   "frozen selected checkpoint with the frozen evaluator, after the same-job DEV32 re-evaluation passed the "
                   "in-job check, on the unit's recorded DEV32 GPU type (L40S)",
        "conditions": [
            "inside a Slurm GPU job submitted with V10_TR_EXPECT_ACTIVATION = sha256 of this file and V10_TR_EXPECT_COMMIT = the "
            "v6 execution worktree commit; the records, the evidence files, the code files above and the bundle copies are "
            "byte-identical",
            "INPUT_LOCK_V5 PASS under the v5 activation; frozen checkpoint sha256 == the manifest; DEV32 re-evaluated first in the "
            "same job and in-job check (a), (e), (f) passed before TEST is read",
            "results are written once and never overwritten; every TEST read is logged"],
        "forbidden": ["any TEST access outside these jobs", "running any of the other 39 units under v6",
                      "rewriting v5 records, the v5 attempt directories or INPUT_LOCK_V5.json",
                      "training, CAL selection, checkpoint choice or criterion changes from TEST content",
                      "pooling TEST_REPEAT with DEV or calling it an independent confirmation"],
        "evaluator_only": True}


def render_v6() -> dict:
    b6 = build_v6()
    out = {C.DECISION_V6: C.dumps(b6["decision_v6"]), C.AMENDMENT_V6: C.dumps(b6["amendment_v6"])}
    shas = {**{k: C.sha_bytes(v.encode()) for k, v in out.items()},
            **{n: C.sha(C.REPO / C.RECORDS / n) for n in C.FROZEN_RECORDS if n not in out}}
    evidence = {r: C.sha(C.BUNDLE / r) for r in (C.LOCK, f"exits/{V5_INPUTS_JOB}.json")}
    out[C.ACTIVATION] = C.dumps(activation_v6(shas, {rel: C.sha(C.REPO / rel) for rel in C.CODE_FILES}, evidence))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", type=Path, help="write the records into this directory only (development)")
    ap.add_argument("--decision-only", action="store_true", help="write OWNER_DECISION_V6.json to the repo records only")
    a = ap.parse_args(argv)
    if a.decision_only:                 # the owner's v6 decision
        path = C.REPO / C.RECORDS / C.DECISION_V6
        text = C.dumps(decision_v6())
        C.require(not path.exists() or path.read_text() == text, f"{path} exists with different content")
        path.write_text(text)
        print("wrote", path, "sha256", C.sha_bytes(text.encode()))
        return 0
    texts = render_v6()              # v1, v2, v4 and v5 (5fdcb942) records are frozen; this adds the v6 records
    ok = True
    if a.dry_run:
        a.dry_run.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            (a.dry_run / name).write_text(text)
            print(f"{name} sha256 {C.sha_bytes(text.encode())} ({len(text)} bytes) -> {a.dry_run}")
        return 0
    for name, text in texts.items():
        for path in (C.REPO / C.RECORDS / name, C.BUNDLE / name):
            if a.check:
                same = path.exists() and path.read_text() == text
                ok &= same
                print(f"{path}: {'identical' if same else 'DIFFERS'}")
            else:
                if path.exists():
                    C.require(path.read_text() == text, f"{path} exists with different content (frozen records are not rewritten)")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
                print("wrote", path)
    for name, text in texts.items():
        print(f"{name} sha256 {C.sha_bytes(text.encode())}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
