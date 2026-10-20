#!/usr/bin/env python3
"""V10 phase-2 results report generator -> v10/V10_PHASE2_RESULTS_REPORT_20261019_CN.md (Chinese) and the machine-readable
sidecar v10/V10_PHASE2_RESULTS_20261019.json.

Scope: the natural-image BF (BF/SHG) development benchmark of V10 after phase 1, as recorded on the server:
  BF1        four frozen families MF_VCS_perm1, MF_VCS_M0, MF_VCS_M1, MLP_VCS_perm1: 18 CAL candidates (tuning seed 1017) and
             12 final seeds (1001-1003), handoff/v10_bf1_20261017; scheduling ledger handoff/v10_parallel8_20261018/LEDGER.json
  supplement seven methods MLP_VCS_perm8, InfoNCE_host[J_perm1], MI_native, NMI_native, CoMIR_native, RuLSIF_exact (v1),
             MIND_2D_adapted: handoff/v10_method_pipeline_20261019/families/{0,1,3,5,6} and
             handoff/v10_native_geometry_recovery_20261019/families/{2,4} (RELEASE.json, results/CAL, results/FINAL_DEV,
             COST_LEDGER.jsonl, exits), the bundles' METHOD_REVIEW.json, LEDGER.json and validation/exits
  replication secondary DEV seeds 1004/1005 of six learned methods, designed after the original DEV readouts:
             handoff/v10_seed_replication_pool_20261019 (+ the three inherited units of handoff/v10_seed_replication_hw_20261019)
  rules      owner decision v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md (byte-equal to
             /home/infres/yinwang/CS_QMI/V10_PHASE1_DECISIONS_20261017_CN.md; every quote is checked verbatim inside its section),
             protocol v10/protocol/V10_PROTOCOL_P2_20261017.json, manifest v10/T00/V10_DATA_MANIFEST.csv,
             accepted L40S T01 records v10/T01/{TORCH_VALIDATION,GEOMETRY_GRADIENT_CHECK}.json
  review     handoff/review_claude_20261019/REVIEW_CN.md (+ SLURM_START_TIMES_20261019T105844Z.txt); quotes checked verbatim
  later BF   RuLSIF_exact_v2 (handoff/v10_rulsif_v2_20261019: CAL release 41313c39, finals release_xhw 12e37693, A1 amendment
             committed in ssl_pilot_v10_rulsif2_exec), the K8/K32 capacity extension (handoff/v10_capacity_ext_20261019 and its
             hw_amendment_20261019 table; unit outputs under outputs_v10/capacity_ext{,_hw}_20261019), the post-DEV
             sensitivities D1-D3 (handoff/v10_sensitivity_20261019) and the cross-hardware T01 diagnostics
             (outputs_v10/xhw_t01/<tag>/T01/parts/*.json); each unit is verified like the units above
  hardware   owner decisions as recorded in the capacity HARDWARE_AMENDMENT.json and the RuLSIF v2 RELEASE_XHW.json (compared
             verbatim), the EPFL cross-hardware gate amendment (committed in ssl_pilot_v10_epfl_release_hw2_20261020) and the
             EPFL gate records it refers to (failed H100 gate 1050341; amended RTX6000PRO / H100 gates)
  EPFL       never recomputed from raw results: the official frozen-protocol analysis written by the EPFL agent,
             handoff/v10_epfl_production_20261019/analysis/EPFL_RESULTS.json, is read when present (its sha256 is printed;
             schema v10.epfl.results.v1 is rendered, another schema only recorded) and EPFL is "pending" otherwise; the
             protocol / amendment / release hashes and the gate rows it cites must equal the records; never pooled with BF
  sacct      v10/tools/phase2_report_sacct_probe.json, written only by --probe-sacct (a read-only sacct call); the report reads
             the snapshot, never Slurm

Verification before anything is rendered (any failure = SystemExit "REFUSE: ...", nothing written):
  * pinned sha256 of every frozen input (decision, protocol, manifest, T01 records, review, BF1 FROZEN/PROTOCOL/DATA_LOCK/GPU_GATE,
    each family's SELECTION.json and RELEASE.json, the replication FREEZE/PROTOCOL files);
  * per unit: COMPLETE status; COMPLETE identity == IDENTITY.json == the identity expected from the frozen family record (seed,
    candidate, schedule, release / freeze, data lock, selection dependency, replication fields); every listed non-checkpoint
    artifact re-hashed (checkpoints too with --verify-checkpoints); no unlisted file other than RUN.lock, COMPLETE.json and
    READS_<job>.json (which must equal DATA_READS.json); checkpoint metadata identity/step/sha chain; exactly one ENV file whose
    job equals the unit's job; recorded precision environment == the BF1 protocol (primary units on NVIDIA L40S);
  * per CAL / DEV evaluation: identity == summary identity, role, cores == manifest role list, 10 perturbations, searcher; rows
    complete and unique, finite, one checkpoint; metrics recomputed == summary == COMPLETE; the unit's checkpoint choice
    recomputed with the frozen lexicographic rule; DEV32 / DEV24 / per-core metrics recomputed; DEV24 == the decision's list;
  * family selection recomputed from the verified candidates; complete_sha256 chain; final dependency == sha256(SELECTION.json);
  * exit receipts: every scientific job exit 0 with the matching release / freeze; ledgers name the same jobs; one CAL job per
    family; no unit trained in more than one job (no resume);
  * data reads only FIT / CAL / DEV of the manifest (DEV only in final units); perturbation truths (theta, t) identical across all
    units for each (role, core, perturbation); RuLSIF_exact v1 labelled a failed adaptation only if F_true < 0 in every row.
Every number in the report is computed here from those records.  Hand-written: prose, headings, labels, the contrast list, the
budget grid levels and the review-item mapping.  Output: no wall-clock time and no live query; two runs give the same bytes.
--check renders in memory and compares with the files on disk and, when tracked, with HEAD (exit 1 on a difference).

    OMP_NUM_THREADS=1 /home/infres/yinwang/CS_QMI/env/bin/python v10/tools/phase2_report.py [--check] [--verify-checkpoints]
    /home/infres/yinwang/CS_QMI/env/bin/python v10/tools/phase2_report.py --probe-sacct      # refresh the sacct snapshot only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SELF_REL = Path(__file__).resolve().relative_to(REPO).as_posix()
REPORT_DATE = "20261019"
REPORT_REL = f"v10/V10_PHASE2_RESULTS_REPORT_{REPORT_DATE}_CN.md"
SIDECAR_REL = f"v10/V10_PHASE2_RESULTS_{REPORT_DATE}.json"
SACCT_PROBE_REL = "v10/tools/phase2_report_sacct_probe.json"
CKPT_CHECK_REL = "v10/tools/phase2_report_checkpoint_check.json"
PYTHON = "/home/infres/yinwang/CS_QMI/env/bin/python"

CS = Path("/home/infres/yinwang/CS_QMI")
HAND = CS / "handoff"
BF1 = HAND / "v10_bf1_20261017"
PAR8 = HAND / "v10_parallel8_20261018"
PIPE = HAND / "v10_method_pipeline_20261019"
NREC = HAND / "v10_native_geometry_recovery_20261019"
POOL = HAND / "v10_seed_replication_pool_20261019"
HWB = HAND / "v10_seed_replication_hw_20261019"
REVIEW_DIR = HAND / "review_claude_20261019"
RULSIF_V2 = HAND / "v10_rulsif_v2_20261019"
CAPACITY = HAND / "v10_capacity_ext_20261019"
CAP_HW = CAPACITY / "hw_amendment_20261019"
CAPACITY_HW_OUT = Path("/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_hw_20261019")
RV2 = HAND / "v10_rulsif_v2_20261019"
RV2_EXEC = CS / "ssl_pilot_v10_rulsif2_exec"           # A1 amendment, committed there
RV2_AMENDMENT_REL = "v10/supplement/RULSIF_V2_AMENDMENT_20261019.json"
RV2_FAMILY = "RuLSIF_exact_v2"
EPFL_PROD = HAND / "v10_epfl_production_20261019"
EPFL_ANALYSIS = EPFL_PROD / "analysis"          # official frozen-protocol EPFL analysis, written by the EPFL agent
EPFL_RESULTS = EPFL_ANALYSIS / "EPFL_RESULTS.json"
EPFL_RESULTS_MD = EPFL_ANALYSIS / "EPFL_RESULTS_CN.md"
EPFL_RESULTS_SCHEMA = "v10.epfl.results.v1"
EPFL_SETS = ("DEV72_primary", "DEV_minus_weaker_relations", "DEV_minus_boundary_and_saturated", "DEV_minus_all")
EPFL_HW = HAND / "v10_epfl_production_hw_20261019"      # read only for the failed H100 gate record (hardware disclosure)
EPFL_HW2 = HAND / "v10_epfl_production_hw2_20261020"    # read only for the amended cross-hardware gate records
EPFL_SRC = CS / "ssl_pilot_v10_epfl_release_20261019"           # frozen EPFL protocol and group record (committed)
EPFL_HW2_SRC = CS / "ssl_pilot_v10_epfl_release_hw2_20261020"   # EPFL cross-hardware gate amendment (committed)
EPFL_RV2_SRC = CS / "ssl_pilot_v10_epfl_rulsif2_rel_20261020"   # EPFL RuLSIF v2 addendum revision 3 (committed)
OUT10 = Path("/projects/EEG-foundation-model/yinghao/outputs_v10")
# the capacity runner writes its unit directories here (ssl_pilot_v10_capacity scripts/v10_capacity/capacity_common.py OUT);
# exit receipts stay in the bundle (exits/)
CAPACITY_OUT = Path("/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_20261019")
SENSITIVITY = HAND / "v10_sensitivity_20261019"
XHW = Path("/projects/EEG-foundation-model/yinghao/outputs_v10/xhw_t01")
DECISION_EXT = CS / "V10_PHASE1_DECISIONS_20261017_CN.md"
DECISION_REL = "v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md"
PROTOCOL_REL = "v10/protocol/V10_PROTOCOL_P2_20261017.json"
MANIFEST_REL = "v10/T00/V10_DATA_MANIFEST.csv"
T01_TORCH_REL = "v10/T01/TORCH_VALIDATION.json"
T01_GEOM_REL = "v10/T01/GEOMETRY_GRADIENT_CHECK.json"
REVIEW_MD = REVIEW_DIR / "REVIEW_CN.md"
REVIEW_STARTTIMES = REVIEW_DIR / "SLURM_START_TIMES_20261019T105844Z.txt"
INPUT_ROOTS = (("BF1", BF1), ("parallel8", PAR8), ("pipeline", PIPE), ("native recovery", NREC), ("复现 pool", POOL),
               ("复现 hw", HWB), ("审查", REVIEW_DIR), ("xhw_t01", XHW), ("RuLSIF v2（BF）", RV2),
               ("RuLSIF v2 A1 修订（工作树）", RV2_EXEC), ("容量扩展", CAPACITY), ("容量扩展输出", CAPACITY_OUT),
               ("容量扩展输出（硬件修订）", CAPACITY_HW_OUT), ("敏感性", SENSITIVITY),
               ("EPFL 官方分析", EPFL_ANALYSIS), ("EPFL 硬件发行（门禁记录）", EPFL_HW), ("EPFL 硬件修订发行（门禁记录）", EPFL_HW2),
               ("EPFL 协议（工作树）", EPFL_SRC), ("EPFL 门禁修订（工作树）", EPFL_HW2_SRC), ("EPFL RuLSIF v2 附录（工作树）", EPFL_RV2_SRC), ("仓库 v10/", REPO / "v10"),
               ("决定（根目录副本）", DECISION_EXT), ("ssl_pilot_v10（协议与 manifest 副本）", CS / "ssl_pilot_v10"))

# sha256 of the frozen inputs: a different byte content is a different analysis input, never silently accepted
PIN = {
    "decision": "2e60f963ad0cf874c35689ebec88da8a03854cec6156ff06850c6924279711dd",
    "protocol_p2": "c750b23baab14b24ec5928383bc47f985e866c932178c49d9e1395410af5bcdc",
    "manifest": "cc309cece34b574b2a518b5775a1e30987ebd9015693cca1543d5610f13564f6",
    "t01_torch": "021f1238f4d368552443014c99b637829cd0017fb5388664505814c4e2fe32d1",
    "t01_geometry": "d79ecd61acbfa761795e4c05a79872c4e90d0e7dded05355bd73cf7c3abe54e6",
    "review": "e59123734a091a7fef8b29944bf26d7af19759ea890bdd813b84ae0270c8c2a9",
    "bf1_frozen": "a03e88277ba61159d4cd70255d72bf13515b8cac9843465329f6f9bcb15f58a2",
    "bf1_protocol": "de4c0c3943d5a54b3a7de73a84785f240f7ee5b1396d173fd2109999a5555e9d",
    "bf1_data_lock": "e65c60a33f095933e673613fc58f2754485933524259e548c6a0f5dc380f72c5",
    "bf1_gpu_gate": "27d6b0e1e2d55ce240b3e57e1e5090b563f532df6f38a6769dda8c95e13dd6e3",
    "pipe_method_review": "ef69c20db9a7a6c8add2cf2025b5b83c77455648dc92886ff36a2ba442cd1cf9",
    "nrec_method_review": "6b7fc447a613f33b8bcd7b78bc1b76b0cd605d66fc3dd0db61cbb0e007055713",
    "pool_freeze": "c5eea843edc5cbb35b74d039ec95809b873c1e276aff93195b1189c3a34b283d",
    "pool_protocol": "d3debde76f0d713d193cbe82a11b02d36ab2e32048d43efe1a1d4f959157fa31",
    "hw_freeze": "d172478823dcd377df737bee3c28101897f4de6600ccf0d6a0a95458f3fe3094",
    "hw_protocol": "67fe1bd8a6de40537f4a0ff44ad5436f86bf575f4163e687aab1eb3cc258f191",
    # completed after the first version of this report
    "rv2_release": "41313c395898f99d911b1e92532cd5fe809f1563165fdbfefa8673859bd6e9be",
    "rv2_release_xhw": "12e376931f4324c3f81096c87dc4bf7b22d90c75cd7f6b8531d046bc3ac40fce",
    "rv2_selection": "cea56cbc1e676e293588e548158e7a4897a4ee253c5a9a7b3c614cbd6e68b919",
    "rv2_amendment": "381f776442e8a8bcfac98814973d2996250be948b9ffd0b6bba7e5b7faf86420",
    "cap_prereg": "f366a8053e0099487586e2e49e10af1fbdc2e30d971914a610b3acbfce5c0d18",
    "cap_base_freeze": "e4157194c5421e406c5abdc3b0decd733c0e1dcb554db1dc6d5def0c39a8023b",
    "cap_hw_freeze": "f4b4748314bbc56de34831c0cc1a84b4ab6b48610f2159d17de959389533c54f",
    "cap_hw_amendment": "aa56b440c821f4e39715b09eefa2b0b47fe8e42515b9efc3af0179b49c2b33f1",
    "cap_table_hw": "3358c9d1f108984f8aafa524b19228ef87d10efebf9c044de3c27a529b699384",
    "sens_protocol": "42a130cb60583b6c602de7680126267f2cfb0957b33d66a4b849a38c252eda72",
    "sens_results": "02688dcb639b77b681d41f7626b80956531d823954ee5d51da3fe58a5f5af730",
    "epfl_release_hw": "7cbd1c21ed7e73b6121723730fda9e9575a113b4a3b3436bcbb6783416e92c15",
    "epfl_release_hw2": "d9d67cf2b987c1a2a7db8e81ffe8cfc52387d71067dc25ce9f89e1fcc3f79a8c",
    "epfl_protocol": "dbde5d9d83d8cfb8931c1a63e59825ced2ef6306a50cdc3932d71b34a84bff40",
    "epfl_xhw_amendment": "cd673b7cbd5b87bb974be3b797d8686b6efe3c0a59ff497d7be2707484bdc7c1",
    "epfl_rv2_T1024": "fd9b6f26ac1460413f8d160d191464b38ac26bdf059e5e5b15b9cfed0c663717",
    "epfl_rv2_T2048": "31aedcec8bb53d6b181af9c2a7bf2a0dd40670c26429df580886e525444851e7",
    "epfl_rv2_addendum3": "98d3f91a27d891d0a7f339f6ad3be83811ab1503159223d27474b1e63d9dab2c",
}
SELECTION_PIN = {
    "MF_VCS_perm1": "af9a46a3b500d3d264300c1bbb706f9645d63051f7d72fe49bb13a7f53a98a81",
    "MF_VCS_M0": "a571492e24fd3fa6724d74b37fed2a28a5f24d73c7363edee7f8dc92e8eef6fe",
    "MF_VCS_M1": "f351a479739cf93679b2e61a41861bdcdabcf727b51c442d79e1f9581b0dca08",
    "MLP_VCS_perm1": "ec2ede048bd5c8e90751c6ba5a39d3972b0935edb2ffbcd5d0aebc4dac0f7fea",
    "MLP_VCS_perm8": "feb04bd1b3a937288dd1c6a799e11626670ff011c642f746bbdbc41be1c41b54",
    "InfoNCE_host[J_perm1]": "89a9eaf0dcbec53e3df7d4a96b728d3af594470d8a02d19d9ba517329b066c12",
    "MI_native": "981e35acf9a84644aca5a1429a58c5c0d772ce593cb1df5cc25dc612d4d57e0c",
    "NMI_native": "45c13384ea2e1bf3e56817411706d660d6152377baf3455bf9a1482353451806",
    "CoMIR_native": "277fa807c73ad82a7338e144bf7fbaf01585c4b7dc4c36202a8e97be6d1bf695",
    "RuLSIF_exact": "bfb638a45ef977f8f7536378e61cf865b199f7178984f2fcce533ded5d2b71f7",
    "MIND_2D_adapted": "726a68626065d24897a06faa3dbec06b6cc3cdf3f499ef8bed4f13c094fd0dd4",
}
RELEASE_PIN = {
    "MLP_VCS_perm8": "f38e93cbeff11c5947004056bd08d52a0e17ad9519d06ec3d70223f09b876145",
    "InfoNCE_host[J_perm1]": "f10c051970a2def3a15a65d4b5176d465ba797ffd951e9e72b4bab774b5d0dbd",
    "MI_native": "87252504ede0de7dc2e022bb50a2485a44480ea8c96b69f568e0e938da275b3c",
    "NMI_native": "9c71fa857cbcb598063e4a0d5e827280cf8a5ef393398cd7a8214decb66ed0bd",
    "CoMIR_native": "a8cd6bf3521af38c5ecd090d8a0d5b6c284bc08fca5760aefeda22cde22eef8d",
    "RuLSIF_exact": "ab25cc0e3a80305fcb2b5dc5b4713d8902ad8022aa0227f4058f24e393c72e21",
    "MIND_2D_adapted": "188204c353a7526121f7286d7a22f0a6adca8b5e3517f31c8ac414b43cb8f961",
}

BF1_FAMILIES = ("MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1", "MLP_VCS_perm1")
BF1_METHOD = {"MF_VCS_perm1": "MF_perm1", "MF_VCS_M0": "MF_M0", "MF_VCS_M1": "MF_M1", "MLP_VCS_perm1": "MLP_perm1"}
SUPP = (("MLP_VCS_perm8", PIPE, 0), ("InfoNCE_host[J_perm1]", PIPE, 1), ("MI_native", NREC, 2), ("NMI_native", PIPE, 3),
        ("CoMIR_native", NREC, 4), ("RuLSIF_exact", PIPE, 5), ("MIND_2D_adapted", PIPE, 6))
ORDER = BF1_FAMILIES + tuple(f for f, _, _ in SUPP)
FINAL_SEEDS = (1001, 1002, 1003)
REPL_SEEDS = (1004, 1005)
L40S = "NVIDIA L40S"
FAILED_V1 = "RuLSIF_exact"          # labelled a failed adaptation only after the F_true < 0 check passes on every row
REFERENCE = "MF_VCS_M1"             # replication protocol comparison_scope: MF_VCS_M1 versus the other learned critics
CONTRASTS = (                       # (A, B, group) paired difference A - B; group label for the table
    ("MF_VCS_M1", "MF_VCS_M0", "ref"), ("MF_VCS_M1", "MF_VCS_perm1", "ref"), ("MF_VCS_M1", "MLP_VCS_perm1", "ref"),
    ("MF_VCS_M1", "MLP_VCS_perm8", "ref"), ("MF_VCS_M1", "InfoNCE_host[J_perm1]", "ref"), ("MF_VCS_M1", "CoMIR_native", "ref"),
    ("MF_VCS_M1", "MI_native", "ref"), ("MF_VCS_M1", "NMI_native", "ref"), ("MF_VCS_M1", "MIND_2D_adapted", "ref"),
    ("MF_VCS_M0", "MF_VCS_perm1", "mech"), ("MLP_VCS_perm1", "MF_VCS_perm1", "mech"),
    ("MLP_VCS_perm8", "MLP_VCS_perm1", "mech"), ("InfoNCE_host[J_perm1]", "MLP_VCS_perm1", "mech"),
)
N_BOOT = 10000
TUNING_BUDGETS_H = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PIPELINE_BUDGETS_H = (1.0, 2.0, 4.0, 8.0, 16.0)
ROT_STRATA = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0))
DEVICE_UNAVAILABLE = "cudaErrorDevicesUnavailable"
EVAL_KEYS = ("err_full", "F_hat", "F_true", "theta_hat", "seconds")


# ------------------------------------------------------------------------------------------------------------------ basics
def refuse(msg: str):
    raise SystemExit("REFUSE: " + msg)


def check(cond, msg: str):
    if not cond:
        refuse(msg)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_time(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def utc_str(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Inputs:
    """Every file read is hashed once; a second read must give the same bytes."""

    def __init__(self):
        self.hashes: dict[str, str] = {}
        self.sizes: dict[str, int] = {}

    def _record(self, path: Path, h: str, n: int):
        key = str(path)
        if key in self.hashes and self.hashes[key] != h:
            refuse(f"input changed during generation: {path}")
        self.hashes[key], self.sizes[key] = h, n

    def raw(self, path) -> bytes:
        p = Path(path)
        check(p.is_file(), f"missing input {p}")
        b = p.read_bytes()
        self._record(p, sha_bytes(b), len(b))
        return b

    def text(self, path) -> str:
        return self.raw(path).decode("utf-8")

    def json(self, path):
        return json.loads(self.raw(path))

    def sha(self, path) -> str:
        return sha_bytes(self.raw(path))

    def sha_stream(self, path) -> str:
        p = Path(path)
        h, n = hashlib.sha256(), 0
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
                n += len(block)
        self._record(p, h.hexdigest(), n)
        return h.hexdigest()

    def digest(self, root: Path) -> tuple[int, int, str]:
        root = str(root)
        keys = sorted(k for k in self.hashes if k == root or k.startswith(root + "/"))
        body = "".join(f"{Path(k).relative_to(root) if k != root else '.'}  {self.hashes[k]}\n" for k in keys)
        return len(keys), sum(self.sizes[k] for k in keys), sha_bytes(body.encode())


def rel_where(p: Path) -> str:
    for base in (HAND, OUT10):
        try:
            return str(Path(p).relative_to(base))
        except ValueError:
            pass
    return str(p)


def jsonl(b: bytes) -> list:
    return [json.loads(s) for s in b.decode("utf-8").splitlines() if s.strip()]


def metrics(rows) -> dict:
    errs = [float(r["err_full"]) for r in rows]
    check(errs and all(math.isfinite(e) and e >= 0 for e in errs), "empty or non-finite registration errors")
    return {"n": len(errs), "success42": sum(e < 42 for e in errs) / len(errs),
            "success9": sum(e < 9 for e in errs) / len(errs), "median_err": statistics.median(errs)}


def sel_key(rec) -> tuple:
    m = rec["metrics"]
    return (-m["success42"], -m["success9"], m["median_err"], rec["step"], rec.get("order", 0))


def fmt_lr(x) -> str:
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def cand_label(c: dict) -> str:
    if c.get("cap") is not None:
        return f"lr {fmt_lr(c['lr'])}, cap {c['cap']}"
    if "epochs" in c:
        return f"{c['epochs']} epoch × {c['steps_per_epoch']} 步 × batch {c['batch_size']}, lr {c['lr']:g}"
    if "lr" in c:
        return f"lr {fmt_lr(c['lr'])}"
    if "bandwidth_multiplier" in c:
        return f"带宽 ×{c['bandwidth_multiplier']:g}, ridge {fmt_lr(c['ridge'])}"
    if "bins" in c:
        lo, hi = c["range_percentiles"]
        return f"{c['bins']} bins, σ_bin {c['sigma_bins']:g}, 范围分位 {lo:g}–{hi:g}"
    if "descriptor" in c:
        d = c["descriptor"]
        return f"r {d['radius']}, σ {d['sigma']:g}, offset {d['offset']}, 方差下限 {d['variance_floor']:g}"
    refuse("unknown candidate shape " + json.dumps(c))


def fx(x, nd: int, sign: bool = False) -> str:
    """Fixed decimals, rounded half up in decimal (0.78125 -> 0.7813), never -0. Binary noise is removed first at 1e-12,
    so a mean such as (186 + 189 + 183) / 960 that the float sum leaves at 0.58124999999999 is rounded as 0.58125."""
    with localcontext() as ctx:
        ctx.prec = 60
        d = Decimal(repr(float(x))).quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN)
        d = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    txt = f"{d:.{nd}f}"
    return ("+" + txt if d >= 0 else txt) if sign else txt


def f4(x) -> str:
    return fx(x, 4)


def s4(x) -> str:
    return fx(x, 4, True)


def f3(x) -> str:
    return fx(x, 3)


def s3(x) -> str:
    return fx(x, 3, True)


def gh(sec: float) -> str:
    return fx(sec / 3600, 3)


def table(head: list, rows: list) -> list:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# ------------------------------------------------------------------------------------------------------- decision quotes
def decision_sections(text: str) -> dict:
    sec, out = None, defaultdict(str)
    for line in text.splitlines():
        m = re.match(r"^#{2,3} (\d+(?:\.\d+)?) ", line)
        if m:
            sec = m.group(1)
        if sec:
            out[sec] += line + "\n"
            if "." in sec:
                out[sec.split(".")[0]] += line + "\n"
    return out


def quote(sections: dict, sec: str, text: str) -> str:
    check(text in sections.get(sec, ""), f"decision quote not found verbatim in section {sec}: {text[:60]}")
    return text


# ----------------------------------------------------------------------------------------------------------------- roles
def load_roles(inp: Inputs, protocol: dict, dec: dict) -> dict:
    check(inp.sha(REPO / MANIFEST_REL) == PIN["manifest"], "manifest sha256 differs from the pin")
    check(inp.sha(CS / "ssl_pilot_v10" / MANIFEST_REL) == PIN["manifest"], "ssl_pilot_v10 manifest differs")
    rows = list(csv.DictReader(io.StringIO(inp.text(REPO / MANIFEST_REL))))
    bf = [r for r in rows if r["dataset"] == "BF_SHG" and r["inclusion"] == "include"]
    fit = sorted(r["record_id"] for r in bf if r["role"] == "FIT")
    cal = sorted(r["record_id"] for r in bf if r["role"] == "CAL")
    dev = sorted(r["record_id"] for r in bf if r["role"] == "DEV" and r["original_split"] != "test")
    check((len(fit), len(cal), len(dev)) == (32, 8, 32), "manifest FIT/CAL/DEV counts are not 32/8/32")

    def pos(c):
        return c.split("_", 1)[1]
    fitcal_pos = {pos(c) for c in fit + cal}
    dev24 = [c for c in dev if pos(c) not in fitcal_pos]
    excluded = [c for c in dev if pos(c) in fitcal_pos]
    listed = re.search(r"DEV24 排除的 8 个 core：`([^`]+)`", dec["6.1"])
    check(listed is not None, "decision 6.1 DEV24 list not found")
    check(excluded == [s.strip() for s in listed.group(1).split(",")], "recomputed DEV24 exclusion differs from decision 6.1")
    tma = protocol["tma_colocation_sensitivity"]["DEV"]
    check(tma["excluded_ids"] == excluded and tma["retained_ids"] == dev24 and tma["full_ids"] == dev,
          "recomputed DEV24 differs from protocol P2")

    def groups(cores):
        g = {}
        for c in cores:
            g.setdefault(pos(c), []).append(c)
        return g
    lock = inp.json(BF1 / "DATA_LOCK.json")
    lock_roles = {(f["core"], f["role"]) for f in lock["files"]}
    check(lock_roles == {(c, "FIT") for c in fit} | {(c, "CAL") for c in cal} | {(c, "DEV") for c in dev},
          "BF1 DATA_LOCK roles differ from the manifest")
    check(lock["manifest_sha256"] == PIN["manifest"] and lock["held_test_read"] is False, "DATA_LOCK manifest / held flag")
    return {"FIT": fit, "CAL": cal, "DEV": dev, "DEV24": dev24, "DEV24_excluded": excluded,
            "groups32": groups(dev), "groups24": groups(dev24), "fitcal_positions": sorted(fitcal_pos),
            "lock_files": len(lock["files"])}


# ------------------------------------------------------------------------------------------------------------- unit load
PRECISION = None   # BF1 PROTOCOL precision block, set in main


def check_precision(env: dict, where: str):
    p = PRECISION
    for k in ("torch", "cuda_runtime", "cudnn", "torch_threads"):
        check(env[k] == p["environment"][k], f"{where}: environment {k} differs from the BF1 protocol")
    for k, v in p["settings"].items():
        check(env["precision"].get(k) == v, f"{where}: precision setting {k} differs from the BF1 protocol")
    check(env["precision"]["env"] == p["env"], f"{where}: precision environment overrides differ")


def load_unit(inp: Inputs, udir: Path, *, schema: str, ident: dict, roles: dict, phase: str, cal_steps: list,
              save_steps: list, steps: int, verify_pt: bool, primary: bool) -> dict:
    where = rel_where(udir)
    comp_b = inp.raw(udir / "COMPLETE.json")
    comp = json.loads(comp_b)
    check(comp.get("status") == "COMPLETE", f"{where}: not COMPLETE")
    check(comp["identity"] == ident, f"{where}: COMPLETE identity differs from the frozen expectation")
    check(inp.json(udir / "IDENTITY.json") == ident, f"{where}: IDENTITY.json differs from COMPLETE")
    check(comp["steps"] == steps, f"{where}: steps {comp['steps']} != {steps}")
    arts = comp["artifacts"]
    required = {"IDENTITY.json", "CONFIG.json", "DATA_READS.json"}
    required |= {f"checkpoints/step_{s:04d}.{x}" for s in save_steps for x in ("pt", "json")}
    required |= {f"CAL/step_{s:04d}.{x}" for s in cal_steps for x in ("jsonl", "summary.json", "identity.json")}
    if steps > 0:
        required.add("TRAIN_STEPS.jsonl")
    if phase == "FINAL_DEV":
        required |= {"DEV.jsonl", "DEV.summary.json", "DEV.identity.json"}
    check(required <= set(arts), f"{where}: COMPLETE lacks required artifacts {sorted(required - set(arts))[:4]}")
    present = {p.relative_to(udir).as_posix() for p in udir.rglob("*") if p.is_file()}
    check(set(arts) <= present, f"{where}: listed artifact missing on disk")
    extra = sorted(present - set(arts) - {"COMPLETE.json", "RUN.lock"})
    for e in extra:
        check(re.fullmatch(r"READS_\d+\.json", e) is not None, f"{where}: unlisted file {e}")
    blobs, pts = {}, []
    for name in sorted(arts):
        p = (udir / name).resolve()
        check(p.is_relative_to(udir.resolve()), f"{where}: artifact outside the unit {name}")
        if name.endswith(".pt"):
            pts.append((f"{where}/{name}", arts[name]))
            if verify_pt:
                check(inp.sha_stream(udir / name) == arts[name], f"{where}/{name}: checkpoint sha256 differs")
            continue
        b = inp.raw(udir / name)
        check(sha_bytes(b) == arts[name], f"{where}/{name}: sha256 differs from COMPLETE")
        blobs[name] = b
    for e in extra:
        check(inp.raw(udir / e) == blobs["DATA_READS.json"], f"{where}/{e}: differs from DATA_READS.json")
    envs = [n for n in arts if re.fullmatch(r"ENV_\d+\.json", n)]
    check(len(envs) == 1, f"{where}: {len(envs)} ENV records (resume or missing)")
    job = str(comp.get("job_id") if schema == "bf1" else comp.get("job"))
    env = json.loads(blobs[envs[0]])
    check(envs[0] == f"ENV_{job}.json" and env["slurm_job_id"] == job, f"{where}: ENV job differs from COMPLETE job")
    check(env.get("device") == "cuda", f"{where}: not a CUDA run")
    check_precision(env, where)
    if primary:
        check(env["gpu_name"] == L40S and env["slurm_partition"] == "L40S", f"{where}: primary unit not on L40S")
    ck = {}
    for s in save_steps:
        meta = json.loads(blobs[f"checkpoints/step_{s:04d}.json"])
        check(meta["identity"] == ident and meta["step"] == s and meta["sha256"] == arts[f"checkpoints/step_{s:04d}.pt"],
              f"{where}: checkpoint metadata chain broken at step {s}")
        ck[s] = meta["sha256"]
    cps = comp["checkpoints"] if schema == "bf1" else comp["cal_checkpoints"]
    check([c["step"] for c in cps] == list(cal_steps), f"{where}: CAL checkpoint steps differ from the schedule")
    cal = {}
    for c in cps:
        s = c["step"]
        rows = jsonl(blobs[f"CAL/step_{s:04d}.jsonl"])
        summ = json.loads(blobs[f"CAL/step_{s:04d}.summary.json"])
        idj = json.loads(blobs[f"CAL/step_{s:04d}.identity.json"])
        check_eval(where + f"/CAL{s}", rows, summ, idj, ident, ck[s], "CAL", roles["CAL"], arts[f"CAL/step_{s:04d}.jsonl"])
        check(c["metrics"] == summ["metrics"] and c["checkpoint_sha256"] == ck[s]
              and c["checkpoint"] == f"checkpoints/step_{s:04d}.pt", f"{where}: CAL record {s} differs from its summary")
        cal[s] = {"metrics": summ["metrics"], "rows": slim(rows), "checkpoint_sha256": ck[s]}
    check(comp["selected"] == min(cps, key=sel_key), f"{where}: recorded checkpoint choice differs from the CAL rule")
    unit = {"dir": udir, "where": where, "schema": schema, "phase": phase, "ident": ident, "job": job, "env": env,
            "host": env["host"], "gpu": env["gpu_name"], "partition": env["slurm_partition"],
            "complete_sha": sha_bytes(comp_b), "complete": comp, "cal": cal, "cal_steps": list(cal_steps),
            "selected": comp["selected"], "steps": steps, "artifact_bytes": sum(len(b) for b in blobs.values()),
            "n_artifacts": len(arts), "pts": pts}
    reads = json.loads(blobs["DATA_READS.json"])
    allowed = {"FIT", "CAL"} | ({"DEV"} if phase == "FINAL_DEV" else set())
    for r in reads:
        check(r["role"] in allowed and r["core"] in roles[r["role"]], f"{where}: data read outside the allowed roles")
    unit["read_roles"] = dict(sorted(Counter(r["role"] for r in reads).items()))
    if phase == "FINAL_DEV":
        rows = jsonl(blobs["DEV.jsonl"])
        summ = json.loads(blobs["DEV.summary.json"])
        idj = json.loads(blobs["DEV.identity.json"])
        sel_ck = comp["selected"]["checkpoint_sha256"]
        check(sel_ck == ck[comp["selected"]["step"]], f"{where}: selected checkpoint sha differs from its metadata")
        check_eval(where + "/DEV", rows, summ, idj, ident, sel_ck, "DEV", roles["DEV"], arts["DEV.jsonl"])
        check(comp["DEV32"] == summ["metrics"], f"{where}: DEV32 differs from DEV summary")
        d24 = set(roles["DEV24"])
        check(comp["DEV24"] == metrics([r for r in rows if r["core"] in d24]), f"{where}: DEV24 recomputation differs")
        check(comp["per_core"] == {c: metrics([r for r in rows if r["core"] == c]) for c in roles["DEV"]},
              f"{where}: per-core metrics differ")
        if "DEV24_cores" in comp:
            check(comp["DEV24_cores"] == roles["DEV24"], f"{where}: DEV24_cores differ")
        unit["dev_rows"] = slim(rows)
        unit["dev32"], unit["dev24"] = comp["DEV32"], comp["DEV24"]
    if "TRAIN_STEPS.jsonl" in blobs:
        unit["train"] = parse_train(blobs["TRAIN_STEPS.jsonl"], steps, job, where, schema, cal_steps)
    unit["config"] = json.loads(blobs["CONFIG.json"])
    return unit


def check_eval(where, rows, summ, idj, ident, ck, role, cores, rows_sha):
    check(idj == summ["identity"] and summ["status"] == "COMPLETE", f"{where}: evaluation identity / status")
    check(idj["unit"] == ident and idj["checkpoint_sha256"] == ck and idj["role"] == role and idj["cores"] == cores
          and idj["perturbations"] == list(range(10)) and idj["searcher"] == "unchanged R6-REG-P2",
          f"{where}: evaluation identity differs (unit/checkpoint/role/cores/perturbations/searcher)")
    check(summ["rows_sha256"] == rows_sha and summ.get("checkpoint_sha256", ck) == ck,
          f"{where}: summary row hash / checkpoint")
    keys = [(r["core"], r["perturbation"]) for r in rows]
    check(len(keys) == len(set(keys)) == len(cores) * 10 and set(keys) == {(c, p) for c in cores for p in range(10)},
          f"{where}: incomplete or duplicated registrations")
    for r in rows:
        check(r["role"] == role and r["checkpoint_sha256"] == ck, f"{where}: mixed role or checkpoint in rows")
        check(all(math.isfinite(float(r[k])) for k in EVAL_KEYS), f"{where}: non-finite registration value")
    check(metrics(rows) == summ["metrics"], f"{where}: recomputed metrics differ from the summary")


def slim(rows: list) -> list:
    out = []
    for r in rows:
        out.append({"core": r["core"], "p": r["perturbation"], "err": float(r["err_full"]), "theta": float(r["theta"]),
                    "t": (float(r["t"][0]), float(r["t"][1])), "F_hat": float(r["F_hat"]), "F_true": float(r["F_true"]),
                    "sec": float(r["seconds"]), "full_sec": float(r.get("full_registration_seconds", r["seconds"])),
                    "n_grid": r["n_grid"]})
    return out


def parse_train(b: bytes, steps: int, job: str, where: str, schema: str, cal_steps: list) -> dict:
    step, grad, m1_stop, m1_iter, m1_rel = [], [], [], [], []
    utc_after = {}
    kernel = None
    wanted = {s + 1 for s in cal_steps}
    jobkey = "job_id" if schema == "bf1" else "job"
    for line in b.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        check(str(r[jobkey]) == job, f"{where}: TRAIN_STEPS row from another job (resume)")
        step.append(r["step"])
        if "grad_norm_preclip" in r:
            grad.append(float(r["grad_norm_preclip"]))
        if "m1_stop_reason" in r:
            m1_stop.append(r["m1_stop_reason"])
            m1_iter.append(int(r["m1_iterations"]))
            m1_rel.append(float(r["m1_gap_relative_to_zero"]))
        if r["step"] in wanted:
            utc_after[r["step"] - 1] = r["utc"]
        if r.get("kind") == "kernel_fit":
            kernel = {k: r[k] for k in ("bandwidth_median", "bandwidth", "bandwidth_left", "bandwidth_right", "negative_coefficients",
                                        "linear_solve_residual", "q_mode", "alpha", "linear_solve_dtype",
                                        "coefficient_postprocessing", "coefficient_count", "constant_coefficient", "J_fit_moments")
                      if k in r}
    check(step == list(range(1, steps + 1)), f"{where}: TRAIN_STEPS steps not exactly 1..{steps}")
    return {"grad": np.array(grad, dtype=float), "m1_stop": m1_stop, "m1_iter": m1_iter, "m1_rel": m1_rel,
            "utc_after": utc_after, "kernel": kernel}


# ------------------------------------------------------------------------------------------------------------------- BF1
def bf1_identity(fam, seed, lr, cap, phase, dep):
    return {"schema": "v10.bf1.unit.v1", "freeze_sha256": PIN["bf1_frozen"], "family": fam, "method": BF1_METHOD[fam],
            "seed": seed, "lr": lr, "cap": cap, "phase": phase, "steps": 3000, "checkpoint_interval": 250, "K": 16,
            "dependency": dep, "data_lock_sha256": PIN["bf1_data_lock"]}


def load_bf1(inp: Inputs, roles: dict, verify_pt: bool) -> dict:
    global PRECISION
    check(inp.sha(BF1 / "FROZEN.json") == PIN["bf1_frozen"], "BF1 FROZEN.json differs from the pin")
    check(inp.sha(BF1 / "PROTOCOL.json") == PIN["bf1_protocol"], "BF1 PROTOCOL.json differs from the pin")
    check(inp.sha(BF1 / "DATA_LOCK.json") == PIN["bf1_data_lock"], "BF1 DATA_LOCK.json differs from the pin")
    check(inp.sha(BF1 / "GPU_GATE.json") == PIN["bf1_gpu_gate"], "BF1 GPU_GATE.json differs from the pin")
    frozen, protocol, gate = inp.json(BF1 / "FROZEN.json"), inp.json(BF1 / "PROTOCOL.json"), inp.json(BF1 / "GPU_GATE.json")
    PRECISION = protocol["precision"]
    check(gate["status"] == "PASS" and gate["freeze_sha256"] == PIN["bf1_frozen"], "BF1 GPU gate not PASS for the freeze")
    check(frozen["run_inputs"] == {"PROTOCOL.json": PIN["bf1_protocol"], "DATA_LOCK.json": PIN["bf1_data_lock"]},
          "FROZEN run inputs differ")
    steps12 = list(range(250, 3001, 250))
    fams = {}
    for fam in BF1_FAMILIES:
        sp = BF1 / "results" / "CAL" / fam / "SELECTION.json"
        sel_b = inp.raw(sp)
        check(sha_bytes(sel_b) == SELECTION_PIN[fam], f"{fam}: SELECTION.json differs from the pin")
        sel = json.loads(sel_b)
        check(sel["status"] == "SELECTED" and sel["family"] == fam and sel["freeze_sha256"] == PIN["bf1_frozen"]
              and sel["budget_exhausted"] is False and sel["planned_candidates"] == len(protocol["candidates"][fam]),
              f"{fam}: selection header")
        check([c["candidate"] for c in sel["completed_candidates"]] == protocol["candidates"][fam],
              f"{fam}: completed candidates differ from the frozen candidate list")
        cands = []
        for c in sel["completed_candidates"]:
            cd = c["candidate"]
            u = load_unit(inp, BF1 / c["unit"], schema="bf1", ident=bf1_identity(fam, 1017, cd["lr"], cd["cap"], "CAL", None),
                          roles=roles, phase="CAL", cal_steps=steps12, save_steps=steps12, steps=3000,
                          verify_pt=verify_pt, primary=True)
            check(u["complete_sha"] == c["complete_sha256"], f"{fam}: candidate COMPLETE hash chain")
            check({**u["selected"], "order": cd["order"]} == c["choice"], f"{fam}: candidate choice differs")
            u.update(method=fam, seed=1017, candidate=cd, order=cd["order"])
            cands.append(u)
        best = min(sel["completed_candidates"], key=lambda r: sel_key(r["choice"]))
        check(best == sel["selected"], f"{fam}: family selection differs from the lexicographic rule")
        check(len({u["job"] for u in cands}) == 1, f"{fam}: CAL candidates ran in more than one job")
        finals = {}
        cd = sel["selected"]["candidate"]
        for seed in FINAL_SEEDS:
            u = load_unit(inp, BF1 / "results" / "FINAL_DEV" / fam / f"seed_{seed}", schema="bf1",
                          ident=bf1_identity(fam, seed, cd["lr"], cd["cap"], "FINAL_DEV", SELECTION_PIN[fam]),
                          roles=roles, phase="FINAL_DEV", cal_steps=steps12, save_steps=steps12, steps=3000,
                          verify_pt=verify_pt, primary=True)
            u.update(method=fam, seed=seed, candidate=cd, order=cd["order"])
            finals[seed] = u
        fams[fam] = {"method": fam, "source": "BF1", "selection": sel, "selection_sha": SELECTION_PIN[fam],
                     "candidates": cands, "finals": finals, "selected_candidate": cd,
                     "selected_choice": sel["selected"]["choice"], "cal_job": cands[0]["job"],
                     "code": {"commit": frozen["source_commit"], "core": cands[0]["env"]["code_hash_src_scripts_v10"],
                              "host": cands[0]["env"]["host_code_hash"], "job_code": cands[0]["env"]["job_code_hash"],
                              "freeze": PIN["bf1_frozen"]}}
        for u in cands + list(finals.values()):
            e = u["env"]
            check(e["git_commit"] == frozen["source_commit"] and e["git_dirty_code"] == ""
                  and e["code_hash_src_scripts_v10"] == "e9b4239450c4e2eb" and e["host_code_hash"] == "6c6f094684fbc50a"
                  and e["job_code_hash"] == e["job_pin_expected"] == "e08ee2c176475e2b", f"{u['where']}: code identity")
    exits = {}
    for p in sorted((BF1 / "exits").glob("*.json")):
        r = inp.json(p)
        check(p.stem == r["job_id"], f"BF1 exit receipt name {p}")
        exits[r["job_id"]] = r
    ledger = [json.loads(s) for s in inp.text(BF1 / "COST_LEDGER.jsonl").splitlines() if s.strip()]
    par8 = inp.json(PAR8 / "LEDGER.json")
    check(par8["status"] == "COMPLETE_VERIFIED" and par8["gate_job"] == "1045975", "parallel8 ledger status")
    return {"families": fams, "exits": exits, "ledger": ledger, "par8": par8, "frozen": frozen, "protocol": protocol,
            "gate_job": par8["gate_job"]}


# ------------------------------------------------------------------------------------------------------------ supplement
def supp_identity(release_sha, fam, cand, seed, phase, schedule, dep):
    return {"schema": "v10.supplement.unit.v1", "release_sha256": release_sha, "family": fam, "candidate": cand,
            "seed": seed, "phase": phase, "schedule": schedule, "data_lock_sha256": PIN["bf1_data_lock"],
            "selection_dependency_sha256": dep}


def load_supp_family(inp: Inputs, fam: str, froot: Path, roles: dict, verify_pt: bool, pin_release=None,
                     pin_selection=None) -> dict:
    rel_b = inp.raw(froot / "RELEASE.json")
    rel_sha = sha_bytes(rel_b)
    if pin_release is not None:
        check(rel_sha == pin_release, f"{fam}: RELEASE.json differs from the pin")
    rel = json.loads(rel_b)
    check(rel["status"] == "FROZEN_VALIDATED" and rel["family"] == fam and rel["data_lock"]["sha256"] == PIN["bf1_data_lock"]
          and rel["parent_protocol_sha256"] == PIN["bf1_protocol"], f"{fam}: release header")
    spec = rel["families"][fam]
    cands_planned, schedule = spec["candidates"], spec["schedule"]
    res = froot / "results"
    sp = res / "CAL" / fam / "SELECTION.json"
    sel_b = inp.raw(sp)
    sel_sha = sha_bytes(sel_b)
    if pin_selection is not None:
        check(sel_sha == pin_selection, f"{fam}: SELECTION.json differs from the pin")
    sel = json.loads(sel_b)
    check(sel["status"] == "SELECTED" and sel["family"] == fam and sel["release_sha256"] == rel_sha
          and sel["budget_exhausted"] is False, f"{fam}: selection header")
    check([c["candidate"] for c in sel["completed_candidates"]] == cands_planned, f"{fam}: candidates differ from release")
    exits = {}
    for p in sorted((froot / "exits").glob("*.json")):
        r = inp.json(p)
        check(p.stem == r["job"], f"{fam}: exit receipt name {p}")
        exits[r["job"]] = r
    ledger = [json.loads(s) for s in inp.text(froot / "COST_LEDGER.jsonl").splitlines() if s.strip()]
    steps, cal_steps, save_steps = schedule["steps"], schedule["cal_steps"], schedule["save_steps"]
    cands = []
    for c in sel["completed_candidates"]:
        cd = c["candidate"]
        u = load_unit(inp, res / "CAL" / fam / f"candidate_{cd['order']:02d}", schema="supp",
                      ident=supp_identity(rel_sha, fam, cd, 1017, "CAL", schedule, None), roles=roles, phase="CAL",
                      cal_steps=cal_steps, save_steps=save_steps, steps=steps, verify_pt=verify_pt, primary=True)
        check(u["complete_sha"] == c["complete_sha256"] and {**u["selected"], "order": cd["order"]} == c["choice"],
              f"{fam}: candidate chain")
        u.update(method=fam, seed=1017, candidate=cd, order=cd["order"])
        cands.append(u)
    check(min(sel["completed_candidates"], key=lambda r: sel_key(r["choice"])) == sel["selected"],
          f"{fam}: family selection differs from the lexicographic rule")
    check({u["job"] for u in cands} == {sel["job"]}, f"{fam}: CAL candidates not all from the selection job")
    rc = exits.get(sel["job"])
    check(rc is not None and rc["exit_code"] == 0 and rc["release_sha256"] == rel_sha, f"{fam}: CAL exit receipt")
    cd = sel["selected"]["candidate"]
    finals = {}
    for seed in schedule["final_seeds"]:
        u = load_unit(inp, res / "FINAL_DEV" / fam / f"seed_{seed}", schema="supp",
                      ident=supp_identity(rel_sha, fam, cd, seed, "FINAL_DEV", schedule, sel_sha), roles=roles,
                      phase="FINAL_DEV", cal_steps=cal_steps, save_steps=save_steps, steps=steps, verify_pt=verify_pt,
                      primary=True)
        rc = exits.get(u["job"])
        check(rc is not None and rc["exit_code"] == 0 and rc["release_sha256"] == rel_sha, f"{fam}: final exit receipt")
        u.update(method=fam, seed=seed, candidate=cd, order=cd["order"])
        finals[seed] = u
    for u in cands + list(finals.values()):
        e = u["env"]
        check(e["git_commit"] == rel["source_commit"] and e["git_dirty_code"] == "", f"{u['where']}: source commit")
    codes = {u["env"]["code_hash_src_scripts_v10"] for u in cands + list(finals.values())}
    check(len(codes) == 1, f"{fam}: more than one code hash")
    return {"method": fam, "source": "SUPP", "release": rel, "release_sha": rel_sha, "selection": sel,
            "selection_sha": sel_sha, "candidates": cands, "finals": finals, "selected_candidate": cd,
            "selected_choice": sel["selected"]["choice"], "cal_job": sel["job"], "exits": exits, "ledger": ledger,
            "schedule": schedule, "root": froot,
            "code": {"commit": rel["source_commit"], "core": codes.pop(), "host": cands[0]["env"]["host_code_hash"],
                     "job_code": cands[0]["env"]["job_code_hash"], "release": rel_sha, "code_sha256": rel["code_sha256"]}}


def load_supplement(inp: Inputs, roles: dict, verify_pt: bool) -> dict:
    check(inp.sha(PIPE / "METHOD_REVIEW.json") == PIN["pipe_method_review"], "pipeline METHOD_REVIEW differs")
    check(inp.sha(NREC / "METHOD_REVIEW.json") == PIN["nrec_method_review"], "recovery METHOD_REVIEW differs")
    review = {"PIPE": inp.json(PIPE / "METHOD_REVIEW.json"), "NREC": inp.json(NREC / "METHOD_REVIEW.json")}
    out = {}
    for fam, bundle, idx in SUPP:
        fr = load_supp_family(inp, fam, bundle / "families" / str(idx), roles, verify_pt, RELEASE_PIN[fam], SELECTION_PIN[fam])
        rv = review["PIPE"]["methods"][fam]
        check(rv["candidates"] == fr["release"]["families"][fam]["candidates"] and rv["schedule"] == fr["schedule"],
              f"{fam}: release differs from the frozen METHOD_REVIEW recipe")
        fr["bundle"], fr["index"] = bundle, idx
        out[fam] = fr
    ledgers = {"PIPE": inp.json(PIPE / "LEDGER.json"), "NREC": inp.json(NREC / "LEDGER.json")}
    vexits = {}
    for b, root in (("PIPE", PIPE), ("NREC", NREC)):
        for p in sorted((root / "validation" / "exits").glob("*.json")):
            r = inp.json(p)
            check(p.stem == r["job"], f"validation exit name {p}")
            vexits[r["job"]] = {**r, "bundle": b}
    return {"families": out, "ledgers": ledgers, "validation_exits": vexits, "review": review}


# ----------------------------------------------------------------------------------------------------------- replication
def load_replication(inp: Inputs, roles: dict, bf1: dict, supp: dict, verify_pt: bool) -> dict:
    check(inp.sha(POOL / "FREEZE.json") == PIN["pool_freeze"] and inp.sha(POOL / "PROTOCOL.json") == PIN["pool_protocol"],
          "replication pool FREEZE / PROTOCOL differ from the pin")
    check(inp.sha(HWB / "FREEZE.json") == PIN["hw_freeze"] and inp.sha(HWB / "PROTOCOL.json") == PIN["hw_protocol"],
          "replication hw FREEZE / PROTOCOL differ from the pin")
    proto, fin, ledger = inp.json(POOL / "PROTOCOL.json"), inp.json(POOL / "FINISHED.json"), inp.json(POOL / "LEDGER.json")
    hw_ledger = inp.json(HWB / "LEDGER.json")
    check(fin["status"] == "PASS" and fin["complete"] == fin["planned_full_units"] == 12 and not fin["failed"],
          "replication FINISHED is not a complete PASS")
    check(proto["new_hyperparameter_search"] is False and proto["held_data_access"] is False, "replication protocol scope")
    inherited = proto["inherited_execution"]
    check(inherited["freeze_sha256"] == PIN["hw_freeze"], "inherited freeze")
    hw_units = {u["key"]: u for u in hw_ledger["units"]}
    units = []
    steps12 = list(range(250, 3001, 250))
    for lu in ledger["units"]:
        key, fam, seed = lu["key"], lu["family"], lu["seed"]
        check(lu["status"] == "COMPLETE_VERIFIED", f"replication {key} not COMPLETE_VERIFIED")
        inh = key in inherited["jobs_by_unit"]
        root = HWB if inh else POOL
        check(Path(lu["result_bundle"]) == root, f"replication {key}: result bundle")
        if inh:
            check(lu["job"] == inherited["jobs_by_unit"][key] == hw_units[key]["job"], f"replication {key}: inherited job")
            submit = hw_units[key]["submit_utc"]
        else:
            submit = lu["submit_utc"]
        udir = root / "results" / key
        method = proto["methods"][fam]
        check(inp.sha(method["selection"]["path"]) == method["selection"]["sha256"] == SELECTION_PIN[fam],
              f"replication {key}: selection pin")
        comp = json.loads(inp.raw(udir / "COMPLETE.json"))
        rid = comp["identity"]
        pshas = PIN["hw_protocol"] if inh else PIN["pool_protocol"]
        extra = {"replication_protocol_sha256": pshas, "result_scope": "ADDITIONAL_SEED_DEV_REPLICATION",
                 "gpu_model": rid.get("gpu_model"), "hardware_protocol_sha256": pshas}
        if "actual_partition" in rid:
            extra["actual_partition"] = rid["actual_partition"]
        if method["kind"] == "bf1":
            cd = method["candidate"]
            ident = {**bf1_identity(fam, seed, cd["lr"], cd["cap"], "FINAL_DEV", SELECTION_PIN[fam]), **extra}
            u = load_unit(inp, udir, schema="bf1", ident=ident, roles=roles, phase="FINAL_DEV", cal_steps=steps12,
                          save_steps=steps12, steps=3000, verify_pt=verify_pt, primary=False)
        else:
            check(method["release"]["sha256"] == RELEASE_PIN[fam], f"replication {key}: release pin")
            fr = supp["families"][fam]
            ident = {**supp_identity(RELEASE_PIN[fam], fam, method["candidate"], seed, "FINAL_DEV", fr["schedule"],
                                     SELECTION_PIN[fam]), **extra}
            u = load_unit(inp, udir, schema="supp", ident=ident, roles=roles, phase="FINAL_DEV", cal_steps=steps12,
                          save_steps=steps12, steps=3000, verify_pt=verify_pt, primary=False)
        check(u["complete_sha"] == lu["complete_sha256"] and u["job"] == lu["job"], f"replication {key}: ledger chain")
        check(u["gpu"] == rid["gpu_model"] and u["env"]["hardware_amendment"]["protocol_sha256"] == pshas,
              f"replication {key}: hardware identity")
        if "actual_partition" in rid:
            check(u["partition"] == rid["actual_partition"], f"replication {key}: partition")
        check(u["gpu"].startswith({"RTX6000PRO": "NVIDIA RTX PRO 6000", "L40S": "NVIDIA L40S", "H100": "NVIDIA H100"}[u["partition"]]),
              f"replication {key}: GPU name vs partition")
        rt = inp.json(root / "runtime" / f"{u['job']}.json")
        check(rt["gpu_name"] == u["gpu"] and rt["host"] == u["host"] and rt["slurm_job_id"] == u["job"],
              f"replication {key}: runtime record differs from ENV")
        rc = inp.json(root / "exits" / f"{u['job']}.json")
        check(rc["job"] == u["job"] and rc["exit_code"] == 0 and rc["freeze_sha256"] == (PIN["hw_freeze"] if inh else PIN["pool_freeze"]),
              f"replication {key}: exit receipt")
        u.update(method=fam, seed=seed, key=key, inherited=inh, receipt=rc, submit_utc=submit,
                 candidate=method["candidate"], order=method["candidate"]["order"])
        units.append(u)
    check(sorted((u["method"], u["seed"]) for u in units) == sorted((m, s) for m in proto["methods"] for s in REPL_SEEDS),
          "replication matrix is not 6 methods x seeds 1004/1005")
    rep_ledger = [json.loads(s) for s in inp.text(POOL / "COST_LEDGER.jsonl").splitlines() if s.strip()]
    hw_cost = [json.loads(s) for s in inp.text(HWB / "COST_LEDGER.jsonl").splitlines() if s.strip()]
    return {"units": units, "protocol": proto, "finished": fin, "ledger_rows": rep_ledger + hw_cost}


# --------------------------------------------------------------------------------------------------------------- checks
def cross_unit_checks(all_units: list) -> dict:
    truth, grids = {}, Counter()
    for u in all_units:
        blocks = [("CAL", u["cal"][s]["rows"]) for s in u["cal"]]
        if "dev_rows" in u:
            blocks.append(("DEV", u["dev_rows"]))
        for role, rows in blocks:
            for r in rows:
                k = (role, r["core"], r["p"])
                v = (r["theta"], r["t"][0], r["t"][1])
                check(truth.setdefault(k, v) == v, f"perturbation truth differs across units at {k}")
                grids[r["n_grid"]] += 1
    return {"truth_keys": len(truth), "n_grid": dict(grids)}


# ----------------------------------------------------------------------------------------------------------- statistics
def core_arrays(units: list, cores: list) -> dict:
    """Per unit and core: successes out of 10 (<42, <9) and the median error of the 10 perturbations."""
    idx = {c: i for i, c in enumerate(cores)}
    out = []
    for u in units:
        errs = defaultdict(list)
        for r in u["dev_rows"]:
            if r["core"] in idx:
                errs[r["core"]].append(r["err"])
        check(all(len(errs[c]) == 10 for c in cores), "core without 10 perturbations")
        out.append({"k42": np.array([sum(e < 42 for e in errs[c]) for c in cores], dtype=np.int64),
                    "k9": np.array([sum(e < 9 for e in errs[c]) for c in cores], dtype=np.int64),
                    "med": np.array([statistics.median(errs[c]) for c in cores], dtype=float)})
    return out


def boot_seed(*parts) -> int:
    return int(sha_bytes("|".join(parts).encode())[:16], 16)


def paired(A: list, B: list, cores: list, groups: dict, key: str) -> dict:
    """Core-paired difference A - B.  Core score = mean over the method's units of the per-core success fraction (or median
    error).  Success: mean over cores of the difference (= difference of the seed-averaged rates); error: median over cores of
    the per-core difference.  Core bootstrap and TMA position-group bootstrap, same random stream."""
    a, b = core_arrays(A, cores), core_arrays(B, cores)
    nA, nB, n = len(a), len(b), len(cores)
    num = {m: sum(x[m] for x in a) * nB - sum(x[m] for x in b) * nA for m in ("k42", "k9")}   # integer, / (10 nA nB)
    den = 10 * nA * nB
    dmed = sum(x["med"] for x in a) / nA - sum(x["med"] for x in b) / nB
    point = {"s42": float(num["k42"].sum()) / (den * n), "s9": float(num["k9"].sum()) / (den * n),
             "med": float(np.median(dmed))}
    sign = {m: (int((num[m] > 0).sum()), int((num[m] < 0).sum()), int((num[m] == 0).sum())) for m in ("k42", "k9")}
    seed = boot_seed("v10-phase2", key, ",".join(cores))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    core_ci = {"s42": ci(num["k42"][idx].sum(axis=1) / (den * n)), "s9": ci(num["k9"][idx].sum(axis=1) / (den * n)),
               "med": ci(np.median(dmed[idx], axis=1))}
    # position groups (record_id suffix); ordered by the first member's core order
    gl = []
    for c in cores:
        g = next(k for k, v in groups.items() if c in v)
        if g not in gl:
            gl.append(g)
    gmem = [np.array([cores.index(c) for c in groups[g]]) for g in gl]
    rng = np.random.default_rng(seed)
    gidx = rng.integers(0, len(gl), size=(N_BOOT, len(gl)))
    if all(len(m) == 1 for m in gmem):
        cidx = np.array([m[0] for m in gmem])[gidx]
        grp_ci = {"s42": ci(num["k42"][cidx].sum(axis=1) / (den * cidx.shape[1])),
                  "s9": ci(num["k9"][cidx].sum(axis=1) / (den * cidx.shape[1])),
                  "med": ci(np.median(dmed[cidx], axis=1))}
        singleton = True
    else:
        s42, s9, med = [], [], []
        for row in gidx:
            ci_ = np.concatenate([gmem[g] for g in row])
            s42.append(num["k42"][ci_].sum() / (den * len(ci_)))
            s9.append(num["k9"][ci_].sum() / (den * len(ci_)))
            med.append(np.median(dmed[ci_]))
        grp_ci = {"s42": ci(np.array(s42)), "s9": ci(np.array(s9)), "med": ci(np.array(med))}
        singleton = False
    return {"point": point, "core_ci": core_ci, "group_ci": grp_ci, "groups": len(gl), "singleton_groups": singleton,
            "sign": sign, "n_cores": n, "nA": nA, "nB": nB, "boot_seed": seed}


def ci(x: np.ndarray) -> tuple:
    lo, hi = np.quantile(x, [0.025, 0.975])
    return (float(lo), float(hi))


def seed_range(fa: dict, fb: dict, cores: list, groups: dict, key: str) -> dict:
    """Descriptive per-seed differences: matched seeds when both methods have them; single-unit methods vs each seed."""
    A, B = fa["finals"], fb["finals"]
    pairs = []
    if len(A) > 1 and len(B) > 1:
        pairs = [([A[s]], [B[s]]) for s in sorted(set(A) & set(B))]
    elif len(B) == 1:
        pairs = [([A[s]], list(B.values())) for s in sorted(A)]
    elif len(A) == 1:
        pairs = [(list(A.values()), [B[s]]) for s in sorted(B)]
    vals = {"s42": [], "s9": [], "med": []}
    for a, b in pairs:
        aa, bb = core_arrays(a, cores)[0], core_arrays(b, cores)[0]
        n = len(cores)
        vals["s42"].append(float((aa["k42"] - bb["k42"]).sum()) / (10 * n))
        vals["s9"].append(float((aa["k9"] - bb["k9"]).sum()) / (10 * n))
        vals["med"].append(float(np.median(aa["med"] - bb["med"])))
    return {m: (min(v), max(v)) for m, v in vals.items()} | {"n": len(pairs)}


# ------------------------------------------------------------------------------------------------------------------ cost
def job_records(bf1: dict, supp: dict, rep: dict) -> list:
    jobs = []
    par8_units = {u["job"]: u for u in bf1["par8"]["units"]}
    led = defaultdict(list)
    for r in bf1["ledger"]:
        led[r["job_id"]].append(r)
    for fam, fr in bf1["families"].items():
        allu = [("CAL", None, u) for u in fr["candidates"]] + [("FINAL_DEV", s, u) for s, u in fr["finals"].items()]
        seen = set()
        for phase, seed, u in allu:
            j = u["job"]
            if j in seen:
                continue
            seen.add(j)
            rc = bf1["exits"].get(j)
            check(rc is not None and rc["exit_code"] == 0, f"BF1 job {j}: missing or non-zero exit receipt")
            pu = par8_units.get(j)
            check(pu is not None and pu["family"] == fam and pu["status"] == "COMPLETE_VERIFIED"
                  and pu["phase"] == ("cal" if phase == "CAL" else "final") and pu["seed"] == seed,
                  f"BF1 job {j}: parallel8 ledger entry differs")
            jobs.append(job_entry(j, fam, phase, seed, rc, rc["start_unix"], pu["submitted_utc"], led[j], u["host"],
                                  u["gpu"], "BF1", "job_id", "slurm_starttime"))
    gate = bf1["gate_job"]
    rc = bf1["exits"][gate]
    check(rc["exit_code"] == 0, "BF1 gate exit")
    jobs.append(job_entry(gate, "BF1（共享）", "GATE", None, rc, rc["start_unix"], None, led[gate], None, L40S, "BF1",
                          "job_id", "slurm_starttime"))
    check(set(bf1["exits"]) == {j["job"] for j in jobs if j["bundle"] == "BF1"}, "BF1 exit receipts without a unit")
    for fam, fr in supp["families"].items():
        bname = "PIPE" if fr["bundle"] == PIPE else "NREC"
        lunits = {u.get("job"): u for u in supp["ledgers"][bname]["units"] if u.get("job")}
        led = defaultdict(list)
        for r in fr["ledger"]:
            led[r["job"]].append(r)
        allu = [("CAL", None, fr["candidates"][0])] + [("FINAL_DEV", s, u) for s, u in fr["finals"].items()]
        for phase, seed, u in allu:
            j = u["job"]
            rc = fr["exits"][j]
            lu = lunits.get(j)
            check(lu is not None and lu["family"] == fam and lu["status"] == "COMPLETE_VERIFIED"
                  and lu["phase"] == ("cal" if phase == "CAL" else "final"), f"{fam} job {j}: bundle ledger entry")
            jobs.append(job_entry(j, fam, phase, seed, rc, rc["start_unix"], lu["submit_utc"], led[j], u["host"], u["gpu"],
                                  bname, "job", "slurm_starttime"))
        check(set(fr["exits"]) == {j["job"] for j in jobs if j["method"] == fam}, f"{fam}: exit receipts without a unit")
    for b, root in (("PIPE", PIPE), ("NREC", NREC)):
        for lu in supp["ledgers"][b]["units"]:
            if lu["phase"] not in ("gate", "full") or not lu.get("job"):
                continue
            rc = supp["validation_exits"].get(lu["job"])
            check(rc is not None and rc["bundle"] == b, f"validation job {lu['job']}: exit receipt")
            jobs.append({"job": lu["job"], "method": lu["family"], "phase": "VALIDATION_" + lu["phase"].upper(), "seed": None,
                         "exit_code": rc["exit_code"], "start": rc["start_unix"], "end": rc["end_unix"],
                         "alloc": rc["batch_seconds"], "submit": parse_time(lu["submit_utc"]), "node": None, "gpu": None,
                         "bundle": b, "basis": "batch_shell", "ledger_end": None, "start_offset": None,
                         "status": lu["status"]})
    led = defaultdict(list)
    for r in rep["ledger_rows"]:
        led[r["job"]].append(r)
    for u in rep["units"]:
        rc = u["receipt"]
        jobs.append(job_entry(u["job"], u["method"], "REPL", u["seed"], rc, rc["start_unix"], u["submit_utc"], led[u["job"]],
                              u["host"], u["gpu"], "HW" if u["inherited"] else "POOL", "job", "batch_shell"))
    return jobs


def job_entry(j, fam, phase, seed, rc, start, submit, led, node, gpu, bundle, jk, basis) -> dict:
    ends = [r for r in led if r["event"] == "END"]
    starts = [r for r in led if r["event"] == "START"]
    check(len(ends) == 1 and len(starts) == 1, f"job {j}: ledger START/END count {len(starts)}/{len(ends)}")
    check(ends[0].get("status") in ("COMPLETE", "PASS"), f"job {j}: ledger END status {ends[0].get('status')}")
    alloc = rc["allocated_seconds"]
    check(abs((rc["end_unix"] - rc["start_unix"]) - alloc) < 1e-6, f"job {j}: receipt arithmetic")
    check(ends[0]["allocated_seconds"] <= alloc + 1e-6, f"job {j}: ledger END after the exit receipt")
    return {"job": j, "method": fam, "phase": phase, "seed": seed, "exit_code": rc["exit_code"], "start": start,
            "end": rc["end_unix"], "alloc": alloc, "submit": parse_time(submit) if submit else None, "node": node,
            "gpu": gpu, "bundle": bundle, "basis": basis, "ledger_end": ends[0]["allocated_seconds"],
            "start_offset": parse_time(starts[0]["utc"]) - start, "status": ends[0].get("status"),
            "checkpoint_events": {(r.get("unit"), r.get("step")): r["allocated_seconds"] for r in led
                                  if r["event"] == "CHECKPOINT"}}


def t_avail_cal(fr: dict, u: dict, job: dict, step: int) -> float:
    """Allocated seconds (job clock) at which the CAL readout of `step` in unit `u` was complete.  BF1: the CHECKPOINT ledger
    event written right after the CAL readout.  Supplement: the first training record after the readout (upper bound), or the
    COMPLETE timestamp for the last CAL step (written right after it, before artifact hashing)."""
    if fr["source"] == "BF1":
        v = job["checkpoint_events"].get((u["dir"].name, step))
        check(v is not None, f"{u['where']}: no CHECKPOINT event at {step}")
        return v
    if step != u["cal_steps"][-1]:
        return parse_time(u["train"]["utc_after"][step]) - job["start"]
    check(u["phase"] == "CAL", "t_avail of the last CAL step of a final unit is not used")
    return parse_time(u["complete"]["utc"]) - job["start"]


def same_total_cost(fams: dict, jobs: list) -> dict:
    byjob = {(j["job"], j["phase"]): j for j in jobs}
    out = {}
    for fam, fr in fams.items():
        cj = byjob[(fr["cal_job"], "CAL")]
        avail = []
        for u in fr["candidates"]:
            for s in u["cal_steps"]:
                t = t_avail_cal(fr, u, cj, s)
                avail.append({"t": t, "order": u["order"], "step": s, "metrics": u["cal"][s]["metrics"]})
        times = [a["t"] for a in avail]
        check(all(0 <= t <= cj["alloc"] + 1e-6 for t in times), f"{fam}: availability outside the CAL allocation")
        sel = fr["selected_choice"]
        sel_t = next(a["t"] for a in avail if a["order"] == sel["order"] and a["step"] == sel["step"])
        levels = []
        for b in TUNING_BUDGETS_H:
            pool = [a for a in avail if a["t"] <= b * 3600]
            best = min(pool, key=lambda a: sel_key(a)) if pool else None
            levels.append({"budget_h": b, "choice": best, "is_final": bool(best) and best["order"] == sel["order"]
                           and best["step"] == sel["step"]})
        finals = {}
        for s, u in fr["finals"].items():
            fj = byjob[(u["job"], "FINAL_DEV")]
            last = u["cal_steps"][-1]
            st = u["selected"]["step"]
            if st == last:
                cmin, how = fj["alloc"], "最后一个 CAL checkpoint 即选中：等于实际成本"
            else:
                ta = t_avail_cal(fr, u, fj, st) if fr["source"] == "BF1" else parse_time(u["train"]["utc_after"][st]) - fj["start"]
                tl = fj["checkpoint_events"].get((u["dir"].name, last))
                check(tl is not None, f"{u['where']}: no CHECKPOINT event at the last step")
                cmin, how = ta + (fj["alloc"] - tl), "选中 checkpoint 的 CAL 完成 + 末段 DEV 阶段（保守）"
            check(0 < cmin <= fj["alloc"] + 1e-6, f"{u['where']}: minimal cost outside the allocation")
            finals[s] = {"actual": fj["alloc"], "cmin": cmin, "how": how, "step": st}
        actual = cj["alloc"] + sum(f["actual"] for f in finals.values())
        minimal = sel_t + sum(f["cmin"] for f in finals.values())
        out[fam] = {"cal_alloc": cj["alloc"], "sel_t": sel_t, "levels": levels, "finals": finals, "actual": actual,
                    "minimal": minimal, "n_avail": len(avail)}
    return out


# ----------------------------------------------------------------------------------------------------------- diagnostics
def cal_saturation(fr: dict) -> dict:
    units = fr["candidates"] + list(fr["finals"].values())
    sel_full = sum(1 for u in units if u["selected"]["metrics"]["success42"] == 1.0 and u["selected"]["metrics"]["success9"] == 1.0)
    full_counts = [sum(1 for s in u["cal"] if u["cal"][s]["metrics"]["success42"] == 1.0 and u["cal"][s]["metrics"]["success9"] == 1.0)
                   for u in units]
    decided = Counter()
    margins = []
    for u in units:
        recs = sorted(({"step": s, "metrics": u["cal"][s]["metrics"]} for s in u["cal"]), key=sel_key)
        if len(recs) < 2:
            continue
        k0, k1 = sel_key(recs[0]), sel_key(recs[1])
        crit = next(i for i in range(4) if k0[i] != k1[i])
        decided[("s42", "s9", "median", "step")[crit]] += 1
        if crit == 2 and k0[0] == -1.0 and k0[1] == -1.0:
            margins.append(k1[2] - k0[2])
    fam_margin = None
    cc = sorted(fr["selection"]["completed_candidates"], key=lambda r: sel_key(r["choice"]))
    if len(cc) > 1:
        k0, k1 = sel_key(cc[0]["choice"]), sel_key(cc[1]["choice"])
        crit = next(i for i in range(5) if k0[i] != k1[i])
        fam_margin = {"criterion": ("s42", "s9", "median", "step", "order")[crit],
                      "selected": cc[0]["choice"]["metrics"]["median_err"], "runner_up": cc[1]["choice"]["metrics"]["median_err"],
                      "runner_order": cc[1]["candidate"]["order"]}
    cal9 = [u["selected"]["metrics"]["success9"] for u in fr["finals"].values()]
    dev9 = [u["dev32"]["success9"] for u in fr["finals"].values()]
    return {"n_units": len(units), "sel_full": sel_full, "full_counts": full_counts, "decided": dict(decided),
            "median_margins": margins, "fam_margin": fam_margin, "cal9_mean": statistics.fmean(cal9),
            "dev9_mean": statistics.fmean(dev9), "n_ckpt": len(units[0]["cal"])}


def stall_stats(u: dict) -> dict:
    g = u["train"]["grad"]
    return {"g1": float(np.median(g[:250])), "g2": float(np.median(g[250:1000])), "g3": float(np.median(g[1000:])),
            "tiny": float(np.mean(g < 1e-6)), "max_s42": max(u["cal"][s]["metrics"]["success42"] for s in u["cal"])}


def clip_fraction(u: dict) -> float:
    clip = u["config"].get("clip_norm")
    check(clip == 1.0, f"{u['where']}: clip_norm {clip}")
    g = u["train"]["grad"]
    check(len(g) == u["steps"], f"{u['where']}: gradient norm per step missing")
    return float(np.mean(g > clip))


def m1_stats(u: dict) -> dict:
    t = u["train"]
    n = len(t["m1_stop"])
    check(n == u["steps"], f"{u['where']}: M1 record per step missing")
    return {"cap": u["candidate"]["cap"], "lr": u["candidate"]["lr"], "stops": dict(sorted(Counter(t["m1_stop"]).items())),
            "cap_frac": sum(r == "iteration_budget_exhausted" for r in t["m1_stop"]) / n,
            "iter_med": statistics.median(t["m1_iter"]), "rel_med": statistics.median(t["m1_rel"])}


def strata(rows: list) -> list:
    out = []
    for lo, hi in ROT_STRATA:
        rs = [r for r in rows if lo <= abs(r["theta"]) < hi or (hi == ROT_STRATA[-1][1] and abs(r["theta"]) == hi)]
        out.append({"lo": lo, "hi": hi, "n": len(rs), "s42": sum(r["err"] < 42 for r in rs) / len(rs),
                    "s9": sum(r["err"] < 9 for r in rs) / len(rs)})
    return out


# ------------------------------------------------------------------------------------------------------------ placeholders
def xhw_diagnostics(inp: Inputs, torch_rec: dict, geom_rec: dict) -> dict:
    if not XHW.is_dir():
        return {"present": False}
    ref = {p: torch_rec["parts"][f"{p}_cuda"] for p in ("verify_torch", "core_tol", "adapter")}
    ref["geometry"] = geom_rec["parts"]["geometry_cuda"]
    tags = []
    for tdir in sorted(p for p in XHW.iterdir() if p.is_dir()):
        parts = {}
        for pf in sorted((tdir / "T01" / "parts").glob("*_cuda.json")):
            d = inp.json(pf)
            name = d["part"]
            fails = [c for c in d.get("checks", []) if c.get("status") == "FAIL" and c.get("required") in (True, "True")]
            texts = [str(c.get("error", "")) + str(c.get("stderr_tail", "")) for c in fails] + [str(d.get("error", ""))]
            infra = d["status"] == "FAIL" and all(DEVICE_UNAVAILABLE in t for t in texts if t) and any(texts)
            prov = d.get("provenance", {})
            r = ref[name]
            same = None
            if d["status"] == "PASS":
                same = ((d["n_checks"], d["n_required"], d["n_pass"], d["n_fail"]) ==
                        (r["n_checks"], r["n_required"], r["n_pass"], r["n_fail"])
                        and sorted(d["diagnostic_failures"]) == sorted(r["diagnostic_failures"]))
            parts[name] = {"status": d["status"], "n": [d.get(k) for k in ("n_checks", "n_required", "n_pass", "n_fail")],
                           "ref_n": [r[k] for k in ("n_checks", "n_required", "n_pass", "n_fail")],
                           "diag": sorted(d.get("diagnostic_failures") or []), "infra": infra, "same_as_l40s": same,
                           "gpu": prov.get("gpu_name"), "host": prov.get("host"), "job": prov.get("slurm_job_id"),
                           "commit": prov.get("git_commit"), "core": prov.get("code_hash_src_scripts_v10"),
                           "pin": (d.get("job_pin") or {}).get("status")}
        tags.append({"tag": tdir.name, "parts": parts})
    return {"present": True, "tags": tags}


def discover_completes(root: Path) -> list:
    return sorted(p for p in root.rglob("COMPLETE.json") if "validation" not in p.parts)


def placeholder_generic(inp: Inputs, root: Path, unit_roots=(), depth=1) -> dict:
    """Root-level JSON records are listed; completed units (COMPLETE.json) are verified and summarised; unit directories
    still running are only counted, so the report does not churn while a bundle fills."""
    if not root.exists():
        return {"state": "absent"}
    files = sorted(p for p in root.glob("*.json") if p.is_file())
    if depth >= 2:
        files += sorted(p for p in root.glob("*/*.json") if p.is_file() and "validation" not in p.parts)
    entries = []
    for p in files:
        b = inp.raw(p)
        try:
            d = json.loads(b)
        except ValueError:
            refuse(f"{p}: not JSON")
        entries.append({"path": str(p.relative_to(root)), "sha": sha_bytes(b),
                        "status": d.get("status") if isinstance(d, dict) else None,
                        "schema": d.get("schema") if isinstance(d, dict) else None})
    prereg = None
    for name in ("PREREGISTRATION.json", "PROTOCOL.json"):
        pp = root / name
        if pp.exists():
            b = inp.raw(pp)
            d = json.loads(b)
            rpt, st = d.get("reporting", {}), d.get("statistics", {})
            prereg = {"file": name, "status": d.get("status"), "sha": sha_bytes(b), "table": rpt.get("table", ""),
                      "implementation": st.get("implementation", ""), "label": d.get("label", "")}
            break
    comps = []
    roots = [root] + [r for r in unit_roots if r.exists()]
    for c in [c for r in roots for c in discover_completes(r)]:
        d = json.loads(inp.raw(c))
        udir = c.parent
        for name, h in sorted(d.get("artifacts", {}).items()):
            if not name.endswith(".pt"):
                check(inp.sha(udir / name) == h, f"{udir}/{name}: sha256 differs from COMPLETE")
        idn = d.get("identity", {})
        job = str(d.get("job_id") or d.get("job") or "")
        rc = None
        for ex in sorted(root.rglob(f"exits/{job}.json")) if job else []:
            rc = json.loads(inp.raw(ex))
        comps.append({"unit": udir.name, "status": d.get("status"), "family": idn.get("family"),
                      "K": idn.get("K"), "seed": idn.get("seed"), "phase": idn.get("phase"), "DEV32": d.get("DEV32"),
                      "DEV24": d.get("DEV24"), "job": job, "exit_code": None if rc is None else rc.get("exit_code"),
                      "alloc": None if rc is None else rc.get("allocated_seconds"),
                      "step": (d.get("selected") or {}).get("step")})
    running = sorted(p.parent.name for r in roots for p in r.rglob("IDENTITY.json")
                     if "validation" not in p.parts and not (p.parent / "COMPLETE.json").exists())
    return {"state": "present", "files": entries, "completes": comps, "prereg": prereg, "running": running}


def sacct_probe_snapshot(inp: Inputs) -> dict | None:
    p = REPO / SACCT_PROBE_REL
    if not p.exists():
        return None
    return inp.json(p)


def probe_sacct(jobs: list):
    ids = sorted({j for j in jobs})
    cmd = ["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "-o", "JobID,Start,End,Elapsed,NodeList,State,ExitCode"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        rec = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        rec = {"returncode": None, "stdout": "", "stderr": repr(e)}
    rec.update(utc=utc_str(dt.datetime.now(dt.timezone.utc).timestamp()), command=" ".join(cmd[:5]) + " <jobs> " + " ".join(cmd[6:]),
               jobs=ids)
    (REPO / SACCT_PROBE_REL).write_text(json.dumps(rec, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print("wrote", SACCT_PROBE_REL, "returncode", rec["returncode"])


# ------------------------------------------------------------------------------------- results completed after 2026-10-19
def committed(inp: Inputs, repo: Path, rel: str) -> bytes:
    """A record read from a worktree must equal its committed version at that worktree's HEAD."""
    b = inp.raw(repo / rel)
    r = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{rel}"], capture_output=True)
    check(r.returncode == 0 and r.stdout == b, f"{repo / rel}: differs from the committed version at HEAD")
    return b


def gate_pass(inp: Inputs, ref: dict, gpu: str, where: str) -> dict:
    check(inp.sha(ref["path"]) == ref["sha256"], f"{where}: gate record hash")
    g = inp.json(ref["path"])
    check(g["status"] == "PASS" and g["environment"]["gpu_name"] == gpu, f"{where}: gate not PASS on {gpu}")
    return g


def load_rulsif_v2_bf(inp: Inputs, roles: dict, verify_pt: bool) -> dict:
    """BF RuLSIF_exact_v2: CAL under release 41313c39 (L40S), finals under the cross-hardware release 12e37693."""
    fam = RV2_FAMILY
    if not (RV2 / "RELEASE.json").exists():
        return {"state": "absent"}
    rel_b, xr_b = inp.raw(RV2 / "RELEASE.json"), inp.raw(RV2 / "RELEASE_XHW.json")
    check(sha_bytes(rel_b) == PIN["rv2_release"] and sha_bytes(xr_b) == PIN["rv2_release_xhw"], "RuLSIF v2 releases differ")
    rel, xr = json.loads(rel_b), json.loads(xr_b)
    am = json.loads(committed(inp, RV2_EXEC, RV2_AMENDMENT_REL))
    check(sha_bytes(inp.raw(RV2_EXEC / RV2_AMENDMENT_REL)) == PIN["rv2_amendment"], "A1 amendment differs from the pin")
    check(rel["amendment"]["sha256"] == xr["amendment"]["sha256"] == PIN["rv2_amendment"]
          and xr["base_release"]["sha256"] == PIN["rv2_release"] and rel["family"] == xr["family"] == fam,
          "RuLSIF v2 release chain")
    spec = rel["families"][fam]
    cands, schedule = spec["candidates"], spec["schedule"]
    rule = am["amended_rule"]
    check([c["order"] for c in cands] == rule["eligible_orders"] and rule["excluded_orders"] == [8],
          "RuLSIF v2 candidates differ from the A1 eligible set")
    res = RV2 / "results"
    sel_p = res / "CAL" / fam / "SELECTION.json"
    if not sel_p.exists():
        return {"state": "pending_cal"}
    sel_b = inp.raw(sel_p)
    check(sha_bytes(sel_b) == PIN["rv2_selection"] == xr["cal_selection"]["sha256"], "RuLSIF v2 SELECTION differs")
    sel = json.loads(sel_b)
    check(sel["status"] == "SELECTED" and sel["family"] == fam and sel["release_sha256"] == PIN["rv2_release"]
          and sel["budget_exhausted"] is False and [c["candidate"] for c in sel["completed_candidates"]] == cands,
          "RuLSIF v2 selection header / candidates")
    exits = {}
    for p in sorted((RV2 / "exits").glob("*.json")):
        r = inp.json(p)
        check(p.stem == r["job"], f"RuLSIF v2 exit receipt name {p}")
        exits[r["job"]] = r
    cand_units = []
    for c in sel["completed_candidates"]:
        cd = c["candidate"]
        u = load_unit(inp, res / "CAL" / fam / f"candidate_{cd['order']:02d}", schema="supp",
                      ident=supp_identity(PIN["rv2_release"], fam, cd, 1017, "CAL", schedule, None), roles=roles, phase="CAL",
                      cal_steps=schedule["cal_steps"], save_steps=schedule["save_steps"], steps=schedule["steps"],
                      verify_pt=verify_pt, primary=True)
        check(u["complete_sha"] == c["complete_sha256"] and {**u["selected"], "order": cd["order"]} == c["choice"]
              and u["env"]["git_commit"] == rel["source_commit"], "RuLSIF v2 candidate chain")
        u.update(method=fam, seed=1017, candidate=cd, order=cd["order"])
        cand_units.append(u)
    check(min(sel["completed_candidates"], key=lambda r: sel_key(r["choice"])) == sel["selected"],
          "RuLSIF v2 family selection differs from the lexicographic rule")
    check({u["job"] for u in cand_units} == {sel["job"]}, "RuLSIF v2 CAL candidates from more than one job")
    rc = exits.get(sel["job"])
    check(rc is not None and rc["exit_code"] == 0 and rc["release_sha256"] == PIN["rv2_release"]
          and inp.sha(RV2 / "exits" / f"{sel['job']}.json") == xr["cal_exit"]["sha256"], "RuLSIF v2 CAL exit receipt")
    selcd = sel["selected"]["candidate"]
    finals, pending = {}, []
    for seed in schedule["final_seeds"]:
        ud = res / "FINAL_DEV" / fam / f"seed_{seed}"
        if not (ud / "COMPLETE.json").exists():
            pending.append(seed)
            continue
        u = load_unit(inp, ud, schema="supp",
                      ident=supp_identity(PIN["rv2_release_xhw"], fam, selcd, seed, "FINAL_DEV", schedule, PIN["rv2_selection"]),
                      roles=roles, phase="FINAL_DEV", cal_steps=schedule["cal_steps"], save_steps=schedule["save_steps"],
                      steps=schedule["steps"], verify_pt=verify_pt, primary=False)
        rcf = exits.get(u["job"])
        check(rcf is not None and rcf["exit_code"] == 0 and rcf["release_sha256"] == PIN["rv2_release_xhw"]
              and u["env"]["git_commit"] == xr["source_commit"], f"RuLSIF v2 final {seed}: receipt / source")
        hw = inp.json(ud / "HARDWARE.json")
        check(hw["gpu_name"] == u["gpu"] and hw["node"] == u["host"] and hw["partition"] == u["partition"]
              and hw["first_job"] == u["job"], f"RuLSIF v2 final {seed}: HARDWARE.json differs from ENV")
        if u["gpu"] == L40S:
            gate_pass(inp, xr["l40s_gate"], L40S, f"RuLSIF v2 final {seed}")
            gate = {"job": "L40S（CAL 发行门禁）", "basis": "l40s_gate"}
        else:
            gate_pass(inp, hw["gate_report"], u["gpu"], f"RuLSIF v2 final {seed}")
            check(inp.sha(hw["gate_receipt"]["path"]) == hw["gate_receipt"]["sha256"]
                  and inp.json(hw["gate_receipt"]["path"])["exit_code"] == 0, f"RuLSIF v2 final {seed}: gate receipt")
            gate = {"job": hw["gate_job"], "basis": hw["gate_basis"]}
        u.update(method=fam, seed=seed, candidate=selcd, order=selcd["order"], receipt=rcf, gate=gate)
        finals[seed] = u
    check(xr["owner_decision"]["chosen_option_verbatim"] == "final 放开，CAL 留 L40S", "RuLSIF v2 XHW owner decision")
    return {"state": "complete" if not pending else "partial", "release": rel, "xhw": xr, "amendment": am,
            "selection": sel, "candidates": cand_units, "finals": finals, "pending": pending, "schedule": schedule,
            "selected_candidate": selcd, "selected_choice": sel["selected"]["choice"], "exits": exits, "cal_job": sel["job"]}


def grid_label(ci95, deltas):
    lo, hi = ci95
    if lo <= 0 <= hi:
        return "B_no_detectable_response"
    sign = 1 if lo > 0 else -1
    return "A_consistent_response" if all(d * sign > 0 for d in deltas) else "C_mixed"


def load_capacity(inp: Inputs, roles: dict, verify_pt: bool, fams: dict) -> dict:
    tp = CAP_HW / "CAPACITY_TABLE_HW.json"
    if not tp.exists():
        return {"state": "table_absent"}
    tb = inp.raw(tp)
    check(sha_bytes(tb) == PIN["cap_table_hw"], "capacity table differs from the pin")
    t = json.loads(tb)
    check(inp.sha(CAPACITY / "FREEZE.json") == PIN["cap_base_freeze"] == t["base_freeze_sha256"]
          and inp.sha(CAP_HW / "FREEZE.json") == PIN["cap_hw_freeze"] == t["hw_freeze_sha256"]
          and inp.sha(CAP_HW / "HARDWARE_AMENDMENT.json") == PIN["cap_hw_amendment"] == t["hardware_amendment_sha256"]
          and inp.sha(CAPACITY / "PREREGISTRATION.json") == PIN["cap_prereg"] and t["complete"] is True,
          "capacity freeze / amendment / preregistration chain")
    pre, hwa = inp.json(CAPACITY / "PREREGISTRATION.json"), inp.json(CAP_HW / "HARDWARE_AMENDMENT.json")
    steps12 = list(range(250, 3001, 250))
    units, cells = [], []
    for cell in t["cells"]:
        fam, K = cell["family"], cell["K"]
        check(cell["status"] == "COMPLETE" and cell["missing"] == [] and K in (8, 32), f"capacity cell {fam} K{K}")
        sel = fams[fam]["selected_candidate"]
        cu = []
        for e in cell["units"]:
            seed, hwu = e["seed"], e["source"] == "hw_amendment"
            check(e["source"] in ("base_freeze", "hw_amendment") and e["unit"] == f"{fam}_K{K}_seed{seed}", "capacity unit key")
            ident = {"schema": "v10.capacity_ext.unit.hw_amendment.v1" if hwu else "v10.capacity_ext.unit.v1",
                     "freeze_sha256": PIN["bf1_frozen"], "family": fam, "method": BF1_METHOD[fam], "seed": seed,
                     "lr": sel["lr"], "cap": sel["cap"], "phase": "FINAL_DEV", "steps": 3000, "checkpoint_interval": 250, "K": K,
                     "dependency": SELECTION_PIN[fam], "data_lock_sha256": PIN["bf1_data_lock"],
                     "extension": "BF1_CAPACITY_EXT_K8_K32", "preregistration_sha256": PIN["cap_prereg"],
                     "capacity_freeze_sha256": PIN["cap_base_freeze"], "result_scope": "CAPACITY_EXTENSION_SEPARATE_TABLE"}
            if hwu:
                ident.update(hardware_amendment_sha256=PIN["cap_hw_amendment"], capacity_hw_freeze_sha256=PIN["cap_hw_freeze"])
            u = load_unit(inp, (CAPACITY_HW_OUT if hwu else CAPACITY_OUT) / e["unit"], schema="bf1", ident=ident, roles=roles,
                          phase="FINAL_DEV", cal_steps=steps12, save_steps=steps12, steps=3000, verify_pt=verify_pt, primary=False)
            k16 = fams[fam]["finals"][seed]
            check(u["dev32"] == e["DEV32"] and u["dev24"] == e["DEV24"] and e["DEV32_k16"] == k16["dev32"]
                  and e["DEV24_k16"] == k16["dev24"] and e["selected_step"] == u["selected"]["step"]
                  and e["k16_selected_step"] == k16["selected"]["step"], f"capacity {e['unit']}: table differs from the records")
            for blk in ("DEV32", "DEV24"):
                for m in MKEYS:
                    check(abs(e[f"{blk}_delta"][m] - (e[blk][m] - e[f"{blk}_k16"][m])) <= 1e-12, f"capacity {e['unit']}: delta")
            check(e["hardware"] == [[u["gpu"], u["partition"], u["host"]]] and e["stratum"] == u["gpu"]
                  and e["k16_reference_hardware"] == [k16["gpu"]] and e["hardware_matched_to_k16"] == (u["gpu"] == k16["gpu"])
                  and e["cost"]["jobs"] == [u["job"]], f"capacity {e['unit']}: hardware / job")
            rc = inp.json(CAPACITY / "exits" / f"{u['job']}.json")
            check(rc["exit_code"] == 0 and rc["job"] == u["job"] and rc["unit"] == e["unit"]
                  and abs(rc["allocated_seconds"] / 3600 - e["cost"]["allocated_gpu_hours"]) < 1e-9, f"capacity {e['unit']}: receipt")
            u.update(method=fam, seed=seed, K=K, key=e["unit"], source=e["source"], receipt=rc, k16=k16)
            cu.append(u)
        views = [("pooled", cell["pooled"], cu)] + [(g, v, [x for x in cu if x["gpu"] == g]) for g, v in sorted(cell["strata"].items())]
        rendered = []
        for name, view, members in views:
            check(view["seeds"] == [x["seed"] for x in members] and members, f"capacity {fam} K{K} {name}: seeds")
            out = {"name": name, "seeds": view["seeds"], "single": len(members) == 1}
            for blk in ("DEV32", "DEV24"):
                for m in ("success42", "success9"):
                    ds = [x[blk.lower()][m] - x["k16"][blk.lower()][m] for x in members]
                    v = view[blk][m]
                    check(len(v["delta_per_seed"]) == len(ds) and all(abs(a - b) <= 1e-12 for a, b in zip(v["delta_per_seed"], ds))
                          and abs(v["delta_mean"] - statistics.fmean(ds)) <= 1e-12, f"capacity {fam} K{K} {name}: delta mean")
                    want = "single_seed_descriptive" if len(members) == 1 else grid_label(v["ci95"], ds)
                    check(v["grid"] == want, f"capacity {fam} K{K} {name} {blk} {m}: grid label {v['grid']} != {want}")
                    out[(blk, m)] = {"mean": statistics.fmean(ds), "ci": tuple(v["ci95"]), "grid": v["grid"], "per_seed": ds}
            rendered.append(out)
        cells.append({"family": fam, "K": K, "views": rendered})
        units += cu
    alloc = sum(u["receipt"]["allocated_seconds"] for u in units)
    check(abs(t["total_allocated_gpu_hours"] - alloc / 3600) < 1e-6, "capacity total allocated hours")
    bytype = defaultdict(float)
    for u in units:
        bytype[u["gpu"]] += u["receipt"]["allocated_seconds"] / 3600
    check({g: round(v, 3) for g, v in bytype.items()} == t["gpu_hours_by_type"], "capacity GPU-h by type")
    return {"state": "complete", "table": t, "cells": cells, "units": units, "prereg": pre, "hwa": hwa,
            "bytype": dict(sorted(bytype.items())), "total_s": alloc, "ceiling_h": t["ceiling_gpu_hours"]}


def load_sensitivity(inp: Inputs, fams: dict) -> dict:
    rp = SENSITIVITY / "RESULTS.json"
    if not rp.exists():
        return {"state": "results_absent"}
    rb = inp.raw(rp)
    check(sha_bytes(rb) == PIN["sens_results"], "sensitivity RESULTS.json differs from the pin")
    r = json.loads(rb)
    check(r["preregistration_sha256"] == PIN["sens_protocol"] == inp.sha(SENSITIVITY / "PROTOCOL.json")
          and all(r[d]["status"] == "COMPLETE" for d in ("D1", "D2", "D3")), "sensitivity preregistration / status")
    comir, mind = fams["CoMIR_native"]["finals"], fams["MIND_2D_adapted"]
    for s in (1001, 1002, 1003):
        uu = r["D1"]["units"][f"FINAL_{s}"]
        check(uu["DEV32_frozen_epoch23"] == comir[s]["dev32"] and uu["DEV24_frozen_epoch23"] == comir[s]["dev24"],
              f"D1 frozen epoch-23 rows differ from the verified CoMIR final {s}")
    fz = r["D2"]["frozen"]
    check(fz["DEV32"] == mind["finals"][1001]["dev32"] and fz["DEV24"] == mind["finals"][1001]["dev24"]
          and fz["CAL"] == mind["candidates"][0]["cal"][0]["metrics"], "D2 frozen rows differ from the verified MIND unit")
    mine = strata(mind["finals"][1001]["dev_rows"])
    for x, key in zip(mine, ("0-10", "10-20", "20-30")):
        check(fz["DEV32_strata"][key]["n"] == x["n"] and fz["DEV32_strata"][key]["success42"] == x["s42"],
              f"D2 frozen stratum {key} differs from §6.6")
    hours = defaultdict(float)
    for rec in r["costs"]["receipts"]:
        e = inp.json(SENSITIVITY / "exits" / f"{rec['job']}.json")
        check(e == rec and e["exit_code"] == 0, f"sensitivity receipt {rec['job']}")
        hours[f"{rec['diagnostic']}|{rec['partition']}"] += rec["allocated_seconds"] / 3600
    check(set(hours) == set(r["costs"]["allocated_gpu_hours"])
          and all(abs(hours[k] - v) < 1e-9 for k, v in r["costs"]["allocated_gpu_hours"].items()), "sensitivity costs")
    d3 = []
    for j in r["D3"]["jobs"]:
        summ = inp.json(SENSITIVITY / "D3" / "summaries" / f"{j['job']}.json")
        gp = SENSITIVITY / "D3" / "validation" / "runs" / j["job"] / j["family"] / "GATE.json"
        check(summ["status"] == j["status"] == "PASS" and summ["gate_sha256"] == inp.sha(gp), f"D3 gate record {j['job']}")
        g = inp.json(gp)
        ex, rs = g["resume"]["expected_second"], g["resume"]["resumed_second"]
        check(set(ex) == set(rs), f"D3 resume fields {j['job']}")
        diffs = {k: (ex[k], rs[k]) for k in sorted(ex) if ex[k] != rs[k]}
        check(j["resume_bitwise_equal_this_gpu"] == (not diffs), f"D3 resume_bitwise_equal_this_gpu {j['job']}")
        d3.append({"job": j["job"], "family": j["family"], "gpu": j["gpu_name"], "diffs": diffs,
                   "same": sorted(k for k in ex if ex[k] == rs[k]), "restored": g["resume"]["optimizer_and_rng_restored"]})
    return {"state": "complete", "r": r, "d3_resume": d3}


def load_hardware_disclosure(inp: Inputs, rv2: dict) -> dict:
    hwa = inp.json(CAP_HW / "HARDWARE_AMENDMENT.json")
    am = json.loads(committed(inp, EPFL_HW2_SRC, "v10/epfl/EPFL_XHW_GATE_AMENDMENT_20261020.json"))
    check(inp.sha(EPFL_HW2_SRC / "v10/epfl/EPFL_XHW_GATE_AMENDMENT_20261020.json") == PIN["epfl_xhw_amendment"],
          "EPFL XHW gate amendment differs from the pin")
    xr = rv2["xhw"]["owner_decision"] if rv2.get("xhw") else None
    cap = hwa["authority"]
    strip = lambda s: s.strip("「」")
    check(xr is None or (strip(cap["owner_choice_verbatim"]) == xr["chosen_option_verbatim"]
                         and cap["option_as_offered_verbatim"] == xr["offered_as_verbatim"]
                         and strip(cap["standing_instruction_verbatim"]) == xr["standing_instruction_0931_utc_verbatim"]),
          "owner decision text differs between the RuLSIF v2 and capacity records")
    fg = am["failed_gate"]
    hw_b, hw2_b = inp.raw(EPFL_HW / "RELEASE.json"), inp.raw(EPFL_HW2 / "RELEASE.json")
    check(sha_bytes(hw_b) == PIN["epfl_release_hw"] and sha_bytes(hw2_b) == PIN["epfl_release_hw2"],
          "EPFL hardware releases differ from the pins")
    hw_rel, hw2_rel = json.loads(hw_b), json.loads(hw2_b)
    check(hw2_rel["xhw_gate_amendment"]["sha256"] == PIN["epfl_xhw_amendment"], "EPFL hw2 release does not cite the XHW amendment")
    g0 = inp.json(EPFL_HW / "xhw_gates" / "H100" / "gpu_gate" / fg["job"] / "GATE.json")
    failed = {k: v["required_fail"] for k, v in g0["parts"].items() if v["required_fail"]}
    check(g0["status"] == "FAIL" and g0["environment"]["slurm_job_id"] == fg["job"] and g0["environment"]["host"] == fg["node"]
          and g0["pins"]["commit"] == fg["commit"] == hw_rel["source_commit"] and failed == {"checkpoint": fg["failed_rows"]}
          and sorted(fg["failed_rows"]) == sorted(am["rule"]["changed_rows"]),
          "EPFL failed H100 gate: record differs from the amendment or from the hw release")
    gates = []
    gdir = EPFL_HW2 / "xhw_gates"
    for gp in sorted(gdir.glob("*/gpu_gate/*/GATE.json")) if gdir.exists() else []:
        g = inp.json(gp)
        rows = {k: v for k, v in g["checkpoint_rows_measured"].items() if k in am["rule"]["changed_rows"]}
        gates.append({"gpu": g["environment"]["gpu_name"], "node": g["environment"]["host"], "job": g["environment"]["slurm_job_id"],
                      "status": g["status"], "rows": rows,
                      "state_hash": g["checkpoint_rows_measured"]["checkpoint.state_hash_equal_after_reload"]["status"]})
    return {"cap_authority": cap, "rv2_owner": xr, "epfl_amendment": am, "epfl_gates": gates}


def check_epfl_disclosure(R: dict):
    """The official EPFL analysis cites the same cross-hardware gates (job, status, measured rows) and owner words as the records."""
    ep, h = R["epfl"], R["hwd"]
    if ep["state"] != "present" or not ep["known"]:
        return
    hwd = ep["data"]["disclosures"]["hardware"]
    label = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX6000PRO", "NVIDIA H100 NVL": "H100"}
    for g in h["epfl_gates"]:
        o = hwd["cross_hardware_gates"][label[g["gpu"]]]
        check(o["job"] == g["job"] and o["status"] == g["status"]
              and all(o["checkpoint_rows_measured"][k]["status"] == row["status"]
                      and o["checkpoint_rows_measured"][k]["gradient_max_abs"] == row["gradient"]["max_abs_err"]
                      and o["checkpoint_rows_measured"][k]["gradient_max_rel"] == row["gradient"]["max_rel_err"]
                      for k, row in g["rows"].items()), f"EPFL official analysis: gate {g['job']} differs from its record")
    check(hwd["owner_quotes"]["2026-10-19"] == h["cap_authority"]["owner_choice_verbatim"].strip("「」")
          and hwd["owner_quotes"]["2026-10-20"] == h["epfl_amendment"]["owner"]["words_verbatim"],
          "EPFL official analysis: owner words differ from the records")


def load_epfl(inp: Inputs) -> dict:
    """EPFL is not recomputed here. The report reads the official frozen-protocol analysis written by the EPFL agent
    (EPFL_RESULTS.json) when it exists and records its sha256; otherwise EPFL is pending. Besides that file only records are
    read: the frozen protocol's reporting rule and the RuLSIF_exact_v2 addendum's labels."""
    proto_b = committed(inp, EPFL_SRC, "v10/epfl/EPFL_PROTOCOL_FROZEN.json")
    check(sha_bytes(proto_b) == PIN["epfl_protocol"], "EPFL frozen protocol differs from the pin")
    add_b = committed(inp, EPFL_RV2_SRC, "v10/epfl/EPFL_RULSIF_V2_ADDENDUM_REV3.json")
    check(sha_bytes(add_b) == PIN["epfl_rv2_addendum3"], "EPFL RuLSIF v2 addendum rev3 differs from the pin")
    add = json.loads(add_b)
    mdict = re.fullmatch(r"a change of kernel dictionary size, (\d+) -> 1024 \(.*", add["reporting"]["must_state"][0])
    check(mdict is not None, "EPFL RuLSIF v2 addendum: dictionary change text")
    out = {"state": "pending", "protocol": json.loads(proto_b), "addendum": add, "dict_from": int(mdict.group(1))}
    if not EPFL_RESULTS.exists():
        return out
    b = inp.raw(EPFL_RESULTS)
    try:
        d = json.loads(b)
    except ValueError as e:
        refuse(f"{EPFL_RESULTS}: not valid JSON ({e})")
    check(isinstance(d, dict), f"{EPFL_RESULTS}: top level is not an object")
    out.update(state="present", sha=sha_bytes(b), bytes=len(b), data=d, known=d.get("schema") == EPFL_RESULTS_SCHEMA,
               md_sha=inp.sha(EPFL_RESULTS_MD) if EPFL_RESULTS_MD.exists() else None)
    if out["known"]:
        hwd = d["disclosures"]["hardware"]
        check(f"{out['dict_from']} -> 1024" in d["disclosures"]["rulsif_v2"],
              f"{EPFL_RESULTS}: RuLSIF v2 dictionary change differs from the addendum")
        check(d["protocol"]["frozen_protocol_sha256"] == PIN["epfl_protocol"]
              and hwd["amendments"]["EPFL_XHW_GATE_AMENDMENT_20261020"] == PIN["epfl_xhw_amendment"]
              and {t: x["release_sha256"] for t, x in d["rulsif_v2"].items()} == {"1024": PIN["epfl_rv2_T1024"], "2048": PIN["epfl_rv2_T2048"]},
              f"{EPFL_RESULTS}: protocol / amendment / RuLSIF v2 release hashes differ from the records")
    return out


# ---------------------------------------------------------------------------------------------------------------- collect
REVIEW_QUOTES = {
    "comir_host": "DEV32 为 0.741 / 0.875 / 0.778，早先同配方的宿主 CoMIR（seed 0–2）为 0.912 / 0.891 / 0.897。",
    "comir_cause": "审查没有发现与 V9 `cmd_comir` 的代码差异，原因未查明。",
    "sacct": "sacct 不可用（slurmdbd 拒绝连接），`scontrol` 在作业结束 300 秒后就查不到，所以已结束单元的真实 allocation 已经无法恢复。",
    "repl_start": "复现的 `run.sbatch:18-19` 用批处理 shell 启动时的 `date +%s` 计时；BF1 和补充方法的 runner 从 Slurm StartTime（`squeue -o %S`）计时。",
    "mind_floor": "方差下限是绝对值 1e-6；作者公开的 MIND-SSC 代码是夹在均值的 0.001–1000 倍之间。",
    "mind_sigma": "patch 加权 σ=0.8 的来源没有说明。",
    "infonce_j": "InfoNCE 的 J 水平是任意的（训练对逐行平移不变），F 值不能当 S 估计来读。",
    "node57": "node57 的时钟比控制器慢约 90 秒，BF1 每个作业因此少记约 1.5 分钟。",
}


def collect(verify_pt: bool) -> dict:
    inp = Inputs()
    dec_text = inp.text(REPO / DECISION_REL)
    check(sha_bytes(dec_text.encode()) == PIN["decision"], "decision differs from the pin")
    check(inp.sha(DECISION_EXT) == PIN["decision"], "root copy of the decision differs from v10/decisions")
    dec = decision_sections(dec_text)
    protocol = inp.json(REPO / PROTOCOL_REL)
    check(inp.sha(REPO / PROTOCOL_REL) == PIN["protocol_p2"], "protocol P2 differs from the pin")
    check(inp.sha(CS / "ssl_pilot_v10" / PROTOCOL_REL) == PIN["protocol_p2"], "ssl_pilot_v10 protocol P2 differs")
    check(inp.sha(REPO / T01_TORCH_REL) == PIN["t01_torch"] and inp.sha(REPO / T01_GEOM_REL) == PIN["t01_geometry"],
          "accepted T01 records differ from the pin")
    torch_rec, geom_rec = inp.json(REPO / T01_TORCH_REL), inp.json(REPO / T01_GEOM_REL)
    review_text = inp.text(REVIEW_MD)
    check(sha_bytes(review_text.encode()) == PIN["review"], "review notes differ from the pin")
    for k, q in REVIEW_QUOTES.items():
        check(q in review_text, f"review quote {k} not found verbatim")
    starttimes = inp.text(REVIEW_STARTTIMES)
    roles = load_roles(inp, protocol, dec)
    bf1 = load_bf1(inp, roles, verify_pt)
    supp = load_supplement(inp, roles, verify_pt)
    fams = {f: (bf1["families"].get(f) or supp["families"][f]) for f in ORDER}
    rep = load_replication(inp, roles, bf1, supp, verify_pt)
    primary_units = [u for fr in fams.values() for u in fr["candidates"] + list(fr["finals"].values())]
    rv2 = load_rulsif_v2_bf(inp, roles, verify_pt)
    rv2_units = (rv2.get("candidates", []) + list(rv2.get("finals", {}).values())) if rv2["state"] != "absent" else []
    capacity = load_capacity(inp, roles, verify_pt, fams)
    if capacity["state"] == "table_absent":
        capacity = placeholder_generic(inp, CAPACITY, (CAPACITY_OUT, CAPACITY_HW_OUT))
    cap_units = capacity.get("units", [])
    sensitivity = load_sensitivity(inp, fams)
    if sensitivity["state"] == "results_absent":
        sensitivity = placeholder_generic(inp, SENSITIVITY, depth=2)
    xchk = cross_unit_checks(primary_units + rep["units"] + rv2_units + cap_units)
    rv = fams[FAILED_V1]
    rv_rows = [r for u in rv["candidates"] + list(rv["finals"].values()) for s in u["cal"] for r in u["cal"][s]["rows"]]
    rv_rows += [r for u in rv["finals"].values() for r in u["dev_rows"]]
    check(all(r["F_true"] < 0 for r in rv_rows), "RuLSIF v1: F_true < 0 does not hold on every row")
    jobs = job_records(bf1, supp, rep)
    stc = same_total_cost(fams, jobs)
    R = {"inp": inp, "dec": dec, "protocol": protocol, "roles": roles, "bf1": bf1, "supp": supp, "fams": fams, "rep": rep,
         "xchk": xchk, "jobs": jobs, "stc": stc, "review_text": review_text, "starttimes": starttimes,
         "torch_rec": torch_rec, "geom_rec": geom_rec, "rv_rows": len(rv_rows), "rv2": rv2, "rv2_units": rv2_units,
         "capacity": capacity, "sensitivity": sensitivity, "hwd": load_hardware_disclosure(inp, rv2),
         "epfl": load_epfl(inp),
         "xhw": xhw_diagnostics(inp, torch_rec, geom_rec), "sacct": sacct_probe_snapshot(inp),
         "primary_units": primary_units, "verify_pt": verify_pt}
    pts = sorted(x for u in primary_units + rep["units"] + rv2_units + cap_units for x in u["pts"])
    R["pt_list"] = pts
    check_epfl_disclosure(R)
    R["pt_digest"] = sha_bytes("".join(f"{a}  {b}\n" for a, b in pts).encode())
    snap = REPO / CKPT_CHECK_REL
    R["ckpt_snapshot"] = inp.json(snap) if snap.exists() else None
    return R




# ---------------------------------------------------------------------------------------------------------------- analyze
MKEYS = ("success42", "success9", "median_err")


def seed_mean(rows: list, block: str) -> dict:
    return {k: statistics.fmean(r[block][k] for r in rows) for k in MKEYS}


def analyze(R: dict) -> dict:
    fams, roles, jobs = R["fams"], R["roles"], R["jobs"]
    A = {}
    # ---- main table (same data / fixed steps)
    main = {}
    for f in ORDER:
        rows = [{"seed": s, "job": u["job"], "host": u["host"], "gpu": u["gpu"], "step": u["selected"]["step"],
                 "dev32": u["dev32"], "dev24": u["dev24"], "cal": u["selected"]["metrics"]}
                for s, u in sorted(fams[f]["finals"].items())]
        main[f] = {"rows": rows, "mean32": seed_mean(rows, "dev32"), "mean24": seed_mean(rows, "dev24")}
    A["main"] = main
    # ---- core-paired contrasts
    con = []
    for a, b, grp in CONTRASTS:
        check(a != FAILED_V1 and b != FAILED_V1, "the failed RuLSIF v1 adaptation is not a contrast arm")
        entry = {"A": a, "B": b, "group": grp}
        for name, cores, groups in (("DEV32", roles["DEV"], roles["groups32"]), ("DEV24", roles["DEV24"], roles["groups24"])):
            p = paired(list(fams[a]["finals"].values()), list(fams[b]["finals"].values()), cores, groups, f"{a}|{b}|{name}")
            check(not p["singleton_groups"] or p["group_ci"] == p["core_ci"],
                  "singleton position groups must reproduce the core bootstrap exactly")
            p["seed_range"] = seed_range(fams[a], fams[b], cores, groups, f"{a}|{b}|{name}")
            entry[name] = p
        con.append(entry)
    A["contrasts"] = con
    A["groups"] = {"DEV32": (len(roles["DEV"]), len(roles["groups32"]), max(len(v) for v in roles["groups32"].values())),
                   "DEV24": (len(roles["DEV24"]), len(roles["groups24"]), max(len(v) for v in roles["groups24"].values()))}
    # ---- cost
    cost = {}
    for f in ORDER:
        fj = [j for j in jobs if j["method"] == f]
        cal = [j for j in fj if j["phase"] == "CAL"]
        fin = sorted((j for j in fj if j["phase"] == "FINAL_DEV"), key=lambda j: j["seed"])
        val = [j for j in fj if j["phase"].startswith("VALIDATION")]
        check(len(cal) == 1 and len(fin) == len(fams[f]["finals"]), f"{f}: job set")
        cost[f] = {"cal": cal[0], "finals": fin, "validation": val,
                   "cal_s": cal[0]["alloc"], "fin_s": sum(j["alloc"] for j in fin), "val_s": sum(j["alloc"] for j in val),
                   "nodes_cal": sorted({cal[0]["node"]}), "nodes_fin": sorted({j["node"] for j in fin})}
    A["cost"] = cost
    A["gate"] = next(j for j in jobs if j["phase"] == "GATE")
    A["node57"] = sorted(((j["job"], j["method"], j["phase"], j["seed"], -j["start_offset"]) for j in jobs
                          if j["basis"] == "slurm_starttime" and j["start_offset"] is not None and j["start_offset"] < 0),
                         key=lambda x: x[0])
    A["node57_nodes"] = sorted({j["node"] for j in jobs if j["basis"] == "slurm_starttime" and j["start_offset"] is not None
                                and j["start_offset"] < 0})
    A["start_offsets_ok"] = [j["start_offset"] for j in jobs if j["basis"] == "slurm_starttime" and j["start_offset"] is not None
                             and j["start_offset"] >= 0]
    A["end_gap"] = [j["alloc"] - j["ledger_end"] for j in jobs if j["ledger_end"] is not None]
    # node speed: DEV registration seconds per (method, node)
    speed = defaultdict(list)
    for f in ORDER:
        for u in fams[f]["finals"].values():
            speed[(f, u["host"])].append(statistics.median(r["sec"] for r in u["dev_rows"]))
    A["speed"] = {k: (len(v), min(v), max(v)) for k, v in sorted(speed.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1]))}
    # ---- diagnostics
    A["saturation"] = {f: cal_saturation(fams[f]) for f in ORDER if len(fams[f]["candidates"][0]["cal"]) > 1}
    A["cal_vs_dev"] = {f: (statistics.fmean(r["cal"]["success9"] for r in main[f]["rows"]),
                           main[f]["mean32"]["success9"]) for f in ORDER}
    m1 = [("CAL", u) for u in fams["MF_VCS_M1"]["candidates"]] + [("FINAL", u) for u in fams["MF_VCS_M1"]["finals"].values()]
    m1 += [("REPL", u) for u in R["rep"]["units"] if u["method"] == "MF_VCS_M1"]
    A["m1"] = [(kind, u.get("order"), u["seed"], m1_stats(u)) for kind, u in m1]
    A["stall"] = [(f, u["order"], u["candidate"], stall_stats(u)) for f in ("MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1")
                  for u in fams[f]["candidates"]]
    clip = []
    for f in ORDER:
        if fams[f]["candidates"][0].get("train") is None or len(fams[f]["candidates"][0]["train"]["grad"]) == 0:
            continue
        clip.append((f, [(u["order"], clip_fraction(u)) for u in fams[f]["candidates"]],
                     [(s, clip_fraction(u)) for s, u in sorted(fams[f]["finals"].items())]))
    A["clip"] = clip
    rv = fams[FAILED_V1]
    rl = []
    for kind, u in [("CAL", u) for u in rv["candidates"]] + [("FINAL", u) for u in rv["finals"].values()]:
        blocks = [("CAL", u["cal"][s]["rows"]) for s in u["cal"]] + ([("DEV", u["dev_rows"])] if "dev_rows" in u else [])
        for role, rows in blocks:
            rl.append({"kind": kind, "order": u["order"], "seed": u["seed"], "role": role, "n": len(rows),
                       "neg": sum(r["F_true"] < 0 for r in rows), "ge": sum(r["F_hat"] >= r["F_true"] for r in rows),
                       "m": metrics([{"err_full": r["err"]} for r in rows]), "kernel": u["train"]["kernel"],
                       "cand": u["candidate"]})
    A["rulsif"] = rl
    cands = rv["release"]["families"][FAILED_V1]["candidates"]
    corner = min(cands, key=lambda c: (c["bandwidth_multiplier"], -c["ridge"]))
    A["rulsif_selected_is_corner"] = corner == rv["selected_candidate"]
    A["rulsif_grid"] = (sorted({c["bandwidth_multiplier"] for c in cands}), sorted({c["ridge"] for c in cands}))
    A["strata"] = {f: strata([r for u in fams[f]["finals"].values() for r in u["dev_rows"]]) for f in ORDER}
    asym = {}
    for f in ORDER:
        fr = fams[f]
        asym[f] = {"n_cand": len(fr["candidates"]), "n_cal_ckpt": len(fr["candidates"][0]["cal_steps"]),
                   "n_saved": len([a for a in fr["candidates"][0]["complete"]["artifacts"] if a.endswith(".pt")]),
                   "tuning_sel": fr["selected_choice"], "cal_s": cost[f]["cal_s"]}
    A["asym"] = asym
    # hard cores: per-core success9 < 0.5 across primary final units except the failed adaptation
    hard_units = [u for f in ORDER if f != FAILED_V1 for u in fams[f]["finals"].values()]
    per = {c: 0 for c in roles["DEV"]}
    per_bf1 = {c: 0 for c in roles["DEV"]}
    s9mean = {c: [] for c in roles["DEV"]}
    for u in hard_units:
        for c, m in u["complete"]["per_core"].items():
            per[c] += m["success9"] < 0.5
            per_bf1[c] += (m["success9"] < 0.5) and u["schema"] == "bf1"
            s9mean[c].append(m["success9"])
    n_bf1 = sum(1 for u in hard_units if u["schema"] == "bf1")
    A["hard"] = {"n_units": len(hard_units), "n_bf1": n_bf1,
                 "cores": sorted(((c, n, per_bf1[c], statistics.fmean(s9mean[c])) for c, n in per.items()
                                  if 2 * n > len(hard_units)), key=lambda x: (-x[1], x[0])),
                 "bf1_10plus": sorted(c for c, n in per_bf1.items() if n >= 10)}
    slide = {}
    for f in ORDER:
        out = {}
        for sl in ("1B", "2B"):
            vals = []
            for u in fams[f]["finals"].values():
                rows = [r for r in u["dev_rows"] if r["core"].startswith(sl + "_")]
                vals.append((sum(r["err"] < 42 for r in rows) / len(rows), sum(r["err"] < 9 for r in rows) / len(rows), len(rows)))
            out[sl] = (statistics.fmean(v[0] for v in vals), statistics.fmean(v[1] for v in vals), vals[0][2] // 10)
        slide[f] = out
    A["slide"] = slide
    # ---- replication
    rep = sorted(R["rep"]["units"], key=lambda u: (ORDER.index(u["method"]), u["seed"]))
    A["rep_units"] = rep
    spread = {}
    for f in R["rep"]["protocol"]["methods"]:
        prim = main[f]["rows"]
        spread[f] = {"primary": [(r["seed"], r["dev32"]) for r in prim],
                     "repl": [(u["seed"], u["gpu"], u["dev32"]) for u in rep if u["method"] == f]}
    A["rep_spread"] = dict(sorted(spread.items(), key=lambda kv: ORDER.index(kv[0])))
    pairs = []
    others = [f for f in ORDER if f in R["rep"]["protocol"]["methods"] and f != REFERENCE]
    for f in others:
        bygpu = defaultdict(list)
        for s in REPL_SEEDS:
            ua = next(u for u in rep if u["method"] == REFERENCE and u["seed"] == s)
            ub = next(u for u in rep if u["method"] == f and u["seed"] == s)
            if ua["gpu"] == ub["gpu"]:
                bygpu[ua["gpu"]].append((s, ua, ub))
        for g, lst in sorted(bygpu.items()):
            p = paired([x[1] for x in lst], [x[2] for x in lst], roles["DEV"], roles["groups32"], f"repl|{REFERENCE}|{f}|{g}")
            per_seed = []
            for s, ua, ub in lst:
                q = paired([ua], [ub], roles["DEV"], roles["groups32"], f"repl|{REFERENCE}|{f}|{g}|{s}")
                per_seed.append((s, q["point"]))
            pairs.append({"B": f, "gpu": g, "seeds": [x[0] for x in lst], "res": p, "per_seed": per_seed})
        if not bygpu:
            pairs.append({"B": f, "gpu": None, "seeds": [], "res": None, "per_seed": []})
    A["rep_pairs"] = pairs
    rj = [j for j in jobs if j["phase"] == "REPL"]
    bg = defaultdict(list)
    for j in rj:
        bg[j["gpu"]].append(j)
    A["rep_cost"] = {g: (len(v), sum(j["alloc"] for j in v), sorted({j["node"] for j in v})) for g, v in sorted(bg.items())}
    # Slurm StartTime recorded by the review for one still-running replication unit
    tz = re.search(r"# TZ on login node: (\w+)", R["starttimes"])
    check(tz is not None and tz.group(1) == "CEST", "start-time snapshot time zone")
    st = {}
    for line in R["starttimes"].splitlines():
        parts = line.split("|")
        if len(parts) == 7 and parts[0].isdigit():
            st[parts[0]] = dt.datetime.fromisoformat(parts[3]).replace(tzinfo=dt.timezone(dt.timedelta(hours=2))).timestamp()
    A["rep_starttime"] = [(j["job"], st[j["job"]], j["start"], j["start"] - st[j["job"]]) for j in rj if j["job"] in st]
    return A


# ----------------------------------------------------------------------------------------------------------------- render
def short(h: str, n=8) -> str:
    return h[:n]


def mtrip(m: dict) -> list:
    return [f4(m["success42"]), f4(m["success9"]), f3(m["median_err"])]


def cand_id(fr: dict, order: int) -> str:
    return f"c{order}"


def method_note(f: str, A: dict) -> str:
    fin = [s["cap_frac"] for k, o, sd, s in A["m1"] if k == "FINAL"]
    return {"MF_VCS_M1": f"{fx(min(fin), 3)}–{fx(max(fin), 3)} 的批次达到 cap 100（近似头，§6.2）",
            "InfoNCE_host[J_perm1]": "InfoNCE 训练的 critic + 共同 J 读出（决定 §4）",
            "MI_native": "无训练；单一确定性单元（seed 标记 1001）",
            "NMI_native": "无训练；单一确定性单元（seed 标记 1001）",
            "MIND_2D_adapted": "无训练；二维适配，非 MIND-SSC 复现；单一单元",
            "CoMIR_native": "只在最终 epoch（第 736 步）读 CAL（§6.7）",
            FAILED_V1: "**失败的适配**（§6.5），不作核精确矩基线"}.get(f, "")


def render(R: dict, A: dict) -> str:
    dec, roles, fams, inp = R["dec"], R["roles"], R["fams"], R["inp"]
    q = lambda sec, text: quote(dec, sec, text)
    L = []
    w = L.append
    rv2s = R["rv2"]["state"]
    ep = R["epfl"]
    q5 = q("5", "可以分别记录进度、失败和实际成本；必需基线补齐之前不发布方法优劣、强基线优势或新颖性比较结论。")
    w("# V10 第二阶段结果分析（草稿）：BF 开发集与 EPFL（分开报告）")
    w("")
    w("**BF：BF1 四族、七个补充方法、RuLSIF_exact_v2、K8/K32 容量扩展、看过 DEV 之后的敏感性与次级种子复现；EPFL：单独一节 · "
      "输入为服务器 handoff 记录（摘要见 §13） · 报告 2026-10-19，2026-10-20 增补 §8–§10**")
    w("")
    rv2txt = {"complete": "已在 BF 上完成，但在标明的赛后修订 A1 下运行（原预登记 level-B 判据未通过；§8.1），它是否满足决定 §5 "
                          "的必需基线由负责人决定", "partial": "在 BF 上部分完成（§8.1）"}.get(rv2s, "尚未完成")
    if ep["state"] != "present":
        eptxt = "EPFL 的官方冻结协议分析尚未写出（§10，pending）"
    elif not ep["known"]:
        eptxt = f"EPFL 的官方分析 `{EPFL_RESULTS.name}`（sha256 `{short(ep['sha'], 12)}`）的 schema 未被本生成器识别，数字未引用（§10）"
    else:
        eptxt = (f"EPFL（官方分析 `{EPFL_RESULTS.name}`，sha256 `{short(ep['sha'], 12)}`，§10）中 RuLSIF_exact_v2 的状态："
                 + "，".join(f"T{t} {x['status']}" for t, x in ep["data"]["rulsif_v2"].items()))
    w(f"> **读法。** 本文是开发结果的描述性汇总，不是方法优劣、强基线优势或新颖性结论。决定 §5 原文：“{q5}”"
      f"必需基线集合仍未确认补齐：RuLSIF_exact（v1）是失败的适配（§6.5）；RuLSIF_exact_v2 {rv2txt}；{eptxt}。"
      "所有区间只作描述，不做显著性检验。")
    w("")
    nfiles = len(inp.hashes)
    w(f"本文全部数字由 `{SELF_REL}` 从服务器记录计算（§13 列出各 bundle 的输入摘要）；没有 wall-clock 时间或实时查询，同一输入下"
      f"两次运行逐字节相同，`--check` 复核。生成时读取并逐个记录 sha256 的输入文件共 {nfiles} 个。手写部分只有文字、标题、对比清单、"
      "预算网格、审查项对照与第 12 节的待决事项。BF 与 EPFL 分开报告，从不合并。")
    w("")
    # ---------------------------------------------------------------- 1
    w("## 1 数据角色、协议与边界")
    w("")
    pdata = R["bf1"]["protocol"]["data"]
    check(pdata["TEST134"] == "HOLD" and pdata["EPFL"] == "NOT_ACTIVATED" and pdata["CONFIRM"] == 0
          and pdata["roles"] == {"FIT": len(roles["FIT"]), "CAL": len(roles["CAL"]), "DEV": len(roles["DEV"])},
          "BF1 protocol data roles differ from the manifest")
    w(f"- 角色由 manifest（sha256 `{short(PIN['manifest'], 12)}`）重算：FIT {len(roles['FIT'])}、CAL {len(roles['CAL'])}、"
      f"DEV {len(roles['DEV'])}，与 BF1 `DATA_LOCK.json` 的 {roles['lock_files']} 个文件角色一致。BF1 协议记录："
      f"TEST134 `{pdata['TEST134']}`、EPFL `{pdata['EPFL']}`、CONFIRM {pdata['CONFIRM']}（BF1 冻结时的记录）；全部 BF 单元"
      "（含 RuLSIF_exact_v2 与容量扩展）的数据读取记录只含 FIT/CAL/DEV 的 manifest core（§2.3），没有读取 TEST 或 EPFL。"
      "EPFL 之后按其冻结协议单独运行；本文不读 EPFL 原始结果，EPFL 的数字只取自官方分析（§10）。"
      "本文不读任何 TEST 数据。")
    w("- 决定 §6.1 原文：")
    for t in ("当前维持 **FIT32、CAL8、DEV32**。DEV32 是开发主表。",
              "同时预登记一个保守敏感性分析：按 record_id 的下划线后部分作为 TMA 位置，排除与当前 FIT/CAL 同位的评估 core。跨 1B/2B 同位是否同一 specimen 尚未证实，此分组是敏感性假设。",
              "DEV24 排除的 8 个 core：`1B_F4, 1B_H4, 1B_H9, 2B_D5, 2B_G3, 2B_G5, 2B_H2, 2B_I5`。",
              "统计以 core 为单位并在方法间配对；每个 core 的多个扰动不当作独立组织样本。另报告按 TMA 位置组重采样的保守敏感性。"):
        w(f"  > {q('6.1', t)}")
    w("")
    w(f"  由 manifest 重算：FIT/CAL 共占 {len(roles['fitcal_positions'])} 个 TMA 位置；DEV 中与之同位的 "
      f"{len(roles['DEV24_excluded'])} 个 core（{', '.join(roles['DEV24_excluded'])}）与决定所列逐字相同，DEV24 保留 "
      f"{len(roles['DEV24'])} 个，也与协议 P2 的 `tma_colocation_sensitivity` 一致。")
    w("- 决定 §8.2 原文（预算与两种视图）：")
    for t in ("每族调参上限为**最多 9 个候选、最多 8 个 L40S 已分配 GPU 小时**，计入训练、checkpoint、CAL 配准及作业内开销；两者先到为准。排队时间另报。",
              "相同步数主表称为 same-data / fixed-steps。same-total-cost 视图使用实际累计成本和预算内已产生的 checkpoint，并只按 CAL 选取；不能用超预算训练完成后的最佳 checkpoint 回填较早预算。预算上限相同也不等于实际花费相同。"):
        w(f"  > {q('8.2', t)}")
    w("")
    w("- 决定 §4 原文（InfoNCE 表注）：")
    w(f"  > {q('4', '表注必须写明：这是“InfoNCE 训练的 critic + 共同 J 读出”，与原包中“原生目标 / 候选归一化”的表述不同。')}")
    w("")
    w("- 决定 §5 原文：")
    w(f"  > {q('5', '完整方法集合和方法内选择全部固定以后才能启用 TEST 重复块。')}")
    w("")
    w("  本文不读 TEST；第 7 节的次级复现及其 checkpoint 选择也属于将来全局冻结必须覆盖的内容。")
    w("")
    # ---------------------------------------------------------------- 2
    w("## 2 完整性与来源")
    w("")
    w("### 2.1 逐方法")
    w("")
    rows = []
    for f in ORDER:
        fr = fams[f]
        c = A["cost"][f]
        src = ("BF1 冻结 `FROZEN` " + short(PIN["bf1_frozen"]) if fr["source"] == "BF1"
               else f"{'pipeline' if fr['bundle'] == PIPE else 'native recovery'} families/{fr['index']} · release "
                    + short(fr["release_sha"]))
        code = fr["code"]
        ident = f"`{code['commit'][:7]}` · core `{code['core']}`" + (f" · host `{code['host']}`" if fr["source"] == "BF1" else "")
        jobsx = f"CAL {c['cal']['job']}；最终 " + "/".join(j["job"] for j in c["finals"])
        nodes = f"CAL {','.join(c['nodes_cal'])}；最终 {','.join(c['nodes_fin'])}"
        ch = fr["selected_choice"]
        selx = (f"{cand_id(fr, ch['order'])}：{cand_label(fr['selected_candidate'])}（调参 ckpt {ch['step']}；CAL "
                f"{f4(ch['metrics']['success42'])}/{f4(ch['metrics']['success9'])}/{f3(ch['metrics']['median_err'])}）")
        seeds = " · ".join(f"{s}@{u['selected']['step']}" for s, u in sorted(fr["finals"].items()))
        planned = (fr["selection"]["planned_candidates"] if fr["source"] == "BF1"
                   else len(fr["release"]["families"][f]["candidates"]))
        rows.append([f + ("（v1）" if f == FAILED_V1 else ""), src, f"{len(fr['candidates'])}/{planned}",
                     len(fr["finals"]), jobsx, ident, f"L40S；{nodes}", selx, seeds])
    rv2 = R["rv2"]
    if rv2["state"] in ("complete", "partial"):
        ch = rv2["selected_choice"]
        nodes_f = ",".join(sorted({f"{u['host']}({gpu_short(u['gpu'])})" for u in rv2["finals"].values()}))
        rows.append([RV2_FAMILY + "（A1）", f"CAL release {short(PIN['rv2_release'])}；最终 release_xhw {short(PIN['rv2_release_xhw'])}",
                     f"{len(rv2['candidates'])}/{len(rv2['release']['families'][RV2_FAMILY]['candidates'])}", len(rv2["finals"]),
                     f"CAL {rv2['cal_job']}；最终 " + "/".join(u["job"] for _, u in sorted(rv2["finals"].items())),
                     f"CAL `{rv2['release']['source_commit'][:7]}` · 最终 `{rv2['xhw']['source_commit'][:7]}`",
                     f"CAL L40S {','.join(sorted({u['host'] for u in rv2['candidates']}))}；最终 {nodes_f}",
                     f"c{ch['order']}：{cand_label(rv2['selected_candidate'])}（CAL {f4(ch['metrics']['success42'])}/"
                     f"{f4(ch['metrics']['success9'])}/{f3(ch['metrics']['median_err'])}）",
                     " · ".join(f"{s}@{u['selected']['step']}" for s, u in sorted(rv2["finals"].items()))])
    for line in table(["方法", "记录", "CAL 候选 完成/计划", "最终单元", "作业（exit 0）", "代码身份", "GPU；节点",
                       "选中配置（调参 seed 1017）", "最终 seed@选中 checkpoint"], rows):
        w(line)
    w("")
    w("CAL 读数格式为 success<42 / success<9 / 中位误差（px），调参 seed 1017 的 8 core × 10 扰动。选中配置按决定 §8.1 的字典序"
      "（<42 高、<9 高、中位误差低、checkpoint 早、候选顺序）在 CAL 上重算后与记录一致。")
    w("")
    w("### 2.2 候选编号")
    w("")
    rows = []
    for f in ORDER:
        fr = fams[f]
        rows.append([f, "；".join(f"c{u['order']} {cand_label(u['candidate'])}" for u in fr["candidates"])])
    if rv2["state"] in ("complete", "partial"):
        rows.append([RV2_FAMILY + "（A1 合格候选；候选 8 排除）", "；".join(f"c{u['order']} {cand_label(u['candidate'])}" for u in rv2["candidates"])])
    for line in table(["方法", "冻结候选（顺序即决定 §8.1 的候选顺序）"], rows):
        w(line)
    w("")
    w("### 2.3 记录校验（任一不符即拒绝生成）")
    w("")
    prim = R["primary_units"]
    ncal = sum(len(u["cal"]) for u in prim + R["rep"]["units"])
    ndev = sum(1 for u in prim + R["rep"]["units"] if "dev_rows" in u)
    nrows = sum(len(u["cal"][s]["rows"]) for u in prim + R["rep"]["units"] for s in u["cal"]) + sum(
        len(u["dev_rows"]) for u in prim + R["rep"]["units"] if "dev_rows" in u)
    nart = sum(u["n_artifacts"] for u in prim + R["rep"]["units"])
    npt = sum(1 for u in prim + R["rep"]["units"] for a in u["complete"]["artifacts"] if a.endswith(".pt"))
    nbf = sum(1 for u in prim if u["schema"] == "bf1")
    w(f"- 单元：主体 {len(prim)} 个（BF1 {nbf}、补充方法 {len(prim) - nbf}）与次级复现 {len(R['rep']['units'])} 个，全部 "
      "`COMPLETE`；COMPLETE 中的 identity、`IDENTITY.json` 与由冻结记录推出的期望 identity（族、seed、候选、调度、"
      "freeze / release、data lock、选择依赖、复现字段）逐字段相同。")
    snap = R["ckpt_snapshot"]
    nall = len(R["pt_list"])
    if snap is None:
        ptxt = f"{npt} 个 checkpoint（.pt）的内容 hash 未重算（`--verify-checkpoints` 未运行）"
    elif snap["digest"] == R["pt_digest"] and snap["n"] == nall and snap["all_match"] is True:
        ptxt = (f"BF 全部 {nall} 个 checkpoint（这些单元的 {npt} 个，加上 RuLSIF_exact_v2 与 K8/K32 容量扩展单元的 {nall - npt} 个；"
                f".pt 共 {fx(snap['bytes'] / 1e9, 2)} GB）由 `--verify-checkpoints` 于 {snap['utc']} 逐个重算，"
                f"全部与 COMPLETE 一致（快照 `{CKPT_CHECK_REL}`，覆盖集合与本次记录相同）")
    else:
        ptxt = f"checkpoint 重算快照 `{CKPT_CHECK_REL}` 与当前记录的 checkpoint 集合不同（过期），需重新运行 `--verify-checkpoints`"
    w(f"- 产物：COMPLETE 共列出 {nart} 个产物，其中 {nart - npt} 个非 checkpoint 产物在本次生成中全部重新计算 sha256 一致；"
      + ptxt + "；另外每个 checkpoint 的 sha256 与其元数据、评估 identity 和每一行配准记录中的 checkpoint 字段逐一相符。"
      "单元目录中没有 COMPLETE 未列出的文件（除 `RUN.lock`、`COMPLETE.json` 与等于 `DATA_READS.json` 的 `READS_<job>.json`）。")
    w(f"- 评估：{ncal} 个 CAL 读数与 {ndev} 个 DEV 读数，共 {nrows} 行配准；每个读数的行完整且不重复、只来自一个 checkpoint，"
      "重算的成功率与中位误差和 summary、COMPLETE 完全相等；每个单元的 checkpoint 选择与每族的候选选择按字典序重算一致；"
      "DEV32、DEV24 与逐 core 指标重算一致。")
    xc = R["xchk"]
    grid = ", ".join(f"{k}（{v} 行）" for k, v in sorted(xc["n_grid"].items()))
    w(f"- 配对：{xc['truth_keys']} 个（角色, core, 扰动）的扰动真值 (θ, t) 在全部 {len(prim) + len(R['rep']['units'])} "
      f"个单元中逐位相同；搜索网格大小 n_grid 的取值：{grid}。")
    w("- 精度与硬件：每个单元恰有一个 ENV 记录，作业号与 COMPLETE 一致；Torch/CUDA/cuDNN/线程与精度开关等于 BF1 协议记录；"
      f"主体单元全部是 {L40S}（`L40S` 分区）。")
    ntr = sum(1 for u in prim + R["rep"]["units"] if "train" in u)
    w(f"- 无续跑：每个单元只有一个 ENV 记录；有训练记录的 {ntr} 个单元，其 TRAIN_STEPS 恰为 1..N 步且只来自该单元的作业。")
    w(f"- RuLSIF_exact（v1）的失败标签有前提检查：其全部 {R['rv_rows']} 行 CAL/DEV 配准记录中真值处 F_true < 0（§6.5）。")
    extra_units = len(R["rv2_units"]) + len(R["capacity"].get("units", []))
    if extra_units:
        w(f"- 之后完成的 BF 单元（RuLSIF_exact_v2 {len(R['rv2_units'])} 个、K8/K32 容量扩展 {len(R['capacity'].get('units', []))} 个）"
          "按同样规则核对（identity、产物 hash、CAL/DEV 重算、exit receipt、硬件记录），扰动真值与上面的单元逐位一致。EPFL 不在本节范围内（§10 只读官方分析）。")
    w("")
    # ---------------------------------------------------------------- 3
    w("## 3 主表：DEV32（same-data / fixed-steps）与 DEV24 敏感性")
    w("")
    n32 = {u["dev32"]["n"] for u in R["primary_units"] if "dev32" in u}
    n24 = {u["dev24"]["n"] for u in R["primary_units"] if "dev24" in u}
    thr = R["protocol"]["bf_batch1"]["selection"]["thresholds_native_px"]
    check(n32 == {10 * len(roles["DEV"])} and n24 == {10 * len(roles["DEV24"])} and thr == [42, 9], "DEV block sizes / thresholds")
    w("每个方法用自己的 CAL 选出配置；每个最终 seed 再在自己的 CAL 读数上选 checkpoint，DEV 只在选中的 checkpoint 上评估。"
      f"DEV32 每单元 {n32.pop()} 次配准（{len(roles['DEV'])} core × 10 扰动），DEV24 为 {n24.pop()} 次。成功率阈值 {thr[0]} 与 "
      f"{thr[1]} native px（协议 P2），中位误差单位 native px。BF1 与七个补充方法的单元都在 L40S 上运行（节点不同，§5.4）；"
      "RuLSIF_exact_v2 的最终单元按负责人决定在三分区运行（§9），逐 seed 标出 GPU。")
    w("")
    w("### 3.1 逐 seed")
    w("")
    rows = []
    for f in ORDER:
        for r in A["main"][f]["rows"]:
            rows.append([f + ("（v1，失败的适配）" if f == FAILED_V1 else ""), r["seed"], r["job"], r["host"], r["step"],
                         *mtrip(r["dev32"]), *mtrip(r["dev24"])])
        if f == FAILED_V1 and "rv2" in A:
            for r in A["rv2"]["rows"]:
                rows.append([RV2_FAMILY + "（A1）", r["seed"], r["job"], f"{r['host']}（{gpu_short(r['gpu'])}）", r["step"],
                             *mtrip(r["dev32"]), *mtrip(r["dev24"])])
    for line in table(["方法", "seed", "作业", "节点", "选中 ckpt", "DEV32 s<42", "s<9", "中位 px", "DEV24 s<42", "s<9",
                       "中位 px"], rows):
        w(line)
    w("")
    w("### 3.2 seed 均值")
    w("")
    rows = []
    for f in ORDER:
        m = A["main"][f]
        rows.append([f, len(m["rows"]), *mtrip(m["mean32"]), *mtrip(m["mean24"]), method_note(f, A)])
        if f == FAILED_V1 and "rv2" in A:
            a = A["rv2"]
            rows.append([RV2_FAMILY + "（A1）合并", len(a["rows"]), *mtrip(a["mean32"]), *mtrip(a["mean24"]),
                         "赛后修订 A1（level-A 合格，候选 8 排除）；硬件混合：" +
                         "、".join(f"{r['seed']} {gpu_short(r['gpu'])}" for r in a["rows"]) + "（§8.1）"])
            for g, st in a["strata"].items():
                rows.append([f"{RV2_FAMILY}（A1）{gpu_short(g)} 层", len(st["seeds"]), *mtrip(st["mean32"]), *mtrip(st["mean24"]),
                             "seed " + ", ".join(map(str, st["seeds"])) + "；不声称跨硬件等价"])
    for line in table(["方法", "单元", "DEV32 s<42", "s<9", "中位 px", "DEV24 s<42", "s<9", "中位 px", "说明"], rows):
        w(line)
    w("")
    w("中位误差的 seed 均值是各 seed 中位误差的算术平均。单一单元方法的“均值”就是该单元。复现的 seeds 1004/1005 不进入本表"
      "（第 7 节）。")
    w("")
    # ---------------------------------------------------------------- 4
    g32, g24 = A["groups"]["DEV32"], A["groups"]["DEV24"]
    w("## 4 以 core 为单位的配对差（描述性）")
    w("")
    w("- 单位是 core。每个单元在每个 core 上汇总 10 个扰动：成功个数（<42、<9 px）与 10 个误差的中位数。方法的 core 得分是"
      "该方法全部主体单元（seeds 1001–1003；MI/NMI/MIND 为单一单元）的平均；两方法在相同的 core 与扰动上配对。")
    w("- 成功率差 = core 差值的平均（等于两方法 seed 平均成功率之差）；误差差 = core 差值的中位数（px）。差值一律为 A − B。")
    w(f"- 区间：core cluster bootstrap，{N_BOOT} 次、百分位 95%；TMA 位置组 bootstrap 按 record_id 下划线后的位置重采样。"
      f"DEV32 的 {g32[0]} 个 core 分属 {g32[1]} 个位置组、DEV24 的 {g24[0]} 个分属 {g24[1]} 个，每组最多 {max(g32[2], g24[2])} 个 core，"
      "所以两种重采样在同一随机流下给出逐位相同的区间（生成器检查）；对 DEV 而言，保守敏感性实际体现在 DEV24 的排除上。")
    w("- “按 seed”：两方法都有 1001–1003 时按相同 seed 配对的 3 个差值的范围；单一单元方法与对方每个 seed 比较。"
      "core 计数是 core 差值为正 / 负 / 零的个数。")
    scope = R["rep"]["protocol"]["comparison_scope"][0]
    check(scope.startswith(REFERENCE + " versus"), "replication comparison scope")
    w(f"- 参照组 A={REFERENCE} 沿用复现协议的比较范围（原文：“{scope}”），另加 CoMIR/MI/NMI/MIND；"
      "机制组每行只改变一个因素（矩估计、critic 类、负参考个数、训练目标）。RuLSIF_exact（v1）与 RuLSIF_exact_v2（A1，硬件混合）"
      "都不进入配对。"
      "这些区间不是检验，不能读作优劣；必需基线集合尚未补齐。")
    w("")
    for mk, title, fm in (("s42", "success<42（比例差）", s4), ("s9", "success<9（比例差）", s4), ("med", "中位误差（px 差）", s3)):
        w(f"### 4.{('s42', 's9', 'med').index(mk) + 1} {title}")
        w("")
        rows = []
        for c in A["contrasts"]:
            cells = [f"{c['A']} − {c['B']}" + ("" if c["group"] == "ref" else "（机制）")]
            for name in ("DEV32", "DEV24"):
                p = c[name]
                lo, hi = p["core_ci"][mk]
                glo, ghi = p["group_ci"][mk]
                sr = p["seed_range"][mk]
                cells += [fm(p["point"][mk]), f"[{fm(lo)}, {fm(hi)}]", f"[{fm(glo)}, {fm(ghi)}]", f"{fm(sr[0])} … {fm(sr[1])}"]
                if mk != "med":
                    sg = p["sign"]["k42" if mk == "s42" else "k9"]
                    cells.append(f"{sg[0]}/{sg[1]}/{sg[2]}")
            rows.append(cells)
        head = ["A − B"]
        for name in ("DEV32", "DEV24"):
            head += [f"{name} 差值", "core 95%", "位置组 95%", "按 seed"] + (["core +/−/0"] if mk != "med" else [])
        for line in table(head, rows):
            w(line)
        w("")
    # ---------------------------------------------------------------- 5
    render_cost(R, A, w, q)
    render_diagnostics(R, A, w, q)
    render_replication(R, A, w, q)
    render_completed(R, A, w, q)
    render_hardware(R, A, w)
    render_epfl(R, A, w)
    render_review(R, A, w)
    render_limits(R, A, w)
    render_repro(R, A, w)
    return "\n".join(L).rstrip("\n") + "\n"


def render_cost(R, A, w, q):
    fams, cost, stc = R["fams"], A["cost"], R["stc"]
    bud = R["bf1"]["protocol"]["budget"]
    capn, caph = bud["cal_max_candidates_per_family"], bud["cal_allocated_L40S_gpu_hours_per_family"]
    w("## 5 成本：已分配 GPU 小时、硬件、排队与 same-total-cost 视图")
    w("")
    w("### 5.1 实际已分配 GPU-h（exit receipt）")
    w("")
    rows, tc, tf, tv = [], 0.0, 0.0, 0.0
    for f in ORDER:
        c = cost[f]
        tc, tf, tv = tc + c["cal_s"], tf + c["fin_s"], tv + c["val_s"]
        fin = " / ".join(gh(j["alloc"]) for j in c["finals"])
        nbad = sum(j["exit_code"] != 0 for j in c["validation"])
        val = gh(c["val_s"]) + f"（{len(c['validation'])} 个作业" + (f"，其中 exit≠0 {nbad} 个）" if nbad else "）")
        rows.append([f, f"{len(fams[f]['candidates'])}/{capn}", c["cal"]["job"], gh(c["cal_s"]), fin, gh(c["fin_s"]),
                     gh(c["cal_s"] + c["fin_s"]), val if c["validation"] else "—"])
    rows.append(["合计", "", "", gh(tc), "", gh(tf), gh(tc + tf), gh(tv)])
    for line in table(["方法", "CAL 候选/上限", "CAL 作业", "CAL GPU-h", "最终 GPU-h（各 seed）", "最终合计", "CAL+最终",
                       "验收作业 GPU-h（不计入比较）"], rows):
        w(line)
    w("")
    bynode = defaultdict(lambda: [0, 0.0, 0.0, set()])
    for j in R["jobs"]:
        if j["phase"] in ("CAL", "FINAL_DEV"):
            b = bynode[(j["gpu"], j["node"])]
            b[0] += 1
            b[1 if j["phase"] == "CAL" else 2] += j["alloc"]
            b[3].add(j["method"])
    w("按节点（主体单元，全部为 L40S）：")
    w("")
    rows = [[gpu_short(g), n, c, gh(a), gh(b), gh(a + b), len(m)] for (g, n), (c, a, b, m) in sorted(bynode.items())]
    for line in table(["GPU", "节点", "作业", "CAL GPU-h", "最终 GPU-h", "合计", "涉及方法数"], rows):
        w(line)
    w("")
    over = [f for f in ORDER if cost[f]["cal_s"] >= caph * 3600 - 30]
    w(f"- 每族 CAL 都是一个作业、按冻结顺序跑完全部候选；实际 CAL 成本最高的是 "
      f"{max(ORDER, key=lambda f: cost[f]['cal_s'])}（{gh(max(cost[f]['cal_s'] for f in ORDER))} GPU-h），"
      + (f"没有任何一族达到 {caph} L40S GPU-h 上限，也没有候选被截断（各族 `budget_exhausted=false`）。" if not over else
         "达到上限的族：" + ", ".join(over) + "。"))
    gate = A["gate"]
    w(f"- BF1 共享 GPU 门禁作业 {gate['job']} 已分配 {fx(gate['alloc'], 1)} s，未计入任一族。补充方法的验收作业（gate 与完整验收）"
      "从批处理脚本启动时刻计时，列在最后一列，不计入科学比较；其中 exit≠0 的是 pipeline 中 MI/CoMIR 的旧 gate，"
      "由 native recovery bundle 的新验收接替。其他 bundle 的更早验收或诊断作业不在本文输入范围内。")
    w("- 单位：L40S 已分配 GPU 小时 = (exit receipt 的 end − start) / 3600；BF1 与补充方法的 start 取 Slurm StartTime"
      "（作业脚本用 `squeue -o %S` 读取），end 为节点时钟；不含排队。")
    w("")
    w("### 5.2 排队（提交 → Slurm StartTime）")
    w("")
    par8 = {u["job"]: u for u in R["bf1"]["par8"]["units"]}
    rows = []
    for f in ORDER:
        c = cost[f]
        cj = c["cal"]
        dep = ""
        if cj["bundle"] == "BF1":
            cmd = " ".join(par8[cj["job"]]["command"])
            m = re.search(r"--dependency=(\S+)", cmd)
            dep = f"（提交时带 `{m.group(1)}`）" if m else ""
        rows.append([f, f"{fx((cj['start'] - cj['submit']) / 3600, 2)}{dep}",
                     " / ".join(f"{fx((j['start'] - j['submit']) / 3600, 2)}" for j in c["finals"])])
    for line in table(["方法", "CAL 等待 h", "最终等待 h（各 seed）"], rows):
        w(line)
    w("")
    w("等待时间 = 账本记录的提交时刻到 exit receipt 的 Slurm StartTime；它混合了依赖等待、调度策略（V9 优先、共享 normal 8 卡"
      "与 pending 上限）和资源等待，没有 sacct 无法分解。")
    w("")
    w("### 5.3 计费口径与对账限制")
    w("")
    eg = A["end_gap"]
    w(f"- 账本 END 记录与 exit receipt 的差（receipt − END）在 {fx(min(eg), 2)}–{fx(max(eg), 2)} s 之间（作业脚本写 receipt 的收尾）。")
    n57 = A["node57"]
    if n57:
        w(f"- 节点时钟：{', '.join(A['node57_nodes'])} 上的 {len(n57)} 个作业，其账本 START 时间戳早于 Slurm StartTime "
          f"{fx(min(x[4] for x in n57), 0)}–{fx(max(x[4] for x in n57), 0)} s（其余节点晚 {fx(min(A['start_offsets_ok']), 0)}–"
          f"{fx(max(A['start_offsets_ok']), 0)} s，即正常启动时间）。这说明该节点时钟落后，这些作业的已分配秒数按节点时钟结束、"
          f"至少少记上述秒数（合计至少 {fx(sum(x[4] for x in n57) / 3600, 3)} GPU-h），本文不作修正。作业："
          + "，".join(f"{x[0]}（{x[1]} {x[2]}{'' if x[3] is None else ' ' + str(x[3])}，≥{fx(x[4], 0)} s）" for x in n57) + "。")
        w(f"  审查原文：“{REVIEW_QUOTES['node57']}”")
    sc = R["sacct"]
    if sc is None:
        w("- sacct 对账：未探测（`--probe-sacct` 尚未运行）。")
    else:
        ok = sc["returncode"] == 0 and sc["stdout"].strip() != ""
        err = sc["stderr"].strip().splitlines()[-1] if sc["stderr"].strip() else ""
        w(f"- sacct 对账：{sc['utc']} 的只读探测（`{SACCT_PROBE_REL}`）" +
          (f"返回 {len(sc['stdout'].splitlines())} 行，本版本未用于对账。" if ok else
           f"没有返回作业记录（returncode {sc['returncode']}；stderr 末行：`{err}`），所以整段 allocation 的 Slurm 对账仍不可用。"))
    w(f"  审查原文：“{REVIEW_QUOTES['sacct']}”")
    w(f"- 次级复现的计时起点不同：“{REVIEW_QUOTES['repl_start']}” 因此第 7 节的复现成本不含 Slurm 分配到脚本启动之间的开销"
      "（未测量），只与同口径的复现单元比较。")
    w("")
    w("### 5.4 节点速度混杂")
    w("")
    rows = []
    for (f, node), (n, lo, hi) in A["speed"].items():
        rows.append([f, node, n, f"{fx(lo, 3)}" if n == 1 else f"{fx(lo, 3)}–{fx(hi, 3)}"])
    for line in table(["方法", "节点", "最终单元", "DEV 每次配准秒数（单元中位数）"], rows):
        w(line)
    w("")
    w("同一方法、同一 GPU 型号在不同节点上的每次配准耗时不同，所以族间成本差异含节点因素；表中秒数只计配准求解本身"
      "（补充方法另有输入/描述符准备时间，计入其已分配成本）。")
    w("")
    w("### 5.5 same-total-cost 视图（决定 §8.2）")
    w("")
    w("规则（手写）：在调参作业的已分配时钟上，某个 checkpoint 只有在其完整 CAL 读数完成后才算“可得”；给定预算 B，只在 B 内"
      "可得的 checkpoint 中按 CAL 字典序选择，不用更晚完成的 checkpoint 回填。可得时刻：BF1 取 CAL 读数之后写入的 CHECKPOINT "
      "账本事件；补充方法取 CAL 读数之后的第一条训练记录时间（上界），最后一个 CAL 步取 COMPLETE 时间。DEV 只在最终报告的"
      "选择上评估过，所以预算内的 CAL 选择若与最终选择不同，其 DEV 记为“未评估”，从不回填。")
    w("")
    w("#### 5.5.1 调参（seed 1017）在不同累计预算下的 CAL 选择")
    w("")
    head = ["方法", "实际 CAL GPU-h", "最终选择首次可得"] + [f"B={b:g} h" for b in TUNING_BUDGETS_H]
    rows = []
    for f in ORDER:
        s = stc[f]
        cells = [f, gh(s["cal_alloc"]), gh(s["sel_t"])]
        for lv in s["levels"]:
            ch = lv["choice"]
            if ch is None:
                cells.append("—")
            else:
                m = ch["metrics"]
                cells.append(f"c{ch['order']}@{ch['step']} {fx(m['success42'], 3)}/{fx(m['success9'], 3)}/{fx(m['median_err'], 2)}"
                             + ("＝" if lv["is_final"] else ""))
        rows.append(cells)
    for line in table(head, rows):
        w(line)
    w("")
    w("单元格为 `候选@checkpoint CAL s<42/s<9/中位 px`，“＝”表示已与最终报告的选择相同，“—”表示预算内还没有完整 CAL 读数。"
      "候选编号见 §2.2。")
    w("")
    w("#### 5.5.2 报告的 DEV 结果所需的最低累计成本")
    w("")
    rows = []
    for f in ORDER:
        s = stc[f]
        per = " · ".join(f"{sd}: {gh(x['cmin'])}/{gh(x['actual'])}" for sd, x in sorted(s["finals"].items()))
        rows.append([f, f"{gh(s['sel_t'])}/{gh(s['cal_alloc'])}", per, f"{gh(s['minimal'])}/{gh(s['actual'])}"])
    for line in table(["方法", "调参：最低/实际 GPU-h", "最终 seed：最低/实际 GPU-h", "整条流水线：最低/实际 GPU-h"], rows):
        w(line)
    w("")
    w("“最低”= 调参选择首次可得的累计成本 + 每个最终 seed 在其选中 checkpoint 的 CAL 读数完成时的成本，再加该 seed 的整段末尾"
      "（最后一个 CAL checkpoint 之后的 DEV 评估与收尾，取实际值，偏保守）；选中的就是最后一个 CAL checkpoint 时等于实际成本。"
      "预算低于“最低”时，报告的 DEV 结果不是该预算内的 CAL 选择。")
    w("")
    w("#### 5.5.3 相同总预算下可得的已报告 DEV32 结果（seed 均值 s<42 / s<9）")
    w("")
    head = ["方法", "最低 GPU-h"] + [f"B={b:g} h" for b in PIPELINE_BUDGETS_H]
    rows = []
    for f in ORDER:
        s = stc[f]
        m = A["main"][f]["mean32"]
        cells = [f, gh(s["minimal"])]
        for b in PIPELINE_BUDGETS_H:
            cells.append(f"{fx(m['success42'], 4)} / {fx(m['success9'], 4)}" if s["minimal"] <= b * 3600 else "未达（未评估）")
        rows.append(cells)
    for line in table(head, rows):
        w(line)
    w("")
    w("B 是“调参 + 全部最终 seed”的累计已分配 L40S GPU-h（单一单元方法只有一个最终单元）。上限相同不等于实际花费相同；"
      "节点速度与 node57 时钟偏差（§5.3、§5.4）都在这些成本里。")
    w("")


def render_diagnostics(R, A, w, q):
    fams, roles = R["fams"], R["roles"]
    w("## 6 诊断")
    w("")
    # 6.1
    w("### 6.1 CAL 判别力饱和与选择幅度")
    w("")
    rows = []
    for f, s in A["saturation"].items():
        fc = s["full_counts"]
        dec = " / ".join(f"{k} {s['decided'].get(k, 0)}" for k in ("s42", "s9", "median", "step"))
        fm = s["fam_margin"]
        fmt = (f"按 {fm['criterion']}：c{fams[f]['selected_choice']['order']} {f3(fm['selected'])} vs c{fm['runner_order']} "
               f"{f3(fm['runner_up'])}") if fm else "—"
        cal9, dev9 = A["cal_vs_dev"][f]
        rows.append([f, s["n_units"], s["sel_full"], f"{statistics.median(fc):g}（{min(fc)}–{max(fc)}）/ {s['n_ckpt']}", dec, fmt,
                     f"{f4(cal9)} → {f4(dev9)}"])
    for line in table(["方法", "选中 checkpoint（调参候选+最终）", "其中 CAL 两阈值均 80/80", "单元内 80/80 checkpoint 数 中位（范围）",
                       "单元内与次优 checkpoint 的区分条件", "族内选中候选 vs 次优候选（中位误差 px）",
                       "最终 seed：CAL s<9 → DEV32 s<9（均值）"], rows):
        w(line)
    w("")
    bf = [A["saturation"][f] for f in BF1_FAMILIES]
    w(f"- BF1 四族共 {sum(s['n_units'] for s in bf)} 个选中 checkpoint，其中 {sum(s['sel_full'] for s in bf)} 个在 CAL 上两阈值都是 "
      "80/80，选择实际由中位误差（字典序第 3 项）决定。补充学习型方法同理（表中各行）。")
    margins = sorted(m for f in A["saturation"] for m in A["saturation"][f]["median_margins"])
    w(f"- 选中 checkpoint 为两阈值 80/80、且与单元内次优 checkpoint 只差在中位误差的单元共 {len(margins)} 个（全部学习型方法），"
      f"两者中位误差之差的中位数为 {fx(statistics.median(margins), 3)} px（范围 {fx(margins[0], 3)}–{fx(margins[-1], 3)} px）。"
      "最终 seed 的 CAL s<9 接近 1，而 DEV32 的 s<9 低得多（最后一列）：CAL8 在 <9 px 上的区分度有限。这是读数性质，"
      "不是对任何方法的判断。")
    w("- 无训练方法与 CoMIR、RuLSIF 每个候选只有一个 CAL 读数，不在此表。")
    w("")
    # 6.2
    w("### 6.2 M1：cap 内的近似头")
    w("")
    w(f"决定 §2.1 原文：“{q('2.1', '主训练保持 `require_converged=False`，达到上限必须标记为近似解并保留实际 gap。')}”")
    w("")
    rows = []
    for kind, order, seed, s in A["m1"]:
        lab = {"CAL": f"调参 c{order}", "FINAL": f"最终 seed {seed}", "REPL": f"复现 seed {seed}"}[kind]
        stops = "，".join(f"{k} {v}" for k, v in s["stops"].items())
        rows.append([lab, fmt_lr(s["lr"]), s["cap"], f4(s["cap_frac"]), f"{s['iter_med']:g}", f"{fx(s['rel_med'], 4)}", stops])
    for line in table(["M1 单元", "lr", "cap", "达到 cap 的批次比例", "迭代数中位", "gap_final/gap_ref 中位", "停止原因（批次数）"], rows):
        w(line)
    w("")
    full_cap = [f"c{o}" for k, o, sd, s in A["m1"] if k == "CAL" and s["cap_frac"] == 1.0]
    sel_o = fams["MF_VCS_M1"]["selected_choice"]["order"]
    selfrac = [s["cap_frac"] for k, o, sd, s in A["m1"] if (k == "CAL" and o == sel_o) or k in ("FINAL", "REPL")]
    check(all(0 < x < 1 for x in selfrac), "M1 selected-configuration cap fractions not strictly between 0 and 1")
    w(f"- 调参候选中每一批都停在上限的有：{', '.join(full_cap)}；这些候选全程使用近似头（达到 cap、未满足相对 gap 容差）。"
      f"选中的 c{sel_o} 及其最终与复现 seed 有 {fx(min(selfrac), 4)}–{fx(max(selfrac), 4)} 的批次停在 cap。gap 只说明固定特征头"
      "问题的次优程度（决定 §2.2）。")
    w("")
    # 6.3
    w("### 6.3 MF 在 lr=1e-3 的训练停滞")
    w("")
    rows = []
    for f, order, cd, s in A["stall"]:
        rows.append([f, f"c{order}", cand_label(cd), f"{s['g1']:.3g} / {s['g2']:.3g} / {s['g3']:.3g}", f4(s["tiny"]),
                     f4(s["max_s42"])])
    for line in table(["方法", "候选", "配置", "pre-clip 梯度范数中位（1–250 / 251–1000 / 1001–3000 步）", "梯度范数 <1e-6 的步比例",
                       "12 个 CAL checkpoint 的最高 s<42"], rows):
        w(line)
    w("")
    hi = [(f, o, s) for f, o, cd, s in A["stall"] if cd["lr"] == 1e-3]
    lo = [s for f, o, cd, s in A["stall"] if cd["lr"] != 1e-3]
    stalled = [(f, o, s) for f, o, s in hi if s["g3"] < 1e-6]
    other = [(f, o, s) for f, o, s in hi if s["g3"] >= 1e-6]
    check(all(s["max_s42"] < 0.5 for _, _, s in hi) and all(s["max_s42"] == 1.0 for s in lo),
          "the two-usable-learning-rates statement needs every lr=1e-3 MF candidate < 0.5 and every other = 1.0")
    w(f"- lr=1e-3 的 {len(hi)} 个 MF 候选中，" + "、".join(f"{f} c{o}" for f, o, _ in stalled)
      + f" 在 1001–3000 步的梯度范数中位 ≤ {max(s['g3'] for _, _, s in stalled):.3g}（训练停滞）；"
      + "、".join(f"{f} c{o}" for f, o, _ in other) + " 梯度没有消失。这 " + str(len(hi))
      + f" 个候选的 12 个 CAL checkpoint 中最高 s<42 不超过 {fx(max(s['max_s42'] for _, _, s in hi), 4)}，而 lr 1e-4/3e-4 的 "
      f"{len(lo)} 个 MF 候选都达到 1.0000；所以 MF 网格实际只有两档学习率可用。这是冻结网格下的观察，不改变已定选择。")
    w("")
    # 6.4
    w("### 6.4 梯度裁剪（各单元 CONFIG 的 clip_norm 均为 1.0，已检查）")
    w("")
    rows = []
    for f, cands, finals in A["clip"]:
        rows.append([f, " / ".join(f"c{o} {fx(x, 4)}" for o, x in cands), " / ".join(f"{sd} {fx(x, 4)}" for sd, x in finals)])
    for line in table(["方法", "调参候选：触发裁剪的步比例", "最终 seed：触发裁剪的步比例"], rows):
        w(line)
    w("")
    w("触发 = 记录的 pre-clip 总梯度范数 > 1（torch `clip_grad_norm_` 只在此时缩放）。MLP critic、InfoNCE 与 CoMIR 的绝大多数步"
      "都在裁剪，实际步长由裁剪决定；CoMIR 的记录值是两个 encoder 各自裁剪前范数的较大者。")
    w("")
    # 6.5
    w("### 6.5 RuLSIF_exact（v1）：失败的适配")
    w("")
    w(f"决定 §5 表中该基线的要求：“{q('5', '| RuLSIF / 核精确矩 | 实现所声明参考下的精确矩；V9 的 RFF ridge perm1 不充当该基线 |')}”")
    w("")
    rows = []
    for r in A["rulsif"]:
        lab = f"调参 c{r['order']}" if r["kind"] == "CAL" else f"最终 seed {r['seed']}"
        k = r["kernel"]
        rows.append([lab, cand_label(r["cand"]), r["role"], r["n"], r["neg"], r["ge"], *mtrip(r["m"]),
                     f"{fx(k['bandwidth'], 3)}", k["negative_coefficients"]])
    for line in table(["单元", "候选", "角色", "行数", "F_true<0", "F_hat≥F_true", "s<42", "s<9", "中位 px", "核带宽", "负系数个数"], rows):
        w(line)
    w("")
    bw, rg = A["rulsif_grid"]
    w(f"- 全部 {R['rv_rows']} 行中真值处 F_true < 0、F_hat ≥ F_true。审查对此的解读是“也就是说真值处的 J 比 T≡0 还差”。"
      f"选中的候选（{cand_label(fams[FAILED_V1]['selected_candidate'])}）"
      + ("位于网格角（带宽倍数最小、ridge 最大；网格为带宽 ×" + "/".join(f"{x:g}" for x in bw) + "、ridge "
         + "/".join(fmt_lr(x) for x in rg) + "）。" if A["rulsif_selected_is_corner"] else "不在网格角。"))
    w("- 本行只作为失败记录与诊断保留，不作为“核精确矩 / RuLSIF 基线”参与任何配对或成本比较；v2 见 §8.1。")
    w("")
    # 6.6
    w("### 6.6 按真值旋转角 |θ*| 分层的成功率")
    w("")
    w(f"决定 §5 表中 MIND 的要求：“{q('5', '| MIND_2D_adapted | 实施真实的二维适配并披露，不能标成已经直接复现原三维方法 |')}”")
    w("")
    head = ["方法"] + [f"|θ*| {int(lo)}–{int(hi)}°：n / s<42 / s<9" for lo, hi in ROT_STRATA]
    rows = []
    for f in ORDER:
        rows.append([f] + [f"{x['n']} / {fx(x['s42'], 4)} / {fx(x['s9'], 4)}" for x in A["strata"][f]])
    for line in table(head, rows):
        w(line)
    w("")
    ms = A["strata"]["MIND_2D_adapted"]
    w(f"- MIND_2D_adapted 的 s<42 从 {fx(ms[0]['s42'], 4)}（0–10°）到 {fx(ms[-1]['s42'], 4)}（20–30°）；表中 n 为该方法全部主体单元"
      "合计的配准数（学习型方法为 3 个 seed 合计）。审查还指出：“" + REVIEW_QUOTES["mind_floor"] + "”“" + REVIEW_QUOTES["mind_sigma"] + "”")
    w("")
    # 6.7
    w("### 6.7 选模自由度不对称与 CoMIR")
    w("")
    rows = []
    for f in ORDER:
        a = A["asym"][f]
        ts = a["tuning_sel"]["metrics"]
        rows.append([f, a["n_cand"], a["n_cal_ckpt"], a["n_saved"], a["n_cand"] * a["n_cal_ckpt"], a["n_cal_ckpt"],
                     f"{fx(ts['success42'], 4)}/{fx(ts['success9'], 4)}/{fx(ts['median_err'], 3)}",
                     f"{gh(a['cal_s'])} / {R['bf1']['protocol']['budget']['cal_allocated_L40S_gpu_hours_per_family']}"])
    for line in table(["方法", "候选数", "每候选 CAL checkpoint", "每单元保存的 checkpoint", "调参可选组合", "每个最终 seed 可选 checkpoint",
                       "调参选中项 CAL s<42/s<9/中位", "调参 CAL GPU-h / 上限"], rows):
        w(line)
    w("")
    asy = A["asym"]
    w(f"- 学习型方法的调参可在 {asy['MF_VCS_perm1']['n_cand']} 个候选（M1 为 {asy['MF_VCS_M1']['n_cand']} 个）× "
      f"{asy['MF_VCS_perm1']['n_cal_ckpt']} 个 CAL checkpoint 中选择；MI/NMI/MIND 只有单一配置；CoMIR 每个单元保存了 "
      f"{asy['CoMIR_native']['n_saved']} 个 checkpoint，但冻结配方只在最终 epoch 读 CAL，没有方法内 checkpoint 选择。"
      + "这些单一配置方法的调参 CAL 成本为 "
      + "、".join(f"{f} {gh(A['asym'][f]['cal_s'])}" for f in ("MI_native", "NMI_native", "MIND_2D_adapted", "CoMIR_native"))
      + f" GPU-h（上限 {R['bf1']['protocol']['budget']['cal_allocated_L40S_gpu_hours_per_family']}）。本文只披露，不做看过 DEV 之后的补选。")
    w(f"- 决定 §5 表中 CoMIR 的要求：“{q('5', '| CoMIR_native | 接入原生完整训练及 encode-then-register；训练和 latent 生成成本计入 |')}”")
    w(f"- 与早先宿主 CoMIR 的差距（审查记录，非本文计算）：“{REVIEW_QUOTES['comir_host']}”“{REVIEW_QUOTES['comir_cause']}”")
    w("")
    # 6.8
    h = A["hard"]
    w("### 6.8 困难 core 与 slide 分层")
    w("")
    rows = []
    for c, n, nb, m in h["cores"]:
        rows.append([c, c.split("_")[0], "是" if c in roles["DEV24"] else "否（与 FIT/CAL 同位）", f"{n}/{h['n_units']}",
                     f"{nb}/{h['n_bf1']}", f4(m)])
    for line in table(["DEV core", "slide", "属于 DEV24", "per-core s<9 < 0.5 的单元（主体，不含 RuLSIF v1）", "其中 BF1 单元",
                       "全部这些单元的 per-core s<9 均值"], rows):
        w(line)
    w("")
    w(f"- 表中只列超过半数单元 per-core s<9 < 0.5 的 core。BF1 的 {h['n_bf1']} 个最终单元中至少 10 个低于 0.5 的 core 共 "
      f"{len(h['bf1_10plus'])} 个。")
    w("")
    rows = []
    for f in ORDER:
        sl = A["slide"][f]
        rows.append([f, f"{fx(sl['1B'][0], 4)} / {fx(sl['1B'][1], 4)}", f"{fx(sl['2B'][0], 4)} / {fx(sl['2B'][1], 4)}"])
    n1b, n2b = A["slide"][ORDER[0]]["1B"][2], A["slide"][ORDER[0]]["2B"][2]
    for line in table(["方法", f"1B（{n1b} core）s<42 / s<9", f"2B（{n2b} core）s<42 / s<9"], rows):
        w(line)
    w("")
    w("slide 前缀 1B/2B 是 manifest 记录的分层；跨 slide 是否同一组织未证实（决定 §6.1）。")
    w("")
    # 6.9
    w("### 6.9 InfoNCE 的 J 水平")
    w("")
    w(f"审查原文：“{REVIEW_QUOTES['infonce_j']}” 本文只使用几何误差（corner error）比较各方法，F 值只用于 §6.5 的符号检查。")
    w("")


def gpu_short(g: str) -> str:
    return {"NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX PRO 6000", "NVIDIA H100 NVL": "H100 NVL",
            "NVIDIA L40S": "L40S"}.get(g, g)


def render_replication(R, A, w, q):
    rp = R["rep"]["protocol"]
    w("## 7 次级种子复现（seeds 1004/1005；看过原 DEV 之后设计）")
    w("")
    w("### 7.1 设计与边界")
    w("")
    w(f"- 复现协议（`{POOL.name}/PROTOCOL.json`，sha256 `{short(PIN['pool_protocol'], 12)}`）原文：“{rp['design_context']}”")
    w(f"- 冻结的配对单位为 `{rp['statistics']['paired_by']}`；硬件记录要求原文：“{rp['statistics']['hardware_reporting']}”")
    w("- 因此本节是次级证据：不与 L40S 的 1001–1003 主表合并，不算独立确认；方法间只在**同一 GPU 型号、同一 seed** 内配对，"
      "这是对冻结 paired_by 的必要收窄（审查 §2.3）。超参数与 CAL 选择沿用主体的冻结选择，没有新的搜索；每个复现 seed 在自己的 "
      "CAL 上选 checkpoint。")
    w("")
    w("### 7.2 单元")
    w("")
    rows = []
    for u in A["rep_units"]:
        wait = (u["receipt"]["start_unix"] - parse_time(u["submit_utc"])) / 3600
        rows.append([u["key"], u["method"], u["seed"], u["job"], "继承（hw bundle）" if u["inherited"] else "pool",
                     gpu_short(u["gpu"]), u["host"], u["selected"]["step"], *mtrip(u["dev32"]), *mtrip(u["dev24"]),
                     gh(u["receipt"]["allocated_seconds"]), f"{fx(wait, 2)}"])
    for line in table(["单元", "方法", "seed", "作业", "执行", "GPU", "节点", "选中 ckpt", "DEV32 s<42", "s<9", "中位 px",
                       "DEV24 s<42", "s<9", "中位 px", "GPU-h（脚本起计，原生型号）", "提交→脚本起点 h"], rows):
        w(line)
    w("")
    w("### 7.3 每个方法的 seed 离散度（L40S 主体 1001–1003 只作背景）")
    w("")
    rows = []
    for f, sp in A["rep_spread"].items():
        pr = sp["primary"]
        rng = lambda k: f"{fx(min(m[k] for _, m in pr), 4)}–{fx(max(m[k] for _, m in pr), 4)}"
        rows.append([f, rng("success42"), rng("success9"), f"{fx(min(m['median_err'] for _, m in pr), 3)}–{fx(max(m['median_err'] for _, m in pr), 3)}",
                     "；".join(f"{s} {gpu_short(g)}: {fx(m['success42'], 4)}/{fx(m['success9'], 4)}/{fx(m['median_err'], 3)}"
                              for s, g, m in sp["repl"])])
    for line in table(["方法", "L40S 1001–1003 s<42 范围", "s<9 范围", "中位 px 范围", "复现 seed（GPU）：s<42/s<9/中位 px"], rows):
        w(line)
    w("")
    w("五个 seed 不合成一个均值：复现单元与主体单元的 GPU 型号不同，型号与 seed 混在一起无法分开。")
    w("")
    w("### 7.4 同一 GPU 型号、同一 seed 内的配对差（DEV32，A = MF_VCS_M1）")
    w("")
    rows = []
    for p in A["rep_pairs"]:
        if p["res"] is None:
            rows.append([f"{REFERENCE} − {p['B']}", "—", "无同型号同 seed 的配对", "", "", ""])
            continue
        r = p["res"]
        cells = [f"{REFERENCE} − {p['B']}", gpu_short(p["gpu"]), ", ".join(str(s) for s in p["seeds"])]
        for mk, fm in (("s42", s4), ("s9", s4), ("med", s3)):
            lo, hi = r["core_ci"][mk]
            per = "; ".join(f"{s}: {fm(v[mk])}" for s, v in p["per_seed"]) if len(p["per_seed"]) > 1 else ""
            cells.append(f"{fm(r['point'][mk])} [{fm(lo)}, {fm(hi)}]" + (f"（{per}）" if per else ""))
        rows.append(cells)
    for line in table(["A − B", "GPU", "配对 seed", "s<42 差 [core 95%]", "s<9 差 [core 95%]", "中位误差差 px [core 95%]"], rows):
        w(line)
    w("")
    bygpu = defaultdict(list)
    for u in A["rep_units"]:
        bygpu[gpu_short(u["gpu"])].append(f"{u['method']}@{u['seed']}")
    minority = sorted(bygpu.items(), key=lambda kv: len(kv[1]))
    w("定义同 §4（core 单位、10 个扰动汇总、core bootstrap）。复现单元的型号："
      + "；".join(f"{g} {len(v)} 个" + (f"（{', '.join(v)}）" if len(v) <= 3 else "") for g, v in minority)
      + "，所以每个对照最多一到两个同型号 seed。描述性，看过 DEV 之后设计，不作结论。")
    w("")
    w("### 7.5 复现成本与启动开销")
    w("")
    rows = [[gpu_short(g), n, gh(s), ", ".join(nodes)] for g, (n, s, nodes) in A["rep_cost"].items()]
    for line in table(["GPU 型号", "单元", "GPU-h 合计（原生型号，脚本起计）", "节点"], rows):
        w(line)
    w("")
    for job, st, start, d in A["rep_starttime"]:
        w(f"- 作业 {job}：审查快照记录的 Slurm StartTime 为 {utc_str(st)}，exit receipt 的脚本起点为 {utc_str(start)}，"
          f"相差 {fx(d, 0)} s 未计入。其余已结束单元的这段开销无法恢复（未测量）。GPU-h 按实际型号报告，不折算为 L40S。")
    w("")
    w("### 7.6 跨硬件 T01 诊断（xhw_t01，不训练，不替代 L40S 门禁）")
    w("")
    xh = R["xhw"]
    if not xh["present"]:
        w("未完成/未运行。")
        w("")
        return
    rows = []
    for t in xh["tags"]:
        for name in ("verify_torch", "core_tol", "adapter", "geometry"):
            p = t["parts"].get(name)
            if p is None:
                rows.append([t["tag"], name, "缺失", "", "", "", "", ""])
                continue
            cls = ("PASS（必需项全过）" if p["status"] == "PASS" else
                   "未完成：CUDA 设备不可用（cudaErrorDevicesUnavailable），无数值结论" if p["infra"] else "FAIL")
            n = "/".join("—" if x is None else str(x) for x in p["n"])
            same = {True: "计数与诊断失败项与 L40S 相同", False: "与 L40S 不同", None: "—"}[p["same_as_l40s"]]
            rows.append([t["tag"], name, cls, n, "/".join(str(x) for x in p["ref_n"]), same,
                         f"{gpu_short(p['gpu'] or '')} {p['host'] or ''} {p['job'] or ''}",
                         f"{(p['commit'] or '')[:7]} · {p['core']} · pin {p['pin']}"])
    for line in table(["tag", "part", "结果", "检查/必需/通过/失败", "L40S 同 part", "与 L40S 比较", "GPU 节点 作业", "代码 · core · pin"], rows):
        w(line)
    w("")
    verdict = {}
    for t in xh["tags"]:
        g = gpu_short(next((p["gpu"] for p in t["parts"].values() if p.get("gpu")), "") or "")
        allpass = len(t["parts"]) == 4 and all(p["status"] == "PASS" for p in t["parts"].values())
        verdict.setdefault(g, []).append((t["tag"], allpass))
    A["xhw_verdict"] = verdict
    txt = "；".join(f"{g}：" + ("有 tag 四个 part 全部 PASS（" + ", ".join(tg for tg, ok in v if ok) + "）" if any(ok for _, ok in v)
                              else "没有一次运行得到完整数值结论（" + ", ".join(tg for tg, _ in v) + "）") for g, v in sorted(verdict.items()))
    w("这些是同一 T01 CUDA parts 在其他 GPU 型号上的诊断性重跑（只写独立 out-root，不合并进已接受的 T01）。按型号：" + txt +
      "。PASS 只说明该型号上必需数值项通过，不构成跨卡等价或硬件无关的结论。")
    w("")


def render_generic(ph, root, w, title, fams=None, closing=""):
    if ph["state"] == "absent":
        w(f"未完成/未运行（`{root}` 不存在）。")
        w("")
        return
    w(f"{title}（`{root}`" + ("，单元目录在 `" + str(CAPACITY_OUT) + "`" if root == CAPACITY else "") +
      f"）：已完成单元 {len(ph['completes'])} 个，尚未完成的单元目录 {len(ph['running'])} 个。")
    w("")
    if ph["completes"]:
        rows = []
        for c in ph["completes"]:
            ref = (fams or {}).get(c["family"], {}).get("finals", {}).get(c["seed"]) if c["K"] not in (None, 16) else None
            delta = ""
            if ref is not None and c["DEV32"]:
                d = {k: c["DEV32"][k] - ref["dev32"][k] for k in MKEYS}
                delta = f"{s4(d['success42'])}/{s4(d['success9'])}/{s3(d['median_err'])}"
            rows.append([c["unit"], c["status"], c["family"], c["K"], c["seed"], c["step"], c["job"],
                         {None: "缺 receipt", 0: "0"}.get(c["exit_code"], str(c["exit_code"])),
                         "" if c["alloc"] is None else gh(c["alloc"]),
                         "" if not c["DEV32"] else "/".join(mtrip(c["DEV32"])), "" if not c["DEV24"] else "/".join(mtrip(c["DEV24"])),
                         delta])
        for line in table(["单元", "状态", "族", "K", "seed", "选中 ckpt", "作业", "exit", "GPU-h", "DEV32 s<42/s<9/中位",
                           "DEV24 s<42/s<9/中位", "DEV32 差（− 同族同 seed 的 K16 最终单元）"], rows):
            w(line)
        w("")
    rows = [[e["path"], e["schema"] or "", e["status"] or "", short(e["sha"], 12)] for e in ph["files"]]
    if rows:
        for line in table(["JSON 记录", "schema", "status", "sha256"], rows):
            w(line)
        w("")
    else:
        w("根目录中还没有 JSON 记录。")
        w("")
    w(closing)
    w("")


# ---------------------------------------------------------------------------------------- analysis of the completed blocks
def analyze_new(R: dict, A: dict):
    rv2 = R["rv2"]
    if rv2["state"] in ("complete", "partial"):
        rows = []
        for s, u in sorted(rv2["finals"].items()):
            pos = sum(r["F_true"] > 0 for r in u["dev_rows"]) / len(u["dev_rows"])
            ge = sum(r["F_hat"] >= r["F_true"] for r in u["dev_rows"]) / len(u["dev_rows"])
            rows.append({"seed": s, "job": u["job"], "host": u["host"], "gpu": u["gpu"], "step": u["selected"]["step"],
                         "dev32": u["dev32"], "dev24": u["dev24"], "cal": u["selected"]["metrics"], "gate": u["gate"],
                         "alloc": u["receipt"]["allocated_seconds"], "ftrue_pos": pos, "fhat_ge": ge})
        strata_ = {}
        for g in sorted({r["gpu"] for r in rows}):
            sub = [r for r in rows if r["gpu"] == g]
            strata_[g] = {"seeds": [r["seed"] for r in sub], "mean32": seed_mean(sub, "dev32"), "mean24": seed_mean(sub, "dev24")}
        A["rv2"] = {"rows": rows, "mean32": seed_mean(rows, "dev32"), "mean24": seed_mean(rows, "dev24"), "strata": strata_,
                    "cal_alloc": rv2["exits"][rv2["cal_job"]]["allocated_seconds"]}


def render_completed(R, A, w, q):
    w("## 8 已完成的补充结果")
    w("")
    render_rv2(R, A, w, q)
    render_capacity(R, A, w, q)
    render_sensitivity(R, A, w)


def render_rv2(R, A, w, q):
    rv2 = R["rv2"]
    w("### 8.1 RuLSIF_exact_v2（BF；在 A1 修订下运行）")
    w("")
    if rv2["state"] in ("absent", "pending_cal"):
        w("pending（尚无 CAL 选择记录）。")
        w("")
        return
    am, xr = rv2["amendment"], rv2["xhw"]
    w(f"- A1 修订（`{RV2_AMENDMENT_REL}`，sha256 `{short(PIN['rv2_amendment'], 12)}`，在 `{RV2_EXEC.name}` 中已提交）标签原文："
      f"“{am['label']}”")
    w(f"- 其报告规则原文：“{am['reporting_rule']}”")
    w("- 相对 v1 的改动（原文）：" + "；".join(f"{k}. {v}" for k, v in am["changes_from_v1"].items()))
    w(f"- 发行：CAL 用 `{short(PIN['rv2_release'])}`（RELEASE.json），最终单元用跨硬件发行 `{short(PIN['rv2_release_xhw'])}`"
      f"（RELEASE_XHW.json，base 为前者）。跨硬件发行的报告规则原文：“{xr['reporting_rule']}”")
    w("")
    sel = rv2["selected_choice"]
    rows = [[f"c{u['order']}", cand_label(u["candidate"]), *[f"{fx(u['selected']['metrics'][k], 4)}" for k in ("success42", "success9")],
             f"{fx(u['selected']['metrics']['median_err'], 3)}", u["train"]["kernel"].get("negative_coefficients", ""),
             "＝选中" if u["order"] == sel["order"] else ""] for u in rv2["candidates"]]
    w(f"CAL（seed 1017，作业 {rv2['cal_job']}，L40S；A1 合格候选 {len(rv2['candidates'])} 个，候选 8 已排除）：")
    w("")
    for line in table(["候选", "配置", "CAL s<42", "s<9", "中位 px", "负系数个数", ""], rows):
        w(line)
    w("")
    a = A["rv2"]
    rows = []
    for r in a["rows"]:
        rows.append([r["seed"], r["job"], gpu_short(r["gpu"]), r["host"], r["gate"]["job"], r["step"], *mtrip(r["dev32"]),
                     *mtrip(r["dev24"]), f"{fx(r['ftrue_pos'], 4)}", gh(r["alloc"])])
    w("最终单元（每个 seed 一个 GPU 型号；GPU-h 为该型号的原生已分配小时）：")
    w("")
    for line in table(["seed", "作业", "GPU", "节点", "门禁", "选中 ckpt", "DEV32 s<42", "s<9", "中位 px", "DEV24 s<42", "s<9",
                       "中位 px", "DEV F_true>0 比例", "GPU-h"], rows):
        w(line)
    w("")
    rows = [["合并（全部硬件）", ", ".join(str(r["seed"]) for r in a["rows"]), *mtrip(a["mean32"]), *mtrip(a["mean24"])]]
    for g, st in a["strata"].items():
        rows.append([f"{gpu_short(g)} 层", ", ".join(map(str, st["seeds"])), *mtrip(st["mean32"]), *mtrip(st["mean24"])])
    for line in table(["视图", "seed", "DEV32 s<42", "s<9", "中位 px", "DEV24 s<42", "s<9", "中位 px"], rows):
        w(line)
    w("")
    v1 = A["main"][FAILED_V1]["mean32"]
    w(f"- 主表 §3.2 中它与 v1（DEV32 seed 均值 s<42 {fx(v1['success42'], 4)}）并列。合并行混有 GPU 型号；L40S 层只有 "
      f"{len(a['strata'].get(L40S, {}).get('seeds', []))} 个 seed。不声称跨硬件等价。")
    w(f"- 成本：CAL {gh(a['cal_alloc'])} L40S GPU-h；最终单元见上表（按型号，不折算为 L40S）。v2 不进入 §5 的 L40S 成本合计、"
      "same-total-cost 视图和 §4 的配对差（硬件混合，且 A1 是否使 v2 满足决定 §5 的必需基线由负责人决定）。")
    w("")


def render_capacity(R, A, w, q):
    c = R["capacity"]
    w("### 8.2 K8/K32 容量扩展（单独成表，不进入主表）")
    w("")
    w(f"决定 §8.1 原文：“{q('8.1', 'K8/K32 如用于自然图像，另记容量扩展，不暗中并入主表，也不增加没有入账的 27 格。')}”")
    w("")
    if c["state"] != "complete":
        render_generic(c, CAPACITY, w, "容量扩展（尚无完整的硬件修订表）", R["fams"], "pending。")
        return
    pre, hwa = c["prereg"], c["hwa"]
    w(f"- 冻结：base `{short(PIN['cap_base_freeze'])}`（FREEZE.json），硬件修订 `{short(PIN['cap_hw_freeze'])}`"
      f"（hw_amendment_20261019/FREEZE.json；HARDWARE_AMENDMENT.json `{short(PIN['cap_hw_amendment'])}`）；表格由该 bundle 的冻结脚本"
      f"写出（CAPACITY_TABLE_HW.json `{short(PIN['cap_table_hw'])}`），本文逐单元核对后引用其区间。")
    w(f"- 预登记的解释范围原文：“{pre['interpretation_scope']}”")
    w(f"- 硬件修订的报告规则原文：“{hwa['reporting']['pairing_caveat']}” “{hwa['reporting']['strata']}” "
      f"“{hwa['reporting']['interpretation']}”")
    w("")
    rows = []
    for u in sorted(c["units"], key=lambda x: (ORDER.index(x["method"]), x["K"], x["seed"])):
        d = {k: u["dev32"][k] - u["k16"]["dev32"][k] for k in MKEYS}
        rows.append([u["method"], u["K"], u["seed"], gpu_short(u["gpu"]), u["host"], u["selected"]["step"], *mtrip(u["dev32"]),
                     s4(d["success42"]), s4(d["success9"]), s3(d["median_err"]), gh(u["receipt"]["allocated_seconds"])])
    for line in table(["族", "K", "seed", "GPU", "节点", "选中 ckpt", "DEV32 s<42", "s<9", "中位 px", "Δs<42（− K16）", "Δs<9",
                       "Δ中位 px", "GPU-h"], rows):
        w(line)
    w("")
    rows = []
    for cell in c["cells"]:
        for v in cell["views"]:
            name = "合并（全部硬件）" if v["name"] == "pooled" else f"{gpu_short(v['name'])} 层"
            cells = [cell["family"], cell["K"], name, ", ".join(map(str, v["seeds"]))]
            for key in (("DEV32", "success42"), ("DEV32", "success9"), ("DEV24", "success42")):
                x = v[key]
                cells.append(f"{s4(x['mean'])} [{s4(x['ci'][0])}, {s4(x['ci'][1])}] {x['grid']}")
            rows.append(cells)
    for line in table(["族", "K", "视图", "seed", "DEV32 Δs<42 [95%] 格", "DEV32 Δs<9 [95%] 格", "DEV24 Δs<42 [95%] 格"], rows):
        w(line)
    w("")
    multi = [x for cell in c["cells"] for v in cell["views"] if not v["single"] for x in (v[(b, m)] for b in ("DEV32", "DEV24")
                                                                                          for m in ("success42", "success9"))]
    labels = sorted({x["grid"] for x in multi})
    single = sorted({x["grid"] for cell in c["cells"] for v in cell["views"] if v["single"]
                     for x in (v[(b, m)] for b in ("DEV32", "DEV24") for m in ("success42", "success9"))})
    allb = labels == ["B_no_detectable_response"]
    w(f"- 合并视图与 L40S 层的全部 {len(multi)} 个格（DEV32/DEV24 × s<42/s<9）的预登记格标签：{', '.join(labels)}"
      + ("，即所有 K 效应都是 B 格。" if allb else "。") + f" 预登记 B 格原文：“{pre['interpretation_grid']['B_no_detectable_response']}”")
    differ = []
    for cell in c["cells"]:
        views = {v["name"]: v for v in cell["views"]}
        l40 = views.get(L40S)
        if l40 is None or l40["single"]:
            continue
        differ += [f"{cell['family']} K{cell['K']} {b} {m}" for b in ("DEV32", "DEV24") for m in ("success42", "success9")
                   if views["pooled"][(b, m)]["grid"] != l40[(b, m)]["grid"]]
    w("- 合并视图与 L40S 层的格标签" + ("处处相同；按硬件修订规则，没有结果需称为 hardware-sensitive。" if not differ else
                                   f"在 {', '.join(differ)} 上不同，按硬件修订规则这些结果称为 hardware-sensitive。"))
    w(f"- 单 seed 的 RTX PRO 6000 层标为 {', '.join(single)}，只作描述，没有格判断。预登记原文：“{pre['interpretation_grid']['all_outcomes']}”")
    bt = "，".join(f"{gpu_short(g)} {fx(v, 3)}" for g, v in c["bytype"].items())
    w(f"- 成本：共 {fx(c['total_s'] / 3600, 3)} GPU-h（原生型号；上限 {c['ceiling_h']:g}）：{bt}。")
    w("")


def render_sensitivity(R, A, w):
    s = R["sensitivity"]
    w("### 8.3 看过 DEV 之后的敏感性 D1–D3（post-DEV sensitivity）")
    w("")
    if s["state"] != "complete":
        render_generic(s, SENSITIVITY, w, "敏感性诊断（尚无 RESULTS.json）", None, "pending。")
        return
    r = s["r"]
    w(f"标签原文：“{r['label']}” 预登记 `PROTOCOL.json`（sha256 `{short(PIN['sens_protocol'], 12)}`），结果 `RESULTS.json`"
      f"（`{short(PIN['sens_results'], 12)}`）。下列冻结行与本文 §3 的已核对单元逐值相同（生成器检查）。")
    w("")
    d1 = r["D1"]
    rows = []
    for k in ("CAL_1017", "FINAL_1001", "FINAL_1002", "FINAL_1003"):
        u = d1["units"][k]
        cells = [k, u["selected_epoch"], u["selected_step"], u["qc_replay_cal_0736"]]
        if "DEV32_selected" in u:
            cells += ["/".join(mtrip(u["DEV32_selected"])), "/".join(mtrip(u["DEV32_frozen_epoch23"])),
                      "/".join(mtrip(u["DEV24_selected"])), "/".join(mtrip(u["DEV24_frozen_epoch23"]))]
        else:
            cells += ["—"] * 4
        rows.append(cells)
    w("**D1 CoMIR_native 在 CAL 上选 epoch**（冻结配方只在第 23 epoch 读 CAL）：")
    w("")
    for line in table(["单元", "选中 epoch", "选中步", "CAL 重放（736）", "DEV32 选中 s<42/s<9/中位", "DEV32 冻结 e23",
                       "DEV24 选中", "DEV24 冻结 e23"], rows):
        w(line)
    w("")
    p = d1["paired_success42_mean_over_seeds"]
    w(f"- 三 seed 平均的逐 core 配对差（success<42，选中 − 冻结）：{s4(p['mean'])}，95% 区间 [{s4(p['ci_low'])}, {s4(p['ci_high'])}]"
      f"（{p['units']} core，{p['resamples']} 次）。预登记格：`{d1['grid_cell']}`——“{d1['grid_text']}” 差距条款原文：“{d1['gap_clause']}”")
    w("")
    d2 = r["D2"]
    rows = []
    for blk in ("CAL", "DEV32", "DEV24"):
        rows.append([blk, "/".join(mtrip(d2["variant"][blk])), "/".join(mtrip(d2["frozen"][blk]))])
    w("**D2 MIND_2D_warp_then_describe**（命名变体，不替代主表的 MIND_2D_adapted 行）：")
    w("")
    for line in table(["读数", "变体 s<42/s<9/中位", "冻结 s<42/s<9/中位"], rows):
        w(line)
    w("")
    rows = [[k, f"{d2['variant']['DEV32_strata'][k]['n']} / {fx(d2['variant']['DEV32_strata'][k]['success42'], 4)}",
             f"{d2['frozen']['DEV32_strata'][k]['n']} / {fx(d2['frozen']['DEV32_strata'][k]['success42'], 4)}"] for k in ("0-10", "10-20", "20-30")]
    for line in table(["DEV32 |θ*| 层（°）", "变体 n / s<42", "冻结 n / s<42"], rows):
        w(line)
    w("")
    p = d2["paired_success42_variant_minus_frozen"]
    w(f"- 逐 core 配对差（success<42，变体 − 冻结）：{s4(p['mean'])}，95% 区间 [{s4(p['ci_low'])}, {s4(p['ci_high'])}]。"
      f"预登记格：`{d2['grid_cell']}`——“{d2['grid_text']}” 原文：“{d2['always']}”")
    w("")
    rows = [[j["job"], j["family"], j["partition"], gpu_short(j["gpu_name"]), j["host"], j["status"], j["cuda_numerics"],
             str(j["resume_bitwise_equal_this_gpu"])] for j in r["D3"]["jobs"]]
    w("**D3 补充方法 GPU 门禁的跨硬件重跑**：")
    w("")
    for line in table(["作业", "方法", "分区", "GPU", "节点", "结果", "CUDA 数值", "续跑逐位相同"], rows):
        w(line)
    w("")
    for d in s["d3_resume"]:
        if d["diffs"]:
            dd = "；".join(f"{k} 不中断 {a!r}、续跑 {b!r}（差 {b - a:.3g}）" for k, (a, b) in d["diffs"].items())
            w(f"- “续跑逐位相同”列比较门禁记录中续跑后的第二步与不中断的第二步。作业 {d['job']}（{d['family']}，{gpu_short(d['gpu'])}）："
              f"{', '.join(d['same'])} 逐位相同；{dd}；优化器与 RNG 恢复记录为 {d['restored']}，门禁状态 PASS。")
    w("- PASS 只说明实现层面的数值恒等式、续跑和搜索机制在该 GPU 型号上成立，不说明与 L40S 等价。")
    w("- 成本（已分配 GPU-h，按诊断与分区，receipt 逐项核对）：" + "，".join(f"{k} {fx(v, 3)}" for k, v in
                                                               sorted(r["costs"]["allocated_gpu_hours"].items())) + "。")
    w("")


def render_hardware(R, A, w):
    h = R["hwd"]
    cap, am = h["cap_authority"], h["epfl_amendment"]
    w("## 9 硬件与修订披露")
    w("")
    w(f"- 负责人决定（{cap['decision_date']}）原文：「{cap['owner_choice_verbatim'].strip('「」')}」。当时提供的选项原文："
      f"“{cap['option_as_offered_verbatim']}” 此前的常设指示（{cap['standing_instruction_time']}）原文："
      f"“{cap['standing_instruction_verbatim'].strip('「」')}”（记录：容量扩展 HARDWARE_AMENDMENT.json 与 RuLSIF v2 的"
      " RELEASE_XHW.json，两处文字经生成器比对一致）。")
    w(f"- 负责人决定（记录日期 {am['date']}，UTC {am['utc']}）原文：「{am['owner']['words_verbatim']}」。背景原文："
      f"“{am['owner']['context_verbatim']}”")
    fg = am["failed_gate"]
    w(f"- EPFL 跨硬件门禁修订（`{EPFL_HW2_SRC.name}/v10/epfl/EPFL_XHW_GATE_AMENDMENT_20261020.json`，sha256 "
      f"`{short(PIN['epfl_xhw_amendment'], 12)}`）标签原文：“{am['label']}” H100 门禁 {fg['job']}（{fg['node']}，{fg['gpu']}）"
      f"只在以下两行失败：{', '.join(r if isinstance(r, str) else r.get('name', str(r)) for r in fg['failed_rows'])}；"
      "这两行在 RTX6000PRO / H100 上改用 T01 往返容差：" + "；".join(f"{k}：{v}" for k, v in am["rule"]["changed_rows"].items())
      + f"。L40S 规则原文：“{am['reasons']['l40s']}”")
    rows = []
    for g in h["epfl_gates"]:
        for name, row in sorted(g["rows"].items()):
            e, gr = row["energy"], row["gradient"]
            rows.append([gpu_short(g["gpu"]), g["node"], g["job"], g["status"], name.replace("checkpoint.energy_and_gradient_bitwise_", ""),
                         f"{e['max_abs_err']:.3g} / {e['max_rel_err']:.3g} / {e['bitwise']}",
                         f"{gr['max_abs_err']:.3g} / {gr['max_rel_err']:.3g} / {gr['bitwise']}", row["status"], g["state_hash"]])
    w("")
    w("EPFL 修订后门禁（hw2 发行）中这两行的实测差值：")
    w("")
    for line in table(["GPU", "节点", "门禁作业", "门禁", "行", "能量 max abs / max rel / 逐位", "梯度 max abs / max rel / 逐位",
                       "行结果", "state hash 行"], rows):
        w(line)
    w("")
    w("各类作业的实际 GPU（BF 由各单元 ENV 记录统计；EPFL 取自官方分析的逐单元 hardware 字段）：")
    w("")
    for line in table(["类别", "单元", "GPU 型号（单元数）"], hardware_rows(R)):
        w(line)
    w("")
    w("BF 的 CAL（调参与选模）全部在 L40S；最终单元在负责人决定之后按 RTX6000PRO / L40S / H100 三分区调度（BF 主体最终单元早于"
      "决定，全部在 L40S）。各块按硬件分层报告，不声称跨硬件等价。EPFL 的 CAL 在 L40S（官方分析的 cal_allocated_l40s_hours，§10.1）。")
    w("")


def hardware_rows(R) -> list:
    def count(units):
        c = Counter(gpu_short(u["gpu"]) for u in units)
        return "，".join(f"{g} {n}" for g, n in sorted(c.items()))
    fams = R["fams"]
    rows = [["BF 调参 CAL（BF1 四族 + 七个补充方法）", sum(len(fr["candidates"]) for fr in fams.values()),
             count([u for fr in fams.values() for u in fr["candidates"]])],
            ["BF 主体最终单元（seeds 1001–1003）", sum(len(fr["finals"]) for fr in fams.values()),
             count([u for fr in fams.values() for u in fr["finals"].values()])],
            ["BF 次级复现（seeds 1004/1005）", len(R["rep"]["units"]), count(R["rep"]["units"])]]
    if R["rv2"]["state"] in ("complete", "partial"):
        rows += [["BF RuLSIF_exact_v2 CAL", len(R["rv2"]["candidates"]), count(R["rv2"]["candidates"])],
                 ["BF RuLSIF_exact_v2 最终单元", len(R["rv2"]["finals"]), count(list(R["rv2"]["finals"].values()))]]
    if R["capacity"]["state"] == "complete":
        rows.append(["BF K8/K32 容量扩展", len(R["capacity"]["units"]), count(R["capacity"]["units"])])
    ep = R["epfl"]
    if ep["state"] == "present" and ep["known"]:
        fin = [{"gpu": r["hardware"]["gpu_name"]} for f in ep["data"]["families"].values() for r in f["finals"].values() if r]
        fin += [{"gpu": r["hardware"]["gpu_name"]} for x in ep["data"]["rulsif_v2"].values() for r in x["finals"].values() if r]
        rows.append(["EPFL 最终单元（取自官方分析，§10）", len(fin), count(fin)])
    return rows


def json_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def json_short_list(v) -> bool:
    return isinstance(v, list) and len(v) <= 12 and all(json_scalar(x) for x in v)


def json_leaves(node, path=()):
    """Scalar leaves of a JSON value in file order; a short list of scalars counts as one leaf."""
    if json_scalar(node) or json_short_list(node):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from json_leaves(v, path + (str(k),))
    else:
        for n, v in enumerate(node):
            yield from json_leaves(v, path + (f"[{n}]",))


def epfl_cell(m: dict, n=None) -> str:
    return f"{f4(m['success42'])} / {f4(m['success9'])}" + (f"（{n}）" if n is not None else "")


def epfl_final_row(label, seed, r) -> list:
    if r is None:
        return [label, seed, "pending"] + [""] * 12
    st = r["stats"]["DEV72_primary"]
    um, rg = st["unit_macro"], st["registration_level"]
    return [label, seed, r["release"], gpu_short(r["hardware"]["gpu_name"]), r["hardware"]["node"], r["job"], r["selected"]["step"],
            f4(um["success42"]), f4(um["success9"]), f3(um["median_native_px"]), f4(rg["success42"]), f4(rg["success9"]),
            f3(rg["median_native_px"]), f3(r["allocated_hours"]), str(r["runtime_statistics_agree"]).lower()]


EPFL_FINAL_HEAD = ["族", "seed", "发行", "GPU", "节点", "作业", "选中 ckpt", "单元宏 s<42", "s<9", "单元中位 native px",
                   "配准级 s<42", "s<9", "配准中位 native px", "GPU-h", "运行时统计一致"]


def epfl_mean_rows(label, summ, strata) -> list:
    if not summ or not summ.get("seed_mean"):
        return [[label, "pending", "", "", "", "", "", ""]]
    m, bi = summ["seed_mean"]["DEV72_primary"], summ["bootstrap95"]["DEV72_primary"]
    um, rg = m["unit_macro"], m["registration_level"]
    ci = lambda k, f: f"{f(um[k])} [{f(bi[k][0])}, {f(bi[k][1])}]"
    gpus = "，".join(f"{sd} {gpu_short(g)}" for sd, g in strata["hardware_per_seed"].items())
    rows = [[label, len(summ["seeds_complete"]), gpus, ci("success42", f4), ci("success9", f4), ci("median_native_px", f3),
             f4(rg["success42"]), f4(rg["success9"])]]
    if strata["n_L40S"] and strata["n_L40S"] < strata["n_pooled"]:
        lo = strata["L40S_only"]
        rows.append([f"{label}（L40S 层）", strata["n_L40S"], "L40S", f4(lo["success42"]), f4(lo["success9"]), "", "", ""])
    return rows


EPFL_MEAN_HEAD = ["族", "seed 数", "GPU（每 seed）", "单元宏 s<42 [95%]", "s<9 [95%]", "单元中位 native px [95%]", "配准级 s<42", "s<9"]


def render_epfl(R, A, w):
    ep = R["epfl"]
    proto, add = ep["protocol"], ep["addendum"]
    w("## 10 EPFL RGB–NIR（单独报告，从不与 BF 合并）")
    w("")
    w(f"- 来源：本节的 EPFL 数字只取自 EPFL agent 写出的官方冻结协议分析 `{rel_where(EPFL_RESULTS)}`（中文说明 "
      f"`{EPFL_RESULTS_MD.name}`）；本文不从 EPFL 原始结果重算，也不把 EPFL 与 BF 合并。冻结协议"
      f"（`{EPFL_SRC.name}/v10/epfl/EPFL_PROTOCOL_FROZEN.json`，sha256 `{short(PIN['epfl_protocol'], 12)}`）原文：评估单位"
      f"“{proto['evaluation']['unit']}”；报告“{proto['reporting']}”")
    ms = add["reporting"]["must_state"]
    w(f"- RuLSIF_exact_v2 的每个 EPFL 结果都须带的标签（附录 rev3，`{EPFL_RV2_SRC.name}/v10/epfl/EPFL_RULSIF_V2_ADDENDUM_REV3.json`，"
      f"sha256 `{short(PIN['epfl_rv2_addendum3'], 12)}`）原文：" + "；".join(f"“{x}”" for x in ms)
      + f"。继承标签原文：“{add['reporting']['inherited']}” 2048 中心的容量敏感性原文：“{add['capacity_sensitivity']['reporting']}”")
    if ep["state"] != "present":
        w(f"- 状态：pending（`{EPFL_RESULTS.name}` 尚不存在）。")
        w("")
        return
    d = ep["data"]
    w(f"- 读入的 `{EPFL_RESULTS.name}`：sha256 `{ep['sha']}`（{ep['bytes']} 字节，schema `{d.get('schema')}`"
      + (f"，文件自记 utc {d['utc']}" if "utc" in d else "") + f"）；`{EPFL_RESULTS_MD.name}`："
      + (f"sha256 `{ep['md_sha']}`" if ep["md_sha"] else "尚不存在") + "。必需基线未补齐，本节不作方法优劣陈述。")
    if not ep["known"]:
        w(f"- schema 不是本生成器认识的 `{EPFL_RESULTS_SCHEMA}`：本文不照录、不解读其中的数字，只记录 sha256 与顶层结构；按新 schema "
          "更新本生成器后再引用。顶层键（标量叶数）：" + "，".join(f"`{k}`（{sum(1 for _ in json_leaves(v))}）" for k, v in d.items()) + "。")
        w("")
        return
    dis, fams = d["disclosures"], d["families"]
    w(f"- 官方分析的披露原文：“{dis['benchmark']}” “{dis['statistical_unit']}” “{dis['thresholds']}” "
      f"“{dis['selection_asymmetry']}” 硬件规则：“{dis['hardware']['rule']}” M1 成本：“{dis['m1_cost']['note']}”")
    bs = d["bootstrap"]
    w(f"- 区间：{bs['B']} 次 bootstrap（{bs['generator']}，seed {bs['seed']}；{bs['resampled']}；{bs['interval']}），只作描述。"
      f"官方分析记录的问题：{'无' if not d['problems'] else '；'.join(map(str, d['problems']))}。")
    nf = sum(1 for f in fams.values() for r in f["finals"].values() if r is not None)
    np_ = sum(1 for f in fams.values() for r in f["finals"].values() if r is None)
    w(f"- 10 族最终单元：{nf} 个已在官方分析中" + (f"，{np_} 个 pending。" if np_ else "，没有 pending。")
      + " RuLSIF_exact_v2 两个发行见 §10.5。")
    w("")
    w("### 10.1 CAL 选择（seed 1017，L40S）")
    w("")
    rows = []
    for f, e in fams.items():
        c = e["cal"]
        rows.append([f, e["kind"], f"c{c['selected']['order']}：{cand_label(c['selected'])}", c["selected_step"],
                     f"{c['completed_candidates']}/{c['planned_candidates']}", str(c["budget_exhausted"]).lower(),
                     f3(c["cal_allocated_l40s_hours"]), c["predicted"]["budget_check"]])
    for line in table(["族", "类型", "选中配置", "CAL 选中步", "候选 完成/计划", "budget_exhausted", "CAL L40S GPU-h",
                       "运行前预算检查"], rows):
        w(line)
    w("")
    w("### 10.2 最终单元：DEV72 单元宏平均与配准级读数（硬件逐单元）")
    w("")
    rows = [epfl_final_row(f, sd, r) for f, e in fams.items() for sd, r in e["finals"].items()]
    for line in table(EPFL_FINAL_HEAD, rows):
        w(line)
    w("")
    w("### 10.3 四个预登记敏感性集合（单元宏平均 s<42 / s<9，括号内为单元数）")
    w("")
    rows = [[f, sd, gpu_short(r["hardware"]["gpu_name"])] + [epfl_cell(r["stats"][n]["unit_macro"], r["stats"][n]["n_units"])
                                                         for n in EPFL_SETS]
            for f, e in fams.items() for sd, r in e["finals"].items() if r is not None]
    for line in table(["族", "seed", "GPU", *EPFL_SETS], rows):
        w(line)
    w("")
    rows = [[f] + [f"{s4(x['success42'])} / {s4(x['success9'])} / {s3(x['median_err_full'])}" for x in
                   (d["sensitivity_deltas_vs_primary"][f][n] for n in EPFL_SETS[1:])] for f in fams]
    w("seed 均值单元宏相对 DEV72_primary 的差（集合 − 主集合；Δs<42 / Δs<9 / Δ中位 err_full，官方分析给出）：")
    w("")
    for line in table(["族", *EPFL_SETS[1:]], rows):
        w(line)
    w("")
    w("### 10.4 族的 seed 均值（单元宏，区间为官方分析的 bootstrap 95%）")
    w("")
    rows = [x for f, e in fams.items() for x in epfl_mean_rows(f, e["summary"], e["strata"])]
    for line in table(EPFL_MEAN_HEAD, rows):
        w(line)
    w("")
    w("混合硬件的族另给官方分析的 L40S 层；不声称跨硬件等价。EPFL 与 BF 的数字从不合并。")
    w("")
    w("### 10.5 RuLSIF_exact_v2（EPFL）")
    w("")
    w(f"官方分析的标签原文：“{dis['rulsif_v2']}”")
    w("")
    for t, x in d["rulsif_v2"].items():
        tag = f"T{t}"
        lab = (f"RuLSIF_exact_v2 {tag}（A1；{ms[2]}；字典 {ep['dict_from']} → {t}"
               + ("；容量敏感性，不替代 T1024）" if x["role"] != "primary" else "）"))
        c = x["cal"]
        mc = x["measured_cost_timing"]
        w(f"- {tag}（role `{x['role']}`，发行 `{short(x['release_sha256'])}`）：状态 {x['status']}；CAL "
          + (f"选中 c{c['selected']['order']}：{cand_label(c['selected'])}（候选完成 {c['completed_candidates']}，budget_exhausted "
             f"{str(c['budget_exhausted']).lower()}）" if c else "尚无选择")
          + f"；计时估计（measured_cost_timing）：CAL {f3(mc['cal_family_hours_charged'])} h，最终单元 {f3(mc['finals_hours'])} h，"
          f"合计 {f3(mc['family_total_hours'])} h。")
        rows = [epfl_final_row(lab, sd, r) for sd, r in x["finals"].items()]
        w("")
        for line in table(EPFL_FINAL_HEAD, rows):
            w(line)
        w("")
        if x.get("summary"):
            for line in table(EPFL_MEAN_HEAD, epfl_mean_rows(lab, x["summary"], x["strata"])):
                w(line)
            w("")


REVIEW_MAP = (
    ("RuLSIF_exact 能量退化（P1-1）", "v1 标为失败的适配（先检查全部行 F_true<0）；A1 修订下的 v2 并列报告", "§6.5、§8.1"),
    ("CAL 判别力饱和、选择由中位误差决定、幅度小（P2）", "逐方法重算 80/80 计数、区分条件与幅度、CAL→DEV 落差", "§6.1"),
    ("MF lr=1e-3 训练停滞（P3）", "梯度范数分段中位、<1e-6 步比例、最高 CAL s<42", "§6.3"),
    ("M1 cap 10/30 为近似头（P3）", "逐单元达到 cap 的批次比例、迭代数与相对 gap", "§6.2"),
    ("MLP 大多数步触发裁剪（P3）", "全部学习型方法逐单元的裁剪步比例", "§6.4"),
    ("节点速度混入成本（P2）", "按方法×节点列出每次配准秒数；成本表注明", "§5.4"),
    ("无 sacct 对账、无排队时间（P1-4、P2）", "由账本重建提交→StartTime 等待；sacct 只读探测快照；口径说明", "§5.2、§5.3"),
    ("node57 时钟偏慢（P3）", "由 START 事件与 StartTime 推出每个作业的少记下界", "§5.3"),
    ("选模不对称、CoMIR 只读最终 epoch（P2）", "列出各方法的选模自由度；看过 DEV 之后的 D1 敏感性单列，不替换冻结行", "§6.7、§8.3"),
    ("CoMIR 低于早先宿主 CoMIR（P2）", "引用审查数字（非本文计算），原因未查明", "§6.7"),
    ("MIND 适配差异与旋转依赖（P2）", "按 |θ*| 分层的成功率；D2 命名变体单列", "§6.6、§8.3"),
    ("InfoNCE 的 J 水平任意（P3）", "只用几何误差比较；不把 F 读作 S", "§6.9"),
    ("复现混入 GPU 型号（P2）", "只在同型号同 seed 内配对；不与 L40S 主体合并；按型号报成本", "§7"),
    ("复现计费起点不一致（P1-4）", "复现成本单列；启动开销写未测量（一个作业有 StartTime 快照）", "§5.3、§7.5"),
    ("RTX/H100 没有 CUDA 数值门禁（P2）", "汇总 xhw_t01 诊断与 D3 跨硬件门禁；EPFL 的 T01 容差修订单列", "§7.6、§8.3、§9"),
    ("困难 core（P2）", "逐 core 计数与 slide 分层", "§6.8"),
    ("续跑路径从未实测（P2）", "检查每个单元只来自一个作业、无续跑", "§2.3"),
)


def render_review(R, A, w):
    w("## 11 独立审查（2026-10-19）各项的处理")
    w("")
    w(f"审查记录 `{REVIEW_MD.relative_to(HAND)}`（sha256 `{short(PIN['review'], 12)}`）。本文不沿用审查中的数字，"
      "全部由记录重算；引用审查原文的地方都注明。")
    w("")
    for line in table(["审查项", "本文处理", "位置"], [list(x) for x in REVIEW_MAP]):
        w(line)
    w("")


def render_limits(R, A, w):
    xh = R["xhw"]
    h100 = [t["tag"] for t in xh.get("tags", []) if any(p["infra"] for p in t["parts"].values())]
    ep = R["epfl"]
    w("## 12 局限与待决事项")
    w("")
    w("局限（记录事实）：")
    w("")
    ep = R["epfl"]
    if ep["state"] != "present":
        etxt = "EPFL 的官方分析尚未写出（§10）。"
    elif not ep["known"]:
        etxt = "EPFL 官方分析的 schema 未被识别，数字未引用（§10）。"
    else:
        etxt = "EPFL 上 RuLSIF_exact_v2：" + "，".join(f"T{t} {x['status']}" for t, x in ep["data"]["rulsif_v2"].items()) + "（§10.5）。"
    w("- 必需基线集合仍未确认补齐：RuLSIF v1 失败；RuLSIF v2 在 A1 修订下运行（§8.1）；" + etxt
      + "按决定 §5 不形成方法优劣、强基线或新颖性结论。")
    w("- DEV32 是开发集，不是确认；CAL8 在 <42/<9 两个阈值上大多饱和，选择主要由中位误差决定（§6.1）。")
    w("- DEV 的 TMA 位置组都只有一个 core，位置组重采样与 core 重采样相同；保守性只来自 DEV24 的排除（§4）。")
    w("- 成本混有节点速度与节点时钟偏差；没有 sacct 对账；复现单元的启动开销未测量（§5）。")
    w("- 选模自由度在方法间不对称；CoMIR 与早先宿主结果的差距原因未查明；MIND 是二维适配（§6.6、§6.7）。D1/D2 是看过 DEV "
      "之后的敏感性，不替换冻结行（§8.3）。")
    passed = [t["tag"] for t in xh.get("tags", []) if len(t["parts"]) == 4 and all(p["status"] == "PASS" for p in t["parts"].values())]
    w("- 复现、RuLSIF v2 最终单元、容量扩展的 seed 1003 与 EPFL 最终单元混有 GPU 型号；只按硬件分层报告，不声称跨硬件等价（§9）。"
      + (f"跨硬件 T01 诊断中 {', '.join(h100)} 因 CUDA 设备不可用而没有数值结论，" if h100 else "")
      + (f"{', '.join(passed)} 四个 part 全部 PASS（§7.6）；PASS 不等于跨卡等价。" if passed else ""))
    w("- EPFL 在 RTX6000PRO / H100 上的两行 checkpoint 能量 / 梯度检查改用 T01 容差，这是 H100 门禁失败之后的判据修改（§9）。")
    w("")
    w("待决事项（需要 owner 决定；本文不预设答案）：")
    w("")
    w("1. RuLSIF_exact_v2（A1）是否视为满足决定 §5 的必需基线，以及是否进入配对差（目前只在主表并列，§3.2、§8.1）。")
    nopass = [g for g, v in sorted(A.get("xhw_verdict", {}).items()) if not any(ok for _, ok in v)]
    if nopass:
        w(f"2. {'、'.join(nopass)} 的跨硬件 T01 诊断是换节点重跑，还是只在报告中声明该型号未过决定 §7 的数值门禁（§7.6）。")
    else:
        w("2. 跨硬件 T01 诊断在复现用到的每个型号上都有完整 PASS（§7.6）；是否把它写入复现的准入记录。")
    w("3. node57 的时钟少记是否需要在成本表中修正，还是保持“按记录、另列下界”（§5.3）。")
    w(f"4. 官方 EPFL 分析 `{EPFL_RESULTS.name}` 写出或更新后重新生成本文（`--check` 会显示差异）；§10 读入它并记录其 sha256。")
    w("")


def render_repro(R, A, w):
    inp = R["inp"]
    w("## 13 复现与输入摘要")
    w("")
    w("```")
    w(f"OMP_NUM_THREADS=1 {PYTHON} {SELF_REL}            # 写报告与 sidecar")
    w(f"OMP_NUM_THREADS=1 {PYTHON} {SELF_REL} --check    # 内存渲染并与磁盘/HEAD 比较")
    w(f"{PYTHON} {SELF_REL} --verify-checkpoints         # 重算全部 checkpoint hash，写 {CKPT_CHECK_REL}")
    w(f"{PYTHON} {SELF_REL} --probe-sacct                # 只读 sacct 探测，写 {SACCT_PROBE_REL}")
    w("```")
    w("")
    w(f"- 机器可读结果：`{SIDECAR_REL}`（与本文同一次运行写出，包含未经舍入的数值）。")
    w(f"- 舍入：表中的小数按十进制四舍五入到所示位数（先在 1e-12 处去掉二进制浮点噪声），例如 0.78125 → {f4(0.78125)}、"
      f"-0.03125 → {s4(-0.03125)}；2026-10-19 的初版直接按二进制浮点值格式化，个别末位因此相差 1。")
    w(f"- 运行环境：Python {sys.version.split()[0]}，numpy {np.__version__}。bootstrap 每次 {N_BOOT} 个重采样，随机种子为"
      "对照键（A|B|集合|core 列表）的 sha256 前 16 位十六进制；位置组重采样用同一种子。")
    w("- 输入摘要（每个根目录下本次读取的文件数、字节数，以及“相对路径  sha256”逐行排序后的 sha256）：")
    w("")
    roots = INPUT_ROOTS
    rows = []
    for name, root in roots:
        n, b, d = inp.digest(root)
        rows.append([name, f"`{root}`", n, b, f"`{short(d, 16)}`" if n else "—"])
    for line in table(["输入", "根", "文件", "字节", "摘要"], rows):
        w(line)
    w("")
    w("- 固定的冻结输入 hash：" + "，".join(f"{k} `{short(v, 12)}`" for k, v in PIN.items()) + "。")
    w("- EPFL 不在 `--verify-checkpoints` 范围内：本文不读 EPFL 原始结果，只读官方分析 `EPFL_RESULTS.json`（存在时）与上列记录。")
    w("")


# ----------------------------------------------------------------------------------------------------------------- sidecar
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, Path):
        return str(x)
    return x


def sidecar_rv2(R, A):
    rv2 = R["rv2"]
    if "rv2" not in A:
        return {"state": rv2["state"]}
    return {"state": rv2["state"], "selected_candidate": rv2["selected_candidate"], "selected_choice": rv2["selected_choice"],
            "cal_job": rv2["cal_job"], "cal_alloc_s": A["rv2"]["cal_alloc"],
            "finals": [{k: v for k, v in r.items()} for r in A["rv2"]["rows"]], "mean32": A["rv2"]["mean32"],
            "mean24": A["rv2"]["mean24"], "strata": A["rv2"]["strata"]}


def sidecar_capacity(R):
    c = R["capacity"]
    if c["state"] != "complete":
        return {"state": c["state"]}
    return {"state": "complete", "total_gpu_h": c["total_s"] / 3600, "gpu_h_by_type": c["bytype"],
            "units": [{"family": u["method"], "K": u["K"], "seed": u["seed"], "gpu": u["gpu"], "host": u["host"], "job": u["job"],
                       "selected_step": u["selected"]["step"], "dev32": u["dev32"], "dev24": u["dev24"],
                       "k16_dev32": u["k16"]["dev32"], "alloc_s": u["receipt"]["allocated_seconds"]} for u in c["units"]],
            "cells": [{"family": x["family"], "K": x["K"],
                       "views": [{"name": v["name"], "seeds": v["seeds"],
                                  **{f"{b}_{m}": v[(b, m)] for b in ("DEV32", "DEV24") for m in ("success42", "success9")}}
                                 for v in x["views"]]} for x in c["cells"]]}


def sidecar_sensitivity(R):
    s = R["sensitivity"]
    if s["state"] != "complete":
        return {"state": s["state"]}
    r = s["r"]
    return {"state": "complete", "label": r["label"],
            "D1": {k: r["D1"][k] for k in ("paired_success42_mean_over_seeds", "paired_success9_mean_over_seeds", "grid_cell")},
            "D2": {k: r["D2"][k] for k in ("paired_success42_variant_minus_frozen", "grid_cell", "stratum_20_30_gain")},
            "D3": [{k: j[k] for k in ("job", "family", "partition", "gpu_name", "status")} for j in r["D3"]["jobs"]],
            "costs": r["costs"]["allocated_gpu_hours"]}


def sidecar_hardware(R):
    h = R["hwd"]
    return {"owner_capacity_authority": h["cap_authority"], "owner_rv2": h["rv2_owner"],
            "epfl_amendment": {k: h["epfl_amendment"][k] for k in ("id", "label", "owner", "failed_gate", "rule")},
            "epfl_gates": h["epfl_gates"], "where_run": hardware_rows(R)}


def sidecar_epfl(R, A):
    ep = R["epfl"]
    out = {"state": ep["state"], "source": rel_where(EPFL_RESULTS), "recomputed_from_raw_results": False}
    if ep["state"] == "present":
        out.update(sha256=ep["sha"], bytes=ep["bytes"], md_sha256=ep["md_sha"], n_leaves=sum(1 for _ in json_leaves(ep["data"])))
    return out


def unit_rec(u: dict) -> dict:
    r = {"where": u["where"], "method": u["method"], "phase": u["phase"], "seed": u["seed"], "order": u.get("order"),
         "candidate": u.get("candidate"), "job": u["job"], "host": u["host"], "gpu": u["gpu"],
         "selected_step": u["selected"]["step"], "cal_selected": u["selected"]["metrics"], "complete_sha256": u["complete_sha"]}
    if "dev32" in u:
        r.update(dev32=u["dev32"], dev24=u["dev24"])
    return r


def headlines(R: dict, A: dict) -> list:
    """Short plain-text summary printed by main(); every number comes from the same objects as the report."""
    L = []
    if "rv2" in A:
        a = A["rv2"]
        for r in a["rows"]:
            L.append(f"BF RuLSIF_exact_v2 (A1 post-gate amendment) seed {r['seed']} {gpu_short(r['gpu'])} {r['host']} job {r['job']}: "
                     f"DEV32 s<42/s<9/median {'/'.join(mtrip(r['dev32']))}")
        L.append(f"BF RuLSIF_exact_v2 pooled (mixed hardware) DEV32 {'/'.join(mtrip(a['mean32']))}; "
                 + "; ".join(f"{gpu_short(g)} stratum seeds {','.join(map(str, st['seeds']))} {'/'.join(mtrip(st['mean32']))}"
                             for g, st in a["strata"].items())
                 + f"; v1 seed mean s<42 {f4(A['main'][FAILED_V1]['mean32']['success42'])}")
    else:
        L.append(f"BF RuLSIF_exact_v2: {R['rv2']['state']}")
    c = R["capacity"]
    if c["state"] == "complete":
        for cell in c["cells"]:
            parts = []
            for v in cell["views"]:
                x = v[("DEV32", "success42")]
                parts.append(f"{'pooled' if v['name'] == 'pooled' else gpu_short(v['name'])} {s4(x['mean'])} "
                             f"[{s4(x['ci'][0])}, {s4(x['ci'][1])}] {x['grid']}")
            L.append(f"capacity {cell['family']} K{cell['K']} DEV32 delta s<42 vs K16: " + "; ".join(parts))
        multi = sorted({v[(b, m)]["grid"] for cell in c["cells"] for v in cell["views"] if not v["single"]
                        for b in ("DEV32", "DEV24") for m in ("success42", "success9")})
        L.append(f"capacity: {len(c['units'])} units, {fx(c['total_s'] / 3600, 3)} GPU-h; pooled and L40S grid labels: {', '.join(multi)}")
    else:
        L.append(f"capacity: {c['state']}")
    sv = R["sensitivity"]
    if sv["state"] == "complete":
        for d, key in (("D1", "paired_success42_mean_over_seeds"), ("D2", "paired_success42_variant_minus_frozen")):
            p = sv["r"][d][key]
            L.append(f"post-DEV sensitivity {d}: paired s<42 {s4(p['mean'])} [{s4(p['ci_low'])}, {s4(p['ci_high'])}] {sv['r'][d]['grid_cell']}")
        L.append("post-DEV sensitivity D3: " + ", ".join(f"{j['job']} {j['family']} {j['partition']} {j['status']}" for j in sv["r"]["D3"]["jobs"]))
    else:
        L.append(f"post-DEV sensitivity: {sv['state']}")
    ep = R["epfl"]
    if ep["state"] != "present":
        L.append(f"EPFL: pending ({EPFL_RESULTS.name} of the official frozen-protocol analysis not written yet; "
                 "no EPFL number is recomputed here)")
    else:
        L.append(f"EPFL: official analysis {EPFL_RESULTS.name} sha256 {ep['sha']} read "
                 f"({sum(1 for _ in json_leaves(ep['data']))} scalar leaves); {EPFL_RESULTS_MD.name} sha256 {ep['md_sha'] or 'absent'}")
        if ep["known"]:
            for f, e in ep["data"]["families"].items():
                sm = e["summary"]
                if not sm or not sm.get("seed_mean"):
                    L.append(f"EPFL DEV72 {f}: pending")
                    continue
                um, bi = sm["seed_mean"]["DEV72_primary"]["unit_macro"], sm["bootstrap95"]["DEV72_primary"]
                txt = (f"EPFL DEV72 {f} ({len(sm['seeds_complete'])} seeds; "
                       + ", ".join(f"{sd} {gpu_short(g)}" for sd, g in e["strata"]["hardware_per_seed"].items())
                       + f"): unit-macro s<42 {f4(um['success42'])} [{f4(bi['success42'][0])}, {f4(bi['success42'][1])}], "
                       f"s<9 {f4(um['success9'])} [{f4(bi['success9'][0])}, {f4(bi['success9'][1])}], median {f3(um['median_native_px'])} px")
                st = e["strata"]
                if st["n_L40S"] and st["n_L40S"] < st["n_pooled"]:
                    txt += f"; L40S stratum ({st['n_L40S']}) {f4(st['L40S_only']['success42'])}/{f4(st['L40S_only']['success9'])}"
                L.append(txt)
            for t, x in ep["data"]["rulsif_v2"].items():
                c = x["cal"]
                L.append(f"EPFL RuLSIF_exact_v2 T{t} ({x['role']}, dictionary {ep['dict_from']} -> {t}): {x['status']}; CAL "
                         + (f"selected c{c['selected']['order']} {cand_label(c['selected'])}" if c else "no selection yet"))
    return L


def sidecar(R: dict, A: dict) -> dict:
    inp = R["inp"]
    jobs = [{k: v for k, v in j.items() if k != "checkpoint_events"} for j in R["jobs"]]
    return {
        "schema": "v10.phase2.results.v1", "generator": SELF_REL, "report": REPORT_REL, "n_boot": N_BOOT,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "inputs": {k: {"root": str(v), **dict(zip(("files", "bytes", "digest"), inp.digest(v)))} for k, v in INPUT_ROOTS},
        "pins": {"frozen": PIN, "selection": SELECTION_PIN, "release": RELEASE_PIN},
        "roles": {k: R["roles"][k] for k in ("FIT", "CAL", "DEV", "DEV24", "DEV24_excluded")},
        "units": [unit_rec(u) for u in R["primary_units"] + R["rep"]["units"]],
        "main": {f: {"mean32": A["main"][f]["mean32"], "mean24": A["main"][f]["mean24"]} for f in ORDER},
        "contrasts": [{k: (v if k in ("A", "B", "group") else {kk: vv for kk, vv in v.items()}) for k, v in c.items()}
                      for c in A["contrasts"]],
        "jobs": jobs,
        "cost": {f: {"cal_s": c["cal_s"], "fin_s": c["fin_s"], "val_s": c["val_s"]} for f, c in A["cost"].items()},
        "same_total_cost": {f: {"cal_alloc": s["cal_alloc"], "sel_t": s["sel_t"], "minimal": s["minimal"], "actual": s["actual"],
                                "finals": s["finals"],
                                "levels": [{"budget_h": l["budget_h"], "is_final": l["is_final"],
                                            "choice": None if l["choice"] is None else
                                            {"order": l["choice"]["order"], "step": l["choice"]["step"], "t": l["choice"]["t"],
                                             "metrics": l["choice"]["metrics"]}} for l in s["levels"]]}
                            for f, s in R["stc"].items()},
        "diagnostics": {"saturation": A["saturation"], "m1": A["m1"], "stall": A["stall"], "clip": A["clip"],
                        "rulsif": [{k: v for k, v in r.items()} for r in A["rulsif"]], "strata": A["strata"], "hard": A["hard"],
                        "slide": A["slide"], "speed": [[k[0], k[1], *v] for k, v in A["speed"].items()],
                        "node57": A["node57"], "cal_vs_dev": A["cal_vs_dev"]},
        "replication": {"pairs": [{"B": p["B"], "gpu": p["gpu"], "seeds": p["seeds"], "res": p["res"], "per_seed": p["per_seed"]}
                                  for p in A["rep_pairs"]], "cost": A["rep_cost"], "starttime": A["rep_starttime"]},
        "cross_unit": R["xchk"], "xhw": R["xhw"], "checkpoints": {"n": len(R["pt_list"]), "digest": R["pt_digest"]},
        "rulsif_v2_bf": sidecar_rv2(R, A), "capacity_ext": sidecar_capacity(R), "sensitivity": sidecar_sensitivity(R),
        "hardware_disclosure": sidecar_hardware(R), "epfl": sidecar_epfl(R, A), "headlines": headlines(R, A),
    }


# -------------------------------------------------------------------------------------------------------------------- main
def git_head_text(rel: str):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True)
    return r.stdout.decode("utf-8") if r.returncode == 0 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V10 phase-2 results report generator (writes only the report and sidecar)")
    ap.add_argument("--check", action="store_true", help="render in memory; compare with the files on disk and at HEAD")
    ap.add_argument("--verify-checkpoints", action="store_true", help=f"re-hash every checkpoint and write {CKPT_CHECK_REL}")
    ap.add_argument("--probe-sacct", action="store_true", help=f"read-only sacct probe, writes {SACCT_PROBE_REL}")
    ap.add_argument("--output", default=str(REPO / REPORT_REL))
    ap.add_argument("--json-output", default=str(REPO / SIDECAR_REL))
    args = ap.parse_args(argv)
    if args.probe_sacct:
        R = collect(verify_pt=False)
        probe_sacct([j["job"] for j in R["jobs"]])
        return 0
    if args.verify_checkpoints:
        R = collect(verify_pt=True)
        rec = {"utc": utc_str(dt.datetime.now(dt.timezone.utc).timestamp()), "n": len(R["pt_list"]),
               "bytes": sum(v for k, v in R["inp"].sizes.items() if k.endswith(".pt")), "digest": R["pt_digest"],
               "all_match": True, "scope": "every .pt artifact listed in the COMPLETE records of the BF units (primary, replication, "
                                           "RuLSIF_exact_v2, K8/K32 capacity extension); EPFL excluded"}
        (REPO / CKPT_CHECK_REL).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        print("wrote", CKPT_CHECK_REL, rec["n"], "checkpoints", rec["bytes"], "bytes")
        return 0
    R = collect(verify_pt=False)
    A = analyze(R)
    analyze_new(R, A)
    md = render(R, A)
    js = json.dumps(jsonable(sidecar(R, A)), ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n"
    outs = ((Path(args.output), md), (Path(args.json_output), js))
    for line in headlines(R, A):
        print(line)
    if args.check:
        ok = True
        for path, text in outs:
            disk = path.read_text() if path.exists() else None
            rel = path.resolve().relative_to(REPO).as_posix() if path.resolve().is_relative_to(REPO) else None
            head = git_head_text(rel) if rel else None
            for label, other in (("disk", disk), ("HEAD", head)):
                if other is None:
                    print(f"{path.name}: no {label} version")
                    ok = ok and label == "HEAD"
                    continue
                if other != text:
                    ok = False
                    diff = list(difflib.unified_diff(other.splitlines(), text.splitlines(), f"{label}", "rendered", n=0, lineterm=""))
                    print(f"{path.name}: differs from {label} ({len(diff)} diff lines)")
                    print("\n".join(diff[:20]))
                else:
                    print(f"{path.name}: identical to {label}")
        return 0 if ok else 1
    for path, text in outs:
        path.write_text(text)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
