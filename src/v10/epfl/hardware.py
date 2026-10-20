"""EPFL finals hardware amendment (owner decision 2026-10-19: 「final 放开，CAL 留 L40S」).  Standard library only.

A release of schema v10.epfl.release_hw.v1 is a FINALS-ONLY version of the 10-family release 3dae6434...: it consumes the CAL selections
made under that release (pinned by sha256 where they exist at build time, verified through the unchanged selection checks otherwise),
writes new final units into the same results root, and accepts three GPU types for finals:
  NVIDIA L40S, NVIDIA RTX PRO 6000 Blackwell Server Edition, NVIDIA H100 NVL     (names exactly as the CUDA device reports them)
CAL is never run under it (CAL and timing stay on L40S, under the parent release; the budget unit is L40S GPU-h).  A final on a non-L40S
GPU additionally requires a PASS cross-hardware item-5 gate of this exact source commit on that GPU type (written by
slurm/v10_epfl/gpu_gate_xhw.sbatch under <bundle>/xhw_gates/<label>/).  Every other environment / precision check is unchanged
(runtime.runtime_environment).  The science code is byte-identical to the parent source 7ceb2f1e (checked here, at release build and at
run time).  Each final records the GPU name in its identity (a unit is single-hardware: a resume on another GPU type is refused, a COMPLETE
unit is reused on any hardware), and the GPU name, node, partition and allocated time in its ledger rows and COMPLETE.json.

A release of schema v10.epfl.release_hw_replicate.v1 (EPFL_HW_REPLICATE_20261020) is a POST-HOC HARDWARE SENSITIVITY DIAGNOSTIC beside the
amended hardware release d9d67cf2: it reruns the two RTX PRO 6000 finals of d9d67cf2 (MF_VCS_M1 and MLP_VCS_perm8, seed 1001) on L40S with
the same science, the same (pinned) CAL selection and the same seed, into its own results root; it never replaces, reuses or touches a
primary unit (their COMPLETE.json hashes are pinned and re-verified), and each replicate unit carries the diagnostic label in its identity.
A release of schema v10.epfl.release_hw_control.v1 (EPFL_HW_CONTROL_20261020) is its L40S-vs-L40S control under the same rules: two L40S
finals of d9d67cf2 (the same families, seed 1002) rerun on L40S into a third results root.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import production as PR

RELEASE_SCHEMA_HW, RELEASE_STATUS_HW = "v10.epfl.release_hw.v1", "FROZEN_VALIDATED"
PARENT_RELEASE_SHA256 = "3dae643414dca842eba8243bbf6424733f9373d19468928fc23349fcc86a3875"
PARENT_SOURCE_COMMIT = "7ceb2f1e77c0c763059476e50b94f02e43457bad"
CAL_GPU = "NVIDIA L40S"
FINAL_GPUS = ("NVIDIA L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 NVL")
GPU_LABEL = {"NVIDIA L40S": "L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX6000PRO", "NVIDIA H100 NVL": "H100"}
FINAL_PARTITIONS = "RTX6000PRO,L40S,H100"
FINAL_EXCLUDE = "node51,node53,node60"
AMENDMENT = PR.EPFL_DIR / "EPFL_HARDWARE_AMENDMENT_20261019.json"
XHW_AMENDMENT = PR.EPFL_DIR / "EPFL_XHW_GATE_AMENDMENT_20261020.json"
XHW_CRITERION_ID = "EPFL_XHW_GATE_AMENDMENT_20261020"      # RTX6000PRO / H100: the two checkpoint energy / gradient rows on T01's tolerance
CHECKPOINT_ROWS = ("checkpoint.state_hash_equal_after_reload", "checkpoint.energy_and_gradient_bitwise_T1_rot+16.26deg",
                   "checkpoint.energy_and_gradient_bitwise_T2_rot-22.62deg")
DELTA_REVIEW_HW = PR.EPFL_DIR / "EPFL_DELTA_REVIEW_HW.json"
# Post-hoc hardware sensitivity diagnostic (2026-10-20): the two RTX PRO 6000 finals of release d9d67cf2 rerun on L40S, same units.
REPLICATE_SCHEMA = "v10.epfl.release_hw_replicate.v1"
REPLICATE_LABEL = "POST-HOC HARDWARE SENSITIVITY DIAGNOSTIC"
HW2_RELEASE_SHA256 = "d9d67cf2b987c1a2a7db8e81ffe8cfc52387d71067dc25ce9f89e1fcc3f79a8c"
HW2_SOURCE_COMMIT = "ee55bb5d7c89c58c98cb0a0118562b728dffd9ed"
REPLICATE_UNITS = {"MF_VCS_M1": [1001], "MLP_VCS_perm8": [1001]}
REPLICATE_PRIMARY = {"MF_VCS_M1/1001": "1050507", "MLP_VCS_perm8/1001": "1050524"}      # the RTX PRO 6000 finals of d9d67cf2 (node59)
REPLICATE_PARTITION, REPLICATE_EXCLUDE = "L40S", "node51,node60"
REPLICATE_RECORD = PR.EPFL_DIR / "EPFL_HW_REPLICATE_20261020.json"
DELTA_REVIEW_REPLICATE = PR.EPFL_DIR / "EPFL_DELTA_REVIEW_HW_REPLICATE.json"
REPLICATE_RELEASE_SHA256 = "ca2bc60a57a5e2b768a7a0b13d16588ab7b6e63a0ef02fe4dabcdc40e252d659"    # its release (source 952c8982)
REPLICATE_SOURCE_COMMIT = "952c898237529852263902f169a64748ec2f5985"
# Its L40S-vs-L40S control (2026-10-20, EPFL_HW_CONTROL_20261020): two L40S finals of d9d67cf2 rerun on L40S, same units (run-to-run
# non-determinism on one GPU type, shown beside the RTX-vs-L40S pairs).
CONTROL_SCHEMA = "v10.epfl.release_hw_control.v1"
CONTROL_UNITS = {"MF_VCS_M1": [1002], "MLP_VCS_perm8": [1002]}
CONTROL_PRIMARY = {"MF_VCS_M1/1002": "1050508", "MLP_VCS_perm8/1002": "1050531"}        # the L40S finals of d9d67cf2 (node50, node57)
CONTROL_RECORD = PR.EPFL_DIR / "EPFL_HW_CONTROL_20261020.json"
DELTA_REVIEW_CONTROL = PR.EPFL_DIR / "EPFL_DELTA_REVIEW_HW_CONTROL.json"
DIAGNOSTICS = {    # schema -> rule of the diagnostic release version
    REPLICATE_SCHEMA: {"role": "RTX_VS_L40S_REPLICATE", "units": REPLICATE_UNITS, "primary": REPLICATE_PRIMARY, "primary_on_l40s": False,
                       "record": REPLICATE_RECORD, "record_key": "replicate_record", "review": DELTA_REVIEW_REPLICATE,
                       "review_key": "delta_review_replicate", "base_source_commit": HW2_SOURCE_COMMIT, "job_prefix": "v10e_hwrep_",
                       "controls_for": None},
    CONTROL_SCHEMA: {"role": "L40S_VS_L40S_CONTROL", "units": CONTROL_UNITS, "primary": CONTROL_PRIMARY, "primary_on_l40s": True,
                     "record": CONTROL_RECORD, "record_key": "control_record", "review": DELTA_REVIEW_CONTROL,
                     "review_key": "delta_review_control", "base_source_commit": REPLICATE_SOURCE_COMMIT, "job_prefix": "v10e_hwctl_",
                     "controls_for": REPLICATE_RELEASE_SHA256},
}
# Files that may differ from the parent source 7ceb2f1e (orchestration, records, tests); everything else under the science paths must be
# byte-identical (git blob sha256 at 7ceb2f1e).
NON_SCIENCE_CHANGES = ("src/v10/epfl/hardware.py", "scripts/v10_epfl/run.py", "scripts/v10_epfl/release_hw.py", "scripts/v10_epfl/hw_decide.py",
                       "scripts/v10_epfl/gpu_gate_xhw.py", "scripts/v10_epfl/hw_decide_xhw.py", "scripts/v10_epfl/xhw_diag.py",
                       "slurm/v10_epfl/xhw_diag.sbatch", "v10/epfl/EPFL_XHW_GATE_AMENDMENT_20261020.json",
                       "slurm/v10_epfl/prod_final_hw.sbatch", "slurm/v10_epfl/gpu_gate_xhw.sbatch", "tests/v10/test_epfl_hardware.py",
                       "v10/epfl/EPFL_HARDWARE_AMENDMENT_20261019.json", "v10/epfl/EPFL_DELTA_REVIEW_HW.json",
                       "scripts/v10_epfl/hw_replicate_decide.py", "v10/epfl/EPFL_HW_REPLICATE_20261020.json",
                       "v10/epfl/EPFL_DELTA_REVIEW_HW_REPLICATE.json", "scripts/v10_epfl/hw_control_decide.py",
                       "v10/epfl/EPFL_HW_CONTROL_20261020.json", "v10/epfl/EPFL_DELTA_REVIEW_HW_CONTROL.json")


def science_files(repo: Path = PR.REPO) -> list:
    """The science paths: every Python file under src/v10 and src/v9, the EPFL gate / timing scripts, the EPFL and supplement records that
    existed in the parent source, the data lock (exact list, sorted)."""
    paths = set()
    for rel in ("src/v10", "src/v9"):
        paths.update(str(p.relative_to(repo)) for p in (repo / rel).rglob("*.py") if "__pycache__" not in p.parts)
    paths.update(("scripts/v10_epfl/gpu_gate.py", "scripts/v10_epfl/timing.py", "scripts/v10_epfl/finalize.py", "scripts/v10_epfl/decide.py",
                  "scripts/v10_epfl/audit.py"))
    paths.update(str(p.relative_to(repo)) for p in (repo / "v10/epfl").glob("*.json"))
    paths.update(str(p.relative_to(repo)) for p in (repo / "v10/supplement").glob("*.json"))
    return sorted(p for p in paths if p not in NON_SCIENCE_CHANGES and (repo / p).exists())


def science_identity(repo: Path = PR.REPO) -> dict:
    """{path: [sha256 here, sha256 at the parent source commit]} for every science file; RuntimeError if any differs or is new."""
    out = {}
    for f in science_files(repo):
        here = PR.sha(repo / f)
        r = subprocess.run(["git", "-C", str(repo), "show", f"{PARENT_SOURCE_COMMIT}:{f}"], capture_output=True)
        parent = hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
        out[f] = [here, parent]
    bad = {f: v for f, v in out.items() if v[0] != v[1]}
    PR.require(not bad, f"science code differs from the parent source {PARENT_SOURCE_COMMIT[:8]}: {sorted(bad)}")
    return out


def gpu_accepted(phase: str, gpu_name: str) -> bool:
    """CAL: L40S only.  Final: the three accepted types (exact CUDA names)."""
    if phase == "cal":
        return gpu_name == CAL_GPU
    if phase == "final":
        return gpu_name in FINAL_GPUS
    raise ValueError(phase)


def checkpoint_rows_ok(path) -> bool:
    """The amended checkpoint part (EPFL_XHW_GATE_AMENDMENT_20261020): the state-hash row bitwise PASS, the two energy / gradient rows PASS
    under T01's float32 round-trip tolerance (atol 1e-5, rtol 1e-4) with their measured comparisons recorded."""
    try:
        rows = {r["name"]: r for r in PR.read(path)["rows"]}
    except (OSError, ValueError, KeyError):
        return False
    if set(rows) != set(CHECKPOINT_ROWS) or rows[CHECKPOINT_ROWS[0]].get("status") != "PASS":
        return False
    for name in CHECKPOINT_ROWS[1:]:
        r = rows[name]
        if r.get("status") != "PASS" or not str(r.get("criterion", "")).startswith(XHW_CRITERION_ID):
            return False
        for part in ("energy", "gradient"):
            c = r.get(part) or {}
            if not (c.get("allclose") is True and c.get("finite") is True and c.get("atol") == 1e-5 and c.get("rtol") == 1e-4
                    and isinstance(c.get("worst_err_over_bound"), float) and c["worst_err_over_bound"] <= 1.0):
                return False
    return True


def xhw_gate(release: dict, gpu_name: str, source_commit: str):
    """The PASS cross-hardware item-5 gate record of this source commit on this GPU type under the amended acceptance (None for L40S: the
    unchanged item-5 gate 1048825 is the L40S gate of the identical science code).  RuntimeError when no such record exists."""
    if gpu_name == CAL_GPU:
        return None
    label = GPU_LABEL[gpu_name]
    found = []
    for p in sorted(glob.glob(os.path.join(release["xhw_gates"]["dir"], label, "gpu_gate", "*", "GATE.json"))):
        g = PR.read(p)
        ok = (g.get("schema") == "v10.epfl.item5_gate.v1" and g.get("status") == "PASS" and g.get("pins", {}).get("commit") == source_commit
              and g.get("pins", {}).get("record_sha256") == PR.GATE_RECORD_SHA256 and g.get("environment", {}).get("gpu_name") == gpu_name
              and len(g.get("parts", {})) == 8 and all(v.get("status") == "PASS" for v in g["parts"].values())
              and set(g.get("data_reads_roles", [])) <= {"FIT", "CAL"}
              and (g.get("xhw_criterion") or {}).get("id") == XHW_CRITERION_ID and checkpoint_rows_ok(Path(p).parent / "checkpoint.json"))
        if ok:
            found.append({"path": p, "sha256": PR.sha(p), "job": g["pins"]["job"]})
    PR.require(bool(found), f"No PASS cross-hardware gate of commit {source_commit[:8]} on {gpu_name} under {release['xhw_gates']['dir']}/{label}: "
                            "a final on this GPU type waits for it")
    return found[-1]


def verify_release_hw(path, expected_sha, family=None, *, phase="final", candidates_fn=None, schedule_fn=None, repo: Path = PR.REPO) -> dict:
    """Every check of the hardware release version except the Slurm requirement (the runner adds it)."""
    path = Path(path).resolve()
    PR.require(bool(expected_sha) and PR.sha(path) == expected_sha, "Missing or changed submission-pinned EPFL release")
    rel = PR.read(path)
    PR.require(rel.get("schema") == RELEASE_SCHEMA_HW and rel.get("status") == RELEASE_STATUS_HW, "Not a frozen EPFL hardware release")
    inv = PR.source_inventory(repo)
    PR.require(rel.get("source_files") == inv and rel.get("code_sha256") == PR.digest(inv), "EPFL source identity changed since the release")
    num = PR.numerics_identity(repo)
    PR.require(rel.get("numerics_identity") == num == PR.gate_numerics_identity(repo), "Numerics files differ from the item-5 gate commit")
    PR.require(rel.get("science_identity") == science_identity(repo), "Science files differ from the release record")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    PR.require(head == rel.get("source_commit"), "The running tree is not the release's source commit")
    parent = rel.get("parent_release") or {}
    PR.require(parent.get("sha256") == PARENT_RELEASE_SHA256 and Path(parent.get("path", "")).exists() and PR.sha(parent["path"]) == PARENT_RELEASE_SHA256,
               "Parent release 3dae6434 missing or changed")
    par = PR.read(parent["path"])
    for key in ("frozen_protocol", "activation_record", "group_record", "data_lock", "item5_gate", "item5_timing", "delta_review"):
        ref = par[key]; p = PR._abs(ref["path"], repo)
        PR.require(p.exists() and PR.sha(p) == ref["sha256"] and rel.get(key) == ref, "Parent pin changed: " + key)
    for key, p in (("hardware_amendment", AMENDMENT), ("delta_review_hw", DELTA_REVIEW_HW), ("xhw_gate_amendment", XHW_AMENDMENT)):
        ref = rel.get(key) or {}
        PR.require(ref.get("sha256") == PR.sha(p) and Path(PR._abs(ref.get("path", ""), repo)) == p, "Pinned input changed: " + key)
    PR.require(rel.get("phases") == ["FINAL_DEV"] and rel.get("final_gpus") == list(FINAL_GPUS) and rel.get("cal_gpu") == CAL_GPU,
               "Hardware rule differs from the amendment")
    PR.require(rel.get("cal_release", {}).get("sha256") == PARENT_RELEASE_SHA256, "CAL selections must come from the parent release")
    PR.require((rel.get("xhw_gates") or {}).get("criterion") == XHW_CRITERION_ID, "Cross-hardware acceptance differs from the amendment")
    PR.require(Path(rel.get("results_root", "")) == Path(par["results_dir"]), "Results root must be the parent's (its CAL selections)")
    if (path.parent / "PAUSE").exists():
        raise RuntimeError("EPFL hardware release is paused (PAUSE marker)")
    if phase != "final":
        raise RuntimeError("The hardware release runs finals only; CAL runs under the parent release on L40S")
    if family is not None:
        entry = rel.get("families", {}).get(family)
        PR.require(entry is not None and entry.get("status") == "READY", f"{family} is not READY in the hardware release")
        PR.require(entry["candidates"] == par["families"][family]["candidates"] and entry["schedule"] == par["families"][family]["schedule"],
                   "Family entry differs from the parent release")
        if candidates_fn is not None:
            PR.require(entry["candidates"] == candidates_fn(family), "Candidate grid differs from the frozen definition: " + family)
        if schedule_fn is not None:
            PR.require(entry["schedule"] == schedule_fn(family), "Schedule differs from the frozen definition: " + family)
    return rel


def verify_release_replicate(path, expected_sha, family, seed, *, phase="final", candidates_fn=None, schedule_fn=None, repo: Path = PR.REPO) -> dict:
    """A post-hoc hardware diagnostic release (DIAGNOSTICS: the RTX-vs-L40S replicates EPFL_HW_REPLICATE_20261020 or their L40S-vs-L40S
    control EPFL_HW_CONTROL_20261020): finals only, L40S only, exactly the version's units, the parent's CAL selections (pinned), its own
    results root (never inside the primary one, nor the replicates' root for the control), the primary units pinned by their COMPLETE.json
    sha256 (unchanged; RTX PRO 6000 for the replicates, L40S for the control), science byte-identical to 7ceb2f1e.  Every check except the
    Slurm requirement (the runner adds it)."""
    path = Path(path).resolve()
    PR.require(bool(expected_sha) and PR.sha(path) == expected_sha, "Missing or changed submission-pinned diagnostic release")
    rel = PR.read(path)
    spec = DIAGNOSTICS.get(rel.get("schema"))
    PR.require(spec is not None and rel.get("status") == RELEASE_STATUS_HW, "Not a frozen EPFL hardware diagnostic release")
    PR.require(str(rel.get("label", "")).startswith(REPLICATE_LABEL) and rel.get("role") == spec["role"],
               "The diagnostic release must carry the diagnostic label and its role")
    inv = PR.source_inventory(repo)
    PR.require(rel.get("source_files") == inv and rel.get("code_sha256") == PR.digest(inv), "EPFL source identity changed since the release")
    PR.require(rel.get("numerics_identity") == PR.numerics_identity(repo) == PR.gate_numerics_identity(repo), "Numerics differ from the item-5 gate commit")
    PR.require(rel.get("science_identity") == science_identity(repo), "Science files differ from the release record")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    PR.require(head == rel.get("source_commit"), "The running tree is not the release's source commit")
    pinned = [("parent_release", PARENT_RELEASE_SHA256), ("base_release", HW2_RELEASE_SHA256), ("cal_release", PARENT_RELEASE_SHA256)]
    if spec["controls_for"]:
        pinned.append(("controls_for", spec["controls_for"]))
    for key, want in pinned:
        ref = rel.get(key) or {}
        PR.require(ref.get("sha256") == want and Path(ref.get("path", "")).exists() and PR.sha(ref["path"]) == want, "Pinned release changed: " + key)
    par = PR.read(rel["parent_release"]["path"])
    for key in ("frozen_protocol", "activation_record", "group_record", "data_lock", "item5_gate", "item5_timing", "delta_review"):
        ref = par[key]; p = PR._abs(ref["path"], repo)
        PR.require(p.exists() and PR.sha(p) == ref["sha256"] and rel.get(key) == ref, "Parent pin changed: " + key)
    for key, p in (("hardware_amendment", AMENDMENT), ("xhw_gate_amendment", XHW_AMENDMENT), (spec["record_key"], spec["record"]),
                   (spec["review_key"], spec["review"])):
        ref = rel.get(key) or {}
        PR.require(ref.get("sha256") == PR.sha(p) and PR._abs(ref.get("path", ""), repo) == p, "Pinned input changed: " + key)
    units = spec["units"]
    PR.require(rel.get("phases") == ["FINAL_DEV"] and rel.get("final_gpus") == [CAL_GPU] and rel.get("units") == {k: list(v) for k, v in units.items()}
               and rel.get("partition") == REPLICATE_PARTITION and rel.get("exclude") == REPLICATE_EXCLUDE,
               "Diagnostic rule differs (finals only, L40S only, --exclude=node51,node60, the version's units)")
    primary_root = Path(par["results_dir"]).resolve(); root = Path(rel.get("results_root", "")).resolve()
    PR.require(root == (path.parent / "results").resolve() and root != primary_root and primary_root not in root.parents
               and root not in primary_root.parents, "Diagnostic reruns live in their own results root, never in the primary one")
    if spec["controls_for"]:
        rep_root = Path(PR.read(rel["controls_for"]["path"])["results_root"]).resolve()
        PR.require(root != rep_root and rep_root not in root.parents and root not in rep_root.parents, "The control never shares the replicates' root")
    PR.require(Path(rel.get("cal_results_root", "")).resolve() == primary_root, "CAL selections come from the parent results")
    pairs = rel.get("pairs") or {}
    PR.require(sorted(pairs) == sorted(f"{f}/{s}" for f, ss in units.items() for s in ss), "Pairs differ from the version's units")
    for key, pair in pairs.items():
        f, s = key.split("/")
        want = primary_root / "FINAL_DEV" / f / f"seed_{s}" / "COMPLETE.json"
        gpu_ok = pair["primary_gpu"] == CAL_GPU if spec["primary_on_l40s"] else (pair["primary_gpu"] in FINAL_GPUS and pair["primary_gpu"] != CAL_GPU)
        PR.require(Path(pair["primary_complete"]).resolve() == want and PR.sha(want) == pair["primary_complete_sha256"]
                   and pair["primary_job"] == spec["primary"][key] and gpu_ok, "Primary unit changed or not the pinned final: " + key)
    if (path.parent / "PAUSE").exists():
        raise RuntimeError("EPFL hardware diagnostic release is paused (PAUSE marker)")
    PR.require(phase == "final", "The diagnostic release runs finals only")
    PR.require(family in units and seed in units[family], f"{family} seed {seed} is not a unit of this diagnostic release")
    entry = (rel.get("families") or {}).get(family) or {}
    PR.require(entry.get("status") == "READY" and entry.get("candidates") == par["families"][family]["candidates"]
               and entry.get("schedule") == par["families"][family]["schedule"], "Family entry differs from the parent release")
    pin = (rel.get("selection_pins") or {}).get(family)
    PR.require(bool(pin) and pairs[f"{family}/{seed}"]["selection_dependency_sha256"] == pin,
               "The rerun needs the pinned CAL selection, equal to the primary unit's selection dependency")
    if candidates_fn is not None:
        PR.require(entry["candidates"] == candidates_fn(family), "Candidate grid differs from the frozen definition")
    if schedule_fn is not None:
        PR.require(entry["schedule"] == schedule_fn(family), "Schedule differs from the frozen definition")
    return rel


def require_release_replicate(path, family, seed, **kw):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("EPFL production runs only through Slurm")
    return verify_release_replicate(path, os.environ.get("V10_EPFL_EXPECT_RELEASE"), family, seed, **kw)


def replicate_label(rel: dict, family: str, seed: int) -> dict:
    """The diagnostic label carried in a rerun unit's identity (so the unit can never pass for a primary one)."""
    spec = DIAGNOSTICS[rel["schema"]]; pair = rel["pairs"][f"{family}/{seed}"]
    return {"kind": REPLICATE_LABEL, "role": spec["role"], "record": spec["record"].stem,
            "replicate_of": {"release_sha256": HW2_RELEASE_SHA256, "job": pair["primary_job"], "gpu_name": pair["primary_gpu"],
                             "complete_sha256": pair["primary_complete_sha256"]}}


def require_release_hw(path, family, **kw):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("EPFL production runs only through Slurm")
    return verify_release_hw(path, os.environ.get("V10_EPFL_EXPECT_RELEASE"), family, **kw)


def hardware_record(env: dict) -> dict:
    return {"gpu_name": env.get("gpu_name"), "label": GPU_LABEL.get(env.get("gpu_name")), "node": env.get("slurm_nodelist") or env.get("host"),
            "partition": env.get("slurm_partition"), "slurm_job": env.get("slurm_job_id")}


def unit_states(results_root, families: dict) -> list:
    """Per final unit: COMPLETE / STARTED (a directory without COMPLETE.json: running or interrupted, never listed) / NOT_STARTED, and whether
    the family's CAL selection exists.  Pure filesystem reads."""
    out = []
    for fam, entry in families.items():
        if entry.get("status") != "READY":
            continue
        sel = Path(results_root) / "CAL" / fam / "SELECTION.json"
        for seed in entry["schedule"]["final_seeds"]:
            unit = Path(results_root) / "FINAL_DEV" / fam / f"seed_{seed}"
            state = "COMPLETE" if (unit / "COMPLETE.json").exists() else ("STARTED" if unit.exists() else "NOT_STARTED")
            out.append({"family": fam, "seed": seed, "state": state, "selection": sel.exists(), "unit": str(unit)})
    return out


def json_dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=False)
