#!/usr/bin/env python3
"""Hardware release version of the EPFL 10-family production (owner 2026-10-19: 「final 放开，CAL 留 L40S」) and its per-unit job lines.  CPU.

  python -I scripts/v10_epfl/release_hw.py release --source <frozen hw tree> [--bundle <dir>]   # from inside that clean tree
      -> <bundle>/RELEASE.json (schema v10.epfl.release_hw.v1), then the job lines (as `jobs`)
  python -I scripts/v10_epfl/release_hw.py jobs --release <bundle>/RELEASE.json                 # re-runnable at any time
  python -I scripts/v10_epfl/release_hw.py replicate --source <frozen replicate tree> [--bundle <dir>]   # from inside that clean tree
      -> <hw2 bundle>/hw_replicates/RELEASE.json (schema v10.epfl.release_hw_replicate.v1: the post-hoc hardware sensitivity diagnostic,
         EPFL_HW_REPLICATE_20261020), then its two job lines (as `replicate-jobs`)
  python -I scripts/v10_epfl/release_hw.py control --source <frozen control tree> [--bundle <dir>]       # from inside that clean tree
      -> <hw2 bundle>/hw_controls/RELEASE.json (schema v10.epfl.release_hw_control.v1: the L40S-vs-L40S control of the replicates,
         EPFL_HW_CONTROL_20261020), then its two job lines
  python -I scripts/v10_epfl/release_hw.py replicate-jobs --release <hw_replicates or hw_controls>/RELEASE.json   # re-runnable
      -> <bundle>/JOBS.md, JOBS_PLAN.json: the version's units on --partition=L40S --exclude=node51,node60 with the L40S final time
         limits, unless the rerun unit is COMPLETE or started
      -> <bundle>/JOBS.md, <bundle>/JOBS_PLAN.json: one line per final unit that is neither COMPLETE nor started (any release), with the
         L40S time limit, --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60; units whose family CAL is still running are
         listed under the feeder rule (submit after that CAL's exit receipt shows exit_code 0); old pending jobs of the same unit are named

The release consumes the parent's CAL selections: each existing SELECTION.json is verified with the runner's own selection check (CAL
units' COMPLETE hashes, the fixed lexicographic rule, the CAL exit receipt exit 0 under 3dae6434) and pinned by sha256; families whose CAL
is still running are verified the same way by each final at run time.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import hardware as HW, production as PR  # noqa: E402

BUNDLE_DEFAULT = PR.PROJECT / "handoff" / "v10_epfl_production_hw2_20261020"      # amended cross-hardware acceptance (release 7cbd1c21 kept)
PARENT_RELEASE = PR.PROJECT / "handoff" / "v10_epfl_production_20261019" / "RELEASE.json"
HW2_RELEASE = BUNDLE_DEFAULT / "RELEASE.json"
REPLICATE_BUNDLE_DEFAULT = BUNDLE_DEFAULT / "hw_replicates"                          # post-hoc hardware sensitivity diagnostic (own root)
RTX_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
SHORT = {"MF_VCS_perm1": "mfp1", "MF_VCS_M0": "mfm0", "MF_VCS_M1": "mfm1", "MLP_VCS_perm1": "mlp1", "MLP_VCS_perm8": "mlp8",
         "InfoNCE_host[J_perm1]": "nce", "MI_native": "mi", "NMI_native": "nmi", "CoMIR_native": "comir", "MIND_2D_adapted": "mind"}


def run_module():
    spec = importlib.util.spec_from_file_location("epfl_run", REPO / "scripts" / "v10_epfl" / "run.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def verified_selection(results_root: Path, family: str, parent_sha: str, data_sha: str, group_sha: str):
    """The runner's own check of a parent CAL selection (require_exit=True); returns its sha256."""
    from v10.epfl import host as H
    run = run_module()
    ctx = run.Context(None, "cpu", {}, schedule_fn=H.production_schedule, candidates_fn=H.production_candidates)
    sel, sha = run.checked_selection(results_root, family, parent_sha, data_sha, group_sha, ctx, require_exit=True)
    return sel, sha


def build_release(args) -> int:
    src = Path(args.source).resolve()
    PR.require(src == REPO.resolve(), "run release_hw.py release from inside the frozen source tree it releases")
    PR.require(subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() == "",
               "the release source must be a clean committed tree")
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    PR.require(PR.sha(PARENT_RELEASE) == HW.PARENT_RELEASE_SHA256, "parent release changed")
    par = PR.read(PARENT_RELEASE); results_root = Path(par["results_dir"])
    num = PR.numerics_identity(); PR.require(num == PR.gate_numerics_identity(), "numerics differ from the item-5 gate commit")
    sci = HW.science_identity()
    inv = PR.source_inventory()
    families, pins, pending = {}, {}, []
    for fam, e in par["families"].items():
        families[fam] = dict(e)
        if e["status"] != "READY":
            continue
        if (results_root / "CAL" / fam / "SELECTION.json").exists():
            sel, sha = verified_selection(results_root, fam, HW.PARENT_RELEASE_SHA256, par["data_lock"]["sha256"], par["group_record"]["sha256"])
            pins[fam] = sha
            families[fam]["cal_selection"] = {"sha256": sha, "selected": sel["selected"]["candidate"], "cal_job": sel["job"],
                                              "budget_exhausted": sel["budget_exhausted"]}
        else:
            pending.append(fam)
    bundle = Path(args.bundle or BUNDLE_DEFAULT); bundle.mkdir(parents=True, exist_ok=True)
    rel = {"schema": HW.RELEASE_SCHEMA_HW, "status": HW.RELEASE_STATUS_HW, "utc": PR.utc(), "source_root": str(src), "source_commit": commit,
           "source_files": inv, "code_sha256": PR.digest(inv), "numerics_identity": num, "gate_numerics_identity": PR.gate_numerics_identity(),
           "gate_commit": PR.GATE_COMMIT, "science_identity": sci, "parent_source_commit": HW.PARENT_SOURCE_COMMIT,
           "parent_release": {"path": str(PARENT_RELEASE), "sha256": HW.PARENT_RELEASE_SHA256},
           "cal_release": {"path": str(PARENT_RELEASE), "sha256": HW.PARENT_RELEASE_SHA256}, "results_root": str(results_root),
           **{k: par[k] for k in ("frozen_protocol", "activation_record", "group_record", "data_lock", "item5_gate", "item5_timing", "delta_review")},
           "hardware_amendment": {"path": str(HW.AMENDMENT.relative_to(REPO)), "sha256": PR.sha(HW.AMENDMENT)},
           "delta_review_hw": {"path": str(HW.DELTA_REVIEW_HW.relative_to(REPO)), "sha256": PR.sha(HW.DELTA_REVIEW_HW)},
           "xhw_gate_amendment": {"path": str(HW.XHW_AMENDMENT.relative_to(REPO)), "sha256": PR.sha(HW.XHW_AMENDMENT)},
           "supersedes_release": {"sha256": "7cbd1c21ed7e73b6121723730fda9e9575a113b4a3b3436bcbb6783416e92c15",
                                  "change": "cross-hardware acceptance of the two checkpoint energy / gradient rows on RTX6000PRO / H100 only"},
           "phases": ["FINAL_DEV"], "final_gpus": list(HW.FINAL_GPUS), "cal_gpu": HW.CAL_GPU, "final_partitions": HW.FINAL_PARTITIONS,
           "final_exclude": HW.FINAL_EXCLUDE,
           "xhw_gates": {"dir": str(bundle / "xhw_gates"), "required_for": [g for g in HW.FINAL_GPUS if g != HW.CAL_GPU],
                         "criterion": HW.XHW_CRITERION_ID,
                         "rule": "a final on RTX6000PRO / H100 requires a PASS item-5 gate of this source commit on that GPU type under "
                                 "<dir>/<label>/gpu_gate/<job>/GATE.json written by slurm/v10_epfl/gpu_gate_xhw.sbatch (scripts/v10_epfl/"
                                 "gpu_gate_xhw.py: the two checkpoint energy / gradient rows on T01's round-trip tolerance, the state hash "
                                 "bitwise, every other row unchanged); L40S: the unchanged item-5 gate 1048825"},
           "families": families, "selection_pins": pins, "selection_pending": pending,
           "reporting": ["hardware recorded per final", "results by hardware stratum as well as pooled", "no claim of cross-hardware equivalence"],
           "held_roles": par["held_roles"]}
    PR.write(bundle / "RELEASE.json", rel)
    HW.verify_release_hw(bundle / "RELEASE.json", PR.sha(bundle / "RELEASE.json"))
    print(f"hardware release {PR.sha(bundle / 'RELEASE.json')} -> {bundle / 'RELEASE.json'}; pinned selections {sorted(pins)}; pending {pending}")
    return write_jobs(bundle / "RELEASE.json")


def pending_old_jobs():
    """{job name: [job ids]} of this user's queued v10e_fin_* jobs (read-only squeue; {} if the controller does not answer)."""
    r = subprocess.run(["squeue", "-u", "yinwang", "-h", "-o", "%i %j %T"], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].startswith("v10e_fin_"):
            out.setdefault(parts[1], []).append((parts[0], parts[2]))
    return out


def write_jobs(release_path) -> int:
    release_path = Path(release_path).resolve(); rel = PR.read(release_path); rsha = PR.sha(release_path); bundle = release_path.parent
    src = rel["source_root"]; old = pending_old_jobs()
    env = f"V10_EPFL_SOURCE={src} V10_EPFL_RELEASE={release_path} V10_EPFL_EXPECT_RELEASE={rsha}"
    states = HW.unit_states(rel["results_root"], rel["families"])
    now, later, skipped = [], [], []
    for u in states:
        fam, seed = u["family"], u["seed"]
        e = rel["families"][fam]; limit = e["time_limits"]["final"]
        name = f"v10e_finhw_{SHORT[fam]}_{seed}"
        cmd = (f"{env} sbatch --export=ALL --partition={HW.FINAL_PARTITIONS} --exclude={HW.FINAL_EXCLUDE} --job-name={name} --time={limit} "
               f"{src}/slurm/v10_epfl/prod_final_hw.sbatch '{fam}' {seed}")
        swap = old.get(f"v10e_fin_{SHORT[fam]}_{seed}", [])
        row = {"family": fam, "seed": seed, "state": u["state"], "time_limit_l40s": limit, "job_name": name, "command": cmd,
               "old_jobs_same_unit": [{"job": j, "state": s} for j, s in swap]}
        if u["state"] != "NOT_STARTED":
            skipped.append(row)
        elif u["selection"]:
            now.append(row)
        else:
            later.append(row)
    lines = ["# EPFL finals on RTX6000PRO / L40S / H100 (hardware release version)", "",
             f"Release `{release_path}` sha256 `{rsha}`; source `{src}` @ `{rel['source_commit']}`; parent release 3dae6434 (CAL selections, "
             f"results root `{rel['results_root']}`).  Generated {PR.utc()} from the unit directories (re-run `release_hw.py jobs` before swapping).",
             "", "Preconditions: a PASS cross-hardware gate on RTX6000PRO and on H100 (`xhw_gates/<label>/`) before any of these lines run (a final "
             "landing on a GPU type without its PASS gate refuses to start).  Keep the normal-QOS count within its cap.", "",
             "## Submit now (CAL selection verified and pinned; unit neither COMPLETE nor started)", ""]
    for r in now:
        swap = ", ".join(f"{o['job']} ({o['state']})" for o in r["old_jobs_same_unit"])
        lines += [f"- {r['family']} seed {r['seed']}" + (f" - swap: cancel old pending job {swap} first, if it is still PENDING" if swap else ""), "",
                  "```bash", r["command"], "```"]
    lines += ["", "## After the family's CAL exit receipt shows exit_code 0 (feeder rule)", "",
              f"Family CAL still running under 3dae6434: {sorted({r['family'] for r in later})}.  Rule: when `{Path(rel['parent_release']['path']).parent}/exits/<CAL job>.json` "
              "exists with exit_code 0 and SELECTION.json exists, submit the family's lines below (each final re-verifies the selection, its CAL "
              "candidates' COMPLETE hashes and the exit receipt before training).", ""]
    for r in later:
        lines += [f"- {r['family']} seed {r['seed']}", "", "```bash", r["command"], "```"]
    lines += ["", "## Not listed (COMPLETE, or started under any release: running or interrupted)", ""]
    lines += [f"- {r['family']} seed {r['seed']}: {r['state']}" for r in skipped]
    (bundle / "JOBS.md").write_text("\n".join(lines) + "\n")
    PR.write(bundle / "JOBS_PLAN.json", {"release_sha256": rsha, "utc": PR.utc(), "submit_now": now, "after_cal": later, "not_listed": skipped,
                                         "feeder_rule": "submit a family's finals after its CAL exit receipt shows exit_code 0"})
    print(f"{len(now)} lines now, {len(later)} after CAL, {len(skipped)} not listed -> {bundle / 'JOBS.md'}")
    return 0


def changed_since(commit: str, repo: Path = REPO) -> list:
    """Every file changed since a commit (committed or not, untracked included)."""
    diff = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", commit], capture_output=True, text=True).stdout.split()
    new = subprocess.run(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
    return sorted(set(diff) | set(new))


def primary_pair(parent_results: Path, family: str, seed: int, selection_sha: str, selected_candidate: dict, spec: dict) -> dict:
    """The primary final of d9d67cf2 that a diagnostic rerun repeats (RTX PRO 6000 for the replicates, L40S for the control): COMPLETE, the
    pinned job, its identity (release d9d67cf2, same family / seed / candidate / selection dependency, the expected GPU) and its exit receipt
    (exit 0 on that GPU).  Read-only."""
    key = f"{family}/{seed}"; job = spec["primary"][key]; gpu = HW.CAL_GPU if spec["primary_on_l40s"] else RTX_GPU
    unit = parent_results / "FINAL_DEV" / family / f"seed_{seed}"
    c = PR.read(unit / "COMPLETE.json"); idn = c["identity"]
    receipt = PR.read(BUNDLE_DEFAULT / "exits" / f"{job}.json")
    PR.require(c.get("status") == "COMPLETE" and c.get("job") == job and idn["release_sha256"] == HW.HW2_RELEASE_SHA256 and idn["family"] == family
               and idn["seed"] == seed and idn["phase"] == "FINAL_DEV" and idn["candidate"] == selected_candidate
               and idn["selection_dependency_sha256"] == selection_sha and (idn.get("hardware") or {}).get("gpu_name") == gpu
               and (c.get("hardware") or {}).get("gpu_name") == gpu, f"{key}: the primary unit is not the expected {gpu} final")
    PR.require(receipt.get("exit_code") == 0 and receipt.get("gpu_name") == gpu and receipt.get("release_sha256") == HW.HW2_RELEASE_SHA256,
               f"{key}: the primary job lacks a successful {gpu} exit receipt")
    return {"family": family, "seed": seed, "primary_unit": str(unit), "primary_complete": str(unit / "COMPLETE.json"),
            "primary_complete_sha256": PR.sha(unit / "COMPLETE.json"), "primary_dev_sha256": PR.sha(unit / "DEV.jsonl"), "primary_job": job,
            "primary_gpu": gpu, "primary_node": receipt.get("node"), "primary_partition": receipt.get("partition"),
            "primary_release_sha256": HW.HW2_RELEASE_SHA256, "primary_selected_step": c["selected"]["step"],
            "primary_allocated_seconds": receipt.get("allocated_seconds"), "selection_dependency_sha256": selection_sha}


LABELS = {HW.REPLICATE_SCHEMA: ": L40S reruns of the two RTX PRO 6000 finals of d9d67cf2 with the same release science, the same CAL selection "
                               "and the same seed, to measure the hardware effect directly; post hoc (decided after the primary results); "
                               "never a primary unit, never pooled into the primary results",
          HW.CONTROL_SCHEMA: ": L40S-VS-L40S CONTROL of the replicates ca2bc60a: L40S reruns of two L40S finals of d9d67cf2 (the same families, "
                             "seed 1002) with the same release science, the same CAL selection and the same seed, showing the run-to-run "
                             "differences on one GPU type beside the RTX-vs-L40S pairs; post hoc (decided after the primary results); never a "
                             "primary unit, never pooled into the primary results"}
TITLES = {HW.REPLICATE_SCHEMA: "EPFL hardware replicates (post-hoc hardware sensitivity diagnostic)",
          HW.CONTROL_SCHEMA: "EPFL L40S-vs-L40S control of the hardware replicates (post-hoc hardware sensitivity diagnostic)"}
BUNDLES_DEFAULT = {HW.REPLICATE_SCHEMA: REPLICATE_BUNDLE_DEFAULT, HW.CONTROL_SCHEMA: BUNDLE_DEFAULT / "hw_controls"}


def build_diagnostic(args, schema: str) -> int:
    spec = HW.DIAGNOSTICS[schema]
    src = Path(args.source).resolve()
    PR.require(src == REPO.resolve(), "build a diagnostic release from inside the frozen source tree it releases")
    PR.require(subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() == "",
               "the release source must be a clean committed tree")
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    PR.require(PR.sha(PARENT_RELEASE) == HW.PARENT_RELEASE_SHA256 and PR.sha(HW2_RELEASE) == HW.HW2_RELEASE_SHA256, "parent or base release changed")
    par = PR.read(PARENT_RELEASE); parent_results = Path(par["results_dir"])
    num = PR.numerics_identity(); PR.require(num == PR.gate_numerics_identity(), "numerics differ from the item-5 gate commit")
    sci = HW.science_identity(); inv = PR.source_inventory()
    review = PR.read(spec["review"])
    unlisted = [f for f in changed_since(spec["base_source_commit"]) if f not in review["changed_files_since_base_source"]]
    PR.require(not unlisted, f"the delta review does not cover: {unlisted}")
    PR.require(review["science_identity"] == {f: v[0] for f, v in sci.items()} and review["science_files_byte_identical"] is True,
               "the delta review's science identity differs from this tree")
    record = PR.read(spec["record"])
    PR.require(record.get("id") == spec["record"].stem and record.get("units") == {k: list(v) for k, v in spec["units"].items()},
               "the record does not name the version's units")
    controls = {}
    if spec["controls_for"]:
        cpath = REPLICATE_BUNDLE_DEFAULT / "RELEASE.json"
        PR.require(PR.sha(cpath) == spec["controls_for"], "the replicate release ca2bc60a changed")
        controls = {"controls_for": {"path": str(cpath), "sha256": spec["controls_for"]}}
        PR.require(review.get("science_identity_equal_to_replicate_release") is True, "the control's science differs from the replicates'")
    bundle = Path(args.bundle or BUNDLES_DEFAULT[schema])
    PR.require(not (bundle / "results").exists(), f"{bundle}/results exists: a rerun unit has started, the release is not rebuilt")
    data_sha, group_sha = par["data_lock"]["sha256"], par["group_record"]["sha256"]
    families, pins, pairs = {}, {}, {}
    for fam, seeds in spec["units"].items():
        e = par["families"][fam]
        PR.require(e["status"] == "READY", f"{fam} is not READY in the parent release")
        sel, ssha = verified_selection(parent_results, fam, HW.PARENT_RELEASE_SHA256, data_sha, group_sha)
        pins[fam] = ssha
        families[fam] = {**e, "cal_selection": {"sha256": ssha, "selected": sel["selected"]["candidate"], "cal_job": sel["job"],
                                                "budget_exhausted": sel["budget_exhausted"]}}
        for s in seeds:
            pairs[f"{fam}/{s}"] = {**primary_pair(parent_results, fam, s, ssha, sel["selected"]["candidate"], spec),
                                   "replicate_unit": str(bundle / "results" / "FINAL_DEV" / fam / f"seed_{s}"), "replicate_gpu": HW.CAL_GPU,
                                   "time_limit_l40s": e["time_limits"]["final"]}
    bundle.mkdir(parents=True, exist_ok=True)
    rel = {"schema": schema, "status": HW.RELEASE_STATUS_HW, "role": spec["role"], "label": HW.REPLICATE_LABEL + LABELS[schema],
           "owner_rules_verbatim": {"2026-10-19": "final 放开，CAL 留 L40S", "2026-10-20": "什么快什么来吧"},
           "utc": PR.utc(), "source_root": str(src), "source_commit": commit, "source_files": inv, "code_sha256": PR.digest(inv),
           "numerics_identity": num, "gate_numerics_identity": PR.gate_numerics_identity(), "gate_commit": PR.GATE_COMMIT, "science_identity": sci,
           "parent_source_commit": HW.PARENT_SOURCE_COMMIT, "base_source_commit": spec["base_source_commit"],
           "parent_release": {"path": str(PARENT_RELEASE), "sha256": HW.PARENT_RELEASE_SHA256},
           "base_release": {"path": str(HW2_RELEASE), "sha256": HW.HW2_RELEASE_SHA256}, **controls,
           "cal_release": {"path": str(PARENT_RELEASE), "sha256": HW.PARENT_RELEASE_SHA256},
           "results_root": str(bundle / "results"), "cal_results_root": str(parent_results),
           **{k: par[k] for k in ("frozen_protocol", "activation_record", "group_record", "data_lock", "item5_gate", "item5_timing", "delta_review")},
           "hardware_amendment": {"path": str(HW.AMENDMENT.relative_to(REPO)), "sha256": PR.sha(HW.AMENDMENT)},
           "xhw_gate_amendment": {"path": str(HW.XHW_AMENDMENT.relative_to(REPO)), "sha256": PR.sha(HW.XHW_AMENDMENT)},
           spec["record_key"]: {"path": str(spec["record"].relative_to(REPO)), "sha256": PR.sha(spec["record"])},
           spec["review_key"]: {"path": str(spec["review"].relative_to(REPO)), "sha256": PR.sha(spec["review"])},
           "phases": ["FINAL_DEV"], "final_gpus": [HW.CAL_GPU], "partition": HW.REPLICATE_PARTITION, "exclude": HW.REPLICATE_EXCLUDE,
           "units": {k: list(v) for k, v in spec["units"].items()}, "families": families, "selection_pins": pins, "pairs": pairs,
           "reporting": ["per-registration differences against the primary unit", "unit-macro differences (rerun - primary) with paired unit "
                         "bootstrap intervals", "labelled post-hoc hardware sensitivity diagnostic beside, never inside, the primary results",
                         "no claim of cross-hardware equivalence"],
           "held_roles": par["held_roles"]}
    PR.write(bundle / "RELEASE.json", rel)
    from v10.epfl import host as H
    for fam, seeds in spec["units"].items():
        for s in seeds:
            HW.verify_release_replicate(bundle / "RELEASE.json", PR.sha(bundle / "RELEASE.json"), fam, s, candidates_fn=H.production_candidates,
                                        schedule_fn=H.production_schedule)
    print(f"{spec['role']} release {PR.sha(bundle / 'RELEASE.json')} -> {bundle / 'RELEASE.json'}; pairs {sorted(pairs)}")
    return write_replicate_jobs(bundle / "RELEASE.json")


def write_replicate_jobs(release_path) -> int:
    release_path = Path(release_path).resolve(); rel = PR.read(release_path); rsha = PR.sha(release_path); bundle = release_path.parent
    spec = HW.DIAGNOSTICS[rel["schema"]]; src = rel["source_root"]
    env = f"V10_EPFL_SOURCE={src} V10_EPFL_RELEASE={release_path} V10_EPFL_EXPECT_RELEASE={rsha}"
    now, skipped = [], []
    for key, pair in rel["pairs"].items():
        fam, seed = pair["family"], pair["seed"]
        unit = Path(rel["results_root"]) / "FINAL_DEV" / fam / f"seed_{seed}"
        state = "COMPLETE" if (unit / "COMPLETE.json").exists() else ("STARTED" if unit.exists() else "NOT_STARTED")
        name = f"{spec['job_prefix']}{SHORT[fam]}_{seed}"; limit = pair["time_limit_l40s"]
        cmd = (f"{env} sbatch --export=ALL --partition={rel['partition']} --exclude={rel['exclude']} --job-name={name} --time={limit} "
               f"{src}/slurm/v10_epfl/prod_final_replicate.sbatch '{fam}' {seed}")
        row = {"family": fam, "seed": seed, "state": state, "time_limit_l40s": limit, "job_name": name, "command": cmd,
               "primary_job": pair["primary_job"], "primary_gpu": pair["primary_gpu"]}
        (now if state == "NOT_STARTED" else skipped).append(row)
    gpu_short = lambda g: g.replace("NVIDIA ", "").replace(" Blackwell Server Edition", "")   # noqa: E731
    lines = [f"# {TITLES[rel['schema']]}", "",
             f"Release `{release_path}` sha256 `{rsha}`; source `{src}` @ `{rel['source_commit']}`; base release d9d67cf2 (the primary finals), "
             f"CAL selections of 3dae6434 read from `{rel['cal_results_root']}`; own results root `{rel['results_root']}`"
             + (f"; control of the replicates `{rel['controls_for']['sha256'][:8]}`" if rel.get("controls_for") else "")
             + f".  Generated {PR.utc()} from the rerun unit directories.", "",
             "L40S only (`--partition=L40S --exclude=node51,node60`), the L40S final time limits.  Owner rules 「final 放开，CAL 留 L40S」 and "
             "「什么快什么来吧」.  Each job re-verifies the pinned CAL selection (CAL candidates' COMPLETE hashes, the CAL exit receipt) and the "
             "pinned primary COMPLETE.json hashes before training; it refuses any GPU other than NVIDIA L40S.  Keep the normal-QOS count within "
             "its cap.", "", "## Submit (rerun unit neither COMPLETE nor started)", ""]
    for r in now:
        lines += [f"- {r['family']} seed {r['seed']} (repeats {r['primary_job']} on {gpu_short(r['primary_gpu'])})", "", "```bash", r["command"], "```"]
    lines += ["", "## Not listed (COMPLETE, or started: running or interrupted)", ""]
    lines += [f"- {r['family']} seed {r['seed']}: {r['state']}" for r in skipped] or ["- none"]
    (bundle / "JOBS.md").write_text("\n".join(lines) + "\n")
    PR.write(bundle / "JOBS_PLAN.json", {"release_sha256": rsha, "utc": PR.utc(), "submit_now": now, "not_listed": skipped})
    print(f"{len(now)} rerun lines, {len(skipped)} not listed -> {bundle / 'JOBS.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("release"); r.add_argument("--source", required=True); r.add_argument("--bundle")
    j = sub.add_parser("jobs"); j.add_argument("--release", required=True)
    rr = sub.add_parser("replicate"); rr.add_argument("--source", required=True); rr.add_argument("--bundle")
    rc = sub.add_parser("control"); rc.add_argument("--source", required=True); rc.add_argument("--bundle")
    rj = sub.add_parser("replicate-jobs"); rj.add_argument("--release", required=True)
    a = ap.parse_args()
    return {"release": build_release, "jobs": lambda x: write_jobs(x.release), "replicate": lambda x: build_diagnostic(x, HW.REPLICATE_SCHEMA),
            "control": lambda x: build_diagnostic(x, HW.CONTROL_SCHEMA), "replicate-jobs": lambda x: write_replicate_jobs(x.release)}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
