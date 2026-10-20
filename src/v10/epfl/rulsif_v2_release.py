"""One-family EPFL RuLSIF_exact_v2 releases (addendum revision 3): T = 1024 primary, T = 2048 capacity sensitivity.  Standard library only.

Rules (v10/epfl/EPFL_RULSIF_V2_RELEASE_RULES_20261020.json):
  source        a descendant of the gate commit c68074a3 (the revision-3 gate tree); every science file byte-identical to c68074a3
  acceptance    the GPU gate 1050041 (L40S, both totals), the T-specific FIT gate (eligible orders) and the T-specific measured timing, all at
                c68074a3 with the GATE_CHECKS record cc20086d and the group record 22434b65
  candidates    the eligible orders of the T-specific FIT gate, in frozen order (A1-eligible rows)
  CAL           L40S only, seed 1017, 8 L40S GPU-h cap (runner-enforced, Slurm StartTime)
  finals        1001-1003 on RTX6000PRO / L40S / H100 (owner 2026-10-19 「final 放开，CAL 留 L40S」, 2026-10-20 「什么快什么来吧」): L40S licensed by
                the GPU gate 1050041; RTX6000PRO / H100 each by a PASS cross-hardware v2 gate (rulsif_v2_gpu_gate.py --xhw) from the RELEASE commit,
                run on that type, under <bundle>/xhw_gates/rulsif_v2_gpu_gate_xhw/<label>/<job>/GATE.json.  The amended acceptance
                EPFL_XHW_GATE_AMENDMENT_20261020 has no identical row in the v2 gate (its checkpoint rows compare the reloaded state bitwise), so the
                v2 gate rows are unchanged.  A final's identity carries its GPU name; COMPLETE units are reused on any GPU type.
"""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
from pathlib import Path

from . import production as PR

RELEASE_SCHEMA, RELEASE_STATUS = "v10.epfl.release_rulsif_v2.v1", "FROZEN_VALIDATED"
FAMILY = "RuLSIF_exact_v2"
GATE_COMMIT = "c68074a30dff7088872a7a1e52e3fdb96c07900c"
GATE_CHECKS_SHA256 = "cc20086d8c7fba4c0a08cd00dbe064b9138da17d97074e10a358f5519da06043"
ACTIVATED_SHA256 = "6fc991bc43118a88322009b104f5e779cc44339578570dd303ece8c6c40c8c2e"
GROUP_SHA256 = "22434b65736051c2e2257ff8d85449f1bb29bacbeb3d1fd9dc98e9e65de28e68"
CENTRES = {1024: "primary", 2048: "capacity_sensitivity"}
CAL_GPU = "NVIDIA L40S"
FINAL_GPUS = ("NVIDIA L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 NVL")
GPU_LABEL = {"NVIDIA L40S": "L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX6000PRO", "NVIDIA H100 NVL": "H100"}
FINAL_PARTITIONS, FINAL_EXCLUDE = "RTX6000PRO,L40S,H100", "node51,node53,node60"
RULES = PR.EPFL_DIR / "EPFL_RULSIF_V2_RELEASE_RULES_20261020.json"
NON_SCIENCE = ("src/v10/epfl/rulsif_v2_release.py",)
RECORD_KEYS = ("activation_gate_checks", "activation_activated", "group_record", "data_lock", "frozen_protocol", "addendum_rev1", "addendum_rev2",
               "addendum_rev3", "preregistration", "amendment_A1", "hardware_amendment", "xhw_gate_amendment", "release_rules")
GATE_PARTS = ("env", "numerics", "family", "capacity2048", "guard")


def git_blob_sha(commit: str, path: str, repo: Path = PR.REPO):
    r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def science_files(repo: Path = PR.REPO) -> list:
    paths = set()
    for rel in ("src/v10", "src/v9"):
        paths.update(str(p.relative_to(repo)) for p in (repo / rel).rglob("*.py") if "__pycache__" not in p.parts)
    for d in ("v10/epfl", "v10/supplement"):
        paths.update(str(p.relative_to(repo)) for p in (repo / d).glob("*.json"))
    return sorted(p for p in paths if p not in NON_SCIENCE and git_blob_sha(GATE_COMMIT, p, repo) is not None)


def science_identity(repo: Path = PR.REPO) -> dict:
    """{path: sha256} of every science file that existed at the gate commit; RuntimeError if any differs from it."""
    out, bad = {}, []
    for f in science_files(repo):
        here, gate = PR.sha(repo / f), git_blob_sha(GATE_COMMIT, f, repo)
        out[f] = here
        if here != gate:
            bad.append(f)
    new_py = [str(p.relative_to(repo)) for rel in ("src/v10", "src/v9") for p in (repo / rel).rglob("*.py")
              if "__pycache__" not in p.parts and str(p.relative_to(repo)) not in NON_SCIENCE and git_blob_sha(GATE_COMMIT, str(p.relative_to(repo)), repo) is None]
    PR.require(not bad and not new_py, f"science code differs from the gate commit {GATE_COMMIT[:8]}: {bad + new_py}")
    return out


def gpu_accepted(phase: str, gpu_name: str) -> bool:
    if phase == "cal":
        return gpu_name == CAL_GPU
    if phase == "final":
        return gpu_name in FINAL_GPUS
    raise ValueError(phase)


def xhw_gate(release: dict, gpu_name: str):
    """None for L40S (the GPU gate 1050041 of the identical science code); for RTX6000PRO / H100 the PASS cross-hardware v2 gate of the release
    commit on that GPU type, else RuntimeError."""
    if gpu_name == CAL_GPU:
        return None
    label = GPU_LABEL[gpu_name]; found = []
    for p in sorted(glob.glob(os.path.join(release["hardware"]["xhw_gates_dir"], "rulsif_v2_gpu_gate_xhw", label, "*", "GATE.json"))):
        g = PR.read(p); pins = g.get("pins", {})
        ok = (g.get("schema") == "v10.epfl.rulsif_v2_gpu_gate.v1" and g.get("status") == "PASS" and g.get("cross_hardware") is True
              and pins.get("commit") == release["source_commit"] and pins.get("record_sha256") == GATE_CHECKS_SHA256
              and pins.get("group_record_sha256") == GROUP_SHA256 and g.get("environment", {}).get("gpu_name") == gpu_name
              and set(g.get("parts", {})) == set(GATE_PARTS) and all(v.get("status") == "PASS" for v in g["parts"].values())
              and set(g.get("data_reads_roles", [])) <= {"FIT", "CAL"} and g.get("dev_read") is False)
        if ok:
            found.append({"path": p, "sha256": PR.sha(p), "job": pins.get("job")})
    PR.require(bool(found), f"No PASS cross-hardware RuLSIF_exact_v2 gate of commit {release['source_commit'][:8]} on {gpu_name}: a final on this "
                            "GPU type waits for it")
    return found[-1]


def verify_release(path, expected_sha, phase, *, candidates_fn=None, schedule_fn=None, repo: Path = PR.REPO) -> dict:
    path = Path(path).resolve()
    PR.require(bool(expected_sha) and PR.sha(path) == expected_sha, "Missing or changed submission-pinned RuLSIF_exact_v2 release")
    rel = PR.read(path)
    PR.require(rel.get("schema") == RELEASE_SCHEMA and rel.get("status") == RELEASE_STATUS and rel.get("family") == FAMILY,
               "Not a frozen EPFL RuLSIF_exact_v2 release")
    inv = PR.source_inventory(repo)
    PR.require(rel.get("source_files") == inv and rel.get("code_sha256") == PR.digest(inv), "Source identity changed since the release")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    PR.require(head == rel.get("source_commit"), "The running tree is not the release's source commit")
    PR.require(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", GATE_COMMIT, head]).returncode == 0,
               "The release source does not descend from the gate commit")
    PR.require(rel.get("gate_commit") == GATE_COMMIT and rel.get("science_identity") == science_identity(repo), "Science differs from the gate commit")
    kd = rel.get("kernel_dictionary") or {}
    T = kd.get("total_centres")
    PR.require(T in CENTRES and kd.get("role") == CENTRES[T], "Total centres must be 1024 (primary) or 2048 (capacity sensitivity)")
    recs = rel.get("records") or {}
    for k in RECORD_KEYS:
        ref = recs.get(k) or {}; p = PR._abs(ref.get("path", ""), repo)
        PR.require(bool(ref.get("path")) and p.exists() and PR.sha(p) == ref.get("sha256"), "Pinned record changed: " + k)
    PR.require(recs["activation_gate_checks"]["sha256"] == GATE_CHECKS_SHA256 and recs["activation_activated"]["sha256"] == ACTIVATED_SHA256
               and recs["group_record"]["sha256"] == GROUP_SHA256, "Activation / group records differ from the pinned ones")
    PR.require(PR.read(PR._abs(recs["activation_activated"]["path"], repo)).get("stage") == "ACTIVATED", "DEV needs the stage-ACTIVATED record")
    acc = rel.get("acceptance") or {}
    for k in ("gpu_gate", "fit_gate", "timing"):
        ref = acc.get(k) or {}
        PR.require(bool(ref.get("path")) and Path(ref["path"]).exists() and PR.sha(ref["path"]) == ref.get("sha256"), "Acceptance record changed: " + k)
    g, f, t = (PR.read(acc[k]["path"]) for k in ("gpu_gate", "fit_gate", "timing"))
    PR.require(g.get("status") == "PASS" and g["pins"]["commit"] == GATE_COMMIT and g["pins"]["record_sha256"] == GATE_CHECKS_SHA256
               and g["pins"].get("group_record_sha256") == GROUP_SHA256 and g["environment"].get("gpu_name") == CAL_GPU
               and all(v["status"] == "PASS" for v in g["parts"].values()) and set(g["parts"]) == set(GATE_PARTS), "GPU gate is not the L40S PASS")
    PR.require(f.get("status") == "PASS" and f["pins"]["commit"] == GATE_COMMIT and f.get("centres_total") == T and f.get("roles_read") == ["FIT"],
               "FIT gate is not the PASS of this centre total")
    PR.require(t.get("status") == "MEASURED" and t["pins"]["commit"] == GATE_COMMIT and t.get("centres_total") == T and t.get("gpu") == CAL_GPU,
               "Timing is not the MEASURED L40S record of this centre total")
    cands = rel.get("candidates") or []
    PR.require([c["order"] for c in cands] == f["eligible_orders"] and 1 <= len(cands) <= PR.MAX_CANDIDATES, "Candidates are not the eligible set")
    if candidates_fn is not None:
        PR.require(cands == [c for c in candidates_fn(FAMILY) if c["order"] in f["eligible_orders"]], "Candidates differ from the frozen rows")
    if schedule_fn is not None:
        PR.require(rel.get("schedule") == schedule_fn(FAMILY), "Schedule differs from the frozen definition")
    PR.require(rel.get("budget") == {"cap_seconds": PR.CAP_SECONDS, "margin_seconds": PR.CAP_MARGIN, "max_candidates": PR.MAX_CANDIDATES},
               "Budget rule differs from decision 4")
    hw = rel.get("hardware") or {}
    PR.require(hw.get("cal_gpu") == CAL_GPU and hw.get("final_gpus") == list(FINAL_GPUS), "Hardware rule differs from the amendment")
    PR.require(Path(rel.get("results_dir", "")) == path.parent / "results", "Results live next to the release")
    if (path.parent / "PAUSE").exists():
        raise RuntimeError("RuLSIF_exact_v2 release paused (PAUSE marker)")
    PR.require(phase in ("cal", "final"), "phase must be cal or final")
    return rel


def require_release(path, phase, **kw):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("EPFL production runs only through Slurm")
    return verify_release(path, os.environ.get("V10_EPFL_EXPECT_RELEASE"), phase, **kw)


def hardware_record(env: dict) -> dict:
    return {"gpu_name": env.get("gpu_name"), "label": GPU_LABEL.get(env.get("gpu_name")), "node": env.get("slurm_nodelist") or env.get("host"),
            "partition": env.get("slurm_partition"), "slurm_job": env.get("slurm_job_id")}
