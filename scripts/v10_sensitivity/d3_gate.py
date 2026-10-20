"""D3 post-DEV diagnostic: the supplement's own GPU gate on RTX PRO 6000 / H100 (pre-registered).

Slurm RTX6000PRO or H100:  d3_gate.py FAMILY             FAMILY: MLP_VCS_perm8 | InfoNCE_host[J_perm1]
Slurm cpu-high:            d3_gate.py FAMILY --dry-run   binds/patches/verifies the manifest, never runs the gate

Runs the frozen pipeline tree's scripts/v10_supplement/validate.py `gate --family FAMILY`
unchanged, against a byte copy of its validation MANIFEST in the new bundle (so the gate
writes only <bundle>/D3/validation/runs/<job>). The one change, in this process only:
run.runtime_environment's L40S-name condition becomes the D3 partition/GPU pool; every
other runtime and precision check is unchanged (v10_sensitivity.hardware).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import traceback

DIAG = Path(__file__).resolve().parents[2]
sys.path.append(str(DIAG / "src"))  # appended: the frozen tree is put first by frozen.import_runner

from v10_sensitivity import common as C  # noqa: E402
from v10_sensitivity import frozen as Z  # noqa: E402
from v10_sensitivity import hardware as H  # noqa: E402

FAMILIES = ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]")
SCOPE = "D3 post-DEV cross-hardware diagnostic of the supplement GPU gate (pre-registered V10-POSTDEV-SENSITIVITY-20261019)"


def numerics_line(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def delta(new, old):
    if isinstance(new, (int, float)) and isinstance(old, (int, float)) and not isinstance(new, bool):
        return new - old
    if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
        return [delta(a, b) for a, b in zip(new, old)]
    return None


def compare(new, accepted):
    keys = {"resume.first.loss": ("resume", "first", "loss"),
            "resume.first.grad_norm_preclip": ("resume", "first", "grad_norm_preclip"),
            "resume.expected_second.loss": ("resume", "expected_second", "loss"),
            "resume.resumed_second.loss": ("resume", "resumed_second", "loss"),
            "resume.expected_second.grad_norm_preclip": ("resume", "expected_second", "grad_norm_preclip"),
            "full_search.F_hat": ("full_search_timing", "F_hat"),
            "full_search.theta_hat_deg": ("full_search_timing", "theta_hat_deg"),
            "full_search.t_hat": ("full_search_timing", "t_hat"),
            "full_search.seconds": ("full_search_timing", "seconds"),
            "geometry_gradient": ("geometry_gradient",), "finite_difference": ("finite_difference",),
            "peak_gpu_gib": ("peak_gpu_gib",)}
    out = {}
    for name, path in keys.items():
        a, b = new, accepted
        for key in path:
            a, b = a[key], b[key]
        out[name] = {"this_gpu": a, "accepted_l40s": b, "difference": delta(a, b)}
    out["resume_bitwise_equal_this_gpu"] = new["resume"]["expected_second"] == new["resume"]["resumed_second"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID")
    C.require(bool(job), "D3 runs only inside Slurm")
    pre = C.prereg()
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if not args.dry_run:
        C.verify_freeze()
        C.require_partition("D3", partition)
    mref = pre["frozen_inputs"]["pipeline_validation_manifest"]
    C.verify_manifest(mref)
    copy = C.BUNDLE / "D3" / "validation" / "MANIFEST.json"
    C.require(copy.exists() and C.sha(copy) == mref["sha256"], "Bundle manifest is not a byte copy of the frozen one")
    C.require(Path(os.environ.get("V10_VALIDATION_MANIFEST", "")).resolve() == copy.resolve()
              and os.environ.get("V10_VALIDATION_SHA256") == mref["sha256"],
              "V10_VALIDATION_MANIFEST/V10_VALIDATION_SHA256 must name the bundle copy")
    summaries = C.BUNDLE / "D3" / "summaries"
    if not args.dry_run and os.environ.get("V10D_D3_ALLOW_REPEAT") != "1":
        for previous in sorted(summaries.glob("*.json")) if summaries.exists() else []:
            record = C.read(previous)
            C.require(not (record.get("family") == args.family and record.get("partition") == partition
                           and record.get("status") == "PASS"),
                      f"A PASS for {args.family} on {partition} already exists ({previous.name}); set V10D_D3_ALLOW_REPEAT=1 to repeat")
    source_root = Path(mref["source_root"])
    R = Z.import_runner(source_root, "run")  # validate.gate() does `import run as R`
    original = R.runtime_environment
    R.runtime_environment = H.amended_runtime_environment(R, scope=SCOPE)
    V = Z.import_script(source_root, "scripts/v10_supplement/validate.py", "v10_frozen_validate")
    C.require(Path(V.__file__).resolve().is_relative_to(source_root) and sys.modules["run"] is R,
              "validate.py would not bind the patched frozen runner")
    if args.dry_run:
        pin, manifest = V.verify_manifest()
        Z.initialize(R, source_root)
        report = {"status": "DRY_RUN_PASS", "family": args.family, "partition": partition, "label": C.LABEL,
                  "manifest_copy": str(pin), "manifest_sha256": C.sha(pin), "would_write": str(pin.parent / "runs" / job),
                  "patched": getattr(R.runtime_environment, "v10d_amended", False),
                  "original_function": f"{original.__module__}.{original.__qualname__}", "utc": C.utc(), "job": job}
        C.write(C.OUT_ROOT / "dryrun" / job / "D3.json", report)
        print("D3_DRY_RUN_PASS", report, flush=True)
        return
    run_dir = copy.parent / "runs" / job
    status, error = "FAIL", None
    try:
        sys.argv = [str(source_root / "scripts/v10_supplement/validate.py"), "gate", "--family", args.family]
        V.main()
        top, gate = C.read(run_dir / "GATE.json"), C.read(run_dir / args.family / "GATE.json")
        environment = gate["environment"]
        C.require(top["status"] == "PASS" and gate["status"] == "PASS", "Gate did not PASS")
        C.require(environment.get("hardware_amendment", {}).get("scope") == SCOPE
                  and environment["slurm_partition"] == partition
                  and environment["gpu_name"].startswith(H.D3_POOL[partition]), "Gate environment is not the D3 hardware")
        status = "PASS"
    except BaseException as exc:  # the gate itself wrote runs/<job>/FAILURE.json
        error = {"error": repr(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        accepted_ref = mref["accepted_l40s_gates"][args.family]
        summary = {"schema": "v10.sensitivity.d3.summary.v1", "status": status, "family": args.family,
                   "partition": partition, "job": job, "label": C.LABEL, "scope": SCOPE, "utc": C.utc(),
                   "gate_dir": str(run_dir), "error": error}
        if status == "PASS":
            C.require(C.sha(accepted_ref["path"]) == accepted_ref["sha256"], "Accepted L40S gate changed")
            accepted = C.read(accepted_ref["path"])
            summary.update(gpu_name=environment["gpu_name"], gpu_capability=environment["gpu_capability"],
                           host=environment["host"], cuda_numerics=numerics_line(run_dir / "CUDA_NUMERICS.txt"),
                           accepted_l40s_job=accepted_ref["job"],
                           comparison=compare(gate, accepted),
                           gate_sha256=C.sha(run_dir / args.family / "GATE.json"))
        C.write(summaries / f"{job}.json", summary)


def interrupted(signum, frame):
    raise InterruptedError(f"Slurm signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    main()
