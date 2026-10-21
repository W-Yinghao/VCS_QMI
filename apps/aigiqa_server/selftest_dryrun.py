"""Dry-run of the server wrappers on a synthetic AGIQA-shaped dataset (login-node safe: file and protocol operations only).

Exercises: stage_data functions (owner/CMIM byte identity + header semantics, image root, image decode, package prepare,
post-prepare disjointness, seal), make_matrix base/register (package make_grid + package register + list-jobs),
resolve_job (unique stem, refusal once a receipt exists) and the lock check. Everything is written under --root.
Synthetic files cannot match the pinned real hashes; those checks are expected to fail here and are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_data  # noqa: E402

WORDS = ["amber", "beacon", "cactus", "dahlia", "ember", "fjord", "galaxy", "harbor", "indigo", "jungle", "koala", "lotus", "maple",
         "nebula", "opal", "prairie", "quartz", "raven", "saffron", "tundra", "umber", "violet", "walnut", "xenon", "yarrow", "zephyr",
         "acorn", "basalt", "cobalt", "dune", "estuary", "fennel", "glacier", "heron", "ivory", "jasper", "kelp", "lagoon", "mesa", "nectar"]


def synth(root: Path):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(7)
    img_dir = root / "data/agiqa_3k/images_official/AGIQA-3K"
    img_dir.mkdir(parents=True)
    rows = []
    for i, w in enumerate(WORDS):
        for j in range(3):
            name = f"synth{j}_{w}_{i:03d}.jpg"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_dir / name, quality=90)
            rows.append([name, f"a photo of a {w} in style {j}", "adj1", "adj2", "style", f"{1 + j + .01 * i:.4f}", "0.5", f"{1 + .7 * j + .02 * i:.4f}", "0.5"])
    header = ["name", "prompt", "adj1", "adj2", "style", "mos_quality", "std_quality", "mos_align", "std_align"]
    owner = root / f"data/agiqa_3k/sources/{stage_data.OWNER_DIR}/data.csv"
    cmim = root / f"data/agiqa_3k/sources/{stage_data.CMIM_DIR}/Database/AGIQA-3k/full.csv"
    for p in (owner, cmim):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
    models = root / "models/imagereward"
    models.mkdir(parents=True)
    (models / "ImageReward.pt").write_bytes(b"synthetic placeholder, not a model")
    (models / "med_config.json").write_text('{"synthetic": true}\n')
    return owner, cmim, img_dir, models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    owner, cmim, img_dir, models = synth(root)
    report = {"expected_fail_with_synthetic_files": [], "must_pass": []}
    c = stage_data.Checks()
    mapping, names = stage_data.owner_mos_mapping(c, owner, cmim)
    image_root = stage_data.find_image_root(c, root / "data/agiqa_3k/images_official", names)
    report["images"] = stage_data.validate_images(c, image_root, names, HERE / "pins/hf_strawhat_agiqa-3k_bf9b7bb.json", root / "inventory.csv")
    prepared = root / "data/agiqa_3k/prepared/AGIQA-3k"
    prep = stage_data.run_prepare(c, HERE.parent / "VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019", cmim, image_root, prepared)
    post = stage_data.post_prepare(c, prepared) if prep["returncode"] == 0 else None
    report["sha256sums_files"] = stage_data.sha256sums(root / "data/agiqa_3k", root / "data/agiqa_3k/SHA256SUMS", {"sealed", "zips"})
    sealed = stage_data.seal(prepared, image_root)
    pinned = {"owner_vs_cmim_byte_identical", "adapter_columns_match_owner_readme_positions", "unique_image_root_holds_every_csv_name",
              "all_images_decode", "no_extra_files_in_image_root", "package_prepare_exit0", "test_inputs_label_free"}
    for item in c.items:
        name = item["check"]
        if name in pinned or name.startswith("disjoint_"):
            report["must_pass"].append({name: item["pass"]})
        else:
            report["expected_fail_with_synthetic_files"].append({name: item["pass"]})
    # fake staging record (synthetic) -> make_matrix base
    staging = {"status": "PASS", "dry_run": False, "synthetic_selftest": True,
               "model_assets": {"ImageReward.pt": {"path": str(models / "ImageReward.pt"), "sha256": stage_data.sha256(models / "ImageReward.pt")},
                                "med_config.json": {"path": str(models / "med_config.json"), "sha256": stage_data.sha256(models / "med_config.json")}}}
    (root / "data/agiqa_3k/STAGING_RECORD.json").write_text(json.dumps(staging))
    real = {k: v for k, v in (line.split("=", 1) for line in (HERE / "paths.env").read_text().splitlines() if "=" in line and not line.startswith("#"))}
    penv = root / "paths.env"
    penv.write_text("\n".join([f"AIG_REPO={real['AIG_REPO']}", f"AIG_PKG={HERE.parent / 'VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019'}",
                               f"AIG_SERVER={HERE}", f"AIG_EVIDENCE={root / 'evidence'}", f"AIG_PY={sys.executable}",
                               f"AIG_DATA={root / 'data/agiqa_3k'}", f"AIG_PREPARED={prepared}", f"AIG_MODELS={models}",
                               f"AIG_OUT={root / 'out'}", f"AIG_PROTOCOL={root / 'out/protocol'}", f"AIG_LOCK={root / 'out/protocol/aigiqa_alignment_lock.json'}",
                               f"AIG_LOGS={root / 'logs'}"]) + "\n")
    env = {**os.environ, "AIG_PATHS_ENV": str(penv)}
    base = subprocess.run([sys.executable, str(HERE / "make_matrix.py"), "base"], env=env, capture_output=True, text=True)
    report["make_matrix_base_rc"] = base.returncode
    lock_torch = json.loads((HERE / "ENV_LOCK.json").read_text())["torch"]["version"]
    fam = {"timed_fit_epoch": {"seconds": 60.0}, "cal_sized_inference": {"seconds": 10.0}, "model_build_seconds_incl_sha256": 30.0,
           "checkpoint_probe": {"bytes": 1.84e9, "save_seconds_target_fs": 15.0}}
    g0 = {"status": "PASS", "slurm": {"SLURM_JOB_PARTITION": "L40S"}, "runtime": {"torch": lock_torch},
          "families": {f: dict(fam) for f in ["task_only", "reference_nce_text", "bce_text", "vcs_exact_text", "vcs_sampled8_text", "corrected_cmim_both"]}}
    g0["families"]["corrected_cmim_both"]["gaussian_warmup_epoch"] = {"seconds": 25.0}
    (root / "fake_G0_RESULT.json").write_text(json.dumps(g0))
    reg = subprocess.run([sys.executable, str(HERE / "make_matrix.py"), "register", "--g0", str(root / "fake_G0_RESULT.json")],
                         env=env, capture_output=True, text=True)
    report["make_matrix_register_rc"] = reg.returncode
    report["make_matrix_register_stdout"] = reg.stdout[-1500:]
    report["make_matrix_register_stderr"] = reg.stderr[-1500:]
    lock = root / "out/protocol/aigiqa_alignment_lock.json"
    pyenv = {**env, "PYTHONPATH": f"{HERE.parent / 'VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019'}:{HERE.parent / 'VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019' / 'shared'}"}
    if reg.returncode == 0:
        chk = subprocess.run([sys.executable, "-m", "vcs_common.experiment", "check", "--lock", str(lock)], env=pyenv, capture_output=True, text=True)
        report["lock_check"] = chk.stdout.strip()
        r1 = subprocess.run([sys.executable, str(HERE / "resolve_job.py"), "--lock", str(lock), "--name", "vcs_exact_text_w0.1_s3101"], env=pyenv, capture_output=True, text=True)
        report["resolve_ok"] = {"rc": r1.returncode, "job_id": r1.stdout.strip()}
        lk = json.loads(lock.read_text())
        job = next(j for j in lk["jobs"] if j["job_id"] == r1.stdout.strip())
        rp = lock.parent / (lock.stem + ".state") / "jobs" / (job["job_id"] + ".json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({"status": "RUNNING", "synthetic_selftest": True}))
        r2 = subprocess.run([sys.executable, str(HERE / "resolve_job.py"), "--lock", str(lock), "--name", "vcs_exact_text_w0.1_s3101"], env=pyenv, capture_output=True, text=True)
        report["resolve_refuses_after_receipt"] = {"rc": r2.returncode, "stderr": r2.stderr.strip()[-300:]}
        lines = (root / "out/protocol/SUBMIT_LINES.sh").read_text().splitlines()
        report["submit_lines"] = len([l for l in lines if l.startswith("sbatch ")])
        report["first_line"] = next(l for l in lines if l.startswith("sbatch "))
        report["lock_jobs"] = len(lk["jobs"]); report["lock_scope"] = lk["scope"]; report["lock_launch_cwd"] = lk["launch_cwd"]
        report["lock_family_budgets"] = lk["family_budgets"]; report["lock_matched_groups"] = lk["matched_family_groups"]
    report["prepare_counts"] = (post or {}).get("image_counts"), (post or {}).get("split_group_counts")
    report["seal"] = sealed
    ok = (all(list(d.values())[0] for d in report["must_pass"]) and report["make_matrix_base_rc"] == 0 and report["make_matrix_register_rc"] == 0
          and report.get("resolve_ok", {}).get("rc") == 0 and report.get("resolve_refuses_after_receipt", {}).get("rc") == 5
          and report.get("submit_lines") == 78 and report.get("lock_jobs") == 78)
    report["status"] = "PASS" if ok else "FAIL"
    (root / "SELFTEST.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1)[:6000])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
