"""Write the r22 configs and JOBS csv files (R22_ALTERNATIVES_CN.md D and E). No scheduler calls.

KD stage 1 (S1): lambda selection on CAL seed 27182 (the previous round's CAL seed) for the two VCS-CRD arms, plus the
added matched control crd_uniform (official CRD NCE, uniform Q, weight 0.8) at the CAL seed and the 3 final seeds.
KD stage 2 (S2): the two VCS-CRD arms at their selected lambda, final seeds 31415/16180/57721 (written by select_kd.py).
AIG: 4 arms x seeds 4101/4102/4103, lambda 0.1, alternating_target_sg, 20 epochs.
Smoke: 2-epoch versions (KD vcscrd/dkd_vcscrd at lambda 10; AIG banked exact+MOS and bce+MOS with the aux weight on
from epoch 1) under outputs_apps/r22/smoke.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS = Path("/home/infres/yinwang/CS_QMI")
OUT = CS / "outputs_apps" / "r22"
KD_PRIOR = CS / "outputs_apps" / "kd_20261020" / "protocol" / "kd_final_grid"
AIG_BASE = (CS / "ssl_pilot_app_round21" / "apps" / "round_20261021" / "package" / "configs" / "aigiqa"
            / "r1_alternating_target_sg_exact__seed4101.json")
KD_CAL_SEED, KD_FINAL_SEEDS, AIG_SEEDS = 27182, (31415, 16180, 57721), (4101, 4102, 4103)
KD_LAMBDAS = (0.8, 3, 10)
FIELDS = ["app", "arm", "lam", "seed", "epochs", "job_name", "config", "config_sha256", "run_dir"]
NOTE = ("r22 (2026-10-22, R22_ALTERNATIVES_CN.md D): crd_original/dkd_crd_uniform recipe of the previous KD round; "
        "only the CRD criterion is replaced (objective vcs) or, for crd_uniform, only the negative mode. PILOT/DEVELOPMENT.")


def lam_tag(lam):
    return f"{lam:g}"


def write(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kd_config(arm, lam, seed, epochs=240):
    base = "dkd_crd_uniform" if arm == "dkd_vcscrd" else "crd_original"
    cfg = json.loads((KD_PRIOR / f"{base}_seed31415.json").read_text())
    cfg["seed"] = seed
    cfg["name"] = f"r22_{arm}_l{lam_tag(lam)}"
    crd = cfg["loss"]["crd"]
    crd["negative_mode"] = "uniform_other_instances"          # product of marginals (already so for dkd_crd_uniform)
    if arm in ("vcscrd", "dkd_vcscrd"):
        crd["objective"] = "vcs"                              # tau = crd temperature 0.07, fixed
    elif arm != "crd_uniform":
        raise ValueError(arm)
    cfg["loss"]["crd_weight"] = float(lam)
    cfg["training"]["epochs"] = epochs
    cfg["notes"] = list(cfg.get("notes", [])) + [NOTE]
    return cfg


# arm -> (method, bank size, P weighting, aux weight lambda).  E2 (2026-10-23, R22_ALTERNATIVES_CN.md §E2): bank + uniform P at the
# round-21 lambda 0.01, after E showed lambda 0.1 itself costs ~0.019 SRCC while the bank recovers ~0.009 of it.
AIG_ARMS = {"exact_bank": ("vcs_exact", 4096, "uniform", 0.1),
            "exact_bank_mosw": ("vcs_exact", 4096, "mos", 0.1),
            "bce_bank_mosw": ("bce", 4096, "mos", 0.1),
            "exact_inbatch": ("vcs_exact", None, None, 0.1),
            "exact_bank_l001": ("vcs_exact", 4096, "uniform", 0.01),
            "bce_bank_l001": ("bce", 4096, "uniform", 0.01)}


def aig_config(arm, seed, run_dir, epochs=20, smoke=False):
    cfg = json.loads(AIG_BASE.read_text())
    cfg["seed"], cfg["output_dir"], cfg["epochs"] = seed, str(run_dir), epochs
    cfg["method_label"] = f"r22_{arm}"
    cfg["evidence_role"] = "R22_PILOT_DEVELOPMENT_FIT_CAL_ONLY_NO_TEST"
    loss = cfg["loss"]
    assert loss["application_update"]["mode"] == "alternating_target_sg"
    method, bank, weighting = AIG_ARMS[arm][:3]
    loss["weight"] = AIG_ARMS[arm][3]
    loss["method"] = method
    if bank is not None:                                       # exact_inbatch = package path (in-batch U), lambda 0.1
        loss["bank_size"], loss["p_weighting"] = bank, weighting
    if smoke:
        loss["application_update"].update(warmup_epochs=0, ramp_epochs=1)
        cfg["evidence_role"] = "R22_GPU_SMOKE_ONLY"
    return cfg


def rows_to_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def kd_row(arm, lam, seed, cdir, rdir, epochs=240, prefix="r22_kd"):
    stem = f"{arm}_l{lam_tag(lam)}_s{seed}"
    path = cdir / f"{stem}.json"
    sha = write(kd_config(arm, lam, seed, epochs), path)
    return dict(app="kd", arm=arm, lam=lam_tag(lam), seed=seed, epochs=epochs, job_name=f"{prefix}_{stem}",
                config=path.name, config_sha256=sha, run_dir=str(rdir / stem))


def aig_row(arm, seed, cdir, rdir, epochs=20, smoke=False, prefix="r22_aig"):
    stem = f"{arm}_s{seed}"
    path = cdir / f"{stem}.json"
    sha = write(aig_config(arm, seed, rdir / stem, epochs, smoke), path)
    return dict(app="aigiqa", arm=arm, lam=lam_tag(AIG_ARMS[arm][3]) if AIG_ARMS[arm][3] != 0.1 else "0.1", seed=seed, epochs=epochs, job_name=f"{prefix}_{stem}",
                config=path.name, config_sha256=sha, run_dir=str(rdir / stem))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=["smoke", "kd_s1", "kd_s2", "kd_s2_all", "aig", "aig_e2"], required=True)
    p.add_argument("--vcscrd-lambda", type=float)
    p.add_argument("--dkd-vcscrd-lambda", type=float)
    a = p.parse_args()
    cfg_root = HERE / "configs"
    if a.part == "smoke":
        cdir = cfg_root / "smoke"
        rows = [kd_row("vcscrd", 10, KD_CAL_SEED, cdir, OUT / "smoke" / "kd", epochs=2, prefix="r22_kd_chk"),
                kd_row("dkd_vcscrd", 10, KD_CAL_SEED, cdir, OUT / "smoke" / "kd", epochs=2, prefix="r22_kd_chk"),
                aig_row("exact_bank_mosw", 4101, cdir, OUT / "smoke" / "aigiqa", epochs=2, smoke=True, prefix="r22_aig_chk"),
                aig_row("bce_bank_mosw", 4101, cdir, OUT / "smoke" / "aigiqa", epochs=2, smoke=True, prefix="r22_aig_chk")]
        rows_to_csv(rows, cdir / "JOBS.csv")
    elif a.part == "kd_s1":
        cdir = cfg_root / "kd"
        rows = [kd_row(arm, lam, KD_CAL_SEED, cdir, OUT / "kd" / "runs") for lam in sorted(KD_LAMBDAS, reverse=True)
                for arm in ("vcscrd", "dkd_vcscrd")]
        rows += [kd_row("crd_uniform", 0.8, s, cdir, OUT / "kd" / "runs") for s in (KD_CAL_SEED,) + KD_FINAL_SEEDS]
        rows_to_csv(rows, cdir / "JOBS_S1.csv")
    elif a.part == "kd_s2_all":   # every (arm, lambda, final seed); select_kd.py picks the 6 rows of the selected lambdas
        cdir = cfg_root / "kd"
        rows = [kd_row(arm, lam, s, cdir, OUT / "kd" / "runs")
                for arm in ("vcscrd", "dkd_vcscrd") for lam in KD_LAMBDAS for s in KD_FINAL_SEEDS]
        rows_to_csv(rows, cdir / "JOBS_S2_ALL.csv")
    elif a.part == "kd_s2":
        if a.vcscrd_lambda is None or a.dkd_vcscrd_lambda is None:
            raise SystemExit("kd_s2 needs both selected lambdas (select_kd.py)")
        cdir = cfg_root / "kd"
        rows = [kd_row(arm, lam, s, cdir, OUT / "kd" / "runs")
                for arm, lam in (("vcscrd", a.vcscrd_lambda), ("dkd_vcscrd", a.dkd_vcscrd_lambda)) for s in KD_FINAL_SEEDS]
        rows_to_csv(rows, cdir / "JOBS_S2.csv")
    elif a.part == "aig_e2":
        cdir = cfg_root / "aigiqa"
        rows = [aig_row(arm, s, cdir, OUT / "aigiqa" / "runs") for arm in ("exact_bank_l001", "bce_bank_l001") for s in AIG_SEEDS]
        rows_to_csv(rows, cdir / "JOBS_E2.csv")
    else:
        cdir = cfg_root / "aigiqa"
        rows = [aig_row(arm, s, cdir, OUT / "aigiqa" / "runs")
                for arm in ("exact_bank_mosw", "bce_bank_mosw", "exact_bank", "exact_inbatch") for s in AIG_SEEDS]
        rows_to_csv(rows, cdir / "JOBS.csv")
    print(json.dumps({"part": a.part, "rows": len(rows)}))


if __name__ == "__main__":
    main()
