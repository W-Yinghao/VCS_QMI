#!/usr/bin/env python3
"""Build the repeat-audit input of the 2026-10-20 review package (v10/REPEAT_INPUT_CONTRACT_CN.md) from existing records.

Pairs only same-checkpoint, same-GPU-model DEV32 re-evaluations of the recorded BF finals listed in the TEST_REPEAT134
manifest. Each source directory must hold DEV.jsonl and ENV.json:
  - TEST_REPEAT134 v1/v2 DEV gates: handoff/v10_test_repeat134_20261020/gate/<type>__<family>/<job>/
  - TEST_REPEAT134 v5 same-job DEV re-evaluations (--v5): results/TEST_REPEAT/<unit>/DEV_REEVAL.jsonl with all 320 rows and
    the job's ENV_<job>.json; only DEV_REEVAL.* and ENV files are opened (no TEST file)
  - extra re-evaluation directories passed with --extra <unit_key>=<dir>
Pairs on a different GPU model or a different effective environment (precision block incl. driver) are not written to the
audit input; they are listed in cross_gpu_pairs.json with the differing fields.
Usage: repeat_rows.py OUT_DIR [--v5] [--extra UNIT=DIR ...]
"""
import argparse, csv, hashlib, json
from pathlib import Path

H = Path("/home/infres/yinwang/CS_QMI/handoff/v10_test_repeat134_20261020")
PREC_KEYS = ("torch", "cuda_runtime", "cudnn", "torch_threads", "float32_matmul_precision", "cudnn_conv_fp32_precision",
             "cuda_matmul_fp32_precision", "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic",
             "cublas_workspace_config", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")
COLS = ("dataset", "eval_set", "method", "seed", "checkpoint_sha256", "gpu_model", "run_id", "unit_id", "record_id",
        "perturbation_id", "err_native_px", "protocol_sha256", "precision_sha256", "input_sha256", "perturbation_sha256",
        "energy_at_truth_sha256")


def sha(b): return hashlib.sha256(b).hexdigest()
def cj(o): return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def precision_hash(env):
    eff = {k: env.get(k) for k in PREC_KEYS}
    eff["precision"] = {k: v for k, v in (env.get("precision") or {}).items() if k != "decision"}  # effective values only
    return sha(cj(eff))


def run_rows(u, run_id, dev, env):
    out = []
    for line in open(dev):
        r = json.loads(line)
        if r["checkpoint_sha256"] != u["checkpoint"]["sha256"]:
            raise SystemExit(f"{run_id}: checkpoint differs from the frozen manifest")
        out.append({"dataset": "BF_SHG", "eval_set": "DEV32", "method": u["identity"].get("family", u["family"]), "seed": u["seed"],
                    "checkpoint_sha256": r["checkpoint_sha256"], "gpu_model": env["gpu_name"], "run_id": run_id,
                    "unit_id": r["core"], "record_id": r["core"], "perturbation_id": r["perturbation"],
                    "err_native_px": repr(float(r["err_full"])),
                    "protocol_sha256": sha(cj(u["identity"])), "precision_sha256": precision_hash(env),
                    "input_sha256": u["identity"]["data_lock_sha256"],
                    "perturbation_sha256": sha(cj({k: r[k] for k in ("core", "perturbation", "theta", "t", "n_grid")})),
                    "energy_at_truth_sha256": sha(cj({"F_true": r["F_true"], "dtype": "float, JSON repr"}))})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--v5", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    man = {u["key"]: u for u in json.load(open(H / "TEST_REPEAT134_MANIFEST.json"))["units"]}
    repeats = []                                              # (unit_key, run_id, dir, dev_file, env_file)
    for g in sorted(H.glob("gate/*/*/GATE.json")):
        repeats.append((json.load(open(g))["unit"], f"gate:{g.parent.parent.name}:{g.parent.name}", g.parent, "DEV.jsonl", "ENV.json"))
    if a.v5:
        for d in sorted((H / "results/TEST_REPEAT").glob("*")):
            dev = d / "DEV_REEVAL.jsonl"; envs = sorted(d.glob("ENV_*.json"))
            if d.name in man and dev.exists() and envs and sum(1 for _ in open(dev)) == 320:
                repeats.append((d.name, f"v5:{d.name}:{envs[-1].stem[4:]}", d, "DEV_REEVAL.jsonl", envs[-1].name))
    for e in a.extra:
        unit, d = e.split("=", 1); repeats.append((unit, f"extra:{unit}:{Path(d).name}", Path(d), "DEV.jsonl", "ENV.json"))
    rows, comps, cross, refs = [], [], [], {}
    for unit, rid, d, devf, envf in repeats:
        u = man[unit]; ud = Path(u["unit_dir"])
        renv = json.load(open(sorted(ud.glob("ENV_*.json"))[-1])); xenv = json.load(open(d / envf))
        ref_id = f"final:{unit}:{renv['slurm_job_id']}"
        pair = {"unit": unit, "reference_run_id": ref_id, "repeat_run_id": rid,
                "reference_gpu": renv["gpu_name"], "repeat_gpu": xenv["gpu_name"]}
        if renv["gpu_name"] != xenv["gpu_name"] or precision_hash(renv) != precision_hash(xenv):
            pr, px = renv.get("precision") or {}, xenv.get("precision") or {}
            pair["differs"] = sorted([k for k in PREC_KEYS if renv.get(k) != xenv.get(k)] +
                                     [f"precision.{k}" for k in set(pr) | set(px) if k != "decision" and pr.get(k) != px.get(k)])
            pair["values"] = {k: [pr.get(k.split(".", 1)[1]), px.get(k.split(".", 1)[1])] if k.startswith("precision.") else
                              [renv.get(k), xenv.get(k)] for k in pair["differs"]}
            cross.append(pair); continue
        if ref_id not in refs:
            refs[ref_id] = run_rows(u, ref_id, ud / "DEV.jsonl", renv); rows += refs[ref_id]
        rows += run_rows(u, rid, d / devf, xenv)
        comps.append({"dataset": "BF_SHG", "eval_set": "DEV32", "method": u["identity"].get("family", u["family"]), "seed": u["seed"],
                      "checkpoint_sha256": u["checkpoint"]["sha256"], "gpu_model": xenv["gpu_name"],
                      "reference_run_id": ref_id, "repeat_run_id": rid})
    with open(a.out / "repeat_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    manifest = {"schema": "v10.repeat-audit-input.v1", "rows_csv": "repeat_rows.csv",
                "rows_csv_sha256": sha((a.out / "repeat_rows.csv").read_bytes()),
                "thresholds_native_px": {"BF_SHG": [42, 9], "EPFL_RGB_NIR": [10.5, 2.25]}, "comparisons": comps}
    (a.out / "repeat_input.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (a.out / "cross_gpu_pairs.json").write_text(json.dumps(cross, indent=1) + "\n")
    (a.out / "HASH_PROVENANCE.json").write_text(json.dumps({
        "generator": "v10/review_20261020/repeat_rows.py", "algorithm": "sha256 over canonical JSON (sorted keys, no spaces)",
        "protocol_sha256": "the unit's frozen identity block in TEST_REPEAT134_MANIFEST.json (freeze or release, family, seed, data lock, ...), canonical JSON",
        "input_sha256": "unit identity data_lock_sha256 (DATA_LOCK of the inputs) from the manifest",
        "precision_sha256": f"ENV json: {list(PREC_KEYS)} plus the effective 'precision' block without its free-text 'decision'; "
                            "GPU driver version is not recorded in these ENV files",
        "perturbation_sha256": "per row: core, perturbation, theta, t, n_grid from DEV.jsonl",
        "energy_at_truth_sha256": "per row: F_true as stored in DEV.jsonl (JSON float repr; the value was computed in float32 "
                                  "and round-trips exactly, so equal hashes mean equal values; the raw tensor bytes are not stored)",
        "sources": {"manifest": str(H / "TEST_REPEAT134_MANIFEST.json"),
                    "manifest_sha256": sha((H / "TEST_REPEAT134_MANIFEST.json").read_bytes()),
                    "repeat_dirs": [str(d) for _, _, d, _, _ in repeats]}}, indent=1) + "\n")
    print(json.dumps({"rows": len(rows), "comparisons": len(comps), "cross_gpu": len(cross)}))


if __name__ == "__main__":
    main()
