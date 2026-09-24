"""P0 preflight: ``python -m vcs_ssl.preflight --config configs/cifar10_pilot_vcs.yaml [--create-manifest]``.

Read-only environment / data / config verification (spec §2) plus creation of the shared split manifest when asked.
Writes ``reports/P0_preflight.md`` and ``reports/P0_preflight.json`` under the repo (or ``--report-dir``).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from .config import ConfigError, load_config
from .data.cifar import CIFAR10_FIRST10_LABELS, load_cifar10_train
from .data.splits import build_manifest, load_manifest, write_manifest
from .utils import atomic_write_json, atomic_write_text, environment_info, git_info, precision_flags, utc_now


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def disk(path: str | Path) -> dict[str, Any]:
    try:
        u = shutil.disk_usage(path)
        st = os.statvfs(path)
        return {"total_gb": u.total / 2**30, "free_gb": u.free / 2**30, "inodes_free": st.f_favail}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--create-manifest", action="store_true", help="create the shared split manifest if it does not exist")
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args(argv)
    report_dir = Path(args.report_dir) if args.report_dir else repo_root() / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"utc": utc_now(), "argv": sys.argv, "status": "PASS", "problems": []}

    def problem(msg: str) -> None:
        rep["problems"].append(msg)
        rep["status"] = "FAIL"

    rep["repo"] = git_info(repo_root())
    rep["environment"] = environment_info()
    rep["precision_flags_default"] = precision_flags()
    rep["env_vars"] = {k: os.environ.get(k) for k in ("REPO_ROOT", "DATA_ROOT", "OUTPUT_ROOT", "MANIFEST_ROOT", "SLURM_JOB_ID", "SLURM_JOB_PARTITION")}
    for k in ("DATA_ROOT", "OUTPUT_ROOT", "MANIFEST_ROOT"):
        v = os.environ.get(k)
        if not v:
            problem(f"{k} is not exported")
        elif not Path(v).is_dir():
            problem(f"{k}={v} is not an existing directory")
    if not torch.cuda.is_available():
        problem("no CUDA device visible to this process (preflight must run inside a GPU allocation)")
    else:
        names = [g["name"] for g in rep["environment"]["gpus"]]
        if any("P100" in n for n in names):
            problem(f"P100 GPU detected ({names}); policy forbids P100")
    rep["resources"] = {"cpu_count": os.cpu_count(), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "slurm_mem": os.environ.get("SLURM_MEM_PER_NODE"),
                        "disk_output_root": disk(os.environ.get("OUTPUT_ROOT", ".")), "disk_data_root": disk(os.environ.get("DATA_ROOT", "."))}
    # config
    try:
        cfg = load_config(args.config)
        rep["config"] = {"path": cfg["_meta"]["config_path"], "file_sha256": cfg["_meta"]["config_file_sha256"], "config_hash": cfg["_meta"]["config_hash"],
                         "method": cfg["run"]["method"], "resolved_data_root": cfg["data"]["root"], "resolved_output_root": cfg["run"]["output_root"],
                         "resolved_manifest": cfg["data"]["manifest"], "policy_checks": "PASS"}
    except ConfigError as e:
        problem(f"config: {e}")
        cfg = None
    # data
    if cfg is not None:
        try:
            data = load_cifar10_train(cfg["data"]["root"])
            rep["data"] = {"n_train": len(data), "file_hashes": data.file_hashes, "source": data.source,
                           "first10_labels": data.targets[:10].tolist(), "first10_expected": CIFAR10_FIRST10_LABELS, "official_test_read": False}
            mpath = Path(cfg["data"]["manifest"])
            if mpath.is_file():
                m = load_manifest(mpath, expected_seed=cfg["data"]["split_seed"], expected_val_per_class=cfg["data"]["val_per_class"])
                rep["manifest"] = {"path": str(mpath), "status": "EXISTS_VERIFIED", "sha256": m["manifest_sha256"], "n_fit": m["n_fit"], "n_selection": m["n_selection"]}
                if m["raw_file_hashes"] != data.file_hashes:
                    problem("existing manifest raw_file_hashes differ from data on disk")
            elif args.create_manifest:
                m = build_manifest(data.targets, split_seed=cfg["data"]["split_seed"], val_per_class=cfg["data"]["val_per_class"], source=data.source,
                                   file_hashes=data.file_hashes)
                write_manifest(m, mpath)
                m2 = load_manifest(mpath, expected_seed=cfg["data"]["split_seed"], expected_val_per_class=cfg["data"]["val_per_class"])
                rep["manifest"] = {"path": str(mpath), "status": "CREATED", "sha256": m2["manifest_sha256"], "n_fit": m2["n_fit"], "n_selection": m2["n_selection"],
                                   "fit_class_counts": m2["fit_class_counts"], "selection_class_counts": m2["selection_class_counts"],
                                   "fit_first5": m2["fit_uids"][:5], "selection_first5": m2["selection_uids"][:5]}
            else:
                problem(f"manifest {mpath} missing (run with --create-manifest)")
        except Exception as e:  # noqa: BLE001
            problem(f"data: {type(e).__name__}: {e}")
            rep["data_traceback"] = traceback.format_exc()
    # quick GPU tensor sanity (tiny; not training)
    if torch.cuda.is_available():
        try:
            x = torch.randn(4, 3, 32, 32, device="cuda")
            from .models import build_models  # noqa: PLC0415

            if cfg is not None:
                built = build_models(cfg, seed=0, device=torch.device("cuda", 0))
                with torch.no_grad():
                    h = built["encoder"](x)
                    p = built["projector"](h)
                rep["gpu_shape_check"] = {"h": list(h.shape), "p_raw": list(p.shape), "params": built["params"]}
        except Exception as e:  # noqa: BLE001
            problem(f"gpu shape check: {e!r}")
    rep["to_implement"] = "all CLIs implemented in this repo (train/evaluate/preflight/summarize/prepare_data); no prior SSL trainer existed in the workspace"
    atomic_write_json(report_dir / "P0_preflight.json", rep)
    atomic_write_text(report_dir / "P0_preflight.md", render_md(rep))
    print(f"preflight {rep['status']}; problems: {rep['problems']}")
    return 0 if rep["status"] == "PASS" else 2


def render_md(r: dict[str, Any]) -> str:
    e = r["environment"]
    lines = [f"# P0 preflight — {r['status']}", "", f"UTC: {r['utc']}", ""]
    if r["problems"]:
        lines += ["## Problems", *[f"- {p}" for p in r["problems"]], ""]
    lines += ["## Repo", f"- root: `{r['repo']['repo_root']}`", f"- branch/commit: `{r['repo']['branch']}` / `{r['repo']['commit']}`",
              f"- dirty: {r['repo']['is_dirty']} ({len(r['repo']['dirty_files'])} files)", "",
              "## Environment", f"- python: `{e['python'].splitlines()[0]}`", f"- torch: `{e['torch']}` (CUDA build {e['torch_cuda_build']}, cuDNN {e['cudnn']})",
              f"- torchvision: `{e.get('torchvision')}`", f"- numpy: `{e.get('numpy')}`", f"- host: `{e['hostname']}`  SLURM job `{e['slurm_job_id']}` partition `{e['slurm_partition']}`",
              f"- GPUs: {e.get('gpus')}", f"- nvidia-smi: {e.get('nvidia_smi')}", f"- precision flags (defaults before policy): {r['precision_flags_default']}", "",
              "## Resources", f"- {r['resources']}", "", "## Paths", f"- {r['env_vars']}", ""]
    if "config" in r:
        lines += ["## Config", *[f"- {k}: `{v}`" for k, v in r["config"].items()], ""]
    if "data" in r:
        d = r["data"]
        lines += ["## Data", f"- n_train: {d['n_train']} (official train partition only; test partition not read)",
                  f"- first10 labels: {d['first10_labels']} (expected {d['first10_expected']})", "- file hashes:",
                  *[f"  - {k}: md5 `{v['md5']}` sha256 `{v['sha256']}`" for k, v in d["file_hashes"].items()], ""]
    if "manifest" in r:
        lines += ["## Manifest", *[f"- {k}: `{v}`" for k, v in r["manifest"].items()], ""]
    if "gpu_shape_check" in r:
        lines += ["## GPU shape check", f"- {r['gpu_shape_check']}", ""]
    lines += ["## To implement / mapping", f"- {r['to_implement']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
