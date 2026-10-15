"""V9-N2 official-baseline driver for N1 NCR+A and N2 L2RM+A (owner 2026-10-15): exports the V9 observed mapping and anchor set of a shared stage
as the two numpy files the patches expect, launches the PATCHED official training in fork A's environments, and collects the official
evaluation into the same unit-dir layout as the V9 runner (result.json with test.primary, done.json), so the aggregate reads all 8 methods alike.

    python scripts/v9/n2_official.py export  --shared <out_root>/<dataset>/rho<rho>/s<seed> --data-root <f30k_precomp dir> --out <unit dir>
    python scripts/v9/n2_official.py launch  --baseline ncr|l2rm --unit <unit dir> --dataset f30k|coco --seed 901 --lr-mult 1 [--host-lock reports/V9/HOST_LOCK.yaml] [--dry-run] [--anchor-off]
    python scripts/v9/n2_official.py collect --baseline ncr|l2rm --unit <unit dir> --dataset f30k|coco

Paths (clones, environments, data layout) come from HOST_LOCK.yaml (fork A); the patched copies live under <unit>/code/<baseline> (patch applied at
launch from v9/patches, hash recorded).  `--anchor-off` launches the native path (no observed index, official noise index) for the A-table
reproduction / the degradation check on real data.  Nothing here reads test labels: the official `evaluation.py` of each baseline does, once, in
`collect`, on the official test split that the frozen protocol names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from v9.n2.noise import load_observed  # noqa: E402
from v9.n2.roles import ANCHOR_ROLES  # noqa: E402

DEFAULT_PATHS = {"ncr": {"clone": "/projects/EEG-foundation-model/yinghao/code_v9/2021-NeurIPS-NCR", "env": "/projects/EEG-foundation-model/yinghao/envs/v9_sgr", "entry": "NCR/run.py"},
                 "l2rm": {"clone": "/projects/EEG-foundation-model/yinghao/code_v9/L2RM", "env": "/projects/EEG-foundation-model/yinghao/envs/v9_l2rm", "entry": "main_L2RM.py"},
                 "data_root": "/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data", "vocab": "/projects/EEG-foundation-model/yinghao/code_v9/SGRAF_py3/vocab",
                 "noise_index": "/projects/EEG-foundation-model/yinghao/code_v9/L2RM/noise_index"}
NAME = {"f30k": "f30k_precomp", "coco": "coco_precomp"}


def host_lock(path: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text()) or {}
    except Exception:
        return {}


def cmd_export(a):
    """observed mapping (int64 per caption, official order) + anchor mask (bool per caption) from the shared stage's observed table and roles."""
    shared = Path(a.shared); rows = load_observed(shared / "pairs.observed_pairs.json"); roles = json.loads((shared / "roles.json").read_text())
    caps_file = Path(a.data_root) / "train_caps.txt"
    n_caps = sum(1 for _ in open(caps_file)) if caps_file.exists() else len(rows)          # fixture shared stages have no official caption file
    t2i = -np.ones(n_caps, dtype=np.int64); anchor = np.zeros(n_caps, dtype=bool)
    anchor_imgs = set(i for r in ANCHOR_ROLES for i in roles["A"][r])
    for r in rows:
        cid = int(r["caption_id"]); t2i[cid] = int(r["observed_image_id"]); anchor[cid] = r["role"] != "U"
    if np.any(t2i < 0):
        raise ValueError(f"{int((t2i < 0).sum())} captions without an observed image (table incomplete)")
    per_image = len(rows) // max(1, len(set(int(r["observed_image_id"]) for r in rows) | set(int(r["observed_image_id"]) for r in rows)))
    assert anchor.sum() == sum(1 for r in rows if r["role"] != "U"), "anchor count mismatch"
    assert all(t2i[int(r["caption_id"])] == int(r["caption_id"]) // 5 for r in rows if r["role"] != "U") or caps_file.exists() is False, "anchor pairs must be clean (official layout: caption i -> image i // 5)"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); np.save(out / "v9_observed_t2i.npy", t2i); np.save(out / "v9_anchor_mask.npy", anchor)
    man = {"shared": str(shared), "n_captions": int(n_caps), "n_anchor_captions": int(anchor.sum()), "n_anchor_images": len(anchor_imgs), "observed_sha256": hashlib.sha256((out / "v9_observed_t2i.npy").read_bytes()).hexdigest(),
           "anchor_sha256": hashlib.sha256((out / "v9_anchor_mask.npy").read_bytes()).hexdigest(), "noise_manifest": json.loads((shared / "pairs.noise_manifest.json").read_text()), "roles_hashes": roles["hashes"]}
    (out / "v9_export_manifest.json").write_text(json.dumps(man, indent=1)); print(json.dumps({k: v for k, v in man.items() if k != "noise_manifest"}, indent=1)); return 0


def _prepare_code(baseline: str, unit: Path, clone: Path) -> tuple[Path, str]:
    code = unit / "code" / baseline
    if not code.exists():
        shutil.copytree(clone, code, ignore=shutil.ignore_patterns(".git", "__pycache__", "runs", "output"))
        patch = ROOT / "v9" / "patches" / f"{baseline}_anchor_assisted.diff"
        r = subprocess.run(["patch", "-p1", "-s", "-i", str(patch)], cwd=code, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("patch failed: " + r.stdout + r.stderr)
        (code / "V9_PATCH.json").write_text(json.dumps({"patch": str(patch), "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "clone": str(clone), "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True).stdout.strip()}))
    return code, json.loads((code / "V9_PATCH.json").read_text())["sha256"][:16]


def cmd_launch(a):
    lock = host_lock(a.host_lock); P = DEFAULT_PATHS[a.baseline]; up = (lock.get("upstream") or {}).get(a.baseline) or {}
    clone = Path(up.get("path") or P["clone"]); env = Path(lock.get("environment_l2rm") if a.baseline == "l2rm" and lock.get("environment_l2rm") else lock.get("environment") or P["env"])
    if a.baseline == "l2rm" and not lock.get("environment_l2rm") and Path(P["env"]).exists():
        env = Path(P["env"])
    unit = Path(a.unit); code, phash = _prepare_code(a.baseline, unit, clone); name = NAME[a.dataset]; tr = (lock.get("training") or {})
    # NCR follows the SGR host schedule (HOST_LOCK training.native_epochs / lr_update per data name; README: f30k 40 / 30, coco 20 / 10);
    # L2RM follows its own train scripts (f30k: 35 epochs, lr_update 15, reg 0.01; coco: train script values) — both recorded in the launch manifest.
    if a.baseline == "ncr":
        ep_lock, lu_lock = tr.get("native_epochs") or {}, tr.get("lr_update") or {}
        epochs = int(ep_lock.get(name, {"f30k_precomp": 40, "coco_precomp": 20}[name])) if isinstance(ep_lock, dict) else {"f30k": 40, "coco": 20}[a.dataset]
        lr_update = int(lu_lock.get(name, {"f30k_precomp": 30, "coco_precomp": 10}[name])) if isinstance(lu_lock, dict) else {"f30k": 30, "coco": 10}[a.dataset]
    else:
        epochs, lr_update = {"f30k": 35, "coco": 20}[a.dataset], 15
    warm = {"f30k": 5, "coco": 10}[a.dataset]
    lr = 2e-4 * a.lr_mult; out_dir = unit / "official_out"; out_dir.mkdir(parents=True, exist_ok=True)
    data_root = a.data_root or DEFAULT_PATHS["data_root"]; vocab = a.vocab or DEFAULT_PATHS["vocab"]
    if a.anchor_off:
        noise_args = ["--noise_ratio", str(a.rho), "--noise_file", str(Path(a.noise_index or DEFAULT_PATHS["noise_index"]) / f"{name}_{a.rho}.npy")] if a.rho > 0 else ["--noise_ratio", "0"]
        v9_args = []
    else:
        noise_args = ["--noise_ratio", "0"]                               # the V9 mapping carries the noise; noise_ratio 0 disables the native generator
        v9_args = ["--observed_index_file", str(unit / "v9_observed_t2i.npy"), "--anchor_mask_file", str(unit / "v9_anchor_mask.npy")]
    if a.baseline == "ncr":
        cmd = [str(env / "bin/python"), "NCR/run.py", "--gpu", "0", "--workers", str(a.workers), "--data_name", name, "--data_path", data_root, "--vocab_path", vocab, "--output_dir", str(out_dir),
               "--num_epochs", str(epochs), "--lr_update", str(lr_update), "--warmup_epoch", str(warm), "--learning_rate", f"{lr:g}", "--batch_size", "128", "--seed", str(a.seed), *noise_args, *v9_args]
    else:
        cmd = [str(env / "bin/python"), "-u", "main_L2RM.py", "--module_name", "SGR", "--gpu", "0", "--batch_size", "128", "--seed", str(a.seed), "--reg", "0.01", "--num_epochs", str(epochs), "--lr", f"{lr:g}", "--lr_cost", "2e-6",
               "--workers", str(a.workers), "--lr_update", str(lr_update), "--data_name", name, "--queue_length", "128", "--warmup_epoch", str(warm), "--data_path", data_root, "--vocab_path", vocab, "--output_dir", str(out_dir), *noise_args, *v9_args]
    man = {"baseline": a.baseline, "dataset": a.dataset, "seed": a.seed, "lr_mult": a.lr_mult, "anchor_off": a.anchor_off, "rho": a.rho, "cmd": cmd, "cwd": str(code), "env": str(env), "patch_sha256_16": phash,
           "hyperparameters": {"epochs": epochs, "lr_update": lr_update, "warmup_epoch": warm, "lr": lr, "batch_size": 128}, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (unit / "launch_manifest.json").write_text(json.dumps(man, indent=1)); print(" ".join(cmd))
    if a.dry_run:
        return 0
    with open(unit / "official_train.log", "a") as log:
        rc = subprocess.run(cmd, cwd=code, stdout=log, stderr=subprocess.STDOUT).returncode
    (unit / "launch_manifest.json").write_text(json.dumps({**man, "rc": rc, "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1)); return rc


def _parse_recalls(text: str) -> dict:
    """The official evaluation prints 'Image to text: r1, r5, r10, medr, meanr' and 'Text to image: ...'; take the LAST pair (NCR: the 2-model average; L2RM: fold-5 mean or 5K)."""
    i2t = re.findall(r"Image to text:\s*([\d.]+),?\s*([\d.]+),?\s*([\d.]+)", text); t2i = re.findall(r"Text to image:\s*([\d.]+),?\s*([\d.]+),?\s*([\d.]+)", text)
    if not i2t or not t2i:
        raise ValueError("no recall lines found in the evaluation output")
    a, b = [float(x) for x in i2t[-1]], [float(x) for x in t2i[-1]]
    out = {"i2t_R1": a[0], "i2t_R5": a[1], "i2t_R10": a[2], "t2i_R1": b[0], "t2i_R5": b[1], "t2i_R10": b[2]}; out["rsum"] = sum(out.values()); out["mR"] = out["rsum"] / 6; out["tie"] = "official (numpy argsort)"
    return out


def cmd_collect(a):
    unit = Path(a.unit); code = unit / "code" / a.baseline; out_dir = unit / "official_out"; lock = host_lock(a.host_lock); name = NAME[a.dataset]
    man = json.loads((unit / "launch_manifest.json").read_text()); env = Path(man["env"]); data_root = a.data_root or DEFAULT_PATHS["data_root"]; vocab = a.vocab or DEFAULT_PATHS["vocab"]
    best = sorted(out_dir.rglob("model_best.pth.tar"))
    if not best:
        raise FileNotFoundError("no model_best.pth.tar under " + str(out_dir))
    split = "testall" if a.dataset == "coco" else "test"
    if a.baseline == "ncr":
        script = f"import sys; sys.path.insert(0, 'NCR'); from evaluation import evalrank; evalrank({str(best[-1])!r}, data_path={data_root!r}, vocab_path={vocab!r}, split={split!r}, fold5={'True' if a.dataset == 'coco' and a.fold5 else 'False'})"
    else:
        script = f"from evaluation import evalrank; evalrank({str(best[-1])!r}, data_path={data_root!r}, split={split!r}, fold5={'True' if a.dataset == 'coco' and a.fold5 else 'False'})"
    r = subprocess.run([str(env / "bin/python"), "-c", script], cwd=code, capture_output=True, text=True); (unit / "official_eval.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError("official evaluation failed; see official_eval.log")
    primary = _parse_recalls(r.stdout)
    result = {"unit": unit.name, "dataset": a.dataset, "method": {"ncr": "NCR_A", "l2rm": "L2RM_A"}[a.baseline] + ("_native" if man.get("anchor_off") else ""), "seed": man["seed"], "rho": man.get("rho"),
              "test": {"primary": primary, "evaluator": f"official {a.baseline} evaluation.py (NCR: two-model averaged sims)", "checkpoint": str(best[-1])}, "launch_manifest": man, "done": {"best": "official best_rsum on dev", "rc": man.get("rc")}}
    (unit / "result.json").write_text(json.dumps(result, indent=1)); (unit / "done.json").write_text(json.dumps({"best": result["done"], "official": True}, indent=1)); print(json.dumps(primary, indent=1)); return 0


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--shared", required=True); e.add_argument("--data-root", required=True); e.add_argument("--out", required=True)
    l = sub.add_parser("launch")
    for k, kw in (("--baseline", {"required": True, "choices": ["ncr", "l2rm"]}), ("--unit", {"required": True}), ("--dataset", {"required": True, "choices": ["f30k", "coco"]}), ("--seed", {"type": int, "required": True}),
                  ("--lr-mult", {"type": float, "default": 1.0}), ("--rho", {"type": float, "default": 0.4}), ("--host-lock", {"default": str(ROOT / "reports/V9/HOST_LOCK.yaml")}), ("--data-root", {"default": ""}), ("--vocab", {"default": ""}),
                  ("--noise-index", {"default": ""}), ("--workers", {"type": int, "default": 4}), ("--dry-run", {"action": "store_true"}), ("--anchor-off", {"action": "store_true"})):
        l.add_argument(k, **kw)
    c = sub.add_parser("collect")
    for k, kw in (("--baseline", {"required": True, "choices": ["ncr", "l2rm"]}), ("--unit", {"required": True}), ("--dataset", {"required": True, "choices": ["f30k", "coco"]}), ("--host-lock", {"default": str(ROOT / "reports/V9/HOST_LOCK.yaml")}),
                  ("--data-root", {"default": ""}), ("--vocab", {"default": ""}), ("--fold5", {"action": "store_true"})):
        c.add_argument(k, **kw)
    a = ap.parse_args(); return {"export": cmd_export, "launch": cmd_launch, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
