"""V9-N2 driver.

    python scripts/v9/n2_run.py smoke   --out <dir> [--device cpu|cuda] [--epochs 2]      # fixture end-to-end: roles, noise, teacher, weights (N0/N3/N4/N7/filter [+N5/N6/KDE if quant is available]), students, dev/test eval, strict check
    python scripts/v9/n2_run.py strict  --out <dir>                                       # the §05.5 strict-interface check alone
    python scripts/v9/n2_run.py unit    --dataset f30k|coco --rho 0.4 --method VCS_Weight --seed 901 --lr-mult 1 --data-root D --vocab V --out-root O [--formal] [--host-lock reports/V9/HOST_LOCK.yaml]
                                                                                           # one formal unit on the real precomp data (needs fork A's data + env + HOST_LOCK)
Every run writes a JSON summary; nothing here reads an official test split unless `unit --test` is given explicitly (the protocol decides when).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from v9.n2.fixtures import make_fixture  # noqa: E402
from v9.n2.pipeline import run_unit  # noqa: E402
from v9.n2.strict import run_strict_check  # noqa: E402
from v9.n2.teacher import HostConfig  # noqa: E402


def fixture_dataset(seed: int, n_images: int = 60):
    tr = make_fixture(n_images=n_images, seed=seed); dev = make_fixture(n_images=20, seed=seed + 100); te = make_fixture(n_images=20, seed=seed + 200)
    ds = dict(tr); ds["dev_images"] = dev["images"]; ds["dev_captions"] = [dev["captions"][i] for i in range(len(dev["captions"]))]
    test = {"images": te["images"], "captions": [te["captions"][i] for i in range(len(te["captions"]))]}
    return ds, test, tr


def cmd_smoke(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    ds, test, tr = fixture_dataset(a.seed, a.n_images)
    # fixture-only host and roles: reduced dims, anchor fraction 0.5 (the formal protocol uses 0.1) and 8 teacher epochs so that the tiny teacher
    # separates P from Q and the rho path is exercised in its identified branch; the formal values come from HOST_LOCK / the protocol.
    cfg = HostConfig(vocab_size=tr["vocab"], img_dim=tr["img_dim"], word_dim=16, embed_size=32, sim_dim=16, num_region=tr["n_region"], batch_size=16,
                     num_epochs=a.epochs, teacher_epochs=8, lr_update=100, learning_rate=2e-3)
    methods = ["Unweighted_A", "VCS_Weight", "BCE_Weight", "RankMass", "PlainFilter"]
    try:
        import v9.quant  # noqa: F401
        methods += ["RuLSIF_Weight", "Nystrom_Weight", "KDE_Weight"]
    except Exception:
        pass
    summary = {"fixture": {"n_images": a.n_images, "seed": a.seed, "host": "reduced dims (fixture only)"}, "units": {}, "strict": None}
    for rho in (0.0, 0.4):
        for m in methods:
            try:
                r = run_unit(ds, cfg, out, "fixture", rho, m, a.seed, a.device, epochs=a.epochs, critic_steps=120, anchor_fraction=0.5,
                             min_sizes={k: 1 for k in ("teacher_warmup", "critic_fit", "critic_select", "reference_moments")}, test=test)
                summary["units"][f"rho{rho:g}/{m}"] = {"best_dev_rsum": r["done"]["best"]["rsum"], "test_rsum": r["test"]["primary"]["rsum"], "realized_rho": r["noise_manifest"]["realized_rho"],
                                                       "rho_hat": r["weights_record"].get("rho_used"), "ess": r["weights_record"]["ess"], "mass": r["weights_record"]["total_mass"], "status": r["weights_record"].get("rho", {}).get("status")}
            except NotImplementedError as e:
                summary["units"][f"rho{rho:g}/{m}"] = {"skipped": str(e)}
    summary["strict"] = run_strict_check(seed=a.seed); summary["seconds"] = time.time() - t0
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=1, default=str)); print(json.dumps(summary, indent=1, default=str)[:4000])
    return 0


def cmd_strict(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res = run_strict_check(seed=a.seed); (out / "strict_check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); return 0


def cmd_unit(a):
    import yaml
    from v9.n2.data import eval_arrays, load_precomp, load_vocab
    name = {"f30k": "f30k_precomp", "coco": "coco_precomp"}[a.dataset]; w2i = load_vocab(Path(a.vocab) / f"{name}_vocab.json")
    tr = load_precomp(Path(a.data_root) / name, "train", w2i); dev = load_precomp(Path(a.data_root) / name, "dev", w2i)
    ds = dict(tr); ds["dev_images"], ds["dev_captions"] = eval_arrays(dev)
    if a.dataset == "coco":                                    # NCR: dev is the first 1000 images of the 5K dev file
        ds["dev_images"], ds["dev_captions"] = ds["dev_images"][:1000], ds["dev_captions"][:5000]
    lock = yaml.safe_load(Path(a.host_lock).read_text()) if a.host_lock and Path(a.host_lock).exists() else None
    cfg = HostConfig.from_host_lock(len(w2i), (lock or {}).get("host", lock))
    test = None
    if a.test:
        split = "testall" if a.dataset == "coco" else "test"; te = load_precomp(Path(a.data_root) / name, split, w2i); test = dict(zip(("images", "captions"), eval_arrays(te)))
    r = run_unit(ds, cfg, Path(a.out_root), a.dataset, a.rho, a.method, a.seed, a.device, lr_mult=a.lr_mult, formal=a.formal, test=test)
    print(json.dumps({k: v for k, v in r.items() if k in ("unit", "done", "scores_hash")}, indent=1, default=str)); return 0


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--out", required=True); s.add_argument("--device", default="cpu"); s.add_argument("--epochs", type=int, default=2); s.add_argument("--seed", type=int, default=0); s.add_argument("--n-images", type=int, default=80)
    s = sub.add_parser("strict"); s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=0)
    u = sub.add_parser("unit")
    for k, kw in (("--dataset", {"required": True, "choices": ["f30k", "coco"]}), ("--rho", {"type": float, "required": True}), ("--method", {"required": True}), ("--seed", {"type": int, "required": True}),
                  ("--lr-mult", {"type": float, "default": 1.0}), ("--data-root", {"required": True}), ("--vocab", {"required": True}), ("--out-root", {"required": True}), ("--host-lock", {"default": "reports/V9/HOST_LOCK.yaml"}),
                  ("--device", {"default": "cuda"}), ("--formal", {"action": "store_true"}), ("--test", {"action": "store_true"})):
        u.add_argument(k, **kw)
    a = ap.parse_args(); return {"smoke": cmd_smoke, "strict": cmd_strict, "unit": cmd_unit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
