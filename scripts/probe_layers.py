"""Read-only per-layer probe diagnostic on frozen checkpoints (synthesis §5.A.3): where does the information stop being linearly readable?

    python scripts/probe_layers.py --runs A,B,... --out reports/PROBE_LAYERS.md

For each run (final checkpoint, eval mode, frozen) and each feature layer
    l2 = ResNet layer2 output (avg-pooled, 128), l3 = layer3 (256), h = layer4/avgpool (512, the frozen endpoint),
    proj_hidden = projector hidden ReLU output (hidden_dim), p_raw = projector output (output_dim), z_l2 = L2(p_raw)
compute on clean-transform features (fit UIDs train, selection UIDs score):
    the frozen linear-probe protocol of the run's own evaluation config (SGD 0.1, 100 ep, cosine, final epoch), kNN (k=200, T=0.1),
    and the effective rank on the first 4096 selection UIDs.
Nothing is trained except probe heads; controls are processed with the same code.  Features are not cached (dimension differs per layer).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vcs_ssl.checkpoint import load_checkpoint
from vcs_ssl.config import load_resolved
from vcs_ssl.data.cifar import load_cifar10_train
from vcs_ssl.data.datasets import LabeledCleanDataset, make_eval_loader
from vcs_ssl.data.splits import load_manifest
from vcs_ssl.data.transforms import build_clean_transform
from vcs_ssl.diagnostics import knn_eval, linear_probe, spectrum_stats
from vcs_ssl.models import build_models
from vcs_ssl.utils import atomic_write_json, utc_now


@torch.no_grad()
def extract_layers(enc: nn.Module, proj: nn.Module, images, targets, uids, tf, device, batch_size=512, num_workers=4, seed=0):
    feats: dict[str, list[torch.Tensor]] = {}
    hooks = []

    def pooled(name):
        def fn(_m, _i, out):
            feats.setdefault(name, []).append(torch.flatten(F.adaptive_avg_pool2d(out, 1), 1).float().cpu())
        return fn

    def flat(name):
        def fn(_m, _i, out):
            feats.setdefault(name, []).append(out.float().cpu())
        return fn

    hooks.append(enc.layer2.register_forward_hook(pooled("l2")))
    hooks.append(enc.layer3.register_forward_hook(pooled("l3")))
    relus = [m for m in proj if isinstance(m, nn.ReLU)]
    for i, m in enumerate(relus):
        hooks.append(m.register_forward_hook(flat("proj_hidden" if len(relus) == 1 else f"proj_hidden{i + 1}")))
    ds = LabeledCleanDataset(images, targets, uids, tf)
    loader = make_eval_loader(ds, batch_size=batch_size, num_workers=num_workers, generator=torch.Generator().manual_seed(seed),
                              pin_memory=device.type == "cuda")
    ys = []
    for x, y, _ in loader:
        h = enc(x.to(device))
        feats.setdefault("h", []).append(h.float().cpu())
        p = proj(h)
        feats.setdefault("p_raw", []).append(p.float().cpu())
        ys.append(y)
    for hk in hooks:
        hk.remove()
    out = {k: torch.cat(v) for k, v in feats.items()}
    out["z_l2"] = F.normalize(out["p_raw"], dim=1)
    return out, torch.cat(ys)


def diagnose(run_dir: Path, device: torch.device, num_workers: int) -> dict:
    cfg = load_resolved(run_dir / "config.resolved.yaml")
    man = load_manifest(run_dir / "manifest.json")
    ck_name = f"epoch_{cfg['train']['epochs']:03d}.pt"
    ck = load_checkpoint(run_dir / "checkpoints" / ck_name)
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
    enc, proj = built["encoder"], built["projector"]
    enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"])
    for m in (enc, proj):
        m.eval()
        for q in m.parameters():
            q.requires_grad_(False)
    data = load_cifar10_train(cfg["data"]["root"])
    fit = np.asarray(man["fit_uids"], dtype=np.int64); sel = np.asarray(man["selection_uids"], dtype=np.int64)
    tf = build_clean_transform(cfg["views"])
    Ff, yf = extract_layers(enc, proj, data.data, data.targets, fit, tf, device, num_workers=num_workers, seed=11)
    Fs, ys = extract_layers(enc, proj, data.data, data.targets, sel, tf, device, num_workers=num_workers, seed=12)
    lcfg = cfg["evaluation"]["linear"]; kc = cfg["evaluation"]["knn"]
    out = {"run": run_dir.name, "method": cfg["run"]["method"], "epochs": cfg["train"]["epochs"], "checkpoint": ck_name,
           "critic_input": cfg["model"]["critic"]["input"] if cfg["model"]["critic"]["enabled"] else None, "K": cfg["pairing"]["k"],
           "negative_detach": cfg["pairing"]["negative_detach"], "layers": {}}
    for name in [k for k in ("l2", "l3", "h", "proj_hidden", "proj_hidden1", "proj_hidden2", "p_raw", "z_l2") if k in Ff]:
        lp = linear_probe(Ff[name], yf, Fs[name], ys, lcfg, device=device)
        kn = knn_eval(Ff[name], yf, Fs[name], ys, k=kc["k"], temperature=kc["temperature"], chunk=kc["query_chunk"], device=device)
        sp = spectrum_stats(Fs[name][:4096])
        out["layers"][name] = {"dim": int(Ff[name].shape[1]), "linear": lp["linear_val_top1_pct"], "knn": kn["knn_val_top1_pct"],
                               "eff_rank": sp["effective_rank"]}
        print(f"  {run_dir.name} {name:12s} dim={Ff[name].shape[1]:5d} linear={lp['linear_val_top1_pct']:.2f} knn={kn['knn_val_top1_pct']:.2f} rank={sp['effective_rank']:.1f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    rows = []
    for r in a.runs.split(","):
        try:
            rows.append(diagnose(Path(a.output_root) / r, device, a.num_workers)); print(f"{r}: done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{r}: FAILED {e!r}", flush=True)
    names = ["l2", "l3", "h", "proj_hidden", "p_raw", "z_l2"]
    L = [f"# Per-layer probe diagnostic — {utc_now()}", "",
         "Frozen final checkpoints, clean-transform features, fit UIDs train the head / selection UIDs score it. Each cell: linear-val % / kNN % (k=200, T=0.1) / effective rank (4096 selection). "
         "l2, l3 = avg-pooled ResNet stage outputs; h = the frozen endpoint; proj_hidden = projector ReLU output; p_raw / z_l2 = projector output before / after L2.", "",
         "| run | method | critic | K | negdet | " + " | ".join(names) + " |", "|---|---|---|---|---|" + "---|" * len(names)]
    for r in rows:
        cells = []
        for n in names:
            v = r["layers"].get(n)
            cells.append(f"{v['linear']:.2f} / {v['knn']:.2f} / {v['eff_rank']:.0f} (d={v['dim']})" if v else "—")
        L.append(f"| {r['run']} | {r['method']} | {r['critic_input']} | {r['K']} | {r['negative_detach']} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    atomic_write_json(Path(a.out).with_suffix(".json"), rows)
    print(f"{len(rows)} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
