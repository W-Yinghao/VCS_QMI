"""P133 CPU gate, part 2 — the check the P100 report requires before any MoCo-consistent run: under the P133 pairs no CONSTANT online map
q(x) = c reaches J > 0, on real CIFAR-10 batches after real P133 training steps (real augmentation, B 256, a full queue, real key-encoder
drift at the PEAK learning rate — the schedule is entered at the end of warm-up, where drift is largest).  Positive control: the same
constant maps under the P100 construction (positives online-online, negatives online vs momentum-queue keys) DO reach J > 0.

    python scripts/p133_gate_constant_map.py --config configs/cifar10_hpMC_vcs_views4_800ep_seed0.yaml --steps 20 \
        --run-dir <scratch> --out reports/P133_GATE_<job>/constant_map.json

Candidates c: 256 random unit directions, the mean current-key direction, the adversarial direction mean(current keys) - mean(queue keys),
and c optimised by gradient ascent on J (Adam, 300 steps, on the sphere).  Also reported: the InfoNCE loss of the best constant map relative
to log(1 + N_neg) (chance level for exchangeable positive / negative keys) and the training J over the gate steps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    sys.path.insert(0, str(_p))

from vcs_ssl.config import load_config  # noqa: E402
from vcs_ssl.data.cifar import load_train_partition  # noqa: E402
from vcs_ssl.objectives import moco_consistent_infonce, moco_consistent_vcs  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.train import Trainer  # noqa: E402


def j_from(tp: torch.Tensor, tq: torch.Tensor) -> float:
    return float(tp.mean() - tq.mean() - 0.5 * tp.square().mean() - 0.5 * tq.square().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = load_config(a.config)
    assert cfg["pairing"].get("moco_consistent") and cfg["run"]["method"] == "vcs_qmi"
    data = load_train_partition(cfg["data"]["name"], cfg["data"]["root"])
    manifest = load_manifest(cfg["data"]["manifest"], expected_seed=cfg["data"]["split_seed"], expected_val_per_class=cfg["data"]["val_per_class"])
    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=manifest, device=torch.device("cpu"), stage="P133_GATE", epoch_eval=False)
    tr.setup()
    tr.step = tr.warmup_steps  # enter the schedule at the peak learning rate (largest per-step drift of the key encoder)
    it = iter(tr.loader)
    train_J, t0 = [], time.time()
    for _ in range(a.steps):
        batch = next(it)
        views, uids = list(batch[:-1]), batch[-1]
        out = tr.train_step(views[0], views[1], uids, False, extra_views=views[2:])
        tr.step += 1
        train_J.append(float(out["J_raw"]))
    step_s = (time.time() - t0) / a.steps
    batch = next(it)
    views, uids = list(batch[:-1]), batch[-1]
    keys = tr.p133_momentum_keys(views)
    queue, crit = tr.p133_queue, tr.critic  # queue is None for variant (a) (moco_use_queue false)
    B, D, V = keys[0].shape[0], keys[0].shape[1], len(keys)
    kall = torch.cat(keys)
    qk = queue.features() if queue is not None else None
    g = torch.Generator().manual_seed(a.seed + 1)
    cands = {f"random_{i}": F.normalize(torch.randn(D, generator=g), dim=0) for i in range(256)}
    cands["mean_current_key"] = F.normalize(kall.mean(0), dim=0)
    if qk is not None:
        cands["adversarial_current_minus_queue"] = F.normalize(kall.mean(0) - qk.mean(0), dim=0)

    def J_p133(c: torch.Tensor) -> torch.Tensor:
        q = [c.expand(B, D) for _ in range(V)]
        return moco_consistent_vcs(q, keys, uids, queue, crit)["J_raw"]

    neg_keys_p100 = qk if qk is not None else kall  # P100 construction: negatives online vs momentum keys (queue, or the batch's if no queue)

    def J_p100(c: torch.Tensor) -> float:  # positives online-online (s = <c, c> = 1), negatives online vs momentum keys
        tp = crit.score_matrix(torch.ones(B * V * (V - 1)))
        tq = crit.score_matrix(neg_keys_p100 @ c)
        return j_from(tp, tq)

    c_par = torch.nn.Parameter(cands.get("adversarial_current_minus_queue", cands["mean_current_key"]).clone())
    opt = torch.optim.Adam([c_par], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        (-J_p133(F.normalize(c_par, dim=0))).backward()
        opt.step()
    cands["gradient_ascent_on_J"] = F.normalize(c_par.detach(), dim=0)
    with torch.no_grad():
        res = {n: {"J_p133": float(J_p133(c)), "J_p100_construction": J_p100(c)} for n, c in cands.items()}
        best = max(res, key=lambda n: res[n]["J_p133"])
        nce = moco_consistent_infonce([cands[best].expand(B, D) for _ in range(V)], keys, uids, queue, float(cfg["objective"]["simclr_temperature"]))
        n_neg_row = nce["n_neg"] / (V * B)
    # sampling tolerance of J for the best constant: half-batch split (images 0..B/2-1 vs the rest)
    half = B // 2
    with torch.no_grad():
        c = cands[best]
        sub = [moco_consistent_vcs([c.expand(len(ix), D) for _ in range(V)], [k[ix] for k in keys], uids[ix], queue, crit)["J_raw"]
               for ix in (torch.arange(half), torch.arange(half, B))]
        se = float(abs(sub[0] - sub[1]) / 2)
    summary = {
        "config": a.config, "gate_steps": a.steps, "step_seconds_cpu": step_s, "lr_factor_at_start": "peak (step = warmup_steps)",
        "use_queue": queue is not None, "momentum_m": float(cfg["pairing"]["momentum_m"]),
        "queue_size": None if queue is None else queue.size, "queue_fill": 0 if queue is None else queue.filled, "train_J_first_last": [train_J[0], train_J[-1]], "train_J": train_J,
        "max_J_p133_constant_map": res[best]["J_p133"], "argmax": best, "half_batch_se": se,
        "J_p133_gradient_ascent": res["gradient_ascent_on_J"]["J_p133"],
        "J_p133_adversarial": res["adversarial_current_minus_queue"]["J_p133"] if "adversarial_current_minus_queue" in res else None,
        "max_J_p100_construction_same_candidates": max(r["J_p100_construction"] for r in res.values()),
        "infonce_best_constant": float(nce["loss"]), "infonce_chance_log1pN": math.log1p(n_neg_row),
        "pass_rule": "max J over constant maps <= 2 x half-batch SE (main-session decision 2026-10-05)",
        "pass_p133": res[best]["J_p133"] <= 2 * se, "pass_positive_control": max(r["J_p100_construction"] for r in res.values()) > 0.0,
        "candidates": {n: r for n, r in res.items() if not n.startswith("random_")},
        "random_J_p133_quantiles": [float(x) for x in torch.tensor([r["J_p133"] for n, r in res.items() if n.startswith("random_")]).quantile(torch.tensor([0.0, 0.5, 1.0]))],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("candidates", "train_J")}, indent=1))
    return 0 if (summary["pass_p133"] and summary["pass_positive_control"]) else 1


if __name__ == "__main__":
    sys.exit(main())
