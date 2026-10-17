"""V9-N2 weighted student training (package §03.F, §05.5): the host's per-anchor hinge terms (row = caption retrieval, column = image retrieval,
hard negatives = max violation, SGRAF default) are obtained per sample BEFORE aggregation; the objective over the merged pool A u U is

    L = (sum_{i in A} ell_i + sum_{i in U} w_i ell_i) / (n_A + sum_{i in U} w_i),

whose denominator is fixed over the whole pool before training; a minibatch estimates it by mean(v_i ell_i) / mean_v_full_pool with v_i = 1 on A
and v_i = w_i on U.  w is read from the locked weight file (sha256 verified), detached, never multiplied into logits; low-w pairs stay in the
negative pool (they are ordinary batch members).  Logged per step: weighted loss, the UNnormalised U risk (mean ell over the batch's U pairs),
batch ESS and total mass.  Checkpoint every epoch (model, optimiser, epoch, best, RNG states); resume from last.pt at the epoch boundary;
selection by the official dev rSum (ties -> earlier epoch); done.json when finished.  N0 (Unweighted+A) is the same loop with w = 1 on U.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .eval import recall_metrics, similarity_matrix
from .noise import EVAL_ONLY_SUFFIX
from .sgr_host import hinge_per_anchor
from .teacher import HostConfig, build_host, iterate_batches, seed_everything
from .weights import REF


def _rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


class WeightedStudent:
    def __init__(self, cfg: HostConfig, pool_rows: list[dict], weights: dict, collate, dev_images: np.ndarray, dev_captions: list[list[int]],
                 out_dir: str | Path, seed: int, lr_mult: float = 1.0, device: str = "cpu", weight_lock: dict | None = None, formal: bool = False,
                 loss_scale: str = "mean", grad_log_window: tuple | None = None, stop_epoch_steps: int | None = None):
        # loss_scale "mean" (default, frozen N2 runs): batch loss = mean_i(v_i l_i) / mean_v_full.  "sum" (A/B diagnostic, main session 2026-10-17):
        # sum_i(v_i l_i) / mean_v_full = batch_size x the default, i.e. the official SGRAF / NCR ContrastiveLoss scale (sum over anchors) for v = 1,
        # under which the locked grad_clip 2.0 acts as in the official code.  grad_log_window = (steps before, steps after) the warm-up -> hard switch:
        # every step in it writes grad_norms.jsonl (total grad norm before / after clipping, loss, U risk).  stop_epoch_steps: stop the LAST epoch
        # after that many steps and validate there (diagnostic only).
        if loss_scale not in ("mean", "sum"):
            raise ValueError(f"loss_scale {loss_scale!r}")
        self.loss_scale, self.grad_log_window, self.stop_epoch_steps = loss_scale, grad_log_window, stop_epoch_steps
        for r in pool_rows:
            if "original_image_id" in r or "mismatched" in r or EVAL_ONLY_SUFFIX in json.dumps(r):
                raise ValueError("a pool row carries evaluation-only information; the trainer refuses")
        if formal and cfg.to_verify:
            raise ValueError(f"HOST_LOCK fields still TO_VERIFY: {cfg.to_verify}; a formal unit cannot start")
        self.cfg, self.rows, self.collate, self.device, self.seed, self.lr_mult = cfg, pool_rows, collate, device, int(seed), float(lr_mult)
        self.dev_images, self.dev_captions = dev_images, dev_captions
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        v = np.array([1.0 if r["role"] != "U" else float(weights[r["pair_id"]]) for r in pool_rows], dtype=np.float64)
        if np.any(v < 0) or np.any(v > 1):
            raise ValueError("weights must lie in [0, 1]")
        self.v = v; self.is_u = np.array([r["role"] == "U" for r in pool_rows])
        n_A = int((~self.is_u).sum()); self.mean_v_full = float(v.sum() / len(v))                  # (n_A + sum_U w) / (n_A + n_U)
        self.manifest = {"n_pool": len(pool_rows), "n_A": n_A, "n_U": int(self.is_u.sum()), "sum_w_U": float(v[self.is_u].sum()),
                         "mean_v_full_pool": self.mean_v_full, "ess_U": REF.effective_sample_size(v[self.is_u]) if self.is_u.any() else 0.0,
                         "weight_lock_sha256": (weight_lock or {}).get("sha256"), "seed": self.seed, "lr_mult": self.lr_mult,
                         "cfg": asdict(cfg), "loss": f"warm-up mean hinge for {cfg.warmup_epochs} epochs (HOST_LOCK 57366c5), then max-violation hinge per anchor (row + column); fixed-pool normaliser", "formal": formal}
        (self.out / "run_manifest.json").write_text(json.dumps(self.manifest, indent=1))

    # ---------------------------------------------------------------------------------------------------------------- checkpoints
    def _save(self, host, opt, epoch, best, log, name="last.pt"):
        torch.save({"model": host.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best, "log": log, "rng": _rng_state()}, self.out / f"{name}.tmp")
        (self.out / f"{name}.tmp").replace(self.out / name)

    def validate(self, host) -> dict:
        sims = similarity_matrix(host, self.dev_images, self.dev_captions, device=self.device)
        return recall_metrics(sims, 5, "stable")

    def fit(self, epochs: int | None = None, log_every: int = 20) -> dict:
        if (self.out / "done.json").exists():
            return json.loads((self.out / "done.json").read_text())
        cfg = self.cfg; epochs = cfg.num_epochs if epochs is None else epochs
        seed_everything(self.seed); host = build_host(cfg).to(self.device); opt = torch.optim.Adam(host.parameters(), lr=cfg.learning_rate * self.lr_mult)
        start, best, log = 0, {"rsum": -1.0, "epoch": -1}, {"epochs": [], "steps": []}
        if (self.out / "last.pt").exists():
            ck = torch.load(self.out / "last.pt", map_location="cpu", weights_only=False)
            host.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start, best, log = ck["epoch"] + 1, ck["best"], ck["log"]; _set_rng_state(ck["rng"])
            log.setdefault("resumed_at", []).append(start)
        t0 = time.time(); step = len(log["steps"])
        for epoch in range(start, epochs):
            lr = cfg.learning_rate * self.lr_mult * (0.1 ** (epoch // cfg.lr_update))
            for g in opt.param_groups:
                g["lr"] = lr
            rng = np.random.default_rng([self.seed, 9_020, epoch]); host.train(); agg = {"loss": 0.0, "u_risk": 0.0, "n": 0, "u_n": 0}; in_epoch = 0
            n_batches = (sum(1 for _ in iterate_batches(len(self.rows), cfg.batch_size, np.random.default_rng([self.seed, 9_020, epoch])))
                         if self.grad_log_window is not None else 0)      # a separate generator: the training permutation is untouched
            for idx in iterate_batches(len(self.rows), cfg.batch_size, rng):
                ims, caps, lens = self.collate([self.rows[i] for i in idx]); v = torch.as_tensor(self.v[idx], dtype=torch.float32, device=self.device)
                sims = host(ims.to(self.device), caps.to(self.device), lens)
                # HOST_LOCK 57366c5: SGRAF with hardest negatives from step 0 collapses under torch 2.9; every SGR row runs NCR's warm-up loss
                # (mean over in-batch negatives, both directions) for the first `warmup_epochs`, then the max-violation hinge.  Weights apply in both phases.
                hard = epoch >= cfg.warmup_epochs
                er, ec = hinge_per_anchor(sims, cfg.margin, hard_negative=hard); ell = er + ec
                loss = ((v.detach() * ell).sum() if self.loss_scale == "sum" else (v.detach() * ell).mean()) / self.mean_v_full
                opt.zero_grad(set_to_none=True); loss.backward()
                if cfg.grad_clip > 0:
                    gn_t = torch.nn.utils.clip_grad_norm_(host.parameters(), cfg.grad_clip)   # total norm BEFORE clipping (returned anyway)
                else:
                    gn_t = None
                opt.step(); step += 1; in_epoch += 1
                u_mask = torch.as_tensor(self.is_u[idx], device=self.device)
                u_risk = float(ell.detach()[u_mask].mean()) if bool(u_mask.any()) else float("nan")
                agg["loss"] += float(loss.detach()); agg["n"] += 1
                if u_mask.any():
                    agg["u_risk"] += u_risk; agg["u_n"] += 1
                if self.grad_log_window is not None:
                    pre, post = self.grad_log_window
                    if (epoch == cfg.warmup_epochs - 1 and in_epoch > n_batches - pre) or (epoch == cfg.warmup_epochs and in_epoch <= post):
                        gnorm = float(gn_t) if gn_t is not None else float(torch.sqrt(sum((q.grad.detach() ** 2).sum() for q in host.parameters() if q.grad is not None)))
                        with open(self.out / "grad_norms.jsonl", "a") as fh:
                            fh.write(json.dumps({"epoch": epoch, "step_in_epoch": in_epoch, "step": step, "phase": "warmup" if epoch < cfg.warmup_epochs else "hard",
                                                 "loss": float(loss.detach()), "u_risk": u_risk, "grad_norm_pre_clip": gnorm,
                                                 "grad_norm_post_clip": min(gnorm, cfg.grad_clip) if cfg.grad_clip > 0 else gnorm,
                                                 "clipped": bool(cfg.grad_clip > 0 and gnorm > cfg.grad_clip), "loss_scale": self.loss_scale}) + "\n")
                if step % log_every == 0:
                    vb = self.v[idx]; log["steps"].append({"step": step, "epoch": epoch, "loss": float(loss), "u_risk_unnormalised": u_risk, "batch_mass": float(vb.sum()),
                                                            "batch_ess": REF.effective_sample_size(vb) if vb.sum() > 0 else 0.0, "lr": lr})
                if self.stop_epoch_steps is not None and epoch == epochs - 1 and in_epoch >= self.stop_epoch_steps:
                    break                                                # diagnostic: validate mid-epoch, right after the switch
            val = self.validate(host)
            rec = {"epoch": epoch, "lr": lr, "phase": "warmup_mean_hinge" if epoch < cfg.warmup_epochs else "max_violation", "steps_in_epoch": in_epoch, "mean_loss": agg["loss"] / max(agg["n"], 1),
                   "mean_u_risk_unnormalised": agg["u_risk"] / max(agg["u_n"], 1), "dev": val, "seconds": time.time() - t0}
            log["epochs"].append(rec)
            with open(self.out / "progress.jsonl", "a") as fh:                   # monitoring only (release gate / watcher); the checkpoint log is authoritative
                fh.write(json.dumps({"epoch": epoch, "phase": rec["phase"], "steps_in_epoch": in_epoch, "mean_loss": rec["mean_loss"], "dev_rsum": val["rsum"], "seconds": rec["seconds"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
            if val["rsum"] > best["rsum"]:                       # ties keep the earlier epoch
                best = {"rsum": val["rsum"], "epoch": epoch}; self._save(host, opt, epoch, best, log, name="best.pt")
            self._save(host, opt, epoch, best, log, name="last.pt")
        done = {"best": best, "epochs_run": epochs, "manifest_sha256": hashlib.sha256((self.out / "run_manifest.json").read_bytes()).hexdigest(), "seconds": time.time() - t0}
        (self.out / "train_log.json").write_text(json.dumps(log, indent=1)); (self.out / "done.json").write_text(json.dumps(done, indent=1))
        return done
