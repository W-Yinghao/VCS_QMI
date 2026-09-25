"""Training CLI: ``python -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml [--smoke-steps 100]``.

Ordinary joint maximization of the reference objective (spec §7): encoder, projector and critic (VCS only) are updated by
one backward pass of ``loss = -J``.  SimCLR / VICReg controls share every other component.  The trainer owns the data
contract, schedule, logging, checkpoints, failure artifacts and the in-training kNN/spectrum/critic diagnostics.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import SCHEMA_VERSION, __version__
from .checkpoint import atomic_torch_save, capture_rng, load_checkpoint, restore_rng, rng_fingerprint
from .config import ConfigError, dump_resolved, load_config
from .data.cifar import CifarTrain, load_cifar10_train
from .data.datasets import SSLTwoViewDataset, make_ssl_loader
from .data.splits import load_manifest
from .data.transforms import build_clean_transform, build_two_view_transform, two_view_transform_signature
from .diagnostics import critic_holdout, extract_features, knn_eval, spectrum_report
from .models import build_models, ema_update
from .objectives import compute_objective, compute_objective_target, critic_steps, forward_features, forward_features_target, pair_symmetric
from .optim import all_grads_finite, build_optimizer, grad_norms, set_lrs, verify_optimizer_coverage
from .schedule import lr_factor, warmup_steps_for
from .utils import (Timer, append_jsonl, apply_precision_policy, atomic_write_json, atomic_write_text, environment_info, git_info,
                    precision_flags, read_json, sha256_file, utc_now)

STATUS = ("NOT_RUN", "RUNNING", "COMPLETED", "FAILED_NUMERICAL", "FAILED_INFRA", "STOPPED_BUDGET")
EXIT_CODES = {"COMPLETED": 0, "FAILED_INFRA": 2, "FAILED_NUMERICAL": 3, "STOPPED_BUDGET": 143}


class StopRequested(Exception):
    pass


class _SignalFlag:
    def __init__(self) -> None:
        self.received: int | None = None

    def install(self) -> None:
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame) -> None:  # noqa: ANN001
        self.received = int(signum)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: dict[str, Any], *, run_dir: Path, data: CifarTrain, manifest: dict[str, Any], device: torch.device,
                 stage: str, smoke_steps: int | None = None, smoke_epoch_steps: int | None = None, stop_after_steps: int | None = None,
                 resume_from: Path | None = None, epoch_eval: bool = True, eval_num_workers: int | None = None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.data = data
        self.manifest = manifest
        self.device = device
        self.stage = stage
        self.method = cfg["run"]["method"]
        self.seed = int(cfg["run"]["seed"])
        self.smoke = smoke_steps is not None
        self.stop_after_steps = stop_after_steps
        self.epoch_eval = epoch_eval
        self.eval_num_workers = cfg["train"]["num_workers"] if eval_num_workers is None else eval_num_workers
        self.flag = _SignalFlag()
        self.resume_from = resume_from

        t = cfg["train"]
        self.batch = int(t["batch_size_images"])
        self.fit_uids = np.asarray(manifest["fit_uids"], dtype=np.int64)
        self.sel_uids = np.asarray(manifest["selection_uids"], dtype=np.int64)
        self.fit_mask = np.zeros(len(data), dtype=bool)
        self.fit_mask[self.fit_uids] = True
        full_steps_per_epoch = len(self.fit_uids) // self.batch  # drop_last
        if self.smoke:
            self.total_steps = int(smoke_steps)
            self.steps_per_epoch = int(smoke_epoch_steps or smoke_steps)
            if self.steps_per_epoch > full_steps_per_epoch:
                raise ConfigError("smoke pseudo-epoch cannot exceed one pass over fit")
            self.epochs = -(-self.total_steps // self.steps_per_epoch)
        else:
            if t["max_steps"] is not None:
                raise ConfigError("train.max_steps must be null for the frozen pilot; use --smoke-steps for smoke runs")
            self.steps_per_epoch = full_steps_per_epoch
            self.epochs = int(t["epochs"])
            self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup_steps = warmup_steps_for(warmup_epochs=t["warmup_epochs"], epochs=t["epochs"],
                                             steps_per_epoch=full_steps_per_epoch, total_steps=self.total_steps)
        self.min_lr_ratio = float(cfg["schedule"]["min_lr_ratio"])
        self.knn_epochs = set(int(e) for e in cfg["evaluation"]["knn_epochs"]) if not self.smoke else {0, self.epochs}
        self.checkpoint_epochs = set(int(e) for e in cfg["logging"]["checkpoint_epochs"]) if not self.smoke else {self.epochs}

        # dedicated RNG streams (spec §13.1)
        self.loader_gen = torch.Generator().manual_seed(self.seed)
        self.pair_gen = torch.Generator().manual_seed(self.seed + int(cfg["pairing"]["rng_seed_offset"]))

        self.two_view = build_two_view_transform(cfg["views"])
        self.clean = build_clean_transform(cfg["views"])
        self.dataset = SSLTwoViewDataset(data.data, self.fit_uids, self.two_view)

        # bookkeeping
        self.step = 0
        self.completed_epoch = 0
        self.seen_base_images = 0
        self.train_seconds = 0.0
        self.eval_seconds = 0.0
        self.status = "NOT_RUN"
        self.failure_reason: str | None = None
        self.collapse_streak = 0
        self.steady_step_times: list[float] = []
        self.git = git_info(repo_root())

    # -- paths ---------------------------------------------------------------------------------------------------------
    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "evaluations"

    # -- setup ---------------------------------------------------------------------------------------------------------
    def setup(self) -> None:
        for d in (self.ckpt_dir, self.log_dir, self.eval_dir, self.run_dir / "artifacts", self.run_dir / "failure"):
            d.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name
        atomic_write_text(self.run_dir / "config.resolved.yaml", dump_resolved(self.cfg))
        atomic_write_json(self.run_dir / "environment.json", {**environment_info(), "precision_flags": precision_flags(),
                                                                 "vcs_ssl_version": __version__, "schema": SCHEMA_VERSION})
        atomic_write_json(self.run_dir / "code_state.json", self.git)
        atomic_write_text(self.run_dir / "code_state.txt", "\n".join(f"{k}: {v}" for k, v in self.git.items()) + "\n")
        src = repo_root() / "sources.json"
        if src.is_file():
            shutil.copy2(src, self.run_dir / "sources.json")
        atomic_write_json(self.run_dir / "manifest.json", self.manifest, indent=None)
        atomic_write_text(self.run_dir / "manifest.sha256", self.manifest["manifest_sha256"] + "  manifest.json\n")

        built = build_models(self.cfg, seed=self.seed, device=self.device)
        self.encoder: nn.Module = built["encoder"]
        self.projector: nn.Module = built["projector"]
        self.critic: nn.Module | None = built["critic"]
        self.predictor: nn.Module | None = built.get("predictor")
        self.teacher = built.get("teacher")
        self.target_branch = self.cfg["train"].get("target_branch", "shared")
        self.init_hashes = built["init_hashes"]
        self.param_counts = built["params"]
        self.critic_impl = built.get("critic_impl")
        atomic_write_text(self.run_dir / "model_strings.txt", "\n\n".join(f"== {k} ==\n{v}" for k, v in built["model_strings"].items()))
        self.optimizer = build_optimizer(self.encoder, self.projector, self.critic, self.cfg["optimizer"], predictor=self.predictor)
        self.coverage = verify_optimizer_coverage(self.optimizer, self.encoder, self.projector, self.critic, self.predictor)

        self.loader = make_ssl_loader(self.dataset, batch_size=self.batch, num_workers=self.cfg["train"]["num_workers"],
                                      pin_memory=self.cfg["train"]["pin_memory"] and self.device.type == "cuda",
                                      persistent_workers=self.cfg["train"]["persistent_workers"], generator=self.loader_gen,
                                      drop_last=self.cfg["train"]["drop_last"])
        if len(self.loader) != len(self.fit_uids) // self.batch:
            raise RuntimeError("unexpected number of batches per pass")

        self.save_view_examples()
        self.run_manifest = {
            "run_id": self.run_id, "method": self.method, "stage": self.stage, "seed": self.seed, "smoke": self.smoke,
            "code_commit": self.git.get("commit"), "code_dirty": self.git.get("is_dirty"),
            "config_hash": self.cfg["_meta"]["config_hash"], "config_file_sha256": self.cfg["_meta"]["config_file_sha256"],
            "split_hash": self.manifest["manifest_sha256"], "physical_batch_images": self.batch, "views_per_image": 2,
            "K": int(self.cfg["pairing"]["k"]) if self.method == "vcs_qmi" else None,
            "pair_sampling": self.cfg["pairing"]["sampler"] if self.method == "vcs_qmi" else
            ("2B-2 in-batch negatives (NT-Xent)" if self.method == "simclr_matched" else "none (VICReg)"),
            "world_size": 1, "encoder_dim": int(self.cfg["model"]["h_dim"]), "projector_dim": int(self.cfg["model"]["projector"]["output_dim"]),
            "critic_params": self.param_counts["critic"] or None, "critic_impl": self.critic_impl, "encoder_params": self.param_counts["encoder"],
            "hparams": {"K": int(self.cfg["pairing"]["k"]), "critic_hidden_dims": list(self.cfg["model"]["critic"]["hidden_dims"]),
                        "critic_last_gain": self.cfg["model"]["critic"]["last_layer_xavier_gain"],
                        "critic_lr_multiplier": self.cfg["optimizer"]["critic_lr_multiplier"], "critic_weight_decay": self.cfg["optimizer"]["critic_weight_decay"],
                        "projector_hidden_dim": self.cfg["model"]["projector"]["hidden_dim"], "projector_output_dim": self.cfg["model"]["projector"]["output_dim"],
                        "critic_input_norm": self.cfg["model"]["normalization"]["vcs_and_simclr"], "batch_size_images": self.batch,
                        "lr": self.cfg["optimizer"]["lr"], "epochs": self.epochs, "warmup_epochs": self.cfg["train"]["warmup_epochs"],
                        "min_lr_ratio": self.min_lr_ratio, "crop_scale_min": self.cfg["views"]["random_resized_crop"]["scale"][0],
                        "color_jitter": [self.cfg["views"]["color_jitter"][k] for k in ("brightness", "contrast", "saturation", "hue")],
                        "gaussian_blur_p": self.cfg["views"]["gaussian_blur_p"],
                        "matrix_weight_decay": self.cfg["optimizer"]["matrix_weight_decay_encoder_projector"],
                        "critic_input": self.cfg["model"]["critic"]["input"], "pair_symmetric": pair_symmetric(self.cfg),
                        "critic_steps": critic_steps(self.cfg), "critic_feature_source": self.cfg["model"]["critic"].get("feature_source", "z"),
                        "negative_detach": self.cfg["pairing"]["negative_detach"], "target_branch": self.target_branch,
                        "predictor": self.predictor is not None, "projector_depth": self.cfg["model"]["projector"].get("depth", 2),
                        "cosine_scale_init": self.cfg["model"]["critic"].get("cosine_scale_init", 1.0)},
            "projector_params": self.param_counts["projector"], "objective_target": self.cfg["objective"]["target"],
            "steps_per_epoch": self.steps_per_epoch, "epochs": self.epochs, "intended_total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps, "min_lr_ratio": self.min_lr_ratio,
            "n_fit": int(len(self.fit_uids)), "n_selection": int(len(self.sel_uids)),
            "loader_generator_seed": self.seed, "pair_generator_seed": self.seed + int(self.cfg["pairing"]["rng_seed_offset"]),
            "init_hashes": self.init_hashes, "optimizer_coverage": self.coverage,
            "two_view_transform_sha256": two_view_transform_signature(self.cfg["views"]),
            "started_utc": utc_now(), "hostname": environment_info()["hostname"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        atomic_write_json(self.run_dir / "run_manifest.json", self.run_manifest)
        self.write_status("RUNNING")

        if self.resume_from is not None:
            self.resume(self.resume_from)
        else:
            if self.cfg["logging"]["save_initial_checkpoint"]:
                self.save_checkpoint(self.ckpt_dir / "initial.pt")
                if self.epoch_eval and 0 in self.knn_epochs:
                    self.epoch_evaluation(self.ckpt_dir / "initial.pt", epoch=0)
                self.append_epoch_record(epoch=0, epoch_stats=None, status="RUNNING")

    def save_view_examples(self, n: int = 16) -> None:
        from torchvision.utils import save_image  # noqa: PLC0415

        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == "cuda" else []):
            torch.manual_seed(self.seed + 999)
            rows = []
            uids = []
            for i in range(n):
                v1, v2, uid = self.dataset[i]
                rows.append(torch.stack((v1, v2)))
                uids.append(uid)
            grid = torch.cat(rows) * 0.5 + 0.5  # undo fixed normalization for viewing
            save_image(grid.clamp(0, 1), self.run_dir / "artifacts" / "view_examples.png", nrow=8)
            atomic_write_json(self.run_dir / "artifacts" / "view_examples_uids.json", {"uids": uids, "layout": "pairs (v1,v2) row-major"})

    # -- status / records ----------------------------------------------------------------------------------------------
    def write_status(self, status: str, **extra: Any) -> None:
        if status not in STATUS:
            raise ValueError(status)
        self.status = status
        atomic_write_json(self.run_dir / "status.json", {
            "run_id": self.run_id, "status": status, "stage": self.stage, "method": self.method, "seed": self.seed,
            "completed_epoch": self.completed_epoch, "optimizer_step": self.step, "intended_total_steps": self.total_steps,
            "epochs_intended": self.epochs, "failure_reason": self.failure_reason, "collapse_suspected": self.collapse_streak >= 2,
            "updated_utc": utc_now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **extra})

    def peak_memory(self) -> dict[str, float | None]:
        if self.device.type != "cuda":
            return {"peak_allocated_mb": None, "peak_reserved_mb": None}
        return {"peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 2**20,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / 2**20}

    def append_epoch_record(self, *, epoch: int, epoch_stats: dict[str, Any] | None, status: str, eval_res: dict[str, Any] | None = None) -> None:
        ev = eval_res or getattr(self, "_last_eval", None) or {}
        rec = {
            "run_id": self.run_id, "method": self.method, "stage": self.stage, "seed": self.seed, "epoch": epoch,
            "code_commit": self.git.get("commit"), "config_hash": self.cfg["_meta"]["config_hash"], "split_hash": self.manifest["manifest_sha256"],
            "physical_batch_images": self.batch, "views_per_image": 2, "K": self.run_manifest["K"], "pair_sampling": self.run_manifest["pair_sampling"],
            "world_size": 1, "encoder_dim": self.run_manifest["encoder_dim"], "projector_dim": self.run_manifest["projector_dim"],
            "critic_params": self.run_manifest["critic_params"], "objective_target": self.run_manifest["objective_target"],
            "optimizer_step": self.step, "seen_base_images": self.seen_base_images,
            "J_raw": None if epoch_stats is None else epoch_stats.get("J_raw"),
            "R_binary": None if epoch_stats is None else epoch_stats.get("R_binary"),
            "loss_epoch_mean": None if epoch_stats is None else epoch_stats.get("loss"),
            "nt_xent": None if epoch_stats is None else epoch_stats.get("nt_xent"),
            "vicreg_invariance": None if epoch_stats is None else epoch_stats.get("vicreg_invariance"),
            "vicreg_variance": None if epoch_stats is None else epoch_stats.get("vicreg_variance"),
            "vicreg_covariance": None if epoch_stats is None else epoch_stats.get("vicreg_covariance"),
            "heldout_J": ev.get("heldout_J") if ev.get("epoch") == epoch else None,
            "linear_val_top1_pct": None,  # filled by vcs_ssl.evaluate (separate frozen-feature protocol)
            "knn_val_top1_pct": ev.get("knn_val_top1_pct") if ev.get("epoch") == epoch else None,
            "h_effective_rank": ev.get("h_effective_rank") if ev.get("epoch") == epoch else None,
            "z_effective_rank": ev.get("z_effective_rank") if ev.get("epoch") == epoch else None,
            "collapse_suspected": self.collapse_streak >= 2,
            "train_seconds": self.train_seconds, "eval_seconds": self.eval_seconds, **self.peak_memory(),
            "status": status, "failure_reason": self.failure_reason, "utc": utc_now(),
        }
        append_jsonl(self.log_dir / "epochs.jsonl", rec)

    # -- checkpoints ---------------------------------------------------------------------------------------------------
    def checkpoint_payload(self) -> dict[str, Any]:
        rng = capture_rng(self.device, self.loader_gen, self.pair_gen)
        return {
            "encoder_state": self.encoder.state_dict(), "projector_state": self.projector.state_dict(),
            "critic_state": None if self.critic is None else self.critic.state_dict(),
            "predictor_state": None if self.predictor is None else self.predictor.state_dict(),
            "teacher_encoder_state": None if self.teacher is None else self.teacher["encoder"].state_dict(),
            "teacher_projector_state": None if self.teacher is None else self.teacher["projector"].state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": {"kind": "linear_warmup_cosine_per_step", "total_steps": self.total_steps, "warmup_steps": self.warmup_steps,
                                "min_lr_ratio": self.min_lr_ratio, "step": self.step},
            "run_id": self.run_id, "code_commit": self.git.get("commit"), "config_hash": self.cfg["_meta"]["config_hash"],
            "manifest_hash": self.manifest["manifest_sha256"], "method": self.method, "seed": self.seed, "stage": self.stage,
            "completed_epoch": self.completed_epoch, "optimizer_step": self.step, "intended_total_steps": self.total_steps,
            "steps_per_epoch": self.steps_per_epoch, "epochs": self.epochs, "batch_size_images": self.batch, "K": self.run_manifest["K"],
            "seen_base_images": self.seen_base_images, "train_seconds": self.train_seconds, "eval_seconds": self.eval_seconds,
            **rng, "precision_flags": precision_flags(), "model_hparams": self.cfg["model"], "best_metric_policy": None,
            "init_hashes": self.init_hashes, "smoke": self.smoke, "saved_utc": utc_now(), "torch_version": torch.__version__,
        }

    def save_checkpoint(self, path: Path) -> None:
        atomic_torch_save(self.checkpoint_payload(), path)

    def resume(self, path: Path) -> None:
        ck = load_checkpoint(path, map_location="cpu")
        for key, mine in (("config_hash", self.cfg["_meta"]["config_hash"]), ("manifest_hash", self.manifest["manifest_sha256"]),
                          ("intended_total_steps", self.total_steps), ("steps_per_epoch", self.steps_per_epoch), ("method", self.method),
                          ("seed", self.seed), ("batch_size_images", self.batch), ("K", self.run_manifest["K"]), ("epochs", self.epochs)):
            if ck.get(key) != mine:
                raise ConfigError(f"resume refused: checkpoint {key}={ck.get(key)!r} differs from current {mine!r}")
        if ck["init_hashes"] != self.init_hashes:
            raise ConfigError("resume refused: initial weight hashes differ (seed/init mismatch)")
        self.encoder.load_state_dict(ck["encoder_state"])
        self.projector.load_state_dict(ck["projector_state"])
        if self.critic is not None:
            self.critic.load_state_dict(ck["critic_state"])
        if self.predictor is not None:
            self.predictor.load_state_dict(ck["predictor_state"])
        if self.teacher is not None:
            self.teacher["encoder"].load_state_dict(ck["teacher_encoder_state"])
            self.teacher["projector"].load_state_dict(ck["teacher_projector_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        restore_rng(ck, self.loader_gen, self.pair_gen)
        self.completed_epoch = int(ck["completed_epoch"])
        self.step = int(ck["optimizer_step"])
        self.seen_base_images = int(ck["seen_base_images"])
        self.train_seconds = float(ck["train_seconds"])
        self.eval_seconds = float(ck["eval_seconds"])
        if self.step != self.completed_epoch * self.steps_per_epoch:
            raise ConfigError("resume refused: checkpoint is not at an epoch boundary")
        self.resumed_from = {"path": str(path), "sha256": sha256_file(path), "completed_epoch": self.completed_epoch, "optimizer_step": self.step}
        append_jsonl(self.log_dir / "resume_events.jsonl", {**self.resumed_from, "utc": utc_now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})

    # -- in-training evaluation (kNN / spectrum / critic hold-out) ------------------------------------------------------
    def epoch_evaluation(self, ckpt_path: Path, *, epoch: int) -> dict[str, Any]:
        """Evaluate a *fresh copy* loaded from ``ckpt_path`` (spec §10.1); training instance and training RNG untouched."""
        before = rng_fingerprint(capture_rng(self.device, self.loader_gen, self.pair_gen))
        ecfg = self.cfg["evaluation"]
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        result: dict[str, Any] = {"epoch": epoch, "checkpoint": ckpt_path.name, "checkpoint_sha256": sha256_file(ckpt_path)}
        with Timer(self.device) as t, torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            built = build_models(self.cfg, seed=self.seed, device=self.device)
            ck = load_checkpoint(ckpt_path)
            enc, proj, crit = built["encoder"], built["projector"], built["critic"]
            enc.load_state_dict(ck["encoder_state"])
            proj.load_state_dict(ck["projector_state"])
            if crit is not None:
                crit.load_state_dict(ck["critic_state"])
            ev_pred, ev_teacher = built.get("predictor"), built.get("teacher")
            if ev_pred is not None and ck.get("predictor_state") is not None:
                ev_pred.load_state_dict(ck["predictor_state"])
            if ev_teacher is not None and ck.get("teacher_encoder_state") is not None:
                ev_teacher["encoder"].load_state_dict(ck["teacher_encoder_state"])
                ev_teacher["projector"].load_state_dict(ck["teacher_projector_state"])
            eval_seed = 7_000_000 + epoch
            sel = extract_features(enc, proj, self.data.data, self.data.targets, self.sel_uids, self.clean, device=self.device,
                                   num_workers=self.eval_num_workers, seed=eval_seed, l2_eps=self.cfg["model"]["normalization"]["eps"])
            fit = extract_features(enc, None, self.data.data, self.data.targets, self.fit_uids, self.clean, device=self.device,
                                   num_workers=self.eval_num_workers, seed=eval_seed + 1)
            kcfg = ecfg["knn"]
            knn = knn_eval(fit["h"], fit["labels"], sel["h"], sel["labels"], k=kcfg["k"], temperature=kcfg["temperature"],
                           chunk=kcfg["query_chunk"], device=self.device)
            result["knn"] = knn
            result["knn_val_top1_pct"] = knn["knn_val_top1_pct"]
            scfg = ecfg["spectrum"]
            if scfg["enabled"]:
                n = int(scfg["selection_first_sorted_ids"])
                sub = {k: sel[k][:n] for k in ("h", "p_raw", "z_l2")}
                spec = spectrum_report(sub, names=tuple(scfg["features"]), center=scfg["center"], ddof=scfg["covariance_ddof"])
                spec["_uids_sha256"] = __import__("hashlib").sha256(self.sel_uids[:n].tobytes()).hexdigest()
                result["spectrum"] = spec
                result["h_effective_rank"] = spec["h"]["effective_rank"]
                result["z_effective_rank"] = spec["z_l2"]["effective_rank"]
                flag = spec["h"]["collapse_flag"] or spec["z_l2"]["collapse_flag"]
                self.collapse_streak = self.collapse_streak + 1 if flag else 0
                result["collapse_flag_now"] = flag
                result["COLLAPSE_SUSPECTED"] = self.collapse_streak >= 2
            cv = ecfg["critic_validation"]
            if cv["enabled"] and crit is not None:
                ch = critic_holdout(enc, proj, crit, self.data.data, self.sel_uids, self.two_view, device=self.device, feature_source=self.cfg["model"]["critic"].get("feature_source", "z"),
                                    batch_size=cv["batch_size"], repeats=cv["repeats"], rng_seed=cv["rng_seed"], k=int(self.cfg["pairing"]["k"]),
                                    num_workers=self.eval_num_workers, l2_eps=self.cfg["model"]["normalization"]["eps"],
                                    normalize_input=self.cfg["model"]["normalization"]["vcs_and_simclr"] != "none", symmetric=pair_symmetric(self.cfg),
                                    target_branch=self.target_branch, teacher=ev_teacher, predictor=ev_pred)
                result["critic_holdout"] = ch
                result["heldout_J"] = ch["heldout_J_mean"]
            del enc, proj, crit, built, ck, sel, fit, ev_pred, ev_teacher
        after = rng_fingerprint(capture_rng(self.device, self.loader_gen, self.pair_gen))
        result["training_rng_untouched"] = before == after
        if before != after:
            raise RuntimeError(f"evaluation consumed training RNG: {[k for k in before if before[k] != after[k]]}")
        result["seconds"] = t.elapsed
        self.eval_seconds += t.elapsed
        tag = f"epoch_{epoch:03d}"
        atomic_write_json(self.eval_dir / f"knn_{tag}.json", {k: v for k, v in result.items() if k not in ("spectrum", "critic_holdout")})
        if "spectrum" in result:
            atomic_write_json(self.eval_dir / f"spectrum_{tag}.json", result["spectrum"])
        if "critic_holdout" in result:
            atomic_write_json(self.eval_dir / f"critic_holdout_{tag}.json", result["critic_holdout"])
        self._last_eval = result
        return result

    # -- training ------------------------------------------------------------------------------------------------------
    def train_step(self, x1: torch.Tensor, x2: torch.Tensor, uids: torch.Tensor, log_this: bool) -> dict[str, Any]:
        if uids.unique().numel() != uids.numel():
            raise RuntimeError("duplicate UID inside a batch")
        if not self.fit_mask[uids.numpy()].all():
            raise RuntimeError("batch contains a UID outside the fit split")
        factor = lr_factor(self.step, self.total_steps, self.warmup_steps, self.min_lr_ratio)
        lrs = set_lrs(self.optimizer, factor)
        self.encoder.train(); self.projector.train()
        if self.critic is not None:
            self.critic.train()
        self.optimizer.zero_grad(set_to_none=True)
        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)
        if self.target_branch == "shared":
            feats = forward_features(self.encoder, self.projector, x1, x2, eps=self.cfg["model"]["normalization"]["eps"])
        else:
            if self.predictor is not None:
                self.predictor.train()
            feats = forward_features_target(self.encoder, self.projector, x1, x2, self.cfg["model"]["normalization"]["eps"],
                                            target_branch=self.target_branch, teacher=self.teacher, predictor=self.predictor)
        n_extra = critic_steps(self.cfg) - 1
        if n_extra > 0 and self.critic is not None:
            # named variant: critic-only updates on detached features before the joint step (encoder/projector grads stay None)
            det = {k: v.detach() for k, v in feats.items()}
            for _ in range(n_extra):
                self.optimizer.zero_grad(set_to_none=True)
                cobj = (compute_objective(self.method, det, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen) if self.target_branch == "shared"
                        else compute_objective_target(det, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen))
                if not torch.isfinite(cobj["loss"]):
                    raise FloatingPointError(f"non-finite critic-only loss at step {self.step}")
                cobj["loss"].backward()
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        obj = (compute_objective(self.method, feats, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen) if self.target_branch == "shared"
               else compute_objective_target(feats, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen))
        loss = obj["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {self.step}")
        loss.backward()
        gn = grad_norms(self.encoder, self.projector, self.critic, self.predictor) if (log_this or self.step == 0) else {}
        if not all_grads_finite(self.optimizer):
            raise FloatingPointError(f"non-finite gradient at step {self.step}")
        if self.step == 0:
            self.first_step_gradient_check(gn)
        self.optimizer.step()
        if self.teacher is not None:
            ema_update(self.teacher, self.encoder, self.projector)
        cos_extra = ({"cos_scale": float(self.critic.scale.detach()), "cos_bias": float(self.critic.bias.detach())}
                     if self.critic is not None and hasattr(self.critic, "scale") and hasattr(self.critic, "bias") else {})
        out = {"loss": float(loss.detach()), **obj["stats"], **cos_extra, "shift": obj["shift"], "n_pos": obj["n_pos"], "n_neg": obj["n_neg"],
               "lr_factor": factor, **{f"lr_{k}": v for k, v in lrs.items()}, **gn}
        return out

    def first_step_gradient_check(self, gn: dict[str, Any]) -> None:
        """Spec §7.2: every module has a nonzero finite gradient on the first real step; parameters then change."""
        problems = []
        critic_on_h = self.method == "vcs_qmi" and self.cfg["model"]["critic"].get("feature_source", "z") == "h_l2"
        for name, m in (("encoder", self.encoder), ("projector", self.projector), ("critic", self.critic), ("predictor", self.predictor)):
            if m is None or (name == "projector" and critic_on_h):  # projector receives no gradient when the critic reads h (disclosed)
                continue
            g = gn.get(f"grad_norm_{name}")
            if g is None or not np.isfinite(g) or g <= 0:
                problems.append(f"{name} gradient norm {g}")
        snap = {n: p.detach().clone() for n, p in list(self.encoder.named_parameters())[:2]}
        self._first_step_snapshot = snap
        if problems:
            self.failure_reason = "first-step gradient check failed: " + "; ".join(problems)
            raise RuntimeError(self.failure_reason)
        atomic_write_json(self.run_dir / "first_step_gradients.json", {**gn, "check": "PASS", "utc": utc_now()})

    def verify_first_step_update(self) -> None:
        snap = getattr(self, "_first_step_snapshot", None)
        if snap is None:
            return
        changed = {n: not torch.equal(snap[n], dict(self.encoder.named_parameters())[n].detach()) for n in snap}
        atomic_write_json(self.run_dir / "first_step_update.json", {"encoder_params_changed": changed, "utc": utc_now()})
        if not any(changed.values()):
            raise RuntimeError("no encoder parameter changed after the first optimizer step")
        self._first_step_snapshot = None

    def save_failure_context(self, exc: BaseException, batch: tuple | None, step_out: dict[str, Any] | None) -> None:
        if not self.cfg["logging"]["keep_failure_artifacts"]:
            return
        payload = {"exception": repr(exc), "traceback": traceback.format_exc(), "step": self.step, "epoch": self.completed_epoch + 1,
                   "shift": None if step_out is None else step_out.get("shift"), "last_step_stats": step_out,
                   "uids": None if batch is None else batch[2].tolist(), **capture_rng(self.device, self.loader_gen, self.pair_gen),
                   "encoder_state": self.encoder.state_dict(), "projector_state": self.projector.state_dict(),
                   "critic_state": None if self.critic is None else self.critic.state_dict(), "config_hash": self.cfg["_meta"]["config_hash"]}
        atomic_torch_save(payload, self.run_dir / "failure" / f"failure_step_{self.step:06d}.pt")
        atomic_write_json(self.run_dir / "failure" / f"failure_step_{self.step:06d}.json",
                          {k: v for k, v in payload.items() if k in ("exception", "traceback", "step", "epoch", "shift", "last_step_stats", "uids")})

    def run(self) -> str:
        self.flag.install()
        batch = None
        out: dict[str, Any] | None = None
        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            for epoch in range(self.completed_epoch + 1, self.epochs + 1):
                sums: dict[str, float] = {}
                counts: dict[str, int] = {}
                step_times: list[float] = []
                interval_t0 = time.perf_counter()
                it = iter(self.loader)
                steps_this_epoch = min(self.steps_per_epoch, self.total_steps - self.step)
                with Timer(self.device) as epoch_timer:
                    for _ in range(steps_this_epoch):
                        if self.flag.received is not None:
                            raise StopRequested(f"signal {self.flag.received}")
                        if self.stop_after_steps is not None and self.step >= self.stop_after_steps:
                            raise StopRequested(f"--stop-after-steps {self.stop_after_steps}")
                        t_wait = time.perf_counter()
                        batch = next(it)
                        data_wait = time.perf_counter() - t_wait
                        log_this = (self.step % self.cfg["logging"]["step_interval"] == 0) or (self.step == self.total_steps - 1)
                        with Timer(self.device) as st:
                            out = self.train_step(batch[0], batch[1], batch[2], log_this)
                        if self.step == 0:
                            self.verify_first_step_update()
                        step_times.append(st.elapsed)
                        if self.step >= self.warmup_steps:
                            self.steady_step_times.append(st.elapsed)
                        self.step += 1
                        self.seen_base_images += int(batch[0].shape[0])
                        for k, v in out.items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("shift", "n_pos", "n_neg"):
                                sums[k] = sums.get(k, 0.0) + float(v)
                                counts[k] = counts.get(k, 0) + 1
                        if log_this:
                            rec = {"step": self.step - 1, "epoch": epoch, **out, "base_images": int(batch[0].shape[0]), "views": 2 * int(batch[0].shape[0]),
                                   "seen_base_images": self.seen_base_images, "step_seconds": st.elapsed, "data_wait_seconds": data_wait,
                                   "interval_mean_step_seconds": float(np.mean(step_times[-self.cfg["logging"]["step_interval"]:])),
                                   "interval_wall_seconds": time.perf_counter() - interval_t0, **self.peak_memory(), "utc": utc_now()}
                            interval_t0 = time.perf_counter()
                            append_jsonl(self.log_dir / "steps.jsonl", rec)
                            print(f"[{self.run_id}] ep {epoch} step {self.step - 1}/{self.total_steps} loss {out['loss']:.5f} "
                                  f"J {out.get('J_raw')} nt {out.get('nt_xent')} inv {out.get('vicreg_invariance')} "
                                  f"lr {out['lr_enc_proj_matrix']:.2e} {st.elapsed * 1000:.0f}ms", flush=True)
                        batch = None
                del it
                self.train_seconds += epoch_timer.elapsed
                self.completed_epoch = epoch
                epoch_stats = {k: sums[k] / counts[k] for k in sums}
                epoch_stats["epoch_seconds"] = epoch_timer.elapsed
                epoch_stats["mean_step_seconds"] = float(np.mean(step_times)) if step_times else None
                self.save_checkpoint(self.ckpt_dir / "last.pt")
                if epoch in self.checkpoint_epochs:
                    shutil.copy2(self.ckpt_dir / "last.pt", self.ckpt_dir / f"epoch_{epoch:03d}.pt")
                ev = None
                if self.epoch_eval and epoch in self.knn_epochs:
                    ev = self.epoch_evaluation(self.ckpt_dir / "last.pt", epoch=epoch)
                self.append_epoch_record(epoch=epoch, epoch_stats=epoch_stats, status="RUNNING", eval_res=ev)
                self.write_status("RUNNING")
                print(f"[{self.run_id}] epoch {epoch}/{self.epochs} done in {epoch_timer.elapsed:.1f}s; "
                      f"kNN {None if ev is None else ev.get('knn_val_top1_pct')}", flush=True)
            self.finish("COMPLETED")
        except StopRequested as e:
            self.failure_reason = f"stopped: {e}; last completed epoch {self.completed_epoch} is in last.pt; mid-epoch progress discarded"
            self.finish("STOPPED_BUDGET")
        except FloatingPointError as e:
            self.failure_reason = f"numerical: {e}"
            self.save_failure_context(e, batch, out)
            self.finish("FAILED_NUMERICAL")
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            self.failure_reason = f"CUDA OOM: {e}"[:500]
            self.save_failure_context(e, batch, out)
            self.finish("FAILED_INFRA")
        except Exception as e:  # noqa: BLE001
            self.failure_reason = f"{type(e).__name__}: {e}"[:1000]
            traceback.print_exc()
            self.save_failure_context(e, batch, out)
            self.finish("FAILED_INFRA")
        return self.status

    def finish(self, status: str) -> None:
        steady = self.steady_step_times
        mem = self.peak_memory()
        summary = {
            **self.run_manifest, "status": status, "failure_reason": self.failure_reason, "completed_epoch": self.completed_epoch,
            "optimizer_step": self.step, "seen_base_images": self.seen_base_images, "seen_views": 2 * self.seen_base_images,
            "train_seconds": self.train_seconds, "eval_seconds": self.eval_seconds,
            "steady_state_step_seconds_mean": float(np.mean(steady)) if steady else None,
            "steady_state_images_per_s": (self.batch / float(np.mean(steady))) if steady else None,
            "steady_state_views_per_s": (2 * self.batch / float(np.mean(steady))) if steady else None,
            **mem, "collapse_suspected": self.collapse_streak >= 2, "finished_utc": utc_now(),
            "last_eval": {k: v for k, v in (getattr(self, "_last_eval", None) or {}).items() if k not in ("spectrum", "critic_holdout", "knn")},
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        self.write_status(status)
        self.append_epoch_record(epoch=self.completed_epoch, epoch_stats=None, status=status)


# ----------------------------------------------------------------------------------------------------------------------
def default_run_id(cfg: dict[str, Any], stage: str) -> str:
    return f"{cfg['run']['method']}_seed{cfg['run']['seed']}_{stage}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VCS-QMI SSL pilot trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke-steps", type=int, default=None, help="P2 smoke: total optimizer steps (own schedule horizon)")
    ap.add_argument("--smoke-epoch-steps", type=int, default=None, help="P2 smoke: steps per pseudo-epoch for checkpoint/resume tests")
    ap.add_argument("--stop-after-steps", type=int, default=None, help="debug: stop cleanly (STOPPED_BUDGET) after this many steps")
    ap.add_argument("--resume", default=None, help="path to last.pt (same config/manifest/horizon; epoch-boundary resume)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--stage", default=None, help="override stage label (default: config run.stage, or P2_smoke)")
    ap.add_argument("--no-epoch-eval", action="store_true", help="skip in-training kNN/spectrum/critic diagnostics (tests only)")
    ap.add_argument("--allow-cpu", action="store_true", help="tests only")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    apply_precision_policy(cfg["train"])
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        if torch.cuda.device_count() != 1:
            print(f"WARNING: {torch.cuda.device_count()} GPUs visible; using cuda:0 only (single-GPU policy)", flush=True)
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        print("ERROR: no CUDA device visible; the pilot trains on one GPU via SLURM", file=sys.stderr)
        return EXIT_CODES["FAILED_INFRA"]
    stage = args.stage or ("P2_smoke" if args.smoke_steps else cfg["run"]["stage"])
    out_root = Path(cfg["run"]["output_root"])
    if args.resume:
        run_dir = Path(args.resume).resolve().parent.parent
        if args.run_id:
            run_dir = out_root / args.run_id
    else:
        run_dir = out_root / (args.run_id or default_run_id(cfg, stage))
        if run_dir.exists() and any(run_dir.iterdir()) and not cfg["run"]["overwrite"]:
            print(f"ERROR: run dir {run_dir} exists and overwrite=false", file=sys.stderr)
            return EXIT_CODES["FAILED_INFRA"]
    run_dir.mkdir(parents=True, exist_ok=True)
    data = load_cifar10_train(cfg["data"]["root"])
    manifest = load_manifest(cfg["data"]["manifest"], expected_seed=cfg["data"]["split_seed"], expected_val_per_class=cfg["data"]["val_per_class"])
    if manifest["raw_file_hashes"] != data.file_hashes:
        raise ConfigError("manifest raw file hashes differ from the data on disk")
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=manifest, device=device, stage=stage, smoke_steps=args.smoke_steps,
                 smoke_epoch_steps=args.smoke_epoch_steps, stop_after_steps=args.stop_after_steps,
                 resume_from=Path(args.resume) if args.resume else None, epoch_eval=not args.no_epoch_eval)
    try:
        tr.setup()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        tr.failure_reason = f"setup: {type(e).__name__}: {e}"[:1000]
        tr.run_id = run_dir.name
        tr.write_status("FAILED_INFRA")
        return EXIT_CODES["FAILED_INFRA"]
    status = tr.run()
    print(f"[{tr.run_id}] final status {status}; run_dir={run_dir}", flush=True)
    return EXIT_CODES.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
