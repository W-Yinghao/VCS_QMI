"""Training CLI: ``python -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml [--smoke-steps 100]``.

Ordinary joint maximization of the reference objective (spec §7): encoder, projector and critic (VCS only) are updated by
one backward pass of ``loss = -J``.  SimCLR / VICReg controls share every other component.  The trainer owns the data
contract, schedule, logging, checkpoints, failure artifacts and the in-training kNN/spectrum/critic diagnostics.
"""
from __future__ import annotations

import copy

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
from .data.cifar import CifarTrain, load_train_partition, n_classes_of
from .data.datasets import SSLMultiViewDataset, SSLTwoViewDataset, make_ssl_loader
from .kernel_cs import median_pairwise_distance
from .data.splits import load_manifest
from .data.transforms import build_clean_transform, build_two_view_transform, two_view_transform_signature
from .diagnostics import critic_holdout, extract_features, knn_eval, spectrum_report
from .models import build_models, ema_update
from .objectives import (NegativeQueue, compute_objective, compute_objective_target, compute_objective_views, critic_feature_dim, critic_steps,
                         critic_input_key, forward_features, forward_features_target, forward_features_views, pair_symmetric, uses_queue)
from .objectives import KeyUidQueue, compute_objective_moco_consistent  # P133
from .optim import all_grads_finite, build_optimizer, grad_norms, has_trainable_params, set_lrs, verify_optimizer_coverage
from .schedule import lr_factor, warmup_steps_for
from .utils import (Timer, append_jsonl, apply_precision_policy, atomic_write_json, atomic_write_text, environment_info, git_info,
                    precision_flags, read_json, sha256_file, utc_now, sha256_bytes)

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
        self.n_classes = n_classes_of(data)  # 10 (CIFAR-10) or 100 (CIFAR-100, P91); drives kNN votes and the probe head
        if manifest.get("n_classes") is not None and int(manifest["n_classes"]) != self.n_classes:
            raise ConfigError(f"manifest n_classes {manifest['n_classes']} differs from the data ({self.n_classes})")
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
        self.n_views = int(cfg["views"]["count"])
        self.dataset = (SSLTwoViewDataset(data.data, self.fit_uids, self.two_view) if self.n_views == 2
                        else SSLMultiViewDataset(data.data, self.fit_uids, self.two_view, self.n_views))

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
        # P104 (package v3 §9.3): cost counters carried by checkpoints (new key; older checkpoints resume with zeros + a flag)
        self.cost = {"positive_pairs": 0, "negative_pairs": 0, "critic_pair_evaluations": 0, "encoder_updates": 0, "critic_updates": 0,
                     "refresh_seconds": 0.0}
        self.git = git_info(repo_root())

    def _p104(self) -> bool:
        c, p = self.cfg["model"]["critic"], self.cfg["pairing"]
        return any(k in c for k in ("affine_mode", "cosine_bias_init", "noise_repeats", "noise_eval_repeats")) or any(k in p for k in ("pair_scope", "all_view_chunk"))

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
        # named variant (wave-2 C-T): FIFO queue of detached features from previous steps as the negative pool
        self.neg_queue: NegativeQueue | None = None
        self.queue_fallback_steps = 0
        if uses_queue(self.cfg):
            self.queue_key = critic_input_key(self.cfg) if self.method == "vcs_qmi" else "z_l2"
            qdim = critic_feature_dim(self.cfg) if self.method == "vcs_qmi" else int(self.cfg["model"]["projector"]["output_dim"])
            self.neg_queue = NegativeQueue(int(self.cfg["pairing"]["queue_size"]), qdim, device=self.device)
        # P100: momentum (EMA) copy of encoder + projector that produces the queue keys (not optimised, no gradient).  Its BatchNorm runs
        # in train mode with its *own* running buffers (MoCo convention; single GPU, so no shuffle-BN); its parameters follow
        # theta_k <- m * theta_k + (1 - m) * theta_q after every optimiser step.  Keys = the critic-input features of view 0 (one key per image).
        self.momentum = bool(self.cfg["pairing"].get("momentum_encoder", False))
        self.key_model: dict[str, Any] | None = None
        if self.momentum:
            self.momentum_m = float(self.cfg["pairing"].get("momentum_m", 0.99))
            self.key_model = {"encoder": copy.deepcopy(self.encoder), "projector": copy.deepcopy(self.projector)}
            for mod in self.key_model.values():
                for prm in mod.parameters():
                    prm.requires_grad_(False)
        # ---- P133 (MoCo-consistent momentum queue; pairing.moco_consistent): every pair is (online query, momentum key).  The key
        # encoder runs in EVAL mode (BN = its running buffers, which follow the online buffers by the same EMA as the parameters), so each key
        # is a per-sample function of the momentum network and carries no batch-statistics signature (single-GPU substitute for shuffle-BN).
        # The P100 queue (self.neg_queue) is replaced by a key + uid ring (self.p133_queue); keys of all V views are computed, view 0 is enqueued.
        self.moco = bool(self.cfg["pairing"].get("moco_consistent", False))
        self.p133_queue: KeyUidQueue | None = None
        self.p133_use_queue = bool(self.cfg["pairing"].get("moco_use_queue", True))
        if self.moco:
            self.neg_queue = None
            if self.p133_use_queue:  # variant (a) (moco_use_queue false): negatives = current-batch momentum keys only, no ring
                self.p133_queue = KeyUidQueue(int(self.cfg["pairing"]["queue_size"]), critic_feature_dim(self.cfg) if self.method == "vcs_qmi"
                                              else int(self.cfg["model"]["projector"]["output_dim"]), device=self.device)
        # ---- end P133

        self.loader = make_ssl_loader(self.dataset, batch_size=self.batch, num_workers=self.cfg["train"]["num_workers"],
                                      pin_memory=self.cfg["train"]["pin_memory"] and self.device.type == "cuda",
                                      persistent_workers=self.cfg["train"]["persistent_workers"], generator=self.loader_gen,
                                      drop_last=self.cfg["train"]["drop_last"])
        if len(self.loader) != len(self.fit_uids) // self.batch:
            raise RuntimeError("unexpected number of batches per pass")

        self.save_view_examples()
        self.calibration = self.calibrate_cosine_bias() if self.cfg["model"]["critic"].get("cosine_bias_calibrate", False) else None
        self.kernel_sigma: float | None = None  # cs_kernel_native only (Server Spec v2 §3.2)
        self.bandwidth_calibration = self.calibrate_bandwidths()  # cs_kernel_native / rff_tanh; None otherwise
        self.run_manifest = {
            "run_id": self.run_id, "method": self.method, "stage": self.stage, "seed": self.seed, "smoke": self.smoke,
            "code_commit": self.git.get("commit"), "code_dirty": self.git.get("is_dirty"),
            "config_hash": self.cfg["_meta"]["config_hash"], "config_file_sha256": self.cfg["_meta"]["config_file_sha256"],
            "split_hash": self.manifest["manifest_sha256"], "physical_batch_images": self.batch, "views_per_image": 2,
            "K": int(self.cfg["pairing"]["k"]) if self.method == "vcs_qmi" else None,
            "pair_sampling": ((f"momentum_key_queue{self.cfg['pairing']['queue_size']}" if self.momentum else f"queue{self.cfg['pairing']['queue_size']}_detached") if uses_queue(self.cfg) else self.cfg["pairing"]["sampler"]) if self.method == "vcs_qmi" else
            ((("2B-2 in-batch + momentum-key queue negatives (NT-Xent)" if self.momentum else "MoCo-style queue negatives (NT-Xent, no momentum encoder)") if uses_queue(self.cfg) else "2B-2 in-batch negatives (NT-Xent)") if self.method == "simclr_matched" else ("kernel CS plug-in on all B x B cross pairs of the batch (native; no negative branch)" if self.method == "cs_kernel_native" else "none (VICReg)")),
            "negative_source": self.cfg["pairing"].get("negative_source", "cyclic"),
            **({"momentum_queue": {"m": self.momentum_m, "queue_size": int(self.cfg["pairing"]["queue_size"]), "keys": "view 0 of each image, critic-input space",
                                    "key_bn": "momentum encoder in train mode with its own BN buffers (single GPU, no shuffle-BN)",
                                    "vcs_negatives": "K keys per anchor from the queue, uniform with replacement (dedicated CPU pair generator)",
                                    "simclr_negatives": "2B-2 in-batch + all queue keys (additional)"}} if self.momentum else {}), "queue_size": int(self.cfg["pairing"]["queue_size"]) if uses_queue(self.cfg) else None,
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
                        "cosine_scale_init": self.cfg["model"]["critic"].get("cosine_scale_init", 1.0),
                        "cosine_scale_fixed": self.cfg["model"]["critic"].get("cosine_scale_fixed", False),
                        "cosine_bias_calibrate": self.cfg["model"]["critic"].get("cosine_bias_calibrate", False),
                        "projector_output_bn": self.cfg["model"]["projector"]["output_batchnorm"], "n_views": self.n_views,
                        "projector_kind": self.cfg["model"]["projector"].get("kind", "mlp"),
                        "rff_features": self.cfg["model"]["critic"].get("rff_features"), "rff_bandwidth_multiple": self.cfg["model"]["critic"].get("rff_bandwidth_multiple"),
                        "kernel_cs_bandwidth_multiple": self.cfg["objective"].get("kernel_cs_bandwidth_multiple"), "kernel_cs_chunk": self.cfg["objective"].get("kernel_cs_chunk"),
                        "p95_residual_lambda": self.cfg["model"]["critic"].get("residual_lambda"),
                        "p95_observation_noise_tau": self.cfg["model"]["critic"].get("observation_noise_tau"),
                        "p95_refresh_every_epochs": self.cfg["model"]["critic"].get("refresh_every_epochs"),
                        "p95_refresh_batches": self.cfg["model"]["critic"].get("refresh_batches"), "objective_loss": self.cfg["objective"]["loss"],
                        "p104_affine_mode": self.cfg["model"]["critic"].get("affine_mode"), "p104_cosine_bias_init": self.cfg["model"]["critic"].get("cosine_bias_init"),
                        "p104_pair_scope": self.cfg["pairing"].get("pair_scope"), "p104_all_view_chunk": self.cfg["pairing"].get("all_view_chunk"),
                        "p104_noise_repeats": self.cfg["model"]["critic"].get("noise_repeats"),
                        "p104_noise_eval_repeats": self.cfg["model"]["critic"].get("noise_eval_repeats"),
                        "trainable_affine": (None if self.critic is None or not hasattr(self.critic, "scale") else
                                             bool(getattr(self.critic, "trainable_affine", True) and any(p.requires_grad for p in (self.critic.scale, self.critic.bias) if isinstance(p, torch.nn.Parameter)))),
                        "critic_trainable_params": None if self.critic is None else int(sum(p.numel() for p in self.critic.parameters() if p.requires_grad)),
                        # P126 (v6 V6-CURVE): only for the fixed curved scorer — anchor kappa, lambda, endpoints, slope bounds and the ACTUAL zero s0
                        **({"p126_curve": self.critic.curve_record()} if hasattr(self.critic, "curve_record") else {})},
            "cosine_bias_calibration": getattr(self, "calibration", None),
            "bandwidth_calibration": self.bandwidth_calibration, "kernel_sigma": self.kernel_sigma,
            "dataset": self.cfg["data"]["name"], "n_classes": int(self.n_classes),
            "projector_params": self.param_counts["projector"], "objective_target": self.cfg["objective"]["target"],
            "steps_per_epoch": self.steps_per_epoch, "epochs": self.epochs, "intended_total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps, "min_lr_ratio": self.min_lr_ratio,
            "n_fit": int(len(self.fit_uids)), "n_selection": int(len(self.sel_uids)),
            "loader_generator_seed": self.seed, "pair_generator_seed": self.seed + int(self.cfg["pairing"]["rng_seed_offset"]),
            "init_hashes": self.init_hashes, "optimizer_coverage": self.coverage,
            "two_view_transform_sha256": two_view_transform_signature(self.cfg["views"]),
            "started_utc": utc_now(), "hostname": environment_info()["hostname"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        if self.moco:  # P133: replaces the P100 description of the momentum queue
            self.run_manifest["pair_sampling"] = "P133 MoCo-consistent: (online query, momentum key) pairs; Q = keys of other images (batch + uid-masked queue)"
            self.run_manifest.pop("momentum_queue", None)
            self.run_manifest["p133_moco_consistent"] = {
                "m": self.momentum_m, "use_queue": self.p133_use_queue,
                "queue_size": int(self.cfg["pairing"]["queue_size"]) if self.p133_use_queue else None, "key_encoder_bn": "eval mode; BN buffers EMA of the online buffers (same m)",
                "keys": "all V views from the momentum encoder (no gradient)" + ("; view-0 keys + image uids enqueued after the optimiser step" if self.p133_use_queue else ""),
                "positives": "(q_i(x), k_j(x)) for all ordered view pairs i != j",
                "negatives": "(q_i(x), k) for keys of other images: current batch (all views)" + (" + queue (uid-masked)" if self.p133_use_queue else " only (no queue)"),
                "vcs": "original J, scorer critic.score_matrix (A-P3: fixed tanh(2s - 1)), P / Q averaged separately", "simclr": "InfoNCE per (x, i, j != i), same negatives, tau from config"}
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

    @torch.no_grad()
    def _calibration_views(self, n_images: int = 1024) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Two train-distribution views of the first ``n_images`` fit UIDs (dedicated RNG streams), forward in training mode (batch
        statistics = the distribution the objective sees at step 0), BN buffers restored afterwards.  Returns z1, z2 (L2-normalised
        projector outputs) and the record of how they were drawn."""
        uids = self.fit_uids[:n_images]
        seed = self.seed + 515151
        from .data.datasets import TwoViewNoLabelEvalDataset, make_eval_loader  # noqa: PLC0415
        loader = make_eval_loader(TwoViewNoLabelEvalDataset(self.data.data, uids, self.two_view), batch_size=256, num_workers=0,
                                  generator=torch.Generator().manual_seed(seed), pin_memory=False)
        enc_snap = {k: v.clone() for k, v in self.encoder.state_dict().items()}
        proj_snap = {k: v.clone() for k, v in self.projector.state_dict().items()}
        self.encoder.train(); self.projector.train()
        z1s, z2s = [], []
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)  # augmentation parameters (num_workers=0 draws them from the global stream)
            for x1, x2, _ in loader:
                f = forward_features(self.encoder, self.projector, x1.to(self.device), x2.to(self.device), eps=self.cfg["model"]["normalization"]["eps"])
                a, b = f["z_l2"].chunk(2, dim=0)
                z1s.append(a); z2s.append(b)
        self.encoder.load_state_dict(enc_snap); self.projector.load_state_dict(proj_snap)
        rec = {"n_images": int(len(uids)), "uid_first": int(uids[0]), "uid_last": int(uids[-1]), "rng_seed": seed, "batch_size": 256,
               "bn_mode": "train (batch statistics; buffers restored afterwards)", "model_state": "initial weights (before the first optimizer step)"}
        return torch.cat(z1s), torch.cat(z2s), rec

    def calibrate_bandwidths(self) -> dict[str, Any] | None:
        """Server Spec v2 §3.2 / §3.4: kernel bandwidths are fixed multiples of the FIT median pairwise distance, measured once at the start
        of the run on the step-0 critic-input distribution (:meth:`_calibration_views`).  cs_kernel_native: sigma for the Gaussian kernels
        on z_l2 (both views pooled).  rff_tanh: sigma of the random-Fourier features on the pair vector w = [z1; z2] (positive pairs plus one
        cyclic-shift partner per image, i.e. an M-like pool).  Recorded in ``bandwidth_calibration.json``, the run manifest and every
        checkpoint; resume refuses a differing value."""
        is_rff = self.critic is not None and hasattr(self.critic, "set_bandwidth")
        if self.method != "cs_kernel_native" and not is_rff:
            return None
        z1, z2, rec = self._calibration_views()
        if self.method == "cs_kernel_native":
            med = median_pairwise_distance(torch.cat((z1, z2)))
            mult = float(self.cfg["objective"]["kernel_cs_bandwidth_multiple"])
            self.kernel_sigma = mult * med
            rec.update({"kind": "cs_kernel_native", "statistic": "median pairwise distance of z_l2 over both views pooled (i < j, first 4096 rows)",
                        "median_pair_distance": med, "bandwidth_multiple": mult, "sigma": self.kernel_sigma,
                        "chunk": int(self.cfg["objective"]["kernel_cs_chunk"]), "kernel": "exp(-||z_i - z_j||^2 / (2 sigma^2)) on each view"})
        else:
            from reference.ssl_core import cyclic_negative_indices  # noqa: PLC0415
            pg = torch.Generator().manual_seed(self.seed + 5151)
            idx, sh = cyclic_negative_indices(len(z1), 1, generator=pg, device=z1.device)
            w = torch.cat((torch.cat((z1, z2), dim=1), torch.cat((z1, z2[idx[0]]), dim=1)))
            med = median_pairwise_distance(w)
            sigma = self.critic.set_bandwidth(med)
            rec.update({"kind": "rff_tanh", "statistic": "median pairwise distance of pair vectors w = [z1; z2] over the positive pairs and one "
                                                         "cyclic-shift partner per image (i < j, first 4096 rows)",
                        "pair_rng_seed": self.seed + 5151, "shift": int(sh[0]), "median_pair_distance": med,
                        "bandwidth_multiple": float(self.critic.bandwidth_multiple), "sigma": sigma, "n_features": int(self.critic.n_features),
                        "omega0_sha256": sha256_bytes(self.critic.omega0.detach().cpu().numpy().tobytes()),
                        "phase_sha256": sha256_bytes(self.critic.phase.detach().cpu().numpy().tobytes()),
                        "trainable_params": int(sum(p.numel() for p in self.critic.parameters() if p.requires_grad)),
                        "features": "sqrt(2/m) cos(Omega w + b), Omega = Omega0 / sigma fixed, read-out theta (m + 1 trainable), tanh output"})
        atomic_write_json(self.run_dir / "bandwidth_calibration.json", rec)
        return rec

    @torch.no_grad()
    def calibrate_cosine_bias(self, n_images: int = 512) -> dict[str, Any]:
        """Named variant (init only): b0 = -a0 * mu, mu = 0.5*(mean s_pos + mean s_neg) of the critic's similarity on a fixed fit
        calibration mini-batch (first n_images fit UIDs, two views with a dedicated RNG), train-mode BN with buffers restored."""
        crit = self.critic
        if crit is None or not (hasattr(crit, "scale") and hasattr(crit, "bias")):
            raise ConfigError("cosine_bias_calibrate requires a cosine/shared-metric critic")
        uids = self.fit_uids[:n_images]
        gen = torch.Generator().manual_seed(self.seed + 424242)
        from .data.datasets import TwoViewNoLabelEvalDataset, make_eval_loader  # noqa: PLC0415
        loader = make_eval_loader(TwoViewNoLabelEvalDataset(self.data.data, uids, self.two_view), batch_size=256, num_workers=0, generator=gen,
                                  pin_memory=False)
        enc_snap = {k: v.clone() for k, v in self.encoder.state_dict().items()}
        proj_snap = {k: v.clone() for k, v in self.projector.state_dict().items()}
        self.encoder.train(); self.projector.train()
        s_pos, s_neg = [], []
        pg = torch.Generator().manual_seed(self.seed + 4242)
        from reference.ssl_core import cyclic_negative_indices  # noqa: PLC0415
        from .objectives import critic_input_key  # noqa: PLC0415
        key = critic_input_key(self.cfg)
        for x1, x2, _ in loader:
            f = forward_features(self.encoder, self.projector, x1.to(self.device), x2.to(self.device), eps=self.cfg["model"]["normalization"]["eps"])
            z1, z2 = f[key].chunk(2, dim=0)
            if hasattr(crit, "W"):
                z1 = torch.nn.functional.normalize(z1 @ crit.W.T, dim=1); z2 = torch.nn.functional.normalize(z2 @ crit.W.T, dim=1)
            idx, _ = cyclic_negative_indices(len(z1), 1, generator=pg, device=z1.device)
            s_pos.append((z1 * z2).sum(-1)); s_neg.append((z1 * z2[idx[0]]).sum(-1))
        self.encoder.load_state_dict(enc_snap); self.projector.load_state_dict(proj_snap)
        mp, mn = float(torch.cat(s_pos).mean()), float(torch.cat(s_neg).mean())
        mu = 0.5 * (mp + mn)
        a0 = float(crit.scale)
        crit.bias.fill_(-a0 * mu)
        rec = {"n_images": int(len(uids)), "uid_first": int(uids[0]), "uid_last": int(uids[-1]), "rng_seed": self.seed + 424242, "pair_rng_seed": self.seed + 4242,
               "bn_mode": "train (buffers restored afterwards)", "mean_s_pos": mp, "mean_s_neg": mn, "mu": mu, "a0": a0, "b0": float(crit.bias)}
        atomic_write_json(self.run_dir / "cosine_bias_calibration.json", rec)
        return rec

    @torch.no_grad()
    def _refresh_scores(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Cosine similarities of positive pairs and K cyclic-shift negatives on ``refresh_batches`` x B FIT images (two views drawn
        from the training augmentation with dedicated RNG streams; encoder / projector in eval(), so BN buffers are neither used from the
        batch nor updated).  Training RNG streams (global, loader, pairing) are untouched (fork_rng + own generators)."""
        from reference.ssl_core import cyclic_negative_indices  # noqa: PLC0415
        from .data.datasets import TwoViewNoLabelEvalDataset, make_eval_loader  # noqa: PLC0415
        nb = int(self.cfg["model"]["critic"].get("refresh_batches", 16))
        k = int(self.cfg["pairing"]["k"])
        seed = self.seed * 1000 + 616161 + epoch
        perm = torch.randperm(len(self.fit_uids), generator=torch.Generator().manual_seed(seed))
        uids = self.fit_uids[perm[: nb * self.batch].numpy()]
        loader = make_eval_loader(TwoViewNoLabelEvalDataset(self.data.data, uids, self.two_view), batch_size=self.batch, num_workers=0,
                                  generator=torch.Generator().manual_seed(seed + 1), pin_memory=False)
        pg = torch.Generator().manual_seed(seed + 2)
        was = (self.encoder.training, self.projector.training)
        self.encoder.eval(); self.projector.eval()
        s_pos, s_neg = [], []
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 3)  # augmentation parameters (num_workers=0 draws them from the global stream)
            for x1, x2, _ in loader:
                f = forward_features(self.encoder, self.projector, x1.to(self.device), x2.to(self.device), eps=self.cfg["model"]["normalization"]["eps"])
                z1, z2 = f["z_l2"].chunk(2, dim=0)
                idx, _ = cyclic_negative_indices(len(z1), k, generator=pg, device=z1.device)
                s_pos.append((z1 * z2).sum(-1))
                s_neg.append((z1.unsqueeze(0) * z2[idx]).sum(-1).reshape(-1))
        self.encoder.train(was[0]); self.projector.train(was[1])
        rec = {"refresh_seed": seed, "n_images": int(len(uids)), "batches": nb, "K": k, "bn_mode": "eval (running buffers; not updated)"}
        return torch.cat(s_pos).double(), torch.cat(s_neg).double(), rec

    def refresh_cosine_critic(self, epoch: int) -> dict[str, Any]:
        """P95 variant 4 (v1 plan §5 / Spec v2 §8.3 caution, tested online): refit (a, b) of the recipe cosine critic to convergence on
        fixed features (full-batch L-BFGS on −J, strong-Wolfe line search, float64), then continue training.  Only the two critic scalars
        change; AdamW moments of a, b are kept as they are (disclosed).  Logged to logs/critic_refresh.jsonl."""
        crit = self.critic
        if crit is None or not (hasattr(crit, "scale") and hasattr(crit, "bias")):
            raise ConfigError("critic refresh needs the cosine critic")
        t0 = time.perf_counter()
        sp, sn, rec = self._refresh_scores(epoch)
        a0, b0 = float(crit.scale.detach()), float(crit.bias.detach())

        def J_of(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            tp, tn = torch.tanh(a * sp + b), torch.tanh(a * sn + b)
            return (tp - 0.5 * tp.square()).mean() + (-tn - 0.5 * tn.square()).mean()

        with torch.enable_grad():
            a = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
            b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = -J_of(a, b)
                loss.backward()
                return loss
            opt.step(closure)
        with torch.no_grad():
            J_before, J_after = float(J_of(torch.tensor(a0, dtype=torch.float64), torch.tensor(b0, dtype=torch.float64))), float(J_of(a, b))
            ok = np.isfinite(J_after) and J_after >= J_before
            if ok:
                crit.scale.fill_(float(a)); crit.bias.fill_(float(b))
        rec.update({"epoch": epoch, "a_before": a0, "b_before": b0, "a_after": float(crit.scale.detach()), "b_after": float(crit.bias.detach()),
                    "J_before": J_before, "J_after": J_after, "delta_J": J_after - J_before, "applied": bool(ok),
                    "seconds": time.perf_counter() - t0, "utc": utc_now()})
        append_jsonl(self.log_dir / "critic_refresh.jsonl", rec)
        print(f"[{self.run_id}] critic refresh at epoch {epoch}: a {a0:.3f}->{rec['a_after']:.3f}, b {b0:.3f}->{rec['b_after']:.3f}, "
              f"J {J_before:.5f}->{J_after:.5f} ({rec['seconds']:.1f}s)", flush=True)
        return rec

    def save_view_examples(self, n: int = 16) -> None:
        from torchvision.utils import save_image  # noqa: PLC0415

        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == "cuda" else []):
            torch.manual_seed(self.seed + 999)
            rows = []
            uids = []
            for i in range(n):
                item = self.dataset[i]
                v1, v2, uid = item[0], item[1], item[-1]
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
            "physical_batch_images": self.batch, "views_per_image": self.n_views, "K": self.run_manifest["K"], "pair_sampling": self.run_manifest["pair_sampling"],
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
            **({"heldout_J_noisy": ev.get("heldout_J_noisy") if ev.get("epoch") == epoch else None,
                "heldout_gate_clean": ev.get("heldout_gate_clean") if ev.get("epoch") == epoch else None,
                "heldout_gate_noisy": ev.get("heldout_gate_noisy") if ev.get("epoch") == epoch else None,
                "cost": dict(self.cost)} if self._p104() else {}),
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
            "neg_queue_state": None if self.neg_queue is None else self.neg_queue.state_dict(), "queue_fallback_steps": self.queue_fallback_steps,
            "kernel_sigma": self.kernel_sigma, "bandwidth_calibration": self.bandwidth_calibration, "p104_cost": dict(self.cost),
            **({"momentum_key_encoder_state": self.key_model["encoder"].state_dict(), "momentum_key_projector_state": self.key_model["projector"].state_dict(),
                "momentum_m": self.momentum_m} if self.key_model is not None else {}),
            **({"p133_queue_state": self.p133_queue.state_dict()} if self.p133_queue is not None else {}),  # P133
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
        if self.neg_queue is not None:
            if ck.get("neg_queue_state") is None:
                raise ConfigError("resume refused: checkpoint has no negative-queue state but the config uses queue negatives")
            self.neg_queue.load_state_dict(ck["neg_queue_state"])
            self.queue_fallback_steps = int(ck.get("queue_fallback_steps", 0))
        if self.key_model is not None:
            if ck.get("momentum_key_encoder_state") is None:
                raise ConfigError("resume refused: checkpoint has no momentum-encoder state but the config uses the momentum queue")
            self.key_model["encoder"].load_state_dict(ck["momentum_key_encoder_state"])
            self.key_model["projector"].load_state_dict(ck["momentum_key_projector_state"])
        if self.p133_queue is not None:  # P133
            if ck.get("p133_queue_state") is None:
                raise ConfigError("resume refused: checkpoint has no P133 key/uid queue state but the config uses pairing.moco_consistent")
            self.p133_queue.load_state_dict(ck["p133_queue_state"])
        if self.method == "cs_kernel_native":
            cs = ck.get("kernel_sigma")
            if cs is None or self.kernel_sigma is None or abs(float(cs) - self.kernel_sigma) > 1e-6 * max(1.0, abs(self.kernel_sigma)):
                raise ConfigError(f"resume refused: checkpoint kernel_sigma={cs!r} differs from the recalibrated value {self.kernel_sigma!r}")
            self.kernel_sigma = float(cs)  # the checkpoint's value is the one every previous step used
        restore_rng(ck, self.loader_gen, self.pair_gen)
        self.completed_epoch = int(ck["completed_epoch"])
        self.step = int(ck["optimizer_step"])
        self.seen_base_images = int(ck["seen_base_images"])
        self.train_seconds = float(ck["train_seconds"])
        self.eval_seconds = float(ck["eval_seconds"])
        if ck.get("p104_cost") is not None:
            self.cost.update(ck["p104_cost"])
        else:
            self.cost["resumed_without_cost_counters"] = True
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
                           chunk=kcfg["query_chunk"], device=self.device, n_classes=self.n_classes)
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
                                    target_branch=self.target_branch, teacher=ev_teacher, predictor=ev_pred,
                                    **({"noise_tau": float(self.cfg["model"]["critic"]["observation_noise_tau"]),
                                        "noise_repeats": int(self.cfg["model"]["critic"]["noise_eval_repeats"])}
                                       if getattr(crit, "is_noisy", False) and "noise_eval_repeats" in self.cfg["model"]["critic"] else {}))
                result["critic_holdout"] = ch
                result["heldout_J"] = ch["heldout_J_mean"]
                result["heldout_gate_clean"] = ch.get("heldout_gate_clean_mean")
                if "heldout_J_noisy_mean" in ch:
                    result["heldout_J_noisy"] = ch["heldout_J_noisy_mean"]
                    result["heldout_gate_noisy"] = ch["heldout_gate_noisy_mean"]
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
    def train_step(self, x1: torch.Tensor, x2: torch.Tensor, uids: torch.Tensor, log_this: bool, extra_views: list | None = None) -> dict[str, Any]:
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
        multi = bool(extra_views)
        if multi:
            if self.target_branch != "shared" or critic_steps(self.cfg) != 1:
                raise ConfigError("views.count > 2 is implemented for the shared branch with a single joint step only")
            views = [x1, x2] + [v.to(self.device, non_blocking=True) for v in extra_views]
            feats = forward_features_views(self.encoder, self.projector, views, eps=self.cfg["model"]["normalization"]["eps"])
        elif self.target_branch == "shared":
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
                cobj = (compute_objective(self.method, det, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen, queue=self.neg_queue) if self.target_branch == "shared"
                        else compute_objective_target(det, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen))
                if not torch.isfinite(cobj["loss"]):
                    raise FloatingPointError(f"non-finite critic-only loss at step {self.step}")
                cobj["loss"].backward()
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        keys = None
        p133_keys = None
        if multi and self.key_model is not None and not self.moco:
            keys = self.momentum_keys(views[0])
        if multi and self.moco:  # P133: (online query, momentum key) objective
            p133_keys = self.p133_momentum_keys(views)
            obj = compute_objective_moco_consistent(feats, p133_keys, uids, cfg=self.cfg, critic=self.critic, queue=self.p133_queue)
        elif multi:
            obj = compute_objective_views(feats, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen, kernel_sigma=self.kernel_sigma,
                                          **({"queue": self.neg_queue} if self.key_model is not None else {}))
        else:
            obj = (compute_objective(self.method, feats, cfg=self.cfg, critic=self.critic, pair_generator=self.pair_gen, queue=self.neg_queue,
                                     kernel_sigma=self.kernel_sigma) if self.target_branch == "shared"
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
        if self.neg_queue is not None:  # queue update after the step: both views' detached features of this batch
            if obj["stats"].get("queue_fallback") == 1.0:
                self.queue_fallback_steps += 1
            self.neg_queue.enqueue(keys if keys is not None else feats[self.queue_key].detach())
        if self.p133_queue is not None:  # P133: view-0 momentum keys of this batch (computed before the step) + their image uids
            self.p133_queue.enqueue(p133_keys[0], uids)
        if self.key_model is not None:
            self.momentum_update()
        if self.teacher is not None:
            ema_update(self.teacher, self.encoder, self.projector)
        cos_extra = ({"cos_scale": float(self.critic.scale.detach()), "cos_bias": float(self.critic.bias.detach())}
                     if self.critic is not None and hasattr(self.critic, "scale") and hasattr(self.critic, "bias") else {})
        out = {"loss": float(loss.detach()), **obj["stats"], **cos_extra, "shift": obj["shift"], "n_pos": obj["n_pos"], "n_neg": obj["n_neg"],
               **({"critic_pair_evals": obj["critic_pair_evals"]} if "critic_pair_evals" in obj else {}),
               "lr_factor": factor, **{f"lr_{k}": v for k, v in lrs.items()}, **gn}
        return out

    @torch.no_grad()
    def momentum_keys(self, x: torch.Tensor) -> torch.Tensor:
        """P100: queue keys of this batch from the momentum encoder (train-mode BN with its own buffers), in the critic-input space."""
        enc, proj = self.key_model["encoder"], self.key_model["projector"]
        enc.train(); proj.train()
        eps = self.cfg["model"]["normalization"]["eps"]
        h = enc(x); p = proj(h)
        key = self.queue_key
        return {"z_l2": torch.nn.functional.normalize(p, dim=1, eps=eps), "p_raw": p, "h_l2": torch.nn.functional.normalize(h, dim=1, eps=eps)}[key].detach()

    @torch.no_grad()
    def momentum_update(self) -> None:
        m = self.momentum_m
        for k_mod, q_mod in ((self.key_model["encoder"], self.encoder), (self.key_model["projector"], self.projector)):
            for kp, qp in zip(k_mod.parameters(), q_mod.parameters()):
                kp.mul_(m).add_(qp.detach(), alpha=1.0 - m)
            if self.moco:  # P133: the eval-mode key encoder's BN buffers follow the online buffers by the same EMA
                for (kn, kb), (qn, qb) in zip(k_mod.named_buffers(), q_mod.named_buffers()):
                    if kn != qn:
                        raise RuntimeError("momentum / online buffer layouts differ")
                    if kb.dtype.is_floating_point:
                        kb.mul_(m).add_(qb.detach(), alpha=1.0 - m)
                    else:
                        kb.copy_(qb)

    @torch.no_grad()
    def p133_momentum_keys(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """P133: momentum keys of all V views in EVAL mode (per-sample; no batch-statistics channel), in the critic-input space, L2-normalised."""
        enc, proj = self.key_model["encoder"], self.key_model["projector"]
        enc.eval(); proj.eval()
        eps = self.cfg["model"]["normalization"]["eps"]
        h = enc(torch.cat(views, dim=0)); p = proj(h)
        src = critic_input_key(self.cfg) if self.method == "vcs_qmi" else "z_l2"
        if src != "z_l2":
            raise RuntimeError("P133 keys are defined in the z (L2-normalised projector) space")
        return list(torch.nn.functional.normalize(p, dim=1, eps=eps).detach().chunk(len(views), dim=0))

    def first_step_gradient_check(self, gn: dict[str, Any]) -> None:
        """Spec §7.2: every module has a nonzero finite gradient on the first real step; parameters then change."""
        problems = []
        critic_on_h = self.method == "vcs_qmi" and self.cfg["model"]["critic"].get("feature_source", "z") == "h_l2"
        for name, m in (("encoder", self.encoder), ("projector", self.projector), ("critic", self.critic), ("predictor", self.predictor)):
            if m is None or (name == "projector" and critic_on_h):  # projector receives no gradient when the critic reads h (disclosed)
                continue
            if not has_trainable_params(m):  # parameter-free module (named variant projector.kind=bn_only: affine-free BN only)
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
                   "uids": None if batch is None else batch[-1].tolist(), **capture_rng(self.device, self.loader_gen, self.pair_gen),
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
                            out = self.train_step(batch[0], batch[1], batch[-1], log_this, extra_views=list(batch[2:-1]) if len(batch) > 3 else None)
                        if self.step == 0:
                            self.verify_first_step_update()
                        step_times.append(st.elapsed)
                        if self.step >= self.warmup_steps:
                            self.steady_step_times.append(st.elapsed)
                        self.step += 1
                        self.seen_base_images += int(batch[0].shape[0])
                        self.cost["positive_pairs"] += int(out.get("n_pos") or 0); self.cost["negative_pairs"] += int(out.get("n_neg") or 0)
                        self.cost["critic_pair_evaluations"] += int(out.get("critic_pair_evals") or ((out.get("n_pos") or 0) + (out.get("n_neg") or 0)))
                        self.cost["encoder_updates"] += 1
                        if self.critic is not None and any(p.requires_grad for p in self.critic.parameters()):
                            self.cost["critic_updates"] += critic_steps(self.cfg)
                        for k, v in out.items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("shift", "n_pos", "n_neg", "critic_pair_evals"):
                                sums[k] = sums.get(k, 0.0) + float(v)
                                counts[k] = counts.get(k, 0) + 1
                        if log_this:
                            rec = {"step": self.step - 1, "epoch": epoch, **out, "base_images": int(batch[0].shape[0]), "views": (len(batch) - 1) * int(batch[0].shape[0]),
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
                R_ref = int(self.cfg["model"]["critic"].get("refresh_every_epochs", 0))
                if R_ref > 0 and epoch % R_ref == 0 and epoch < self.epochs:  # P95 critic refresh, before the epoch's checkpoint
                    self.cost["refresh_seconds"] += float(self.refresh_cosine_critic(epoch)["seconds"])  # package v3 §12: counted separately
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
            "optimizer_step": self.step, "seen_base_images": self.seen_base_images, "seen_views": self.n_views * self.seen_base_images,
            "train_seconds": self.train_seconds, "eval_seconds": self.eval_seconds,
            "steady_state_step_seconds_mean": float(np.mean(steady)) if steady else None,
            "steady_state_images_per_s": (self.batch / float(np.mean(steady))) if steady else None,
            "steady_state_views_per_s": (2 * self.batch / float(np.mean(steady))) if steady else None,
            # package v3 §12: the field above assumed 2 views; the real number of encoded views per step is n_views * batch (new field;
            # historical summaries are not rewritten)
            "steady_state_views_per_s_actual": (self.n_views * self.batch / float(np.mean(steady))) if steady else None,
            "cost": {**self.cost, "base_images": self.seen_base_images, "encoded_views": self.n_views * self.seen_base_images,
                     "training_seconds_total": self.train_seconds + float(self.cost.get("refresh_seconds", 0.0))},
            **mem, "collapse_suspected": self.collapse_streak >= 2, "finished_utc": utc_now(), "queue_fallback_steps": self.queue_fallback_steps,
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
    data = load_train_partition(cfg["data"]["name"], cfg["data"]["root"])
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
