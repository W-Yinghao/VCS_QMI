"""Strict loading of the frozen pilot configurations.

Rules (spec §2, §15.2 item 8): unknown fields are errors, missing fields are errors, ``${VAR}`` placeholders must
resolve from the environment to existing directories, and first-round policy invariants are asserted so that a
config can never silently enable something the spec forbids.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .utils import sha256_file, sha256_json

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

METHODS = ("vcs_qmi", "simclr_matched", "vicreg_matched_128", "cs_kernel_native")
DATASETS = {"cifar10": {"val_per_class": 500, "manifest_tag": "cifar10_"}, "cifar100": {"val_per_class": 50, "manifest_tag": "cifar100_"}}


class ConfigError(ValueError):
    pass


# --- schema: nested dict of key -> type spec.  A tuple of types means "one of"; None means allow None. -----------
_Num = (int, float)


class _Opt:
    """Optional field: absent in older frozen configs -> filled with ``default`` (so downstream code always sees it)."""

    def __init__(self, types: Any, default: Any, fill: bool = True) -> None:
        self.types = types if isinstance(types, tuple) else (types,)
        self.default = default
        self.fill = fill  # False (P95 fields): validated when present, never inserted, so the resolved dict / config_hash of every
        #                   existing config is unchanged (resume and evaluation compare that hash); code reads them with .get(default)


SCHEMA: dict[str, Any] = {
    "schema_version": str,
    "run": {"stage": str, "method": str, "seed": int, "output_root": str, "resume": (str, type(None)), "overwrite": bool,
            "control_tuning": _Opt(bool, False)},
    "data": {"name": str, "root": str, "download": bool, "split": str, "split_seed": int, "val_per_class": int,
             "manifest": str, "ssl_labels_accessible": bool, "official_test_accessible": bool},
    "views": {"count": int,
              "random_resized_crop": {"size": int, "scale": list, "ratio": list, "interpolation": str, "antialias": bool},
              "horizontal_flip_p": _Num,
              "color_jitter": {"brightness": _Num, "contrast": _Num, "saturation": _Num, "hue": _Num, "p": _Num},
              "grayscale_p": _Num, "gaussian_blur_p": _Num, "solarize_p": _Num,
              "normalize_mean": list, "normalize_std": list},
    "model": {"backbone": str, "weights": (str, type(None)), "h_dim": int,
              "stem": {"kernel_size": int, "stride": int, "padding": int, "bias": bool, "maxpool": bool},
              "projector": {"hidden_dim": int, "output_dim": int, "hidden_batchnorm": bool, "hidden_linear_bias": bool,
                            "output_linear_bias": bool, "output_batchnorm": bool, "depth": _Opt(int, 2), "predictor": _Opt(bool, False), "kind": _Opt(str, "mlp")},
              "normalization": {"vcs_and_simclr": str, "eps": _Num, "vicreg": str},
              "critic": {"enabled": bool, "input": str, "hidden_dims": list, "activation": str, "output": str,
                         "batchnorm": bool, "dropout": _Num, "last_layer_xavier_gain": _Num, "last_layer_bias": _Num,
                         "feature_source": _Opt(str, "z"), "cosine_scale_init": _Opt((int, float), 1.0),
                         "cosine_scale_fixed": _Opt(bool, False), "cosine_bias_calibrate": _Opt(bool, False),
                         "rff_features": _Opt(int, 1024), "rff_bandwidth_multiple": _Opt((int, float), 1.0),
                         # P95 (package v1 improvements inside SSL), not filled when absent:
                         "residual_lambda": _Opt((int, float), 0.5, fill=False), "observation_noise_tau": _Opt((int, float), 0.0, fill=False),
                         "refresh_every_epochs": _Opt(int, 0, fill=False), "refresh_batches": _Opt(int, 16, fill=False),
                         # P104 (package v3), not filled when absent:
                         "affine_mode": _Opt(str, "learned", fill=False), "cosine_bias_init": _Opt((int, float), 0.0, fill=False),
                         "noise_repeats": _Opt(int, 1, fill=False), "noise_eval_repeats": _Opt(int, 16, fill=False)}},
    "objective": {"target": str, "loss": str, "positive_weight": _Num, "negative_weight": _Num,
                  "training_cs_transform": bool, "clip_J": bool, "extra_regularizers": list, "simclr_temperature": _Num,
                  "vicreg_weights": {"invariance": _Num, "variance": _Num, "covariance": _Num}, "vicreg_variance_eps": _Num,
                  "kernel_cs_bandwidth_multiple": _Opt((int, float), 1.0), "kernel_cs_chunk": _Opt(int, 0),
                  # P107 (package v4 A-L1) explicit marker, not filled when absent:
                  "js_fixed_scorer": _Opt(bool, False, fill=False)},
    "pairing": {"sampler": str, "k": int, "unique_shifts": bool, "allow_self": bool, "label_filter": bool, "queue": bool,
                "negative_detach": bool, "rng": str, "rng_seed_offset": int,
                "negative_source": _Opt(str, "cyclic"), "queue_size": _Opt(int, 4096),
                # P100 (momentum-encoder key queue), not filled when absent so older configs keep their resolved dict / config_hash:
                "momentum_encoder": _Opt(bool, False, fill=False), "momentum_m": _Opt((int, float), 0.99, fill=False),
                # P104 U line (package v3 §6), not filled when absent:
                "pair_scope": _Opt(str, "cross_view_k", fill=False), "all_view_chunk": _Opt(int, 256, fill=False)},
    "train": {"mode": str, "target_branch": _Opt(str, "shared"), "epochs": int, "warmup_epochs": _Num, "batch_size_images": int, "drop_last": bool, "shuffle": bool,
              "replacement": bool, "world_size": int, "grad_accumulation_steps": int, "precision": str, "allow_tf32": bool,
              "compile": bool, "num_workers": int, "pin_memory": bool, "persistent_workers": bool,
              "grad_clip_norm": (_Num[0], _Num[1], type(None)), "encoder_projector_forward": str, "max_steps": (int, type(None))},
    "optimizer": {"name": str, "lr": _Num, "betas": list, "eps": _Num, "matrix_weight_decay_encoder_projector": _Num,
                  "weight_decay_bias_norm": _Num, "critic_weight_decay": _Num, "critic_lr_multiplier": _Num},
    "schedule": {"kind": str, "unit": str, "min_lr_ratio": _Num, "scale_lr_with_batch": bool},
    "evaluation": {"official_test_enabled": bool, "checkpoint_rule": str, "knn_epochs": list, "linear_epochs_of_pretrain": list,
                   "clean_transform": str, "feature": str, "freeze_encoder_parameters": bool, "freeze_bn_buffers": bool,
                   "knn": {"k": int, "temperature": _Num, "normalize_h": bool, "query_chunk": int},
                   "linear": {"normalize_h": bool, "head": str, "epochs": int, "batch_size": int, "optimizer": str, "lr": _Num,
                              "momentum": _Num, "weight_decay": _Num, "schedule": str, "min_lr_ratio": _Num, "seed": int,
                              "checkpoint_rule": str, "feature_cache_dtype": str},
                   "critic_validation": {"enabled": bool, "augmentation": str, "repeats": int, "update_weights": bool,
                                         "model_mode": str, "batch_size": int, "rng_seed": int},
                   "spectrum": {"enabled": bool, "selection_first_sorted_ids": int, "features": list, "center": bool,
                                "covariance_ddof": int}},
    "logging": {"step_interval": int, "gradient_norm_interval": int, "epoch_jsonl": bool, "save_initial_checkpoint": bool,
                "checkpoint_epochs": list, "resume_checkpoint": str, "keep_failure_artifacts": bool, "external_logging": bool},
    "execution": {"max_concurrent_jobs": int, "max_gpus_per_job": int, "automatic_hyperparameter_search": bool,
                  "auto_start_full_200ep": bool, "requires_preflight": bool},
}


def _validate(node: Any, schema: Any, path: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise ConfigError(f"{path}: expected mapping, got {type(node).__name__}")
        unknown = sorted(set(node) - set(schema))
        missing = sorted(k for k in set(schema) - set(node) if not isinstance(schema[k], _Opt))
        if unknown:
            raise ConfigError(f"{path}: unknown field(s) {unknown}; fields are never silently ignored")
        if missing:
            raise ConfigError(f"{path}: missing field(s) {missing}")
        for k, sub in schema.items():
            if isinstance(sub, _Opt) and k not in node:
                if not sub.fill:
                    continue  # P95 optional field absent: left out (hash of older configs unchanged)
                node[k] = sub.default  # optional field absent (older frozen config): filled with its default
            _validate(node[k], sub.types if isinstance(sub, _Opt) else sub, f"{path}.{k}")
        return
    types = schema if isinstance(schema, tuple) else (schema,)
    if bool in types and isinstance(node, bool):
        return
    if isinstance(node, bool) and bool not in types:
        raise ConfigError(f"{path}: boolean not allowed here")
    if not isinstance(node, types):
        names = "/".join(t.__name__ for t in types)
        raise ConfigError(f"{path}: expected {names}, got {type(node).__name__} ({node!r})")


def _resolve_env(node: Any, path: str, env: dict[str, str]) -> Any:
    if isinstance(node, dict):
        return {k: _resolve_env(v, f"{path}.{k}", env) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_env(v, f"{path}[{i}]", env) for i, v in enumerate(node)]
    if isinstance(node, str):
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name not in env or not env[name]:
                raise ConfigError(f"{path}: placeholder ${{{name}}} is unresolved; export {name} before running")
            return env[name]
        return _ENV_RE.sub(sub, node)
    return node


def _p95_policy(cfg: dict[str, Any]) -> None:
    """P95 named variants (package v1 improvements inside full SSL, owner 2026-09-29).  Each is allowed only on the recipe wiring it
    was designed for; none changes J (except the js_matched control, which is labelled as a control loss)."""
    c = cfg["model"]["critic"]
    loss = cfg["objective"]["loss"]
    tau = float(c.get("observation_noise_tau", 0.0))
    R = int(c.get("refresh_every_epochs", 0))
    lam = c.get("residual_lambda")
    if c["input"] == "residual_cosine_mlp":
        if lam is None or not (0.0 < float(lam) < 1.0):
            raise ConfigError("residual_cosine_mlp needs critic.residual_lambda in (0, 1)")
    elif lam is not None:
        raise ConfigError("critic.residual_lambda is only defined for residual_cosine_mlp")
    if tau < 0:
        raise ConfigError("critic.observation_noise_tau must be >= 0")
    if tau > 0 and c["input"] != "cosine":
        raise ConfigError("observation noise is defined for the recipe cosine critic only")
    if R < 0 or int(c.get("refresh_batches", 16)) < 1:
        raise ConfigError("critic.refresh_every_epochs must be >= 0 and refresh_batches >= 1")
    if R > 0 and (c["input"] != "cosine" or c.get("cosine_scale_fixed", False)):
        raise ConfigError("critic refresh refits (a, b) of the recipe cosine critic only")
    if loss == "js_matched_logistic":
        n2 = "noise_repeats" in c  # P104 N2 marker: the noisy matched-JS control is declared explicitly (P95 rule kept otherwise)
        # P107 A-L1 (package v4 §A1): the matched JS loss on the FIXED angular scorer f = 2s - 1 (affine_mode 'fixed', no noise) is the
        # named JS counterpart of P104 G2; every other refusal is kept.
        fixed_js = bool(cfg["objective"].get("js_fixed_scorer", False)) and c.get("affine_mode", "learned") == "fixed" and tau == 0
        if cfg["objective"].get("js_fixed_scorer", False) and not fixed_js:
            raise ConfigError("objective.js_fixed_scorer marks the P107 A-L1 cell: affine_mode 'fixed', no observation noise")
        if c["input"] != "cosine" or (tau > 0 and not n2) or R > 0 or c.get("cosine_scale_fixed", False) or (c.get("affine_mode", "learned") != "learned" and not fixed_js):
            raise ConfigError("js_matched_logistic is the control for the recipe cosine critic (learned a, b, or the P107 fixed scorer; no refresh); "
                              "observation noise is allowed only as the P104 N2 control (critic.noise_repeats set)")
        if cfg["pairing"]["sampler"] != "random_nonzero_cyclic_shift" or cfg["pairing"].get("negative_source", "cyclic") != "cyclic":
            raise ConfigError("js_matched_logistic uses the recipe's cyclic-shift pairing")
    if cfg["objective"].get("js_fixed_scorer", False) and loss != "js_matched_logistic":
        raise ConfigError("objective.js_fixed_scorer is only meaningful with the matched JS loss (P107 A-L1)")
    p95 = c["input"] in ("residual_cosine_mlp", "dictionary_simplex") or tau > 0 or R > 0 or loss == "js_matched_logistic"
    if p95:
        if cfg["train"].get("target_branch", "shared") != "shared" or cfg["train"]["mode"] != "joint" or c.get("feature_source", "z") != "z":
            raise ConfigError("P95 variants use the recipe wiring (shared branch, joint mode, critic on z)")
        if cfg["model"]["normalization"]["vcs_and_simclr"] != "l2":
            raise ConfigError("P95 variants read the L2-normalised projector output")


def _p104_policy(cfg: dict[str, Any]) -> None:
    """P104 (package v3, owner 2026-09-30): G (fixed angular scale, full negative routing), U (all view tokens), N (noise-draw average,
    noisy matched-JS control).  Each option is allowed only on the wiring it was specified for; J / tanh / product negatives unchanged."""
    c, p = cfg["model"]["critic"], cfg["pairing"]
    m, loss = cfg["run"]["method"], cfg["objective"]["loss"]
    tau = float(c.get("observation_noise_tau", 0.0))
    am = c.get("affine_mode", "learned")
    scope = p.get("pair_scope", "cross_view_k")
    R = c.get("noise_repeats", 1)
    Re = c.get("noise_eval_repeats")
    new = any(x in c for x in ("affine_mode", "cosine_bias_init", "noise_repeats", "noise_eval_repeats")) or any(x in p for x in ("pair_scope", "all_view_chunk"))
    if not new:
        return
    if m != "vcs_qmi":
        raise ConfigError("P104 options are VCS-only")
    if c["input"] != "cosine":
        raise ConfigError("P104 options are defined for the angular (cosine) critic")
    if am not in ("learned", "fixed"):
        raise ConfigError("critic.affine_mode must be 'learned' or 'fixed'")
    if am == "fixed":
        if tau > 0 or c.get("cosine_scale_fixed", False) or c.get("cosine_bias_calibrate", False) or int(c.get("refresh_every_epochs", 0)) > 0:
            raise ConfigError("affine_mode 'fixed' (G line): a, b are buffers; no noise / scale_fixed / bias calibration / refresh on top")
        if not (loss == "negative_J" or (loss == "js_matched_logistic" and cfg["objective"].get("js_fixed_scorer", False))):
            raise ConfigError("the fixed-scale critic is used with the original J, or with the matched JS loss only when objective.js_fixed_scorer "
                              "is set (P107 A-L1)")
    if cfg["objective"].get("js_fixed_scorer", False) and (loss != "js_matched_logistic" or am != "fixed"):
        raise ConfigError("objective.js_fixed_scorer marks the P107 A-L1 cell: matched JS loss on the fixed angular scorer only")
    if not isinstance(c.get("cosine_bias_init", 0.0), (int, float)):
        raise ConfigError("critic.cosine_bias_init must be a number")
    if c.get("cosine_bias_calibrate", False) and "cosine_bias_init" in c:
        raise ConfigError("cosine_bias_init and cosine_bias_calibrate are exclusive")
    if scope not in ("cross_view_k", "all_view_tokens"):
        raise ConfigError("pairing.pair_scope must be 'cross_view_k' or 'all_view_tokens'")
    if scope == "all_view_tokens":
        if cfg["views"]["count"] <= 2:
            raise ConfigError("all_view_tokens is implemented for the multi-view path (views.count > 2)")
        if tau > 0:
            raise ConfigError("package v3 §6.4: a noisy critic with the all-view matrix path is refused (the matrix hook would skip the noise)")
        if loss != "negative_J" or p["sampler"] != "random_nonzero_cyclic_shift" or p.get("negative_source", "cyclic") != "cyclic":
            raise ConfigError("all_view_tokens uses the original J and replaces the cyclic-shift sampler (keep the frozen sampler field)")
        if not (isinstance(p.get("all_view_chunk", 256), int) and p.get("all_view_chunk", 256) >= 1):
            raise ConfigError("pairing.all_view_chunk must be a positive int")
    elif "all_view_chunk" in p:
        raise ConfigError("pairing.all_view_chunk is only meaningful with pair_scope 'all_view_tokens'")
    if not (isinstance(R, int) and R >= 1):
        raise ConfigError("critic.noise_repeats must be an int >= 1")
    if R > 1 and (tau <= 0 or cfg["views"]["count"] <= 2 or scope != "cross_view_k"):
        raise ConfigError("noise_repeats > 1 needs observation noise, the multi-view path and the cross-view K pairing")
    if Re is not None and (not isinstance(Re, int) or Re < 1 or tau <= 0):
        raise ConfigError("critic.noise_eval_repeats must be an int >= 1 and requires observation noise")
    if loss == "js_matched_logistic" and tau > 0 and cfg["views"]["count"] <= 2:
        raise ConfigError("the noisy matched-JS control (N2) is implemented for the multi-view path")


def policy_checks(cfg: dict[str, Any]) -> None:
    """First-round invariants from spec §1.2, §6, §8, §12, §15.2(8)."""
    if cfg["schema_version"] != "vcs_ssl_agent_1.0":
        raise ConfigError(f"unsupported schema_version {cfg['schema_version']!r}")
    m = cfg["run"]["method"]
    if m not in METHODS:
        raise ConfigError(f"run.method must be one of {METHODS}, got {m!r}")
    dname = cfg["data"]["name"]
    if dname not in DATASETS:
        raise ConfigError(f"data.name must be one of {tuple(DATASETS)}, got {dname!r}")
    if cfg["data"]["split"] != "dev45k_val5k" or not (isinstance(cfg["data"]["val_per_class"], int) and cfg["data"]["val_per_class"] > 0):
        raise ConfigError(f"{dname}: the identity split is dev45k_val5k with a positive integer val_per_class (500 for cifar10, 50 for cifar100 on the real "
                          "manifests; the value is cross-checked against the manifest at load time)")
    if DATASETS[dname]["manifest_tag"] not in Path(cfg["data"]["manifest"]).name or (dname == "cifar10" and "cifar100" in Path(cfg["data"]["manifest"]).name):
        raise ConfigError(f"data.manifest file name must carry the dataset tag {DATASETS[dname]['manifest_tag']!r} (no cross-dataset manifest reuse)")
    if cfg["model"]["weights"] is not None:
        raise ConfigError("model.weights must be null (no pretrained weights)")
    if cfg["objective"]["extra_regularizers"] != []:
        raise ConfigError("objective.extra_regularizers must be [] in the first round")
    if cfg["objective"]["training_cs_transform"] or cfg["objective"]["clip_J"]:
        raise ConfigError("training on transformed CS or clipped J is forbidden")
    if cfg["objective"]["positive_weight"] != 1.0 or cfg["objective"]["negative_weight"] != 1.0:
        raise ConfigError("positive/negative weights must both be 1.0 (reference objective)")
    p = cfg["pairing"]
    if not (1 <= p["k"] <= cfg["train"]["batch_size_images"] - 1):
        raise ConfigError("pairing.k must satisfy 1 <= K <= batch_size_images - 1 (K distinct nonzero cyclic shifts)")
    ns = p.get("negative_source", "cyclic")
    if ns not in ("cyclic", "queue"):
        raise ConfigError("pairing.negative_source must be 'cyclic' (frozen: in-batch cyclic shifts) or 'queue' (named variant: FIFO queue of detached features from previous steps)")
    if p["queue"] != (ns == "queue"):
        raise ConfigError("pairing.queue must be true exactly when pairing.negative_source == 'queue'")
    if p["sampler"] not in ("random_nonzero_cyclic_shift", "random_nonzero_cyclic_shift_symmetric", "all_pairs_matrix") or not p["unique_shifts"] \
            or p["allow_self"] or p["label_filter"] or p["rng"] != "dedicated_cpu_generator":
        raise ConfigError("pairing policy deviates from the frozen definition (symmetric scoring of both orders is the only named variant)")
    if ns == "queue":
        if m not in ("vcs_qmi", "simclr_matched"):
            raise ConfigError("queue negatives are implemented for vcs_qmi and simclr_matched only (VICReg has no negatives)")
        if p["sampler"] != "random_nonzero_cyclic_shift":
            raise ConfigError("queue negatives require the default sampler (no symmetric / all-pairs combination)")
        if not isinstance(p["queue_size"], int) or p["queue_size"] < max(2, p["k"]):
            raise ConfigError("pairing.queue_size must be an int >= max(2, K)")
        mom = bool(p.get("momentum_encoder", False))
        if (cfg["views"]["count"] != 2 and not mom) or cfg["train"].get("target_branch", "shared") != "shared" or cfg["train"]["mode"] != "joint":
            raise ConfigError("queue negatives are implemented for 2 views (4 views with the P100 momentum encoder), the shared branch and a single joint step only")
    if p.get("momentum_encoder", False):
        # P100: momentum (EMA) copy of encoder + projector produces the queue keys; the objective is unchanged
        if ns != "queue":
            raise ConfigError("pairing.momentum_encoder requires pairing.negative_source == 'queue'")
        mm = p.get("momentum_m", 0.99)
        if not (isinstance(mm, (int, float)) and 0.0 < float(mm) < 1.0):
            raise ConfigError("pairing.momentum_m must be in (0, 1)")
        if cfg["views"]["count"] != 4:
            raise ConfigError("the momentum-encoder queue is implemented for the 4-view recipe (views.count = 4) only")
    elif "momentum_m" in p:
        raise ConfigError("pairing.momentum_m is only meaningful with pairing.momentum_encoder: true")
    if p["negative_detach"] and m != "vcs_qmi":
        raise ConfigError("negative_detach is a VCS-only named variant")
    if m == "vcs_qmi" and cfg["model"]["critic"]["cosine_scale_init"] <= 0:
        raise ConfigError("critic.cosine_scale_init must be > 0")
    if m == "vcs_qmi" and cfg["model"]["critic"]["feature_source"] not in ("z", "h_l2"):
        raise ConfigError("critic.feature_source must be 'z' (projector output, default) or 'h_l2' (L2-normalized encoder output, named variant)")
    t = cfg["train"]
    if not (t["mode"] == "joint" or re.fullmatch(r"joint_critic_steps_([2-9]|10)", t["mode"])):
        raise ConfigError("train.mode must be 'joint' or the named variant 'joint_critic_steps_N' (N extra-1 critic-only steps on detached features, 2<=N<=10)")
    if m != "vcs_qmi" and (t["mode"] != "joint" or p["sampler"] != "random_nonzero_cyclic_shift"):
        raise ConfigError("control runs keep joint mode and the default sampler")
    tb = t["target_branch"]
    if not (tb in ("shared", "stopgrad") or re.fullmatch(r"ema_0\.\d+", tb)):
        raise ConfigError("train.target_branch must be 'shared' (default), 'stopgrad' or 'ema_<tau>' (named variants)")
    if m != "vcs_qmi" and (tb != "shared" or cfg["model"]["projector"]["predictor"] or cfg["model"]["projector"]["depth"] != 2):
        raise ConfigError("control runs keep the shared two-view branch, no predictor, 2-layer projector")
    if cfg["model"]["projector"]["depth"] < 1 or cfg["model"]["projector"]["depth"] > 4:
        raise ConfigError("projector depth must be in [1, 4] (1 = single linear layer, VCS-only named variant)")
    if cfg["model"]["projector"]["kind"] not in ("mlp", "bn_only"):
        raise ConfigError("projector.kind must be 'mlp' (default) or 'bn_only' (affine-free BN on h, VCS-only named variant)")
    if m != "vcs_qmi" and (cfg["model"]["projector"]["kind"] != "mlp" or cfg["model"]["projector"]["depth"] != 2):
        raise ConfigError("control runs keep the 2-layer MLP projector")
    if cfg["model"]["projector"]["kind"] == "bn_only" and cfg["model"]["projector"]["output_dim"] != cfg["model"]["h_dim"]:
        raise ConfigError("projector.kind=bn_only requires output_dim == h_dim")
    if cfg["model"]["projector"]["predictor"] and tb == "shared":
        raise ConfigError("a predictor requires a stop-gradient or EMA target branch")
    if t["world_size"] != 1 or t["grad_accumulation_steps"] != 1 or t["precision"] != "fp32" or t["allow_tf32"] or t["compile"]:
        raise ConfigError("single-GPU FP32, no accumulation, no TF32, no compile in the first round")
    if t["grad_clip_norm"] is not None:
        raise ConfigError("gradient clipping is disabled in the first round")
    if not t["drop_last"] or not t["shuffle"] or t["replacement"]:
        raise ConfigError("loader must shuffle without replacement and drop the last incomplete batch")
    if t["encoder_projector_forward"] != "concat_2B":
        raise ConfigError("forward policy must be concat_2B")
    if t["batch_size_images"] < 2:
        raise ConfigError("batch_size_images must be >= 2")
    if cfg["data"]["official_test_accessible"] or cfg["evaluation"]["official_test_enabled"]:
        raise ConfigError("official test set must stay inaccessible in the first round")
    if cfg["data"]["ssl_labels_accessible"]:
        raise ConfigError("SSL training must not see labels")
    if cfg["data"]["download"]:
        raise ConfigError("data.download must be false; data is prepared explicitly in P0")
    if cfg["execution"]["automatic_hyperparameter_search"] or cfg["execution"]["auto_start_full_200ep"]:
        raise ConfigError("automatic search / auto 200-epoch start are not authorized")
    if cfg["execution"]["max_gpus_per_job"] != 1:
        raise ConfigError("max_gpus_per_job must be 1")
    if cfg["views"]["count"] not in (2, 4, 8, 16):
        raise ConfigError("views.count must be 2 (frozen) or 4 / 8 / 16 (named variants: the objective averaged over all view pairs)")
    ct = cfg["run"].get("control_tuning", False)
    if ct and m == "vcs_qmi":
        raise ConfigError("run.control_tuning applies to control methods only")
    if cfg["views"]["count"] != 2 and m not in ("vcs_qmi", "cs_kernel_native") and not (ct and cfg["views"]["count"] == 4):
        raise ConfigError("control runs keep two views unless run.control_tuning=true (then 4 views: the control loss averaged over the 6 view pairs); "
                          "cs_kernel_native (S-CS table B) uses the same multi-view averaging as VCS natively")
    if cfg["pairing"]["sampler"] == "all_pairs_matrix" and (m != "vcs_qmi" or cfg["model"]["critic"]["input"] not in ("cosine", "shared_metric", "mono_spline", "diag_metric")):
        raise ConfigError("all_pairs_matrix is a VCS-only sampler for similarity-type critics (cosine|shared_metric|mono_spline|diag_metric)")
    if cfg["pairing"]["sampler"] == "all_pairs_matrix" and float(cfg["model"]["critic"].get("observation_noise_tau", 0.0)) > 0:
        raise ConfigError("package v3 §6.4: a noisy critic with the all_pairs_matrix hook is refused (the matrix hook would skip the noise)")
    if cfg["views"]["solarize_p"] != 0.0:
        raise ConfigError("solarize is not part of any recipe here")
    if not 0.0 <= cfg["views"]["gaussian_blur_p"] <= 1.0:
        raise ConfigError("gaussian_blur_p must be in [0, 1]")
    if m != "vcs_qmi" and cfg["views"]["gaussian_blur_p"] != 0.0:
        raise ConfigError("control runs keep the frozen augmentation recipe (no blur)")
    if cfg["model"]["backbone"] != "resnet18_cifar" or cfg["model"]["h_dim"] != 512:
        raise ConfigError("backbone must be resnet18_cifar with h_dim 512")
    if cfg["optimizer"]["name"] != "adamw" or cfg["schedule"]["kind"] != "linear_warmup_cosine" \
            or cfg["schedule"]["unit"] != "optimizer_step" or cfg["schedule"]["scale_lr_with_batch"]:
        raise ConfigError("optimizer/schedule deviate from the frozen recipe")
    # method-specific consistency
    crit = cfg["model"]["critic"]["enabled"]
    cv = cfg["evaluation"]["critic_validation"]["enabled"]
    loss = cfg["objective"]["loss"]
    expect = {"vcs_qmi": ("negative_J", True), "simclr_matched": ("nt_xent", False),
              "vicreg_matched_128": ("variance_invariance_covariance", False), "cs_kernel_native": ("negative_kernel_cs", False)}[m]
    js_matched = m == "vcs_qmi" and loss == "js_matched_logistic"  # P95 control (Server Spec v2 §2.2), same critic / pairing / detach
    if loss != expect[0] and not js_matched:
        raise ConfigError(f"objective.loss {loss!r} inconsistent with method {m!r}")
    if crit != expect[1] or cv != expect[1]:
        raise ConfigError(f"critic.enabled / critic_validation.enabled must be {expect[1]} for method {m!r}")
    if m == "cs_kernel_native":
        # CS-K-native (Server Spec v2 §3.2): classical kernel CS-QMI on L2-normalised pair representations, native autodiff, no detach,
        # no negative branch; bandwidth = objective.kernel_cs_bandwidth_multiple x FIT median pairwise distance (calibrated at run start).
        if not (cfg["objective"]["kernel_cs_bandwidth_multiple"] > 0):
            raise ConfigError("objective.kernel_cs_bandwidth_multiple must be > 0")
        if cfg["objective"]["kernel_cs_chunk"] < 0:
            raise ConfigError("objective.kernel_cs_chunk must be >= 0 (0 = dense)")
        if cfg["model"]["normalization"]["vcs_and_simclr"] != "l2":
            raise ConfigError("cs_kernel_native computes its kernels on the L2-normalised projector output (same input as the VCS critic)")
        if cfg["objective"]["target"] != "classical_kernel_CS_QMI":
            raise ConfigError("cs_kernel_native must declare objective.target = classical_kernel_CS_QMI (fixed reference measure, not S)")
    if m == "vcs_qmi":
        c = cfg["model"]["critic"]
        if c["input"] not in ("ordered_concat", "concat_interact", "bilinear_concat", "cosine", "interact_only", "shared_metric", "mono_spline", "diag_metric", "rff_tanh",
                              "residual_cosine_mlp", "dictionary_simplex") or c["activation"] != "relu" \
                or c["output"] != "tanh" or c["batchnorm"] or c["dropout"] != 0.0 or c["last_layer_bias"] != 0.0:
            raise ConfigError("critic input must be a named variant (ordered_concat|concat_interact|bilinear_concat|cosine|interact_only|shared_metric|mono_spline|diag_metric|rff_tanh); ReLU, tanh, no BN/dropout")
        if c["input"] == "rff_tanh":
            # S-Kernel (Server Spec v2 §3.4): fixed random-Fourier features of [z1; z2], trainable read-out + intercept, tanh; same J / K / detach
            if not (isinstance(c["rff_features"], int) and c["rff_features"] >= 1) or not (c["rff_bandwidth_multiple"] > 0):
                raise ConfigError("critic.rff_features must be a positive int and critic.rff_bandwidth_multiple > 0")
            if c["feature_source"] != "z":
                raise ConfigError("rff_tanh reads the projector output z (feature_source 'z')")
        _p95_policy(cfg)
        _p104_policy(cfg)
        hd = c["hidden_dims"]
        if not (isinstance(hd, list) and len(hd) >= 1 and all(isinstance(w, int) and w > 0 for w in hd)):
            raise ConfigError("critic.hidden_dims must be a non-empty list of positive ints (hyper-parameter variant)")
        if not (isinstance(c["last_layer_xavier_gain"], (int, float)) and c["last_layer_xavier_gain"] > 0):
            raise ConfigError("critic.last_layer_xavier_gain must be > 0 (exactly zero blocks encoder gradients at step 1)")
        if cfg["optimizer"]["critic_lr_multiplier"] <= 0 or cfg["optimizer"]["critic_weight_decay"] < 0:
            raise ConfigError("critic_lr_multiplier must be > 0 and critic_weight_decay >= 0")
    n = cfg["model"]["normalization"]
    if n["vcs_and_simclr"] not in ("l2", "none") or n["vicreg"] != "none":
        raise ConfigError("normalization policy: vcs/simclr in {l2, none} (none = raw projector output into the critic), vicreg none")
    if m == "simclr_matched" and n["vcs_and_simclr"] != "l2":
        raise ConfigError("SimCLR control always uses l2-normalized views")
    pr = cfg["model"]["projector"]
    if not (isinstance(pr["hidden_dim"], int) and pr["hidden_dim"] > 0 and isinstance(pr["output_dim"], int) and pr["output_dim"] > 0
            and pr["hidden_batchnorm"] and not pr["hidden_linear_bias"]):
        raise ConfigError("projector must stay Linear(no bias)-BN-ReLU-...-Linear with positive widths")
    if pr["output_batchnorm"]:
        if m != "vcs_qmi":
            raise ConfigError("output BN is a VCS-only named variant")
        if pr["output_linear_bias"]:
            raise ConfigError("output BN variant requires output_linear_bias=false (bias absorbed by BN)")
    elif not pr["output_linear_bias"]:
        raise ConfigError("without output BN the last projector layer keeps its bias (frozen recipe)")
    if m != "vcs_qmi" and (pr["hidden_dim"] != 512 or pr["output_dim"] != 128):
        raise ConfigError("control runs keep the frozen 512/128 projector")
    if cfg["evaluation"]["feature"] != "h_before_projector" or not cfg["evaluation"]["freeze_encoder_parameters"] \
            or not cfg["evaluation"]["freeze_bn_buffers"]:
        raise ConfigError("evaluation must use frozen h before the projector")


def load_config(path: str | Path, *, env: dict[str, str] | None = None, require_dirs: bool = True) -> dict[str, Any]:
    """Load, validate, resolve and policy-check a frozen config.  Returns the resolved dict with a ``_meta`` block."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate(raw, SCHEMA, "config")
    env = dict(os.environ) if env is None else env
    resolved = _resolve_env(copy.deepcopy(raw), "config", env)
    policy_checks(resolved)
    if require_dirs:
        for key, val in (("data.root", resolved["data"]["root"]), ("run.output_root", resolved["run"]["output_root"])):
            if not Path(val).is_dir():
                raise ConfigError(f"{key} resolved to {val!r}, which is not an existing directory")
        mdir = Path(resolved["data"]["manifest"]).parent
        if not mdir.is_dir():
            raise ConfigError(f"manifest directory {mdir} does not exist")
    resolved["_meta"] = {
        "config_path": str(path.resolve()),
        "config_file_sha256": sha256_file(path),
        "config_hash": sha256_json(_strip_meta(resolved)),
        "env_placeholders": sorted({m.group(1) for m in _ENV_RE.finditer(path.read_text(encoding="utf-8"))}),
    }
    return resolved


def _strip_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != "_meta"}


def dump_resolved(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(_strip_meta(cfg), sort_keys=False, allow_unicode=True) + \
        "# _meta:\n" + "".join(f"#   {k}: {v}\n" for k, v in cfg.get("_meta", {}).items())


def load_resolved(path: str | Path) -> dict[str, Any]:
    """Load a config.resolved.yaml written by the trainer (already resolved, comments carry meta)."""
    text = Path(path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    _validate(cfg, SCHEMA, "config.resolved")
    policy_checks(cfg)
    meta: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#   ") and ":" in line:
            k, v = line[4:].split(":", 1)
            meta[k.strip()] = v.strip()
    cfg["_meta"] = meta
    cfg["_meta"]["config_hash_recomputed"] = sha256_json(_strip_meta(cfg))
    return cfg
