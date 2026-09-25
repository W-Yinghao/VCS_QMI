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

METHODS = ("vcs_qmi", "simclr_matched", "vicreg_matched_128")


class ConfigError(ValueError):
    pass


# --- schema: nested dict of key -> type spec.  A tuple of types means "one of"; None means allow None. -----------
_Num = (int, float)


class _Opt:
    """Optional field: absent in older frozen configs -> filled with ``default`` (so downstream code always sees it)."""

    def __init__(self, types: Any, default: Any) -> None:
        self.types = types if isinstance(types, tuple) else (types,)
        self.default = default


SCHEMA: dict[str, Any] = {
    "schema_version": str,
    "run": {"stage": str, "method": str, "seed": int, "output_root": str, "resume": (str, type(None)), "overwrite": bool},
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
                            "output_linear_bias": bool, "output_batchnorm": bool, "depth": _Opt(int, 2), "predictor": _Opt(bool, False)},
              "normalization": {"vcs_and_simclr": str, "eps": _Num, "vicreg": str},
              "critic": {"enabled": bool, "input": str, "hidden_dims": list, "activation": str, "output": str,
                         "batchnorm": bool, "dropout": _Num, "last_layer_xavier_gain": _Num, "last_layer_bias": _Num,
                         "feature_source": _Opt(str, "z"), "cosine_scale_init": _Opt((int, float), 1.0),
                         "cosine_scale_fixed": _Opt(bool, False), "cosine_bias_calibrate": _Opt(bool, False)}},
    "objective": {"target": str, "loss": str, "positive_weight": _Num, "negative_weight": _Num,
                  "training_cs_transform": bool, "clip_J": bool, "extra_regularizers": list, "simclr_temperature": _Num,
                  "vicreg_weights": {"invariance": _Num, "variance": _Num, "covariance": _Num}, "vicreg_variance_eps": _Num},
    "pairing": {"sampler": str, "k": int, "unique_shifts": bool, "allow_self": bool, "label_filter": bool, "queue": bool,
                "negative_detach": bool, "rng": str, "rng_seed_offset": int},
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


def policy_checks(cfg: dict[str, Any]) -> None:
    """First-round invariants from spec §1.2, §6, §8, §12, §15.2(8)."""
    if cfg["schema_version"] != "vcs_ssl_agent_1.0":
        raise ConfigError(f"unsupported schema_version {cfg['schema_version']!r}")
    m = cfg["run"]["method"]
    if m not in METHODS:
        raise ConfigError(f"run.method must be one of {METHODS}, got {m!r}")
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
    if p["sampler"] not in ("random_nonzero_cyclic_shift", "random_nonzero_cyclic_shift_symmetric") or not p["unique_shifts"] \
            or p["allow_self"] or p["label_filter"] or p["queue"] or p["rng"] != "dedicated_cpu_generator":
        raise ConfigError("pairing policy deviates from the frozen definition (symmetric scoring of both orders is the only named variant)")
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
    if cfg["model"]["projector"]["depth"] < 2 or cfg["model"]["projector"]["depth"] > 4:
        raise ConfigError("projector depth must be in [2, 4]")
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
    if cfg["views"]["count"] != 2:
        raise ConfigError("two views only")
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
              "vicreg_matched_128": ("variance_invariance_covariance", False)}[m]
    if loss != expect[0]:
        raise ConfigError(f"objective.loss {loss!r} inconsistent with method {m!r}")
    if crit != expect[1] or cv != expect[1]:
        raise ConfigError(f"critic.enabled / critic_validation.enabled must be {expect[1]} for method {m!r}")
    if m == "vcs_qmi":
        c = cfg["model"]["critic"]
        if c["input"] not in ("ordered_concat", "concat_interact", "bilinear_concat", "cosine", "interact_only", "shared_metric") or c["activation"] != "relu" \
                or c["output"] != "tanh" or c["batchnorm"] or c["dropout"] != 0.0 or c["last_layer_bias"] != 0.0:
            raise ConfigError("critic input must be a named variant (ordered_concat|concat_interact|bilinear_concat|cosine|interact_only|shared_metric); ReLU, tanh, no BN/dropout")
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
