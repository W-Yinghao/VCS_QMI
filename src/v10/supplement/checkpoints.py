"""Complete resumable states for host, native, kernel and training-free methods."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path

import torch

from v10 import adapter as A
from .contract import read, sha
from .records import write
from .training import configuration, restore_rng, rng_state


def canonical_config(trainer):
    return json.loads(json.dumps(configuration(trainer), allow_nan=False))


def save(trainer, path, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema": "v10.supplement.checkpoint.v1", "identity": identity,
             "config": canonical_config(trainer), "step": trainer.n_steps,
             "modules": [module.state_dict() for module in trainer.modules()],
             "optimizers": [optimizer.state_dict() for optimizer in trainer.optimizers],
             "local_rng": rng_state(trainer), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else [],
             "model_sha256": A.state_hash(*trainer.modules())}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, temporary); os.replace(temporary, path)
    metadata = {"identity": identity, "step": trainer.n_steps, "sha256": sha(path),
                "model_sha256": state["model_sha256"]}
    write(path.with_suffix(".json"), metadata)
    return metadata


def metadata(path, identity):
    path = Path(path)
    result = read(path.with_suffix(".json"))
    if result["identity"] != identity or result["sha256"] != sha(path):
        raise RuntimeError("Checkpoint identity or content changed")
    return result


def load(trainer, path, identity):
    meta = metadata(path, identity)
    state = torch.load(path, map_location=trainer.device, weights_only=True)
    if state["identity"] != identity or state["config"] != canonical_config(trainer):
        raise RuntimeError("Checkpoint configuration differs from the current unit")
    if state["schema"] != "v10.supplement.checkpoint.v1" or state["step"] != meta["step"]:
        raise RuntimeError("Checkpoint schema/step mismatch")
    for module, value in zip(trainer.modules(), state["modules"], strict=True):
        module.load_state_dict(value, strict=True)
    for optimizer, value in zip(trainer.optimizers, state["optimizers"], strict=True):
        optimizer.load_state_dict(value)
    trainer.n_steps = state["step"]
    restore_rng(trainer, state["local_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if trainer.device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    restored = A.state_hash(*trainer.modules())
    if restored != state["model_sha256"] or restored != meta["model_sha256"]:
        raise RuntimeError("Restored model differs from saved complete pair")
    return meta


@contextmanager
def evaluation_state(trainer):
    modules = trainer.modules()
    # Preserve every submodule's training flag, including mixed native states.
    modes = [(submodule, submodule.training) for module in modules for submodule in module.modules()]
    flags = [(parameter, parameter.requires_grad) for module in modules for parameter in module.parameters()]
    local_rng = rng_state(trainer)
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else []
    before = A.state_hash(*modules)
    A.freeze_for_test(*modules)
    try:
        yield
    finally:
        changed = A.state_hash(*modules) != before
        for module, mode in modes:
            module.training = mode
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        restore_rng(trainer, local_rng)
        torch.set_rng_state(cpu_rng)
        if trainer.device.type == "cuda":
            torch.cuda.set_rng_state_all(gpu_rng)
        if changed:
            raise RuntimeError("Evaluation altered model parameters or buffers")
