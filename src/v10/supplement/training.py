"""Full method construction, including FIT-only kernel centres/bandwidth."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from v10 import adapter as A
from .comir import CoMIRConfig, CoMIRTrainer
from .contract import HOST_METHODS
from .registration import patches
from .rulsif import ExactRuLSIF


@dataclass(frozen=True)
class FixedConfig:
    method: str
    seed: int
    candidate: dict


class FixedTrainer:
    """No optimizer or fictional gradient steps for MI/NMI/MIND."""
    def __init__(self, family, candidate, seed, device):
        self.cfg, self.device = FixedConfig(family, seed, candidate), torch.device(device)
        self.n_steps, self.optimizers = 0, []

    def modules(self):
        return ()

    def rng_state(self):
        return {}

    def restore_rng(self, state):
        if state:
            raise ValueError("Training-free method received unexpected RNG state")


class KernelTrainer(FixedTrainer):
    def __init__(self, candidate, seed, guard, device):
        super().__init__("RuLSIF_exact", candidate, seed, device)
        fit = A.fit_cores(guard, 1.0, seed)
        if len(fit) != 32:
            raise RuntimeError("Kernel fitting requires the full frozen FIT32")
        self.scenes = []
        g = np.random.default_rng(A.Q.seed_of(f"V10|RuLSIF|FIT|{seed}"))
        for core in fit:
            guard.check(core, {"FIT"}, "V10 RuLSIF positive patches and exact product moments")
            left, right = A.P.load_img(guard.r6_root, core, ("bf", "shg"))
            image_left = torch.as_tensor(left, device=self.device, dtype=torch.float32)[None]
            image_right = torch.as_tensor(right, device=self.device, dtype=torch.float32)[None]
            positions = torch.as_tensor(g.uniform(5, A.P.SIZE - 6, (candidate["fit_positions_per_core"], 2)),
                                        device=self.device, dtype=torch.float32)
            self.scenes.append((patches(image_left, positions), patches(image_right, positions)))
        # Sample paired kernel centres equally from every FIT scene (4 per core).
        if candidate["centres"] % len(fit):
            raise ValueError("Kernel centres must be balanced over FIT scenes")
        centre_left, centre_right = [], []
        count = candidate["centres"] // len(fit)
        for left, right in self.scenes:
            ids = torch.as_tensor(g.choice(len(left), count, replace=False), device=self.device)
            centre_left.append(left[ids]); centre_right.append(right[ids])
        cx, cy = torch.cat(centre_left), torch.cat(centre_right)
        # Median distance uses only these preselected paired FIT centres. No CAL
        # or evaluation images set the kernel scale; multipliers are fixed candidates.
        distances = torch.pdist(torch.cat((cx, cy), -1).double())
        positive_distances = distances[distances > 0]
        if positive_distances.numel() == 0:
            raise RuntimeError("Degenerate FIT-only kernel bandwidth")
        median = float(positive_distances.median())
        self.ratio = ExactRuLSIF(cx, cy, median * candidate["bandwidth_multiplier"],
                                 candidate["ridge"], candidate["alpha"])
        self.fit_metadata = {"fit_cores": fit, "bandwidth_median": median,
                             "bandwidth": self.ratio.bandwidth,
                             "centres_selection": "paired, equal count per FIT core, without replacement",
                             "fit_positions_per_core": candidate["fit_positions_per_core"]}

    def modules(self):
        return (self.ratio,)

    def step(self):
        if self.n_steps != 0:
            raise RuntimeError("Kernel fit is one complete analytic fit")
        info = self.ratio.fit(self.scenes, q_mode="v")
        self.n_steps = 1
        return {"step": 1, "kind": "kernel_fit", **self.fit_metadata, **info}


def new_trainer(family, candidate, seed, guard, device):
    if family in HOST_METHODS:
        fit = A.fit_cores(guard, 1.0, seed)
        if len(fit) != 32:
            raise RuntimeError("Host supplemental method requires full FIT32")
        images = A.load_learned_inputs(guard, fit, {"FIT"}, "V10 supplemental full FIT training", device)
        config = A.TrainConfig(method=HOST_METHODS[family], seed=seed,
                               encoder_lr=candidate["lr"], head_lr=candidate["lr"],
                               n_sc=16, n_pos=128, crop=160, mlp_perms=8)
        trainer = A.Trainer(config, images, fit, device)
        trainer.optimizers = [trainer.opt]
        return trainer
    if family == "CoMIR_native":
        config = CoMIRConfig(seed=seed, epochs=candidate["epochs"], steps_per_epoch=candidate["steps_per_epoch"],
                             batch_size=candidate["batch_size"], learning_rate=candidate["lr"])
        return CoMIRTrainer(config, guard, device)
    if family == "RuLSIF_exact":
        return KernelTrainer(candidate, seed, guard, device)
    if family == "RuLSIF_exact_v2":
        from .rulsif_v2 import KernelTrainerV2  # separate registered variant; v1 above unchanged
        return KernelTrainerV2(candidate, seed, guard, device)
    if family in ("MI_native", "NMI_native", "MIND_2D_adapted"):
        return FixedTrainer(family, candidate, seed, device)
    raise ValueError("Unknown supplemental family: " + family)


def rng_state(trainer):
    if hasattr(trainer, "sampler"):
        return {"sampler": trainer.sampler.g.bit_generator.state, "sampler_step": trainer.sampler.step,
                "extra": trainer.g_extra.bit_generator.state}
    return trainer.rng_state()


def restore_rng(trainer, state):
    if hasattr(trainer, "sampler"):
        trainer.sampler.g.bit_generator.state = state["sampler"]
        trainer.sampler.step = state["sampler_step"]
        trainer.g_extra.bit_generator.state = state["extra"]
        if trainer.sampler.step != trainer.n_steps:
            raise RuntimeError("Host sampling step differs from checkpoint step")
    else:
        trainer.restore_rng(state)


def configuration(trainer):
    config = asdict(trainer.cfg)
    if isinstance(trainer, KernelTrainer) or getattr(trainer, "kernel_v2", False):
        config["resolved_kernel"] = trainer.fit_metadata
    if isinstance(trainer, CoMIRTrainer):
        config["official_architecture"] = trainer.architecture
        config["official_source"] = trainer.source
    return config
