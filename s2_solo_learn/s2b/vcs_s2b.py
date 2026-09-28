# S2b (CS_QMI P77): the P75 VCS port with the critic scalars (a, b) updated by the recipe's optimiser instead of the
# LARS-excluded plain SGD-momentum step at the protocol's lr (P75 deviation 2, named there as the first suspect).
# Everything else — projector, K = 8 cyclic shifts, negative detach, loss = -J, augmentation, LARS for backbone /
# projector / classifier, schedule, precision, test-set rule — is inherited unchanged from solo/methods/vcs.py.
#
# Implementation: a second optimiser (torch.optim.AdamW, lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight_decay 0 — the
# ssl_pilot recipe's optimizer block for the critic, critic_lr_multiplier 1) with the recipe's schedule shape (linear
# warm-up over scheduler.warmup_epochs from 0, cosine to min_lr_ratio x peak).  Lightning 2.x requires manual
# optimisation for two optimisers: training_step calls manual_backward, steps both optimisers through the precision
# plugin (GradScaler unscale/step per optimiser under 16-mixed) and steps both schedulers per step.
from typing import Any, Dict, List, Sequence

import omegaconf
import torch
from solo.methods.vcs import VCS
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR


class VCSS2B(VCS):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.critic_adamw_lr: float = float(cfg.method_kwargs.critic_adamw_lr)
        self.critic_adamw_min_lr_ratio: float = float(cfg.method_kwargs.critic_adamw_min_lr_ratio)
        self.automatic_optimization = False

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = VCS.add_and_assert_specific_cfg(cfg)
        cfg.method_kwargs.critic_adamw_lr = omegaconf.OmegaConf.select(cfg, "method_kwargs.critic_adamw_lr", default=1e-3)
        cfg.method_kwargs.critic_adamw_min_lr_ratio = omegaconf.OmegaConf.select(
            cfg, "method_kwargs.critic_adamw_min_lr_ratio", default=0.01
        )
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        # the critic is NOT in the LARS optimiser here; it gets its own AdamW in configure_optimizers
        return [g for g in super().learnable_params if g.get("name") != "critic"]

    def configure_optimizers(self):
        base = super().configure_optimizers()
        if not isinstance(base, (tuple, list)) or len(base) != 2:
            raise RuntimeError("S2b expects solo-learn's ([optimizer], [scheduler]) return")
        optimizers, schedulers = base
        copt = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_adamw_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        if self.scheduler_interval == "step":
            warm = self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
            total = self.trainer.estimated_stepping_batches
        else:
            warm, total = self.warmup_epochs, self.max_epochs
        csched = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                copt,
                warmup_epochs=warm,
                max_epochs=total,
                warmup_start_lr=0.0,
                eta_min=self.critic_adamw_min_lr_ratio * self.critic_adamw_lr,
            ),
            "interval": self.scheduler_interval,
            "frequency": 1,
        }
        return list(optimizers) + [copt], list(schedulers) + [csched]

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        loss = super().training_step(batch, batch_idx)  # -J + online classifier loss, metrics logged there
        opts = self.optimizers()
        for o in opts:
            o.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        for o in opts:
            o.step()
        if self.scheduler_interval == "step":
            for s in self.lr_schedulers():
                s.step()
        self.log_dict(
            {"critic_lr": float(opts[1].param_groups[0]["lr"]), "main_lr": float(opts[0].param_groups[0]["lr"])},
            on_step=True, on_epoch=False, sync_dist=False,
        )
        return loss

    def on_train_epoch_end(self):
        if self.scheduler_interval == "epoch":
            for s in self.lr_schedulers():
                s.step()
        return super().on_train_epoch_end()
