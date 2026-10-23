"""Application optimization policy; the original dependence objective is reused.

No new P/Q, entropy term, information measure, label filtering, or score clamp.
This is a finite-step optimizer recipe, not an exact inner supremum solver.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math

import torch


MODES = {"joint_scheduled", "alternating_target_sg"}


def _finite_nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def validate_policy(cfg, default_weight):
    """Validate the complete, explicitly named policy; no silent new defaults."""
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    allowed = {"mode", "warmup_epochs", "ramp_epochs", "critic_steps", "critic_lr",
               "critic_weight_decay", "critic_lr_step_epochs", "critic_lr_gamma"}
    if set(cfg) != allowed:
        raise ValueError(f"Application policy fields differ: {set(cfg) ^ allowed}")
    if cfg["mode"] not in MODES:
        raise ValueError("Unknown application update mode")
    _finite_nonnegative(default_weight, "outer weight")
    for name in ("warmup_epochs", "ramp_epochs"):
        if isinstance(cfg[name], bool) or int(cfg[name]) != cfg[name] or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if cfg["ramp_epochs"] < 1:
        raise ValueError("The fixed recipe requires a positive ramp length")
    if cfg["critic_steps"] != 1:
        raise ValueError("This small-budget recipe fixes exactly one critic step")
    if not math.isfinite(cfg["critic_lr"]) or cfg["critic_lr"] <= 0:
        raise ValueError("critic_lr must be finite and positive")
    if cfg["critic_weight_decay"] != 0:
        raise ValueError("The critic is fitted to the unregularized original objective")
    if int(cfg["critic_lr_step_epochs"]) != cfg["critic_lr_step_epochs"] or cfg["critic_lr_step_epochs"] < 1:
        raise ValueError("critic_lr_step_epochs must be a positive integer")
    if not math.isfinite(cfg["critic_lr_gamma"]) or not 0 < cfg["critic_lr_gamma"] <= 1:
        raise ValueError("Invalid fixed critic LR decay")
    return cfg


def scheduled_weight(base_weight, epoch, warmup_epochs, ramp_epochs):
    """Epoch is zero based; warmup2/ramp4 gives 0,0,.25,.5,.75,1 times lambda."""
    if epoch < 0 or int(epoch) != epoch:
        raise ValueError("Expected a nonnegative integer epoch")
    if epoch < warmup_epochs:
        return 0.0
    return float(base_weight) * min((epoch - warmup_epochs + 1) / ramp_epochs, 1.0)


@contextmanager
def frozen_critic(module):
    """Freeze both projection maps and W while retaining input autograd.

    Restoring requires_grad after forward does not add parameter edges to the
    already-built graph. Never use no_grad here: x must receive the outer grad.
    """
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def assert_disjoint_optimizers(*optimizers):
    seen = set()
    for optimizer in optimizers:
        if optimizer is None:
            continue
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise AssertionError("A parameter occurs in more than one optimizer role")
        seen.update(ids)


def representation_gradient_diagnostics(anchor, weighted_aux, representation):
    """Observed gradients only: never rescale or modify the training objective."""
    def grad(loss):
        if loss is None or not loss.requires_grad:
            return None
        return torch.autograd.grad(loss, representation, retain_graph=True, allow_unused=True)[0]
    ga, gu = grad(anchor), grad(weighted_aux)
    na = 0.0 if ga is None else float(ga.detach().norm())
    nu = 0.0 if gu is None else float(gu.detach().norm())
    cosine = None
    if na > 0 and nu > 0:
        cosine = float((ga.detach().flatten() @ gu.detach().flatten()) / (na * nu))
    return {"anchor_gradient_norm": na, "weighted_aux_gradient_norm": nu,
            "weighted_aux_over_anchor": nu / na if na > 0 else None,
            "anchor_aux_cosine": cosine}


class ApplicationUpdate:
    """Plain controller: regularizers stay registered exactly once by their owner.

    Critic optimizers are constructed lazily, after the owning nn.Module.to().
    Only original exact VCS and its bounded matched BCE are supported here.
    The target detach removes the direct auxiliary target edge; shared host
    parameters can still receive gradients through x and the supervised task.
    """
    def __init__(self, regularizers, cfg, default_weight):
        self.regularizers = regularizers
        self.cfg = validate_policy(cfg, default_weight)
        self.base_weight = float(default_weight)
        self.epoch = 0
        self.optimizers = {}
        self.schedulers = {}
        self.updates = 0
        self.fit_diagnostics = {}
        if self.cfg:
            for reg in regularizers.values():
                if reg.method not in {"vcs_exact", "bce"} or reg.q_mode != "u":
                    raise ValueError("Application recipe supports exact/BCE and unchanged full off-diagonal U only")

    @property
    def alternating(self):
        return bool(self.cfg and self.cfg["mode"] == "alternating_target_sg")

    @property
    def weight(self):
        if not self.cfg:
            return self.base_weight
        return scheduled_weight(self.base_weight, self.epoch,
                                self.cfg["warmup_epochs"], self.cfg["ramp_epochs"])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def critic_parameter_ids(self):
        if not self.alternating:
            return set()
        return {id(p) for reg in self.regularizers.values() for p in reg.parameters()}

    def _ensure_optimizers(self):
        if not self.alternating or self.optimizers:
            return
        for name, reg in self.regularizers.items():
            opt = torch.optim.AdamW([p for p in reg.parameters() if p.requires_grad],
                                   lr=self.cfg["critic_lr"], betas=(.9, .999), eps=1e-8,
                                   weight_decay=0.0)
            self.optimizers[name] = opt
            self.schedulers[name] = torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.cfg["critic_lr_step_epochs"], gamma=self.cfg["critic_lr_gamma"])
        assert_disjoint_optimizers(*self.optimizers.values())

    def assert_outer_optimizer(self, *outer_optimizers):
        self._ensure_optimizers()
        assert_disjoint_optimizers(*outer_optimizers, *self.optimizers.values())

    def fit_critics(self, pairs):
        if not self.alternating:
            raise RuntimeError("fit_critics is only valid for alternating_target_sg")
        if set(pairs) != set(self.regularizers):
            raise ValueError("Every registered branch must use the same named input pair")
        self._ensure_optimizers()
        self.fit_diagnostics = {}
        for name, (x, y) in pairs.items():
            reg, opt = self.regularizers[name], self.optimizers[name]
            # These are the current minibatch values: no new augmentation,
            # label-conditioned sampling, reference bank, or host forward.
            xd, yd = x.detach(), y.detach()
            opt.zero_grad(set_to_none=True)
            value = reg(xd, yd)
            before = float(value.detach())
            value.backward()
            if not bool(torch.isfinite(value)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in reg.parameters()):
                raise FloatingPointError("Nonfinite unweighted critic fit")
            opt.step()
            reg.project_()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                after = float(reg(xd, yd))
            self.fit_diagnostics.update({f"{name}_critic_fit_loss_before": before,
                                         f"{name}_critic_fit_loss_after": after,
                                         f"{name}_critic_learning_rate": opt.param_groups[0]["lr"],
                                         f"{name}_x_batch_std_mean": float(xd.std(0, unbiased=False).mean()),
                                         f"{name}_y_batch_std_mean": float(yd.std(0, unbiased=False).mean())})
        self.updates += 1

    def outer_loss(self, name, x, y):
        reg = self.regularizers[name]
        if not self.alternating:
            return reg(x, y)
        with frozen_critic(reg):
            return reg(x, y.detach())

    def weighted(self, loss):
        # A true task-only warmup: do not manufacture zero critic gradients,
        # because AdamW could otherwise apply decay to a zero-gradient head.
        return self.weight * (loss if self.weight else loss.detach())

    def epoch_end(self):
        if self.alternating:
            for scheduler in self.schedulers.values():
                scheduler.step()

    def state_dict(self):
        return {"policy": copy.deepcopy(self.cfg), "base_weight": self.base_weight,
                "epoch": self.epoch, "updates": self.updates,
                "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
                "schedulers": {name: sch.state_dict() for name, sch in self.schedulers.items()}}

    def load_state_dict(self, state):
        if state["policy"] != self.cfg or state["base_weight"] != self.base_weight:
            raise ValueError("Application update policy changed on resume")
        self.epoch, self.updates = state["epoch"], state["updates"]
        if self.alternating:
            self._ensure_optimizers()
            if set(state["optimizers"]) != set(self.optimizers):
                raise ValueError("Missing critic optimizer state on resume")
            for name, opt in self.optimizers.items():
                opt.load_state_dict(state["optimizers"][name])
                self.schedulers[name].load_state_dict(state["schedulers"][name])

    def diagnostics(self):
        return dict(self.fit_diagnostics, active_aux_weight=self.weight,
                    critic_updates_total=self.updates)
