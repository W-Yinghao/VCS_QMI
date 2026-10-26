"""Task anchors and explicitly named pooled-feature distillation controls."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def kd_loss(student, teacher, temperature=4.0):
    return F.kl_div(F.log_softmax(student / temperature, dim=1),
                    F.softmax(teacher.detach() / temperature, dim=1), reduction="batchmean") * temperature**2


def dkd_loss(student, teacher, labels, alpha=1.0, beta=8.0, temperature=4.0):
    """DKD equations, matching pinned mdistiller with stable log-softmax masking."""
    teacher = teacher.detach()
    mask = torch.zeros_like(student, dtype=torch.bool).scatter_(1, labels[:, None], True)
    ps = F.softmax(student / temperature, dim=1)
    pt = F.softmax(teacher / temperature, dim=1)
    merged_s = torch.cat(((ps * mask).sum(1, keepdim=True), (ps * ~mask).sum(1, keepdim=True)), 1)
    merged_t = torch.cat(((pt * mask).sum(1, keepdim=True), (pt * ~mask).sum(1, keepdim=True)), 1)
    tckd = F.kl_div(merged_s.clamp_min(torch.finfo(student.dtype).tiny).log(), merged_t, reduction="batchmean")
    ls_other = F.log_softmax(student / temperature - mask.to(student.dtype) * 1000., dim=1)
    pt_other = F.softmax(teacher / temperature - mask.to(teacher.dtype) * 1000., dim=1)
    nckd = F.kl_div(ls_other, pt_other, reduction="batchmean")
    return (alpha * tckd + beta * nckd) * temperature**2


class FeatureControl(nn.Module):
    def __init__(self, student_dim, teacher_dim, mode, init_variance=5.0, eps=1e-5):
        super().__init__()
        self.mode, self.eps = mode, eps
        if mode == "vid_pooled":
            self.regressor = nn.Sequential(nn.Linear(student_dim, teacher_dim, bias=False), nn.ReLU(),
                                            nn.Linear(teacher_dim, teacher_dim, bias=False), nn.ReLU(),
                                            nn.Linear(teacher_dim, teacher_dim, bias=False))
            # Registered and optimized variance; Gaussian log-variance is retained.
            self.log_scale = nn.Parameter(torch.full((teacher_dim,), math.log(math.expm1(init_variance - eps))))
        elif mode in ("mse", "cosine"):
            self.regressor = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            raise ValueError(mode)

    def forward(self, student, teacher):
        target = teacher.detach()
        pred = self.regressor(student)
        if self.mode == "mse":
            return F.mse_loss(pred, target)
        if self.mode == "cosine":
            return (1.0 - F.cosine_similarity(pred, target, dim=1)).mean()
        variance = F.softplus(self.log_scale) + self.eps
        return 0.5 * (((pred - target).square() / variance) + variance.log()).mean()


class DistillationLoss(nn.Module):
    def __init__(self, student_dim, teacher_dim, config):
        super().__init__()
        self.cfg = config
        dep = config.get("dependence", {})
        self.dependence = None
        if dep.get("method", "none") != "none":
            from vcs_common.dependence import DependenceRegularizer
            self.dependence = DependenceRegularizer(student_dim, teacher_dim, **dep)
        self.feature = None
        feature = config.get("feature", "none")
        if feature == "itrd":
            from .itrd_reference import ITRDFeatureLoss
            self.feature = ITRDFeatureLoss(student_dim, teacher_dim, **config.get("itrd", {}))
        elif feature != "none":
            self.feature = FeatureControl(student_dim, teacher_dim, feature)

    def prepare_step(self, student_features, teacher_features):
        if self.dependence is not None:
            if teacher_features.requires_grad:
                raise ValueError("Teacher features must be produced under no_grad")
            # M1 internally detaches its quadratic solve while retaining the
            # student graph for the outer step; forward must reuse these objects.
            self.dependence.prepare_step(student_features, teacher_features)

    def project_(self):
        if self.dependence is not None:
            self.dependence.project_()

    def forward(self, logits_s, logits_t, feat_s, feat_t, labels, epoch):
        task = F.cross_entropy(logits_s, labels) * self.cfg.get("ce_weight", 1.0)
        self.last_task = task
        self.last_representation_loss = None
        total = task
        parts = {"task": float(task.detach())}
        anchor = self.cfg.get("anchor", "none")
        if anchor != "none":
            if logits_t is None:
                raise ValueError("A teacher is required by the selected anchor")
            t = self.cfg.get("temperature", 4.0)
            if anchor == "kd":
                loss = kd_loss(logits_s, logits_t, t)
            elif anchor == "dkd":
                loss = dkd_loss(logits_s, logits_t, labels, self.cfg.get("dkd_alpha", 1.), self.cfg.get("dkd_beta", 8.), t)
                warm = self.cfg.get("dkd_warmup", 20)
                loss = loss * min((epoch + 1) / max(warm, 1), 1.)
            else:
                raise ValueError(anchor)
            total = total + self.cfg.get("anchor_weight", 1.0) * loss
            parts[anchor] = float(loss.detach())
        if self.feature is not None:
            loss = self.feature(feat_s, feat_t)
            weighted = self.cfg.get("feature_weight", 1.0) * loss
            total = total + weighted
            self.last_representation_loss = weighted
            parts["feature"] = float(loss.detach())
        if self.dependence is not None:
            loss = self.dependence(feat_s, feat_t)
            weighted = self.cfg.get("dependence_weight", 1.0) * loss
            total = total + weighted
            self.last_representation_loss = weighted if self.last_representation_loss is None else self.last_representation_loss + weighted
            parts["dependence"] = float(loss.detach())
        decision=self.cfg.get("decision",{})
        if decision.get("method","none")!="none":
            from .decision import decision_loss
            auxiliary,self.decision_diagnostics=decision_loss(logits_s,logits_t,decision["method"])
            weighted=decision["weight"]*auxiliary
            total=total+weighted
            self.last_representation_loss=weighted
            parts["decision"]=float(auxiliary.detach())
        return total, parts
