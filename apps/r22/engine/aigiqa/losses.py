"""Explicit baselines: reference CPC, complete Gaussian NLL, shared CS/BCE."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceCPC(nn.Module):
    def __init__(self, fusion_dim, modality_dim, temperature=1.0):
        super().__init__()
        self.project = nn.Linear(fusion_dim, modality_dim)
        self.temperature = temperature

    def forward(self, fusion, modality):
        prediction = F.normalize(self.project(fusion), dim=-1, eps=1e-8)
        modality = F.normalize(modality, dim=-1, eps=1e-8)
        # Same reference orientation: anchor modality, classify predicted fusion.
        logits = modality @ prediction.T / self.temperature
        return F.cross_entropy(logits, torch.arange(len(fusion), device=fusion.device))


class CorrectedGaussian(nn.Module):
    """Complete normalized Gaussian density; trainable target entropy is optional.

    This is a predictive regularizer when the textual target is trainable. An
    optional diagonal Gaussian entropy is explicitly a surrogate, not a Shannon
    entropy estimator or a generally valid Barber-Agakov lower bound.
    """
    def __init__(self, visual_dim, text_dim, entropy_weight=0.0):
        super().__init__()
        self.mean = nn.Sequential(nn.Linear(visual_dim, text_dim), nn.ReLU(), nn.Linear(text_dim, text_dim))
        self.logvar = nn.Sequential(nn.Linear(visual_dim, text_dim), nn.ReLU(), nn.Linear(text_dim, text_dim))
        self.entropy_weight = entropy_weight
        self.diagnostics = {}

    def forward(self, visual, text):
        mu, logvar = self.mean(visual), self.logvar(visual).clamp(-6, 6)
        nll = .5 * (math.log(2 * math.pi) + logvar + (text - mu).square() * (-logvar).exp())
        nll = nll.mean()  # Dimension-normalized, explicit difference from ref sum.
        entropy = .5 * (1 + math.log(2 * math.pi) + text.var(0, unbiased=False).add(1e-6).log()).mean()
        self.diagnostics = {"gaussian_nll_per_dim": float(nll.detach()),
                            "gaussian_diagonal_entropy_surrogate_per_dim": float(entropy.detach())}
        return nll - self.entropy_weight * entropy


class BankedDependence(nn.Module):
    """r22 (2026-10-22): same rank-r simplex-bilinear boxed critic and objective, new (P, Q) samples.

    T(x, y) = a(x)^T W b(y) in [-1, 1] (package DependenceRegularizer head; its init and RNG use are unchanged).
    Q = product of marginals, sampled as every (x_i, y_j), i != j, of the current batch (the package's U support)
        together with every (x_i, y_k) for the y_k in a FIFO bank of detached text embeddings of previous FIT
        batches; one denominator over all Q pairs. The bank is a buffer: no gradient reaches past text features.
    P = matched (x_i, y_i); with p_weighting="mos", P_w puts mass w_i / B on pair i, w_i = MOS_i / mean_batch MOS
        (a probability measure, sum_i w_i / B = 1). P and Q are normalized separately.
    vcs_exact: loss = -J(T) = -(E_P T - E_Q T - E_P T^2/2 - E_Q T^2/2).
    bce: the package's bounded matched BCE on the same score and the same P_w / Q:
        loss = -(E_P log(1 + T) + E_Q log(1 - T)) = 2*(balanced BCE - log 2).
    With bank_size=0 and uniform P this equals the package vcs_exact / bce with q_mode="u".
    """

    def __init__(self, x_dim, y_dim, method, rank=32, temperature=1.0, bank_size=4096, p_weighting="uniform", **unused):
        super().__init__()
        from vcs_common.dependence import DependenceRegularizer
        if method not in ("vcs_exact", "bce"):
            raise ValueError("The banked branch supports vcs_exact and its matched bce only")
        if p_weighting not in ("uniform", "mos"):
            raise ValueError("p_weighting must be uniform or mos")
        if isinstance(bank_size, bool) or int(bank_size) != bank_size or bank_size < 0:
            raise ValueError("bank_size must be a nonnegative integer")
        # The same head class, constructed by the package regularizer (identical parameter init / RNG draws).
        self.head = DependenceRegularizer(x_dim, y_dim, method=method, rank=rank, q_mode="u",
                                          temperature=temperature).head
        self.method, self.q_mode, self.p_weighting, self.bank_size = method, "u", p_weighting, int(bank_size)
        self.register_buffer("bank", torch.zeros(self.bank_size, y_dim))
        self.register_buffer("bank_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("bank_pointer", torch.zeros((), dtype=torch.long))
        self.weights = None
        self.diagnostics = {}

    @property
    def W(self):
        return self.head.W

    @torch.no_grad()
    def project_(self):
        self.head.project_head_()

    def set_labels(self, labels):
        """MOS weights of the current batch (labels are MOS/5, so the ratio to the batch mean is scale free)."""
        if self.p_weighting == "uniform" or labels is None:
            self.weights = None
            return
        labels = labels.detach().reshape(-1)
        labels = labels if labels.is_floating_point() else labels.to(torch.float32)
        if not bool(torch.isfinite(labels).all()) or bool((labels < 0).any()) or float(labels.mean()) <= 0:
            raise ValueError("MOS weights need finite nonnegative MOS with a positive batch mean")
        self.weights = labels / labels.mean()

    def _bank_features(self):
        n = int(self.bank_count)
        if n == 0:
            return None
        return torch.softmax(self.head.right(self.bank[:n]) / self.head.temperature, -1)

    def forward(self, x, y):
        from vcs_common.dependence import _paired, q_values
        _paired(x, y, "u")
        a, b = self.head.features(x, y)
        scores = a @ self.W @ b.T
        pos = scores.diagonal()
        negatives = [q_values(scores, "u")]
        bank_b = self._bank_features()
        if bank_b is not None:
            negatives.append((a @ self.W @ bank_b.T).reshape(-1))
        neg = torch.cat(negatives)
        if self.p_weighting == "mos":
            if self.weights is None or len(self.weights) != len(x):
                raise RuntimeError("MOS-weighted P needs set_labels() for this batch")
            w = self.weights.to(pos.dtype)
        else:
            w = torch.ones_like(pos)
        p1, p2 = (w * pos).mean(), (w * pos.square()).mean()
        q1, q2 = neg.mean(), neg.square().mean()
        j = p1 - q1 - .5 * (p2 + q2)
        if self.method == "vcs_exact":
            loss = -j
        else:
            eps = torch.finfo(scores.dtype).eps
            loss = -((w * torch.log1p(pos.clamp_min(-1 + 2 * eps))).mean() + torch.log1p((-neg).clamp_min(-1 + 2 * eps)).mean())
        self.diagnostics = dict(method=self.method, q_mode="u_plus_bank", batch=len(x), loss=float(loss.detach()),
                                raw_j=float(j.detach()), p1=float(p1.detach()), p2=float(p2.detach()),
                                q1=float(q1.detach()), q2=float(q2.detach()), q_pairs=int(neg.numel()),
                                bank_count=int(self.bank_count), p_weight_max=float(w.max()), p_weight_min=float(w.min()),
                                head_abs_max=float(self.W.detach().abs().max()))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite dependence loss; preserve run as FAILED")
        return loss

    @torch.no_grad()
    def enqueue(self, y):
        """FIFO: write the current batch's detached text embeddings after this step's losses are computed."""
        if self.bank_size == 0:
            return
        y = y.detach().to(self.bank.dtype)
        n = len(y)
        idx = (int(self.bank_pointer) + torch.arange(n, device=self.bank.device)) % self.bank_size
        self.bank.index_copy_(0, idx, y[-self.bank_size:] if n > self.bank_size else y)
        self.bank_pointer.fill_((int(self.bank_pointer) + n) % self.bank_size)
        self.bank_count.fill_(min(int(self.bank_count) + n, self.bank_size))


class IQALosses(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        from vcs_common.dependence import DependenceRegularizer
        self.cfg = cfg
        method = cfg.get("method", "none")
        self.method = method
        self.forward_regularizer = None
        if cfg.get("forward", "none") == "corrected_gaussian":
            self.forward_regularizer = CorrectedGaussian(model.visual_dim, model.text_dim,
                                                         cfg.get("entropy_surrogate_weight", 0.0))
        elif cfg.get("forward", "none") != "none":
            raise ValueError("forward must be none or corrected_gaussian")
        self.regularizers = nn.ModuleDict()
        self.branches = cfg.get("branches", ["text"])
        if len(set(self.branches)) != len(self.branches) or any(b not in {"text", "visual"} for b in self.branches):
            raise ValueError("branches must be distinct text / visual names")
        if method != "none":
            for branch in self.branches:
                dim = model.text_dim if branch == "text" else model.visual_dim
                if method == "reference_infonce":
                    regularizer = ReferenceCPC(model.fusion_dim, dim, cfg.get("temperature", 1.0))
                elif "bank_size" in cfg or "p_weighting" in cfg:
                    if cfg.get("q_mode", "u") != "u":
                        raise ValueError("The banked branch extends the U support only")
                    regularizer = BankedDependence(model.fusion_dim, dim, method, rank=cfg.get("rank", 32),
                                                   temperature=cfg.get("temperature", 1.0),
                                                   bank_size=cfg.get("bank_size", 0),
                                                   p_weighting=cfg.get("p_weighting", "uniform"))
                else:
                    regularizer = DependenceRegularizer(
                        x_dim=model.fusion_dim, y_dim=dim, method=method,
                        rank=cfg.get("rank", 32), q_mode=cfg.get("q_mode", "u"),
                        sampled_draws=cfg.get("sampled_draws", 8), temperature=cfg.get("temperature", 1.0),
                        solver_max_iter=cfg.get("solver_max_iter", 30),
                        solver_relative_tol=cfg.get("solver_relative_tol", .01))
                self.regularizers[branch] = regularizer
        from vcs_common.application_update import ApplicationUpdate
        self.application_update = ApplicationUpdate(self.regularizers, cfg.get("application_update"), cfg.get("weight", .1))
        if self.application_update.cfg and (self.forward_regularizer is not None or self.branches != ["text"]):
            raise ValueError("This application experiment fixes one text branch and no Gaussian term")
        self.diagnostics = {}

    def _set_labels(self, labels):
        for regularizer in self.regularizers.values():
            if isinstance(regularizer, BankedDependence):
                regularizer.set_labels(labels)

    def prepare_step(self, outputs, labels=None):
        self._set_labels(labels)
        if self.application_update.alternating:
            self.application_update.fit_critics({branch: (outputs["fusion"], outputs[branch]) for branch in self.regularizers})
            return
        for branch, regularizer in self.regularizers.items():
            if hasattr(regularizer, "prepare_step"):
                regularizer.prepare_step(outputs["fusion"], outputs[branch])

    def project_(self):
        for regularizer in self.regularizers.values():
            if hasattr(regularizer, "project_"):
                regularizer.project_()

    def forward(self, outputs, labels):
        task = F.mse_loss(outputs["prediction"], labels)
        self._set_labels(labels)
        dependency = task.new_zeros(())
        diagnostics = {"task_mse": float(task.detach())}
        for branch, regularizer in self.regularizers.items():
            loss = self.application_update.outer_loss(branch, outputs["fusion"], outputs[branch])
            dependency = dependency + loss
            diagnostics[f"{branch}_dependence_loss"] = float(loss.detach())
            for key, value in getattr(regularizer, "diagnostics", {}).items():
                if isinstance(value, (float, int)):
                    diagnostics[f"{branch}_{key}"] = value
        # Per-branch mean preserves the named weight when branch count changes.
        if self.regularizers:
            dependency = dependency / len(self.regularizers)
        forward = task.new_zeros(())
        if self.forward_regularizer is not None:
            forward = self.forward_regularizer(outputs["visual"], outputs["text"])
            diagnostics.update(self.forward_regularizer.diagnostics)
        total = task + self.application_update.weighted(dependency) + self.cfg.get("forward_weight", .1) * forward
        diagnostics.update(self.application_update.diagnostics())
        diagnostics["total_loss"] = float(total.detach())
        if self.training:  # bank update after both the critic fit and the outer loss of this step
            for branch, regularizer in self.regularizers.items():
                if isinstance(regularizer, BankedDependence):
                    regularizer.enqueue(outputs[branch])
        self.diagnostics = diagnostics
        return total, task, dependency
