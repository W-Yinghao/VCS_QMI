"""Official CRD loss/memory with explicit FIT indexing and negative provenance.

The pinned upstream's unused alias sampler unconditionally calls cuda(). We
neutralize only that placement during construction, then ALWAYS supply indices.
Official embedding, score normalization, criterion and memory updates execute.

r22 (2026-10-22): optional ``objective: "vcs"`` keeps the official embedding
heads, memory buffers, momentum update and negative indices, and replaces only
the NCE criterion by the variational CS objective (PDF eqs. 33/36/40):
T = tanh(cos(e_s, e_t) / tau), P = same-image (student, teacher) embedding pairs
of the current batch, Q = anchor embedding x memory embeddings of the other
modality at uniformly drawn other FIT instances (product of marginals; P and Q
normalized separately), both CRD directions, loss = -(J_s + J_t).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .models import verify_upstream


class OfficialCRD(nn.Module):
    def __init__(self, student_dim, teacher_dim, fit_ids, fit_labels, cfg, seed):
        super().__init__()
        verify_upstream(cfg["source_root"], "repdistiller")
        folder = Path(cfg["source_root"]).resolve() / "crd"
        name = "_kd_official_crd"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, folder / "__init__.py", submodule_search_locations=[str(folder)])
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        criterion = importlib.import_module(name + ".criterion")
        memory = importlib.import_module(name + ".memory")
        original_cuda = memory.AliasMethod.cuda
        memory.AliasMethod.cuda = lambda unused: None
        try:
            self.loss = criterion.CRDLoss(SimpleNamespace(s_dim=student_dim, t_dim=teacher_dim,
                              feat_dim=cfg.get("feat_dim", 128), nce_k=cfg.get("negatives", 16384),
                              nce_t=cfg.get("temperature", .07), nce_m=cfg.get("momentum", .5), n_data=len(fit_ids)))
        finally:
            memory.AliasMethod.cuda = original_cuda
        self.k = cfg.get("negatives", 16384)
        self.mode = cfg.get("negative_mode", "class_excluding_original")
        if self.mode not in ("class_excluding_original", "uniform_other_instances"):
            raise ValueError("CRD negative mode must explicitly identify its Q")
        self.objective = cfg.get("objective", "nce")
        if self.objective not in ("nce", "vcs"):
            raise ValueError("CRD objective must be nce (official) or vcs")
        if self.objective == "vcs" and self.mode != "uniform_other_instances":
            raise ValueError("VCS-CRD needs Q = product of marginals: negative_mode uniform_other_instances")
        self.tau = float(cfg.get("temperature", .07))
        self.last_parts = {}
        self.seed, self.epoch = seed, 0
        ids = np.asarray(fit_ids, dtype=np.int64)
        labels = np.asarray(fit_labels, dtype=np.int64)
        self.lookup = {int(idx): i for i, idx in enumerate(ids)}
        self.labels = labels
        self.n_data = len(ids)
        self.class_negative = {int(y): np.where(labels != y)[0] for y in np.unique(labels)}
        if any(len(v) == 0 for v in self.class_negative.values()):
            raise ValueError("CRD requires at least two classes in FIT")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def forward(self, student_features, teacher_features, global_indices):
        global_indices = global_indices.detach().cpu().tolist()
        positives = np.asarray([self.lookup[int(i)] for i in global_indices], dtype=np.int64)
        contrast = np.empty((len(positives), self.k + 1), dtype=np.int64)
        contrast[:, 0] = positives
        all_ids = np.arange(self.n_data)
        for row, (global_idx, pos) in enumerate(zip(global_indices, positives)):
            pool = self.class_negative[int(self.labels[pos])] if self.mode == "class_excluding_original" else all_ids[all_ids != pos]
            g = np.random.default_rng(self.seed + self.epoch * 1000003 + int(global_idx) * 9176)
            contrast[row, 1:] = g.choice(pool, size=self.k, replace=self.k > len(pool))
        device = student_features.device
        if self.objective == "vcs":
            return self.vcs(student_features, teacher_features.detach(), torch.from_numpy(positives).to(device),
                            torch.from_numpy(contrast[:, 1:]).to(device))
        return self.loss(student_features, teacher_features.detach(),
                         torch.from_numpy(positives).to(device), torch.from_numpy(contrast).to(device))

    def vcs(self, student_features, teacher_features, positives, negatives):
        """-(J_s + J_t); J = E_P T - E_Q T - E_P T^2/2 - E_Q T^2/2 with T = tanh(cos / tau)."""
        crd = self.loss
        f_s, f_t = crd.embed_s(student_features), crd.embed_t(teacher_features)   # unit-norm embeddings
        memory = crd.contrast
        batch, dim = f_s.shape
        flat = negatives.reshape(-1)
        # Memory rows are buffers (no parameters); detach makes the sampled Q side constant.
        mem_t = torch.index_select(memory.memory_v2, 0, flat).detach().view(batch, -1, dim)
        mem_s = torch.index_select(memory.memory_v1, 0, flat).detach().view(batch, -1, dim)
        t_p = torch.tanh((f_s * f_t).sum(1) / self.tau)                                  # P: B pairs
        t_qs = torch.tanh(torch.bmm(mem_t, f_s.unsqueeze(2)).squeeze(2) / self.tau)      # student anchor: B*K pairs
        t_qt = torch.tanh(torch.bmm(mem_s, f_t.unsqueeze(2)).squeeze(2) / self.tau)      # teacher anchor: B*K pairs

        def objective(tp, tq):
            return tp.mean() - tq.mean() - .5 * tp.square().mean() - .5 * tq.square().mean()

        j_s, j_t = objective(t_p, t_qs), objective(t_p, t_qt)
        # Official ContrastMemory update (momentum, renormalize) at the positives' indices.
        with torch.no_grad():
            momentum = float(memory.params[4])
            for bank, value in ((memory.memory_v1, f_s), (memory.memory_v2, f_t)):
                row = torch.index_select(bank, 0, positives).mul_(momentum).add_(value.detach() * (1 - momentum))
                bank.index_copy_(0, positives, row.div(row.pow(2).sum(1, keepdim=True).pow(.5)))
        self.last_parts = {"vcs_j_s": float(j_s.detach()), "vcs_j_t": float(j_t.detach()),
                           "vcs_tp_mean": float(t_p.detach().mean()), "vcs_tqs_mean": float(t_qs.detach().mean()),
                           "vcs_tqt_mean": float(t_qt.detach().mean()),
                           "vcs_cos_p_mean": float((f_s * f_t).detach().sum(1).mean())}
        return -(j_s + j_t)
