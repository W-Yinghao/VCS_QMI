"""V9-N2 common teacher (package §05.2–05.3): the SGR host trained on the A-teacher-warmup pairs only, for the fixed number of full epochs,
with the host's warm-up loss convention (NCR / L2RM `mode='warmup'`: hinge with the MEAN over in-batch negatives, both directions), then frozen.
Its single pairing score s(image, caption) = the SGR similarity (sigmoid output) of that pair, computed in eval mode (BatchNorm uses running
statistics, so a pair's score does not depend on the batch it is scored in).  The same teacher, cost and anchor information are shared by every
method; the critic-fit / select / reference roles never influence teacher selection (there is no selection: the fixed epoch count is used).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .sgr_host import SGRHost, hinge_per_anchor

# Host hyper-parameters as read from the official code (SGRAF opts.py @50d0c6f, NCR run.py @60256ba, L2RM opt.py @a766c22); every field is
# overridden by reports/V9/HOST_LOCK.yaml once fork A has verified it against the official data.  Fixture runs change the sizes only.
HOST_DEFAULTS = {
    "img_dim": 2048, "word_dim": 300, "embed_size": 1024, "sim_dim": 256, "num_layers": 1, "bi_gru": True, "num_region": 36, "sgr_step": 3,
    "margin": 0.2, "learning_rate": 2e-4, "grad_clip": 2.0, "batch_size": 128, "lr_update": 30, "num_epochs": 40,
    "teacher_epochs": 5, "teacher_loss": "warmup_mean_hinge (NCR/L2RM mode='warmup')", "student_loss": "max_violation_hinge (SGRAF default, NCR/L2RM 'train')",
    "provenance": "SGRAF opts.py 50d0c6f / NCR run.py 60256ba / L2RM opt.py a766c22 (code reading; data-dependent fields TO_VERIFY in HOST_LOCK)",
}


@dataclass
class HostConfig:
    vocab_size: int
    img_dim: int = HOST_DEFAULTS["img_dim"]; word_dim: int = HOST_DEFAULTS["word_dim"]; embed_size: int = HOST_DEFAULTS["embed_size"]
    sim_dim: int = HOST_DEFAULTS["sim_dim"]; num_layers: int = 1; bi_gru: bool = True; num_region: int = HOST_DEFAULTS["num_region"]; sgr_step: int = 3
    margin: float = HOST_DEFAULTS["margin"]; learning_rate: float = HOST_DEFAULTS["learning_rate"]; grad_clip: float = HOST_DEFAULTS["grad_clip"]
    batch_size: int = HOST_DEFAULTS["batch_size"]; lr_update: int = HOST_DEFAULTS["lr_update"]; num_epochs: int = HOST_DEFAULTS["num_epochs"]
    teacher_epochs: int = HOST_DEFAULTS["teacher_epochs"]
    to_verify: list = field(default_factory=list)      # HOST_LOCK fields still marked TO_VERIFY: block a formal run, not a smoke

    @classmethod
    def from_host_lock(cls, vocab_size: int, lock: dict | None, **overrides):
        """Build from a HOST_LOCK mapping (fields with value 'TO_VERIFY' are listed in `to_verify`)."""
        kw, tv = {}, []
        for k in ("img_dim", "word_dim", "embed_size", "sim_dim", "num_layers", "bi_gru", "num_region", "sgr_step", "margin", "learning_rate",
                  "grad_clip", "batch_size", "lr_update", "num_epochs", "teacher_epochs"):
            v = (lock or {}).get(k, HOST_DEFAULTS.get(k))
            if isinstance(v, str) and v.upper().startswith("TO_VERIFY"):
                tv.append(k); v = HOST_DEFAULTS.get(k)
            if v is not None:
                kw[k] = v
        kw.update(overrides)
        return cls(vocab_size=vocab_size, to_verify=tv, **kw)


def build_host(cfg: HostConfig) -> SGRHost:
    return SGRHost(cfg.vocab_size, cfg.img_dim, cfg.word_dim, cfg.embed_size, cfg.sim_dim, cfg.num_layers, cfg.bi_gru, cfg.num_region, cfg.sgr_step)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iterate_batches(n: int, batch_size: int, rng: np.random.Generator, drop_singletons: bool = True):
    perm = rng.permutation(n)
    for s in range(0, n, batch_size):
        idx = perm[s:s + batch_size]
        if drop_singletons and len(idx) < 2:
            continue
        yield idx


def train_teacher(cfg: HostConfig, rows: list[dict], collate, out_dir: str | Path, seed: int, device: str = "cpu", log_every: int = 50) -> dict:
    """rows = the observed pairs of the teacher-warmup role (A is clean, so observed = original).  Writes teacher.pt + teacher_log.json."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed); host = build_host(cfg).to(device); opt = torch.optim.Adam(host.parameters(), lr=cfg.learning_rate)
    rng = np.random.default_rng([seed, 9_010]); log = {"epochs": [], "n_pairs": len(rows), "loss": "warmup_mean_hinge", "seed": seed, "cfg": asdict(cfg)}
    t0 = time.time(); step = 0
    for epoch in range(cfg.teacher_epochs):
        for g in opt.param_groups:
            g["lr"] = cfg.learning_rate * (0.1 ** (epoch // cfg.lr_update))
        host.train(); tot, nb = 0.0, 0
        for idx in iterate_batches(len(rows), cfg.batch_size, rng):
            ims, caps, lens = collate([rows[i] for i in idx]); ims, caps = ims.to(device), caps.to(device)
            sims = host(ims, caps, lens); er, ec = hinge_per_anchor(sims, cfg.margin, hard_negative=False)
            loss = er.sum() + ec.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(host.parameters(), cfg.grad_clip)
            opt.step(); tot += float(loss.detach()); nb += 1; step += 1
        log["epochs"].append({"epoch": epoch, "mean_batch_loss": tot / max(nb, 1), "batches": nb, "seconds": time.time() - t0})
    host.eval()
    torch.save({"model": host.state_dict(), "cfg": asdict(cfg), "seed": seed, "epochs": cfg.teacher_epochs}, out / "teacher.pt")
    (out / "teacher_log.json").write_text(json.dumps(log, indent=1))
    return log


def load_teacher(path: str | Path, device: str = "cpu") -> tuple[SGRHost, HostConfig]:
    ck = torch.load(path, map_location="cpu", weights_only=False); cfg = HostConfig(**ck["cfg"])
    host = build_host(cfg); host.load_state_dict(ck["model"]); host.to(device).eval()
    for p in host.parameters():
        p.requires_grad_(False)
    return host, cfg


@torch.no_grad()
def score_pairs(host: SGRHost, rows: list[dict], collate, batch_size: int = 128, device: str = "cpu") -> np.ndarray:
    """Pair score s_i = SGR similarity of (observed image, caption) of row i, eval mode, batch-independent.  Order = rows order."""
    host.eval(); out = np.zeros(len(rows), dtype=np.float64)
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]; ims, caps, lens = collate(chunk)
        sims = host(ims.to(device), caps.to(device), lens)
        out[s:s + len(chunk)] = sims.diag().cpu().numpy()
    return out
