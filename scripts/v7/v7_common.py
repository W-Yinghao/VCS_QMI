"""V7 shared pieces: paths, the FGA (EvalMuse@45c8bb4) import, manifests, the archive image source, element pooling, the matched element
losses (M0 L1 / M1 balanced logistic / M2 VCS (1 - J) / 4), prior correction and the CAL metrics.  No model weights are loaded here."""
from __future__ import annotations

import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FGA_REPO = Path("/projects/EEG-foundation-model/yinghao/third_party/EvalMuse@45c8bb4")
MODELS = Path("/projects/EEG-foundation-model/yinghao/models")
BERT_DIR = MODELS / "bert-base-uncased"
TORCH_HOME = MODELS / "torch_home"
DATA_OUT = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data")                    # manifests, audit (small)
BIG_OUT = Path("/projects/EEG-foundation-model/yinghao/outputs_v7")                 # checkpoints, staged images, predictions
N_QUERY = 32
OBJECTIVES = ("l1", "logistic", "vcs")
ARMS = {"M0_semantic_l1": "l1", "M1_semantic_logistic": "logistic", "M2_semantic_vcs": "vcs"}
COEF = {"overall": 1.0, "element": 0.1, "mask": 0.1}                               # FGA active branch: overall L1 + 0.1 token + 0.1 mask


def import_lavis():
    """The read-only FGA tree on sys.path; offline caches; the local 'bert-base-uncased' resolved through the working directory (see stage_workdir)."""
    if str(FGA_REPO) not in sys.path:
        sys.path.insert(0, str(FGA_REPO))
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME)); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import lavis  # noqa: F401
    return lavis


def stage_workdir(path: Path) -> Path:
    """A working directory holding the 'bert-base-uncased' link that LAVIS / FGA load by that name (transformers treats it as a local dir)."""
    path.mkdir(parents=True, exist_ok=True); link = path / "bert-base-uncased"
    if not link.exists():
        link.symlink_to(BERT_DIR)
    os.chdir(path); return path


def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# --------------------------------------------------------------------------------------------------------------------------------- images
class ArchiveImages:
    """PIL images of EvalMuse TRAIN members read straight from the 6-part archive (one handle per process; DataLoader workers re-open)."""

    def __init__(self, root=None):
        self.root = root; self._z = None; self._pid = None

    def _zip(self):
        if self._z is None or self._pid != os.getpid():
            sys.path.insert(0, str(Path(__file__).resolve().parent)); import v7_data as VD
            self._z = VD.open_archive(self.root or VD.EVALMUSE); self._pid = os.getpid()
        return self._z

    def __call__(self, image_id: str):
        from PIL import Image
        return Image.open(io.BytesIO(self._zip().read("dataset/images/" + image_id))).convert("RGB")


# -------------------------------------------------------------------------------------------------------------------------------- tokens
def token_spans(offsets, spans):
    """Token positions (in the tokenized sequence; special tokens have offset (0, 0)) whose character range overlaps any element span."""
    idx = []
    for t, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if any(a < e and s < b for s, e in spans):
            idx.append(t)
    return idx


def pool_logits(z_tok, owner, tok_idx, tok_mask):
    """z_tok [B, L] token Bernoulli logits; owner [E] image row of each element; tok_idx [E, K] token positions, tok_mask [E, K] -> z_e [E]:
    the mean of the element's span-token logits (each element once, padding never counted)."""
    import torch
    g = z_tok[owner[:, None], tok_idx]; m = tok_mask.to(g.dtype)
    return (g * m).sum(1) / m.sum(1).clamp(min=1)


# -------------------------------------------------------------------------------------------------------------------------------- losses
def element_loss(z, y, w, prior: float, normalizer: float, objective: str):
    """The matched fractional-vote objective on Bernoulli logits z (T = tanh(z / 2)), fp32.  w = sampling (base) weights, never class balance;
    normalizer = the step's weight total, independent of which labels occur.  vcs = R_M / 4 = (1 - J) / 4; logistic = the balanced soft BCE with
    the same class law; l1 = |sigmoid(z) - y| (M0 task control, not a proper posterior loss)."""
    import torch
    import torch.nn.functional as F
    if not 0 < prior < 1 or normalizer <= 0:
        raise ValueError("invalid prior or normalizer")
    z = z.float(); y = y.float(); w = w.float()
    pos, neg = y / (2 * prior), (1 - y) / (2 * (1 - prior))
    if objective == "vcs":
        d = torch.sigmoid(z); terms = pos * (1 - d).square() + neg * d.square()
    elif objective == "logistic":
        terms = pos * F.softplus(-z) + neg * F.softplus(z)
    elif objective == "l1":
        terms = (torch.sigmoid(z) - y).abs()
    else:
        raise ValueError(objective)
    return (w * terms).sum() / normalizer


def natural_probability(z, prior: float):
    """p_pi = sigmoid(z + log(pi / (1 - pi))): the balanced posterior read back under the FIT prior (no external label involved)."""
    z = np.asarray(z, dtype=np.float64); s = z + math.log(prior / (1 - prior)); e = np.exp(-np.abs(s))
    return np.where(s >= 0, 1 / (1 + e), e / (1 + e))


def expected_brier(p, y):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    return y * (1 - p) ** 2 + (1 - y) * p ** 2


def soft_nll(p, y, eps=1e-7):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps); y = np.asarray(y, np.float64)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def hierarchical_mean(prompt_ids, image_ids, values):
    """Elements -> images -> prompts uniform (the V7 macro unit); returns (mean, n_prompts)."""
    g = defaultdict(lambda: defaultdict(list))
    for p, i, v in zip(prompt_ids, image_ids, values):
        g[str(p)][str(i)].append(float(v))
    per = [np.mean([np.mean(v) for v in imgs.values()]) for imgs in g.values()]
    return (float(np.mean(per)) if per else float("nan")), len(per)


def common_j(t, y, w=None):
    """Empirical class-conditional J on the balanced scale (evaluation-label masses only normalize P and Q; never recalibrate)."""
    t, y = np.asarray(t, np.float64), np.asarray(y, np.float64); w = np.ones_like(t) if w is None else np.asarray(w, np.float64)
    wp, wn = w * y, w * (1 - y)
    return float(np.dot(wp, t - 0.5 * t * t) / wp.sum() + np.dot(wn, -t - 0.5 * t * t) / wn.sum())


def avg_rank_spearman(a, b):
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
