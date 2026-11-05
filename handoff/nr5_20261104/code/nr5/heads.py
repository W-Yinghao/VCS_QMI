"""NR5 scoring heads. G / G_ZERO = the unchanged NR1 DirectScorer (G_ZERO: fresh construction at a0,b0 with zero last residual layer,
seed 1001 under fork_rng). COS = s = exp(log_scale) * cos(u, v) + bias, the same expression/ops as the first part of
DirectScorer.pairs, so at zero residual COS and G_ZERO produce bitwise-identical pair scores."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from nr1.models import DirectScorer


class CosHead(nn.Module):
    architecture = 'COS'

    def __init__(self, a0: float, b0: float, **_):
        super().__init__()
        if not a0 > 0: raise ValueError('a0 must be positive')
        self.log_scale = nn.Parameter(torch.tensor(math.log(a0), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(b0, dtype=torch.float32))

    def pairs(self, x, y, **_):
        dt = self.log_scale.dtype
        x, y = F.normalize(x.to(dt), dim=-1), F.normalize(y.to(dt), dim=-1)
        s = (x * y).sum(-1, keepdim=True)
        return (self.log_scale.exp() * s + self.bias).squeeze(-1)


def make_head(model_config):
    mc = dict(model_config)
    if mc.get('readout') == 'COS':
        return CosHead(mc['a0'], mc['b0'])
    mc.pop('readout', None)
    return DirectScorer(**mc)
