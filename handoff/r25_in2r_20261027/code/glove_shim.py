"""Drop-in for torchtext.vocab.GloVe(name='840B', dim=300) as used by IN2R EncoderText.init_weights.

torchtext is not installable against torch 2.9 here. The cache was built by tools/glove_build.py with
torchtext's parsing rules and keeps every token that init_weights can look up, so `stoi` membership and
`vectors[stoi[w]]` are identical for those tokens. Shared by every arm.
"""
import os
import numpy as np
import torch

CACHE = os.environ.get('R25_GLOVE_NPZ', '/projects/EEG-foundation-model/yinghao/models/glove840b/glove840b_r25_subset.npz')
_LOADED = None


class GloVe:
    def __init__(self, name='840B', dim=300, cache=None, max_vectors=None):
        global _LOADED
        assert name == '840B' and dim == 300 and max_vectors is None
        if _LOADED is None:
            z = np.load(CACHE, allow_pickle=True)
            words = list(z['words'])
            _LOADED = ({w: i for i, w in enumerate(words)}, torch.from_numpy(z['vectors']))
        self.stoi, self.vectors = _LOADED
