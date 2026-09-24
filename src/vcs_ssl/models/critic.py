"""The critic is the collaborator's reference ``PairCritic``; this module only checks the config matches it."""
from __future__ import annotations

from typing import Any

from reference.ssl_core import PairCritic


def build_critic(c: dict[str, Any], *, feature_dim: int) -> PairCritic:
    if not c["enabled"]:
        raise ValueError("build_critic called for a method without a critic")
    if c["input"] != "ordered_concat" or c["activation"] != "relu" or c["output"] != "tanh" or c["batchnorm"] or c["dropout"] != 0.0:
        raise ValueError("critic config is not the reference PairCritic (ordered concat, ReLU, tanh, no BN/dropout)")
    if list(c["hidden_dims"]) != [512, 512]:
        raise ValueError("reference PairCritic uses two hidden layers of equal width 512")
    if c["last_layer_xavier_gain"] != 0.1 or c["last_layer_bias"] != 0.0:
        raise ValueError("reference PairCritic last layer: xavier_uniform gain 0.1, bias 0")
    return PairCritic(feature_dim=feature_dim, hidden_dim=int(c["hidden_dims"][0]))
