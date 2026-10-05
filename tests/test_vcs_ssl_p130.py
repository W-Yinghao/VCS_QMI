"""P130 (v6 §8.3): CIFAR-stem ResNet-50 backbone.  Existing ResNet-18 configs keep byte-identical config hashes and init weights; the config policy
accepts only (resnet18_cifar, 512) and (resnet50_cifar, 2048); the three P130 configs differ from their ResNet-18 parents only in backbone, h_dim
and stage; the ResNet-50 encoder maps [N, 3, 32, 32] to h [N, 2048]."""
import copy
import json
import os

import pytest
import torch
import yaml

os.environ.setdefault("OUTPUT_ROOT", "/tmp/p130_test_out")
os.environ.setdefault("DATA_ROOT", "/home/infres/yinwang/CS_QMI/data/cifar10")
os.environ.setdefault("MANIFEST_ROOT", "/home/infres/yinwang/CS_QMI/manifests")
from vcs_ssl.config import ConfigError, load_config, policy_checks  # noqa: E402
from vcs_ssl.models import build_models  # noqa: E402
from vcs_ssl.models.backbone import build_backbone  # noqa: E402

REF = {  # recorded before the P130 edit (scratchpad ref_hashes.json, 2026-10-04)
    "cifar100_hpY_AP3_c100_views4_800ep_seed0": ("bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767",
                                                 "e54b48428f7add7fb0cff4c70e90e7c8a7cf5b6eba839854efdadf40726f039b",
                                                 "b0ecf86f425782e9a6a47f4aceba688c649a701b89709eb33f800e98de0cec20"),
    "cifar100_hpS_simclr_views4_800ep_seed0": ("bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767",
                                               "e54b48428f7add7fb0cff4c70e90e7c8a7cf5b6eba839854efdadf40726f039b",
                                               "07da1fcaa37c000d8dad93a649add4889a64e55195826b923d8e235a57e3b603"),
    "cifar10_hpY_AP3_views4_800ep_seed0": ("bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767",
                                           "e54b48428f7add7fb0cff4c70e90e7c8a7cf5b6eba839854efdadf40726f039b",
                                           "2f3e24e0c5d7386fa68c223ad8991752e16776378bbe66882554e0a0a5c60140"),
}


@pytest.mark.parametrize("name", sorted(REF))
def test_resnet18_configs_unchanged(name):
    cfg = load_config(f"configs/{name}.yaml")
    b = build_models(cfg, seed=0)
    enc, proj, ch = REF[name]
    assert b["init_hashes"]["encoder_init_sha256"] == enc and b["init_hashes"]["projector_init_sha256"] == proj
    assert cfg["_meta"]["config_hash"] == ch


def test_resnet50_shapes():
    stem = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False, "maxpool": False}
    net = build_backbone("resnet50_cifar", stem).eval()
    with torch.no_grad():
        h = net(torch.randn(2, 3, 32, 32))
    assert tuple(h.shape) == (2, 2048)
    with pytest.raises(ValueError):
        build_backbone("resnet34_cifar", stem)


def test_policy_pairs():
    cfg = load_config("configs/cifar100_hpR50_AP3_views4_800ep_seed0.yaml")
    for bb, hd in (("resnet50_cifar", 512), ("resnet18_cifar", 2048)):
        bad = copy.deepcopy(cfg); bad["model"]["backbone"], bad["model"]["h_dim"] = bb, hd
        bad.pop("_meta", None)
        with pytest.raises(ConfigError):
            policy_checks(bad)


def test_p130_configs_minimal_diff():
    import sys; sys.path.insert(0, "configs")
    from make_p104_configs import flat
    m = json.load(open("configs/P130_SHA256.json"))["configs"]
    assert len(m) == 3
    for n, r in m.items():
        fa, fb = flat(yaml.safe_load(open("configs/" + r["parent"]))), flat(yaml.safe_load(open("configs/" + n)))
        diff = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
        assert diff == {"model.backbone", "model.h_dim", "run.stage"}, diff
        cfg = load_config("configs/" + n)
        assert cfg["model"]["backbone"] == "resnet50_cifar" and cfg["model"]["h_dim"] == 2048
