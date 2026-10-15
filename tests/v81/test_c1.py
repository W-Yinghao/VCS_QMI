"""V8.1 C1 — CPU tests on a toy model with the real attribute layout (no BLIP-2 weights, no data, no GPU).

    /projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I -m pytest -q -p no:cacheprovider tests/v81/test_c1.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v81"))
import c1_train as C1  # noqa: E402


class Blk(torch.nn.Module):
    def __init__(self, cross):
        super().__init__(); self.dense = torch.nn.Linear(4, 4); self.drop = torch.nn.Dropout(0.1); self.has_cross_attention = cross

    def forward(self, x):
        return self.drop(torch.tanh(self.dense(x)))


class Toy(torch.nn.Module):
    """visual_encoder -> ln_vision -> (query_tokens + image mean) -> embeddings -> 4 encoder blocks -> v8_head; plus unused itm_head / Qformer.cls."""
    def __init__(self):
        super().__init__()
        self.visual_encoder = torch.nn.Linear(4, 4); self.ln_vision = torch.nn.LayerNorm(4); self.query_tokens = torch.nn.Parameter(torch.randn(1, 4))
        self.Qformer = torch.nn.Module(); self.Qformer.bert = torch.nn.Module(); self.Qformer.bert.embeddings = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.1))
        self.Qformer.bert.encoder = torch.nn.Module(); self.Qformer.bert.encoder.layer = torch.nn.ModuleList([Blk(k % 2 == 0) for k in range(4)])
        self.Qformer.cls = torch.nn.Linear(4, 3); self.itm_head = torch.nn.Linear(4, 2); self.v8_head = torch.nn.Linear(4, 5)
        for n, p in self.named_parameters():                                               # V8's rule as in v8_train.build
            p.requires_grad_(n.startswith(("Qformer.", "ln_vision.")) or n == "query_tokens" or n.startswith("v8_head."))

    def forward(self, x):
        h = self.ln_vision(self.visual_encoder(x)); q = self.Qformer.bert.embeddings(self.query_tokens + h)
        for blk in self.Qformer.bert.encoder.layer:
            q = blk(q)
        return self.v8_head(q)


def test_cmd_train_is_v8_source_plus_declared_substitutions():
    derived = C1.derive_c1_source(C1.v8_cmd_train_source())
    assert inspect.getsource(C1.cmd_train).rstrip("\n") == derived.rstrip("\n")
    assert len(C1.V8_TO_C1) == 5 and sum(new.count("[C1]") for _, new in C1.V8_TO_C1) == 4


def test_mask_resolves_last_two_blocks_by_enumeration():
    m = Toy(); inv = C1.apply_c1_mask(m)
    tr = {n for n, p in m.named_parameters() if p.requires_grad}
    assert tr == {"Qformer.bert.encoder.layer.2.dense.weight", "Qformer.bert.encoder.layer.2.dense.bias", "Qformer.bert.encoder.layer.3.dense.weight",
                  "Qformer.bert.encoder.layer.3.dense.bias", "ln_vision.weight", "ln_vision.bias", "query_tokens", "v8_head.weight", "v8_head.bias"}
    assert inv["c1_block_indices"] == [2, 3] and inv["c1_blocks_have_cross_attention"] == [True, False]
    ch = inv["checks"]; assert ch["block_names_match_indices"] and ch["c1_subset_of_v8"] and ch["vit_frozen"] and not ch["v8_total_matches_audit"]   # toy size != BLIP-2
    assert inv["n_trainable_c1"] == sum(p.numel() for n, p in m.named_parameters() if n in tr) and inv["newly_trainable_vs_v8"] == 0
    assert inv["by_group"]["qformer_frozen"]["trainable_c1"] == 0 and inv["n_trainable_v8"] > inv["n_trainable_c1"]


def test_modes_during_training():
    m = Toy(); C1.apply_c1_mask(m); m.train(); C1.set_train_c1(m); r = C1.mode_report(m)
    assert r["encoder_layers"] == [False, False, True, True] and not r["embeddings"] and not r["visual_encoder"] and not r["Qformer_root"]
    assert r["dropout_in_c1_blocks_training"] == [True] and r["dropout_elsewhere_training"] == [False] and r["ln_vision"] and r["v8_head"]


def test_one_step_changes_only_trainable_tensors():
    torch.manual_seed(0); m = Toy(); C1.apply_c1_mask(m); C1.set_train_c1(m)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}; opt = C1.optimizer(m, 1e-2)
    m(torch.randn(8, 4)).pow(2).mean().backward()
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    opt.step()
    for n, p in m.named_parameters():
        assert torch.equal(before[n], p.detach()) != p.requires_grad, n                    # frozen bitwise unchanged, trainable changed


def test_epoch_artifact_holds_only_trainable(tmp_path):
    m = Toy(); C1.apply_c1_mask(m); C1.save_epoch_trainable(m, tmp_path, 3)
    ck = torch.load(tmp_path / "epochs" / "epoch3.pt", weights_only=False)
    assert ck["epoch"] == 3 and set(ck["trainable"]) == {n for n, p in m.named_parameters() if p.requires_grad}


def test_parent_lr_from_the_formal_units():
    if not (C1.V8_UNITS / "V8-FIT-U-CS-s0" / "log.json").exists():
        pytest.skip("V8 units not mounted")
    assert {(a, s): C1.parent_lr(a, s) for a in ("U-CS", "U-RPS", "U-MSE") for s in (0, 1, 2)} == {
        **{("U-CS", s): 1e-05 for s in (0, 1, 2)}, **{("U-RPS", s): 5e-06 for s in (0, 1, 2)}, **{("U-MSE", s): 5e-06 for s in (0, 1, 2)}}


def test_export_refuses_non_development_roles():
    with pytest.raises(SystemExit):
        C1.main(["export", "--unit-dir", "/nonexistent", "--roles", "CAL_SELECT", "GENAI_RANK"])
