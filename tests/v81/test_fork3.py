"""V8.1 fork 3 — CPU tests of the intervention / audit / gradient-check code paths (no model, no data, no GPU).

    /projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I -m pytest -q -p no:cacheprovider tests/v81/test_fork3.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v81"))
import v81_fork3_fit_audit as FA  # noqa: E402
import v81_fork3_gpu as G  # noqa: E402

_spec = importlib.util.spec_from_file_location("v81core", REPO / "v8_1" / "package_v81_20261015" / "reference" / "core.py")
core = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(core)


def test_fit_audit_selection_rule():
    per = {f"p{i}": [f"p{i}_img{j}" for j in range(7 if i % 2 else 15)] for i in range(300)}
    sel = FA.select(per)
    assert len(sel) == 128 and sum(len(s[2]) for s in sel) <= 2048
    keys = [k for _, k, _ in sel]; assert keys == sorted(keys)                         # hash order
    assert keys == [FA.prompt_key(p) for p, _, _ in sel]
    for p, _, imgs in sel:
        assert imgs == sorted(per[p], key=str)                                            # whole prompts, lexical candidates
    small = FA.select(per, max_prompts=128, max_images=100)
    assert sum(len(s[2]) for s in small) <= 100 and all(len(s[2]) == len(per[s[0]]) for s in small)   # image cap keeps whole prompts
    assert FA.prompt_key("abc") == __import__("hashlib").sha256(b"v81-fit-audit-2718|abc").hexdigest()


def test_frozen_fit_audit_file_matches_rule():
    p = REPO / "reports" / "V8_1_DIAGNOSTICS" / "results" / "fit_audit_ids.json"
    if not p.exists():
        pytest.skip("id file not built")
    d = json.loads(p.read_text()); assert d["n_prompts"] <= 128 and d["n_images"] <= 2048
    hashes = [q["hash"] for q in d["prompts"]]; assert hashes == sorted(hashes) and all(FA.prompt_key(q["prompt_id"]) == q["hash"] for q in d["prompts"])
    ids = [i for q in d["prompts"] for i in q["image_ids"]]; assert ids == d["image_ids"] and len(set(ids)) == len(ids)
    assert d["image_ids_sha256"] == __import__("hashlib").sha256(json.dumps(ids).encode()).hexdigest()


def test_derangement_moves_every_image_to_another_prompt():
    prompt_of = [f"p{i // 7}" for i in range(7 * 40)] + ["p_single"]
    perm = G.derangement_across_prompts(prompt_of, seed=2718)
    assert sorted(perm) == list(range(len(prompt_of)))                                   # a permutation
    assert all(prompt_of[perm[i]] != prompt_of[i] for i in range(len(prompt_of)))
    assert perm == G.derangement_across_prompts(prompt_of, seed=2718)                    # fixed by the seed
    assert perm != G.derangement_across_prompts(prompt_of, seed=1)


def test_rotation_and_candidate_indices():
    prompt_of = ["a", "a", "a", "b", "c", "c"]; image_of = ["a_3", "a_1", "a_2", "b_1", "c_2", "c_1"]
    nxt = G.rotation_within_prompt(prompt_of, image_of)
    assert nxt == [1, 2, 0, 3, 5, 4]                                                      # a: a_1 -> a_2 -> a_3 -> a_1 (lexical, cyclic); b singleton -> itself; c: c_1(5) -> c_2(4) -> c_1
    assert G.candidate_indices(prompt_of, image_of) == [2, 0, 1, 0, 1, 0]


def test_schema_rows_validate_with_the_package_core():
    rng = np.random.default_rng(0); rows = []
    for k in range(12):
        logits = rng.normal(size=5); q = np.exp(logits) / np.exp(logits).sum(); u = G.tails_from_q(q)
        rec = {"prompt_id": f"p{k // 4}", "image_id": f"img{k:02d}", "votes": [int(v) for v in rng.integers(1, 6, size=3)], "mean": None}
        rec["mean"] = float(np.mean(rec["votes"]))
        pred = {"q": q.astype(np.float32).tolist(), "u": u.astype(np.float32).tolist(), "mu": float(np.float32(1 + u.sum()))}
        rows.append(G.schema_row(rec, pred, "FIT_AUDIT", "U-CS", 0, "best_e2", k % 4))
    out = core.validate_rows(rows, labelled=True); assert len(out) == 12
    assert all(abs(r["score"] - (1 + sum(G.tails_from_q(r["q"])))) < 1e-9 for r in out)
    with pytest.raises(ValueError):
        core.validate_rows([dict(rows[0], role="GENAI_RANK")], labelled=True)           # forbidden labelled role
    with pytest.raises(ValueError):
        core.validate_rows(rows + [rows[0]], labelled=True)                               # duplicate row


class _Data:
    def __init__(self, rows):
        self.rows = rows


def test_label_metrics_perfect_and_inverted():
    rng = np.random.default_rng(1); rows, pred = [], []
    for k in range(30):
        votes = [int(v) for v in rng.integers(1, 6, size=4)]; h = np.bincount(votes, minlength=6)[1:] / 4; t = np.flip(np.cumsum(np.flip(h)))[1:]
        rows.append({"image_id": f"i{k}", "prompt_id": f"p{k // 6}", "votes": votes, "hist": h.tolist(), "tail": t.tolist(), "mean": float(np.mean(votes)), "n": 4, "w": 1.0})
    for r in rows:
        u = np.array(r["tail"]); pred.append({"image_id": r["image_id"], "prompt_id": r["prompt_id"], "q": r["hist"], "u": u.tolist(), "mu": float(1 + u.sum())})
    m = G.label_metrics(pred, _Data(rows), [0.9, 0.7, 0.5, 0.2])
    assert m["mean_mse_prompt_macro"] < 1e-12 and abs(m["spearman_pooled"] - 1) < 1e-9 and m["n_prompts"] == 5
    inv = [dict(p, mu=6 - p["mu"]) for p in pred]
    assert abs(G.label_metrics(inv, _Data(rows), [0.9, 0.7, 0.5, 0.2])["spearman_pooled"] + 1) < 1e-9


def test_grad_stats_norms_cos_and_ratio():
    import torch
    lin = torch.nn.Linear(3, 2); head = torch.nn.Linear(2, 1)
    groups = {"head": list(head.named_parameters()), "qformer": list(lin.named_parameters()), "ln_vision": [], "query_tokens": []}
    g_point = [p.detach().clone() * 0 + 1.0 for grp in groups.values() for _, p in grp]
    g_rank = [g * 2.0 for g in g_point]
    st = G.grad_stats(groups, g_point, g_rank)
    assert abs(st["head"]["cos_point_rank"] - 1) < 1e-6 and abs(st["head"]["rank_over_point"] - 2) < 1e-6
    assert st["all_trainable"]["n_params"] == sum(p.numel() for p in list(lin.parameters()) + list(head.parameters()))
    assert abs(st["qformer"]["point_norm"] - (lin.weight.numel() + lin.bias.numel()) ** 0.5) < 1e-6 and not st["qformer"]["point_nonfinite"]
    g_rank[0] = -g_rank[0]                                                                # flip the head weight gradient -> negative cosine on the head
    st2 = G.grad_stats(groups, g_point, g_rank); assert st2["head"]["cos_point_rank"] < 0
    g_rank[1] = g_rank[1] * float("nan"); assert G.grad_stats(groups, g_point, g_rank)["head"]["rank_nonfinite"]
    assert G.grad_stats(groups, g_point, None)["head"].get("rank_norm") is None


def test_point_and_ranking_terms_separate_as_in_v8_core():
    """The ranking-term gradient taken as grad(total) - grad(point) equals the gradient of v8_core's ranking part alone."""
    import torch
    sys.path.insert(0, str(REPO / "scripts" / "v81" / "v8_src" / "v8")); import v8_core as C
    torch.manual_seed(0); logits = torch.randn(6, 5, requires_grad=True); hist = torch.softmax(torch.randn(6, 5), -1); w = torch.ones(6)
    prior = torch.tensor([0.9, 0.7, 0.5, 0.2]); sub = [0, 0, 0, 1, 1, 1]; nsub = [1] * 6
    lp, _, o = C.batch_loss(logits, hist, w, prior, "mse", sub, nsub); lt, parts, _ = C.batch_loss(logits, hist, w, prior, "rank", sub, nsub)
    (gp,) = torch.autograd.grad(lp, logits, retain_graph=True); (gr,) = torch.autograd.grad(lt - lp, logits, retain_graph=True)
    hbar = (hist * torch.arange(1, 6).float()).sum(-1); num = 0
    for idx in ([0, 1, 2], [3, 4, 5]):
        num = num + C.ranking_loss(o["mu"][idx], hbar[idx], "rank")
    (g_direct,) = torch.autograd.grad(num / 2, logits)
    assert torch.allclose(gr, g_direct, atol=1e-6) and parts["ranking_pairs"] == 6 and parts["ranking_subgroups"] == 2
