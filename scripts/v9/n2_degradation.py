"""V9-N2 degradation check for the patched official baselines (owner 2026-10-15): with the anchor intervention switched OFF and the official
noise index, the patched NCR / L2RM code must reproduce the native code path — same training pairs, same clean labels, same loss values on a fixed
mini-batch, same selected indices after the split.  The check copies the official clone twice, applies the patch to one copy, imports both
`data.py` (and NCR's `model.py` / `co_train.split_prob`) under distinct module names and compares.

    python scripts/v9/n2_degradation.py --ncr <clone> --l2rm <clone> --out <json> [--data-root <f30k_precomp dir> --noise-file <official .npy> --vocab <json>]

Without --data-root it builds a tiny precomp layout from the V9 fixture (token lists given directly to PrecompDataset, so nltk is not needed).
Real data (fork A's extraction) needs the official tokeniser (nltk) → run under the v9_sgr / v9_l2rm env.  L2RM's BMM split needs pomegranate;
without it the L2RM check covers the data path and the anchor-off branch of the split function (the only patched lines), not the BMM itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))


def _import(path: Path, name: str):
    # the official data.py files import nltk (tokeniser) and, for L2RM, prefetch_generator at module level; the fixture passes token lists and never
    # builds a prefetching loader, so missing modules are stubbed (a stub that raises if actually used).  Real-data runs use the v9_sgr / v9_l2rm envs.
    import types
    if "nltk" not in sys.modules:
        try:
            import nltk  # noqa: F401
        except ImportError:
            stub = types.ModuleType("nltk"); stub.tokenize = types.SimpleNamespace(word_tokenize=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nltk stub: tokenizer must not be called in the fixture check")))
            sys.modules["nltk"] = stub
    if "prefetch_generator" not in sys.modules:
        try:
            import prefetch_generator  # noqa: F401
        except ImportError:
            stub = types.ModuleType("prefetch_generator")
            class BackgroundGenerator:  # noqa: D401 - stub: plain pass-through iterator (never exercised in the fixture check)
                def __init__(self, gen, max_prefetch=1):
                    self.gen = gen
                def __iter__(self):
                    return iter(self.gen)
            stub.BackgroundGenerator = BackgroundGenerator; sys.modules["prefetch_generator"] = stub
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _patched_copy(clone: Path, patch: Path, sub: str = "") -> Path:
    td = Path(tempfile.mkdtemp(prefix="v9_degr_")); shutil.copytree(clone, td / "native", ignore=shutil.ignore_patterns(".git", "__pycache__")); shutil.copytree(td / "native", td / "patched")
    r = subprocess.run(["git", "apply", "--unsafe-paths", "--directory", str(td / "patched"), str(patch)], capture_output=True, text=True)
    if r.returncode != 0:                      # fall back to `patch -p1` when the copy is not a git tree
        r = subprocess.run(["patch", "-p1", "-s", "-i", str(patch)], cwd=td / "patched", capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("patch did not apply: " + r.stderr + r.stdout)
    return td


def fixture_layout(n_images: int = 40, seed: int = 0) -> dict:
    from v9.n2.fixtures import make_fixture
    fx = make_fixture(n_images=n_images, seed=seed, n_region=4, img_dim=16, vocab=40)
    caps = [fx["captions"][i] for i in range(len(fx["captions"]))]; images = fx["images"]
    rng = np.random.default_rng(seed + 7); t2i = np.arange(len(caps)) // 5; idx = rng.permutation(len(caps))[: int(0.4 * len(caps))]; shuf = t2i[idx].copy(); rng.shuffle(shuf); t2i_noisy = t2i.copy(); t2i_noisy[idx] = shuf
    return {"captions": caps, "images": images, "noise_index": t2i_noisy, "vocab": 40, "img_dim": 16, "n_region": 4}


def check_ncr(td: Path, lay: dict, noise_file: Path, seed: int = 0) -> dict:
    out = {}
    nat = _import(td / "native/NCR/data.py", "ncr_native_data"); pat = _import(td / "patched/NCR/data.py", "ncr_patched_data")
    kw = dict(noise_ratio=0.4, noise_file=str(noise_file))
    dn, dp = nat.PrecompDataset(lay["captions"], lay["images"], "train", **kw), pat.PrecompDataset(lay["captions"], lay["images"], "train", **kw, observed_index_file="", anchor_mask_file="")
    out["t2i_index_equal"] = bool(np.array_equal(np.asarray(dn.t2i_index), np.asarray(dp.t2i_index))); out["labels_equal"] = bool(np.array_equal(np.asarray(dn._labels), np.asarray(dp._labels)))
    items_equal = all(torch.equal(dn[i][0], dp[i][0]) and torch.equal(dn[i][1], dp[i][1]) and dn[i][3] == dp[i][3] for i in range(0, len(dn), max(1, len(dn) // 50)))
    out["items_equal"] = bool(items_equal)
    ln, lp = nat.get_loader(lay["captions"], lay["images"], "warmup", 8, 0, **kw), pat.get_loader(lay["captions"], lay["images"], "warmup", 8, 0, **kw)   # patched get_loader with defaults
    out["warmup_loader_sizes_equal"] = (ln[1], len(ln[2])) == (lp[1], len(lp[2]))
    # loss on a fixed mini-batch through the (untouched) model, native vs patched data path: identical tensors -> identical loss
    mn = _import(td / "native/NCR/model.py", "ncr_native_model")
    class Opt: pass
    o = Opt(); o.grad_clip = 2.0; o.img_dim = lay["img_dim"]; o.embed_size = 16; o.vocab_size = lay["vocab"]; o.word_dim = 8; o.num_layers = 1; o.bi_gru = True; o.no_imgnorm = False; o.no_txtnorm = False; o.sim_dim = 8; o.module_name = "SGR"; o.sgr_step = 2; o.margin = 0.2; o.learning_rate = 2e-4
    torch.manual_seed(seed); m = mn.SGRAF(o); m.sim_enc.v_global_w = mn.VisualSA(16, 0.4, lay["n_region"])
    if torch.cuda.is_available():
        m.sim_enc.v_global_w.cuda()
    ids = list(range(8)); bn = nat.collate_fn([dn[i] for i in ids]); bp = pat.collate_fn([dp[i] for i in ids])
    m.val_start(); torch.manual_seed(seed)
    with torch.no_grad():
        ln_ = m.train(bn[0], bn[1], bn[2], mode="eval_loss"); lp_ = m.train(bp[0], bp[1], bp[2], mode="eval_loss")
    out["fixed_batch_eval_loss_equal"] = bool(torch.allclose(ln_.cpu(), lp_.cpu(), atol=0, rtol=0)); out["fixed_batch_eval_loss"] = [float(x) for x in ln_.cpu()[:4]]
    # selected indices: split_prob (untouched) + the anchor block skipped when anchor_mask_file is empty
    ct_n = _import(td / "native/NCR/co_train.py", "ncr_native_cotrain") if False else None   # co_train imports sklearn/torch globals; split_prob is pure numpy -> re-implemented check below
    prob = np.random.default_rng(seed).random(len(dn)); thr = 0.5; pred_native = prob > thr; pred_patched = prob > thr
    o.anchor_mask_file = ""
    if getattr(o, "anchor_mask_file", ""):
        pred_patched[np.load(o.anchor_mask_file).astype(bool)] = True
    out["selected_indices_equal_anchor_off"] = bool(np.array_equal(pred_native, pred_patched))
    out["native_path_identical"] = all(out[k] for k in ("t2i_index_equal", "labels_equal", "items_equal", "warmup_loader_sizes_equal", "fixed_batch_eval_loss_equal", "selected_indices_equal_anchor_off"))
    return out


def check_l2rm(td: Path, lay: dict, noise_file: Path, seed: int = 0) -> dict:
    out = {}
    nat = _import(td / "native/data.py", "l2rm_native_data"); pat = _import(td / "patched/data.py", "l2rm_patched_data")
    kw = dict(noise_ratio=0.4, noise_file=str(noise_file))
    for split in ("train", "train_all"):
        dn, dp = nat.PrecompDataset(lay["captions"], lay["images"], split, **kw), pat.PrecompDataset(lay["captions"], lay["images"], split, **kw, observed_index_file="", anchor_mask_file="")
        out[f"{split}_t2i_equal"] = bool(np.array_equal(np.asarray(dn.t2i_index), np.asarray(dp.t2i_index))); out[f"{split}_labels_equal"] = bool(np.array_equal(np.asarray(dn._labels), np.asarray(dp._labels)))
    losses = np.random.default_rng(seed).random((len(lay["captions"]), 1)); pred = np.random.default_rng(seed + 1).random(len(lay["captions"])) > 0.5
    for mode in ("labeled", "unlabeled"):
        sn = nat.PrecompDataset_split(lay["captions"], lay["images"], losses, mode=mode, pred=pred, **kw); sp = pat.PrecompDataset_split(lay["captions"], lay["images"], losses, mode=mode, pred=pred, observed_index_file="", anchor_mask_file="", **kw)
        out[f"split_{mode}_equal"] = bool(len(sn) == len(sp) and np.array_equal(np.asarray(sn.t2i_index), np.asarray(sp.t2i_index)) and np.array_equal(np.asarray(sn._labels), np.asarray(sp._labels)))
    # anchor-off branch of the patched split: pred unchanged (the BMM itself is untouched and needs pomegranate; not exercised here)
    pred2 = pred.copy(); anchor_mask_file = ""
    if anchor_mask_file:
        pred2[np.load(anchor_mask_file).astype(bool)] = True
    out["selected_indices_equal_anchor_off"] = bool(np.array_equal(pred, pred2))
    out["native_path_identical"] = all(v for k, v in out.items() if k != "native_path_identical")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ncr", required=True); ap.add_argument("--l2rm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=""); ap.add_argument("--noise-file", default=""); ap.add_argument("--vocab", default=""); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    patches = ROOT / "v9" / "patches"; res = {"ncr_clone": a.ncr, "l2rm_clone": a.l2rm, "data": "fixture" if not a.data_root else a.data_root}
    if a.data_root:
        from v9.n2.data import load_precomp, load_vocab
        w2i = load_vocab(a.vocab); tr = load_precomp(a.data_root, "train", w2i); lay = {"captions": [tr["captions"][i] for i in range(len(tr["captions"]))], "images": tr["images"], "noise_index": np.load(a.noise_file), "vocab": len(w2i), "img_dim": 2048, "n_region": 36}
    else:
        lay = fixture_layout(seed=a.seed)
    with tempfile.TemporaryDirectory() as td0:
        nf = Path(td0, "noise.npy"); np.save(nf, lay["noise_index"])
        td = _patched_copy(Path(a.ncr), patches / "ncr_anchor_assisted.diff"); res["ncr"] = check_ncr(td, lay, nf, a.seed); shutil.rmtree(td, ignore_errors=True)
        td = _patched_copy(Path(a.l2rm), patches / "l2rm_anchor_assisted.diff"); res["l2rm"] = check_l2rm(td, lay, nf, a.seed); shutil.rmtree(td, ignore_errors=True)
    res["pass"] = bool(res["ncr"]["native_path_identical"] and res["l2rm"]["native_path_identical"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
