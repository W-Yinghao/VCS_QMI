"""r7 R7-30 D1-20 CUDA check of the selector's biased attention (requested after smokes 1034990 / 1035047 / 1035048 failed in backward with
"LSE is not correctly aligned (strideH)": the memory-efficient SDPA kernel cannot back-propagate into a broadcast per-key bias, which the gate
phase needs).  The selector now computes attention explicitly.  Checks, forward AND backward, on the GPU it runs on:
(1) one transformer block, key length 577 (odd), per-key additive bias that REQUIRES GRAD: CUDA fp32 vs an explicit CPU fp64 reference with the
    same weights — outputs, input gradient, bias gradient, every parameter gradient;
(2) the full selector: estimator step (phase A) and gate step (phase B: gradient through log m into the gate head), question 9 + 576 = 585 keys:
    CUDA fp32 vs CPU fp64, same weights, deterministic soft mask (no Gumbel draw) — loss and gate-head gradients.
Exit 0 only if every comparison is within tolerance (relative 1e-3).
    python scripts/r7_d1_attn_check.py --out O
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_selectors as SEL  # noqa: E402


def rel(a, b):
    return float((a.double().cpu() - b.double().cpu()).abs().max() / (b.double().abs().max().cpu() + 1e-12))


def block_check(dev):
    torch.manual_seed(0); blk = SEL.Block(512, 8, 2048, 0.0).eval(); ref = copy.deepcopy(blk).double()
    x = torch.randn(4, 577, 512); bias = torch.log(torch.rand(4, 577) + 1e-3); bias[:, 500:] = -1e9
    outs = {}
    for name, mod, dtype, d in (("cuda", blk.to(dev), torch.float32, dev), ("cpu64", ref, torch.float64, "cpu")):
        xx = x.to(d, dtype).requires_grad_(True); bb = bias.to(d, dtype).requires_grad_(True); mod.zero_grad(set_to_none=True)
        y = mod(xx, bb[:, None, None, :]); (y.pow(2).mean()).backward()
        outs[name] = {"y": y.detach(), "gx": xx.grad, "gb": bb.grad, "gp": {n: p.grad for n, p in mod.named_parameters()}}
    a, b = outs["cuda"], outs["cpu64"]
    r = {"key_length": 577, "rel_out": rel(a["y"], b["y"]), "rel_grad_x": rel(a["gx"], b["gx"]), "rel_grad_bias": rel(a["gb"], b["gb"]),
         "rel_grad_params": max(rel(a["gp"][n], b["gp"][n]) for n in b["gp"])}
    r["match"] = all(v < 1e-3 for k, v in r.items() if k.startswith("rel_"))
    return r


def selector_check(dev):
    torch.manual_seed(1); cfg = SEL.Cfg(dropout=0.0); m = SEL.Selector(cfg, "vcs").eval(); ref = copy.deepcopy(m).double()
    B = 4; q = torch.randn(B, 9, 4096); pad = torch.zeros(B, 9, dtype=torch.bool); pad[0, 6:] = True; v = torch.randn(B, 576, 512)
    keep = torch.rand(B, 3, 576) < 0.3; C = torch.tensor([1, -1, 1, -1]); h = torch.rand(B, 3); d = torch.rand(B, 3)
    res = {}
    for name, mod, dtype, dd in (("cuda", m.to(dev), torch.float32, dev), ("cpu64", ref, torch.float64, "cpu")):
        batch = {"q_emb": q.to(dd, dtype), "q_pad": pad.to(dd), "v": v.to(dd, dtype), "keep": keep.to(dd), "C": C.to(dd), "h": h.to(dd, dtype), "d": d.to(dd, dtype)}
        mod.zero_grad(set_to_none=True); la = SEL.estimator_loss(mod, batch, torch.tensor([1.0, 1.0], dtype=dtype, device=dd)); la.backward()
        ga = {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}
        mod.zero_grad(set_to_none=True)
        tok, pb = mod.encode(batch["q_emb"], batch["q_pad"], batch["v"], torch.zeros(B, 576, dtype=dtype, device=dd))
        msoft = torch.sigmoid(mod.gates(tok) / 0.5)                                      # deterministic soft mask, gradient into the gate head
        lb = mod.readout(pb, mod.encode(batch["q_emb"], batch["q_pad"], batch["v"], torch.log(msoft + 1e-6))[1]).mean(); lb.backward()
        gb = {n: p.grad.clone() for n, p in mod.named_parameters() if n.startswith("gate.") and p.grad is not None}
        res[name] = {"la": la.detach(), "lb": lb.detach(), "ga": ga, "gb": gb}
    a, b = res["cuda"], res["cpu64"]
    r = {"keys": 585, "rel_loss_phaseA": rel(a["la"], b["la"]), "rel_loss_phaseB": rel(a["lb"], b["lb"]),
         "rel_grad_phaseA": max(rel(a["ga"][n], b["ga"][n]) for n in b["ga"]), "rel_grad_gate_phaseB": max(rel(a["gb"][n], b["gb"][n]) for n in b["gb"]),
         "gate_grad_nonzero": bool(any(float(g.abs().max()) > 0 for g in a["gb"].values()))}
    r["match"] = all(v < 1e-3 for k, v in r.items() if k.startswith("rel_")) and r["gate_grad_nonzero"]
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); dev = "cuda"
    rep = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    for name, fn in (("block_577_bias_requires_grad", block_check), ("selector_phaseA_phaseB", selector_check)):
        try:
            rep[name] = fn(dev)
        except Exception as e:
            rep[name] = {"match": False, "error": f"{type(e).__name__}: {e}"}
    rep["pass"] = all(rep[k]["match"] for k in ("block_577_bias_requires_grad", "selector_phaseA_phaseB"))
    Path(a.out).mkdir(parents=True, exist_ok=True); tag = rep["gpu"].replace(" ", "_")
    (Path(a.out) / f"attn_check_{tag}.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 6


if __name__ == "__main__":
    sys.exit(main())
