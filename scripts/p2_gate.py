"""P2 smoke gate: decide whether the three 100-step real-data smoke runs (and the GPU resume check) allow P3.

Gates (hard): all three smoke runs COMPLETED; first-step gradient check PASS for every module; all logged losses/grads finite;
VCS J_raw within [-3, 1] and J == 1 - R; encoder parameters changed; peak memory recorded; in-training evaluation left training
RNG untouched.  The resume comparison on CUDA is *reported* (max |Δparam|, loss trajectory diff), not required to be bit-exact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def jl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def js(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    root = Path(a.output_root)
    lines = [f"# P2 smoke gate (job {a.job})", ""]
    ok = True

    def gate(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    runs = {}
    for m in ("vcs", "simclr", "vicreg"):
        rd = root / f"smoke_{m}_{a.job}"
        runs[m] = rd
        st = js(rd / "status.json") or {}
        summ = js(rd / "summary.json") or {}
        steps = jl(rd / "logs" / "steps.jsonl")
        lines += ["", f"## {m} — {rd.name}", ""]
        gate(f"{m} status COMPLETED", st.get("status") == "COMPLETED", f"status={st.get('status')} reason={st.get('failure_reason')}")
        g = js(rd / "first_step_gradients.json") or {}
        gate(f"{m} first-step gradients", g.get("check") == "PASS", json.dumps({k: v for k, v in g.items() if k.startswith('grad_norm')}))
        u = js(rd / "first_step_update.json") or {}
        gate(f"{m} encoder params changed after step 1", any((u.get("encoder_params_changed") or {}).values()))
        finite = all(math.isfinite(r["loss"]) for r in steps) and all(
            (r.get(k) is None or math.isfinite(r[k])) for r in steps for k in ("grad_norm_encoder", "grad_norm_projector", "grad_norm_critic"))
        gate(f"{m} all logged losses/gradients finite", bool(steps) and finite, f"{len(steps)} logged steps")
        gate(f"{m} peak memory recorded", summ.get("peak_allocated_mb") is not None,
             f"alloc {summ.get('peak_allocated_mb')} MB reserved {summ.get('peak_reserved_mb')} MB")
        if m == "vcs":
            jr = [r["J_raw"] for r in steps if r.get("J_raw") is not None]
            gate("vcs J_raw within [-3,1]", bool(jr) and all(-3.0 <= j <= 1.0 for j in jr), f"first {jr[:1]} last {jr[-1:]}")
            gate("vcs J == 1 - R", all(abs(r["J_raw"] - (1 - r["R_binary"])) < 1e-4 for r in steps if r.get("J_raw") is not None))
            gate("vcs shifts in [1,B-1]", all(1 <= r["shift"] <= 255 for r in steps if r.get("shift") is not None))
        evs = sorted((rd / "evaluations").glob("knn_epoch_*.json"))
        gate(f"{m} in-training eval ran and left training RNG untouched", bool(evs) and all(js(e)["training_rng_untouched"] for e in evs),
             ", ".join(f"{e.stem}: kNN {js(e)['knn_val_top1_pct']:.2f}% h_rank {js(e).get('h_effective_rank')}" for e in evs))
        if steps:
            r = steps[-1]
            lines.append(f"- last step: loss {r['loss']:.5f}, step {r['interval_mean_step_seconds'] * 1000:.0f} ms, data wait {r['data_wait_seconds'] * 1000:.0f} ms, "
                         f"lr {r['lr_enc_proj_matrix']:.2e}; steady images/s {summ.get('steady_state_images_per_s')}")
    # resume comparison
    lines += ["", "## GPU resume check (VCS, stop@50 -> resume to 100 vs continuous)", ""]
    cont, intr = runs["vcs"], root / f"smoke_vcs_interrupted_{a.job}"
    st_i = js(intr / "status.json") or {}
    gate("interrupted run resumed to COMPLETED", st_i.get("status") == "COMPLETED", f"status={st_i.get('status')}")
    try:
        ca = torch.load(cont / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
        cb = torch.load(intr / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
        diffs = {}
        for mod in ("encoder_state", "projector_state", "critic_state"):
            diffs[mod] = max(float((ca[mod][k].float() - cb[mod][k].float()).abs().max()) for k in ca[mod])
        la = {r["step"]: r["loss"] for r in jl(cont / "logs" / "steps.jsonl")}
        lb = {r["step"]: r["loss"] for r in jl(intr / "logs" / "steps.jsonl")}
        common = sorted(set(la) & set(lb) & set(range(50, 100)))
        ld = max(abs(la[s] - lb[s]) for s in common) if common else None
        lines.append(f"- max |Δparam| after 100 steps: {diffs}")
        lines.append(f"- max |Δloss| over resumed steps {common[:1]}..{common[-1:]}: {ld}")
        lines.append(f"- resume events: {jl(intr / 'logs' / 'resume_events.jsonl')}")
        exact = all(v == 0.0 for v in diffs.values()) and (ld == 0.0)
        lines.append(f"- bitwise identical on this GPU: {exact} (epoch-boundary resume level; CUDA nondeterminism may prevent exactness — reported, not gated)")
        gate("resume produced a full 100-step trajectory", bool(common) and common[-1] >= 99 - 50 + 50)
    except Exception as e:  # noqa: BLE001
        gate("resume comparison could be computed", False, repr(e))
    lines += ["", f"**GATE {'PASS' if ok else 'FAIL'}** — {'P3 may start' if ok else 'P3 is blocked; fix and re-run P2'}", ""]
    Path(a.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
