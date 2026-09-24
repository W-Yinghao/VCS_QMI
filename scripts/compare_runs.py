"""Compare two run directories (same config/seed): checkpoint parameter differences (params vs BN buffers) and logged losses."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).is_file() else []

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--report"); ap.add_argument("--title", default="run comparison")
    x = ap.parse_args(); A, B = Path(x.a), Path(x.b)
    ca = torch.load(A / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    cb = torch.load(B / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    lines = [f"# {x.title}", "", f"- A: `{A}` step {ca['optimizer_step']}", f"- B: `{B}` step {cb['optimizer_step']}", ""]
    for mod in ("encoder_state", "projector_state", "critic_state"):
        if ca.get(mod) is None: continue
        p_max = b_max = 0.0; p_rel = 0.0; n_exact = n_total = 0
        for k in ca[mod]:
            d = (ca[mod][k].float() - cb[mod][k].float()).abs().max().item(); n_total += 1; n_exact += (d == 0.0)
            if k.endswith(("running_mean", "running_var", "num_batches_tracked")): b_max = max(b_max, d)
            else:
                p_max = max(p_max, d); p_rel = max(p_rel, d / (ca[mod][k].float().abs().max().item() + 1e-12))
        lines.append(f"- {mod}: max|Δ| params {p_max:.3e} (max rel {p_rel:.3e}); max|Δ| BN buffers {b_max:.3e}; tensors bit-identical {n_exact}/{n_total}")
    la = {r["step"]: r for r in jl(A / "logs" / "steps.jsonl")}; lb = {r["step"]: r for r in jl(B / "logs" / "steps.jsonl")}
    common = sorted(set(la) & set(lb))
    lines.append("- logged steps: " + ", ".join(f"s{s}: Δloss {abs(la[s]['loss'] - lb[s]['loss']):.3e}" + (f" shiftA={la[s].get('shift')} shiftB={lb[s].get('shift')}" if la[s].get('shift') is not None else "") for s in common))
    same_shift = all(la[s].get("shift") == lb[s].get("shift") for s in common)
    lines.append(f"- identical pair shifts at logged steps: {same_shift}")
    lines.append(f"- init hashes equal: {ca['init_hashes'] == cb['init_hashes']}")
    out = "\n".join(lines) + "\n"; print(out)
    if x.report: Path(x.report).write_text(out)

if __name__ == "__main__": sys.exit(main())
