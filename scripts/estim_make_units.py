"""Write the P85 unit files (pilot + full grid, cost-balanced chunks) and the fresh-stream staircase probe used by the CPU smoke.

    python scripts/estim_make_units.py --out slurm/p85_units --chunks 8

P85 grid = vcs_estim.p85_grid.axes(): irrelevant dimensions (d_total 2/10/20/50/100, signal d = 2), independent sample size (N 256..16384),
batch (B 64/256/1024 at equal updates and at equal exposure), dependence staircase (gaussian / cubic / xor_mixture x 5 levels); 3 seeds.
Pilot = spec §5.1 trial: two dimensions x two N, VCS-N + kernel controls, seed 0 (memory / time measurement before the full submission).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim.p85_grid import axes, cost_weight, full, pilot, unit_line  # noqa: E402

STAIRCASE_EV = [("vcs", v) for v in ("single", "cyclic8", "inbatch")] + [("js", v) for v in ("single", "cyclic8", "inbatch")] + [(k, "inbatch") for k in ("infonce", "nwj", "dv", "smile")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="slurm/p85_units"); ap.add_argument("--chunks", type=int, default=8); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.txt").write_text("# P85 pilot (spec §5.1): setting d_signal d_total I N B updates seed methods\n" + "\n".join(unit_line(c) for c in pilot()) + "\n")
    cells = full(); w = [(unit_line(c), cost_weight(c)) for c in cells]; w.sort(key=lambda t: -t[1])
    bins = [[] for _ in range(a.chunks)]; load = [0.0] * a.chunks
    for line, c in w:
        i = load.index(min(load)); bins[i].append(line); load[i] += c
    for i, b in enumerate(bins):
        (out / f"full_{i:02d}.txt").write_text(f"# P85 full grid chunk {i} (relative cost {load[i]:.1f})\n" + "\n".join(b) + "\n")
    (out / "staircase_probe.txt").write_text("\n".join(f"gaussian {k} {v} 64 5e-4 0" for k, v in STAIRCASE_EV) + "\n")
    (out / "smoke.txt").write_text("gaussian 2 6 1.5 128 64 60 0 all\nxor_mixture 10 10 4 128 64 60 0 all\n")
    per_axis = {k: len(v) for k, v in axes().items()}
    print(f"pilot {len(pilot())} cells; full {len(cells)} distinct cells (per axis incl. shared cells: {per_axis}) in {a.chunks} files, "
          f"relative loads {[round(x, 1) for x in load]}; staircase probe {len(STAIRCASE_EV)} cells; smoke 2 cells")
