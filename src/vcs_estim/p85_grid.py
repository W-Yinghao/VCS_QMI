"""P85 grid (package v2 spec §5.1 axes) shared by the unit generator, the runner and the aggregator.

Fixed choices (this version's implementation defaults, disclosed in the P85 pre-registration):
  I_PAD   generator MI (nats) of the two signal coordinates in the irrelevant-dimension axis (mid dependence; S is printed by the truth table);
  I_MID   the mid-dependence level of the sample-size and batch axes at d = 20 (the P81/P82 I = 4 condition, S ≈ 0.80);
  EXPOSURE the sample exposure of the equal-exposure batch variant (2000 updates x 256 = 512k positive pairs).
Roles per cell: FIT = N, TUNE = SELECT = max(64, N // 4), EVAL = 32768, GRAD = 20000 (channel derivative), TRUTH = 200000.
"""
from __future__ import annotations

I_PAD = 1.5
I_MID = 4.0
EXPOSURE = 512_000
SEEDS = (0, 1, 2)
DEP_LEVELS = (2.0, 4.0, 6.0, 8.0, 10.0)


def cell_name(c: dict) -> str:
    return f"P85_{c['setting']}_ds{c['d_signal']}_dt{c['d_total']}_I{c['I']:g}_N{c['N']}_B{c['B']}_U{c['updates']}_s{c['seed']}"


XOR_PAIRS = 10


def cell(setting, d_signal, d_total, I, N, B, updates, seed, methods="all") -> dict:
    if setting == "xor_mixture":            # the mixture setting fixes its own dimension (10 pairs); no padding on the dependence axis
        d_signal = d_total = XOR_PAIRS
    return {"setting": setting, "d_signal": d_signal, "d_total": d_total, "I": float(I), "N": int(N), "B": int(min(B, N)), "updates": int(updates),
            "seed": int(seed), "methods": methods}


def axes() -> dict[str, list[dict]]:
    """Axis name -> list of cells (a cell may belong to several axes; names coincide and the runner skips finished cells)."""
    A = {}
    A["pad"] = [cell("gaussian", 2, dt, I_PAD, 4096, 256, 2000, s) for dt in (2, 10, 20, 50, 100) for s in SEEDS]
    A["nsize"] = [cell("gaussian", 20, 20, I_MID, N, 256, 2000, s) for N in (256, 1024, 4096, 16384) for s in SEEDS]
    A["batch_equal_updates"] = [cell("gaussian", 20, 20, I_MID, 4096, B, 2000, s) for B in (64, 256, 1024) for s in SEEDS]
    A["batch_equal_exposure"] = [cell("gaussian", 20, 20, I_MID, 4096, B, EXPOSURE // B, s) for B in (64, 256, 1024) for s in SEEDS]
    A["dependence"] = [cell(st, 20, 20, I, 4096, 256, 2000, s) for st in ("gaussian", "cubic", "xor_mixture") for I in DEP_LEVELS for s in SEEDS]
    return A


def pilot() -> list[dict]:
    """Spec §5.1: first a small trial on two dimensions x two N with VCS-N and the kernel controls to measure memory and time."""
    return [cell("gaussian", 2, dt, I_PAD, N, 256, 2000, 0, methods="vcs_kernel") for dt in (2, 100) for N in (256, 16384)]


def full() -> list[dict]:
    seen, out = set(), []
    for cells in axes().values():
        for c in cells:
            n = cell_name(c)
            if n not in seen:
                seen.add(n); out.append(c)
    return out


def axis_value(axis: str, c: dict):
    return {"pad": c["d_total"], "nsize": c["N"], "batch_equal_updates": c["B"], "batch_equal_exposure": c["B"], "dependence": (c["setting"], c["I"])}[axis]


def unit_line(c: dict) -> str:
    return f"{c['setting']} {c['d_signal']} {c['d_total']} {c['I']:g} {c['N']} {c['B']} {c['updates']} {c['seed']} {c['methods']}"


def parse_unit_line(line: str) -> dict | None:
    f = line.split()
    if not f or f[0].startswith("#"):
        return None
    return cell(f[0], int(f[1]), int(f[2]), float(f[3]), int(f[4]), int(f[5]), int(f[6]), int(f[7]), f[8] if len(f) > 8 else "all")


def cost_weight(c: dict) -> float:
    """Rough relative wall-clock (neural in-batch training and O(N^2) kernels dominate)."""
    n = c["N"] / 4096; d = c["d_total"] / 20; b = c["B"] / 256; u = c["updates"] / 2000
    return 0.5 + u * (0.6 + 0.4 * b ** 1.5) + 0.8 * n ** 2 * (0.5 + 0.5 * d) + (0.3 if c["methods"] == "all" else 0.0)
