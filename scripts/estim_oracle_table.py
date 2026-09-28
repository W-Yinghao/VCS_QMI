"""Re-derive the owner's oracle-resolution table (CS_QMI/RESULTS_20260928.md, single shuffled negative) and write it as markdown + JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim.oracle import oracle_table  # noqa: E402

OWNER = {0.9: (1.62, 0.97, 2.08, 0.352), 0.99: (3.01, 0.99, 3.35, 0.692), 0.9999: (10.78, 1.04, 9.28, 0.953), 0.999999: (37.73, 1.06, 26.51, 0.994)}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "reports/P70_estim_oracle_table"
    rows = oracle_table()
    L = ["# Oracle resolution δρ = SD / |dE/dρ| (bivariate Gaussian, n = 10 000, single shuffled negative) — re-derivation vs the owner's table", "",
         "| ρ | I (nats) | S (ours / owner) | NWJ/VCS (ours / owner) | JS/VCS (ours / owner) | VCS/CRB (ours / owner) |", "|---|---|---|---|---|---|"]
    for r in rows:
        o = OWNER[r["rho"]]
        L.append(f"| {r['rho']} | {r['I_nats']:.2f} | {r['S']:.3f} / {o[3]} | {r['NWJ/VCS']:.2f} / {o[0]} | {r['JS/VCS']:.3f} / {o[1]} | {r['VCS/CRB']:.2f} / {o[2]} |")
    L += ["", "Method: diagonalised PMI (P: c + ρ(u² − v²)/2; Q: c + ρu²/(2(1+ρ)) − ρv²/(2(1−ρ))), 4 M common draws, central-difference derivative; NWJ product-side moments analytic."]
    Path(out + ".md").write_text("\n".join(L) + "\n"); Path(out + ".json").write_text(json.dumps(rows, indent=1)); print("\n".join(L))
