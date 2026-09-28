"""P85 aggregate: per-axis tables over seeds (selected rows + pre-declared fixed rows), dependence-level ordering probabilities from the
unit bootstrap, cost, QC, missing cells, and the spec §13.2 YAML blocks per method.

    python scripts/estim_p85_aggregate.py --inputs <dir with P85_*.json> --out <prefix>      -> <prefix>.md, <prefix>.json, <prefix>_yaml.md
Same-target columns (S): |J_eval − S| and posterior MSE for bounded posteriors (vcs, js, s_kde, s_kernel, rls_tanh); native columns: |value − own
truth| per estimator on its own scale (never mixed into the S columns); CS-K-native only against its Lebesgue truth where defined.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim.p85_grid import axes, axis_value, cell_name  # noqa: E402

FIXED_ROWS = ("cs_k:mult1", "s_kde:scott")          # reported next to the selected rows


def load(d: Path) -> dict:
    cells = {}
    for f in sorted(d.glob("P85_*.json")):
        try:
            R = json.loads(f.read_text()); cells[R["cell"]["name"]] = R
        except Exception as e:  # noqa: BLE001
            print("skip", f.name, e)
    return cells


def method_key(r: dict) -> str:
    if r["family"] == "neural":
        return f"neural:{r['kind']}:{r['negative_construction']}"
    return r["method"].split(":lr")[0] if r["kind"].startswith("s_kernel") else (r["method"] if r["kind"] in ("cs_k_native", "s_kde") else f"rls:{r['kind'].split('_')[1]}")


def rows_of(R: dict) -> dict:
    out = {}
    for r in R["rows"]:
        if r.get("selected") or r["method"] in FIXED_ROWS:
            out[method_key(r)] = r
    return out


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (float("nan"), float("nan"), 0)


def fmt(m, s, n):
    return "—" if n == 0 else f"{m:.4f} ± {s:.4f} (n={n})"


def mem_of(r):
    m = r.get("memory") or {}; return m.get("cuda_peak_alloc_mb", m.get("cpu_maxrss_mb"))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cells = load(Path(a.inputs)); A = axes(); out = {"n_cells_loaded": len(cells), "axes": {}, "missing": {}, "qc": []}
    L = [f"# P85 aggregate — {len(cells)} cells loaded from {a.inputs}", ""]
    for name, R in cells.items():
        t = R["truth"]; out["qc"].append({"cell": name, "oracle_J_within_3se": abs(t["J_oracle"] - t["S"]) <= 3 * (t["J_oracle_se"] + t["S_se"]),
                                          "nonfinite": sum((r.get("numerical") or {}).get("nonfinite_steps", 0) for r in R["rows"]), "wall_s": R.get("wall_seconds")})
    for axis, cl in A.items():
        groups = defaultdict(list); missing = []
        for c in cl:
            n = cell_name(c)
            if n in cells:
                groups[axis_value(axis, c)].append(cells[n])
            else:
                missing.append(n)
        out["missing"][axis] = missing
        if not groups:
            L += [f"## {axis}: no cells yet ({len(missing)} missing)", ""]; continue
        values = sorted(groups, key=lambda v: (str(v[0]), v[1]) if isinstance(v, tuple) else v)
        methods = sorted({k for g in groups.values() for R in g for k in rows_of(R)})
        tab = {}
        for mkey in methods:
            tab[mkey] = {}
            for v in values:
                rs = [rows_of(R).get(mkey) for R in groups[v]]; rs = [r for r in rs if r]
                if not rs:
                    continue
                trs = [R["truth"] for R in groups[v] if rows_of(R).get(mkey)]
                tab[mkey][str(v)] = {"native_abs_error": ms([r["native"]["abs_error"] for r in rs]), "native_signed_error": ms([r["native"]["signed_error"] for r in rs]),
                                     "native_value": ms([r["native"]["value"] for r in rs]),
                                     "J_eval_minus_S": ms([(r.get("posterior") or {}).get("J_eval_minus_S") for r in rs]),
                                     "posterior_mse": ms([(r.get("posterior") or {}).get("posterior_mse") for r in rs]),
                                     "excess_to_bayes_risk": ms([(r.get("posterior") or {}).get("excess_to_bayes_risk") for r in rs]),
                                     "fit_seconds": ms([r["fit_seconds"] for r in rs]), "eval_seconds": ms([r["evaluation_seconds"] for r in rs]), "peak_mem_mb": ms([mem_of(r) for r in rs]),
                                     "S_truth": ms([t["S"] for t in trs]), "own_truth": ms([r["native"]["own_truth"] for r in rs]), "estimand": rs[0]["estimand"],
                                     "boot_values": [bv for r in rs for bv in (r["native"].get("boot_values") or [])]}
        out["axes"][axis] = {"values": [str(v) for v in values], "table": tab}
        L += [f"## Axis `{axis}` — values {values}", "", "### Same-target error (S): posterior MSE E_M(T − η)² (mean ± sd over seeds)", "",
              "| method | " + " | ".join(str(v) for v in values) + " |", "|---|" + "---|" * len(values)]
        for mkey in methods:
            cells_m = [tab[mkey].get(str(v)) for v in values]
            if all(c is None or c["posterior_mse"][2] == 0 for c in cells_m):
                continue
            L.append(f"| {mkey} | " + " | ".join("—" if c is None else fmt(*c["posterior_mse"]) for c in cells_m) + " |")
        L += ["", "### Native error |value − own truth| on each estimator's own scale (not comparable across targets)", "",
              "| method | estimand | " + " | ".join(str(v) for v in values) + " |", "|---|---|" + "---|" * len(values)]
        for mkey in methods:
            cells_m = [tab[mkey].get(str(v)) for v in values]; est = next((c["estimand"] for c in cells_m if c), "")
            L.append(f"| {mkey} | {est} | " + " | ".join("—" if c is None else fmt(*c["native_abs_error"]) for c in cells_m) + " |")
        L += ["", "### Cost: fit seconds / peak memory MB (mean over seeds)", "", "| method | " + " | ".join(str(v) for v in values) + " |", "|---|" + "---|" * len(values)]
        for mkey in methods:
            cells_m = [tab[mkey].get(str(v)) for v in values]
            L.append(f"| {mkey} | " + " | ".join("—" if c is None else f"{c['fit_seconds'][0]:.1f} s / {c['peak_mem_mb'][0]:.0f} MB" for c in cells_m) + " |")
        if axis == "dependence":
            L += ["", "### Ordering probability P(V_{l+1} > V_l) between adjacent levels (unit bootstrap pooled over seeds; 0.5 = no resolution) and |Δmean| / pooled boot sd", ""]
            for st in ("gaussian", "cubic", "xor_mixture"):
                lv = [v for v in values if v[0] == st]
                if len(lv) < 2:
                    continue
                L += [f"**{st}**", "", "| method | " + " | ".join(f"{lv[i][1]:g}→{lv[i + 1][1]:g}" for i in range(len(lv) - 1)) + " |", "|---|" + "---|" * (len(lv) - 1)]
                for mkey in methods:
                    cells_m = [tab[mkey].get(str(v)) for v in lv]; parts = []
                    for i in range(len(lv) - 1):
                        c0, c1 = cells_m[i], cells_m[i + 1]
                        if not c0 or not c1 or not c0["boot_values"] or not c1["boot_values"]:
                            parts.append("—"); continue
                        b0, b1 = np.array(c0["boot_values"]), np.array(c1["boot_values"]); rng = np.random.default_rng(0)
                        p = float((rng.choice(b1, 4000) > rng.choice(b0, 4000)).mean()); res = abs(b1.mean() - b0.mean()) / np.sqrt(0.5 * (b0.var() + b1.var()) + 1e-300)
                        parts.append(f"{p:.2f} / {res:.1f}")
                    L.append(f"| {mkey} | " + " | ".join(parts) + " |")
                L.append("")
        L.append("")
    # cells outside the frozen axes (pilot / smoke): per-cell summary of the selected rows
    in_axes = {cell_name(c) for cl in A.values() for c in cl}; others = [R for n, R in cells.items() if n not in in_axes]
    if others:
        L += ["## Cells outside the axes (pilot / smoke)", ""]
        for R in others:
            c = R["cell"]; L += [f"### {c['name']} — device {c['device']}, wall {R.get('wall_seconds', float('nan')):.0f} s, S = {R['truth']['S']:.4f}, methods {c['methods']}", "",
                                 "| method | estimand | native value | own truth | abs err | J_eval − S | posterior MSE | fit s | eval s | peak MB |", "|---|---|---|---|---|---|---|---|---|---|"]
            for s in R["summary"]:
                pm = s["peak_mem"] or {}; m = pm.get("cuda_peak_alloc_mb", pm.get("cpu_maxrss_mb"))
                f = lambda v, d=4: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
                L.append(f"| {s['method']} | {s['estimand']} | {f(s['native_value'])} | {f(s['own_truth'])} | {f(s['native_abs_error'])} | "
                         f"{f(None if s['J_eval'] is None else s['J_eval'] - R['truth']['S'])} | {f(s['posterior_mse'])} | {f(s['fit_seconds'], 1)} | {f(s['evaluation_seconds'], 1)} | {f(m, 0)} |")
            cd = R.get("channel_derivative", {}); fd = (cd.get("oracle_fd") or [{}])[1] if len(cd.get("oracle_fd") or []) > 1 else {}
            L += ["", f"channel derivative ({cd.get('parameter')} = {cd.get('theta0')}): oracle FD {fd.get('dS_dtheta')} ± {fd.get('se')}; envelope {cd.get('oracle_envelope')}; "
                      f"learned: " + "; ".join(f"{k}: {v.get('dJ_dtheta')} (±{v.get('se')}, ratio {v.get('ratio_to_oracle')})" if 'dJ_dtheta' in v else f"{k}: {v.get('error')}" for k, v in (cd.get("learned") or {}).items()), ""]
    qc_bad = [q for q in out["qc"] if not q["oracle_J_within_3se"]]
    L += ["## QC", "", f"- oracle J on TRUTH within 3 se of S: {len(out['qc']) - len(qc_bad)} / {len(out['qc'])} cells" + (f"; failing: {[q['cell'] for q in qc_bad]}" if qc_bad else ""),
          f"- non-finite training steps (sum over rows): {sum(q['nonfinite'] for q in out['qc'])}", f"- missing cells per axis: { {k: len(v) for k, v in out['missing'].items()} }", ""]
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    # spec §13.2 YAML blocks: one per method (from a representative selected row of the first loaded cell containing it)
    Y = ["# P85 — per-method delivery blocks (spec §13.2), representative selected rows", ""]
    seen = set()
    for R in cells.values():
        for mkey, r in rows_of(R).items():
            if mkey in seen:
                continue
            seen.add(mkey); keys = ("experiment_family", "protocol_id", "source_commit", "source_basis", "estimator", "estimand", "evaluation_readout", "loss_scale", "reference_measure",
                                    "critic_class", "gradient_routing", "n_independent_units", "n_positive_pairs", "n_negative_pairs", "split_manifest_hash", "noise_target_kind", "noise_tau",
                                    "noise_sigma_coordinate", "fit_seconds", "evaluation_seconds", "status")
            Y += [f"## {mkey} (cell {R['cell']['name']})", "", "```yaml"] + [f"{k}: {json.dumps(r.get(k)) if not isinstance(r.get(k), str) else r.get(k)}" for k in keys] + ["```", ""]
    Path(a.out + "_yaml.md").write_text("\n".join(Y) + "\n"); print("->", a.out + ".md", f"({len(cells)} cells)"); return 0


if __name__ == "__main__":
    sys.exit(main())
