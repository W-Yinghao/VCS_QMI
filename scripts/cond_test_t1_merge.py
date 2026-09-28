"""P73 addendum 1 — merge the T1 conditional-test cells of one (encoder, family) from the monolithic job (power cells only) and the split
null / level jobs (null cells only), and write the merged JSON + markdown in the runner's format.

    python scripts/cond_test_t1_merge.py --power reports/P74_cond_test_t1_vcs4v800_colour.partial.json \
        --nulls 'reports/P74_cond_test_t1_split_vcs4v800_colour_null_*.json' --out reports/P74_cond_test_t1_vcs4v800_colour_merged

Rules (frozen in P73_ADDENDUM1_RUNTIME_SPLIT_FROZEN_20260928.md): planted (power) cells come from the monolithic file; every null cell comes from
the split files — null cells present in the monolithic file are dropped and listed under "dropped_monolithic_null_cells"; a null cell that appears
in two split files is an error (no silent overwrite); the seeds of every source file are recorded.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

TESTS = ("vcs_perm", "vcs_hoeff", "hsic_perm", "hsic_class", "hsic_deep", "c2st_logit", "js_perm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", required=True, help="monolithic JSON (final or .partial)")
    ap.add_argument("--nulls", required=True, help="glob of split-job JSONs (final, not .partial)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    power = json.load(open(a.power))
    merged = {"utc": utc_now(), "sources": {"power": {"file": a.power, "seed": power["settings"]["seed"], "utc": power["utc"]}, "nulls": []},
              "dropped_monolithic_null_cells": [], "families": power["families"], "tests": TESTS, "cases": {}}
    for cname, c in power["cases"].items():
        if c["mode"] == "planted":
            merged["cases"][cname] = c
        else:
            merged["dropped_monolithic_null_cells"].append({"case": cname, "n": sorted(c["by_n"], key=int)})
    files = sorted(glob.glob(a.nulls))
    if not files:
        sys.exit(f"no split files match {a.nulls}")
    for f in files:
        if f.endswith(".partial.json"):
            continue
        d = json.load(open(f))
        merged["sources"]["nulls"].append({"file": f, "seed": d["settings"]["seed"], "only": d["settings"].get("only"), "sizes": d["settings"]["sizes"], "utc": d["utc"]})
        for cname, c in d["cases"].items():
            assert c["mode"] != "planted", f"{f}: planted case {cname} in a null job"
            tgt = merged["cases"].setdefault(cname, {"family": c["family"], "mode": c["mode"], "strength": c["strength"], "by_n": {}})
            for n, r in c["by_n"].items():
                if n in tgt["by_n"]:
                    sys.exit(f"duplicate null cell {cname} n={n} in {f}")
                tgt["by_n"][n] = r
    atomic_write_json(Path(a.out + ".json"), merged)
    fams = ", ".join(f"{k} ({Path(v['dir']).name})" for k, v in merged["families"].items())
    L = [f"# T1 merged (P73 addendum 1) — {fams} — {merged['utc']}", "",
         f"Power cells: `{a.power}` (seed {merged['sources']['power']['seed']}); null cells: {len(merged['sources']['nulls'])} split files (seeds "
         f"{sorted(s['seed'] for s in merged['sources']['nulls'])}); monolithic null cells dropped: {merged['dropped_monolithic_null_cells'] or 'none'}.", "",
         "| case | mode | s | n | R | " + " | ".join(TESTS) + " | hoeff lin / mlp / closed | mean J (VCS) | mean JS |", "|---|---|---|---|---|" + "---|" * len(TESTS) + "---|---|---|"]
    for cname, c in merged["cases"].items():
        for n in sorted(c["by_n"], key=int):
            r = c["by_n"][n]; sm = r["summary"]
            L.append(f"| {cname} | {c['mode']} | {c['strength']:g} | {n} | {r['repeats']} | " + " | ".join(f"{sm[t]['power']:.2f}" for t in TESTS)
                     + f" | {sm['vcs_hoeff_lin']['power']:.2f} / {sm['vcs_hoeff_mlp']['power']:.2f} / {sm['vcs_hoeff_closed']['power']:.2f} | "
                     f"{sm['vcs_perm']['J_mean']:.4f} ± {sm['vcs_perm']['J_sd']:.4f} | {sm['js_perm']['JS_mean']:.4f} |")
    L += ["", "## Smallest n with power ≥ 0.8 (planted cases)", "", "| case | " + " | ".join(TESTS) + " |", "|---|" + "---|" * len(TESTS)]
    for cname, c in merged["cases"].items():
        if c["mode"] != "planted":
            continue
        row = []
        for t in TESTS:
            ns = [int(n) for n, r in c["by_n"].items() if r["summary"][t]["power"] >= 0.8]
            row.append(str(min(ns)) if ns else "—")
        L.append(f"| {cname} | " + " | ".join(row) + " |")
    L += ["", "## Null cells: rejection rate (level; nominal 0.05) with binomial 95 % half-width", "", "| case | n | R | " + " | ".join(TESTS) + " |", "|---|---|---|" + "---|" * len(TESTS)]
    for cname, c in merged["cases"].items():
        if c["mode"] == "planted":
            continue
        for n in sorted(c["by_n"], key=int):
            r = c["by_n"][n]; R = r["repeats"]
            L.append(f"| {cname} | {n} | {R} | " + " | ".join(f"{r['summary'][t]['power']:.3f} ± {1.96 * np.sqrt(max(r['summary'][t]['power'] * (1 - r['summary'][t]['power']), 0.05 * 0.95) / R):.3f}" for t in TESTS) + " |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print(f"-> {a.out}.md ({sum(len(c['by_n']) for c in merged['cases'].values())} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
