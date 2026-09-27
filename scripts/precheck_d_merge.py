"""Merge per-case pre-check D result JSONs (from split jobs) into one JSON + markdown per encoder.
    python scripts/precheck_d_merge.py --inputs a.json,b.json,... --out <prefix>"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from vcs_ssl.utils import atomic_write_json, utc_now

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    merged = None
    for f in a.inputs.split(","):
        d = json.load(open(f))
        if merged is None:
            merged = {"manifest": d["manifest"], "settings": d["settings"], "cases": {}, "sources": []}
        merged["cases"].update(d["cases"]); merged["sources"].append(f)
    merged["utc_merged"] = utc_now(); atomic_write_json(Path(a.out + ".json"), merged)
    man = merged["manifest"]; st = merged["settings"]
    order = sorted(merged["cases"].keys(), key=lambda c: (c.startswith("cond_"), merged["cases"][c].get("strength", 0), c))
    L = [f"# Pre-check D — independence-test power on planted nuisances ({man['run']}, {man['checkpoint']}) — merged {merged['utc_merged']}", "",
         f"Pool: {man['n_fit']} fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = {st['delta']}; permutations {st['perms']}; "
         f"critics fitted on 80 % of FIT and early-stopped/selected on 20 % (linear signed, MLP, closed-form); repeats {st['repeats']} (n ≥ 2000: {st['repeats_large']}; n ≥ 10000: 10).", "",
         "| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in order:
        cr = merged["cases"][case]
        for n, r in sorted(cr["by_n"].items(), key=lambda kv: int(kv[0])):
            s = r["summary"]; h = s["hsic_perm"]["power"] if s.get("hsic_perm") else float("nan")
            L.append(f"| {case} | {cr['strength']:g} | {n} | {r['repeats']} | {s['vcs_hoeff']['power']:.2f} | {s['vcs_hoeff_lin']['power']:.2f} | {s['vcs_hoeff_mlp']['power']:.2f} | {s['vcs_hoeff_closed']['power']:.2f} | {s['vcs_perm']['power']:.2f} | {h:.2f} | {s['c2st']['power']:.2f} | {s['vcs_hoeff']['J_eval_mean']:.4f} ± {s['vcs_hoeff']['J_eval_sd']:.4f} | {s['vcs_hoeff']['tau']:.3f} | {s['c2st']['acc_mean']:.3f} |")
            if "unconditional_power" in s:
                u = s["unconditional_power"]; L.append(f"| {case} (unconditional test) | | {n} | {r['repeats']} | {u['vcs_hoeff']:.2f} | {u['vcs_hoeff_lin']:.2f} | {u['vcs_hoeff_mlp']:.2f} | {u['vcs_hoeff_closed']:.2f} | {u['vcs_perm']:.2f} | {u['hsic_perm']:.2f} | {u['c2st']:.2f} | | | |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md", "cases:", len(order)); return 0

if __name__ == "__main__":
    sys.exit(main())
