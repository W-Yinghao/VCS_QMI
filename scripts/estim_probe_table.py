"""P82: results table of the estimator package probe (P81).  Reads reports/P82_estim_probe_I*_d20_s0.json, writes reports/P82_estim_probe.md."""
import glob
import json
import sys


def f(v, n=4):
    return "—" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main(out="reports/P82_estim_probe.md"):
    files = sorted(glob.glob("reports/P82_estim_probe_I*_d20_s0.json"), key=lambda p: float(p.split("_I")[1].split("_")[0]))
    L = ["# P82 — estimator package v1 probe: Gaussian d = 20, seed 0 (results only)", "",
         "J_eval / posterior MSE on EVAL (32 768 pairs per distribution); FIT columns = the same quantities on the FIT pairs; "
         "JS rows are evaluated with T = tanh f (common-posterior VCS regression evaluation), native JS in its own column.", ""]
    for p in files:
        R = json.load(open(p)); c = R["condition"]
        L += [f"## I = {c['I_generator']} nats (ρ = {c['rho']:.4f}): S = {c['S_truth']:.4f} ± {c['S_truth_se']:.4f}; oracle J on TRUTH {c['J_oracle_truth']:.4f} ± {c['J_oracle_truth_se']:.4f}; wall {R['wall_seconds']:.0f} s", "",
              "| model | params | sel. update | J_eval ± se | S − J_eval | posterior MSE | excess/Bayes | FIT J | FIT pmse | gate+ mean | gate− mean | native JS | fit s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in R["rows"]:
            sd = r["score_diagnostics"]
            L.append(f"| {r['run_id'].split('_', 1)[1]} | {r.get('trainable_parameters', '—')} | {r.get('selected_checkpoint', '—')} | {r['J_eval']:.4f} ± {r['J_se']:.4f} | "
                     f"{-r['signed_value_error']:.4f} | {r['posterior_mse']:.4f} | {f(r.get('excess_to_bayes_risk'), 3)} | {f(r.get('FIT_J'))} | {f(r.get('FIT_posterior_mse'))} | "
                     f"{sd['gate_pos']['mean']:.3f} | {sd['gate_neg']['mean']:.3f} | {f(r.get('JS_native_eval'))} | {f(r.get('fit_seconds'), 1)} |")
        L += ["", "| combination | J_eval ± se | posterior MSE | weights / λ | notes |", "|---|---|---|---|---|"]
        cb = R["combos"]
        for k in ("vcs_best_single_by_select", "vcs_C2_best_of_3_restarts"):
            L.append(f"| {k} | {cb[k]['J_eval']:.4f} | {cb[k]['posterior_mse']:.4f} | — | {cb[k].get('family') or cb[k].get('selected_run')} |")
        m = cb["vcs_mix"]; L.append(f"| vcs_mix (C0, C1, C2) | {m['J_eval']:.4f} ± {m['J_se']:.4f} | {m['posterior_mse']:.4f} | {[round(x, 3) for x in m['combination_weights']]} | KKT {m['kkt_gap']:.1e}; TUNE obj {m['tune_objective']:.4f} vs vertices {[round(x, 4) for x in m['tune_J_vertices']]} |")
        j = cb["js_mix"]; L.append(f"| js_mix (C0, C1, C2) | {j['J_eval']:.4f} ± {j['J_se']:.4f} | {j['posterior_mse']:.4f} | {[round(x, 3) for x in j['weights']]} | native JS {j['JS_native_eval']:.4f}; KKT {j['kkt_gap']:.1e} |")
        s = cb["vcs_residual"]; L.append(f"| vcs_residual ({s['critic_family']}) | {s['J_eval']:.4f} ± {s['J_se']:.4f} | {s['posterior_mse']:.4f} | λ = {s['lambda']:.3f} | A {s['A']:.4f}, B {s['B']:.4f}, pred. TUNE gain {s['predicted_tune_gain']:.4f}; U alone J {s['U_alone']['J']:.4f}; base J {s['base_J_eval']:.4f} |")
        g = R["gradients"]
        L += ["", "| dJ/dρ (critic fixed, through the generator) | value ± se |", "|---|---|"]
        for d in g["oracle_fd"]:
            L.append(f"| oracle dS/dρ, central difference δ = {d['delta']} | {d['dS_drho']:.4f} ± {d['se']:.4f} |")
        L.append(f"| oracle envelope | {g['oracle_envelope']['dJ_drho']:.4f} ± {g['oracle_envelope']['se']:.4f} |")
        for k, v in g.items():
            if k.startswith(("vcs_", "js_")):
                L.append(f"| {k} | {v['dJ_drho']:.4f} ± {v['se']:.4f} |")
        L.append("")
    open(out, "w").write("\n".join(L)); print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
