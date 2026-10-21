#!/usr/bin/env python3
"""V10_BASELINE_QUALIFICATION.csv (review package §3.2/§6 item 2) from existing records.

Statuses come from the two method ledgers, the RuLSIF FIT-gate records, the BF RuLSIF v2 release and the EPFL results;
the condition and scope text follows the package's classification. Usage: baseline_qualification.py OUT.csv
"""
import csv, json, sys
from pathlib import Path

HO = Path("/home/infres/yinwang/CS_QMI/handoff")
LEDGERS = {"pipeline_1047321": HO / "v10_method_pipeline_20261019/LEDGER.json",
           "native_recovery_1047364": HO / "v10_native_geometry_recovery_20261019/LEDGER.json"}
OWNER = json.load(open(HO / "V10_CURRENT_EXECUTION.json"))["authoritative_method_ownership"]
GATE = HO / "v10_rulsif_v2_20261019/gate"


def ledger_status(family):
    key = "native_recovery_1047364" if OWNER.get(family) == "1047364" else "pipeline_1047321"
    units = [u for u in json.load(open(LEDGERS[key]))["units"] if u.get("family") == family]
    by = {}
    for u in units:
        by.setdefault(u.get("phase"), []).append(u.get("status"))
    return "; ".join(f"{p}: {'/'.join(sorted(set(s)))} ({len(s)})" for p, s in sorted(by.items())), str(LEDGERS[key])


def fit_gate(name):
    d = json.load(open(GATE / name)); fam = list(d["families"].values())[0]
    per = [f"{c['candidate']['order']}:{c['status']}" for c in fam["candidates"]]
    return d.get("status"), fam["status"], " ".join(per), d.get("job")


def main():
    out = Path(sys.argv[1]); rows = []
    v1_rec, v1_fam, v1_per, v1_job = fit_gate("FIT_GATE_RuLSIF_exact_v1.json")
    v2_rec, v2_fam, v2_per, v2_job = fit_gate("FIT_GATE_RuLSIF_exact_v2.json")
    a1_rec, a1_fam, a1_per, a1_job = fit_gate("FIT_GATE_RuLSIF_exact_v2_amended.json")
    c5 = json.load(open(HO / "v10_review_actions_20261020/rulsif_v2/cand5_B_export.json"))["levels"]
    rel = json.load(open(HO / "v10_rulsif_v2_20261019/RELEASE.json"))
    ep = json.load(open(HO / "v10_epfl_production_20261019/analysis/EPFL_RESULTS.json"))["rulsif_v2"]
    st, src = ledger_status("RuLSIF_exact")
    rows.append({"baseline": "RuLSIF_exact (v1)", "dataset": "BF_SHG",
                 "math_numeric_gate": "method-pipeline GPU gate and full validation (ledger)",
                 "adaptation_health_gate": f"FIT functional gate on v1 code (job {v1_job}): all 9 candidates {v1_fam}, "
                                           f"F_true < 0 at every sample on both levels ({v1_per})",
                 "tuning_budget": "CAL 8 L40S GPU-h per family", "completion": st,
                 "original_failures": "truth-pose energies all negative: the adaptation failed",
                 "allowed_scope": "kept as the failed-adaptation row; not evidence that kernel methods are weak",
                 "label": "failed adaptation (retained)", "evidence": f"{src}; {GATE / 'FIT_GATE_RuLSIF_exact_v1.json'}"})
    v2 = HO / "v10_rulsif_v2_20261019/results/FINAL_DEV/RuLSIF_exact_v2"
    done = [f"{d.name}:{json.load(open(sorted(d.glob('ENV_*.json'))[-1]))['gpu_name']}" for d in sorted(v2.glob("seed_*"))
            if d.is_dir() and (d / "COMPLETE.json").exists()]
    st, src = f"CAL COMPLETE; finals COMPLETE {len(done)}/3 ({', '.join(done)})", str(HO / "v10_rulsif_v2_20261019")
    b, a = c5["B"]["overall"], c5["A"]["overall"]
    rows.append({"baseline": "RuLSIF_exact_v2 (A1)", "dataset": "BF_SHG",
                 "math_numeric_gate": "small-n algebra tests (tests/v10/test_rulsif_v2.py, hash bound in RELEASE source_files) and "
                                      f"GPU gate 1049287 PASS ({rel['acceptance']['GPU_GATE']['sha256'][:12]})",
                 "adaptation_health_gate": f"original pre-registered gate (job {v2_job}): every candidate must pass level A (same-core new "
                                           f"positions) and level B (4-fold leave-core); result {v2_fam} ({v2_per}). Candidate 5: A mean "
                                           f"{a['mean_F_true']:.8f}, {a['n_F_true_positive']}/320 positive, {a['n_true_beats_imposed']}/320 "
                                           f"beat imposed; B mean {b['mean_F_true']:.8f}, {b['n_F_true_positive']}/320, "
                                           f"{b['n_true_beats_imposed']}/320",
                 "tuning_budget": f"A1 amendment after the gate (job {a1_job}): B demoted to diagnostic, per-candidate admission by A, "
                                  f"orders 0-7 admitted, 8 excluded ({a1_per}); CAL 8 L40S GPU-h",
                 "completion": st,
                 "original_failures": "level B failed for all 9 candidates (positive fraction .625-.678); candidates 4 and 8 also below "
                                      ".75 on truth>imposed; candidate 8 failed level A",
                 "allowed_scope": "development-table row as a post-gate-amended, finite-capacity external kernel baseline; not a baseline "
                                  "that passed the original level-B health gate; algebra checks do not show cross-core generalisation",
                 "label": "A1 门控后修订、有限容量、已实施的外部核基线", "evidence": f"{src}; {HO / 'v10_review_actions_20261020/rulsif_v2'}"})
    for T, role in (("1024", "primary"), ("2048", "pre-registered capacity sensitivity")):
        e = ep[T]
        rows.append({"baseline": f"RuLSIF_exact_v2 (A1) T{T}", "dataset": "EPFL_RGB_NIR",
                     "math_numeric_gate": "inherits the BF v2 numeric checks; EPFL GPU gate rev3 and cross-hardware v2 gates",
                     "adaptation_health_gate": "inherits A1 labels; 1024 centres (10688 -> 1024; ~3.07 per FIT window) "
                                               "or 2048 for the sensitivity; all FIT windows and per-window V kept",
                     "tuning_budget": f"CAL {e['measured_cost_timing']['cal_family_hours_charged']:.2f} L40S GPU-h measured",
                     "completion": f"{e['status']}: seeds {e['finals'] and ','.join(map(str, e['summary']['seeds_complete']))}",
                     "original_failures": "none recorded beyond the inherited A1 classification",
                     "allowed_scope": f"{role}; T2048 never replaces T1024; capacity- and cost-constrained kernel baseline, "
                                      "not RuLSIF's best capability",
                     "label": "容量与成本共同受约束的核基线", "evidence": str(HO / f"v10_epfl_rulsif_v2_T{T}_20261020")})
    notes = {"CoMIR_native": ("single candidate, one checkpoint", "below the earlier host CoMIR; cause not found (records comparison pending)",
                              "native CoMIR row; selection freedom differs from learned families"),
             "MIND_2D_adapted": ("single configuration, one final", "-", "keep the name MIND_2D_adapted; single-config row"),
             "MI_native": ("single configuration, one final", "-", "single-config row"),
             "NMI_native": ("single configuration, one final", "-", "single-config row"),
             "InfoNCE_host[J_perm1]": ("3 candidates x 12 CAL checkpoints", "-",
                                       "trained with InfoNCE, registered with the common J(tanh f): not a native InfoNCE energy")}
    for fam, (budget, fail, scope) in notes.items():
        st, src = ledger_status(fam)
        rows.append({"baseline": fam, "dataset": "BF_SHG", "math_numeric_gate": "GPU gate and full validation (ledger 'gate'/'full')",
                     "adaptation_health_gate": "method gate as frozen in the supplement protocol (ledger)", "tuning_budget": budget,
                     "completion": st, "original_failures": fail, "allowed_scope": scope, "label": "", "evidence": src})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
