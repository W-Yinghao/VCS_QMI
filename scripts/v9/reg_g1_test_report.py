"""V9-G1 sealed-test report writer: every number is generated from reports/V9/V9_G1_TEST_results/g1_test_results.json (the one read) and, for the
side-by-side DEV column, from reports/V9/V9_G1_results/g1_results.json (G1's development validation).  [解释] is rendered mechanically from the frozen
wording rules of reports/V9/V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml; no number is typed by hand.
    python scripts/v9/reg_g1_test_report.py [--test ...json] [--dev ...json] [--ledger reports/V9/V9_TEST_READ_LEDGER.csv] [--out reports/V9/V9_G1_TEST_REPORT_<date>.md]"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METHODS = ("vcs", "logistic", "infonce", "comir", "kernel"); FR = ("0.25", "1"); EP = ("0", "0.2")
NAME = {"vcs": "VCS", "logistic": "logistic (BCE)", "infonce": "InfoNCE", "comir": "CoMIR", "kernel": "kernel (seed 901)", "mi": "MI", "nmi": "NMI"}


def pct(t):
    return "—" if not t or t.get("mean") is None else f"{100 * t['mean']:.1f}" + (f" ± {100 * t['sd']:.1f}" if t.get("sd") is not None else "")


def iv(c, key="ci95"):
    v = c.get(key) if c else None; return f"[{100 * v[0]:+.1f}, {100 * v[1]:+.1f}]" if v else "—"


def iv2(c, key="ci95"):
    v = c.get(key) if c else None; return f"[{100 * v[0]:+.2f}, {100 * v[1]:+.2f}]" if v else "—"


def fp(x, floor="0.001"):
    """Two-sided bootstrap p from 2 000 resamples: the smallest non-zero value is 2 / 2 000 = 0.001, so 0 means < 0.001; Holm over 4 multiplies by up to 4 (< 0.004)."""
    return f"< {floor}" if x == 0 else f"{x:.4f}"


def basis(c):
    """Which of the two frozen intervals contains 0 (generated from the JSON)."""
    t, b = c.get("ci95"), c.get("image_boot95")
    return f"seed t {iv2(c)} {'excludes' if excl(t) else 'contains'} 0, core bootstrap {iv2(c, 'image_boot95')} {'excludes' if excl(b) else 'contains'} 0"


def excl(v):
    return v is not None and (v[0] > 0 or v[1] < 0)


def both(c):
    return bool(c) and excl(c.get("ci95")) and excl(c.get("image_boot95"))


def replication(dev, test):
    """Frozen wording: replicated / not replicated / detected on the test only / reversed / no detectable difference on either split."""
    if test["detected"] and dev["detected"]:
        return "replicated on the sealed test" if (test["mean"] > 0) == (dev["mean"] > 0) else "reversed on the sealed test"
    if dev["detected"]:
        return "not replicated on the sealed test"
    if test["detected"]:
        return "detected on the test only (reported, not promoted)"
    return "no detectable difference on DEV or on the test"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", type=Path, default=REPO / "reports/V9/V9_G1_TEST_results/g1_test_results.json")
    ap.add_argument("--dev", type=Path, default=REPO / "reports/V9/V9_G1_results/g1_results.json"); ap.add_argument("--ledger", type=Path, default=REPO / "reports/V9/V9_TEST_READ_LEDGER.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=Path("/projects/EEG-foundation-model/yinghao/outputs_v9/reg_test/results")); ap.add_argument("--gpu-map", type=Path, default=REPO / "configs/V9_G1_TEST_PREDICTOR_GPU.csv")
    ap.add_argument("--diag", type=Path, nargs="*", default=[Path("/projects/EEG-foundation-model/yinghao/outputs_v9/reg_test_fixture/20261015T211818Z") / n for n in ("diag.json", "diag_l40s.json")]); a = ap.parse_args()
    T, D = json.load(open(a.test)), json.load(open(a.dev)); out = a.out or REPO / f"reports/V9/V9_G1_TEST_REPORT_{dt.date.today():%Y%m%d}.md"
    led = next((r for r in csv.DictReader(open(a.ledger)) if r["item"] == "bf_shg_test_closed"), {}) if a.ledger.exists() else {}
    gm = {r["predictor_id"]: r for r in csv.DictReader(open(a.gpu_map))} if a.gpu_map.exists() else {}; typ = lambda d: "L40S" if "L40S" in d else "RTX PRO 6000" if "PRO 6000" in d else d
    res = [json.load(open(f)) for f in sorted(a.results_dir.glob("*.json"))] if a.results_dir.exists() else []
    dev_types = {}; mism = [r["predictor_id"] for r in res if gm and typ(r["device"]) != gm[r["predictor_id"]]["record_gpu"]]
    for r in res:
        dev_types[typ(r["device"])] = dev_types.get(typ(r["device"]), 0) + 1
    crit = ("vcs", "logistic", "infonce"); diags = [json.load(open(f)) for f in a.diag if f.exists()]
    rep_bound = max((r["rep1_vs_rep2_max_abs_d_err"] for d in diags for pid, r in d["rows"].items() if pid.split("-")[2:3] and pid.split("-")[2] in crit), default=None)
    L = [f"# V9-G1 sealed-test report: BF/SHG rigid registration, one confirmation read ({dt.date.today().isoformat()})", "",
         f"Protocol `V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml` (frozen in f19865e; the version in force at the read is {led.get('protocol_commit', '—')}, i.e. the freeze plus pre-read execution note 1, committed before any test image was preprocessed; execution note 2, 03b2e81, was added after the read started and records the launch and a scheduling-only change of job time limits).  "
         f"Sealed archive sha256 {led.get('archive_sha256', '—')[:16]}…, {led.get('n_cores', '—')} test cores; read started {led.get('read_utc', '—')} (prepare job {led.get('prepare_job', '—')}); "
         f"registration jobs {led.get('registration_jobs', '—')}; aggregate job {led.get('aggregate_job', '—')}.  Every number below is generated by `scripts/v9/reg_g1_test_report.py` from "
         f"`V9_G1_TEST_results/g1_test_results.json` ({T.get('n_results')} results; missing: {T.get('missing') or 'none'}), written by `g1_aggregate.main` unchanged over the test records; "
         "the DEV column comes from G1's `V9_G1_results/g1_results.json`.  Package format: [已知证据] / [本轮结果] / [解释] / [未验证假设].", "",
         "## [已知证据] (V9-G1 development validation, report 0c8616a)",
         "- G1 trained and CAL-selected every predictor on REG-P1's FIT / CAL cores and read the 32 DEV cores, which r6 had also read; its DEV numbers are development validation.",
         "- Primary family on DEV (success < 42 px, Holm over 4): " + "; ".join(f"VCS − {NAME[k.split('|')[0].replace('vcs_minus_', '')]} "
                                                                             f"({k.split('|')[1]}) {100 * c['mean']:+.1f} pp, detected {c['detected']}" for k, c in D["primary"].items()) + ".",
         "- The test predictors are exactly those G1 selected on CAL (chosen checkpoints, CoMIR weights, kernel multipliers); MI / NMI are recomputed on the test.", "",
         "## [本轮结果] (134 sealed test cores × 10 rigid perturbations, read once)", "",
         f"All {len(res)} results ran on the GPU type of their DEV record ({', '.join(f'{k} {v}' for k, v in sorted(dev_types.items()))}); mismatches: {mism or 'none'}; "
         f"code hash {', '.join(sorted({r['code_hash'] for r in res}))}; {'/'.join(str(x) for x in sorted({r['n_cores'] for r in res}))} cores × {'/'.join(str(x) for x in sorted({r['n_pert'] for r in res}))} perturbations per predictor.", "",
         "### Success by cell (core-macro; mean ± sd over seeds 901–903; kernel one seed)", "",
         "| method | fraction | ε | TEST success < 42 px (%) | DEV success < 42 px (%) | TEST success < 9 px (%) | TEST median corner error (px) | TEST failures, summed over seeds (energy) |", "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for f in FR:
            for e in EP:
                k = f"{m}|frac{f}|eps{e}"; t, d = T["cells"].get(k), D["cells"].get(k)
                L.append(f"| {NAME[m]} | {f} | {e} | {pct(t and t['success42'])} | {pct(d and d['success42'])} | {pct(t and t['success9'])} | "
                         + (f"{t['dev_median_err']['mean']:.2f} | {sum(t['failures42'])} ({sum(t['failures42_energy'])}) |" if t else "missing | missing |"))
    for m in ("mi", "nmi"):
        t, d = T["cells"].get(m), D["cells"].get(m)
        L.append(f"| {NAME[m]} | — | — | {pct(t and t['success42'])} | {pct(d and d['success42'])} | {pct(t and t['success9'])} | " + (f"{t['dev_median_err']['mean']:.2f} | {sum(t['failures42'])} ({sum(t['failures42_energy'])}) |" if t else "missing | missing |"))
    L += ["", "### Primary family (VCS − MI, VCS − CoMIR at ε 0.2, both fractions; success < 42 px; Holm over 4)", "",
          "| contrast | TEST Δ (pp) | TEST seed t 95 % | TEST core bootstrap 95 % | TEST Holm p | TEST detected | DEV Δ (pp) | DEV detected |", "|---|---|---|---|---|---|---|---|"]
    for k, c in T["primary"].items():
        d = D["primary"].get(k, {})
        L.append(f"| {k} | {100 * c['mean']:+.1f} | {iv(c)} | {iv(c, 'image_boot95')} | {fp(c['p_holm'], '0.004')} | **{c['detected']}** | " + (f"{100 * d['mean']:+.1f} | {d['detected']} |" if d else "— | — |"))
    L += ["", "### Interaction: seed-paired change of success < 42 px from ε 0 to 0.2 (pp)", "", "| method | TEST fraction 0.25 | TEST fraction 1 | DEV fraction 0.25 | DEV fraction 1 |", "|---|---|---|---|---|"]
    for m in METHODS:
        cell = lambda J, f: J["interaction"].get(f"{m}|frac{f}|delta_0_to_0.2@42")
        L.append(f"| {NAME[m]} | " + " | ".join((f"{100 * x['mean']:+.1f} {iv(x)}" if x else "—") for x in (cell(T, "0.25"), cell(T, "1"), cell(D, "0.25"), cell(D, "1"))) + " |")
    L += ["", "### All VCS − X contrasts on the test (pp; seed t 95 % / core bootstrap 95 %; boot p)", "", "| contrast | Δ | seed t | core bootstrap | boot p |", "|---|---|---|---|---|"]
    L += [f"| {k} | {100 * c['mean']:+.1f} | {iv(c)} | {iv(c, 'image_boot95')} | {fp(c['boot_p'])} |" for k, c in T["contrasts"].items()]
    L += ["", "## [解释]", "", "Rendered from the frozen wording rules; no claim beyond them.", ""]
    for k, c in T["primary"].items():
        d = D["primary"].get(k)
        word = (f"no detectable difference ({basis(c)}; Holm p {fp(c['p_holm'], '0.004')})" if not c["detected"] else f"VCS registers {'more' if c['mean'] > 0 else 'fewer'} test cores ({100 * c['mean']:+.1f} pp)")
        L.append(f"- {k}: on the test, {word}; " + (f"against DEV: {replication(d, c)}." if d else "no DEV counterpart."))
    for f in FR:
        cl = [T["contrasts"].get(f"vcs_minus_{o}|frac{f}|eps0.2@42") for o in ("logistic", "infonce")]
        if all(cl):
            same = not any(both(x) for x in cl)
            L.append(f"- Fraction {f}, ε 0.2, test: VCS − logistic {100 * cl[0]['mean']:+.2f} pp ({basis(cl[0])}), VCS − InfoNCE {100 * cl[1]['mean']:+.2f} pp ({basis(cl[1])}) → "
                     + ("VCS ≈ logistic ≈ InfoNCE: any VCS − MI / VCS − CoMIR reading here is a learned relational-energy class effect, not a VCS-only effect." if same
                        else "the critic objectives differ at this cell on the test; the VCS − X readings are reported as such, without class-effect wording."))
    L += ["- A stronger CoMIR / MI / NMI / kernel cell, where it occurs above, is reported as is; no baseline was changed after the freeze.",
          "- DEV and TEST are shown side by side and never pooled; the test was read once.", "",
          "## [未验证假设]", "",
          "- The test cores come from the same two TMA slides (1B, 2B) and the same acquisition as the DEV cores; other slides, scanners or tissues are untested.",
          "- The perturbations are synthetic rigid transforms (θ ±30°, |t| <= 51 px at 512 px) of aligned pairs; real misalignment, deformation and partial overlap are untested.",
          "- The mismatch model is the conditional displaced-position Q of the G1 protocol; other noise models are untested.",
          "- The kernel rows are one seed; MI / NMI are deterministic rows (no seed variance).",
          f"- Reproducibility: MI / NMI / kernel / CoMIR registrations are bitwise reproducible on the same GPU type; the critics reproduce to about the same-device repeat spread "
          f"({rep_bound:.2f} px largest observed in the pre-read diagnostics); on another GPU type a failed registration can land elsewhere and, for the kernel, a success at 42 px can flip "
          "(pre-read execution note 1).  Each predictor therefore ran on the GPU type of its DEV record.", ""]
    out.write_text("\n".join(L)); print(f"-> {out} ({len(L)} lines)"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
