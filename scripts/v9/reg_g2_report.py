"""V9-G2 report writer.  Every number is generated: G2 readings from reports/V9/V9_G2_results/g2_results.json (scripts/v9/reg_g2.py aggregate),
job ids from reports/V9/V9_JOB_IDS.json, devices from the unit result files, the r6 REG-P3 context from its result files, the G1 context from
reports/V9/V9_G1_results/g1_results.json and reports/V9/V9_G1_TEST_results/g1_test_results.json, the critics' repeat bound from the G1 test-read
diagnostics.  [解释] is rendered mechanically from the frozen wording rules of reports/V9/V9_G2_PROTOCOL_FROZEN_20261015.yaml; no number is typed.
The output depends only on these files and the arguments (regenerating gives identical bytes on the same day).
    python scripts/v9/reg_g2_report.py --results-commit <hash> [--out reports/V9/V9_G2_REPORT_<date>.md]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METHODS = ("vcs", "logistic", "infonce", "comir", "kernel"); FR = ("0.25", "1"); EP = ("0", "0.2"); CRIT = ("vcs", "logistic", "infonce")
NAME = {"vcs": "VCS", "logistic": "logistic (BCE)", "infonce": "InfoNCE", "comir": "CoMIR", "kernel": "kernel (seed 901)", "mi": "MI", "nmi": "NMI"}
UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/reg_epfl/units"); P3 = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3/results")
DIAG = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/reg_test_fixture/20261015T211818Z")


def pct(t):
    return "—" if not t or t.get("mean") is None else f"{100 * t['mean']:.1f}" + (f" ± {100 * t['sd']:.1f}" if t.get("sd") is not None else "")


def iv(c, key="ci95", nd=1):
    v = c.get(key) if c else None; return f"[{100 * v[0]:+.{nd}f}, {100 * v[1]:+.{nd}f}]" if v else "—"


def fp(x, floor="0.001"):
    """Two-sided bootstrap p from 2 000 resamples: the smallest non-zero value is 2 / 2 000 = 0.001, so 0 means < 0.001; Holm over 4 multiplies by up to 4 (< 0.004)."""
    return "—" if x is None else f"< {floor}" if x == 0 else f"{x:.4f}"


def excl(v):
    return v is not None and (v[0] > 0 or v[1] < 0)


def basis(c):
    return (f"seed t {iv(c, nd=2)} {'excludes' if excl(c.get('ci95')) else 'contains'} 0, unit bootstrap {iv(c, 'unit_boot95', 2)} "
            f"{'excludes' if excl(c.get('unit_boot95')) else 'contains'} 0")


def gpu(d):
    return "L40S" if "L40S" in d else "RTX PRO 6000" if "PRO 6000" in d else "H100" if "H100" in d else d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, default=REPO / "reports/V9/V9_G2_results/g2_results.json")
    ap.add_argument("--results-commit", default="—"); ap.add_argument("--registry", type=Path, default=REPO / "reports/V9/V9_JOB_IDS.json"); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(); R = json.load(open(a.results)); out = a.out or REPO / f"reports/V9/V9_G2_REPORT_{dt.date.today():%Y%m%d}.md"
    C, K, Pm, I, ceil = R["cells"], R["contrasts"], R["primary"], R["interaction"], R["ceiling_readout"]
    reg = [e for e in json.load(open(a.registry)) if e.get("unit", "").startswith("V9-G2")]
    jobs = "; ".join(f"{', '.join(e['jobs'])} ({e['unit'].split(':')[0].split('(')[0].strip()})" for e in reg if e.get("jobs"))
    cancelled = sorted({j for e in reg for j in e.get("cancelled_before_start_deferred", {})})
    res = [json.load(open(f)) for f in sorted(UNITS.glob("*/p3/results/*.json"))]; dev = {}
    for r in res:
        dev[gpu(r.get("device", "?"))] = dev.get(gpu(r.get("device", "?")), 0) + 1
    hashes = sorted({r.get("g2_code_hash") for r in res if r.get("g2_code_hash")})
    r6 = {}
    for m in METHODS + ("mi", "nmi"):
        fs = sorted(P3.glob(f"{m}_s*.json")) if m not in ("mi", "nmi") else [P3 / f"{m}.json"]
        v = [json.load(open(f))["dev_success42_unit"] for f in fs if f.exists()]
        if v:
            r6[m] = (sum(v) / len(v), len(v))
    g1d = json.load(open(REPO / "reports/V9/V9_G1_results/g1_results.json"))["primary"]; g1t = json.load(open(REPO / "reports/V9/V9_G1_TEST_results/g1_test_results.json"))["primary"]
    diags = [json.load(open(DIAG / n)) for n in ("diag.json", "diag_l40s.json") if (DIAG / n).exists()]
    rep = max((r["rep1_vs_rep2_max_abs_d_err"] for d in diags for p, r in d["rows"].items() if p.split("-")[2:3] and p.split("-")[2] in CRIT), default=None)
    L = [f"# V9-G2 report: EPFL RGB–NIR rigid registration under less data and mismatched training correspondences ({dt.date.today().isoformat()})", "",
         f"Protocol `V9_G2_PROTOCOL_FROZEN_20261015.yaml` (frozen before any formal unit).  Jobs: {jobs}" + (f"; cancelled before start: {', '.join(cancelled)}" if cancelled else "") + ".  "
         f"Results-only commit {a.results_commit}: `V9_G2_results/g2_results.json` and `g2_tables.md`, written by `scripts/v9/reg_g2.py aggregate` from the per-unit results under "
         f"`{UNITS}`.  Missing units: {R['missing'] or 'none'}.  MI / NMI reused from r6 REG-P3 after the hash check (perturbations {R['mi_nmi_reuse']['perturbations_sha256']}, "
         f"manifest {R['mi_nmi_reuse']['manifest_sha256']}).  Every number below is generated by `scripts/v9/reg_g2_report.py` from these files.  "
         "Package format: [已知证据] / [本轮结果] / [解释] / [未验证假设].", "",
         "## [已知证据]",
         "- r6 REG-P3 (seeds 0–2, all 330 FIT units, clean correspondences) defined the data, roles, perturbations, thresholds and unit pooling used here and read the same 72 DEV units; "
         "its DEV unit success < 10.5 native px: " + ", ".join(f"{NAME[m].split(' (')[0]} {100 * v:.1f} %" + (f" (mean of {n} seeds)" if n > 1 else "") for m, (v, n) in r6.items()) + ".",
         "- V9-G1 (BF/SHG) primary family, DEV / sealed TEST: " + "; ".join(f"VCS − {NAME[k.split('|')[0].replace('vcs_minus_', '')]} (fraction {k.split('|')[1].replace('frac', '')}) "
                                                                            f"{100 * g1d[k]['mean']:+.1f} / {100 * g1t[k]['mean']:+.1f} pp, detected {g1d[k]['detected']} / {g1t[k]['detected']}" for k in g1d) + ".",
         "- Nothing here assumes that a bounded objective is robust to noisy supervision (docs/06 §1); G1 and G2 are two datasets reported side by side, not pooled.", "",
         "## [本轮结果]", "",
         f"{len(res)} unit results (48 trainable units + 4 kernel rows); devices: {', '.join(f'{k} {v}' for k, v in sorted(dev.items()))}; G2 code hashes: {', '.join(hashes) or '—'}.  "
         f"Ceiling readout (pre-registered): every CAL unit success < 10.5 px >= 0.95 → {ceil['all_cal_success42_ge_0.95']} (minimum {100 * ceil['min_cal_success42']:.1f} %), "
         + ("so the < 2.25 px success is the discriminating readout (truth caveat)." if ceil["all_cal_success42_ge_0.95"] else "so success < 10.5 px stays primary."), "",
         "### DEV success by cell (unit-macro over 72 DEV scenes; mean ± sd over seeds 901–903; kernel one seed)", "",
         "| method | fraction | ε | n FIT units | success < 10.5 px (%) | success < 2.25 px (%) | median corner error, err_full px (native px) | failures, summed over seeds (energy) | s / registration |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for f in FR:
            for e in EP:
                c = C.get(f"{m}|frac{f}|eps{e}")
                if not c:
                    L.append(f"| {NAME[m]} | {f} | {e} | missing | | | | | |"); continue
                med = c["dev_median_err"]["mean"]
                L.append(f"| {NAME[m]} | {f} | {e} | {'/'.join(str(x) for x in sorted(set(c['n_fit_units'])))} | {pct(c['success42'])} | {pct(c['success9'])} | {med:.2f} ({med / 4:.2f}) | "
                         f"{sum(c['failures42'])} ({sum(c['failures42_energy'])}) | {c['dev_sec_per_registration']['mean']:.2f} |")
    for m in ("mi", "nmi"):
        c = C.get(m)
        if c:
            med = c["dev_median_err"]["mean"]
            L.append(f"| {NAME[m]} (r6 REG-P3, reused) | — | — | — | {pct(c['success42'])} | {pct(c['success9'])} | {med:.2f} ({med / 4:.2f}) | {sum(c['failures42'])} ({sum(c['failures42_energy'])}) | — |")
    L += ["", "### Primary family (VCS − MI, VCS − CoMIR at ε 0.2, both fractions; success < 10.5 px; Holm over 4; detected = seed t and unit bootstrap exclude 0 and Holm p < 0.05)", "",
          "| contrast | Δ (pp) | seed t 95 % | unit bootstrap 95 % | boot p | Holm p | detected | seed pairs |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {100 * c['mean']:+.1f} | {iv(c)} | {iv(c, 'unit_boot95')} | {fp(c['boot_p'])} | {fp(c['p_holm'], '0.004')} | **{c['detected']}** | {len(c['pairs'])} |" for k, c in Pm.items()]
    L += ["", "### Interaction: seed-paired change of success < 10.5 px from ε 0 to 0.2 (pp; seed t 95 %)", "", "| method | fraction 0.25 | fraction 1 |", "|---|---|---|"]
    for m in METHODS:
        row = [I.get(f"{m}|frac{f}|delta_0_to_0.2@42") for f in FR]
        L.append(f"| {NAME[m]} | " + " | ".join((f"{100 * t['mean']:+.1f} {iv(t)}" if t else "—") for t in row) + " |")
    L += ["", "### All VCS − X contrasts (pp; seed t 95 % / unit bootstrap 95 %; boot p)", "", "| contrast | Δ | seed t | unit bootstrap | boot p |", "|---|---|---|---|---|"]
    L += [f"| {k} | {100 * c['mean']:+.1f} | {iv(c)} | {iv(c, 'unit_boot95')} | {fp(c['boot_p'])} |" for k, c in K.items()]
    L += ["", "## [解释]", "", "Rendered from the frozen wording rules; no claim beyond them.", ""]
    for k, c in Pm.items():
        word = (f"VCS registers {'more' if c['mean'] > 0 else 'fewer'} DEV scenes ({100 * c['mean']:+.1f} pp; Holm p {fp(c['p_holm'], '0.004')})" if c["detected"]
                else f"no detectable difference ({basis(c)}; Holm p {fp(c['p_holm'], '0.004')})")
        L.append(f"- {k}: {word}" + ("" if c["complete"] else "; INCOMPLETE: fewer than 3 seed pairs") + ".")
    for f in FR:
        cl = [K.get(f"vcs_minus_{o}|frac{f}|eps0.2@42") for o in ("logistic", "infonce")]
        if all(cl):
            same = not any(excl(x.get("ci95")) and excl(x.get("unit_boot95")) for x in cl)
            L.append(f"- Fraction {f}, ε 0.2: VCS − logistic {100 * cl[0]['mean']:+.2f} pp ({basis(cl[0])}), VCS − InfoNCE {100 * cl[1]['mean']:+.2f} pp ({basis(cl[1])}) → "
                     + ("VCS ≈ logistic ≈ InfoNCE: any VCS − MI / VCS − CoMIR reading at this cell is a learned relational-energy class effect, not a VCS-only effect." if same
                        else "the critic objectives differ at this cell; the VCS − X readings are reported as such, without class-effect wording."))
    L += ["- A stronger CoMIR / MI / NMI / kernel cell, where it occurs above, is reported as is; no baseline was weakened (recipes frozen).",
          "- The interaction rows are descriptive (intervals shown); differences of changes are not tested.", "",
          "## [未验证假设]", "",
          "- DEV results are development validation: r6 REG-P3 read the same DEV units, and EPFL has no sealed split, so G2 makes no confirmation claim.",
          "- Success < 2.25 native px is read with the truth caveat: the dataset's own RGB–NIR alignment is accurate to about 2 native px.",
          "- The mismatch model is the conditional displaced-position Q of the protocol (same crop, >= 2 patch radii); other noise models are untested.",
          "- The kernel rows are one seed; MI / NMI are deterministic rows reused from r6 REG-P3 (no seed variance).",
          f"- Reproducibility: the G2 units ran on mixed GPU types ({', '.join(f'{k} {v}' for k, v in sorted(dev.items()))}).  The G1 sealed-test pre-read diagnostics showed that CoMIR / kernel / "
          f"MI / NMI registrations are bitwise reproducible only on the same GPU type and the critics to about their same-device repeat spread ({rep:.2f} px largest observed); across GPU types a "
          "fragile registration can change outcome.  G2 was not re-run on a single GPU type.",
          "- Transfer between BF/SHG (G1) and RGB–NIR (G2) is not tested; the two datasets stand side by side.", ""]
    out.write_text("\n".join(L)); print(f"-> {out} ({len(L)} lines)"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
