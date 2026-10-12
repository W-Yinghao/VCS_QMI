"""Execution note E4 helper (D1-20): early stage-2 units for the objectives whose learning-rate choice is already fully determined by the frozen
rule (scripts/r7_d1_plan20.py: per objective, 3e-4 if the two seed-0 CAL Spearmans are within 1e-4, else the higher) — logistic and direct.
Uses the frozen planner's own functions; writes the provisional choices and the 7 lines (logistic s1-2, direct s1-2, A2 direct --target d s0-2).
    python scripts/r7_d1_20_early_stage2.py --selectors-dir D --out-choice F --lines L"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_plan20 as P  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--selectors-dir", required=True); ap.add_argument("--out-choice", required=True); ap.add_argument("--lines", required=True)
a = ap.parse_args(); sel = Path(a.selectors_dir); ch = {}
for o in ("logistic", "direct"):
    logs = {}
    for lr in P.LRS:
        f, pt = sel / f"{P.tag(o, lr, 0)}.json", sel / f"{P.tag(o, lr, 0)}.pt"
        assert f.exists() and pt.exists(), f"seed-0 unit {P.tag(o, lr, 0)} not complete"
        logs[lr] = json.loads(f.read_text())["selected_cal_spearman"]
    best = 3e-4 if abs(logs[1e-4] - logs[3e-4]) <= 1e-4 else max(P.LRS, key=lambda lr: logs[lr])   # identical to P.choose
    ch[o] = {"chosen_lr": best, "cal_spearman": {f"{lr:g}": v for lr, v in logs.items()}}
lines = [P.line(o, ch[o]["chosen_lr"], s) for o in ("logistic", "direct") for s in (1, 2)] + [P.line("direct", ch["direct"]["chosen_lr"], s, "--target d") for s in (0, 1, 2)]
Path(a.out_choice).write_text(json.dumps(ch, indent=1)); Path(a.lines).write_text("# E4 early stage-2 lines (logistic / direct determined)\n" + "\n".join(lines) + "\n")
print(json.dumps(ch, indent=1)); print("\n".join(lines))
