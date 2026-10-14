"""V7X-40 scorer of the X21 common-feature fits on EvalMuse DEV (panel C0): the ONLY X21 step that reads DEV labels, run after every X21
prediction file exists (V7X_20 evaluation rule).  Metrics come from the frozen V7 scorer (scripts/v7/v7_score.py unit_table / paired_boot):
natural element Brier (prompt -> image -> element macro) of the prior-corrected output (primary), raw balanced and CAL_PROB-Platt readouts, NLL,
majority accuracy, prompt-macro AUROC, ECE.  Each X21 fit is converted to a unit-style directory (z = the balanced log ratio ell; y joined from
the V7-01 element manifest).  Contrasts (same-target family, docs/05): N-VCS-F minus each other method, per seed for neural methods (same seed
pairs) and against each deterministic kernel fit; prompt-cluster paired bootstrap 5,000 (seed 2718); native-CV kernel rows reported, not compared.
    python -I scripts/v7x/x40_x21_score.py --x21 DIR --out F"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v7"))
import v7_common as VC  # noqa: E402
import v7_score as VS  # noqa: E402


def convert(fit: Path, dst: Path, labels):
    log = json.loads((fit / "log.json").read_text()); dst.mkdir(parents=True, exist_ok=True); (dst / "pred").mkdir(exist_ok=True)
    if "prior" not in log:                                                     # a kernel's native-CV row: the parent fit's FIT prior
        log["prior"] = json.loads((fit.parent / "log.json").read_text())["prior"]; log["method"] = fit.parent.name + "-nativeCV"
    for role in ("CAL_PROB", "DEV"):
        with open(dst / "pred" / f"elements_{role}.jsonl", "w") as fh:
            for r in VC.read_jsonl(fit / f"pred_{role}.jsonl"):
                fh.write(json.dumps({"element_id": r["element_id"], "image_id": r["image_id"], "prompt_id": r["prompt_id"], "z": r["ell"], "y": labels[r["element_id"]]}) + "\n")
    (dst / "log.json").write_text(json.dumps({"prior_fit_spanned": log["prior"], "arm": log.get("method") or fit.name, "seed": log.get("seed", "det"), "lr": log.get("lr")}))
    return log


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--x21", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=str(VC.BIG_OUT / "x40_units")); a = ap.parse_args(); X = Path(a.x21); work = Path(a.work)
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists: the scorer never overwrites")
    labels = {e["element_id"]: e["y"] for e in VC.read_jsonl(VC.DATA_OUT / "manifest_elements.jsonl") if e.get("y") is not None}
    fits = sorted(p for p in (X / "formal").iterdir() if (p / "log.json").exists()) + sorted(p for p in (X / "kernels").iterdir() if (p / "log.json").exists())
    native = sorted(p / "native_cv" for p in (X / "kernels").iterdir() if (p / "native_cv" / "log.json").exists())
    tabs = {}
    for f in fits + native:
        name = f.name if f.name != "native_cv" else f.parent.name + "-nativeCV"; convert(f, work / name, labels); tabs[name] = VS.unit_table(work / name)
    vcs = {n: t for n, t in tabs.items() if n.startswith("N-VCS-F_s")}; contrasts = {}
    for n, t in tabs.items():
        if n.startswith("N-VCS-F") or n.endswith("-nativeCV"):
            continue
        if n.startswith("N-"):
            s = n.rsplit("_s", 1)[1]; ref = vcs.get(f"N-VCS-F_s{s}")
            if ref:
                contrasts[f"N-VCS-F_s{s} - {n}"] = VS.paired_boot(ref["per_prompt_brier"], t["per_prompt_brier"])
        else:
            for vn, ref in sorted(vcs.items()):
                contrasts[f"{vn} - {n}"] = VS.paired_boot(ref["per_prompt_brier"], t["per_prompt_brier"])
    out = {"units": [{k: v for k, v in t.items() if k != "per_prompt_brier"} for t in tabs.values()], "contrasts_natural_brier_dev": contrasts,
           "rule": "negative = N-VCS-F lower risk; neural per seed (same seed pairs), kernels deterministic vs each VCS seed; prompt-cluster paired bootstrap"}
    Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps({k: round(v["delta"], 5) for k, v in contrasts.items()}, indent=1)); return 0


if __name__ == "__main__":
    sys.exit(main())
