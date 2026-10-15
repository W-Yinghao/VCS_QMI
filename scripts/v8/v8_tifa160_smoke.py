"""V8-05 §B — functional smoke of `tifa160-run` on the REAL TIFA160 predictions (24 V8 units, B0 / B1 / B4T files) with SYNTHETIC labels.

No sealed file is opened: the module's label paths and request file are pointed at synthetic files written under OUT before the run (asserted).
Synthetic labels follow the public formats — a DSG-schema csv (item_id = tifa160_<n>, source_id = TIFA text id, DSG model names, 5 workers,
integer answers) and a TIFA-style json keyed by image file name with two integer ratings — and are a noisy monotone function of B1 VQAScore, so a
broken image-id alignment between labels and predictions shows up as a low B1 level.  Checks: every method on 800 images, both label sources
parsed for 800 images with the expected raters, families / comparators / Holm present, levels for 35 methods, B1 level high, runtime.  No ledger
row, no request-file change; the output is a code check only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v8_confirm as C  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); lab = out / "synthetic_labels"; lab.mkdir()
    cands = C.read_jsonl(C.TIFA160_INPUTS); ims = [c["image_id"] for c in cands]
    b1 = C.load_external("B1_VQAScore", C.TIFA160_EXTERNAL["B1_VQAScore"]); rng = np.random.default_rng(0)
    r = C.ranks(np.array([b1[i] for i in ims])) / len(ims)
    lat = {im: 1 + 4 * float(np.clip(x + rng.normal(0, 0.1), 0, 1)) for im, x in zip(ims, r)}
    tag2dsg = {v: k for k, v in C.DSG_T2I_TO_TAG.items()}; pidx = {p: i for i, p in enumerate(sorted({c["prompt_id"] for c in cands}))}
    rows, tifa = ["item_id,source_id,t2i_model,worker_id,answer"], {}
    for c in cands:
        im = c["image_id"]
        for w in range(5):
            rows.append(f"tifa160_{pidx[c['prompt_id']]},{c['prompt_id']},{tag2dsg[c['generator_id']]},W{w},{int(np.clip(round(lat[im] + rng.normal(0, 0.7)), 1, 5))}")
        tifa[im.split("/")[-1]] = {"text": c["prompt"], "human_scores": [int(np.clip(round(lat[im] + rng.normal(0, 0.9)), 1, 5)) for _ in range(2)]}
    (lab / "tifa160-likert-anns.csv").write_text("\n".join(rows) + "\n"); (lab / "human_annotations.json").write_text(json.dumps(tifa))
    req = lab / "request.md"; req.write_text("synthetic smoke only: TIFA160 CONFIRMED (not a confirmation)\n")
    C.TIFA160_LABELS = {"tifa_2rater": lab / "human_annotations.json", "dsg_5rater": lab / "tifa160-likert-anns.csv"}; C.REQUEST = req
    assert all(str(p).startswith(str(out)) for p in list(C.TIFA160_LABELS.values()) + [C.REQUEST]), "smoke must not touch sealed files"
    t0 = time.time(); rc = C.cmd_tifa160_run(argparse.Namespace(out=str(out / "run"), allow_scoring=True)); secs = time.time() - t0
    res = json.load(open(out / "run" / "tifa160_confirm.json")); md = (out / "run" / "tifa160_confirm.md").read_text()
    L = res["labels"]; ck = {"rc": rc == 0, "images_every_method_800": res["images_scored_by_every_method"] == 800,
                             "comparators": res["comparators_present"] == ["B0_FGA_released_O", "B1_VQAScore", "B4T_TIFA"] and res["comparators_absent"] == ["B4D_DSG"],
                             "tifa_parsed_800": L["tifa_2rater"]["images_with_rating"] == 800 and L["tifa_2rater"]["meta"]["identified_by"] == {"key:image": 800}
                             and L["tifa_2rater"]["meta"]["ratings_per_image"] == {"2": 800} and not L["tifa_2rater"]["error"],
                             "dsg_parsed_800": L["dsg_5rater"]["images_with_rating"] == 800 and L["dsg_5rater"]["raters_per_image"] == [5] and not L["dsg_5rater"]["error"]
                             and L["dsg_5rater"]["meta"]["counts"].get("rows_used") == 4000}
    for lb in ("tifa_2rater", "dsg_5rater"):
        R = res["readouts"][lb]; F = R.get("families", {})
        ck[f"{lb}_n"] = R.get("n_images") == 800 and R.get("n_prompts") == 160 and R.get("n_dropped_no_rating") == 0
        ck[f"{lb}_families"] = (list(F.get("tifa_primary", {})) == ["B0_FGA_released_O", "B1_VQAScore", "B4T_TIFA"] and list(F.get("tifa_attribution", {})) == ["U-BCE"]
                                and len(F.get("tifa_same_data_descriptive", {})) == 6
                                and all("p_holm" in F[f][c][cn] for f in ("tifa_primary", "tifa_attribution") for c in F[f] for cn in C.CORR))
        ck[f"{lb}_levels"] = len(R["levels"]) == 35 and R["levels"]["B1_VQAScore"]["spearman"] > 0.7 and len(R["spearman_per_generator"]["U-CS"]) == 5
    ck["markdown"] = "## labels = tifa_2rater" in md and "## labels = dsg_5rater" in md and "B4T_TIFA" in md
    rep = {"checks": ck, "all_ok": all(ck.values()), "seconds_run": round(secs, 1), "b1_level_spearman_synthetic": {lb: res["readouts"][lb]["levels"]["B1_VQAScore"]["spearman"] for lb in ("tifa_2rater", "dsg_5rater")}}
    (out / "smoke_checks.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
