"""Consolidate every completed run (all stages) into one table: hyper-parameters + final metrics + spectrum/saturation summaries.

    python scripts/consolidate_results.py --output-root $OUTPUT_ROOT --out reports/ALL_RUNS.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def hp(rd: Path) -> dict:
    rm = jload(rd / "run_manifest.json") or {}
    c = yaml.safe_load((rd / "config.resolved.yaml").read_text())
    h = rm.get("hparams") or {}
    out = {
        "K": c["pairing"]["k"], "critic_input": c["model"]["critic"]["input"], "critic_hidden": c["model"]["critic"]["hidden_dims"],
        "gain": c["model"]["critic"]["last_layer_xavier_gain"], "clr": c["optimizer"]["critic_lr_multiplier"], "cwd": c["optimizer"]["critic_weight_decay"],
        "proj": f'{c["model"]["projector"]["hidden_dim"]}/{c["model"]["projector"]["output_dim"]}/d{c["model"]["projector"].get("depth", 2)}',
        "norm": c["model"]["normalization"]["vcs_and_simclr"], "feat_src": c["model"]["critic"].get("feature_source", "z"),
        "sampler": "sym" if c["pairing"]["sampler"].endswith("symmetric") else "cyc", "neg_detach": c["pairing"]["negative_detach"],
        "mode": c["train"]["mode"], "target": c["train"].get("target_branch", "shared"), "pred": c["model"]["projector"].get("predictor", False),
        "B": c["train"]["batch_size_images"], "lr": c["optimizer"]["lr"], "floor": c["schedule"]["min_lr_ratio"], "wd": c["optimizer"]["matrix_weight_decay_encoder_projector"],
        "crop": c["views"]["random_resized_crop"]["scale"][0], "cj": c["views"]["color_jitter"]["brightness"], "blur": c["views"]["gaussian_blur_p"],
        "epochs": c["train"]["epochs"], "cos_a0": c["model"]["critic"].get("cosine_scale_init", 1.0),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for rd in sorted(p for p in Path(a.output_root).iterdir() if p.is_dir()):
        st = jload(rd / "status.json")
        if not st or "FAILED_ARCHIVE" in str(st.get("stage")) or st.get("stage") in ("P2_smoke", "TEST") or rd.name.startswith(("smoke_", "det_")):
            continue
        evs = sorted((rd / "evaluations").glob("evaluation_epoch_*.json")) if (rd / "evaluations").is_dir() else []
        ev = jload(evs[-1]) if evs else None
        ev0 = jload(rd / "evaluations" / "evaluation_initial.json")
        summ = jload(rd / "summary.json") or {}
        h = hp(rd) if (rd / "config.resolved.yaml").is_file() else {}
        sp = (ev or {}).get("spectrum") or {}
        p_eig = (sp.get("p_raw") or {}).get("eigenvalues_desc")
        cliff = next((i for i in range(1, len(p_eig)) if p_eig[i - 1] > 5 * p_eig[i]), None) if p_eig else None
        ch = (ev or {}).get("critic_holdout") or {}
        rep = ch.get("per_repeat") or []
        rows.append({
            "run": rd.name, "stage": st.get("stage"), "method": st.get("method"), "seed": st.get("seed"), "status": st.get("status"),
            "epochs_done": st.get("completed_epoch"), **h,
            "linear": None if ev is None else ev.get("linear_val_top1_pct"), "linear_ep0": None if ev0 is None else ev0.get("linear_val_top1_pct"),
            "knn": None if ev is None else ev.get("knn_val_top1_pct"), "h_rank": None if ev is None else ev.get("h_effective_rank"),
            "z_rank": None if ev is None else ev.get("z_effective_rank"), "p_cliff": cliff, "h_norm": (sp.get("h") or {}).get("norm_mean"),
            "heldout_J": None if ev is None else ev.get("heldout_J"),
            "sat_pos": (sum(r["sat_pos_frac"] for r in rep) / len(rep)) if rep else None, "sat_neg": (sum(r["sat_neg_frac"] for r in rep) / len(rep)) if rep else None,
            "collapse": summ.get("collapse_suspected"), "train_s": summ.get("train_seconds"), "img_s": summ.get("steady_state_images_per_s"),
        })
    rows.sort(key=lambda r: (-(r["linear"] or -1)))
    L = ["# All runs (completed or not), sorted by final linear-val", "",
         "| run | stage | method | K | critic | clr | proj | src | target | pred | B | lr | aug(crop/cj/blur) | ep | linear | kNN | h-rank | z-rank | p-cliff | ‖h‖ | J | sat+/− | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, nd=2: "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    for r in rows:
        crit = r.get("critic_input", "—")
        if crit == "ordered_concat":
            crit = f"mlp{r.get('critic_hidden')}"
        L.append(f"| {r['run']} | {r['stage']} | {r['method']} | {r.get('K','—')} | {crit} | {r.get('clr','—')} | {r.get('proj','—')} | {r.get('feat_src','—')} | "
                 f"{r.get('target','—')}{'+pred' if r.get('pred') else ''}{' negdet' if r.get('neg_detach') else ''}{' ' + r['mode'] if r.get('mode') not in (None, 'joint') else ''} | "
                 f"{r.get('pred','—')} | {r.get('B','—')} | {r.get('lr','—')} | {r.get('crop','—')}/{r.get('cj','—')}/{r.get('blur','—')} | {r.get('epochs','—')} | "
                 f"{f(r['linear'])} | {f(r['knn'])} | {f(r['h_rank'], 1)} | {f(r['z_rank'], 1)} | {f(r['p_cliff'])} | {f(r['h_norm'], 1)} | {f(r['heldout_J'], 3)} | "
                 f"{f(r['sat_pos'])}/{f(r['sat_neg'])} | {r['status']}{' COLLAPSE' if r['collapse'] else ''} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"{len(rows)} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
