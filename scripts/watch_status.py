"""Live status of all runs (hyper-parameter watcher).

    python scripts/watch_status.py --output-root $OUTPUT_ROOT --out reports/WATCH_status.md [--stages P12_vcs_hparamA,P8_long800,...]

For every run: status, epoch progress, ETA, latest monitored-epoch kNN / heldout-J / h-rank, final linear-val (when evaluated), delta vs
baseline, the changed hyper-parameter, SLURM job, and flags (FAILED, STALLED, COLLAPSE, NaN).  Prints a one-line "event key" so a
watcher loop can detect changes (set of completed/failed runs + number of monitored evaluations).
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
from pathlib import Path

import yaml

BASELINES = {  # (stage, seed) -> reference linear/kNN for deltas
    ("P12_vcs_hparamA", 0): {"run": "P5_vcs_seed0", "linear": 74.84, "knn": 64.64, "h_rank": 12.87},
    ("P10_kstudy200", 0): {"run": "P5_vcs_seed0", "linear": 74.84, "knn": 64.64, "h_rank": 12.87},
    ("P14_vcs_hparamB_lr", 0): {"run": "P5_vcs_seed0", "linear": 74.84, "knn": 64.64, "h_rank": 12.87},
    ("P16_vcs_hparamB_set", 0): {"run": "P5_vcs_seed0", "linear": 74.84, "knn": 64.64, "h_rank": 12.87},
    ("P18_vcs_critic_variants", 0): {"run": "P10_vcs_k8_seed0 (K=8)", "linear": 76.12, "knn": 67.44, "h_rank": 15.59},
    ("P20_vcs_ssl_wiring", 0): {"run": "P16_vcs_k8_clr10_seed0 (K=8, clr x10)", "linear": 77.44, "knn": 68.40, "h_rank": 21.44},
    ("P22_vcs_target_branch", 0): {"run": "P16_vcs_k8_clr10_seed0 (K=8, clr x10)", "linear": 77.44, "knn": 68.40, "h_rank": 21.44},
    ("P24_vcs_cosine_base", 0): {"run": "P18_vcs_crit_cosine_seed0 (cosine critic, K=8)", "linear": 78.32, "knn": 72.76, "h_rank": 57.74},
    ("P26_vcs_negdetach_base", 0): {"run": "P24_vcs_cos_negdetach_seed0 (cosine, K=8, neg-detach)", "linear": 80.48, "knn": 74.62, "h_rank": 30.87},
    ("P26_vcs_negdetach_base", 1): {"run": "P24_vcs_cos_negdetach_seed0 (seed-0 reference)", "linear": 80.48, "knn": 74.62, "h_rank": 30.87},
    ("P26_vcs_negdetach_base", 2): {"run": "P24_vcs_cos_negdetach_seed0 (seed-0 reference)", "linear": 80.48, "knn": 74.62, "h_rank": 30.87},
    ("P10_kstudy200", 1): {"run": "P5_vcs_seed1", "linear": 74.24, "knn": 63.64, "h_rank": 13.46},
    ("P10_kstudy200", 2): {"run": "P5_vcs_seed2", "linear": 73.94, "knn": 63.50, "h_rank": 13.22},
    ("P8_long800", 0): {"run": "P5_vcs_seed0 (200ep)", "linear": 74.84, "knn": 64.64, "h_rank": 12.87},
    ("P8_long800", 1): {"run": "P5_vcs_seed1 (200ep)", "linear": 74.24, "knn": 63.64, "h_rank": 13.46},
    ("P8_long800", 2): {"run": "P5_vcs_seed2 (200ep)", "linear": 73.94, "knn": 63.50, "h_rank": 13.22},
}
BASE_TRAJ = {0: {0: 36.54, 10: 42.02, 20: 47.42, 50: 54.60, 100: 60.88, 150: 63.78, 200: 64.64},
             1: {0: 37.42, 10: 43.44, 20: 48.50, 50: 55.70, 100: 60.54, 150: 63.34, 200: 63.64},
             2: {0: 37.08, 10: 41.34, 20: 49.26, 50: 56.76, 100: 61.54, 150: 63.18, 200: 63.50}}
BASE_HP = {"K": 1, "critic_hidden_dims": [512, 512], "critic_last_gain": 0.1, "critic_lr_multiplier": 1.0, "critic_weight_decay": 0.0,
           "projector_hidden_dim": 512, "projector_output_dim": 128, "critic_input_norm": "l2", "batch_size_images": 256, "lr": 0.001,
           "min_lr_ratio": 0.01, "warmup_epochs": 10, "crop_scale_min": 0.2, "color_jitter": [0.4, 0.4, 0.4, 0.1], "gaussian_blur_p": 0.0,
           "matrix_weight_decay": 0.0001, "critic_input": "ordered_concat", "pair_symmetric": False, "critic_steps": 1,
           "critic_feature_source": "z", "negative_detach": False, "target_branch": "shared", "predictor": False, "projector_depth": 2, "cosine_scale_init": 1.0, "cosine_scale_fixed": False, "cosine_bias_calibrate": False, "projector_output_bn": False}


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def jsonl(p: Path):
    try:
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def hparams(rd: Path):
    rm = jload(rd / "run_manifest.json") or {}
    if rm.get("hparams"):
        return rm["hparams"]
    p = rd / "config.resolved.yaml"
    if not p.is_file():
        return {}
    c = yaml.safe_load(p.read_text())
    return {"K": c["pairing"]["k"], "critic_hidden_dims": c["model"]["critic"]["hidden_dims"], "critic_last_gain": c["model"]["critic"]["last_layer_xavier_gain"],
            "critic_lr_multiplier": c["optimizer"]["critic_lr_multiplier"], "critic_weight_decay": c["optimizer"]["critic_weight_decay"],
            "projector_hidden_dim": c["model"]["projector"]["hidden_dim"], "projector_output_dim": c["model"]["projector"]["output_dim"],
            "critic_input_norm": c["model"]["normalization"]["vcs_and_simclr"], "batch_size_images": c["train"]["batch_size_images"],
            "lr": c["optimizer"]["lr"], "epochs": c["train"]["epochs"], "min_lr_ratio": c["schedule"]["min_lr_ratio"], "warmup_epochs": c["train"]["warmup_epochs"],
            "crop_scale_min": c["views"]["random_resized_crop"]["scale"][0], "color_jitter": [c["views"]["color_jitter"][k] for k in ("brightness", "contrast", "saturation", "hue")],
            "gaussian_blur_p": c["views"]["gaussian_blur_p"], "matrix_weight_decay": c["optimizer"]["matrix_weight_decay_encoder_projector"],
            "critic_input": c["model"]["critic"]["input"], "pair_symmetric": c["pairing"]["sampler"].endswith("symmetric"),
            "critic_steps": 1 if c["train"]["mode"] == "joint" else int(c["train"]["mode"].rsplit("_", 1)[1]),
            "critic_feature_source": c["model"]["critic"].get("feature_source", "z"), "negative_detach": c["pairing"]["negative_detach"],
            "target_branch": c["train"].get("target_branch", "shared"), "predictor": c["model"]["projector"].get("predictor", False),
            "projector_depth": c["model"]["projector"].get("depth", 2), "cosine_scale_init": c["model"]["critic"].get("cosine_scale_init", 1.0),
            "cosine_scale_fixed": c["model"]["critic"].get("cosine_scale_fixed", False), "cosine_bias_calibrate": c["model"]["critic"].get("cosine_bias_calibrate", False),
            "projector_output_bn": c["model"]["projector"]["output_batchnorm"]}


def changed_hp(h: dict) -> str:
    diffs = [f"{k}={h[k]}" for k, v in BASE_HP.items() if k in h and h[k] != v]
    return ", ".join(diffs) if diffs else "baseline recipe"


def squeue_jobs() -> dict[str, tuple[str, str]]:
    out = os.popen("squeue -h -u $USER -o '%i|%j|%T|%M|%N' 2>/dev/null").read().strip().splitlines()
    return {l.split("|")[1]: tuple(l.split("|")) for l in out if "|" in l}


def fmt(v, nd=2):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def scan(output_root: Path, stages: set[str]):
    rows = []
    now = time.time()
    for rd in sorted(p for p in output_root.iterdir() if p.is_dir()):
        st = jload(rd / "status.json")
        if not st or st.get("stage") not in stages:
            continue
        E = jsonl(rd / "logs" / "epochs.jsonl")
        mon = [r for r in E if r.get("knn_val_top1_pct") is not None]
        last_mon = mon[-1] if mon else None
        ev_files = sorted((rd / "evaluations").glob("evaluation_epoch_*.json")) if (rd / "evaluations").is_dir() else []
        ev = jload(ev_files[-1]) if ev_files else None
        summ = jload(rd / "summary.json") or {}
        ep, tot = st.get("completed_epoch", 0), st.get("epochs_intended") or 0
        # progress / stall detection from last epoch record time
        last_rec = E[-1] if E else None
        age_min = None
        if last_rec and last_rec.get("utc"):
            age_min = (now - calendar.timegm(time.strptime(last_rec["utc"], "%Y-%m-%dT%H:%M:%SZ"))) / 60
        train_s = (E[-1].get("train_seconds") if E else None) or 0
        eta_min = None
        if st["status"] == "RUNNING" and ep > 0 and tot and train_s:
            eta_min = train_s / ep * (tot - ep) / 60
        h = hparams(rd)
        base = BASELINES.get((st.get("stage"), st.get("seed")))
        base_traj = BASE_TRAJ.get(st.get("seed"), {})
        if st.get("stage") == "P18_vcs_critic_variants":
            base_traj = {0: 36.58, 10: 45.58, 20: 52.44, 50: 60.30, 100: 64.76, 150: 66.82, 200: 67.44}  # P10_vcs_k8_seed0 (K=8 baseline)
        if st.get("stage") == "P26_vcs_negdetach_base":
            base_traj = {0: 36.58, 10: 53.94, 20: 60.88, 50: 68.14, 100: 71.84, 150: 74.26, 200: 74.62}  # P24_vcs_cos_negdetach_seed0
        if st.get("stage") == "P24_vcs_cosine_base":
            base_traj = {0: 36.58, 10: 56.54, 20: 64.00, 50: 69.16, 100: 71.72, 150: 72.84, 200: 72.76}  # P18_vcs_crit_cosine_seed0
        if st.get("stage") in ("P20_vcs_ssl_wiring", "P22_vcs_target_branch"):
            base_traj = {0: 36.58, 50: 63.96, 100: 66.70, 150: 68.38, 200: 68.40}  # P16_vcs_k8_clr10_seed0
        if h.get("epochs") == 800 and st.get("stage") != "P8_long800":
            base = {"run": "P8_vcs800_seed0 (800ep)", "linear": 79.24, "knn": 71.36, "h_rank": 20.13}
        lin = ev["linear_val_top1_pct"] if ev else None
        knn_final = ev["knn_val_top1_pct"] if ev else (last_mon["knn_val_top1_pct"] if last_mon else None)
        flags = []
        if st["status"].startswith("FAILED") or st["status"] == "STOPPED_BUDGET":
            flags.append(st["status"])
        if st["status"] == "RUNNING" and age_min is not None and age_min > 30:
            flags.append(f"STALLED? no epoch for {age_min:.0f} min")
        if st.get("collapse_suspected") or summ.get("collapse_suspected"):
            flags.append("COLLAPSE_SUSPECTED")
        signal = None
        if lin is not None and base:
            d = lin - base["linear"]
            signal = ("HELPS" if d > 1.0 else "HURTS" if d < -1.0 else "neutral") + f" ({d:+.2f})"
        rows.append({
            "run": rd.name, "stage": st["stage"], "seed": st.get("seed"), "status": st["status"], "epoch": ep, "total": tot,
            "eta_min": eta_min, "changed": changed_hp(h), "last_mon_epoch": last_mon["epoch"] if last_mon else None,
            "knn": knn_final, "heldout_J": (ev.get("heldout_J") if ev else (last_mon.get("heldout_J") if last_mon else None)),
            "h_rank": (ev.get("h_effective_rank") if ev else (last_mon.get("h_effective_rank") if last_mon else None)),
            "linear": lin, "signal": signal, "base": base["run"] if base else None, "base_knn": base["knn"] if base else None,
            "flags": flags, "job": st.get("slurm_job_id"), "n_mon": len(mon), "knn_traj": {r["epoch"]: r["knn_val_top1_pct"] for r in mon},
            "base_traj": base_traj,
        })
    return rows


def render(rows, stages):
    L = [f"# Watcher status — {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", "",
         f"Stages: {', '.join(sorted(stages))}.  Baseline for deltas: same-seed K=1 200-epoch run (P5).  "
         "Signal rule (pre-registered): HELPS if final linear-val > baseline + 1.0, HURTS if < baseline − 1.0, else neutral.", ""]
    for stg in sorted(stages):
        rs = [r for r in rows if r["stage"] == stg]
        if not rs:
            continue
        L += [f"## {stg}", "", "| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: (-(r["linear"] or -1), r["run"])):
            bt = r.get("base_traj") or {}
            dk = None if r["knn"] is None or r["last_mon_epoch"] not in bt else r["knn"] - bt[r["last_mon_epoch"]]
            L.append(f"| {r['run']} | {r['changed']} | {r['status']} | {r['epoch']}/{r['total']} | {fmt(r['eta_min'], 0)} | "
                     f"{fmt(r['knn'])} ({r['last_mon_epoch']}) | {fmt(dk)} | {fmt(r['heldout_J'], 4)} | {fmt(r['h_rank'])} | {fmt(r['linear'])} | "
                     f"{r['signal'] or '—'} | {'; '.join(r['flags']) or '—'} | {r['job']} |")
        L.append("")
        traj = [r for r in rs if len(r["knn_traj"]) > 1]
        if traj:
            L += ["kNN trajectories:", ""]
            for r in sorted(traj, key=lambda r: r["run"]):
                L.append(f"- {r['run']}: " + ", ".join(f"ep{e}: {v:.1f}" for e, v in sorted(r["knn_traj"].items())))
            L.append("")
    done = sorted(r["run"] for r in rows if r["status"] != "RUNNING")
    key = f"done={len(done)} mon_evals={sum(r['n_mon'] for r in rows)} flags={sum(1 for r in rows if r['flags'])}"
    L += ["---", f"event_key: {key}", ""]
    return "\n".join(L), key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs"))
    ap.add_argument("--out", default="/home/infres/yinwang/CS_QMI/ssl_pilot/reports/WATCH_status.md")
    ap.add_argument("--stages", default="P24_vcs_cosine_base,P26_vcs_negdetach_base")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    stages = set(a.stages.split(","))
    rows = scan(Path(a.output_root), stages)
    text, key = render(rows, stages)
    Path(a.out).write_text(text)
    if a.print:
        print(text)
    else:
        print(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
