"""P4 aggregation: ``python -m vcs_ssl.summarize --stage P3_pilot --output reports/P4_ssl_pilot_report.md``.

Collects *every* run directory under ``$OUTPUT_ROOT`` for the stage (failed and stopped runs included) and writes a neutral
results table plus a JSON dump.  Observed values only; ``null`` where a number does not exist.  Interpretation (sections D/E
of the report) is written by a human after reading this file, not generated here.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .utils import atomic_write_json, atomic_write_text, read_json, utc_now


def _load(p: Path) -> Any:
    return read_json(p) if p.is_file() else None


def _jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.is_file():
        return []
    import json  # noqa: PLC0415

    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def collect(output_root: Path, stage: str) -> list[dict[str, Any]]:
    rows = []
    for rd in sorted(p for p in output_root.iterdir() if p.is_dir()):
        status = _load(rd / "status.json")
        if status is None or status.get("stage") != stage:
            continue
        summary = _load(rd / "summary.json") or {}
        rm = _load(rd / "run_manifest.json") or {}
        final_evals = sorted((rd / "evaluations").glob("evaluation_epoch_*.json")) if (rd / "evaluations").is_dir() else []
        ev20 = _load(final_evals[-1]) if final_evals else None  # evaluation of the highest evaluated epoch (final checkpoint)
        final_eval_name = final_evals[-1].name if final_evals else None
        ev0 = _load(rd / "evaluations" / "evaluation_initial.json")
        epochs = _jsonl(rd / "logs" / "epochs.jsonl")
        knn_by_epoch = {r["epoch"]: r.get("knn_val_top1_pct") for r in epochs if r.get("knn_val_top1_pct") is not None}
        heldout_by_epoch = {r["epoch"]: r.get("heldout_J") for r in epochs if r.get("heldout_J") is not None}
        last_train = [r for r in epochs if r.get("J_raw") is not None or r.get("nt_xent") is not None or r.get("vicreg_invariance") is not None]
        rows.append({
            "run_id": rd.name, "method": status.get("method"), "seed": status.get("seed"), "status": status.get("status"),
            "failure_reason": status.get("failure_reason"), "completed_epoch": status.get("completed_epoch"), "epochs_intended": status.get("epochs_intended"),
            "optimizer_step": status.get("optimizer_step"), "code_commit": rm.get("code_commit"), "code_dirty": rm.get("code_dirty"),
            "config_hash": rm.get("config_hash"), "split_hash": rm.get("split_hash"), "init_hashes": rm.get("init_hashes"),
            "linear_val_top1_pct": None if ev20 is None else ev20.get("linear_val_top1_pct"),
            "linear_val_ce": None if ev20 is None else ev20.get("linear_val_ce"),
            "linear_val_top1_pct_epoch0": None if ev0 is None else ev0.get("linear_val_top1_pct"),
            "knn_val_top1_pct": None if ev20 is None else ev20.get("knn_val_top1_pct"),
            "knn_val_top1_pct_epoch0": None if ev0 is None else ev0.get("knn_val_top1_pct"),
            "knn_by_epoch": knn_by_epoch, "heldout_J": None if ev20 is None else ev20.get("heldout_J"),
            "heldout_J_sd": None if ev20 is None else ev20.get("heldout_J_sd"), "heldout_J_by_epoch": heldout_by_epoch,
            "heldout_J_epoch0": None if ev0 is None else ev0.get("heldout_J"),
            "h_effective_rank": None if ev20 is None else ev20.get("h_effective_rank"),
            "z_effective_rank": None if ev20 is None else ev20.get("z_effective_rank"),
            "h_effective_rank_epoch0": None if ev0 is None else ev0.get("h_effective_rank"),
            "final_train_J_raw": last_train[-1].get("J_raw") if last_train else None,
            "final_train_R_binary": last_train[-1].get("R_binary") if last_train else None,
            "final_train_nt_xent": last_train[-1].get("nt_xent") if last_train else None,
            "final_train_vicreg": None if not last_train else {k: last_train[-1].get(k) for k in ("vicreg_invariance", "vicreg_variance", "vicreg_covariance")},
            "train_seconds": summary.get("train_seconds"), "eval_seconds_in_training": summary.get("eval_seconds"),
            "eval_seconds_epoch20": None if ev20 is None else ev20.get("eval_seconds"),
            "steady_state_images_per_s": summary.get("steady_state_images_per_s"), "steady_state_views_per_s": summary.get("steady_state_views_per_s"),
            "steady_state_step_seconds": summary.get("steady_state_step_seconds_mean"),
            "peak_allocated_mb": summary.get("peak_allocated_mb"), "peak_reserved_mb": summary.get("peak_reserved_mb"),
            "collapse_suspected": summary.get("collapse_suspected"), "seen_base_images": summary.get("seen_base_images"),
            "critic_params": rm.get("critic_params"), "encoder_params": rm.get("encoder_params"), "projector_params": rm.get("projector_params"),
            "slurm_job_id": rm.get("slurm_job_id"), "hostname": rm.get("hostname"), "run_dir": str(rd),
            "has_epoch0_eval": ev0 is not None, "has_epoch20_eval": ev20 is not None, "final_eval_file": final_eval_name,
        })
    return rows


def fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render(rows: list[dict[str, Any]], stage: str) -> str:
    L = [f"# {stage} — neutral results table (observed values only)", "", f"Generated {utc_now()}. Failed / stopped runs are listed, never dropped.", ""]
    L += ["| run | method | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        hj = "null" if r["heldout_J"] is None else f"{r['heldout_J']:.4f}±{fmt(r['heldout_J_sd'], 4)}"
        L.append(f"| {r['run_id']} | {r['method']} | {r['seed']} | {r['completed_epoch']}/{r['epochs_intended']} | {fmt(r['linear_val_top1_pct'])} | "
                 f"{fmt(r['knn_val_top1_pct'])} | {hj} | {fmt(r['h_effective_rank'])} | {fmt(r['train_seconds'], 0)} | "
                 f"{fmt(r['peak_allocated_mb'], 0)}/{fmt(r['peak_reserved_mb'], 0)} | {r['status']}{' COLLAPSE_SUSPECTED' if r['collapse_suspected'] else ''} |")
    L += ["", "## Epoch-0 (random init, same seed) reference and deltas", "",
          "| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dl = None if r["linear_val_top1_pct"] is None or r["linear_val_top1_pct_epoch0"] is None else r["linear_val_top1_pct"] - r["linear_val_top1_pct_epoch0"]
        dk = None if r["knn_val_top1_pct"] is None or r["knn_val_top1_pct_epoch0"] is None else r["knn_val_top1_pct"] - r["knn_val_top1_pct_epoch0"]
        L.append(f"| {r['run_id']} | {fmt(r['linear_val_top1_pct_epoch0'])} | {fmt(r['linear_val_top1_pct'])} | {fmt(dl)} | {fmt(r['knn_val_top1_pct_epoch0'])} | "
                 f"{fmt(r['knn_val_top1_pct'])} | {fmt(dk)} | {fmt(r['h_effective_rank_epoch0'])} | {fmt(r['h_effective_rank'])} | {fmt(r['heldout_J_epoch0'], 4)} |")
    L += ["", "## kNN trajectory (in-training monitor, selection top-1 %)", ""]
    for r in rows:
        L.append(f"- {r['run_id']}: " + ", ".join(f"ep{e}: {fmt(v)}" for e, v in sorted(r["knn_by_epoch"].items())))
    L += ["", "## Held-out J trajectory (VCS only)", ""]
    for r in rows:
        if r["heldout_J_by_epoch"]:
            L.append(f"- {r['run_id']}: " + ", ".join(f"ep{e}: {fmt(v, 4)}" for e, v in sorted(r["heldout_J_by_epoch"].items())))
    L += ["", "## Final-epoch training objective values (epoch means)", ""]
    for r in rows:
        L.append(f"- {r['run_id']}: J_raw {fmt(r['final_train_J_raw'], 4)}, R_binary {fmt(r['final_train_R_binary'], 4)}, nt_xent {fmt(r['final_train_nt_xent'], 4)}, "
                 f"vicreg {r['final_train_vicreg']}")
    L += ["", "## Cost", "", "| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run_id']} | {fmt(r['steady_state_step_seconds'], 4)} | {fmt(r['steady_state_images_per_s'], 0)} | {fmt(r['steady_state_views_per_s'], 0)} | "
                 f"{fmt(r['train_seconds'], 0)} | {fmt(r['eval_seconds_in_training'], 0)} | {fmt(r['eval_seconds_epoch20'], 0)} | {r['seen_base_images']} | "
                 f"{r['critic_params']} | {r['slurm_job_id']}@{r['hostname']} |")
    L += ["", "## Provenance", ""]
    for r in rows:
        L.append(f"- {r['run_id']}: commit `{r['code_commit']}` dirty={r['code_dirty']}, config `{r['config_hash']}`, split `{r['split_hash']}`, "
                 f"init enc `{(r['init_hashes'] or {}).get('encoder_init_sha256')}`")
    inits = {(r["init_hashes"] or {}).get("encoder_init_sha256") for r in rows}
    L += ["", f"- identical encoder init across runs: {len(inits) == 1 and None not in inits}",
          f"- identical split across runs: {len({r['split_hash'] for r in rows}) == 1}", ""]
    L += ["## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)", "",
          "| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |", "|---|---|---|---|---|---|---|---|"]
    import statistics as st
    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "null"
        return f"{st.mean(vals):.2f} ± {st.stdev(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"
    for m in sorted({r["method"] for r in rows if r["method"]}):
        rs = [r for r in rows if r["method"] == m and r["status"] == "COMPLETED" and r["has_epoch20_eval"]]
        dl = [r["linear_val_top1_pct"] - r["linear_val_top1_pct_epoch0"] for r in rs if r["linear_val_top1_pct"] is not None and r["linear_val_top1_pct_epoch0"] is not None]
        L.append(f"| {m} | {len(rs)} | {agg([r['linear_val_top1_pct'] for r in rs])} | {agg([r['linear_val_top1_pct_epoch0'] for r in rs])} | {agg(dl)} | "
                 f"{agg([r['knn_val_top1_pct'] for r in rs])} | {agg([r['h_effective_rank'] for r in rs])} | {agg([r['heldout_J'] for r in rs])} |")
    L += ["", "## Coverage checks", ""]
    for r in rows:
        L.append(f"- {r['run_id']}: epoch0 eval {r['has_epoch0_eval']}, final eval {r['has_epoch20_eval']} ({r['final_eval_file']}), status {r['status']}, failure {r['failure_reason']}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"))
    args = ap.parse_args(argv)
    if not args.output_root:
        print("OUTPUT_ROOT not set", file=sys.stderr)
        return 2
    rows = collect(Path(args.output_root), args.stage)
    out = Path(args.output)
    atomic_write_text(out, render(rows, args.stage))
    atomic_write_json(out.with_suffix(".json"), rows)
    print(f"{len(rows)} runs summarized -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
