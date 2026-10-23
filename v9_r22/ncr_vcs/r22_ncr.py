"""R22-B unit driver: the strong NCR host (NCR_A of B1: official NCR co-training + V9 anchor term, same schedule, same DEV-only adapter)
with the online VCS posterior (arm vcs) or the matched balanced-BCE posterior (arm bce) in place of the per-epoch GMM clean probability.

    python r22_ncr.py run --arm vcs|bce --schedule chk|full --seed S --unit <new unit dir>
    python r22_ncr.py table [--root <R22 output root>] [--json <out>]

Steps (B1 driver v9_n2p/ab/b0/n2p_baseline.py reused; only prepare and the trainer arguments differ):
  gate     full schedule: the NCR B0 gate B0GATE_NCR_A_s917_r04 must be B0_PASS with the same frozen config / base patch layers / forward
           and guard scripts as this tree (n2p_baseline.check_gate) -- the base host is exactly NCR_A's
  prepare  official clone + ncr_anchor_assisted.diff + ncr_n2p_devonly.diff + ncr_r22_posterior.diff; V9 export (observed mapping, anchor
           mask) and the R22 anchor-role vector, both under the train read guard
  train    patched official trainer with --r22_posterior <arm>; posterior ledger in official_out/r22 (moved with an unfinished attempt)
  forward / score / select   unchanged B1 steps (full-DEV 1014 x 5070 per candidate, package scorer, best full-DEV rSum)
  summary  R22_RESULT.json: selected candidate, six recalls, NCR_A same-seed reference, posterior / fallback ledger summary
chk = smoke: both networks start from the same seed's B1 NCR_A post-warm-up checkpoint (warmup_04), 2 co-training epochs, 2 candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "v9_n2p/ab/b0"))
import n2p_baseline as nb  # noqa: E402

nb.REPO = REPO                                                   # patches, export script and scorer from this worktree
nb.PKG_EVIDENCE = REPO / "v9_n2p/VCS_V9_N2P_V10_Review_Server_Package_v1_20261020/v9/evidence"
B1_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs_v9_n2p_20261020")
GATE = B1_ROOT / "B0GATE_NCR_A_s917_r04/B0_DEV_ADAPTER_REPORT.json"
R22_OUT = Path(os.environ.get("R22_OUT", "/home/infres/yinwang/CS_QMI/outputs_v9_r22/ncr_vcs"))
ARMS = {"vcs": "NCR_A+R22_VCS_posterior", "bce": "NCR_A+R22_BCE_posterior"}
SEEDS = (901, 902, 903)
_ORIG_TRAINER_ARGS = nb.trainer_args


def unit_name(arm, schedule, seed):
    return f"R22B_{'CHK_' if schedule == 'chk' else ''}f30k_r04_s{seed}_{ARMS[arm].replace('+', '_')}"


def effective_frozen(schedule):
    frozen = nb.jload(nb.HERE / "B0_FROZEN.json")
    if schedule == "chk":                                        # smoke only; the full schedule keeps the frozen config unchanged
        frozen["schedules"]["ncr"]["chk"] = {"warmup_epoch": 5, "num_epochs": 2, "lr_update": 30,
                                             "warmup_model_path": str(B1_ROOT / "V9N2P_f30k_r04_s{seed}_NCR_A/candidates/warmup_04.pth.tar")}
        frozen["candidate_maps"]["ncr"]["chk"] = [{"order": k, "candidate_id": f"main:{k}", "phase": "main", "epoch": k, "file": f"main_{k:02d}.pth.tar"}
                                                  for k in range(2)]
    return frozen


def prepare(a, frozen, unit):
    cfg = {"baseline": "ncr", "arm": a.arm, "schedule": a.schedule, "seed": a.seed, "rho": a.rho}
    if (unit / "unit.json").exists():
        old = nb.jload(unit / "unit.json")
        if {k: old.get(k) for k in cfg} != cfg:
            raise SystemExit(f"{unit} belongs to another configuration: {old}")
        return old
    if unit.exists() and any(unit.iterdir()):
        raise SystemExit(f"refusing to use a non-empty directory without unit.json: {unit}")
    unit.mkdir(parents=True, exist_ok=True)
    shutil.copytree(nb.HERE, unit / "adapter", ignore=shutil.ignore_patterns("__pycache__"))
    nb.jdump(unit / "adapter/B0_FROZEN.json", frozen)
    for p in (REPO / "v9_n2p/ab/patches").glob("*.diff"):
        shutil.copy(p, unit / "adapter" / p.name)
    shutil.copytree(HERE, unit / "adapter_r22", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    code = unit / "code/ncr"
    shutil.copytree(frozen["paths"]["clones"]["ncr"], code,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "runs", "output", "noise_index", "*.png", "cc152k"))
    patches = [REPO / "v9/patches/ncr_anchor_assisted.diff", unit / "adapter/ncr_n2p_devonly.diff", unit / "adapter_r22/ncr_r22_posterior.diff"]
    for p in patches:
        r = subprocess.run(["patch", "-p1", "-s", "-i", str(p)], cwd=code, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"patch failed: {p}\n{r.stdout}{r.stderr}")
    if nb.sha(code / "NCR/r22_posterior.py") != nb.sha(HERE / "r22_posterior.py"):
        raise SystemExit("the patch layer's r22_posterior.py differs from v9_r22/ncr_vcs/r22_posterior.py: regenerate the patch")
    view = unit / "data_view/f30k_precomp"
    view.mkdir(parents=True)
    for f in nb.DATA_FILES:
        (view / f).symlink_to(Path(frozen["paths"]["data_dir"]) / f)
    shared = frozen["paths"]["shared_stage"].format(rho=f"{a.rho:g}", seed=a.seed)
    for trace, script, args, log in (
            ("export_trace.jsonl", REPO / "scripts/v9/n2_official.py",
             ["export", "--shared", shared, "--data-root", str(view), "--out", str(unit / "v9_export")], "export.log"),
            ("export_roles_trace.jsonl", HERE / "r22_export_roles.py",
             ["--shared", shared, "--v9-export", str(unit / "v9_export"), "--out", str(unit / "r22_export")], "export_roles.log")):
        r = subprocess.run(nb.guard_cmd(nb.PY_DRIVER, unit, trace, "train", frozen, script, args), capture_output=True, text=True)
        (unit / log).write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"{script.name} failed; see {log}")
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "v9_r22", "v9_n2p/ab"], capture_output=True, text=True).stdout.strip()
    info = {**cfg, "unit": str(unit), "method": ARMS[a.arm], "host_method": "NCR_A", "prepared_utc": nb.utc(), "adapter_git_head": git,
            "adapter_tree_dirty": bool(dirty), "shared_stage": shared, "patch_sha256": {p.name: nb.sha(p) for p in patches},
            "effective_code_sha256": {str(p.relative_to(code)): nb.sha(p) for p in sorted(code.rglob("*.py"))},
            "export_manifest": nb.jload(unit / "v9_export/v9_export_manifest.json"),
            "r22_roles_manifest": nb.jload(unit / "r22_export/r22_roles_manifest.json"),
            "schedule_values": frozen["schedules"]["ncr"][a.schedule], "lr_multiplier": frozen["lr_multiplier"]["value"],
            "frozen_config_sha256": nb.sha(unit / "adapter/B0_FROZEN.json"),
            "gate_report": {"path": str(GATE), "sha256": nb.sha(GATE)} if a.schedule == "full" else None}
    nb.jdump(unit / "unit.json", info)
    return info


def trainer_args(a, frozen, unit):
    script, args = _ORIG_TRAINER_ARGS(a, frozen, unit)
    args = args + ["--r22_posterior", a.arm, "--r22_roles_file", str(unit / "r22_export/r22_anchor_roles.npy"),
                   "--r22_log", str(unit / "official_out/r22")]
    sched = frozen["schedules"]["ncr"][a.schedule]
    if "warmup_model_path" in sched:
        args += ["--warmup_model_path", sched["warmup_model_path"].format(seed=a.seed)]
    return script, args


nb.trainer_args = trainer_args                                   # n2p_baseline.train builds its command through this name


def ledger(unit):
    p = unit / "official_out/r22/posterior.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    per = []
    for r in rows:
        for m in ("A", "B"):
            x = r[m]
            per.append({"epoch": r["epoch"], "model": m, "deployed": x["deployed"], "status": x["rho"].get("status"),
                        "rho_raw": x["rho"].get("raw"), "gap": x["rho"].get("gap"), "c_U": x["rho"].get("c"),
                        "critic_a": (x.get("critic") or {}).get("a"), "critic_b": (x.get("critic") or {}).get("b"),
                        "select_J": (x.get("critic") or {}).get("select_common_J"), "w_mean": x.get("w_U", {}).get("mean"),
                        "w_clean_frac": x.get("w_U", {}).get("frac_gt_0.5"), "gmm_clean_frac": x["gmm_U"].get("frac_gt_0.5"),
                        "split_agree": x.get("agreement", {}).get("split_agree_frac"), "fit_seconds": x.get("fit_seconds"),
                        "q_score_seconds": x.get("q_score_seconds")})
    return {"ledger": str(p), "n_epoch_models": len(per), "n_fallbacks": sum(not r["deployed"] for r in per),
            "fallbacks": [{k: r[k] for k in ("epoch", "model", "status", "rho_raw", "gap")} for r in per if not r["deployed"]], "per_epoch": per}


def summary(a, unit):
    res = nb.jload(unit / "RESULT.json")
    ref_p = B1_ROOT / f"V9N2P_f30k_r04_s{a.seed}_NCR_A/RESULT.json"
    ref = nb.jload(ref_p)["primary_fullDEV_best"] if ref_p.exists() else None
    best = res["primary_fullDEV_best"]
    out = {"schema": "r22b-unit-result-v1", "status": "PILOT_DEVELOPMENT (DEV only, no TEST read)", "unit": str(unit), "arm": a.arm,
           "method": ARMS[a.arm], "schedule": a.schedule, "seed": a.seed, "rho": a.rho,
           "selected": {k: best[k] for k in ("candidate_id", "fullDEV", "checkpoint_sha256")},
           "NCR_A_same_seed": None if ref is None else {"result": str(ref_p), "candidate_id": ref["candidate_id"], "fullDEV": ref["fullDEV"],
                                                         "hardware": nb.jload(ref_p.parent / "train_receipt.json").get("gpu")},
           "rsum_minus_NCR_A": None if ref is None else best["fullDEV"]["rsum"] - ref["fullDEV"]["rsum"],
           "posterior": ledger(unit), "costs": res["costs"], "read_trace": {k: {kk: v[kk] for kk in ("denied", "test_or_eval_only_opened")}
                                                                             for k, v in res["read_trace"].items()}}
    out["posterior"]["roles_trace"] = nb.trace_summary(unit, "export_roles_trace.jsonl")
    nb.jdump(unit / "R22_RESULT.json", out)
    return 0


def run(a):
    frozen = effective_frozen(a.schedule)
    if a.schedule not in frozen["candidate_maps"]["ncr"]:
        raise SystemExit("unknown schedule " + a.schedule)
    if a.schedule == "full":
        a.gate_report = str(GATE)
        if nb.check_gate(a, frozen):                             # base host identical to the B0_PASS NCR gate, before anything is written
            return 7
    unit = Path(a.unit)
    print(f"[r22b] {nb.utc()} {a.arm} {a.schedule} s{a.seed} rho{a.rho} host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID')}", flush=True)
    steps = [("prepare", lambda: (prepare(a, frozen, unit), 0)[1]), ("train", lambda: nb.train(a, frozen, unit)),
             ("forward", lambda: nb.forward(a, frozen, unit)), ("score", lambda: nb.score(a, frozen, unit)),
             ("select", lambda: nb.select(a, frozen, unit)), ("summary", lambda: summary(a, unit))]
    for name, fn in steps:
        rc = fn()
        print(f"[r22b] {nb.utc()} {name} rc={rc}", flush=True)
        if rc:
            return rc
    return 0


def table(a):
    rows = []
    for seed in SEEDS:
        ref_p = B1_ROOT / f"V9N2P_f30k_r04_s{seed}_NCR_A/RESULT.json"
        if ref_p.exists():
            b = nb.jload(ref_p)["primary_fullDEV_best"]
            rows.append({"method": "NCR_A (B1)", "seed": seed, "candidate": b["candidate_id"], **b["fullDEV"], "fallbacks": None,
                         "gpu": nb.jload(ref_p.parent / "train_receipt.json").get("gpu")})
        for arm in ARMS:
            u = Path(a.root) / unit_name(arm, "full", seed)
            if (u / "R22_RESULT.json").exists():
                r = nb.jload(u / "R22_RESULT.json")
                rows.append({"method": ARMS[arm], "seed": seed, "candidate": r["selected"]["candidate_id"], **r["selected"]["fullDEV"],
                             "fallbacks": f"{r['posterior']['n_fallbacks']}/{r['posterior']['n_epoch_models']}",
                             "gpu": (r["costs"].get("train") or {}).get("gpu")})
            else:
                rows.append({"method": ARMS[arm], "seed": seed, "candidate": "(not finished)"})
    cols = list(nb.RECALLS) + ["rsum"]
    print(f"{'method':26s} {'seed':>4s} {'cand':>8s} " + " ".join(f"{c:>7s}" for c in cols) + "  fallbacks  gpu")
    for r in rows:
        vals = " ".join(f"{r[c]:7.2f}" if c in r else f"{'':>7s}" for c in cols)
        print(f"{r['method']:26s} {r['seed']:4d} {r['candidate']:>8s} {vals}  {str(r.get('fallbacks') or ''):9s}  {r.get('gpu') or ''}")
    for m in ["NCR_A (B1)"] + list(ARMS.values()):
        got = {r["seed"]: r["rsum"] for r in rows if r["method"] == m and "rsum" in r}
        ref = {r["seed"]: r["rsum"] for r in rows if r["method"] == "NCR_A (B1)" and "rsum" in r}
        if got:
            d = [got[s] - ref[s] for s in got if s in ref]
            print(f"{m:26s} mean rSum {sum(got.values()) / len(got):.2f} over {len(got)} seeds"
                  + ("" if m.startswith("NCR_A") or not d else f"; minus NCR_A per seed {['%+.2f' % x for x in d]}, mean {sum(d) / len(d):+.2f}"))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--schedule", required=True, choices=["chk", "full"])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rho", type=float, default=0.4)
    r.add_argument("--unit", default="")
    t = sub.add_parser("table")
    t.add_argument("--root", default=str(R22_OUT))
    t.add_argument("--json", default="")
    a = ap.parse_args()
    if a.cmd == "table":
        return table(a)
    a.baseline, a.gate_report = "ncr", ""
    a.unit = a.unit or str(R22_OUT / unit_name(a.arm, a.schedule, a.seed))
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
