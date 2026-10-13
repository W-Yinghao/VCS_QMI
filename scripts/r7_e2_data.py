"""r7 E2 data stage and checks (CPU, main env, python -I; numpy only — torch only for the frozen10k PCA comparison).  E2 reuses D1-FULL's frozen
manifest (outputs/R7_D1_FULL/data: sha256 c4404f7b…, frozen by D1-FULL's data stage; FIT_FULL 70,256 images / 280,787 questions; CAL and DEV are
the frozen D1 sets) and R7-40a's Qwen teacher records of the frozen D1 FIT and CAL (FIT-000..039 = the first 10,000 FIT_FULL images; CAL-000..007).
DEV is never planned, computed, copied or read here (the sealed Qwen DEV teacher records stay R7-40c's).
  precheck [--out O]     : source verification + the static reuse check of the R7-40a CAL / FIT records (+ the GPU replay result if present) + plan
                           counts for both reuse outcomes -> O/precheck.json (dry run: nothing copied, nothing planned)
  data --lines L         : the driver's data stage (refuses without 'r7_e2a_staged_freeze'): verify -> static check -> replay file (required) -> E2/data
                           (byte copies of D1-FULL's manifest + meta; E2's shard plan from D1-FULL's make_plan, imported unchanged: FIT bundles of the
                           shards not reused) -> verified copies of the reused records into E2/teacher (CAL always; FIT-000..039 iff static + replay
                           pass, else they are recomputed by 5 more bundles) -> teacher lines -> E2/data/data_ready.json (written last)
  cachecheck --basis B   : the cache job's outputs against E2's manifest (role image lists and order, token sums, offsets, question ids, PCA image
                           count for B, merged grids within the selectors' 64 × 64 tables; frozen10k: PCA tensors vs R7-40b's) -> E2/cache/e2_cache_check.json
  reuserows --rows R     : R7-40c's selector-independent rows (full, uniform grid) against E2's run: identical inputs (set sha256 in every shard marker
                           = the file = the D1-30 build summary; L40S in every marker; runner code blobs at HEAD = those of R7-40c's run, commit 4174400;
                           R7-40c complete) and E2's in-job full row equal to R7-40c's full row on every item -> E2/eval/reuse/reuse_check.json; with
                           R = selectors and a pass, R7-40c's four uniform-grid rows are copied (sha256-checked, never overwriting) into E2's pred dir
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_30_run as RUN  # noqa: E402  (frozen D1-30: prediction-file layout)
import r7_d1_40_masks as M4  # noqa: E402  (frozen R7-40a: orderings, budgets)
import r7_d1_40c_plan as PL  # noqa: E402  (frozen R7-40c: benchmarks, sharding)
import r7_d1_data as DA  # noqa: E402  (frozen D1 data helpers)
import r7_d1full_data as FD  # noqa: E402  (D1-FULL's data module: make_plan, shard_plan, parse_rows, copy rule; imported unchanged)
import r7_d1_teacher as TE  # noqa: E402  (frozen D1-10 order encodings)
import r7_e2_plan as EP  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
D1 = OUT / "R7_D1"
DF = OUT / "R7_D1_FULL"
E2 = EP.E2
R40 = D1 / "r7_40"
R40C = R40 / "eval_40c"
D30_SETS = D1 / "d1_30" / "sets"
FROZEN_SHA = FD.FROZEN_MANIFEST_SHA256                                                  # 32c72126… (R7-40a's manifest)
FULL_SHA = "c4404f7be19640385072c39e27ce026589acbfcb1187aedffa1bd58c638fb21e"         # D1-FULL's frozen manifest (job 1035862, frozen 2026-10-12T15:58:50Z)
PROTOCOL = REPO / "reports" / "R7" / "R7_D1_PROTOCOL_FROZEN_20261012.yaml"
FREEZE_KEY = "r7_e2a_staged_freeze"                                                   # the teacher stage (E2a) is frozen first, alone
TEACHER_CODE = ("scripts/r7_d1_40_teacher.py", "scripts/r7_d1_40_masks.py", "scripts/r7_d1_40_host.py", "scripts/r7_d1_teacher.py", "scripts/r7_d1_masks.py",
                "scripts/r7_vl_common.py")
TEACHER_DEFAULTS = {"images_per_chunk": 16, "max_tokens": 16384, "max_seqs": 64}       # r7_d1_40_teacher.py bundle defaults (every R7-40a bundle)
TEACHER_GPU = "NVIDIA L40S"                                                            # D40a-4: every R7-40a record
QUALITY_GPU = "NVIDIA L40S"                                                            # R7-40c decision 7: every R7-40c quality shard
QWEN_ENV_FREEZE = Path("/projects/EEG-foundation-model/yinghao/envs/vl_baselines/FREEZE.txt")
QWEN_MODEL = Path("/projects/EEG-foundation-model/yinghao/models/Qwen2.5-VL-7B-Instruct")
RUNNER_CODE = ("scripts/r7_d1_40c_run.py", "scripts/r7_d1_40c_plan.py", "scripts/r7_d1_40_host.py", "scripts/r7_d1_40_masks.py", "scripts/r7_d1_40_selectors.py",
               "scripts/r7_d1_selectors.py", "scripts/r7_d1_30_run.py", "scripts/r7_d1_masks.py", "scripts/r7_vl_common.py")
R740C_RUN_COMMIT = "4174400"           # execution note 40c_3: the tracked R7-40c code is unchanged since 4174400 for every relaunched quality unit
GRID_ROWS = tuple(f"static:grid:{k}" for k in M4.FKEYS)
jread, jwrite = FD.jread, FD.jwrite


def utc(ts=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def require_freeze(protocol):
    p = Path(protocol)
    if not p.exists() or not re.search(rf"^{FREEZE_KEY}\s*:", p.read_text(), re.M):
        raise SystemExit(f"the E2 teacher stage is not frozen: top-level key '{FREEZE_KEY}' absent in {p}")


def git(*args, repo=REPO):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------------------------------------------------------------------ sources
def verify_sources(df_data: Path, d1_data: Path, full_sha=FULL_SHA, frozen_sha=FROZEN_SHA):
    """D1-FULL's manifest is frozen with the expected sha256 and verified by D1-FULL; R7-40a's (frozen D1) manifest is its byte prefix; the
    frozen CAL / FIT shards (R7-40a's) have the same image lists in D1-FULL's plan; D1-FULL's plan equals make_plan(…, reuse_fit=True); no DEV
    shard anywhere in D1-FULL's plan.  Returns (record, manifest rows, frozen plan)."""
    prob = []; meta = jread(df_data / "d1_manifest_meta.json"); fb = (df_data / "d1_manifest.jsonl").read_bytes(); zb = (d1_data / "d1_manifest.jsonl").read_bytes()
    s_full, s_froz = FD.sha_bytes(fb), FD.sha_bytes(zb)
    if meta.get("frozen") is not True:
        prob.append("the D1-FULL manifest is not frozen")
    if s_full != meta.get("manifest_sha256") or (full_sha and s_full != full_sha):
        prob.append(f"D1-FULL manifest sha256 {s_full[:12]} differs from its meta / the expected {str(full_sha)[:12]}")
    if frozen_sha and s_froz != frozen_sha:
        prob.append(f"frozen D1 manifest sha256 {s_froz[:12]} differs from {frozen_sha[:12]}")
    prefix = fb.startswith(zb)
    if not prefix:
        prob.append("the frozen D1 manifest is not a byte prefix of D1-FULL's")
    vf = df_data / "d1full_verify.json"; ver = jread(vf) if vf.exists() else {}
    if ver.get("ok") is not True or ver.get("d1full_manifest_sha256") != s_full:
        prob.append("D1-FULL's own verification (d1full_verify.json) is missing, failing or for another manifest")
    rows = FD.parse_rows(fb); fplan = jread(d1_data / "d1_shards.json"); dplan = jread(df_data / "d1_shards.json")
    reused = [s for s in fplan["shards"] if s.startswith(("CAL-", "FIT-"))]
    diff = [s for s in reused if dplan["shards"].get(s) != fplan["shards"][s]]
    if diff:
        prob.append(f"frozen CAL / FIT shards differ in D1-FULL's plan: {diff[:5]}")
    if any(s.startswith("DEV-") for s in dplan["shards"]) or any(s.startswith("DEV-") for v in dplan["bundles"].values() for s in v):
        prob.append("D1-FULL's plan holds a DEV shard")
    mine = FD.make_plan(rows, fplan, True, fplan["images_per_shard"], dplan["shards_per_bundle"])
    for k in ("shards", "bundles", "questions_per_shard"):
        if mine[k] != dplan[k]:
            prob.append(f"D1-FULL's plan '{k}' differs from make_plan(reuse_fit=True)")
    counts = {role: {"images": len({r["image_id"] for r in rows if r["role"] == role}), "questions": sum(r["role"] == role for r in rows)} for role in ("FIT", "CAL")}
    for role in ("FIT", "CAL"):
        mc = meta.get("counts", {}).get(role, {})
        if (mc.get("images"), mc.get("questions")) != (counts[role]["images"], counts[role]["questions"]):
            prob.append(f"{role} counts {counts[role]} differ from the meta {mc}")
    frozen_fit = [s for s in reused if s.startswith("FIT-")]; frozen_imgs = [i for s in frozen_fit for i in fplan["shards"][s]]
    first = FD.role_images(rows, "FIT")[:len(frozen_imgs)]        # manifest order = the cache's FIT image order (r7_d1_40_cache.images_of)
    rec = {"ok": not prob, "problems": prob, "d1full_manifest_sha256": s_full, "frozen_manifest_sha256": s_froz, "frozen_is_byte_prefix": prefix,
           "d1full_verify_ok": ver.get("ok"), "counts": counts, "class_weights_from_FIT": meta.get("class_weights_from_FIT"), "reused_shards": reused,
           "frozen_fit_images": len(frozen_imgs), "frozen_fit_are_first_fit_images": frozen_imgs == first}
    if not rec["frozen_fit_are_first_fit_images"]:
        prob.append("the frozen FIT images are not the first FIT_FULL images in manifest order"); rec["ok"] = False
    return rec, rows, fplan


# ------------------------------------------------------------------------------------------------------------------------------ static reuse check
def check_records40(recs, mrows, done, jsonl_bytes):
    """Static checks of one R7-40a shard's records against the manifest rows of its images (the D1-FULL manifest; byte-prefix rows)."""
    exp = {r["qid"]: r for r in mrows}; prob = []
    if FD.sha_bytes(jsonl_bytes) != done.get("jsonl_sha256"):
        prob.append("jsonl sha256 differs from the done record")
    if done.get("present") != len(exp) or done.get("expected") != len(exp) or done.get("duplicates") or done.get("skipped_images"):
        prob.append(f"done record {({k: done.get(k) for k in ('expected', 'present', 'duplicates')})}, skipped {len(done.get('skipped_images') or [])}")
    qids = [r["qid"] for r in recs]
    if len(qids) != len(set(qids)) or set(qids) != set(exp):
        prob.append(f"record qids differ from the manifest ({len(set(qids))} vs {len(exp)}; duplicates {len(qids) - len(set(qids))})")
    n_dig = 0
    for r in recs:
        m = exp.get(r["qid"])
        if m is None:
            continue
        if (r["image_id"], r["C"], r["role"], r.get("structural"), r.get("semantic")) != (m["image_id"], m["C"], m["role"], m.get("structural"), m.get("semantic")):
            prob.append(f"{r['qid']}: record fields differ from the manifest"); continue
        rows_, cols_ = (int(x) for x in r["grid"]); n = int(r["n_vis"])
        if n != rows_ * cols_ or r["budgets"] != M4.budgets(n):
            prob.append(f"{r['qid']}: grid / budgets inconsistent"); continue
        bad = False
        for mech, (order, logp) in (("random", M4.order_random(r["qid"], rows_, cols_)), ("grid", M4.order_grid(r["qid"], rows_, cols_))):
            if TE.digest(order) != r["order_digest"][mech] or any(r["logp"][mech][k] != logp[k] for k in M4.FKEYS):
                prob.append(f"{r['qid']}: {mech} ordering / log-prob differs from the current mask code"); bad = True; break
        q = TE.unb64(r["qsim_order"])
        if len(q) != n or len(set(q.tolist())) != n or TE.digest(q) != r["order_digest"]["qsim"]:
            prob.append(f"{r['qid']}: qsim ordering is not the recorded permutation"); bad = True
        vals = [r["full"], r["text"]] + [r["masked"][mm][k] for mm in M4.MECHANISMS for k in M4.FKEYS]
        if set(r["masked"]) != set(M4.MECHANISMS) or not all(len(v) == 2 and np.isfinite(v).all() and 0 <= min(v) and sum(v) <= 1 + 1e-6 for v in vals):
            prob.append(f"{r['qid']}: probability fields malformed"); bad = True
        n_dig += not bad
    return {"records": len(recs), "expected": len(exp), "digest_checked": n_dig, "gpu": done.get("gpu"), "problems": prob[:10], "n_problems": len(prob), "ok": not prob}


def code_identity40(run_start_ts: float, files=TEACHER_CODE):
    """The teacher code E2 runs (working tree) is the code that wrote the R7-40a records: clean against HEAD, last commit before the first
    R7-40a shard started (minus the 10-minute host load)."""
    out = {}
    for f in files:
        blob = git("hash-object", f); head = git("rev-parse", f"HEAD:{f}"); last = git("log", "-1", "--format=%ct %h", "--", f).split()
        out[f] = {"blob": blob, "head_blob": head, "clean": bool(blob) and blob == head and git("status", "--porcelain", "--", f) == "",
                  "last_commit": last[1] if len(last) > 1 else None, "last_commit_utc": utc(int(last[0])) if last else None,
                  "before_run": bool(last) and int(last[0]) < run_start_ts - 600}
    return out, all(v["clean"] and v["before_run"] for v in out.values())


def env_identity40(run_start_ts: float, freeze=QWEN_ENV_FREEZE, model=QWEN_MODEL):
    paths = [Path(freeze)] + (sorted(p for p in Path(model).glob("*") if p.is_file()) if Path(model).exists() else [])
    rec = {str(p): {"mtime_utc": utc(p.stat().st_mtime), "before_run": p.stat().st_mtime < run_start_ts} for p in paths if p.exists()}
    return {"files": rec, "freeze_txt_sha256": DA.sha256_file(freeze) if Path(freeze).exists() else None}, bool(rec) and all(v["before_run"] for v in rec.values())


def settings_check40(teacher_src=REPO / "scripts" / "r7_d1_40_teacher.py", r40_wrapper=REPO / "slurm" / "r7_d1_40_gpu.sbatch",
                     e2_wrapper=REPO / "slurm" / "r7_e2_gpu.sbatch"):
    """R7-40a bundles ran 'r7_d1_40_gpu.sbatch bundle bNNN' with the teacher's defaults; E2's bundle line passes no chunk / batch option
    either, so both use the same defaults (16 images per chunk, prefill batches <= 16,384 tokens / 64 sequences)."""
    src = Path(teacher_src).read_text(); defaults = {}
    for opt, key in (("--images-per-chunk", "images_per_chunk"), ("--max-tokens", "max_tokens"), ("--max-seqs", "max_seqs")):
        m = re.search(re.escape(f'add_argument("{opt}", type=int, default=') + r"(\d+)", src); defaults[key] = int(m.group(1)) if m else None
    lines = {}
    for name, w in (("r7_40a_wrapper", r40_wrapper), ("e2_wrapper", e2_wrapper)):
        bl = [x for x in Path(w).read_text().splitlines() if x.strip().startswith("bundle)")] if Path(w).exists() else []
        lines[name] = len(bl) == 1 and "r7_d1_40_teacher.py bundle" in bl[0] and not any(o in bl[0] for o in ("--images-per-chunk", "--max-tokens", "--max-seqs", "--allow-dev"))
    return {"teacher_defaults": defaults, "expected": TEACHER_DEFAULTS, **lines, "ok": defaults == TEACHER_DEFAULTS and all(lines.values())}


def reuse_check40(rows, fplan, r40_teacher: Path, replay=None, code=True, env=True, settings=True):
    """Static reuse check of R7-40a's CAL / FIT records (never DEV).  FIT-000..039 are reused iff every static check passes AND the GPU replay
    (smoke) passes; otherwise they are recomputed by E2 bundles.  CAL-000..007 are always R7-40a's records (R7-40b's selection set and the
    pre-registered CAL readout); a failed CAL check stops the driver (coordinator decision)."""
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image_id"], []).append(r)
    shards = [s for s in fplan["shards"] if s.startswith(("CAL-", "FIT-"))]; res = {"shards": {}}; starts = []
    for s in shards:
        done_p, rec_p = Path(r40_teacher) / f"{s}.done.json", Path(r40_teacher) / f"{s}.jsonl"
        if not (done_p.exists() and rec_p.exists()):
            res["shards"][s] = {"ok": False, "problems": ["missing R7-40a record or done file"], "jsonl_sha256": None}; continue
        done = jread(done_p); b = rec_p.read_bytes(); recs = FD.parse_rows(b)
        e = check_records40(recs, [q for i in fplan["shards"][s] for q in by_img.get(i, [])], done, b)
        e["jsonl_sha256"] = done.get("jsonl_sha256")
        if done.get("gpu") != TEACHER_GPU:
            e["problems"].append(f"made on {done.get('gpu')}, not {TEACHER_GPU} (D40a-4)"); e["ok"] = False
        res["shards"][s] = e; starts.append(done_p.stat().st_mtime - float(done.get("seconds", 0.0)))
    run_start = min(starts) if starts else 0.0; res["r7_40a_first_shard_start_utc"] = utc(run_start)
    res["code"], code_ok = code_identity40(run_start) if code else ({}, True)
    res["env"], env_ok = env_identity40(run_start) if env else ({}, True)
    res["settings_check"] = settings_check40() if settings else {"ok": True, "skipped": True}
    static_extra = code_ok and env_ok and res["settings_check"]["ok"]
    rp = jread(replay) if replay and Path(replay).exists() else None
    res["replay"] = {"file": str(replay) if replay else None, "pass": None if rp is None else bool(rp.get("pass")),
                     "summary": None if rp is None else {k: rp.get(k) for k in ("gpu", "bit_identical", "shards_checked", "records", "max_abs_dP_yes", "qsim_orders_identical")}}
    fit_ok = all(v["ok"] for s, v in res["shards"].items() if s.startswith("FIT-")); cal_ok = all(v["ok"] for s, v in res["shards"].items() if s.startswith("CAL-"))
    res["code_ok"], res["env_ok"] = code_ok, env_ok
    res["fit"] = {"shards": sum(s.startswith("FIT-") for s in res["shards"]), "records": sum(v.get("records", 0) for s, v in res["shards"].items() if s.startswith("FIT-")),
                  "static_ok": fit_ok and static_extra}
    res["cal"] = {"shards": sum(s.startswith("CAL-") for s in res["shards"]), "records": sum(v.get("records", 0) for s, v in res["shards"].items() if s.startswith("CAL-")),
                  "ok": cal_ok and static_extra}
    res["gpus"] = dict(Counter(v.get("gpu") for v in res["shards"].values()))
    res["fit_reuse"] = bool(res["fit"]["static_ok"] and res["replay"]["pass"])
    res["rule"] = ("FIT-000..039 reused iff qids / manifest rows / mask-seed digests and log-probs / record sha256 / L40S / code / env / settings are "
                   "identical AND the GPU replay is bit-identical on an L40S; otherwise recomputed by E2 bundles.  CAL-000..007 are always R7-40a's records.")
    return res


def plan_counts(rows, fplan, shards_per_bundle):
    dry = {}
    for reuse in (True, False):
        pl = FD.make_plan(rows, fplan, reuse, fplan["images_per_shard"], shards_per_bundle); cs = pl["d1full"]["computed_shards"]
        dry["reuse_fit" if reuse else "recompute_fit"] = {"shards": len(pl["shards"]), "bundles": len(pl["bundles"]), "computed_shards": len(cs),
                                                          "computed_images": sum(len(pl["shards"][s]) for s in cs),
                                                          "computed_questions": sum(pl["questions_per_shard"][s] for s in cs),
                                                          "reused_shards": len(pl["d1full"]["reused_shards"])}
    return dry


def cmd_precheck(a):
    """Pre-freeze dry run: verify the sources, statically check R7-40a's CAL / FIT records, count both plan outcomes; nothing copied."""
    v, rows, fplan = verify_sources(Path(a.df_data), Path(a.d1_data), a.full_sha, a.frozen_sha)
    res = reuse_check40(rows, fplan, Path(a.r40_teacher), a.replay, code=not a.no_code, env=not a.no_code, settings=not a.no_code)
    spb = jread(Path(a.df_data) / "d1_shards.json")["shards_per_bundle"]
    rec = {"utc": utc(), "verify": v, "reuse": {k: res[k] for k in ("fit", "cal", "fit_reuse", "replay", "gpus", "code_ok", "env_ok", "settings_check", "r7_40a_first_shard_start_utc", "rule")},
           "shard_problems": {s: e["problems"] for s, e in res["shards"].items() if not e["ok"]}, "code": res["code"], "env": res["env"],
           "plan_dryrun": plan_counts(rows, fplan, spb), "outcome_now": "reuse_fit" if res["fit_reuse"] else ("reuse_fit if the GPU replay passes" if res["fit"]["static_ok"] else "recompute_fit")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); jwrite(out / "precheck.json", rec)
    print(json.dumps({k: rec[k] for k in ("verify", "reuse", "shard_problems", "plan_dryrun", "outcome_now")}, indent=1, default=str)[:20000])
    return 0 if v["ok"] and res["cal"]["ok"] else 3


# ------------------------------------------------------------------------------------------------------------------------------ data stage
def copy_same(src: Path, dst: Path):
    """Byte copy (sha256-checked); an existing identical file is kept, an existing different file is never overwritten."""
    want = DA.sha256_file(src)
    if dst.exists():
        if DA.sha256_file(dst) != want:
            raise SystemExit(f"{dst} exists and differs from {src}; not overwritten")
        return False
    tmp = dst.with_name(dst.name + ".tmp"); tmp.write_bytes(src.read_bytes())
    if DA.sha256_file(tmp) != want:
        tmp.unlink(); raise SystemExit(f"copy of {src} does not match its sha256")
    tmp.replace(dst); return True


def cmd_data(a):
    """The driver's data stage (CPU job): verify -> static reuse check -> replay -> E2/data -> verified copies -> teacher lines -> data_ready.json."""
    require_freeze(a.protocol)
    root = Path(a.root); dd, td = root / "data", root / "teacher"; dd.mkdir(parents=True, exist_ok=True); td.mkdir(parents=True, exist_ok=True)
    v, rows, fplan = verify_sources(Path(a.df_data), Path(a.d1_data), a.full_sha, a.frozen_sha); jwrite(dd / "e2_verify.json", v)
    if not v["ok"]:
        raise SystemExit(f"source verification failed: {v['problems'][:3]}")
    if not (a.replay and Path(a.replay).exists()):
        raise SystemExit(f"the GPU smoke's replay file {a.replay} is missing (slurm/r7_e2_gpu.sbatch smoke)")
    res = reuse_check40(rows, fplan, Path(a.r40_teacher), a.replay, code=not a.no_code, env=not a.no_code, settings=not a.no_code)
    res["checked_utc"] = utc()
    if not res["cal"]["ok"]:
        jwrite(td / "reuse_check.json", res); raise SystemExit("R7-40a's CAL records failed the static check: STOP (coordinator decision)")
    for name in ("d1_manifest.jsonl", "d1_manifest_meta.json"):
        copy_same(Path(a.df_data) / name, dd / name)
    if DA.sha256_file(dd / "d1_manifest.jsonl") != jread(dd / "d1_manifest_meta.json")["manifest_sha256"]:
        raise SystemExit("copied manifest does not match its meta")
    spb = jread(Path(a.df_data) / "d1_shards.json")["shards_per_bundle"]
    plan = FD.make_plan(rows, fplan, res["fit_reuse"], fplan["images_per_shard"], spb); info = plan.pop("d1full")
    plan["e2"] = {"reused_shards": info["reused_shards"], "computed_shards": info["computed_shards"], "reuse_fit": res["fit_reuse"], "roles_planned": ["CAL", "FIT"],
                  "manifest_sha256": v["d1full_manifest_sha256"], "r7_40a_teacher_dir": str(a.r40_teacher), "source_plan": str(Path(a.df_data) / "d1_shards.json"),
                  "note": "DEV is never planned in E2; CAL is always R7-40a's record set; bundles from D1-FULL's make_plan (imported unchanged)"}
    pp = dd / "d1_shards.json"
    if pp.exists() and jread(pp) != plan:
        raise SystemExit(f"{pp} exists with a different plan (the reuse decision changed?): STOP (coordinator)")
    if not pp.exists():
        jwrite(pp, plan)
    res["copied"] = FD.copy_verified(res, Path(a.r40_teacher), td); jwrite(td / "reuse_check.json", res)
    pairs = EP.teacher_lines(plan, root); EP.write_lines(a.lines, pairs, "r7 E2 teacher bundles (frozen R7-40a teacher, L40S; driver slurm/r7_e2_submit.sh)")
    nq = sum(plan["questions_per_shard"][s] for s in plan["e2"]["computed_shards"])
    rec = {"utc": utc(), "manifest_sha256": v["d1full_manifest_sha256"], "fit_reuse": res["fit_reuse"], "bundles": len(plan["bundles"]),
           "computed_shards": len(plan["e2"]["computed_shards"]), "computed_questions": nq, "reused_shards": len(plan["e2"]["reused_shards"]), "lines": str(a.lines)}
    jwrite(dd / "data_ready.json", rec); print(json.dumps(rec, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ cache check
def cache_check(cache: Path, data_dir: Path, basis: str, r40_cache: Path = R40 / "cache"):
    """The frozen cache job's outputs against E2's manifest (no feature value is read: the .npy headers only)."""
    prob = []; cache, data_dir = Path(cache), Path(data_dir); rec = jread(cache / "cache_d1_40.json"); rows = DA.load_manifest(data_dir)
    meta = jread(data_dir / "d1_manifest_meta.json"); res = {"basis": basis, "roles": {}}
    for role in ("FIT", "CAL"):
        ids = FD.role_images(rows, role); ix = jread(cache / f"images_{role}.json"); n = np.asarray(ix["n"], np.int64)
        grid = np.asarray(ix["grid"], np.int64).reshape(-1, 2); off = np.asarray(ix["offsets"], np.int64)
        arr = np.load(cache / f"feats_{role}.npy", mmap_mode="r")
        r = {"images": len(ix["images"]), "tokens": int(n.sum()), "mean_tokens": float(n.mean()) if len(n) else None, "max_tokens": int(n.max()) if len(n) else None,
             "grid_max": [int(grid[:, 0].max()), int(grid[:, 1].max())] if len(grid) else None, "feats_shape": list(arr.shape), "feats_dtype": str(arr.dtype)}
        if ix["images"] != ids:
            prob.append(f"{role}: cached image list differs from the manifest's role order")
        if rec.get(f"{role}_images") != len(ids) or rec.get(f"{role}_tokens") != r["tokens"]:
            prob.append(f"{role}: cache record counts differ ({rec.get(f'{role}_images')} / {rec.get(f'{role}_tokens')})")
        if len(n) and (not np.array_equal(off, np.concatenate([[0], np.cumsum(n)[:-1]])) or not np.array_equal(n, grid[:, 0] * grid[:, 1])):
            prob.append(f"{role}: offsets or grids inconsistent with the token counts")
        if tuple(arr.shape) != (r["tokens"], 512) or arr.dtype != np.float16:
            prob.append(f"{role}: feats shape {arr.shape} / dtype {arr.dtype}")
        if r["grid_max"] and max(r["grid_max"]) > 64:
            prob.append(f"{role}: merged grid beyond the selectors' 64 × 64 tables (D40b-4: the unit would stop)")
        res["roles"][role] = r
    want_q = [r["qid"] for r in rows if r["role"] in ("FIT", "CAL")]
    got_q = [json.loads(x)["qid"] for x in (cache / "question_ids.jsonl").read_text().splitlines() if x.strip()]
    if got_q != want_q:
        prob.append(f"question_ids.jsonl: {len(got_q)} rows, expected the {len(want_q)} FIT + CAL questions in manifest order")
    exp_pca = EP.pca_images(meta, basis) * 64
    if rec.get("pca_tokens") != exp_pca:
        prob.append(f"PCA fitted on {rec.get('pca_tokens')} tokens, expected {exp_pca} (basis {basis}: 64 seeded tokens × images)")
    for f in ("pca.pt", "embed_table.pt"):
        if not (cache / f).exists():
            prob.append(f"{f} missing")
    res.update({"pca_tokens": rec.get("pca_tokens"), "explained_variance": rec.get("explained_variance"), "questions": len(got_q),
                "gpu": rec.get("env", {}).get("gpu"), "seconds": rec.get("seconds"), "peak_mem_gb": rec.get("peak_mem_gb")})
    if basis == "frozen10k" and (Path(r40_cache) / "pca.pt").exists() and (cache / "pca.pt").exists():     # reported, never a gate
        try:
            import torch
            A, B = torch.load(cache / "pca.pt", weights_only=False), torch.load(Path(r40_cache) / "pca.pt", weights_only=False)
            res["pca_vs_r7_40b"] = {k: {"bit_identical": bool(torch.equal(A[k], B[k])), "max_abs_diff": float((A[k] - B[k]).abs().max())} for k in ("mean", "basis")}
        except Exception as e:                                                          # noqa: BLE001  (a comparison failure is reported only)
            res["pca_vs_r7_40b"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    res["problems"] = prob; res["pass"] = not prob
    return res


def cmd_cachecheck(a):
    res = cache_check(Path(a.root) / "cache", Path(a.root) / "data", a.basis, Path(a.r40_cache)); res["utc"] = utc()
    jwrite(Path(a.root) / "cache" / "e2_cache_check.json", res); print(json.dumps(res, indent=1, default=str))
    return 0 if res["pass"] else 6


# ------------------------------------------------------------------------------------------------------------------------------ R7-40c rows
def preds(bench_dir: Path, cfg: str):
    """First complete record per qid over every file of one configuration (the frozen readout's rule, D1-30 E1)."""
    out = {}
    for p in RUN.cfg_paths(Path(bench_dir), cfg):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and "qid" in r and "pred" in r and r["qid"] not in out:
                out[r["qid"]] = r
    return out


def runner_code_identity(ref=R740C_RUN_COMMIT, files=RUNNER_CODE):
    out = {}
    for f in files:
        head, refb, blob = git("rev-parse", f"HEAD:{f}"), git("rev-parse", f"{ref}:{f}"), git("hash-object", f)
        out[f] = {"head_blob": head, "r7_40c_blob": refb, "same": bool(head) and head == refb == blob and git("status", "--porcelain", "--", f) == ""}
    return out, all(v["same"] for v in out.values())


def reuse_rows(e2_pred: Path, r40c: Path, sets_dir: Path, rows: str, min_identical=1.0, copy=True, code=True, benches=PL.BENCHES):
    """R7-40c's selector-independent rows (full and the uniform grid at 0.125 / 0.25 / 0.5 / 0.75) are E2's rows iff their inputs are identical:
    the same set files (sha256 in every R7-40c and E2 shard marker = the file = the D1-30 build summary), the same GPU class in every marker
    (L40S), the same runner code (blobs at HEAD = at R7-40c's commit, clean), R7-40c complete — and, empirically, E2's in-job full row equals
    R7-40c's full row on every item (answer, kept and visual-token counts, generated length).  rows = 'selectors': on a pass the four grid rows
    are copied into E2's prediction dir; rows = 'core': E2 generated its own grid rows and the comparison is descriptive."""
    e2_pred, r40c, sets_dir = Path(e2_pred), Path(r40c), Path(sets_dir); prob = []; res = {"rows": rows, "min_identical": min_identical, "benches": {}}
    comp = r40c / "completeness.json"; cj = jread(comp) if comp.exists() else {}
    if cj.get("complete") is not True:
        prob.append("R7-40c completeness.json missing or not complete")
    summ = jread(sets_dir / "d1_30_sets.json") if (sets_dir / "d1_30_sets.json").exists() else {}
    res["code"], code_ok = runner_code_identity() if code else ({}, True)
    if not code_ok:
        prob.append(f"runner code differs from R7-40c's ({R740C_RUN_COMMIT})")
    ident_ok = True
    for b in benches:
        n = PL.QUALITY_SHARDS[b]; s = DA.sha256_file(sets_dir / f"{b}.jsonl"); B = {"sets_sha256": s, "markers": {}}
        if summ.get(b, {}).get("sha256") != s:
            prob.append(f"{b}: set file sha256 differs from the D1-30 build summary")
        for side, d in (("r7_40c", r40c / "pred" / b), ("e2", e2_pred / b)):
            mk = [d / f"done_shard{i}of{n}.json" for i in range(n)]; js = [jread(m) if m.exists() else None for m in mk]
            gpus = sorted({str((j or {}).get("env", {}).get("gpu")) for j in js}); shas = sorted({str((j or {}).get("sets_file_sha256")) for j in js})
            B["markers"][side] = {"present": sum(j is not None for j in js), "expected": n, "gpus": gpus, "sets_sha256_ok": shas == [s]}
            if None in js:
                prob.append(f"{b}: {side} shard markers missing")
            if gpus != [QUALITY_GPU]:
                prob.append(f"{b}: {side} shards ran on {gpus}, not only {QUALITY_GPU}")
            if shas != [s]:
                prob.append(f"{b}: {side} markers carry another set sha256")
        qids = [json.loads(x)["qid"] for x in (sets_dir / f"{b}.jsonl").read_text().splitlines() if x.strip()]
        cmp_rows = ["full"] + (list(GRID_ROWS) if rows == "core" else [])
        for cfg in cmp_rows:
            A, E = preds(r40c / "pred" / b, cfg), preds(e2_pred / b, cfg)
            same = sum(1 for q in qids if q in A and q in E and A[q] == E[q]); ans = sum(1 for q in qids if q in A and q in E and A[q]["pred"] == E[q]["pred"])
            cov = (sum(q in A for q in qids), sum(q in E for q in qids))
            B[cfg] = {"items": len(qids), "covered_r7_40c": cov[0], "covered_e2": cov[1], "identical_records": same, "identical_answers": ans,
                      "identical_fraction": same / len(qids) if qids else None}
            if cfg == "full" and (cov != (len(qids), len(qids)) or same < min_identical * len(qids)):
                ident_ok = False
        res["benches"][b] = B
    res["static_ok"] = not prob; res["full_row_identity_ok"] = ident_ok
    if not ident_ok:
        prob.append(f"E2's in-job full row differs from R7-40c's on some item (threshold {min_identical})")
    res["problems"] = prob; res["pass"] = res["static_ok"] and ident_ok; res["gate"] = rows == "selectors"; res["copied"] = []
    if rows == "selectors" and res["pass"] and copy:
        for b in benches:
            for cfg in GRID_ROWS:
                srcs = RUN.cfg_paths(r40c / "pred" / b, cfg)
                if not srcs:
                    raise SystemExit(f"{b}: R7-40c has no file for {cfg}")
                for src in srcs:
                    dst = e2_pred / b / src.name
                    if copy_same(src, dst):
                        res["copied"].append({"file": f"{b}/{src.name}", "sha256": DA.sha256_file(dst)})
                    else:
                        res["copied"].append({"file": f"{b}/{src.name}", "sha256": DA.sha256_file(dst), "already_present": True})
    return res


def cmd_reuserows(a):
    res = reuse_rows(Path(a.root) / "eval" / "pred", Path(a.r40c), Path(a.sets_dir), a.rows, a.min_identical); res["utc"] = utc()
    out = Path(a.root) / "eval" / "reuse"; out.mkdir(parents=True, exist_ok=True); jwrite(out / "reuse_check.json", res)
    print(json.dumps({k: res[k] for k in ("rows", "pass", "gate", "static_ok", "full_row_identity_ok", "problems")} | {"benches": res["benches"]}, indent=1)[:8000])
    return 0 if (res["pass"] or not res["gate"]) else 8


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(E2)); sub = ap.add_subparsers(dest="cmd", required=True)

    def src(p):
        p.add_argument("--df-data", default=str(DF / "data")); p.add_argument("--d1-data", default=str(D1 / "data")); p.add_argument("--r40-teacher", default=str(R40 / "teacher"))
        p.add_argument("--replay", default=None, help="default: <root>/smoke/reuse_replay.json"); p.add_argument("--full-sha", default=FULL_SHA); p.add_argument("--frozen-sha", default=FROZEN_SHA)
        p.add_argument("--no-code", action="store_true", help="tests only: skip the git / env / wrapper identity checks")
    p = sub.add_parser("precheck"); src(p); p.add_argument("--out", default=str(E2 / "precheck"))
    d = sub.add_parser("data"); src(d); d.add_argument("--lines", required=True); d.add_argument("--protocol", default=str(PROTOCOL))
    c = sub.add_parser("cachecheck"); c.add_argument("--basis", choices=EP.BASES, required=True); c.add_argument("--r40-cache", default=str(R40 / "cache"))
    r = sub.add_parser("reuserows"); r.add_argument("--rows", choices=EP.ROWS, required=True); r.add_argument("--r40c", default=str(R40C))
    r.add_argument("--sets-dir", default=str(D30_SETS)); r.add_argument("--min-identical", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd in ("precheck", "data") and a.replay is None:
        a.replay = str(Path(a.root) / "smoke" / "reuse_replay.json")
    return {"precheck": cmd_precheck, "data": cmd_data, "cachecheck": cmd_cachecheck, "reuserows": cmd_reuserows}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
