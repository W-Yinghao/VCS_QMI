"""r7 R7-30 D1-FULL data: the pre-registered D1 study repeated with every eligible FIT image (owner 2026-10-12: '还是多做全量实验').
FIT_FULL = every image eligible under the FROZEN D1 rule except the frozen CAL and DEV images.  The frozen builder is called unchanged
(scripts/r7_d1_data.build_manifest: GQA train_all minus every LLaVA-1.5-seen (image, normalised question) pair, >= 1 binary question per
image, images in sha256('r7-30/d1-role/<id>') order with the 5:1:1 interleave, yes / no alternation per role, <= 4 binary questions per image
in sha256('r7-30/d1-q/<qid>') order) with n_fit = eligible - n_cal - n_dev.  The first 14,000 images of the sha256 order then receive exactly
the frozen roles and questions (same interleave, same per-role running yes / no counts) and every later image goes to FIT, so the frozen
manifest is a byte prefix of the D1-FULL manifest (verified, with CAL / DEV row identity and sha256 of their id lists).
  build    --out DF/data                       : D1-FULL manifest + meta (not frozen); sources re-hashed against the frozen meta (python -I)
  verify   --out DF/data                       : CAL / DEV identical, frozen FIT a prefix of FIT_FULL, byte prefix, counts, every FIT_FULL / CAL /
                                                 DEV image present in the extracted GQA folder (never a zip) -> d1full_verify.json
  freeze   --out DF/data --protocol P          : only when P carries 'd1_full_staged_freeze'; the frozen r7_d1_data freeze (sha256 re-check,
                                                 image-exposure disclosure from mix665k)
  reuse    --out DF/data --teacher DF/teacher [--replay R] [--copy] : static reuse check of the frozen D1-10 FIT / CAL teacher records (qids,
                                                 rows, mask-seed digests, record sha256, code / env identity); with --copy the verified files are
                                                 copied into DF/teacher (FIT only if static + GPU replay pass; CAL always, it is the frozen set)
  plan     --out DF/data --lines L             : frozen 250-image shards (CAL / FIT only; DEV never planned) and teacher bundles of the FIT shards
                                                 not reused; sbatch lines with completion markers (#MARK)
  complete --out DF/data --teacher DF/teacher  : every CAL / FIT shard complete (present == expected, no duplicate, no skipped image, sha256)
  data     --out DF/data --teacher DF/teacher --protocol P --lines L : the driver's data stage = build if absent, verify, freeze, reuse --copy, plan
  precheck --out DF/data                       : pre-freeze verification; dry-run files only (d1full_verify.json, reuse_check_dryrun.json, plan_dryrun.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_data as DA  # noqa: E402   (frozen builder, reader allow-list, hash tags, freeze; read-only use)

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
D1 = OUT / "R7_D1"
DF = OUT / "R7_D1_FULL"
FROZEN_MANIFEST_SHA256 = "32c72126ec9d40a362c67883f3a0b92a7212ae063b7ab9a369c47a696706d794"
GQA = Path("/projects/EEG-foundation-model/yinghao/data/gqa")
QUESTIONS = GQA / "train_all_questions"
IMAGES = GQA / "images"
MIX = Path("/projects/EEG-foundation-model/yinghao/data/llava_mix665k/llava_v1_5_mix665k.json")
PROTOCOL = REPO / "reports" / "R7" / "R7_D1_PROTOCOL_FROZEN_20261012.yaml"
FREEZE_KEY = "d1_full_staged_freeze"
TEACHER_CODE = ("scripts/r7_d1_teacher.py", "scripts/r7_d1_masks.py", "scripts/r7_vl_common.py")
TEACHER_SETTINGS = {"images_per_chunk": 16, "max_tokens": 16384, "max_seqs": 64}      # r7_d1_teacher.py bundle defaults used by every D1-10 bundle
SHARDS_PER_BUNDLE = 8
GPU_LINE = "--partition=H100,L40S --exclude=node51,node52,node60"                      # LLaVA env: no Blackwell kernels


def jread(p):
    return json.loads(Path(p).read_text())


def jwrite(p, obj):
    p = Path(p); tmp = p.with_name(p.name + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def id_sha(ids) -> str:
    return sha_bytes("\n".join(ids).encode())


def parse_rows(b: bytes):
    return [json.loads(x) for x in b.decode().splitlines() if x.strip()]


def role_images(rows, role):
    return list(dict.fromkeys(r["image_id"] for r in rows if r["role"] == role))


def frozen_meta(frozen_dir: Path, expected_sha=FROZEN_MANIFEST_SHA256):
    meta = jread(frozen_dir / "d1_manifest_meta.json")
    if meta.get("frozen") is not True:
        raise SystemExit(f"frozen D1 manifest in {frozen_dir} is not marked frozen")
    sha = DA.sha256_file(frozen_dir / "d1_manifest.jsonl")
    if not (sha == meta["manifest_sha256"] and (expected_sha is None or sha == expected_sha)):
        raise SystemExit(f"frozen D1 manifest sha256 {sha} differs from the frozen record")
    return meta


# ------------------------------------------------------------------------------------------------------------------------------ build
def build(rows_iter, fmeta, seen):
    """The frozen builder with n_fit = eligible - n_cal - n_dev (every eligible image assigned).  Returns (rows, meta)."""
    n_cal, n_dev = fmeta["counts"]["CAL"]["images"], fmeta["counts"]["DEV"]["images"]
    n_fit = fmeta["eligible_images"] - n_cal - n_dev
    rows, meta = DA.build_manifest(rows_iter, n_fit, n_cal, n_dev, fmeta["per_image"], seen, balance=True)
    meta["d1full"] = {"rule": "frozen r7_d1_data.build_manifest with n_fit = eligible_images - n_cal - n_dev (FIT_FULL = every eligible image "
                              "except the frozen CAL / DEV); first 14,000 images in sha256 order keep the frozen roles and questions",
                      "n_fit_requested": n_fit, "frozen_eligible_images": fmeta["eligible_images"],
                      "frozen_manifest_sha256": fmeta["manifest_sha256"], "frozen_counts": fmeta["counts"]}
    return rows, meta


def cmd_build(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if (out / "d1_manifest.jsonl").exists():
        raise SystemExit(f"{out / 'd1_manifest.jsonl'} exists; refusing to overwrite (delete it explicitly to rebuild)")
    fmeta = frozen_meta(Path(a.frozen_dir)); src = Path(a.questions)
    files = sorted(f for f in src.iterdir() if DA.TRAIN_ALL_RE.match(f.name)); qsha = {f.name: DA.sha256_file(f) for f in files}
    if qsha != fmeta["questions_sha256"]:
        raise SystemExit(f"question files differ from the frozen meta: {qsha}")
    msha = DA.sha256_file(Path(a.mix665k))
    if msha != fmeta["mix665k_sha256"]:
        raise SystemExit(f"mix665k sha256 {msha} differs from the frozen meta")
    seen, st = DA.llava_seen_pairs(Path(a.mix665k), with_stats=True)
    rows, meta = build(DA.read_questions(src, "all"), fmeta, seen); meta["llava_exclusion"].update(st)
    import vcs_vl_reference as REF
    REF.assert_group_disjoint({k: [r["image_id"] for r in rows if r["role"] == k] for k in ("FIT", "CAL", "DEV")})
    meta.update({"questions_source": str(src), "question_set": "train_all (unbalanced)", "questions_member": "train_all_questions_<n>.json",
                 "questions_sha256": qsha, "mix665k": str(a.mix665k), "mix665k_sha256": msha,
                 "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)})
    tmp = out / "d1_manifest.jsonl.tmp"; tmp.write_text("".join(json.dumps(r) + "\n" for r in rows)); os.replace(tmp, out / "d1_manifest.jsonl")
    meta["manifest_sha256"] = DA.sha256_file(out / "d1_manifest.jsonl"); jwrite(out / "d1_manifest_meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("eligible_images", "counts", "balance", "class_weights_from_FIT", "manifest_sha256", "seconds")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ verify
def verify(out: Path, frozen_dir: Path, images_dir: Path | None = IMAGES, expected_sha=FROZEN_MANIFEST_SHA256):
    """Every check of the D1-FULL manifest against the frozen one; returns the result dict (res['ok'])."""
    fmeta = frozen_meta(frozen_dir, expected_sha); dmeta = jread(out / "d1_manifest_meta.json")
    fb, db = (frozen_dir / "d1_manifest.jsonl").read_bytes(), (out / "d1_manifest.jsonl").read_bytes(); fr, dr = parse_rows(fb), parse_rows(db)
    R, ok = {}, []
    R["frozen_manifest_sha256"] = sha_bytes(fb); R["d1full_manifest_sha256"] = sha_bytes(db)
    ok.append(R["d1full_manifest_sha256"] == dmeta["manifest_sha256"])
    R["frozen_manifest_is_byte_prefix"] = db.startswith(fb); ok.append(R["frozen_manifest_is_byte_prefix"])
    R["roles"] = {}
    for role in ("CAL", "DEV"):
        f, d = [r for r in fr if r["role"] == role], [r for r in dr if r["role"] == role]
        e = {"rows_identical": f == d, "images_sha256_frozen": id_sha(role_images(fr, role)), "images_sha256_d1full": id_sha(role_images(dr, role)),
             "qids_sha256_frozen": id_sha([r["qid"] for r in f]), "qids_sha256_d1full": id_sha([r["qid"] for r in d]),
             "images": len(role_images(dr, role)), "questions": len(d), "yes": sum(r["C"] == 1 for r in d)}
        e["identical"] = e["rows_identical"] and e["images_sha256_frozen"] == e["images_sha256_d1full"] and e["qids_sha256_frozen"] == e["qids_sha256_d1full"]
        R["roles"][role] = e; ok.append(e["identical"])
    f, d = [r for r in fr if r["role"] == "FIT"], [r for r in dr if r["role"] == "FIT"]
    fi, di = role_images(fr, "FIT"), role_images(dr, "FIT")
    e = {"frozen_rows_are_prefix_in_role_order": d[:len(f)] == f, "frozen_images_are_first_in_role_order": di[:len(fi)] == fi,
         "frozen_images_subset": set(fi) <= set(di), "frozen_qids_subset": {r["qid"] for r in f} <= {r["qid"] for r in d},
         "frozen_images_sha256": id_sha(fi), "first_frozen_n_images_sha256_d1full": id_sha(di[:len(fi)]),
         "images": len(di), "questions": len(d), "yes": sum(r["C"] == 1 for r in d), "frozen_images": len(fi), "frozen_questions": len(f),
         "additional_images": len(di) - len(fi), "additional_questions": len(d) - len(f), "additional_yes": sum(r["C"] == 1 for r in d[len(f):])}
    e["pi_yes"] = e["yes"] / max(e["questions"], 1); e["additional_pi_yes"] = e["additional_yes"] / max(e["additional_questions"], 1)
    e["superset"] = all(e[k] for k in ("frozen_rows_are_prefix_in_role_order", "frozen_images_are_first_in_role_order", "frozen_images_subset", "frozen_qids_subset"))
    R["roles"]["FIT"] = e; ok.append(e["superset"])
    per = Counter(r["image_id"] for r in dr); roles_of = {}
    for r in dr:
        roles_of.setdefault(r["image_id"], set()).add(r["role"])
    R["structure"] = {"eligible_images_frozen": fmeta["eligible_images"], "eligible_images_d1full": dmeta["eligible_images"],
                      "assigned_images": len(per), "max_q_per_image": max(per.values()), "min_q_per_image": min(per.values()),
                      "images_with_two_roles": sum(len(v) > 1 for v in roles_of.values()), "binary_answers_only": all(r["answer"] in ("yes", "no") for r in dr),
                      "C_matches_answer": all(r["C"] == (1 if r["answer"] == "yes" else -1) for r in dr), "duplicate_qids": len(dr) - len({r["qid"] for r in dr}),
                      "trimmed": {k: v["trimmed"] for k, v in dmeta["balance"].items()}, "pi_yes": {k: v["pi_yes"] for k, v in dmeta["balance"].items()},
                      "class_weights_from_FIT_FULL": dmeta["class_weights_from_FIT"], "class_weights_frozen": fmeta["class_weights_from_FIT"]}
    s = R["structure"]
    ok += [s["eligible_images_frozen"] == s["eligible_images_d1full"] == s["assigned_images"], s["max_q_per_image"] <= fmeta["per_image"],
           s["min_q_per_image"] >= 1, s["images_with_two_roles"] == 0, s["binary_answers_only"], s["C_matches_answer"], s["duplicate_qids"] == 0]
    R["sources"] = {"questions_sha256_equal": dmeta.get("questions_sha256") == fmeta["questions_sha256"], "mix665k_sha256_equal": dmeta.get("mix665k_sha256") == fmeta["mix665k_sha256"]}
    ok += list(R["sources"].values())
    if images_dir is not None:
        names = {n[:-4] for n in os.listdir(images_dir) if n.endswith(".jpg")}
        miss = {role: sorted(set(role_images(dr, role)) - names) for role in ("FIT", "CAL", "DEV")}
        R["images"] = {"folder": str(images_dir), "jpg_files": len(names), "missing": {k: len(v) for k, v in miss.items()},
                       "missing_examples": {k: v[:10] for k, v in miss.items() if v}, "zip_fallback": "never (zips deleted 2026-10-12; D1-FULL readers fail on a missing file)"}
        ok.append(not any(miss.values()))
    R["counts"] = {role: {"images": len(role_images(dr, role)), "questions": sum(r["role"] == role for r in dr), "yes": sum(r["role"] == role and r["C"] == 1 for r in dr)}
                   for role in ("FIT", "CAL", "DEV")}
    R["ok"] = all(ok)
    return R


def cmd_verify(a):
    R = verify(Path(a.out), Path(a.frozen_dir), None if a.no_images else IMAGES); R["checked_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jwrite(Path(a.out) / "d1full_verify.json", R); print(json.dumps(R, indent=1))
    return 0 if R["ok"] else 5


# ------------------------------------------------------------------------------------------------------------------------------ freeze
def protocol_freeze_line(protocol: Path):
    lines = [x for x in Path(protocol).read_text().splitlines() if FREEZE_KEY in x]
    return lines[0].strip() if lines else None


def cmd_freeze(a):
    out = Path(a.out); line = protocol_freeze_line(Path(a.protocol))
    if line is None:
        raise SystemExit(f"D1-FULL is not frozen: '{FREEZE_KEY}' absent in {a.protocol}")
    meta = jread(out / "d1_manifest_meta.json")
    if meta.get("frozen") is True:
        if DA.sha256_file(out / "d1_manifest.jsonl") != meta["manifest_sha256"]:
            raise SystemExit("frozen D1-FULL manifest changed since it was frozen")
        print("[r7_d1full_data] already frozen"); return 0
    ver = jread(out / "d1full_verify.json") if (out / "d1full_verify.json").exists() else {}
    if ver.get("ok") is not True or ver.get("d1full_manifest_sha256") != meta["manifest_sha256"]:
        raise SystemExit("verify the D1-FULL manifest first (d1full_verify.json ok for this sha256)")
    go = f"{FREEZE_KEY} in {Path(a.protocol).name} (sha256 {DA.sha256_file(Path(a.protocol))[:16]}): {line[:160]}"
    return DA.cmd_freeze(argparse.Namespace(out=str(out), protocol="R7-30-D1-FULL", go=go, mix665k=str(a.mix665k) if a.mix665k else None))


# ------------------------------------------------------------------------------------------------------------------------------ shards
def shard_plan(rows, images_per_shard=250, roles=("CAL", "FIT", "DEV")):
    """r7_d1_data.plan_shards' shards and per-shard question counts (same names, image lists and role order), in linear time."""
    qn = Counter(r["image_id"] for r in rows); shards = {}
    for role in ("CAL", "FIT", "DEV"):
        if role not in roles:
            continue
        imgs = sorted({r["image_id"] for r in rows if r["role"] == role}, key=lambda i: DA.h("d1-role", i))
        for j in range(0, len(imgs), images_per_shard):
            shards[f"{role}-{j // images_per_shard:03d}"] = imgs[j:j + images_per_shard]
    return shards, {s: sum(qn[i] for i in v) for s, v in shards.items()}


def make_plan(rows, frozen_plan, reuse_fit: bool, images_per_shard=250, shards_per_bundle=SHARDS_PER_BUNDLE):
    """CAL / FIT shards (DEV never planned); bundles 'f000'... over the FIT shards that are not reused.  The frozen CAL / FIT shards must be
    identical to the frozen plan's (same names and image lists)."""
    shards, qps = shard_plan(rows, images_per_shard, roles=("CAL", "FIT"))
    frozen_cf = {s: v for s, v in frozen_plan["shards"].items() if s.startswith(("CAL-", "FIT-"))}
    bad = [s for s, v in frozen_cf.items() if shards.get(s) != v]
    if bad or frozen_plan["images_per_shard"] != images_per_shard:
        raise SystemExit(f"D1-FULL shards differ from the frozen plan: {bad[:5]}")
    reused = sorted(s for s in frozen_cf if s.startswith("CAL-") or reuse_fit)
    todo = [s for s in shards if s.startswith("FIT-") and s not in reused]
    bundles = {f"f{b // shards_per_bundle:03d}": todo[b:b + shards_per_bundle] for b in range(0, len(todo), shards_per_bundle)}
    return {"images_per_shard": images_per_shard, "shards_per_bundle": shards_per_bundle, "shards": shards, "bundles": bundles, "questions_per_shard": qps,
            "d1full": {"reused_shards": reused, "computed_shards": todo, "reuse_fit": reuse_fit, "roles_planned": ["CAL", "FIT"],
                       "note": "DEV is never planned in D1-FULL; CAL is always the frozen D1-10 record set"}}


def teacher_lines(plan, teacher_dir: Path, wrapper="slurm/r7_d1full_gpu.sbatch"):
    return [(f"sbatch --parsable --job-name=r7dFt_{b} {GPU_LINE} --time=04:00:00 {wrapper} bundle {b}", teacher_dir / f"bundle_{b}.ok.json")   # ≈ 3 × an L40S bundle
            for b in plan["bundles"]]


def write_lines(path: Path, pairs, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}: <sbatch line> #MARK <completion marker>\n" + "".join(f"{c} #MARK {m}\n" for c, m in pairs))


def cmd_plan(a):
    out = Path(a.out); rows = DA.load_manifest(out); fplan = jread(Path(a.frozen_dir) / "d1_shards.json")
    rc = jread(Path(a.teacher) / "reuse_check.json") if (Path(a.teacher) / "reuse_check.json").exists() else None
    if rc is None:
        raise SystemExit("run the reuse check first (reuse_check.json)")
    if not rc["cal"]["ok"]:
        raise SystemExit("the frozen CAL records failed the static check: STOP (CAL is the frozen D1-10 set; coordinator decision)")
    plan = make_plan(rows, fplan, bool(rc["fit_reuse"]), shards_per_bundle=a.shards_per_bundle)
    plan["d1full"]["manifest_sha256"] = jread(out / "d1_manifest_meta.json")["manifest_sha256"]
    jwrite(out / "d1_shards.json", plan)
    write_lines(Path(a.lines), teacher_lines(plan, Path(a.teacher)), "R7-30 D1-FULL teacher bundles (driver slurm/r7_d1full_submit.sh)")
    nq = sum(plan["questions_per_shard"][s] for s in plan["d1full"]["computed_shards"])
    print(json.dumps({"shards": len(plan["shards"]), "bundles": len(plan["bundles"]), "computed_shards": len(plan["d1full"]["computed_shards"]),
                      "computed_questions": nq, "reused_shards": len(plan["d1full"]["reused_shards"]), "lines": str(a.lines)}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ reuse
def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def code_identity(run_start_ts: float):
    """The teacher code used by D1-FULL (working tree) equals the code that produced the D1-10 records: each file is clean against HEAD and
    its last commit predates the first D1-10 shard by more than the 10-minute host load."""
    out = {}
    for f in TEACHER_CODE:
        blob = git("hash-object", f); head = git("rev-parse", f"HEAD:{f}"); last = git("log", "-1", "--format=%ct %h", "--", f).split()
        out[f] = {"blob": blob, "head_blob": head, "clean": blob == head and git("status", "--porcelain", "--", f) == "",
                  "last_commit": last[1] if len(last) > 1 else None, "last_commit_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(last[0]))) if last else None,
                  "before_run": bool(last) and int(last[0]) < run_start_ts - 600}
    return out, all(v["clean"] and v["before_run"] for v in out.values())


def env_identity(run_start_ts: float):
    paths = [Path("/projects/EEG-foundation-model/yinghao/envs/sparsevlm/FREEZE.txt")]
    for d in ("llava-v1.5-7b", "clip-vit-large-patch14-336"):
        md = Path("/projects/EEG-foundation-model/yinghao/models") / d
        paths += sorted(p for p in md.glob("*") if p.is_file()) if md.exists() else []
    rec = {str(p): {"mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)), "before_run": p.stat().st_mtime < run_start_ts}
           for p in paths if p.exists()}
    fz = paths[0]
    return {"files": rec, "freeze_txt_sha256": DA.sha256_file(fz) if fz.exists() else None}, bool(rec) and all(v["before_run"] for v in rec.values())


def check_records(shard, recs, rows, done, jsonl_bytes):
    """Static checks of one frozen shard's records against the D1-FULL manifest rows of its images."""
    import numpy as np
    import r7_d1_masks as MK
    import r7_d1_teacher as TE
    exp = {r["qid"]: r for r in rows}; prob = []
    if sha_bytes(jsonl_bytes) != done.get("jsonl_sha256"):
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
        orders = {"random": MK.order_random(r["qid"]), "grid": MK.order_grid(r["qid"])}
        for mech, (order, logp) in orders.items():
            if TE.digest(order) != r["order_digest"][mech] or any(r["logp"][mech][str(k)] != logp[k] for k in MK.BUDGETS):
                prob.append(f"{r['qid']}: {mech} ordering / log-prob differs from the current mask code"); break
        q = TE.unb64(r["qsim_order"])
        if len(q) != MK.N or len(set(q.tolist())) != MK.N or TE.digest(q) != r["order_digest"]["qsim"]:
            prob.append(f"{r['qid']}: qsim ordering is not the recorded permutation")
        vals = [r["full"], r["text"]] + [r["masked"][mm][str(k)] for mm in MK.MECHANISMS for k in MK.BUDGETS]
        if set(r["masked"]) != set(MK.MECHANISMS) or not all(len(v) == 2 and np.isfinite(v).all() and 0 <= min(v) and sum(v) <= 1 + 1e-6 for v in vals):
            prob.append(f"{r['qid']}: probability fields malformed")
        n_dig += 1
    return {"records": len(recs), "expected": len(exp), "digest_checked": n_dig, "gpu": done.get("gpu"), "problems": prob[:10], "n_problems": len(prob), "ok": not prob}


def settings_check(feed_lines=REPO / "slurm" / "r7_d1_10_feed_lines.txt", teacher_src=REPO / "scripts" / "r7_d1_teacher.py",
                   wrapper=REPO / "slurm" / "r7_d1_gpu.sbatch"):
    """D1-10 bundles ran 'r7_d1_gpu.sbatch bundle bNNN' with the r7_d1_teacher.py defaults (no chunk / batch options); D1-FULL passes
    TEACHER_SETTINGS explicitly, which must equal those defaults."""
    import re
    fl = [x for x in Path(feed_lines).read_text().splitlines() if x.strip() and not x.startswith("#")] if Path(feed_lines).exists() else []
    lines_ok = bool(fl) and all(x.rstrip().split(" bundle ")[-1].count(" ") == 0 for x in fl)
    src = Path(teacher_src).read_text(); defaults = {}
    for opt, key in (("--images-per-chunk", "images_per_chunk"), ("--max-tokens", "max_tokens"), ("--max-seqs", "max_seqs")):
        m = re.search(re.escape(f'add_argument("{opt}", type=int, default=') + r"(\d+)", src); defaults[key] = int(m.group(1)) if m else None
    wl = [x for x in Path(wrapper).read_text().splitlines() if x.strip().startswith("bundle)")] if Path(wrapper).exists() else []
    wrapper_ok = len(wl) == 1 and not any(o in wl[0] for o in ("--images-per-chunk", "--max-tokens", "--max-seqs"))
    return {"feed_lines": len(fl), "feed_lines_without_options": lines_ok, "teacher_defaults": defaults, "d1full_settings": TEACHER_SETTINGS,
            "wrapper_bundle_line_without_options": wrapper_ok, "ok": lines_ok and wrapper_ok and defaults == TEACHER_SETTINGS}


def reuse_check(out: Path, frozen_dir: Path, frozen_teacher: Path, replay=None, feed_lines=REPO / "slurm" / "r7_d1_10_feed_lines.txt"):
    """Static reuse check of the frozen D1-10 CAL / FIT teacher records (never DEV) for D1-FULL.  FIT records are reused if and only if
    every static check passes and the GPU replay (smoke) passes; otherwise FIT-000..039 are recomputed.  CAL is the frozen record set."""
    rows = DA.load_manifest(out); fplan = jread(frozen_dir / "d1_shards.json"); by_img = {}
    for r in rows:
        by_img.setdefault(r["image_id"], []).append(r)
    shards = [s for s in fplan["shards"] if s.startswith(("CAL-", "FIT-"))]
    dshards, _ = shard_plan(rows, fplan["images_per_shard"], roles=("CAL", "FIT"))
    res = {"shards": {}, "settings": TEACHER_SETTINGS}; starts = []
    for s in shards:
        done_p, rec_p = frozen_teacher / f"{s}.done.json", frozen_teacher / f"{s}.jsonl"
        if not (done_p.exists() and rec_p.exists()):
            res["shards"][s] = {"ok": False, "problems": ["missing frozen record or done file"]}; continue
        done = jread(done_p); b = rec_p.read_bytes(); recs = parse_rows(b)
        mrows = [q for i in fplan["shards"][s] for q in by_img.get(i, [])]
        e = check_records(s, recs, mrows, done, b); e["same_images_as_d1full_plan"] = dshards.get(s) == fplan["shards"][s]
        e["ok"] = e["ok"] and e["same_images_as_d1full_plan"]; e["jsonl_sha256"] = done.get("jsonl_sha256"); res["shards"][s] = e
        starts.append(done_p.stat().st_mtime - float(done.get("seconds", 0.0)))
    run_start = min(starts) if starts else 0.0
    res["d1_10_first_shard_start_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start))
    res["code"], code_ok = code_identity(run_start); res["env"], env_ok = env_identity(run_start)
    res["settings_check"] = settings_check(feed_lines); settings_ok = res["settings_check"]["ok"]; res["settings_ok"] = settings_ok
    rp = jread(replay) if replay and Path(replay).exists() else None
    res["replay"] = {"file": str(replay) if replay else None, "pass": None if rp is None else bool(rp.get("pass")),
                     "summary": None if rp is None else {k: rp.get(k) for k in ("gpu", "same_gpu_shards", "bit_identical_same_gpu", "max_abs_dT_same_gpu", "max_abs_dT_cross_gpu", "cross_gpu_shards")}}
    fit_ok = all(v["ok"] for s, v in res["shards"].items() if s.startswith("FIT-")); cal_ok = all(v["ok"] for s, v in res["shards"].items() if s.startswith("CAL-"))
    res["fit"] = {"shards": sum(s.startswith("FIT-") for s in res["shards"]), "static_ok": fit_ok and code_ok and env_ok and settings_ok}
    res["cal"] = {"shards": sum(s.startswith("CAL-") for s in res["shards"]), "ok": cal_ok and code_ok and env_ok and settings_ok}
    res["fit_reuse"] = bool(res["fit"]["static_ok"] and res["replay"]["pass"])
    res["rule"] = ("FIT-000..039 reused iff qids / rows / mask-seed digests / record sha256 / code / env / settings are identical AND the GPU replay passes; "
                   "otherwise recomputed by D1-FULL bundles.  CAL-000..007 are always the frozen D1-10 records (a failed static check stops the driver).")
    return res


def copy_verified(res, frozen_teacher: Path, teacher: Path):
    """Copy the reused shard files (jsonl + done record) into the D1-FULL teacher dir; sha256 re-checked; never overwrite a different file."""
    teacher.mkdir(parents=True, exist_ok=True); copied = []
    for s, e in res["shards"].items():
        if not (s.startswith("CAL-") or res["fit_reuse"]):
            continue
        for name, sha in ((f"{s}.jsonl", e["jsonl_sha256"]), (f"{s}.done.json", None)):
            src, dst = frozen_teacher / name, teacher / name; want = sha or DA.sha256_file(src)
            if dst.exists():
                if DA.sha256_file(dst) != want:
                    raise SystemExit(f"{dst} exists and differs from the frozen file; not overwritten")
                continue
            tmp = dst.with_name(dst.name + ".tmp"); shutil.copyfile(src, tmp)
            if DA.sha256_file(tmp) != want:
                tmp.unlink(); raise SystemExit(f"copy of {src} does not match its sha256")
            os.replace(tmp, dst); copied.append(name)
    return copied


def cmd_reuse(a):
    res = reuse_check(Path(a.out), Path(a.frozen_dir), Path(a.frozen_teacher), a.replay); res["checked_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.teacher).mkdir(parents=True, exist_ok=True); name = "reuse_check.json" if a.copy else "reuse_check_dryrun.json"
    if a.copy:
        if not res["cal"]["ok"]:
            jwrite(Path(a.teacher) / name, res); raise SystemExit("frozen CAL records failed the static check: STOP")
        res["copied"] = copy_verified(res, Path(a.frozen_teacher), Path(a.teacher))
    jwrite(Path(a.teacher) / name, res)
    short = {k: res[k] for k in ("fit", "cal", "fit_reuse", "settings_ok", "replay", "d1_10_first_shard_start_utc")}
    short["shard_problems"] = {s: v["problems"] for s, v in res["shards"].items() if not v["ok"]}
    short["code_ok"] = all(v["clean"] and v["before_run"] for v in res["code"].values()); print(json.dumps(short, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ complete
def completeness(out: Path, teacher: Path):
    plan = jread(out / "d1_shards.json"); meta = jread(out / "d1_manifest_meta.json"); rows = DA.load_manifest(out)
    per_role = Counter(r["role"] for r in rows if r["role"] in ("CAL", "FIT")); bad, tot, gpus = [], Counter(), Counter()
    for s, imgs in plan["shards"].items():
        f, j = teacher / f"{s}.done.json", teacher / f"{s}.jsonl"
        if not (f.exists() and j.exists()):
            bad.append(f"{s}: missing"); continue
        d = jread(f); exp = plan["questions_per_shard"][s]
        if d["present"] != exp or d["expected"] != exp or d["duplicates"] or d["skipped_images"] or DA.sha256_file(j) != d["jsonl_sha256"]:
            bad.append(f"{s}: present {d['present']} expected {d['expected']}/{exp} dup {d['duplicates']} skipped {len(d['skipped_images'])}")
        tot[s.split("-")[0]] += d["present"]; gpus[d.get("gpu")] += 1
    res = {"shards": len(plan["shards"]), "problems": bad, "present_per_role": dict(tot), "manifest_per_role": dict(per_role),
           "manifest_sha256": DA.sha256_file(out / "d1_manifest.jsonl"), "frozen_flag": meta.get("frozen"), "gpus": dict(gpus)}
    res["complete"] = (not bad and res["manifest_sha256"] == meta["manifest_sha256"] and meta.get("frozen") is True
                       and all(tot.get(k) == per_role[k] for k in per_role))
    return res


def cmd_complete(a):
    res = completeness(Path(a.out), Path(a.teacher)); jwrite(Path(a.teacher) / "d1full_completeness.json", res); print(json.dumps(res, indent=1))
    return 0 if res["complete"] else 7


# ------------------------------------------------------------------------------------------------------------------------------ data stage
def cmd_data(a):
    """The driver's data stage (CPU job): build if absent -> verify -> freeze -> reuse --copy -> plan; marker data_ready.json."""
    out = Path(a.out)
    if protocol_freeze_line(Path(a.protocol)) is None:
        raise SystemExit(f"D1-FULL is not frozen: '{FREEZE_KEY}' absent in {a.protocol}")
    if not (out / "d1_manifest.jsonl").exists():
        rc = cmd_build(argparse.Namespace(out=a.out, frozen_dir=a.frozen_dir, questions=a.questions, mix665k=a.mix665k))
        if rc:
            return rc
    for step in (lambda: cmd_verify(argparse.Namespace(out=a.out, frozen_dir=a.frozen_dir, no_images=False)),
                 lambda: cmd_freeze(argparse.Namespace(out=a.out, protocol=a.protocol, mix665k=a.mix665k)),
                 lambda: cmd_reuse(argparse.Namespace(out=a.out, frozen_dir=a.frozen_dir, frozen_teacher=a.frozen_teacher, teacher=a.teacher, replay=a.replay, copy=True)),
                 lambda: cmd_plan(argparse.Namespace(out=a.out, frozen_dir=a.frozen_dir, teacher=a.teacher, lines=a.lines, shards_per_bundle=a.shards_per_bundle))):
        rc = step()
        if rc:
            return rc
    jwrite(out / "data_ready.json", {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "manifest_sha256": jread(out / "d1_manifest_meta.json")["manifest_sha256"],
                                     "lines": str(a.lines)})
    return 0


def cmd_precheck(a):
    """Pre-freeze verification (writes dry-run files only; the driver's data stage redoes every step): verify, static reuse check (+ the GPU
    replay result if present), plan counts for the reuse outcome -> d1full_verify.json, teacher/reuse_check_dryrun.json, data/plan_dryrun.json."""
    rc = cmd_verify(argparse.Namespace(out=a.out, frozen_dir=a.frozen_dir, no_images=False))
    res = reuse_check(Path(a.out), Path(a.frozen_dir), Path(a.frozen_teacher), a.replay); res["checked_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.teacher).mkdir(parents=True, exist_ok=True); jwrite(Path(a.teacher) / "reuse_check_dryrun.json", res)
    rows = DA.load_manifest(Path(a.out)); fplan = jread(Path(a.frozen_dir) / "d1_shards.json"); dry = {}
    for reuse in (True, False):
        pl = make_plan(rows, fplan, reuse, shards_per_bundle=a.shards_per_bundle)
        dry["reuse_fit" if reuse else "recompute_fit"] = {"shards": len(pl["shards"]), "bundles": len(pl["bundles"]), "computed_shards": len(pl["d1full"]["computed_shards"]),
                                                          "computed_images": sum(len(pl["shards"][s]) for s in pl["d1full"]["computed_shards"]),
                                                          "computed_questions": sum(pl["questions_per_shard"][s] for s in pl["d1full"]["computed_shards"]),
                                                          "reused_shards": len(pl["d1full"]["reused_shards"])}
    dry["outcome_now"] = "reuse_fit" if res["fit_reuse"] else "recompute_fit"; jwrite(Path(a.out) / "plan_dryrun.json", dry)
    print(json.dumps({"verify_ok": rc == 0, "fit_static_ok": res["fit"]["static_ok"], "cal_ok": res["cal"]["ok"], "replay": res["replay"], "fit_reuse": res["fit_reuse"],
                      "settings": res["settings_check"], "code_ok": all(v["clean"] and v["before_run"] for v in res["code"].values()),
                      "shard_problems": {s: v["problems"] for s, v in res["shards"].items() if not v["ok"]}, "plan_dryrun": dry}, indent=1))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p, teacher=False):
        p.add_argument("--out", default=str(DF / "data")); p.add_argument("--frozen-dir", default=str(D1 / "data"))
        if teacher:
            p.add_argument("--teacher", default=str(DF / "teacher")); p.add_argument("--frozen-teacher", default=str(D1 / "teacher"))
        return p
    b = common(sub.add_parser("build")); b.add_argument("--questions", default=str(QUESTIONS)); b.add_argument("--mix665k", default=str(MIX))
    v = common(sub.add_parser("verify")); v.add_argument("--no-images", action="store_true")
    f = common(sub.add_parser("freeze")); f.add_argument("--protocol", default=str(PROTOCOL)); f.add_argument("--mix665k", default=str(MIX))
    r = common(sub.add_parser("reuse"), True); r.add_argument("--replay", default=str(DF / "smoke" / "reuse_replay.json")); r.add_argument("--copy", action="store_true")
    p = common(sub.add_parser("plan"), True); p.add_argument("--lines", default=str(DF / "lines" / "teacher_lines.txt"))
    p.add_argument("--shards-per-bundle", type=int, default=SHARDS_PER_BUNDLE)
    c = common(sub.add_parser("complete"), True)
    pc = common(sub.add_parser("precheck"), True); pc.add_argument("--replay", default=str(DF / "smoke" / "reuse_replay.json"))
    pc.add_argument("--shards-per-bundle", type=int, default=SHARDS_PER_BUNDLE)
    d = common(sub.add_parser("data"), True); d.add_argument("--protocol", default=str(PROTOCOL)); d.add_argument("--questions", default=str(QUESTIONS))
    d.add_argument("--mix665k", default=str(MIX)); d.add_argument("--replay", default=str(DF / "smoke" / "reuse_replay.json"))
    d.add_argument("--lines", default=str(DF / "lines" / "teacher_lines.txt")); d.add_argument("--shards-per-bundle", type=int, default=SHARDS_PER_BUNDLE)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "verify": cmd_verify, "freeze": cmd_freeze, "reuse": cmd_reuse, "plan": cmd_plan, "complete": cmd_complete, "data": cmd_data,
            "precheck": cmd_precheck}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
