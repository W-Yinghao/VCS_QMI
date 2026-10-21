"""Stage AGIQA-3K + ImageReward after the data_fetch download; then run the package's own `aigiqa.cli prepare`.

Run inside the AIGIQA env with common.sh sourced (stage_data.sbatch does this). Server-side wrapper only:
the split, adapter, image hashing and sealed files are produced by the unchanged package code.

Label safety: MOS values are never printed or stored here. The owner-MOS mapping check compares the two label
tables byte-for-byte and reports only counts; prepare() itself reads every label (as the package requires) and writes
TEST labels to prepared/.../sealed/, which this script then makes unreadable (chmod 000) until the TEST unseal step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = HERE / "pins"
OWNER_DIR = "AGIQA-3k-Database-30ded2bed96fc78070636548e811ece37e5d8f66"
CMIM_DIR = "CMIM-AIGIQA-4108d1d6c26c5494ddac7bb71b11ecdf67203d95"
EXPECT = {
    "csv_git_blob": "49f88aa19b6c0327be21d7eab6cb90a44847f2f9",
    "full_csv_sha256": "4b1945c2137e196d9378b9cf4a114de2267b97a4724f0743e260287eb2ddd6c0",
    "imagereward_pt_sha256": "d71dc8d4912bc633c06ab8ef1ef6a81e8199d4369f14b47b29baf1d19de561f6",
    "imagereward_pt_bytes": 1786880927,
    "med_config_git_blob": "0ffad0a6f3c2f9f11b8faa84529d9860bb70327a",
    "med_config_bytes": 485,
    "imagereward_revision": "5736be03b2652728fb87788c9797b0570450ab72",
    "bert_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "n_images": 2982,
    "zip_bytes": 97950738,
    "fetch_items": ["agiqa3k_images", "agiqa3k_owner_repo", "cmim_aigiqa_4108d1d", "imagereward_v1"],
}
BERT_FILES = ["vocab.txt", "tokenizer_config.json", "config.json", "tokenizer.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, **detail):
        self.items.append({"check": name, "pass": bool(ok), **detail})
        print(("PASS " if ok else "FAIL ") + name + (f" {detail}" if not ok else ""), flush=True)
        return ok

    @property
    def ok(self):
        return all(i["pass"] for i in self.items)


def fetch_records(c: Checks, manifest_path: Path, data: Path, models: Path) -> dict:
    m = load(manifest_path)["items"]
    out = {}
    official = m.get("agiqa3k_images", {}).get("status") == "done"
    mirror = m.get("agiqa3k_images_hf_mirror", {}).get("status") == "done"
    c.add("exactly_one_image_source_fetched", official != mirror, official=official, hf_mirror=mirror)
    items = [i for i in EXPECT["fetch_items"] if i != "agiqa3k_images" or official] + (["agiqa3k_images_hf_mirror"] if mirror else [])
    for item in items:
        rec = m.get(item, {})
        c.add(f"fetch_record_done:{item}", rec.get("status") == "done", status=rec.get("status"))
        out[item] = {k: rec.get(k) for k in ("status", "started", "finished", "revision", "files", "bytes")}
    out["image_source"] = ("official author Google Drive AGIQA-3K.zip (README @30ded2b)" if official else
                           "HF mirror strawhat/agiqa-3k@bf9b7bb8 (fallback; per-file LFS sha256 verified by data_fetch)" if mirror else None)
    if mirror:
        c.add("hf_mirror_revision_pinned", m["agiqa3k_images_hf_mirror"].get("revision") == "bf9b7bb8d633f44832d6ba51ff496a25d8e8e2ae")
        c.add("hf_mirror_all_lfs_sha256_ok", all(f.get("sha256_ok") is not False for f in m["agiqa3k_images_hf_mirror"].get("files") or []))
    for item, sub in ((("agiqa3k_images", "images_official"),) if official else ()) + (("agiqa3k_owner_repo", "sources"), ("cmim_aigiqa_4108d1d", "sources")):
        for f in (m.get(item, {}).get("files") or []):
            p = data / sub / "zips" / f["name"]
            got = sha256(p) if p.is_file() else None
            c.add(f"archive_sha256_matches_fetch_record:{f['name']}", got == f.get("sha256"), recorded=f.get("sha256"), recomputed=got)
            if item == "agiqa3k_images":
                c.add("agiqa3k_zip_bytes_equal_head_content_length", p.is_file() and p.stat().st_size == EXPECT["zip_bytes"],
                      bytes=p.stat().st_size if p.is_file() else None)
    ir = m.get("imagereward_v1", {})
    c.add("imagereward_fetch_revision_pinned", ir.get("revision") == EXPECT["imagereward_revision"], revision=ir.get("revision"))
    lfs = {f["name"]: f for f in (ir.get("files") or [])}
    c.add("imagereward_pt_lfs_sha256_ok_in_fetch_record", (lfs.get("ImageReward.pt") or {}).get("sha256_ok") is True)
    return out


def git_tree(c: Checks, root: Path, tree_json: Path, label: str) -> dict:
    tree = load(tree_json)
    blobs = [t for t in tree["tree"] if t["type"] == "blob"]
    bad, missing = [], []
    for t in blobs:
        p = root / t["path"]
        if not p.is_file():
            missing.append(t["path"])
        elif p.stat().st_size != t["size"] or git_blob(p) != t["sha"]:
            bad.append(t["path"])
    listed = {t["path"] for t in blobs}
    extra = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and str(p.relative_to(root)) not in listed)
    c.add(f"git_tree_exact:{label}", not bad and not missing and not extra, mismatched=bad[:20], missing=missing[:20], extra=extra[:20])
    return {"commit": tree["sha"], "blobs_checked": len(blobs), "mismatched": bad, "missing": missing, "extra": extra}


def owner_mos_mapping(c: Checks, owner_csv: Path, cmim_csv: Path) -> dict:
    """Owner data.csv vs CMIM full.csv: byte identity + column semantics + aggregate validity (no values kept)."""
    res = {"owner_csv": str(owner_csv), "cmim_full_csv": str(cmim_csv),
           "owner_sha256": sha256(owner_csv), "cmim_sha256": sha256(cmim_csv),
           "owner_git_blob": git_blob(owner_csv), "cmim_git_blob": git_blob(cmim_csv)}
    res["byte_identical"] = owner_csv.read_bytes() == cmim_csv.read_bytes()
    c.add("owner_vs_cmim_byte_identical", res["byte_identical"])
    c.add("csv_git_blob_equals_pinned_trees", res["owner_git_blob"] == EXPECT["csv_git_blob"] == res["cmim_git_blob"])
    c.add("cmim_full_csv_sha256_equals_package_record", res["cmim_sha256"] == EXPECT["full_csv_sha256"])
    with open(cmim_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    res["header"] = header
    # Owner README (section 2B, 1-based): 1 image name, 2 prompt, 3-4 adjectives, 5 style, 6-7 perception MOS/STD, 8-9 alignment MOS/STD.
    # Package adapter (aigiqa/manifest.py _adapt): name, prompt, mos_quality -> quality, mos_align -> alignment.
    positions = {k: header.index(k) if k in header else None for k in ("name", "prompt", "mos_quality", "mos_align")}
    res["adapter_column_positions_0based"] = positions
    c.add("adapter_columns_match_owner_readme_positions", len(header) == 9 and positions == {"name": 0, "prompt": 1, "mos_quality": 5, "mos_align": 7},
          header=header)
    names = [r[positions["name"]] for r in rows] if positions["name"] is not None else []
    res["rows"] = len(rows)
    res["unique_names"] = len(set(names))
    res["empty_prompts"] = sum(1 for r in rows if not r[1].strip()) if rows else None
    for col in ("mos_quality", "mos_align"):
        i = positions[col]
        ok = 0
        for r in rows:
            try:
                v = float(r[i])
                ok += math.isfinite(v) and 0.0 <= v <= 5.0
            except (ValueError, IndexError, TypeError):
                pass
        res[f"{col}_finite_within_0_5_count"] = ok
    c.add("csv_rows_2982_unique_names", res["rows"] == EXPECT["n_images"] == res["unique_names"], rows=res["rows"], unique=res["unique_names"])
    c.add("csv_mos_all_finite_in_declared_0_5_scale",
          res["mos_quality_finite_within_0_5_count"] == res["rows"] == res["mos_align_finite_within_0_5_count"])
    res["label_values_reported"] = False
    res["note"] = "Byte identity with the owner's data.csv at 30ded2b makes the CMIM name->MOS mapping the owner's for all rows; only counts are reported."
    return res, names


def find_image_root(c: Checks, parent: Path, names: list[str]) -> Path | None:
    wanted = set(names)
    hits = defaultdict(int)
    for p in parent.rglob("*"):
        if p.is_file() and "zips" not in p.relative_to(parent).parts and p.name in wanted:
            hits[p.parent] += 1
    full = [d for d, n in hits.items() if n == len(wanted)]
    c.add("unique_image_root_holds_every_csv_name", len(full) == 1 and len(hits) == 1, candidates={str(k): v for k, v in hits.items()})
    return full[0] if len(full) == 1 else None


def validate_images(c: Checks, root: Path, names: list[str], mirror_json: Path, inventory_csv: Path) -> dict:
    from PIL import Image
    mirror = load(mirror_json)["files"]
    rows, fmt, mode, dims, by_hash = [], Counter(), Counter(), Counter(), defaultdict(list)
    decode_fail, mirror_mismatch, mirror_missing = [], [], []
    for name in sorted(names):
        p = root / name
        digest = sha256(p)
        rec = {"name": name, "bytes": p.stat().st_size, "sha256": digest}
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                rec.update(format=im.format, mode=im.mode, width=im.size[0], height=im.size[1])
        except Exception as exc:  # recorded, not skipped
            decode_fail.append(name)
            rec.update(format=None, mode=None, width=None, height=None, error=type(exc).__name__)
        m = mirror.get("images/" + name)
        rec["mirror_sha256_match"] = None if m is None else (m.get("sha256_lfs") == digest)
        if m is None:
            mirror_missing.append(name)
        elif not rec["mirror_sha256_match"]:
            mirror_mismatch.append(name)
        fmt[rec["format"]] += 1; mode[rec["mode"]] += 1; dims[f"{rec['width']}x{rec['height']}"] += 1
        by_hash[digest].append(name)
        rows.append(rec)
    extra = sorted(p.name for p in root.iterdir() if p.is_file() and p.name not in set(names))
    dup = [v for v in by_hash.values() if len(v) > 1]
    inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(inventory_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "bytes", "sha256", "format", "mode", "width", "height", "mirror_sha256_match", "error"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    c.add("all_images_decode", not decode_fail, failed=decode_fail[:20])
    c.add("all_images_match_hf_mirror_lfs_sha256", not mirror_mismatch and not mirror_missing,
          mismatched=mirror_mismatch[:20], missing_in_mirror=mirror_missing[:20])
    c.add("no_extra_files_in_image_root", not extra, extra=extra[:20])
    return {"image_root": str(root), "n_images": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
            "formats": dict(fmt), "modes": dict(mode), "sizes": dict(dims.most_common()),
            "duplicate_byte_groups": dup, "decode_failures": decode_fail, "extra_files": extra,
            "mirror": {"repo": load(mirror_json)["repo"], "revision": load(mirror_json)["revision"],
                       "matched": sum(1 for r in rows if r["mirror_sha256_match"]), "mismatched": mirror_mismatch, "missing": mirror_missing},
            "inventory_csv": str(inventory_csv), "inventory_sha256": sha256(inventory_csv)}


def model_assets(c: Checks, models: Path) -> dict:
    pt, med, readme = models / "ImageReward.pt", models / "med_config.json", models / "README.md"
    res = {"ImageReward.pt": {"path": str(pt), "bytes": pt.stat().st_size if pt.is_file() else None, "sha256": sha256(pt) if pt.is_file() else None},
           "med_config.json": {"path": str(med), "bytes": med.stat().st_size if med.is_file() else None,
                               "sha256": sha256(med) if med.is_file() else None, "git_blob": git_blob(med) if med.is_file() else None}}
    if readme.is_file():
        res["README.md"] = {"path": str(readme), "sha256": sha256(readme), "git_blob": git_blob(readme)}
    c.add("imagereward_pt_sha256_and_bytes", res["ImageReward.pt"]["sha256"] == EXPECT["imagereward_pt_sha256"]
          and res["ImageReward.pt"]["bytes"] == EXPECT["imagereward_pt_bytes"], got=res["ImageReward.pt"])
    c.add("med_config_git_blob_and_bytes", res["med_config.json"]["git_blob"] == EXPECT["med_config_git_blob"]
          and res["med_config.json"]["bytes"] == EXPECT["med_config_bytes"], got=res["med_config.json"])
    if med.is_file():
        res["med_config.json"]["keys"] = sorted(load(med))
    return res


def tokenizer_cache(c: Checks, hf_home: Path, bert_local: Path, dry: bool) -> dict:
    pins = load(PINS / "hf_google-bert_bert-base-uncased_86b5e09.json")["files"]
    rev = EXPECT["bert_revision"]
    repo = hf_home / "hub" / "models--bert-base-uncased"
    snap = repo / "snapshots" / rev
    res = {"hf_home": str(hf_home), "repo_id_requested_by_imagereward": "bert-base-uncased",
           "resolved_revision": rev, "source": str(bert_local), "files": {}}
    for f in BERT_FILES:
        blob = git_blob(bert_local / f)
        res["files"][f] = {"sha256": sha256(bert_local / f), "git_blob": blob, "pinned_git_blob": pins[f]["git_blob"]}
        c.add(f"bert_local_matches_pinned_revision:{f}", blob == pins[f]["git_blob"])
    if not dry:
        snap.mkdir(parents=True, exist_ok=True)
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(rev)
        for f in BERT_FILES:
            link = snap / f
            if link.is_symlink() or link.exists():
                if os.readlink(link) != str(bert_local / f):
                    raise RuntimeError(f"unexpected existing cache entry {link}")
            else:
                link.symlink_to(bert_local / f)
        prov = {"linked": [{"repo": "bert-base-uncased (google-bert/bert-base-uncased)", "revision": rev, "linked_files": BERT_FILES,
                            "source": str(bert_local), "data_manifest_item": "bert_base_uncased"}],
                "purpose": "ImageReward BLIP init_tokenizer() -> BertTokenizer.from_pretrained('bert-base-uncased') under HF_HUB_OFFLINE=1",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (hf_home / "CACHE_PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n")
        probe = subprocess.run([sys.executable, "-c",
            "from ImageReward.models.BLIP.blip import init_tokenizer; t=init_tokenizer(); "
            "ids=t('a red cube on a blue sphere', padding='max_length', truncation=True, max_length=35).input_ids; "
            "import json; print(json.dumps({'len': len(t), 'enc_token_id': t.enc_token_id, 'bos_token_id': t.bos_token_id, 'n_ids': len(ids), 'ids_head': ids[:8]}))"],
            capture_output=True, text=True, env={**os.environ, "HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        ok = probe.returncode == 0
        res["offline_init_tokenizer_probe"] = json.loads(probe.stdout.strip().splitlines()[-1]) if ok else {"stderr": probe.stderr[-2000:]}
        c.add("imagereward_init_tokenizer_loads_offline", ok and res["offline_init_tokenizer_probe"].get("len") == 30524,
              probe=res["offline_init_tokenizer_probe"])
    return res


def run_prepare(c: Checks, pkg: Path, cmim_csv: Path, image_root: Path, prepared: Path) -> dict:
    cmd = [sys.executable, "-m", "aigiqa.cli", "prepare", "--dataset", "AGIQA-3k", "--source-csv", str(cmim_csv),
           "--image-root", str(image_root), "--output", str(prepared)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=pkg, capture_output=True, text=True)
    res = {"command": cmd, "cwd": str(pkg), "returncode": p.returncode, "seconds": time.time() - t0, "stderr_tail": p.stderr[-3000:]}
    c.add("package_prepare_exit0", p.returncode == 0, stderr=p.stderr[-500:])
    if p.returncode == 0:
        res["prepare_output"] = json.loads(p.stdout)
    return res


def post_prepare(c: Checks, prepared: Path) -> dict:
    from aigiqa.manifest import verify_prepared, read_csv
    info = verify_prepared(prepared)  # package check: FIT/CAL hashes, sample/prompt/image-byte disjointness
    fit, cal = read_csv(prepared / "fit.csv"), read_csv(prepared / "cal.csv")
    test = read_csv(prepared / "test_inputs.csv")  # label-free
    c.add("test_inputs_label_free", all("alignment_mos" not in r and "quality_mos" not in r for r in test))
    splits = {"fit": fit, "cal": cal, "test": test}
    res = {"manifest_sha256": sha256(prepared / "manifest.json"), "split_seed": info["split_seed"],
           "fractions": [info["fit_fraction"], info["cal_fraction"]], "near_threshold": info["near_threshold"],
           "image_counts": info["image_counts"], "split_group_counts": info["split_group_counts"],
           "unique_image_byte_hashes": info["unique_image_byte_hashes"], "duplicate_image_byte_rows": info["duplicate_image_byte_rows"],
           "candidate_prompt_groups": {k: len({r["prompt_group"] for r in v}) for k, v in splits.items()},
           "generators": {k: dict(Counter(r["generator"] for r in v)) for k, v in splits.items()},
           "input_and_label_hashes": info["input_and_label_hashes"]}
    for a, b in (("fit", "test"), ("cal", "test"), ("fit", "cal")):
        for key in ("sample_id", "prompt_group", "image_sha256"):
            inter = {r[key] for r in splits[a]} & {r[key] for r in splits[b]}
            c.add(f"disjoint_{a}_{b}_{key}", not inter, n=len(inter))
    c.add("all_2982_images_assigned", sum(info["image_counts"].values()) == EXPECT["n_images"], counts=info["image_counts"])
    with open(prepared / "group_merges.csv", newline="") as f:
        merges = list(csv.DictReader(f))
    res["group_merges"] = dict(Counter(m["reason"] for m in merges))
    return res


def seal(prepared: Path, image_root: Path) -> dict:
    """TEST labels unreadable until the release step; everything else read-only (the lock also hashes it)."""
    sealed = prepared / "sealed"
    for p in sorted(sealed.iterdir()):
        os.chmod(p, 0)
    os.chmod(sealed, 0)
    for p in prepared.iterdir():
        if p.is_file():
            os.chmod(p, 0o444)
    for p in image_root.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o444)
    return {"sealed_dir_mode": oct(stat.S_IMODE(os.stat(sealed).st_mode)), "sealed_files_mode": "000",
            "prepared_files_mode": "444", "images_mode": "444",
            "unseal_command_at_TEST_release_only": f"chmod 700 {sealed} && chmod 400 {sealed}/*.csv",
            "note": "workflow boundary, not access control (package EXPERIMENT_LOCK_CN.md)"}


def sha256sums(root: Path, out: Path, skip: set[str]) -> int:
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in skip for part in rel.parts) or not p.is_file() or p == out:
            continue
        lines.append(f"{sha256(p)}  {rel}")
    out.write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True); ap.add_argument("--models", required=True); ap.add_argument("--hf-home", required=True)
    ap.add_argument("--bert-local", required=True); ap.add_argument("--pkg", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--dry-run", action="store_true", help="verify only; no cache, prepare, chmod or record under /projects")
    a = ap.parse_args()
    data, models, evidence = Path(a.data), Path(a.models), Path(a.evidence)
    c = Checks()
    record = {"schema": "aigiqa_server_staging_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "dry_run": a.dry_run,
              "python": sys.executable}
    record["fetch"] = fetch_records(c, Path(a.manifest), data, models)
    record["owner_repo_tree"] = git_tree(c, data / "sources" / OWNER_DIR, PINS / "gh_tree_AGIQA-3k-Database_30ded2b.json", "AGIQA-3k-Database@30ded2b")
    record["cmim_tree"] = git_tree(c, data / "sources" / CMIM_DIR, PINS / "gh_tree_CMIM-AIGIQA_4108d1d.json", "CMIM-AIGIQA@4108d1d")
    owner_csv = data / "sources" / OWNER_DIR / "data.csv"
    cmim_csv = data / "sources" / CMIM_DIR / "Database" / "AGIQA-3k" / "full.csv"
    record["owner_mos_mapping_check"], names = owner_mos_mapping(c, owner_csv, cmim_csv)
    root = find_image_root(c, data / "images_official", names)
    if root is not None:
        record["images"] = validate_images(c, root, names, PINS / "hf_strawhat_agiqa-3k_bf9b7bb.json",
                                           (evidence if a.dry_run else data) / "image_inventory.csv")
    record["model_assets"] = model_assets(c, models)
    record["tokenizer_cache"] = tokenizer_cache(c, Path(a.hf_home), Path(a.bert_local), a.dry_run)
    prepared = data / "prepared" / "AGIQA-3k"
    if c.ok and not a.dry_run:
        record["prepare"] = run_prepare(c, Path(a.pkg), cmim_csv, root, prepared)
        if c.ok:
            record["prepared"] = post_prepare(c, prepared)
            if c.ok:
                # hash lists first (sealed files are covered by manifest.json's input_and_label_hashes), then seal
                record["sha256sums"] = {"data": sha256sums(data, data / "SHA256SUMS", {"sealed", "zips"}),
                                        "models": sha256sums(models, models / "SHA256SUMS", {".cache", "hf_home"})}
                record["seal"] = seal(prepared, root)
    record["checks"] = c.items
    record["status"] = "PASS" if c.ok else "FAIL"
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "STAGING_RECORD.json").write_text(text)
    if not a.dry_run:
        (data / "STAGING_RECORD.json").write_text(text)
    print(json.dumps({"status": record["status"], "failed": [i["check"] for i in c.items if not i["pass"]]}, indent=1))
    raise SystemExit(0 if c.ok else 1)


if __name__ == "__main__":
    main()
