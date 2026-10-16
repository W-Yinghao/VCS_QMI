#!/usr/bin/env python3
"""V10 T00 - data manifest and exposure record for BF-SHG and EPFL RGB-NIR, built from what R6 / V9 actually used.

METADATA ONLY.  Inputs are existing manifests, lists and ledgers plus file-system metadata (os.listdir / os.stat):
  BF-SHG  R6 REG-P1 cache manifest (outputs/R6_REG/manifest.json: FIT / CAL / DEV roles, per-file sha256, full shape; the cache V9-G1 used),
          configs/V9_G1_TEST_CORES.csv (V9-G1's frozen 134-core test list: archive member names, per-file sha256, TIFF header facts),
          reports/V9/V9_TEST_READ_LEDGER.csv (the one V9-G1 test read), data/DATA_MANIFEST.json (archive facts);
  EPFL    R6 REG-P3 cache manifest (outputs/R6_REG_P3/manifest.json: 477 pairs, alignment / size exclusions, near-duplicate links, units,
          roles, windows; the cache V9-G2 used), PROVENANCE.json of both on-server copies, P60's 60-pair id lists (reports/P60_task_B_T2_rgbnir*.json).
It never opens an image or a cache file, never opens / lists / hashes the sealed archive (closed/test.zip: os.stat only), reads no labels and
imports no trainer or host code (stdlib only).  Deterministic: the same inputs give a byte-identical V10_DATA_MANIFEST.csv.

Outputs (default <repo>/v10/T00/): V10_DATA_MANIFEST.csv (exactly the columns of the package's configs/DATA_MANIFEST_TEMPLATE.csv),
MANIFEST_BUILD_SUMMARY.json (counts, checks, relations, exposure events), MANIFEST_SHA256.txt (sha256sum format), EXPOSURE_NOTES.md.
If <out>/PREFLIGHT.json exists (the package's scripts/preflight.py run on this manifest), its status is copied into the summary / notes and
the manifest sha256 it recorded must equal the rebuilt manifest's.

  /home/infres/yinwang/CS_QMI/env/bin/python -I scripts/v10/t00_build_manifest.py [--out v10/T00]
Exit 0 when every check passes, 1 otherwise (the outputs are written in both cases, failures included).
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "v10" / "package_v10_server_v1_20261016"
TEMPLATE = PKG / "configs" / "DATA_MANIFEST_TEMPLATE.csv"
DATA_ROOT = Path("/projects/EEG-foundation-model/yinghao/data")
BF_DIR = DATA_ROOT / "bf_shg_4550300"
BF_ARCHIVE = BF_DIR / "closed" / "test.zip"
BF_SPLITS = ("train", "validation1", "validation2")
EPFL_DIR = DATA_ROOT / "epfl_rgb_nir"
EPFL_HOME = Path("/home/infres/yinwang/CS_QMI/data/rgb_nir")              # P60's copy (same archive), not used by R6 / V9 REG
R6_OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
P1_MANIFEST = R6_OUT / "R6_REG" / "manifest.json"
P1_CACHE = R6_OUT / "R6_REG" / "cache"
P3_MANIFEST = R6_OUT / "R6_REG_P3" / "manifest.json"
P3_CACHE = R6_OUT / "R6_REG_P3" / "cache"
V9_OUT = Path("/projects/EEG-foundation-model/yinghao/outputs_v9")
V9_TEST_RAW = V9_OUT / "reg_test" / "raw"
V9_TEST_CACHE = V9_OUT / "reg_test" / "cache"
V9_G2_UNITS = V9_OUT / "reg_epfl" / "units"
V9_TEST_CORES = REPO / "configs" / "V9_G1_TEST_CORES.csv"
V9_LEDGER = REPO / "reports" / "V9" / "V9_TEST_READ_LEDGER.csv"
DATA_MANIFEST_JSON = REPO / "data" / "DATA_MANIFEST.json"
P60_JSONS = {"full": REPO / "reports" / "P60_task_B_T2_rgbnir.json", "patch_ctf": REPO / "reports" / "P60_task_B_T2_rgbnir_patch_ctf.json",
             "smoke": REPO / "reports" / "P60_task_B_T2_rgbnir_smoke.json"}
WIN = 512                                                                   # REG-P3 registration window (native px)

# values recorded by earlier frozen documents (cross-checked below, never used to fill the manifest)
OFFICIAL = {"BF_SHG": {"pairs": 206, "train": 40, "validation1": 25, "validation2": 7, "test": 134,
                       "source": "Zenodo 4550300 record (full-size TMA cores) and CoMIR (NeurIPS 2020), as cited in the package docs/PROTOCOL_SOURCES_CN.md 2.1-2.2"},
            "EPFL_RGB_NIR": {"pairs": 477, "categories": 9, "resolution": "1024x768 (as stated on the dataset page)",
                             "source": "EPFL IVRL dataset page and Brown & Suesstrunk CVPR 2011, as cited in the package docs/PROTOCOL_SOURCES_CN.md 2.3"}}
FROZEN_REFS = {"test_cores_csv_sha256": "2d395c6ce203143d3ae9d89cb4eefea57efa368367ff89709c0fe95502766bdc",   # V9-G1 test-read protocol frozen_files
               "p60_id_list_sha256": "6c262c3ed830baf97f6f7020f9659cba5ffb55c05d680b4c9b2dd468f971b19e",      # P60 prereg appendix
               "p1_manifest_sha256_prefix": "b8a9e3e7",                                                      # V9-G1 gate record
               "p3_manifest_sha256_prefix": "3ecf9fe8",                                                      # V9-G2 manifest_check
               "v9_g1_fixture_dev_cores": ["1B_B5", "1B_B9", "1B_D1"]}                                         # V9-G1 test-read execution note 1
EXPOSURE_VOCAB = {"used_for_design", "never_read", "unknown"}               # the package preflight's vocabulary
ROLE_VOCAB = {"FIT", "CAL", "DEV", "CONFIRM"}

BF_GROUP_BASIS = ("core: one TMA core = one BF/SHG pair = the physical-specimen unit of R6/V9; patient / tissue-block mapping not published "
                  "(patient-level independence unverified); slide prefix 1B/2B = stratum")
BF_ALIGN = ("pair aligned by the dataset authors (method not described on the record); host truth = identity at 512 px after x4 area "
            "downsampling; thresholds 42 / 9 native px (CoMIR 5 % / 1 % of the 834-px centre patch)")
EPFL_GROUP_BASIS = ("scene unit of R6 REG-P3: one RGB/NIR pair; near-duplicates (thumbnail corr > 0.9; same category and thumbnail shape) "
                    "merged transitively, also through excluded scenes")
HOLD_REASON = ("HOLD for V10 phase 1 (not a data-quality exclusion): official test core read once by V9-G1 as its sealed-test confirmation "
               "(prepare job {prepare_job}; read_utc {read_utc}; results {results_commit}); V10 status = development / repeat benchmark "
               "(role DEV; never CONFIRM); no phase-1 read; activation needs a frozen V10 protocol and a new manifest version")

GUARDED_SUFFIXES = {".tif", ".tiff", ".zip", ".npz", ".npy", ".pt", ".pth", ".png", ".jpg", ".jpeg"}


# ------------------------------------------------------------------------------------------------------------ guarded helpers
def read_meta(path: Path) -> bytes:
    """Read a metadata file (JSON / CSV / YAML / text).  Images, archives, caches, checkpoints and anything under a closed/ directory are refused."""
    p = Path(path)
    if p.suffix.lower() in GUARDED_SUFFIXES or "closed" in p.parts:
        raise SystemExit(f"[guard] refusing to read {p}: images, archives, caches and closed/ are metadata-only (os.stat)")
    return p.read_bytes()


def load_json(path: Path):
    return json.loads(read_meta(path).decode("utf-8"))


def load_csv(path: Path) -> list:
    return list(csv.DictReader(io.StringIO(read_meta(path).decode("utf-8-sig"))))


def sha256_meta(path: Path) -> str:
    return hashlib.sha256(read_meta(path)).hexdigest()


def git(*args: str) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, env=env, stderr=subprocess.STDOUT, timeout=30).strip()


def first_commit(rel: str) -> str:
    try:
        out = git("log", "--diff-filter=A", "--format=%h %cI", "--", rel).splitlines()
        return out[-1] if out else "untracked"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def code_hash() -> dict:
    """V10 code hash = sha256 of the concatenated files under src/v10 and scripts/v10 in sorted path order, first 16 hex.
    'head': the files committed at HEAD (reproducible from the commit); 'worktree': the regular files on disk (__pycache__ skipped), equal to
    find src/v10 scripts/v10 -type f ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16
    (they differ when the working tree holds uncommitted V10 files, e.g. another agent's work in progress)."""
    files = []
    for d in ("src/v10", "scripts/v10"):
        base = REPO / d
        if base.is_dir():
            files += [p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: str(q.relative_to(REPO)).encode()):
        h.update(p.read_bytes())
    out = {"worktree": h.hexdigest()[:16]}
    try:
        names = sorted(git("ls-tree", "-r", "--name-only", "HEAD", "--", "src/v10", "scripts/v10").splitlines(), key=str.encode)
        hh = hashlib.sha256()
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        for n in names:
            hh.update(subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{n}"], env=env, timeout=30))
        out["head"] = hh.hexdigest()[:16]
    except (OSError, subprocess.SubprocessError) as exc:
        out["head"] = f"unknown ({exc})"
    out["equal"] = out["head"] == out["worktree"]
    return out


def dir_bytes(root: Path) -> tuple:
    """Total bytes and file count under root from os.stat only (no file is opened)."""
    total = n = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            total += os.stat(os.path.join(dp, fn)).st_size; n += 1
    return total, n


class Checks:
    def __init__(self):
        self.items = []

    def add(self, cid: str, ok: bool, detail) -> bool:
        self.items.append({"id": cid, "status": "PASS" if ok else "FAIL", "detail": detail}); return ok

    @property
    def failed(self) -> list:
        return [c["id"] for c in self.items if c["status"] != "PASS"]


# ------------------------------------------------------------------------------------------------------------ BF-SHG
def bf_disk() -> tuple:
    cores, extra, unpaired = {}, [], []
    for split in BF_SPLITS:
        d = BF_DIR / split; names = sorted(os.listdir(d)); have = set(names)
        for n in names:
            if n.endswith("_WB.tif"):
                core = n[: -len("_WB.tif")]; mi = core + "_MI.tif"
                if mi not in have:
                    unpaired.append(f"{split}/{n}"); continue
                if core in cores:
                    extra.append(f"{split}/{n} (core id also in {cores[core]['split']})"); continue
                cores[core] = {"split": split, "fixed": f"bf_shg_4550300/{split}/{n}", "moving": f"bf_shg_4550300/{split}/{mi}",
                               "fixed_bytes": os.stat(d / n).st_size, "moving_bytes": os.stat(d / mi).st_size}
            elif n.endswith("_MI.tif"):
                if n[: -len("_MI.tif")] + "_WB.tif" not in have:
                    unpaired.append(f"{split}/{n}")
            else:
                extra.append(f"{split}/{n}")
    return cores, extra, unpaired


def bf_section(ck: Checks) -> dict:
    p1 = load_json(P1_MANIFEST); p1_sha = sha256_meta(P1_MANIFEST)
    disk, extra, unpaired = bf_disk()
    by_split = collections.Counter(v["split"] for v in disk.values())
    ck.add("bf.disk.pairs_per_split", dict(by_split) == {s: OFFICIAL["BF_SHG"][s] for s in BF_SPLITS} and not extra and not unpaired,
           {"pairs": dict(by_split), "files": {s: len(os.listdir(BF_DIR / s)) for s in BF_SPLITS}, "extra": extra, "unpaired": unpaired})
    ck.add("bf.p1_manifest.is_v9_g1_cache", p1_sha.startswith(FROZEN_REFS["p1_manifest_sha256_prefix"]),
           {"sha256": p1_sha, "expected_prefix": FROZEN_REFS["p1_manifest_sha256_prefix"]})
    roles = p1["roles"]; files = p1["files"]
    ck.add("bf.p1_manifest.cores_equal_disk", set(roles) == set(files) == set(disk),
           {"manifest_roles": len(roles), "manifest_files": len(files), "disk": len(disk),
            "only_disk": sorted(set(disk) - set(files)), "only_manifest": sorted(set(files) - set(disk))})
    ck.add("bf.p1_manifest.split_equal_disk", all(files[c]["split"] == disk[c]["split"] for c in disk if c in files),
           [c for c in disk if c in files and files[c]["split"] != disk[c]["split"]])
    role_counts = collections.Counter(roles.values())
    split_role = collections.Counter((files[c]["split"], roles[c]) for c in roles)
    ck.add("bf.roles.counts", dict(role_counts) == {"FIT": 32, "CAL": 8, "DEV": 32}
           and all(r in ("FIT", "CAL") for (s, r) in split_role if s == "train") and all(r == "DEV" for (s, r) in split_role if s != "train"),
           {"roles": dict(role_counts), "split_x_role": {f"{s}|{r}": n for (s, r), n in sorted(split_role.items())}})
    train = sorted((c for c in disk if disk[c]["split"] == "train"), key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    recomputed = {c: ("CAL" if i < 8 else "FIT") for i, c in enumerate(train)}
    recomputed.update({c: "DEV" for c in disk if disk[c]["split"] != "train"})
    ck.add("bf.roles.rule_recomputed_from_ids", recomputed == roles,
           {"rule": "R6 REG-P1 roles(): train cores sorted by sha256(core id), first 8 CAL, rest FIT; validation1 + validation2 DEV",
            "mismatch": sorted(c for c in roles if recomputed.get(c) != roles[c])})
    shapes = collections.Counter(tuple(files[c]["full_shape"]) for c in files)
    ck.add("bf.trainval.full_shape", set(shapes) == {(2048, 2048, 3)}, {f"{k}": v for k, v in shapes.items()})
    sizes = collections.Counter((v["fixed_bytes"], v["moving_bytes"]) for v in disk.values())

    test = load_csv(V9_TEST_CORES); test_sha = sha256_meta(V9_TEST_CORES)
    ledger = load_csv(V9_LEDGER)
    ck.add("bf.test.ledger_one_read", len(ledger) == 1 and ledger[0]["item"] == "bf_shg_test_closed",
           {"rows": len(ledger), "items": [r["item"] for r in ledger]})
    L = ledger[0]
    ck.add("bf.test.core_list_is_frozen", test_sha == FROZEN_REFS["test_cores_csv_sha256"] == L["cores_csv_sha256"],
           {"sha256": test_sha, "frozen": FROZEN_REFS["test_cores_csv_sha256"], "ledger": L["cores_csv_sha256"]})
    tcores = [r["core"] for r in test]
    ck.add("bf.test.count_unique", len(tcores) == len(set(tcores)) == OFFICIAL["BF_SHG"]["test"] == int(L["n_cores"]),
           {"rows": len(tcores), "unique": len(set(tcores)), "ledger_n_cores": L["n_cores"], "by_slide": dict(collections.Counter(r["slide"] for r in test))})
    ck.add("bf.test.member_names", all(r["bf_member"] == f"test/{r['core']}_BF.tif" and r["shg_member"] == f"test/{r['core']}_SHG.tif" for r in test),
           [r["core"] for r in test if r["bf_member"] != f"test/{r['core']}_BF.tif" or r["shg_member"] != f"test/{r['core']}_SHG.tif"])
    tiff = collections.Counter((r["bf_w"], r["bf_h"], r["bf_spp"], r["bf_bits"], r["bf_comp"], r["shg_w"], r["shg_h"], r["shg_spp"], r["shg_bits"], r["shg_comp"]) for r in test)
    ck.add("bf.test.header_facts", set(tiff) == {("2048", "2048", "3", "8", "1", "2048", "2048", "1", "8", "1")}, {"|".join(k): v for k, v in tiff.items()})
    ck.add("bf.test.ids_disjoint_from_trainval", not (set(tcores) & set(disk)), sorted(set(tcores) & set(disk)))

    dm = load_json(DATA_MANIFEST_JSON)["items"]["bf_shg_test_closed"]["files"][0]
    st = os.stat(BF_ARCHIVE)                                                   # size only: the sealed archive is not opened
    ck.add("bf.test.archive_metadata", st.st_size == dm["bytes"] == int(L["archive_bytes"]) and dm["sha256"] == L["archive_sha256"] and dm["md5"] == L["archive_md5"],
           {"stat_bytes": st.st_size, "data_manifest_bytes": dm["bytes"], "ledger_bytes": L["archive_bytes"], "sha256_recorded": dm["sha256"],
            "md5_recorded": dm["md5"], "note": "os.stat only; T00 did not open the archive (no member list read, no hash computed)"})
    raw_ok, raw_bad = 0, []
    for r in test:
        for kind, col in (("BF", "bf_bytes"), ("SHG", "shg_bytes")):
            p = V9_TEST_RAW / f"{r['core']}_{kind}.tif"
            if p.is_file() and os.stat(p).st_size == int(r[col]):
                raw_ok += 1
            else:
                raw_bad.append(str(p))
    ck.add("bf.test.v9_extracted_copies_present", raw_ok == 2 * len(test) and not raw_bad,
           {"files_ok": raw_ok, "bad": raw_bad[:10], "dir": str(V9_TEST_RAW), "rule": "is_file and os.stat size == frozen-list bytes (not opened)",
            "dir_entries": len(os.listdir(V9_TEST_RAW))})

    # exact byte duplicates among all 206 x 2 recorded sha256 values (R6 REG-P1 + V9-G1 frozen list)
    owners = collections.defaultdict(list)
    for c, f in files.items():
        owners[f["sha256_bf"]].append(f"{c}:BF"); owners[f["sha256_shg"]].append(f"{c}:SHG")
    for r in test:
        owners[r["bf_sha256"]].append(f"{r['core']}:BF"); owners[r["shg_sha256"]].append(f"{r['core']}:SHG")
    dups = {h: o for h, o in owners.items() if len(o) > 1}
    ck.add("bf.sha256.no_exact_duplicates", not dups, {"files_hashed_by_r6_v9": sum(len(o) for o in owners.values()), "duplicates": dups})

    # TMA grid positions shared by the two slides (a possible relation that metadata cannot resolve)
    allroles = {c: (disk[c]["split"], roles[c]) for c in disk}
    allroles.update({c: ("test", "TEST(V9)") for c in tcores})
    pos = collections.defaultdict(dict)
    for c, (s, r) in allroles.items():
        slide, p = c.split("_", 1); pos[p][slide] = (c, s, r)
    shared = {p: v for p, v in pos.items() if len(v) > 1}
    straddle = collections.Counter(tuple(sorted(v[s][2] for s in v)) for v in shared.values())
    total, nfiles = dir_bytes(BF_DIR)
    dm_tv = load_json(DATA_MANIFEST_JSON)["items"]["bf_shg_trainval"]
    ck.add("bf.dir_bytes_equal_data_manifest", total == dm_tv["bytes"],
           {"dir": str(BF_DIR), "stat_total_bytes": total, "files": nfiles, "data_manifest_bf_shg_trainval_bytes": dm_tv["bytes"],
            "note": "the DATA_MANIFEST byte count equals the whole directory (zips + extracted train/val + closed/test.zip), from os.stat"})
    zips = {z: os.stat(BF_DIR / "zips" / z).st_size for z in sorted(os.listdir(BF_DIR / "zips"))}
    cache = {"r6_reg_p1_cache_npz": len([n for n in os.listdir(P1_CACHE) if n.endswith(".npz")]),
             "v9_g1_test_cache_npz": len([n for n in os.listdir(V9_TEST_CACHE) if n.endswith(".npz")])}
    ck.add("bf.child_caches_complete", cache["r6_reg_p1_cache_npz"] == len(disk) and cache["v9_g1_test_cache_npz"] == len(test),
           {**cache, "rule": "file counts by os.listdir (the caches are not opened)"})
    return {"p1": p1, "p1_sha": p1_sha, "disk": disk, "roles": roles, "files": files, "test": test, "test_sha": test_sha, "ledger": L,
            "sizes": {f"{a}|{b}": n for (a, b), n in sizes.items()}, "zips": zips, "cache": cache, "dir_total": total,
            "shared_positions": {p: {s: list(v[s]) for s in sorted(v)} for p, v in sorted(shared.items())},
            "shared_position_role_pairs": {" / ".join(k): n for k, n in sorted(straddle.items())}}


# ------------------------------------------------------------------------------------------------------------ EPFL RGB-NIR
def epfl_disk(root: Path) -> tuple:
    pairs, extra, unpaired = {}, [], []
    base = root / "nirscene1"
    cats = sorted(d for d in os.listdir(base) if (base / d).is_dir())
    extra += [n for n in os.listdir(base) if not (base / n).is_dir()]
    for cat in cats:
        names = sorted(os.listdir(base / cat)); have = set(names)
        for n in names:
            if n.endswith("_rgb.tiff"):
                sid = n[: -len("_rgb.tiff")]; nir = sid + "_nir.tiff"
                if nir not in have:
                    unpaired.append(f"{cat}/{n}"); continue
                pairs[f"{cat}/{sid}"] = {"category": cat, "fixed": f"epfl_rgb_nir/nirscene1/{cat}/{n}", "moving": f"epfl_rgb_nir/nirscene1/{cat}/{nir}",
                                         "fixed_bytes": os.stat(base / cat / n).st_size, "moving_bytes": os.stat(base / cat / nir).st_size}
            elif n.endswith("_nir.tiff"):
                if n[: -len("_nir.tiff")] + "_rgb.tiff" not in have:
                    unpaired.append(f"{cat}/{n}")
            else:
                extra.append(f"{cat}/{n}")
    return pairs, cats, extra, unpaired


def epfl_section(ck: Checks) -> dict:
    prov = load_json(EPFL_DIR / "PROVENANCE.json"); prov_home = load_json(EPFL_HOME / "PROVENANCE.json")
    disk, cats, extra, unpaired = epfl_disk(EPFL_DIR)
    per_cat = collections.Counter(v["category"] for v in disk.values())
    ck.add("epfl.disk.pairs_per_category", len(disk) == prov["n_pairs"] == OFFICIAL["EPFL_RGB_NIR"]["pairs"] and dict(per_cat) == prov["pairs_per_category"]
           and len(cats) == OFFICIAL["EPFL_RGB_NIR"]["categories"] and not extra and not unpaired,
           {"pairs": len(disk), "categories": len(cats), "per_category": dict(sorted(per_cat.items())), "provenance": prov["pairs_per_category"],
            "extra": extra, "unpaired": unpaired})
    home, home_cats, home_extra, home_unpaired = epfl_disk(EPFL_HOME)
    same = (set(home) == set(disk) and all((home[k]["fixed_bytes"], home[k]["moving_bytes"]) == (disk[k]["fixed_bytes"], disk[k]["moving_bytes"]) for k in disk)
            and prov_home == prov)
    ck.add("epfl.second_copy_is_same_release", same,
           {"projects_copy": str(EPFL_DIR / "nirscene1"), "home_copy": str(EPFL_HOME / "nirscene1"), "home_pairs": len(home),
            "provenance_identical": prov_home == prov, "archive_sha256": prov["sha256"],
            "rule": "same relative names and os.stat sizes; identical PROVENANCE.json (bytes of the images not compared)"})
    p3 = load_json(P3_MANIFEST); p3_sha = sha256_meta(P3_MANIFEST)
    ck.add("epfl.p3_manifest.is_v9_g2_cache", p3_sha.startswith(FROZEN_REFS["p3_manifest_sha256_prefix"]),
           {"sha256": p3_sha, "expected_prefix": FROZEN_REFS["p3_manifest_sha256_prefix"]})
    P = p3["pairs"]
    ck.add("epfl.p3_manifest.pairs_equal_disk", set(P) == set(disk), {"manifest": len(P), "disk": len(disk),
           "only_disk": sorted(set(disk) - set(P))[:10], "only_manifest": sorted(set(P) - set(disk))[:10]})
    status = collections.Counter(v["status"] for v in P.values())
    ex_al = sorted(k for k, v in P.items() if v["status"] == "excluded_alignment"); ex_sz = sorted(k for k, v in P.items() if v["status"] == "excluded_size")
    kept = sorted(k for k, v in P.items() if v["status"] == "kept")
    ck.add("epfl.exclusions.lists", ex_al == sorted(p3["excluded_alignment"]) and ex_sz == sorted(p3["excluded_size"]) and len(ex_al) == 33
           and len(ex_sz) == 24 and not (set(ex_al) & set(ex_sz)) and len(kept) + len(ex_al) + len(ex_sz) == len(P),
           {"status": dict(status), "alignment": len(ex_al), "size": len(ex_sz), "kept": len(kept),
            "historical": "R6 REG-P3 protocol: 33 alignment + 24 size exclusions, 420 kept (disjoint lists)"})
    mx = lambda k: max(abs(P[k]["best_shift_half_res"][0]), abs(P[k]["best_shift_half_res"][1]))
    ck.add("epfl.exclusions.alignment_rule", all(mx(k) >= 2 for k in ex_al) and all(mx(k) <= 1 for k in kept + ex_sz),
           {"rule": p3["check_rule"], "max_shift_kept_and_size": max(mx(k) for k in kept + ex_sz), "min_shift_alignment_excluded": min(mx(k) for k in ex_al)})
    ck.add("epfl.exclusions.size_rule", all(min(P[k]["shape"]) < WIN for k in ex_sz) and all(min(P[k]["shape"]) >= WIN for k in kept),
           {"rule": "min(H, W) < 512 -> excluded_size (checked after the alignment rule)", "size_excluded_widths": sorted(P[k]["shape"][1] for k in ex_sz)})
    # windows
    W = p3["windows"]; bad_w = []
    for k in kept:
        w = k.replace("/", "_"); H_, W_ = P[k]["shape"]
        centred = ((H_ - WIN) // 2, (W_ - WIN) // 2)
        if w not in W or W[w]["pair"] != k or (W[w]["y0"], W[w]["x0"]) != centred or (P[k]["y0"], P[k]["x0"]) != centred:
            bad_w.append(k)
    ck.add("epfl.windows.one_centred_window_per_kept_pair", len(W) == len(kept) and not bad_w and set(p3["roles"]) == set(W),
           {"windows": len(W), "kept": len(kept), "bad": bad_w[:10], "rule": p3["window_rule"] + "; (y0, x0) = ((H - 512) // 2, (W - 512) // 2)"})
    # near-duplicate closure over all 477 pairs (also through excluded pairs), compared with the REG-P3 units
    parent = {k: k for k in P}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    links = p3["near_duplicate_links"]
    for a, b, _ in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comps = collections.defaultdict(list)
    for k in sorted(P):
        comps[find(k)].append(k)
    group_of, bad_u = {}, []
    for members in comps.values():
        km = sorted(m for m in members if P[m]["status"] == "kept")
        name = "+".join(km) if km else "+".join(sorted(members))
        if km and {P[m]["unit"] for m in km} != {name}:
            bad_u.append(members)
        for m in members:
            group_of[m] = name
    units = p3["units"]
    ck.add("epfl.units.equal_recomputed_closure", not bad_u and set(units) == {group_of[k] for k in kept}
           and all(units[u]["pairs"] == u.split("+") for u in units),
           {"units": len(units), "links": len(links), "multi_member_components": sum(1 for c in comps.values() if len(c) > 1), "bad": bad_u[:5]})
    roles_u = {u: units[u]["role"] for u in units}
    ck.add("epfl.units.single_role_and_category", all(P[k]["role"] == roles_u[P[k]["unit"]] and units[P[k]["unit"]]["category"] == k.split("/")[0] for k in kept)
           and all(p3["roles"][k.replace("/", "_")] == P[k]["role"] for k in kept), "every kept pair carries its unit's role and category; window roles equal pair roles")
    by_cat = collections.defaultdict(list)
    for u in units:
        by_cat[units[u]["category"]].append(u)
    rec = {}
    for cat in sorted(by_cat):
        for i, u in enumerate(sorted(by_cat[cat], key=lambda u: hashlib.sha256(f"R6-REG-P3|EPFL|unit|{u}".encode()).hexdigest())):
            rec[u] = "DEV" if i < 8 else ("CAL" if i < 9 else "FIT")
    ck.add("epfl.roles.rule_recomputed_from_ids", rec == roles_u, {"rule": p3["role_rule"], "mismatch": sorted(u for u in units if rec.get(u) != roles_u[u])})
    ucount = collections.Counter(roles_u.values()); wcount = collections.Counter(P[k]["role"] for k in kept)
    ck.add("epfl.roles.counts", dict(ucount) == {"DEV": 72, "CAL": 9, "FIT": 330} and dict(wcount) == {"DEV": 74, "CAL": 10, "FIT": 336},
           {"units": dict(ucount), "windows": dict(wcount)})
    # cross-role relations through excluded near-duplicates
    excl_links = [[a, b, c] for a, b, c in links if P[a]["status"] != "kept" or P[b]["status"] != "kept"]
    owners = collections.defaultdict(list)
    for k, v in P.items():
        owners[v["sha256_rgb"]].append(f"{k}:RGB"); owners[v["sha256_nir"]].append(f"{k}:NIR")
    dups = {h: o for h, o in owners.items() if len(o) > 1}
    ck.add("epfl.sha256.no_exact_duplicates", not dups, {"files_hashed_by_r6": sum(len(o) for o in owners.values()), "duplicates": dups})
    cache_n = len([n for n in os.listdir(P3_CACHE) if n.endswith(".npz")])
    ck.add("epfl.child_cache_complete", cache_n == len(kept), {"r6_reg_p3_cache_npz": cache_n, "rule": "file count by os.listdir (not opened)"})
    total, nfiles = dir_bytes(EPFL_DIR)
    dm = load_json(DATA_MANIFEST_JSON)["items"]["epfl_rgb_nir"]
    ck.add("epfl.dir_bytes_equal_data_manifest", total == dm["bytes"], {"dir": str(EPFL_DIR), "stat_total_bytes": total, "files": nfiles, "data_manifest_bytes": dm["bytes"]})
    # P60's earlier read of 60 pairs (home copy)
    p60 = {k: load_json(v) for k, v in P60_JSONS.items()}
    ids = p60["full"]["image_ids"]
    ck.add("epfl.p60.id_list", hashlib.sha256(",".join(ids).encode()).hexdigest() == FROZEN_REFS["p60_id_list_sha256"] == p60["full"]["settings"]["id_list_sha256"]
           and len(ids) == 60 and set(ids) <= set(P) and p60["patch_ctf"]["image_ids"] == ids and set(p60["smoke"]["image_ids"]) <= set(ids),
           {"n": len(ids), "data": p60["full"]["settings"]["data"], "smoke": p60["smoke"]["image_ids"]})
    p60_dist = collections.Counter((P[k].get("role") or P[k]["status"]) for k in ids)
    shapes = collections.Counter("1024 wide" if P[k]["shape"][1] == 1024 else ("768 high" if P[k]["shape"][0] == 768 else "other") for k in P)
    g2 = sorted(V9_G2_UNITS.glob("*/p3/results/*.json")) if V9_G2_UNITS.is_dir() else []
    return {"p3": p3, "p3_sha": p3_sha, "prov": prov, "disk": disk, "per_cat": dict(sorted(per_cat.items())), "kept": kept, "ex_al": ex_al, "ex_sz": ex_sz,
            "group_of": group_of, "units": units, "links": links, "excl_links": excl_links, "p60_ids": ids, "p60_dist": dict(sorted(p60_dist.items())),
            "p60_smoke": p60["smoke"]["image_ids"], "shape_classes": dict(shapes), "dir_total": total, "cache_n": cache_n,
            "h_range_1024_wide": [min(v["shape"][0] for v in P.values() if v["shape"][1] == 1024), max(v["shape"][0] for v in P.values() if v["shape"][1] == 1024)],
            "w_range_768_high": [min(v["shape"][1] for v in P.values() if v["shape"][1] != 1024), max(v["shape"][1] for v in P.values() if v["shape"][1] != 1024)],
            "h_of_non_1024_wide": sorted({v["shape"][0] for v in P.values() if v["shape"][1] != 1024}),
            "exact_1024x768": sum(1 for v in P.values() if v["shape"] == [768, 1024]),
            "merged_units": sorted(u for u in units if "+" in u), "g2_result_files": len(g2),
            "g2_units": len([d for d in os.listdir(V9_G2_UNITS)]) if V9_G2_UNITS.is_dir() else 0}


# ------------------------------------------------------------------------------------------------------------ manifest rows
def bf_rows(B: dict) -> list:
    rows = []
    L = B["ledger"]
    for c, d in sorted(B["disk"].items()):
        f = B["files"][c]; H, W_, _ = f["full_shape"]
        rows.append({"dataset": "BF_SHG", "record_id": c, "group_id": c, "fixed_path": d["fixed"], "moving_path": d["moving"], "role": B["roles"][c],
                     "exposure": "used_for_design",
                     "source_version": f"zenodo4550300 full-size cores {d['split']}.zip; sha256_fixed={f['sha256_bf']}; sha256_moving={f['sha256_shg']} (R6 REG-P1 manifest)",
                     "inclusion": "include", "exclusion_reason": "", "original_split": d["split"], "group_basis": BF_GROUP_BASIS,
                     "native_width": str(W_), "native_height": str(H), "resize_factor": "0.25", "alignment_note": BF_ALIGN})
    reason = HOLD_REASON.format(prepare_job=L["prepare_job"], read_utc=L["read_utc"], results_commit=L["results_commit"])
    for r in sorted(B["test"], key=lambda r: r["core"]):
        c = r["core"]
        rows.append({"dataset": "BF_SHG", "record_id": c, "group_id": c, "fixed_path": str(V9_TEST_RAW / f"{c}_BF.tif"), "moving_path": str(V9_TEST_RAW / f"{c}_SHG.tif"),
                     "role": "DEV", "exposure": "used_for_design",
                     "source_version": (f"zenodo4550300 closed/test.zip members {r['bf_member']} / {r['shg_member']} (archive not opened by T00); "
                                        f"sha256_fixed={r['bf_sha256']}; sha256_moving={r['shg_sha256']} (V9-G1 frozen list); path = V9-G1 extracted byte copy"),
                     "inclusion": "exclude", "exclusion_reason": reason, "original_split": "test", "group_basis": BF_GROUP_BASIS,
                     "native_width": r["bf_w"], "native_height": r["bf_h"], "resize_factor": "0.25", "alignment_note": BF_ALIGN})
    return sorted(rows, key=lambda x: x["record_id"])


def epfl_rows(E: dict) -> list:
    rows = []; P = E["p3"]["pairs"]; arch = E["prov"]["sha256"]
    for k in sorted(P):
        v = P[k]; d = E["disk"][k]; H, W_ = v["shape"]; dy, dx = v["best_shift_half_res"]
        shift = f"best half-res shift (dy,dx)=({dy},{dx}), gradcorr zero {v['gradcorr_zero']:.3f} -> best {v['gradcorr_best']:.3f}"
        row = {"dataset": "EPFL_RGB_NIR", "record_id": k, "group_id": E["group_of"][k], "fixed_path": d["fixed"], "moving_path": d["moving"],
               "role": v.get("role", ""), "exposure": "used_for_design",
               "source_version": f"EPFL IVRL nirscene1.zip sha256 {arch}; sha256_fixed={v['sha256_rgb']}; sha256_moving={v['sha256_nir']} (R6 REG-P3 manifest)",
               "original_split": "none_official", "group_basis": EPFL_GROUP_BASIS, "native_width": str(W_), "native_height": str(H)}
        if v["status"] == "kept":
            row.update({"inclusion": "include", "exclusion_reason": "", "resize_factor": "1.0",
                        "alignment_note": (f"release alignment = authors' SIFT/RANSAC similarity registration (not pixel truth); REG-P3 check {shift}: kept "
                                           f"(< 2 half-res px; residual about 2 native px); window 512x512 native at y0={v['y0']} x0={v['x0']}; thresholds 10.5 / 2.25 native px")})
        elif v["status"] == "excluded_alignment":
            row.update({"inclusion": "exclude", "resize_factor": "",
                        "exclusion_reason": f"alignment: R6 REG-P3 rule max(|dy|,|dx|) >= 2 half-res px (>= 4 native px); best shift ({dy},{dx}); never used in any role",
                        "alignment_note": f"REG-P3 check {shift}: excluded for alignment"})
        elif v["status"] == "excluded_size":
            row.update({"inclusion": "exclude", "resize_factor": "",
                        "exclusion_reason": f"size: native {W_}x{H} narrower than the 512-px registration window (R6 REG-P3 size rule); never used in any role",
                        "alignment_note": f"REG-P3 check {shift}: alignment rule passed; excluded for size"})
        else:
            raise SystemExit(f"{k}: unknown status {v['status']}")
        if E["group_of"][k] != k and v["status"] != "kept":
            row["alignment_note"] += f"; near-duplicate of group {E['group_of'][k]} (would inherit its role if ever re-included)"
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------------------ exposure events
def events() -> list:
    ev = [
        {"id": "BF-E1", "dataset": "BF_SHG", "when_utc": "2026-10-11/12", "stage": "R6 REG-P1 (translation-only pilot)",
         "scope": "all 72 train / validation1 / validation2 cores", "use": "pixels read by prepare (x4 area cache); FIT 32 training, CAL 8 checkpoint / bandwidth selection, DEV 32 evaluation",
         "evidence": ["reports/R6/R6_REG_P1_PROTOCOL_FROZEN_20261011.yaml", "reports/R6/R6_REG_P1_REPORT_20261012.md"]},
        {"id": "BF-E2", "dataset": "BF_SHG", "when_utc": "2026-10-12", "stage": "R6 REG-P2 (rigid, shared optimiser)",
         "scope": "CAL 8 / DEV 32 (FIT for training)", "use": "rigid registrations with perturbations.json; MI / NMI / kernel / CoMIR / critics on CAL and DEV",
         "evidence": ["reports/R6/R6_REG_P2_PROTOCOL_FROZEN_20261012.yaml", "reports/R6/R6_REG_P2_REPORT_20261012.md"]},
        {"id": "BF-E3", "dataset": "BF_SHG", "when_utc": "2026-10-15", "stage": "V9-G1 (fit fraction x training mismatch)",
         "scope": "FIT 32 (25 % / 100 % subsets), CAL 8, DEV 32", "use": "48 trainable units + 4 kernel rows trained / fitted on FIT, selected on CAL, evaluated on DEV; DEV results reported",
         "evidence": ["reports/V9/V9_G1_PROTOCOL_FROZEN_20261015.yaml", "reports/V9/V9_G1_REPORT_20261015.md"]},
        {"id": "BF-E4", "dataset": "BF_SHG", "when_utc": "2026-10-15", "stage": "V9-G1 test-read fixture (job 1042290)",
         "scope": "3 DEV cores " + ", ".join(FROZEN_REFS["v9_g1_fixture_dev_cores"]), "use": "re-registration of DEV cores to check the test-read path (within DEV)",
         "evidence": ["reports/V9/V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml"]},
        {"id": "BF-E5", "dataset": "BF_SHG", "when_utc": "2026-10-15T23:36:09Z (read start)", "stage": "V9-G1 sealed-test confirmation read (one read)",
         "scope": "all 134 official test cores", "use": "archive extracted (byte copies under outputs_v9/reg_test/raw), cached, registered by 54 predictors; results and report committed",
         "evidence": ["reports/V9/V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml", "reports/V9/V9_TEST_READ_LEDGER.csv", "reports/V9/V9_G1_TEST_REPORT_20261016.md"]},
        {"id": "BF-X1", "dataset": "BF_SHG", "when_utc": "external (2020-2021)", "stage": "official split roles and published results",
         "scope": "validation1 25 / validation2 7 / test 134", "use": "officially validation1 = CNN hyper-parameters, validation2 = registration tuning; CoMIR (NeurIPS 2020) published results on the 134 test cores",
         "evidence": ["v10/package_v10_server_v1_20261016/docs/PROTOCOL_SOURCES_CN.md", "reports/R6/R6_DATA_NOTES.md"]},
        {"id": "EP-E1", "dataset": "EPFL_RGB_NIR", "when_utc": "2026-09-27/28", "stage": "P60 B-T2 pre-check (home copy)",
         "scope": "60 pairs (stratified, seed 20260927; 3-pair smoke subset; patch coarse-to-fine re-run on the same 60)",
         "use": "pixels read (square centre crop resized to 256 px, scripts/task_b_rgbnir.py); closed-form J* / MI / NMI registration, no training; its outcome is disclosed by REG-P3 as a reason for choosing EPFL",
         "evidence": ["reports/P60_TASK_B_RGBNIR_PREREG_FROZEN_20260927.md", "reports/P60_task_B_T2_rgbnir.json", "reports/P60_task_B_T2_rgbnir_patch_ctf.json"]},
        {"id": "EP-E2", "dataset": "EPFL_RGB_NIR", "when_utc": "2026-10-12", "stage": "R6 REG-P3 data audit + prepare + pilot",
         "scope": "all 477 pairs (alignment / size / near-duplicate checks); 420 windows cached", "use": "exclusions decided from pixels; FIT 336 windows training, CAL 10 selection, DEV 74 evaluation; DEV results reported",
         "evidence": ["reports/R6/R6_REG_P3_PROTOCOL_FROZEN_20261012.yaml", "reports/R6/R6_REG_P3_REPORT_20261012.md"]},
        {"id": "EP-E3", "dataset": "EPFL_RGB_NIR", "when_utc": "2026-10-15/16", "stage": "V9-G2 (fit fraction x training mismatch)",
         "scope": "FIT 330 units (25 % / 100 % subsets), CAL 9 units, DEV 72 units", "use": "48 trainable units + 4 kernel rows; per-unit DEV registrations computed; aggregate paused (execution note 7)",
         "evidence": ["reports/V9/V9_G2_PROTOCOL_FROZEN_20261015.yaml"]},
    ]
    for e in ev:
        e["evidence"] = [{"file": f, "sha256": sha256_meta(REPO / f), "first_commit": first_commit(f)} for f in e["evidence"]]
    return ev


# ------------------------------------------------------------------------------------------------------------ final checks on the rows
def row_checks(ck: Checks, rows: list, cols: list) -> dict:
    ck.add("manifest.columns_equal_template", all(list(r) == cols for r in rows), cols)
    keys = collections.Counter((r["dataset"], r["record_id"]) for r in rows)
    ck.add("manifest.unique_records", all(n == 1 for n in keys.values()), [k for k, n in keys.items() if n > 1])
    inc = [r for r in rows if r["inclusion"] == "include"]
    ck.add("manifest.vocabulary", all(r["inclusion"] in ("include", "exclude") and r["exposure"] in EXPOSURE_VOCAB for r in rows)
           and all(r["role"] in ROLE_VOCAB for r in inc) and all(r["exclusion_reason"].strip() for r in rows if r["inclusion"] == "exclude"),
           "inclusion in {include, exclude}; exposure in the preflight vocabulary; included roles in {FIT, CAL, DEV, CONFIRM}; every exclusion has a reason")
    ck.add("manifest.no_confirm_role", not any(r["role"] == "CONFIRM" for r in rows), "no group is evidenced as never read")
    grp = collections.defaultdict(set)
    for r in rows:
        if r["role"]:
            grp[(r["dataset"], r["group_id"])].add(r["role"])
    ck.add("manifest.group_single_role_all_rows", all(len(v) == 1 for v in grp.values()), {f"{d}|{g}": sorted(v) for (d, g), v in grp.items() if len(v) > 1})
    paths = collections.Counter()
    for r in rows:
        paths[r["fixed_path"]] += 1; paths[r["moving_path"]] += 1
    ck.add("manifest.paths_unique_all_rows", all(n == 1 for n in paths.values()), [p for p, n in paths.items() if n > 1])
    missing = []
    for r in inc:
        for side in ("fixed_path", "moving_path"):
            p = Path(r[side]); p = p if p.is_absolute() else DATA_ROOT / p
            if not p.resolve().is_file():
                missing.append(str(p))
    ck.add("manifest.included_files_exist", not missing, {"missing": missing[:10], "rule": "Path.resolve().is_file() as in preflight.py (no file opened)"})
    ck.add("manifest.no_included_path_in_closed_or_v9_test", not any("/closed/" in r[s] or str(V9_TEST_RAW) in r[s] for r in inc for s in ("fixed_path", "moving_path")),
           "phase 1: no included row points into the sealed archive directory or the V9-G1 test extraction")
    cnt = collections.Counter((r["dataset"], r["inclusion"], r["role"] or "-", r["original_split"]) for r in rows)
    return {"|".join(k): n for k, n in sorted(cnt.items())}


# ------------------------------------------------------------------------------------------------------------ notes
def render_notes(S: dict) -> str:
    B, E, C = S["bf_shg"], S["epfl"], S["counts"]
    ck_rows = "\n".join(f"| `{c['id']}` | {c['status']} |" for c in S["checks"])
    ev_rows = "\n".join(f"| {e['id']} | {e['when_utc']} | {e['stage']} | {e['scope']} | {e['use']} | "
                        + "; ".join(f"`{x['file']}` ({x['first_commit'].split(' ')[0]})" for x in e["evidence"]) + " |" for e in S["exposure_events"])
    pf = S["preflight"]
    pf_line = (f"`PREFLIGHT.json` status **{pf['status']}**, {pf['errors']} errors, {pf['warnings']} warnings; manifest sha256 recorded by the preflight "
               f"{'equals' if pf['manifest_sha256_match'] else 'DIFFERS FROM'} this manifest; current_commit {pf['current_commit']}, dirty tracked paths {pf['dirty_paths']}; "
               f"included records by role: {pf['included']}; declared independent groups {pf['independent_groups']}, declared CONFIRM groups {pf['confirm_groups']}.  "
               f"Per the package README this status only says the inventory was written: it does not certify host-adapter fidelity, V9 completion or "
               f"independent confirmation (preflight `does_not_certify`)."
               if pf.get("status") not in (None, "NOT_RUN") else "`PREFLIGHT.json`: NOT_RUN at the time this file was generated.")
    sp = B["slide_position_overlap"]
    ok = {c["id"]: c["status"] == "PASS" for c in S["checks"]}
    counts_agree = all(ok.get(i, False) for i in ("bf.disk.pairs_per_split", "bf.test.count_unique", "bf.test.core_list_is_frozen", "epfl.disk.pairs_per_category"))
    agree_line = ("The official pair counts agree with what is on disk for both datasets (BF–SHG train / validation1 / validation2 extracted, test inside the archive "
                  "by its frozen member list; EPFL pairs and categories)." if counts_agree else
                  "**The official pair counts DISAGREE with the server for at least one dataset — see the FAILED checks in section 8.**")
    conflict_ids = ("manifest.group_single_role_all_rows", "manifest.paths_unique_all_rows", "bf.sha256.no_exact_duplicates", "epfl.sha256.no_exact_duplicates",
                    "bf.test.ids_disjoint_from_trainval", "epfl.units.equal_recomputed_closure", "epfl.units.single_role_and_category")
    no_conflict = all(ok.get(i, False) for i in conflict_ids)
    conflict_line = "**No cross-role conflict was found.**" if no_conflict else "**A cross-role conflict check FAILED — see section 8 and the summary JSON.**"
    no_confirm = ok.get("manifest.no_confirm_role", False) and C["rows_never_read"] == 0 and C["rows_unknown"] == 0
    confirm_line = ("**No CONFIRM source exists in this round.**" if no_confirm else
                    "**CONFIRM status needs review: a row is never_read / unknown or carries a CONFIRM role — see the summary JSON.**")
    copies_line = ("Two on-server copies of EPFL are the same release (identical PROVENANCE.json and archive sha256; same names and sizes)" if ok.get("epfl.second_copy_is_same_release")
                   else "**The two on-server EPFL copies differ in names / sizes / provenance (check FAILED)**")
    n_excl = E["excluded_alignment"] + E["excluded_size"]
    g2_line = (f"V9-G2 at build time: {E['g2_result_files']} unit result files under {E['g2_units']} unit directories (V9-G2 execution note 7: two units await "
               f"rerun job 1042476 and the G2 aggregate is paused).")
    return f"""# V10 T00 — data manifest, preflight and exposure record (BF–SHG, EPFL RGB–NIR)

摘要：V10 沿用 R6/V9 的 BF–SHG 与 EPFL 分组、角色和排除。BF–SHG 的 {C['bf_total_records']} 个 core 与 EPFL 的 {C['epfl_total_records']} 个场景对都已被本项目读过（BF–SHG 的 {B['test_cores']} 个官方 test core 已由 V9-G1 在 {B['test_read_utc']} 做过一次确认读取），因此本轮**没有可登记为 CONFIRM 的从未读取的独立物理/场景组**；V10 的结果只能是开发/重复基准结果。134 个 test core 在本 manifest 中为 HOLD（phase 1 不读）。

Generated by `scripts/v10/t00_build_manifest.py` at {S['generated_utc']} on {S['host']} (git HEAD {S['git_head']}, builder sha256 {S['builder_sha256'][:16]}, V10 code hash at HEAD {S['code_hash']['head']}, working tree {S['code_hash']['worktree']}).
Metadata only: existing manifests / lists / ledgers and `os.stat` / `os.listdir`; no image, cache or checkpoint was opened; the sealed archive was not opened (no member list read, no hash computed; size by `os.stat`); no label was read; no trainer or host code was imported.
Build checks: {S['checks_passed']} / {len(S['checks'])} PASS{(' — FAILED: ' + ', '.join(S['checks_failed'])) if S['checks_failed'] else ''}.
Preflight (package `scripts/preflight.py`, run read-only on this manifest): {pf_line}

Files in `v10/T00/`: `V10_DATA_MANIFEST.csv` (one row per BF–SHG core / EPFL scene pair, exactly the package template's columns), `MANIFEST_SHA256.txt` (`sha256sum -c` format), `MANIFEST_BUILD_SUMMARY.json` (every count, check, relation and exposure event below, with input hashes), `PREFLIGHT.json`, `PREFLIGHT_RUN.log`, this file.

## 1. What the server has versus the official counts

| dataset | official | on this server | used by R6 / V9 (= V10 roles) |
|---|---|---|---|
| BF–SHG (Zenodo 4550300, full-size TMA cores, 2048 × 2048) | {OFFICIAL['BF_SHG']['pairs']} pairs: train {OFFICIAL['BF_SHG']['train']} / validation1 {OFFICIAL['BF_SHG']['validation1']} / validation2 {OFFICIAL['BF_SHG']['validation2']} / test {OFFICIAL['BF_SHG']['test']} | extracted: train {B['disk_pairs']['train']} / validation1 {B['disk_pairs']['validation1']} / validation2 {B['disk_pairs']['validation2']} pairs ({B['disk_files']} files `<core>_WB.tif` = BF, `<core>_MI.tif` = SHG); test {B['test_cores']} pairs inside the sealed `closed/test.zip` ({B['archive_bytes']} B; members `test/<core>_BF.tif`, `_SHG.tif` per the V9-G1 frozen list), plus V9-G1's byte copies of all {B['test_raw_files']} test files under `outputs_v9/reg_test/raw` | FIT {B['roles']['FIT']} + CAL {B['roles']['CAL']} (train), DEV {B['roles']['DEV']} (validation1 + validation2); the {B['test_cores']} test cores were V9-G1's sealed TEST, now read |
| EPFL RGB–NIR Scene (nirscene1) | {OFFICIAL['EPFL_RGB_NIR']['pairs']} pairs, {OFFICIAL['EPFL_RGB_NIR']['categories']} categories, stated as 1024 × 768 TIFF | {E['pairs']} pairs ({E['files']} TIFFs) in {E['categories']} categories ({', '.join(f'{k} {v}' for k, v in E['per_category'].items())}); sizes are the authors' registered crops: {E['shape_classes'].get('1024 wide', 0)} are 1024 wide × {E['h_range_1024_wide'][0]}–{E['h_range_1024_wide'][1]} high, {E['shape_classes'].get('768 high', 0)} are {E['w_range_768_high'][0]}–{E['w_range_768_high'][1]} wide × {'/'.join(str(h) for h in E['h_of_non_1024_wide'])} high, {E['exact_1024x768']} are exactly 1024 × 768; a second copy of the same release (same file names and sizes, same archive sha256) at `/home/infres/yinwang/CS_QMI/data/rgb_nir` (P60's source) | kept {E['kept']} windows in {E['units_total']} units: FIT {E['units']['FIT']} units / {E['windows']['FIT']} windows, CAL {E['units']['CAL']} / {E['windows']['CAL']}, DEV {E['units']['DEV']} / {E['windows']['DEV']}; excluded {E['excluded_alignment']} for alignment + {E['excluded_size']} for size (disjoint lists, never used in any role) |

{agree_line}  The stated EPFL resolution does not match: {E['exact_1024x768']} of {E['pairs']} pairs are exactly 1024 × 768.  The kept count {E['kept']} = {E['pairs']} − {E['excluded_alignment']} − {E['excluded_size']} is the REG-P3 status recorded for every pair and re-checked against the alignment and size rules (section 8), not arithmetic.

## 2. Manifest conventions (`V10_DATA_MANIFEST.csv`)

- `dataset` BF_SHG / EPFL_RGB_NIR (names of `configs/V10_PROTOCOL.json`); `record_id` = core id (BF–SHG) or `<category>/<id>` (EPFL); `group_id` = the R6/V9 independent unit: the core, or the REG-P3 scene unit (near-duplicates merged; {len(E['merged_units'])} merged units: {', '.join(E['merged_units'])}).
- `fixed_path` = reference (BF; RGB), `moving_path` = moving image (SHG; NIR), as in REG-P1 / REG-P3.  Relative paths resolve against the data root `{S['data_root']}`; the held test rows give V9-G1's absolute extracted-copy paths.
- `role` = V10 role = the R6/V9 role.  Rows excluded from every role (the {n_excl} EPFL exclusions) have an empty role.  The 134 BF–SHG test cores carry `role=DEV` (their only admissible V10 role: development / repeat benchmark) with `inclusion=exclude` and a `HOLD` reason: they are not part of the phase-1 manifest, and activating them needs a frozen V10 protocol and a new manifest version.  If activated they form a separate block (original_split = test) that is reported next to the V9-DEV block and never pooled with it, as V9 did.
- `exposure` uses the preflight vocabulary {{used_for_design, never_read, unknown}}.  Every row is `used_for_design`: read by an earlier project stage with results known before V10 runs (section 3).  For the test cores this means "read once as V9-G1's confirmation"; the V10 package's design snapshot (historical host commit 27117c9, 2026-10-16T01:49:26Z) predates the test results commit {B['results_commit']}, so the package design is not shown to have used the test results, but the project knew them before any V10 execution and the V10 rule (design §4.3) makes them development / repeat-benchmark data either way.  No row is `never_read`; no row is `unknown`.
- `source_version` records the release, the source archive (member) and the per-file sha256 recorded by R6 REG-P1 / REG-P3 prepare or the V9-G1 frozen list.  T00 did not re-hash any image (metadata-only rule); sizes were checked with `os.stat` where a reference exists.
- `original_split` = official split (BF–SHG train / validation1 / validation2 / test); EPFL has no official split (`none_official`).  `resize_factor` = model px per native px (BF–SHG 0.25: ×4 area downsampling to 512 px; EPFL 1.0: one 512 × 512 native window; empty for excluded EPFL pairs, which have no window).
- `alignment_note` carries the per-pair REG-P3 alignment-check values (EPFL) or the dataset-level alignment statement (BF–SHG).

## 3. Exposure ledger (what read what, and when)

| id | when (UTC) | stage | scope | use | evidence (first commit) |
|---|---|---|---|---|---|
{ev_rows}

V9-G1's one test read (ledger `reports/V9/V9_TEST_READ_LEDGER.csv`): archive sha256 {B['archive_sha256'][:16]}…, read_utc {B['test_read_utc']}, prepare job {B['prepare_job']}, registration jobs {B['registration_jobs']}, aggregate job {B['aggregate_job']}, results commit {B['results_commit']}.  {g2_line}

P60's 60 EPFL pairs (id-list sha256 {FROZEN_REFS['p60_id_list_sha256'][:16]}…, read from the home copy) fall in REG-P3's later roles as: {', '.join(f'{k} {v}' for k, v in E['p60_distribution'].items())}.  REG-P3's roles were therefore not drawn from scenes unseen by the project; P60 trained nothing, and REG-P3 disclosed P60 as a reason for choosing EPFL and its translation strata.

## 4. CONFIRM assessment

{confirm_line}  CONFIRM requires an evidenced, never-read, independent physical / scene group.  BF–SHG: all {B['trainval_cores']} train / validation cores were read by R6 REG-P1 / P2 and V9-G1, and all {B['test_cores']} test cores by V9-G1's confirmation read (BF-E5); a new download or a new crop / perturbation seed of the same cores does not restore unseen status.  EPFL: all {E['pairs']} pairs were read by REG-P3's audit (including the {E['excluded_alignment'] + E['excluded_size']} excluded ones, whose exclusion was computed from their pixels), {len(E['p60_ids'])} of them also by P60.  CONFIRM rows in the manifest: {C['rows_confirm']} (the preflight reports the declared CONFIRM groups as `confirmation_group_count_declared`).  V10 results on either dataset are development / repeat-benchmark results.
Other data on the server (for example Zurich Summer, `data/zurich_summer_5914759`, of which R6 REG-P3's draft smoke-ran 4 CAL windows before dropping it) are outside the V10 protocol; T00 did not assess or designate them, and any confirmation source would be a separate proposal.

## 5. Cross-role conflicts and unresolved relations

Checked conflicts (all rows, included and excluded): no group carries two roles; no image path appears twice; no recorded sha256 is shared by two files ({B['files_hashed']} BF–SHG files, {E['files_hashed']} EPFL files; checks `bf.sha256.no_exact_duplicates`, `epfl.sha256.no_exact_duplicates`); BF–SHG test core ids are disjoint from the 72 train / validation ids; every EPFL near-duplicate component lies inside one unit and one role.  {conflict_line}

Relations that metadata cannot resolve (recorded, not fixed):
- BF–SHG patient / tissue-block mapping is not published: the core is the verified unit; patient-level independence of FIT / CAL / DEV / TEST is unverified.  {sp['shared_positions']} TMA grid positions occur on both slides 1B and 2B (role pairs of those positions: {', '.join(f'{k} {v}' for k, v in sp['role_pairs'].items())}).  Whether 1B and 2B are sections of one block (same tissue at the same position) is not documented; if they were, those positions would be the same specimen in two roles.  No image was compared to test this (metadata-only rule).
- EPFL near-duplicate detection (REG-P3) compares only scenes of the same category whose half-resolution ÷ 16 thumbnails have identical shape, at correlation > 0.9; near-duplicates across categories, between differently shaped crops, or below 0.9 (for example consecutive captures of one place) are not detected.  {len(E['excluded_links'])} of the {E['links']} links involve an excluded scene (those scenes carry the linked unit as `group_id`).
- EPFL release alignment is the authors' SIFT / RANSAC similarity registration: REG-P3 keeps pairs whose best integer shift is within ±1 half-resolution px, i.e. a residual of about 2 native px; the strict 2.25-native-px threshold sits at that limit and recovers a known synthetic transform, not the sensors' true correspondence.
- {copies_line}: P60 read the home copy, R6 / V9 read the `/projects` copy.  Any path-based role check must treat them as the same records.

## 6. Parent / child relations (derived artefacts inherit the parent's group, role and exposure)

| parent | child | count on the server | relation |
|---|---|---|---|
| BF–SHG full core pair (2048 px TIFF) | R6 REG-P1 cache `outputs/R6_REG/cache/<core>.npz` | {B['cache']['r6_reg_p1_cache_npz']} | ×4 area downsampling to 512 px (both input paths); V9-G1 trained / registered on it |
| BF–SHG test member in `closed/test.zip` | V9-G1 byte copy `outputs_v9/reg_test/raw/<core>_{{BF,SHG}}.tif` → cache `outputs_v9/reg_test/cache/<core>.npz` | {B['test_raw_files']} files → {B['cache']['v9_g1_test_cache_npz']} | identical bytes (sha256 checked against the frozen list at extraction by V9-G1); same core, same exposure |
| BF–SHG core | 160-px training crops, disk support r = 129, the central 834-px full-resolution patch whose corners define the error | computed on the fly | crops / patches of one core never cross roles |
| BF–SHG core | Zenodo 3874362 cropped release (834 × 834 crops, transformed test set) | not on this server | if ever added: children of the same cores, same groups and exposure, never mixed with the full-size release |
| EPFL scene pair | R6 REG-P3 window cache `outputs/R6_REG_P3/cache/<category>_<id>.npz` | {E['cache_n']} | one centred 512 × 512 native window per kept pair; V9-G2 trained / registered on it |
| EPFL scene pair | P60 square centre crop resized to 256 px | 60 (on the fly) | read from the home copy, 2026-09-27/28 |
| EPFL release on `/projects` | the home copy `CS_QMI/data/rgb_nir` | {E['pairs']} pairs | same release, not independent data |

## 7. Phase-1 access rule

Phase-1 units (T01 / T02) may read only rows with `inclusion=include` (BF–SHG FIT / CAL / DEV from train / validation1 / validation2; EPFL FIT / CAL / DEV windows), and only small batches.  The {B['test_cores']} held test rows (`original_split=test`, `HOLD` reason) and everything under `closed/` or `outputs_v9/reg_test/` are not read in phase 1.  Check `manifest.no_included_path_in_closed_or_v9_test` enforces this for the manifest; code that bypasses the manifest must apply the same rule.

## 8. Build checks

| check | status |
|---|---|
{ck_rows}

Details of each check are in `MANIFEST_BUILD_SUMMARY.json` → `checks`.

## 9. Not done in T00 (explicit)

- No image bytes were re-hashed or decoded; per-file sha256 values are those recorded by R6 / V9 (re-hashing would read image content; it can be done as a separate CPU job if wanted).
- The sealed `closed/test.zip` was not opened: no member list was read and no hash was computed (its `os.stat` size equals the recorded size).
- `HOST_ADAPTER_AUDIT.md` (host interface mapping of `src/v9/reg/g1.py` / `r6_reg_pilot.py`) is a separate T00 deliverable, not part of this data record.
- Licence / redistribution: unchanged from the V9 data audit (BF–SHG CC BY 4.0; EPFL page has no licence text, usage cleared by the owner 2026-10-15); no data, image, crop or label is in the repository.
"""


# ------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "v10" / "T00")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else (Path.cwd() / a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = next(csv.reader(io.StringIO(read_meta(TEMPLATE).decode("utf-8-sig"))))
    ck = Checks()
    B = bf_section(ck)
    E = epfl_section(ck)
    rows = bf_rows(B) + epfl_rows(E)
    rows = [{c: r[c] for c in cols} for r in rows]
    counts_by_key = row_checks(ck, rows, cols)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    man_bytes = buf.getvalue().encode("utf-8")
    man_sha = hashlib.sha256(man_bytes).hexdigest()
    (out / "V10_DATA_MANIFEST.csv").write_bytes(man_bytes)

    pf_path = out / "PREFLIGHT.json"; pf = {"status": "NOT_RUN"}
    if pf_path.is_file():
        p = json.loads(pf_path.read_text(encoding="utf-8")); di = p.get("data_inventory", {})
        pf = {"status": p.get("status"), "errors": len(p.get("errors", [])), "warnings": len(di.get("warnings", [])),
              "manifest_sha256": di.get("sha256"), "manifest_sha256_match": di.get("sha256") == man_sha, "current_commit": p.get("current_commit"),
              "dirty_paths": len(p.get("dirty_paths", [])),
              "included": ", ".join(f"{x['dataset']} {x['role']} {x['count']}" for x in di.get("included_records_by_dataset_role", [])),
              "confirm_groups": di.get("confirmation_group_count_declared"), "independent_groups": di.get("independent_group_count_declared")}
        ck.add("preflight.recorded_this_manifest", pf["manifest_sha256_match"], {"preflight": pf["manifest_sha256"], "manifest": man_sha})

    P = E["p3"]["pairs"]
    try:
        head = git("rev-parse", "HEAD"); dirty = git("status", "--porcelain", "--untracked-files=no").splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        head, dirty = f"unknown ({exc})", []
    S = {
        "stage": "V10 T00 data manifest + exposure record", "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(), "python": sys.version.split()[0], "git_head": head, "git_dirty_tracked_paths": dirty,
        "builder": str(Path(__file__).resolve().relative_to(REPO)), "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "code_hash": code_hash(), "data_root": str(DATA_ROOT), "metadata_only": True,
        "manifest": {"file": "V10_DATA_MANIFEST.csv", "sha256": man_sha, "rows": len(rows), "columns": cols, "rows_by_dataset_inclusion_role_split": counts_by_key},
        "inputs": {str(p): {"sha256": sha256_meta(p), "bytes": os.stat(p).st_size} for p in
                   [TEMPLATE, P1_MANIFEST, V9_TEST_CORES, V9_LEDGER, DATA_MANIFEST_JSON, P3_MANIFEST, EPFL_DIR / "PROVENANCE.json", EPFL_HOME / "PROVENANCE.json", *P60_JSONS.values()]},
        "official": OFFICIAL,
        "counts": {"bf_total_records": len(B["disk"]) + len(B["test"]), "epfl_total_records": len(P),
                   "rows_never_read": sum(r["exposure"] == "never_read" for r in rows), "rows_unknown": sum(r["exposure"] == "unknown" for r in rows),
                   "rows_confirm": sum(r["role"] == "CONFIRM" for r in rows)},
        "bf_shg": {
            "disk_pairs": dict(collections.Counter(v["split"] for v in B["disk"].values())), "disk_files": sum(len(os.listdir(BF_DIR / s)) for s in BF_SPLITS),
            "trainval_cores": len(B["disk"]), "roles": dict(collections.Counter(B["roles"].values())),
            "roles_by_slide": {f"{r}|{s}": n for (r, s), n in sorted(collections.Counter((B["roles"][c], c.split("_", 1)[0]) for c in B["roles"]).items())},
            "test_cores": len(B["test"]), "test_by_slide": dict(collections.Counter(r["slide"] for r in B["test"])),
            "test_raw_files": sum(1 for r in B["test"] for k in ("BF", "SHG") if (V9_TEST_RAW / f"{r['core']}_{k}.tif").is_file()),
            "archive": str(BF_ARCHIVE), "archive_bytes": os.stat(BF_ARCHIVE).st_size, "archive_sha256": B["ledger"]["archive_sha256"],
            "test_read_utc": B["ledger"]["read_utc"], "prepare_job": B["ledger"]["prepare_job"], "registration_jobs": B["ledger"]["registration_jobs"],
            "aggregate_job": B["ledger"]["aggregate_job"], "results_commit": B["ledger"]["results_commit"], "protocol_commit": B["ledger"]["protocol_commit"],
            "file_size_pairs_trainval": B["sizes"], "zips": B["zips"], "cache": B["cache"], "dir_total_bytes": B["dir_total"],
            "files_hashed": 2 * (len(B["files"]) + len(B["test"])),
            "slide_position_overlap": {"shared_positions": len(B["shared_positions"]), "role_pairs": B["shared_position_role_pairs"], "positions": B["shared_positions"]},
            "p1_manifest_sha256": B["p1_sha"], "test_cores_csv_sha256": B["test_sha"]},
        "epfl": {
            "pairs": len(P), "files": 2 * len(E["disk"]), "categories": len(E["per_cat"]), "per_category": E["per_cat"], "shape_classes": E["shape_classes"],
            "h_range_1024_wide": E["h_range_1024_wide"], "w_range_768_high": E["w_range_768_high"], "h_of_non_1024_wide": E["h_of_non_1024_wide"],
            "exact_1024x768": E["exact_1024x768"], "kept": len(E["kept"]),
            "excluded_alignment": len(E["ex_al"]), "excluded_size": len(E["ex_sz"]), "excluded_alignment_ids": E["ex_al"], "excluded_size_ids": E["ex_sz"],
            "units_total": len(E["units"]), "units": dict(collections.Counter(u["role"] for u in E["units"].values())),
            "windows": dict(collections.Counter(P[k]["role"] for k in E["kept"])), "merged_units": E["merged_units"], "links": len(E["links"]),
            "excluded_links": E["excl_links"], "files_hashed": 2 * len(P), "cache_n": E["cache_n"], "dir_total_bytes": E["dir_total"],
            "p60_ids": E["p60_ids"], "p60_smoke_ids": E["p60_smoke"], "p60_distribution": E["p60_dist"],
            "g2_result_files": E["g2_result_files"], "g2_units": E["g2_units"], "p3_manifest_sha256": E["p3_sha"], "archive_sha256": E["prov"]["sha256"]},
        "exposure_events": events(),
        "confirm_assessment": {"eligible_groups": 0, "result": "no CONFIRM source in this round",
                               "reason": "every BF-SHG core (72 train/validation by R6/V9; 134 test by V9-G1's one confirmation read) and every EPFL pair (477 by R6 REG-P3's audit; 60 also by P60) has been read"},
        "preflight": pf,
    }
    S["checks"] = ck.items; S["checks_failed"] = ck.failed; S["checks_passed"] = len(ck.items) - len(ck.failed)
    (out / "MANIFEST_BUILD_SUMMARY.json").write_text(json.dumps(S, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "EXPOSURE_NOTES.md").write_text(render_notes(S), encoding="utf-8")
    lines = [f"{hashlib.sha256((out / f).read_bytes()).hexdigest()}  {f}" for f in ("V10_DATA_MANIFEST.csv", "MANIFEST_BUILD_SUMMARY.json")]
    (out / "MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out / "V10_DATA_MANIFEST.csv"), "sha256": man_sha, "rows": len(rows), "checks_passed": S["checks_passed"],
                      "checks_failed": ck.failed, "preflight": pf.get("status")}))
    return 1 if ck.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
