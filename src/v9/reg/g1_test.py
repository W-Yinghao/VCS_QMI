"""V9-G1 sealed-test confirmation read: ONE read of the BF/SHG test split with the G1 predictors exactly as selected on CAL.

Protocol reports/V9/V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml (owner go 2026-10-15 ~20:55 UTC 「这俩都可以提交」, main commit df4ced4; decision note 1
of reports/V9/V9_G1_PROTOCOL_FROZEN_20261015.yaml).  Test = Zenodo 4550300 test.zip, sealed under data/bf_shg_4550300/closed: 134 BF / SHG core pairs
named <core>_BF.tif / <core>_SHG.tif; no core is shared with REG-P1's train / validation cores.
  preprocessing  REG-P1's cmd_prepare verbatim (its core discovery and role assignment pointed at the extracted test files; role TEST) -> TEST_ROOT
  perturbations  REG-P2's generator (REG-P1's magnitude-stratified translations + θ uniform on ±30°, 10 per core) with the new seed tags TEST_TAGS
  registration   REG-P2's Inputs / register_images / RigidEnergy / optimise_rigid verbatim; F profiles for the first 5 test cores (as REG-P2's DEV)
  predictors     every G1 critic unit's CAL-chosen checkpoint, every CoMIR unit's comir.pt, the 4 kernel rows refitted by g1.fit_kernel_g1 on the
                 unit's FIT with the unit's CAL-chosen multiplier, MI / NMI computed on the test: no re-selection anywhere
  readings       g1_aggregate.main verbatim over the test records (cells, all contrasts, primary family with Holm over 4, interaction)
Guards: the formal steps refuse unless the protocol and the three frozen files (cores, perturbations, predictors) are committed and unchanged; the
archive, every test file and every predictor file must match their frozen sha256; the test root is prepared once; results are never overwritten.
The fixture (guards-need-smoke) runs the same functions on REG-P1 DEV cores packed into a fixture zip with the test naming; it never opens the
sealed archive and never creates the test root."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import resource
import struct
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from . import g1 as G
from . import g1_aggregate as A

P = G.P; Q = G.Q
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
TEST_ZIP = Path("/projects/EEG-foundation-model/yinghao/data/bf_shg_4550300/closed/test.zip")
ZIP_FACTS = {"sha256": "98f5eccdc0d6251643bacf39eda1c9847435508128521e3c6719952744820fd1", "md5": "caa9ae81837a41664ed7b61df81e96fa", "bytes": 1152656423}   # DATA_MANIFEST bf_shg_test_closed
G1_ROOT = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/reg")              # V9-G1 units: predictors, read-only
P1_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG")                         # REG-P1 cache: kernel refit on the unit's FIT, fixture comparison; read-only
P2_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P2")                      # REG-P2: MI / NMI DEV records and perturbations for the fixture; read-only
TEST_ROOT = Path(os.environ.get("V9_REG_TEST_ROOT", "/projects/EEG-foundation-model/yinghao/outputs_v9/reg_test"))
FIX_ROOT = Path(os.environ.get("V9_REG_TEST_FIXTURE_ROOT", "/projects/EEG-foundation-model/yinghao/outputs_v9/reg_test_fixture"))
PROTOCOL = REPO / "reports/V9/V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml"
CORES_CSV = REPO / "configs/V9_G1_TEST_CORES.csv"; PERT_JSON = REPO / "configs/V9_G1_TEST_PERTURBATIONS.json"; PRED_CSV = REPO / "configs/V9_G1_TEST_PREDICTORS.csv"
LEDGER = REPO / "reports/V9/V9_TEST_READ_LEDGER.csv"
TEST_TAGS = ("V9-G1-TEST-20261015|P1", "V9-G1-TEST-20261015|P2")                    # new fixed seed tags; REG-P2's own are P2_TAGS
P2_TAGS = ("R6-REG-P1", "R6-REG-P2")
N_PROFILE = 5
CODE_FILES = ("g1_test.py", "g1.py", "g1_aggregate.py", "r6_reg_pilot.py", "r6_reg_p2.py", "r6_reg_aggregate.py")
LEDGER_FIELDS = ["item", "archive", "archive_sha256", "archive_md5", "archive_bytes", "n_cores", "cores_csv_sha256", "perturbations_sha256", "predictors_sha256",
                 "protocol", "protocol_commit", "read_utc", "prepare_job", "registration_jobs", "aggregate_job", "results_commit"]


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------------------ perturbations
def perturbations_tagged(core: str, tag_t: str = P2_TAGS[0], tag_th: str = P2_TAGS[1]) -> list:
    """REG-P2's generator with its two seed tags as parameters: REG-P1's 10 translations (magnitude strata STRATA by index, uniform direction) and θ
    uniform on [−30°, 30°].  With P2_TAGS it equals Q.perturbations_p2(core) (tests/v9/test_reg_g1_test.py); the test uses TEST_TAGS."""
    g = np.random.default_rng(int(hashlib.sha256(f"{tag_t}|{core}".encode()).hexdigest()[:16], 16)); t = []
    for i in range(P.N_PERT):
        lo, hi = P.STRATA[i % 3]; r = math.sqrt(g.uniform(lo * lo, hi * hi)); th = g.uniform(0, 2 * math.pi); t.append((r * math.sin(th), r * math.cos(th)))
    gth = np.random.default_rng(Q.seed_of(f"{tag_th}|{core}")); th = gth.uniform(-Q.THMAX, Q.THMAX, P.N_PERT)
    return [[float(th[i]), float(t[i][0]), float(t[i][1])] for i in range(P.N_PERT)]


# ------------------------------------------------------------------------------------------------------------ inventory (archive members, no pixel decoding)
def tiff_header(b: bytes) -> dict:
    """REG-P1 read_tiff's tag parsing without decoding pixels: width, height, samples, bits, compression, planar (classic little-endian TIFF)."""
    keys = ("w", "h", "spp", "bits", "comp", "planar")
    if b[:4] != b"II*\x00":
        return {k: None for k in keys}
    off = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, off)[0]; tags = {}
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 16: 8}; fmt = {1: "B", 3: "H", 4: "I"}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack_from("<HHII", b, off + 2 + 12 * i)
        if typ not in fmt:
            continue
        nbytes = sizes[typ] * cnt; pos = off + 2 + 12 * i + 8 if nbytes <= 4 else val
        tags[tag] = list(struct.unpack_from("<" + fmt[typ] * cnt, b, pos))
    return dict(zip(keys, (tags[256][0], tags[257][0], tags.get(277, [1])[0], tags.get(258, [8])[0], tags.get(259, [1])[0], tags.get(284, [1])[0])))


def inventory(zpath) -> list:
    """One row per core: BF / SHG member name, bytes, sha256 of the member's bytes, TIFF header facts."""
    z = zipfile.ZipFile(zpath); by = {}
    for i in z.infolist():
        if i.is_dir():
            continue
        name = i.filename.rsplit("/", 1)[-1]
        if not name.endswith(".tif") or name[:-4].rsplit("_", 1)[-1] not in ("BF", "SHG"):
            raise SystemExit(f"unexpected archive member {i.filename}")
        core, kind = name[:-4].rsplit("_", 1); k = kind.lower(); b = z.read(i)
        by.setdefault(core, {"core": core, "slide": core.split("_")[0]}).update(
            {f"{k}_member": i.filename, f"{k}_bytes": len(b), f"{k}_sha256": hashlib.sha256(b).hexdigest(), **{f"{k}_{x}": v for x, v in tiff_header(b).items()}})
    bad = [c for c, r in by.items() if "bf_sha256" not in r or "shg_sha256" not in r]
    if bad:
        raise SystemExit(f"cores without a BF / SHG pair: {bad}")
    return [by[c] for c in sorted(by)]


def write_cores_csv(rows, path):
    keys = ["core", "slide"] + [f"{k}_{x}" for k in ("bf", "shg") for x in ("member", "bytes", "sha256", "w", "h", "spp", "bits", "comp", "planar")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_cores_csv(path=CORES_CSV) -> dict:
    return {r["core"]: r for r in csv.DictReader(open(path))}


# ------------------------------------------------------------------------------------------------------------ predictors (fixed before the read)
def predictor_rows(g1_root=G1_ROOT) -> list:
    rows = []
    for uid, m, f, e, s in G.all_units() + G.kernel_units():
        res = Path(g1_root) / "units" / uid / "p2" / "results" / f"{m}_s{s}.json"; r = json.load(open(res)); run = Path(g1_root) / "units" / uid / "run"
        if m == "comir":
            file, choice = run / "comir.pt", "comir.pt (REG-P1 CoMIR recipe; nothing selected)"
        elif m == "kernel":
            file, choice = res, f"chosen_mult={r['chosen_mult']:g} (CAL); refit by g1.fit_kernel_g1 on the unit's {len(r['fit'])} FIT cores, eps {e:g}, seed {s}"
        else:
            file, choice = run / r["chosen_ckpt"], r["chosen_ckpt"]
        rows.append({"predictor_id": uid, "method": m, "fit_fraction": f, "train_mismatch": e, "seed": s, "choice": choice, "file": str(file), "file_sha256": sha(file),
                     "g1_result": str(res), "g1_result_sha256": sha(res), "g1_dev_sec_per_registration": r.get("dev_sec_per_registration")})
    for m in ("mi", "nmi"):
        r = json.load(open(P2_ROOT / "results" / f"{m}.json"))
        rows.append({"predictor_id": m, "method": m, "fit_fraction": "", "train_mismatch": "", "seed": "", "choice": "per-image soft 64-bin energy (REG-P2), computed on the test",
                     "file": "", "file_sha256": "", "g1_result": "", "g1_result_sha256": "", "g1_dev_sec_per_registration": r.get("dev_sec_per_registration")})
    return rows


def write_predictors_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_predictors(path=PRED_CSV) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        r = dict(r); r["fit_fraction"] = float(r["fit_fraction"]) if r["fit_fraction"] else None
        r["train_mismatch"] = float(r["train_mismatch"]) if r["train_mismatch"] else None; r["seed"] = int(r["seed"]) if r["seed"] else None; out[r["predictor_id"]] = r
    return out


# ------------------------------------------------------------------------------------------------------------ guards
def refuse_if_frozen():
    if PROTOCOL.exists():
        raise SystemExit(f"[guard] {PROTOCOL.name} exists: the test-side files are frozen and are not regenerated")


def require_frozen(files=None, repo=REPO) -> dict:
    """The protocol and the frozen files must be committed and unchanged against HEAD, with the sha256 recorded in the protocol."""
    import yaml
    files = [Path(f) for f in (files or (PROTOCOL, CORES_CSV, PERT_JSON, PRED_CSV))]
    for f in files:
        if not f.exists():
            raise SystemExit(f"[guard] {f} missing: freeze first")
        try:
            rel = str(f.resolve().relative_to(Path(repo).resolve()))
        except ValueError:
            raise SystemExit(f"[guard] {f} is outside the repository")
        if subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel], capture_output=True).returncode != 0:
            raise SystemExit(f"[guard] {rel} is not committed")
        if subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel]).returncode != 0:
            raise SystemExit(f"[guard] {rel} differs from HEAD")
    fz = yaml.safe_load(open(files[0]))["frozen_files"]
    for f in files[1:]:
        if sha(f) != fz.get(f.name):
            raise SystemExit(f"[guard] sha256({f.name}) differs from the frozen value")
    commit = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h", "--", str(files[0])], capture_output=True, text=True).stdout.strip()
    return {"protocol_commit": commit, **fz}


def ledger_update(path=LEDGER, **kw):
    rows = list(csv.DictReader(open(path))) if Path(path).exists() else []
    row = next((r for r in rows if r["item"] == "bf_shg_test_closed"), None)
    if row is None:
        row = {k: "" for k in LEDGER_FIELDS}; row["item"] = "bf_shg_test_closed"; rows.append(row)
    for k, v in kw.items():
        if k == "read_utc" and row.get(k):
            continue                                                             # the first preparation start stays the read time
        row[k] = ";".join(x for x in (row.get(k), str(v)) if x) if k in ("prepare_job", "registration_jobs", "aggregate_job") else str(v)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS); w.writeheader(); w.writerows(rows)
    return row


# ------------------------------------------------------------------------------------------------------------ preparation (REG-P1's builder, verbatim)
def prepare(root, zpath, cores: dict, formal: bool = True, ledger=None) -> dict:
    root, zpath = Path(root), Path(zpath)
    if (root / "manifest.json").exists():
        raise SystemExit(f"[guard] {root} is already prepared: one read, nothing is rebuilt")
    if formal and (zpath != TEST_ZIP or root != TEST_ROOT):
        raise SystemExit("[guard] the formal preparation reads only the sealed archive into the test root")
    if not formal and (zpath == TEST_ZIP or root == TEST_ROOT):
        raise SystemExit("[guard] the fixture never opens the sealed archive or writes the test root")
    if ledger is not None:
        ledger_update(ledger, read_utc=now(), prepare_job=os.environ.get("SLURM_JOB_ID", "local"))
    if formal:
        h, md = hashlib.sha256(), hashlib.md5()
        with open(zpath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk); md.update(chunk)
        if (h.hexdigest(), md.hexdigest(), zpath.stat().st_size) != (ZIP_FACTS["sha256"], ZIP_FACTS["md5"], ZIP_FACTS["bytes"]):
            raise SystemExit("[guard] the archive differs from DATA_MANIFEST bf_shg_test_closed")
    raw = root / "raw"; raw.mkdir(parents=True, exist_ok=True); z = zipfile.ZipFile(zpath); files = {}
    for core, row in sorted(cores.items()):
        for kind in ("BF", "SHG"):
            b = z.read(row[f"{kind.lower()}_member"])
            if hashlib.sha256(b).hexdigest() != row[f"{kind.lower()}_sha256"]:
                raise SystemExit(f"[guard] {core} {kind}: sha256 differs from the frozen core list")
            (raw / f"{core}_{kind}.tif").write_bytes(b)
        files[core] = ("test", raw / f"{core}_BF.tif", raw / f"{core}_SHG.tif")
    orig = (P.core_files, P.roles); P.core_files = lambda: files; P.roles = lambda cs: {c: "TEST" for c in cs}
    try:
        P.cmd_prepare(argparse.Namespace(root=root, smoke=False))                 # REG-P1's builder: ×4 area down to 512 px, both input paths, sha256 per file
    finally:
        P.core_files, P.roles = orig
    m = json.load(open(root / "manifest.json"))
    bad = [c for c, r in cores.items() if (m["files"][c]["sha256_bf"], m["files"][c]["sha256_shg"]) != (r["bf_sha256"], r["shg_sha256"])]
    if bad or sorted(m["roles"]) != sorted(cores):
        raise SystemExit(f"[guard] prepared manifest differs from the frozen core list: {bad[:5]}")
    return m


# ------------------------------------------------------------------------------------------------------------ registration (REG-P2, verbatim parts)
def register(pid, root, pert: dict, preds: dict, n_pert: int = P.N_PERT, out=None) -> Path:
    """One predictor on every core of `pert` (REG-P2's register_images); the G1 CAL choice is used as is; the result is written once."""
    row = preds[pid]; out = Path(out or Path(root) / "results"); dst = out / f"{pid}.json"
    if dst.exists():
        raise SystemExit(f"[guard] {dst} exists: one read, results are never overwritten")
    for k in ("file", "g1_result"):
        if row[k] and sha(row[k]) != row[f"{k}_sha256"]:
            raise SystemExit(f"[guard] {row[k]} changed since the freeze")
    torch, nn, F = P.torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); m = row["method"]; seed = row["seed"]
    cores = sorted(pert); I = Q.Inputs(torch, F, Path(root), pert, dev); prof = cores[:N_PROFILE]
    reg = lambda maps, kind, kw=None: Q.register_images(torch, F, cores, pert, maps, kind, dev, kw, n_pert, prof)
    res = {"predictor_id": pid, "method": m, "seed": seed, "fit_fraction": row["fit_fraction"], "train_mismatch": row["train_mismatch"], "choice": row["choice"],
           "predictor_file": row["file"], "predictor_sha256": row["file_sha256"], "root": str(root), "n_cores": len(cores), "n_pert": n_pert,
           "selection": "none: the G1 CAL choice is used as is", "r_omega": Q.R_OMEGA, "rho": Q.RHO, "box_exact": Q.BOX1, "box_grid": Q.BOX2, "n_pos": P.N_POS,
           "grid": {"theta_step_deg": Q.TH_STEP, "theta_max_deg": Q.THMAX, "t_step": P.GRID_STEP, "t_max": Q.TMAX}, "starts": P.N_STARTS, "adam": [P.N_ADAM, P.ADAM_LR],
           "corner_half_px512": Q.HALF834, "code_hash": code_hash(), "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "started_utc": now()}
    t0 = time.time()
    if m in ("mi", "nmi"):
        def maps(c, i):
            U, V = I.reference(c)[0], I.moving(c, i)[0]; rg = lambda M: tuple(float(x) for x in np.percentile(M.cpu().numpy(), [0.5, 99.5]))
            return U, V, {"ranges": (rg(U), rg(V))}
        res["test"] = reg(maps, m)
    elif m == "kernel":
        r1 = json.load(open(row["g1_result"])); kw = G.fit_kernel_g1(torch, F, P1_ROOT, r1["fit"], r1["chosen_mult"], seed, dev, float(row["train_mismatch"]))
        res.update({"fit": r1["fit"], "chosen_mult": r1["chosen_mult"], "kernel_h": kw["h"], "kernel_median": kw["median"]})
        res["test"] = reg(lambda c, i: (I.reference(c)[0], I.moving(c, i)[0], {}), "kernel", {k: kw[k] for k in ("W", "b", "w")})
    elif m == "comir":
        run = Path(row["file"]).parent; st = torch.load(row["file"], map_location=dev); mA, mB, _ = P.comir_models(torch, seed, dev)
        mA.load_state_dict(st["A"]); mB.load_state_dict(st["B"]); mA.eval(); mB.eval(); LU = {}; res["comir_run"] = json.load(open(run / "comir.json"))
        for p_ in list(mA.parameters()) + list(mB.parameters()):
            p_.requires_grad_(False)
        for mod in mA.modules():                                                 # as REG-P2: no activation checkpointing in the refinement (same values)
            if hasattr(mod, "efficient"):
                mod.efficient = False
        def maps(c, i):                                                          # encode-once: A = SHG (log1p path), B = BF RGB, as REG-P1 / REG-P2
            with torch.no_grad():
                if c not in LU:
                    LU[c] = mB(I.reference(c)[1][None])[0]
                return LU[c], mA(I.moving(c, i)[1][None])[0], {}
        res["test"] = reg(maps, "comir")
    else:                                                                        # vcs | logistic | infonce: the unit's CAL-chosen checkpoint only
        run = Path(row["file"]).parent; res["train"] = json.load(open(run / "train.json")); res["chosen_ckpt"] = Path(row["file"]).name
        eU, eV, cr = (x.to(dev).eval() for x in P.build_models(nn)); st = torch.load(row["file"], map_location=dev)
        eU.load_state_dict(st["eU"]); eV.load_state_dict(st["eV"]); cr.load_state_dict(st["critic"]); DU = {}
        for p_ in list(eU.parameters()) + list(eV.parameters()) + list(cr.parameters()):
            p_.requires_grad_(False)
        def maps(c, i):
            if c not in DU:
                with torch.no_grad():
                    DU[c] = eU(I.reference(c)[1][None])[0]
            return DU[c], I.moving(c, i)[1], {}
        res["test"] = reg(maps, "critic", {"eV": eV, "critic": cr})
    Q.summarise(res, "test")
    res["seconds_total"] = time.time() - t0; res["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20; res["finished_utc"] = now()
    out.mkdir(parents=True, exist_ok=True); tmp = out / f".{pid}.json.tmp"; json.dump(res, open(tmp, "w"), indent=1); tmp.rename(dst)
    print(f"[test-read {pid}] {len(cores)} cores × {n_pert}: {res['seconds_total']:.0f} s, {res['test_sec_per_registration']:.2f} s / registration -> {dst}", flush=True)
    return dst


# ------------------------------------------------------------------------------------------------------------ readings (g1_aggregate.main, verbatim)
def dev_view(r: dict) -> dict:
    """The test records under the names g1_aggregate.main reads (dev, dev_*); nothing else changes."""
    v = {k: x for k, x in r.items() if k != "test" and not k.startswith("test_")}; v["dev"] = r["test"]
    v.update({"dev_" + k[len("test_"):]: x for k, x in r.items() if k.startswith("test_")}); return v


def aggregate(root, out, preds=None, split_label="BF/SHG sealed TEST (Zenodo 4550300 test.zip), read once") -> dict:
    preds = preds or load_predictors(); runs, missing = {}, []
    for pid, row in preds.items():
        f = Path(root) / "results" / f"{pid}.json"
        if not f.exists():
            missing.append(pid); continue
        key = (row["method"], None, None, None) if row["method"] in ("mi", "nmi") else (row["method"], row["fit_fraction"], row["train_mismatch"], row["seed"])
        runs[key] = dev_view(json.load(open(f)))
    chk = {"mi_nmi": "computed on the test with REG-P2's per-image energy (not reused)"}; out = Path(out); orig = A.load_runs
    A.load_runs = lambda smoke=False: (runs, missing, chk)
    try:
        A.main(out)
    finally:
        A.load_runs = orig
    (out / "g1_results.json").replace(out / "g1_test_results.json"); (out / "g1_tables.md").replace(out / "g1_test_tables.md")
    j = json.load(open(out / "g1_test_results.json")); j.update({"split": split_label, "protocol": str(PROTOCOL.relative_to(REPO)), "n_results": len(runs), "code_hash": code_hash()})
    json.dump(j, open(out / "g1_test_results.json", "w"), indent=1); return j


# ------------------------------------------------------------------------------------------------------------ fixture (non-test smoke of the same path)
def fixture(n_cores=3, n_pert=1) -> int:
    """REG-P1 DEV cores (validation, already read by G1) packed with the test naming; every formal function on them; reproduction of the G1 / REG-P2
    DEV records of the same images and perturbations; every guard exercised.  Never opens the sealed archive; never creates the test root."""
    frozen = require_frozen(); preds = load_predictors(); root = FIX_ROOT / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"); root.mkdir(parents=True)
    test_root_existed = TEST_ROOT.exists(); m1 = P.load_manifest(P1_ROOT, False); cores = sorted(c for c, r in m1["roles"].items() if r == "DEV")[:n_cores]; files = P.core_files()
    zf = root / "fixture.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for c in cores:
            z.write(files[c][1], f"test/{c}_BF.tif"); z.write(files[c][2], f"test/{c}_SHG.tif")
    rows = inventory(zf); write_cores_csv(rows, root / "cores.csv"); cr = read_cores_csv(root / "cores.csv"); checks = {}
    for name, fn in (("formal_prepare_refuses_fixture", lambda: prepare(root, zf, cr, formal=True)),
                     ("fixture_prepare_refuses_test_root", lambda: prepare(TEST_ROOT, zf, cr, formal=False))):
        try:
            fn(); checks[name] = "NOT REFUSED"
        except SystemExit as e:
            checks[name] = f"refused ({e})"
    prepare(root, zf, cr, formal=False, ledger=root / "ledger_fixture.csv")
    same = {}
    for c in cores:
        a, b = np.load(root / "cache" / f"{c}.npz"), np.load(P1_ROOT / "cache" / f"{c}.npz")
        same[c] = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    try:
        prepare(root, zf, cr, formal=False); checks["prepare_twice"] = "NOT REFUSED"
    except SystemExit as e:
        checks["prepare_twice"] = f"refused ({e})"
    pert = {c: perturbations_tagged(c, *P2_TAGS) for c in cores}; p2 = json.load(open(P2_ROOT / "perturbations.json"))
    checks["perturbations_equal_reg_p2"] = all(pert[c] == p2[c] for c in cores); json.dump(pert, open(root / "perturbations.json", "w"), indent=1)
    rep = {}
    for pid in preds:
        register(pid, root, pert, preds, n_pert=n_pert); r = json.load(open(root / "results" / f"{pid}.json"))
        ref = json.load(open(preds[pid]["g1_result"] or P2_ROOT / "results" / f"{pid}.json")); pairs = [(r["test"][c][i], ref["dev"][c][i]) for c in cores for i in range(n_pert)]
        rep[pid] = {"max_abs_d_err_full": max(abs(a["err_full"] - b["err_full"]) for a, b in pairs), "max_abs_d_theta_hat": max(abs(a["theta_hat"] - b["theta_hat"]) for a, b in pairs),
                    "flips42": sum((a["err_full"] < 42) != (b["err_full"] < 42) for a, b in pairs), "flips9": sum((a["err_full"] < 9) != (b["err_full"] < 9) for a, b in pairs),
                    "sec_per_registration": r["test_sec_per_registration"], "seconds_total": r["seconds_total"]}
    try:
        register(next(iter(preds)), root, pert, preds, n_pert=n_pert); checks["register_twice"] = "NOT REFUSED"
    except SystemExit as e:
        checks["register_twice"] = f"refused ({e})"
    agg = aggregate(root, root / "aggregate", preds, split_label="FIXTURE (REG-P1 DEV cores, already read by G1) - code check only")
    ok_guards = all(v.startswith("refused") for k, v in checks.items() if k != "perturbations_equal_reg_p2")
    rep_ok = sum(v["max_abs_d_err_full"] < 0.5 for v in rep.values())
    summary = {"root": str(root), "frozen": frozen, "cores": cores, "n_pert": n_pert, "preprocessing_identical_to_reg_p1": same, "checks": checks,
               "reproduction": rep, "reproduced_within_0.5px": f"{rep_ok}/{len(rep)}", "test_root_created": (not test_root_existed) and TEST_ROOT.exists(),
               "aggregate_primary_keys": sorted(agg["primary"]), "aggregate_missing": agg["missing"], "code_hash": code_hash(), "device_note": "G1 DEV records came from mixed GPUs; small numerical differences are possible"}
    json.dump(summary, open(root / "fixture_report.json", "w"), indent=1)
    ok = all(same.values()) and ok_guards and checks["perturbations_equal_reg_p2"] and rep_ok == len(rep) and not summary["test_root_created"] and len(agg["primary"]) == 4
    print(json.dumps({k: summary[k] for k in ("preprocessing_identical_to_reg_p1", "checks", "reproduced_within_0.5px", "test_root_created", "aggregate_primary_keys")}, indent=1))
    print(f"[fixture] {'PASS' if ok else 'FAIL'} -> {root / 'fixture_report.json'}", flush=True); return 0 if ok else 1
