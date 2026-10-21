"""TEST_REPEAT134 input diagnostic: one core whose prepared arrays differ from V9-G1's cache (pinned Slurm CPU job only).

    diagnose_inputs.py <core>

Scope (DIAGNOSTIC_AUTHORIZATION_V4D.json): numbers only.  The job reads the core's two raw members, the V10-rebuilt and
the V9-G1 prepared arrays of that core, the two prepare manifests and the frozen member inventories, and stats the
prepared cache files.  It rebuilds the core with REG-P1's frozen cmd_prepare in fresh child processes under the default
numpy CPU dispatch (twice) and with the AVX-512 (X86_V4) or all SIMD dispatch targets disabled (float64 log1p is the only
dispatch-dependent kernel of cmd_prepare in numpy 2.5.2).  It evaluates nothing, computes no perturbation or truth,
reads no registration result and changes no record, lock or prepared input; the rebuilds live in a job-local directory
that is deleted at the end.  Output: <bundle>/diagnostics/<core>/<job>/DIAG.json (written once).
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402
import prepare_inputs as PI  # noqa: E402

AUTHORIZATION = "DIAGNOSTIC_AUTHORIZATION_V4D.json"
SCHEMA = "v10.test_repeat134.input_diagnostic.v1"
MARGIN = 120                                           # s, node vs NFS-server clock
# (tag, NPY_DISABLE_CPU_FEATURES): default twice (determinism), then without AVX-512, then baseline SIMD only
VARIANTS = (("default_1", ""), ("default_2", ""), ("no_X86_V4", "X86_V4"), ("baseline_X86_V2", "X86_V3 X86_V4"))


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ------------------------------------------------------------------------------------------------------------ numbers
def _ordered32(x):
    """float32 -> int64 that is monotone in the float order (ULP distance = |difference|)."""
    import numpy as np
    i = x.view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def array_stats(a, b) -> dict:
    """a: the array under test, b: the reference.  Counts and magnitudes only; no element value is reported."""
    import numpy as np
    out = {"dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)]}
    if a.shape != b.shape:
        return {**out, "comparable": False, "identical": False}
    fa, fb = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.isnan(fa), np.isnan(fb)
    diff = ~((fa == fb) | (na & nb))
    num = diff & ~(na | nb)
    d = np.where(num, np.abs(fa - fb), 0.0)
    n = int(diff.sum())
    fin = np.isfinite(fb)
    out.update(comparable=True, identical=bool(n == 0 and a.dtype == b.dtype), n_elements=int(a.size), n_differ=n,
               frac_differ=n / a.size if a.size else 0.0, max_abs=float(d.max()) if a.size else 0.0,
               mean_abs=float(d.mean()) if a.size else 0.0, mean_abs_over_differing=float(d[num].mean()) if num.any() else 0.0,
               reference_abs_max=float(np.abs(fb[fin]).max()) if fin.any() else None,
               nan=[int(na.sum()), int(nb.sum())], nan_mismatch=int((na ^ nb).sum()))
    if n and a.dtype == b.dtype == np.float32:
        u = np.abs(_ordered32(a) - _ordered32(b))[diff]
        out["ulp"] = {"max": int(u.max()), "eq1": int((u == 1).sum()), "eq2": int((u == 2).sum()),
                      "3_to_16": int(((u >= 3) & (u <= 16)).sum()), "gt16": int((u > 16).sum())}
    if n:
        idx = np.argwhere(diff)
        out["differing_index_range"] = [[int(idx[:, k].min()), int(idx[:, k].max())] for k in range(a.ndim)]
        if n <= 16:
            out["differing_indices"] = idx.tolist()                         # positions only
        both = fin & np.isfinite(fa)
        x, y = fb[both], fa[both]
        sxx = float(((x - x.mean()) ** 2).sum()) if x.size else 0.0
        if sxx > 0:                                                      # a - (s b + c): is the change a re-normalisation?
            s = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
            c = float(y.mean() - s * x.mean())
            out["affine_fit_a_on_b"] = {"slope_minus_1": s - 1.0, "intercept": c,
                                        "max_abs_residual": float(np.abs(y - (s * x + c)).max())}
    if a.ndim == 3:
        ax = a.ndim - 1 if a.shape[-1] <= 4 else (0 if a.shape[0] <= 4 else None)
        if ax is not None:
            out["channel_axis"] = ax
            out["per_channel"] = [array_stats(np.take(a, k, axis=ax), np.take(b, k, axis=ax)) for k in range(a.shape[ax])]
    return out


def compare(p: Path, q: Path) -> dict:
    """Every array of npz p against npz q (q is the reference)."""
    import numpy as np
    with np.load(p) as a, np.load(q) as b:
        keys = sorted(set(a.files) | set(b.files))
        arrays = {k: (array_stats(a[k], b[k]) if k in a.files and k in b.files else
                      {"present": [k in a.files, k in b.files], "identical": False}) for k in keys}
        return {"keys_equal": sorted(a.files) == sorted(b.files) == sorted(PI.ARRAY_KEYS), "arrays": arrays,
                "identical": sorted(a.files) == sorted(b.files) and all(v["identical"] for v in arrays.values())}


def brief(cmp: dict) -> dict:
    return {"identical": cmp["identical"],
            "differing": {k: {"n_differ": v.get("n_differ"), "max_ulp": v.get("ulp", {}).get("max"), "max_abs": v.get("max_abs")}
                          for k, v in cmp["arrays"].items() if not v["identical"]}}


def npz_meta(path: Path) -> dict:
    st = path.stat()
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        members = [{"name": i.filename, "date_time": list(i.date_time), "crc32": f"{i.CRC:08x}", "file_size": i.file_size,
                    "compress_size": i.compress_size} for i in z.infolist()]
    return {"path": str(path), "sha256": C.sha(path), "bytes": st.st_size, "mtime_utc": iso(st.st_mtime),
            "mtime_unix": st.st_mtime, "ctime_utc": iso(st.st_ctime), "zip_testzip_first_bad_member": bad, "members": members}


def mtimes(root: Path, cores: list, core: str, window: dict | None) -> dict:
    """stat only: the cache files of every core, in cmd_prepare's write order (sorted core ids)."""
    order = sorted(cores)
    t = {c: (root / "cache" / f"{c}.npz").stat().st_mtime for c in order}
    v = sorted(t.values())
    i = order.index(core)
    prev_t, next_t = (t[order[i - 1]] if i else None), (t[order[i + 1]] if i + 1 < len(order) else None)
    out = {"n_files": len(v), "min_utc": iso(v[0]), "median_utc": iso(v[len(v) // 2]), "max_utc": iso(v[-1]),
           "core_mtime_utc": iso(t[core]), "core_rank_by_mtime": sorted(t, key=t.get).index(core),
           "previous_core": order[i - 1] if i else None, "previous_mtime_utc": iso(prev_t) if prev_t else None,
           "next_core": order[i + 1] if i + 1 < len(order) else None, "next_mtime_utc": iso(next_t) if next_t else None,
           "core_written_between_its_neighbours": bool((prev_t is None or prev_t <= t[core]) and (next_t is None or t[core] <= next_t))}
    if window:
        lo, hi = window["start_unix"] - MARGIN, window["end_unix"] + MARGIN
        out["window"] = {**window, "margin_seconds": MARGIN}
        out["n_outside_window"] = sum(not (lo <= x <= hi) for x in v)
        out["core_inside_window"] = bool(lo <= t[core] <= hi)
    return out


def dispatch_info() -> dict:
    import numpy as np
    from numpy.lib import introspect
    from numpy._core._multiarray_umath import __cpu_features__ as F

    def cur(f, sig):
        return introspect.opt_func_info(func_name=f"^{f}$")[f][sig]["current"]
    model = next((s.split(":", 1)[1].strip() for s in open("/proc/cpuinfo") if s.startswith("model name")), None)
    return {"host": socket.gethostname(), "cpu_model": model, "numpy": np.__version__, "numpy_path": str(Path(np.__file__).parent),
            "X86_V3": bool(F.get("X86_V3")), "X86_V4": bool(F.get("X86_V4")), "log1p_float64": cur("log1p", "dd"),
            "add_float64": cur("add", "ddd"), "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES", "")}


# ------------------------------------------------------------------------------------------------------------ rebuilds
def rebuild_child(spec: dict) -> int:
    """Child process: one core through PI.prepare (raw sha256 check, REG-P1 cmd_prepare verbatim) into a job-local dir."""
    job = os.environ.get("SLURM_JOB_ID")
    C.require(job and os.environ.get("V10_TR_DIAG_PARENT") == job, "rebuilds run only as children of the diagnostic job")
    dest = Path(spec["dest"])
    C.require(dest.parent.name.startswith(f"v10t_diag_{job}_"), "rebuild destination outside the job-local directory")
    P, host_hash = PI.host_modules(Path(spec["host_source_root"]), spec["host_code_hash"])
    PI.prepare(P, {spec["core"]: {"bf_sha256": spec["bf_sha256"], "shg_sha256": spec["shg_sha256"]}}, Path(spec["raw_root"]), dest)
    print("REBUILD " + json.dumps({"dispatch": dispatch_info(), "host_code_hash": host_hash}), flush=True)
    return 0


def rebuild(spec: dict, disable: str, job: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "NPY_DISABLE_CPU_FEATURES"}
    env.update(V10_TR_DIAG_PARENT=job, OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "2"))
    if disable:
        env["NPY_DISABLE_CPU_FEATURES"] = disable
    r = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--rebuild", json.dumps(spec)], env=env,
                       capture_output=True, text=True)
    C.require(r.returncode == 0, f"rebuild {Path(spec['dest']).name} failed: {r.stderr.strip().splitlines()[-1:] }")
    line = [s for s in r.stdout.splitlines() if s.startswith("REBUILD ")][-1]
    return json.loads(line[len("REBUILD "):])


# ------------------------------------------------------------------------------------------------------------ diagnosis
def diagnose(core: str, expected: dict, raw_root: Path, v10_root: Path, v9_root: Path, tmp_base: Path, job: str,
             host_source_root: Path, host_code_hash: str, windows: dict | None = None) -> dict:
    """expected: inventory name -> (bf sha256, shg sha256) of the core's raw members."""
    windows = windows or {}
    raw = {}
    for kind in ("BF", "SHG"):
        p = raw_root / f"{core}_{kind}.tif"
        st = p.stat()
        raw[kind] = {"path": str(p), "sha256": C.sha(p), "bytes": st.st_size, "mtime_utc": iso(st.st_mtime)}
    got = (raw["BF"]["sha256"], raw["SHG"]["sha256"])
    v9m_p, v10m_p = v9_root / "manifest.json", v10_root / "manifest.json"
    v9m, v10m = C.read(v9m_p), C.read(v10m_p)
    e9, e10 = v9m["files"][core], v10m["files"][core]
    sources = {**{k: tuple(v) for k, v in expected.items()},
               "v9_g1_prepared_manifest": (e9["sha256_bf"], e9["sha256_shg"]),
               "v10_prepared_manifest": (e10["sha256_bf"], e10["sha256_shg"])}
    raw["matches"] = {k: v == got for k, v in sources.items()}
    v10_npz, v9_npz = v10_root / "cache" / f"{core}.npz", v9_root / "cache" / f"{core}.npz"
    cores = sorted(v9m["files"])
    provenance = {
        "v9_g1_cache": {**npz_meta(v9_npz), "manifest_entry": e9, "manifest_entry_keys": sorted(e9),
                        "manifest_pins_cache_sha256": any(("npz" in k or "cache" in k) for k in e9),
                        "manifest": {"path": str(v9m_p), "sha256": C.sha(v9m_p), "mtime_utc": iso(v9m_p.stat().st_mtime),
                                     "n_files": len(v9m["files"]), "top_level_keys": sorted(v9m)},
                        "all_cache_mtimes": mtimes(v9_root, cores, core, windows.get("v9_g1_prepare"))},
        "v10_rebuild": {**npz_meta(v10_npz), "manifest_entry": e10,
                        "manifest": {"path": str(v10m_p), "sha256": C.sha(v10m_p), "mtime_utc": iso(v10m_p.stat().st_mtime)},
                        "all_cache_mtimes": mtimes(v10_root, sorted(v10m["files"]), core, windows.get("v10_prepare"))},
        "prepared_manifests_byte_identical": v9m_p.read_bytes() == v10m_p.read_bytes(),
        "prepared_manifest_entries_equal": {"core": e9 == e10, "all_cores": v9m["files"] == v10m["files"],
                                            "roles": v9m["roles"] == v10m["roles"],
                                            "omega_size_down": [v9m.get(k) for k in ("omega", "size", "down")] ==
                                                               [v10m.get(k) for k in ("omega", "size", "down")]}}
    v10_vs_v9 = compare(v10_npz, v9_npz)
    tmp = Path(tempfile.mkdtemp(prefix=f"v10t_diag_{job}_", dir=str(tmp_base)))
    try:
        runs, paths = {}, {}
        for tag, disable in VARIANTS:
            spec = {"core": core, "bf_sha256": got[0], "shg_sha256": got[1], "raw_root": str(raw_root), "dest": str(tmp / tag),
                    "host_source_root": str(host_source_root), "host_code_hash": host_code_hash}
            runs[tag] = {"NPY_DISABLE_CPU_FEATURES": disable, **rebuild(spec, disable, job)}
            paths[tag] = tmp / tag / "cache" / f"{core}.npz"
        for tag, p in paths.items():
            runs[tag]["npz_sha256"] = C.sha(p)
            runs[tag]["arrays_vs_v10_rebuild"] = brief(compare(p, v10_npz))
            runs[tag]["arrays_vs_v9_g1_cache"] = brief(compare(p, v9_npz))
        determinism = compare(paths["default_1"], paths["default_2"])
        dispatch_effect = brief(compare(paths["no_X86_V4"], paths["default_1"]))
        baseline_effect = brief(compare(paths["baseline_X86_V2"], paths["default_1"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)                                # no TEST-derived copy stays on the node
    differing = sorted(k for k, v in v10_vs_v9["arrays"].items() if not v["identical"])
    return {
        "schema": SCHEMA, "core": core, "evaluated": False, "labels_or_results_read": False,
        "summary": {
            "raw_members_match": raw["matches"], "v10_vs_v9_identical": v10_vs_v9["identical"],
            "v10_vs_v9_differing_arrays": {k: {"n_differ": v10_vs_v9["arrays"][k].get("n_differ"),
                                               "max_ulp": v10_vs_v9["arrays"][k].get("ulp", {}).get("max"),
                                               "max_abs": v10_vs_v9["arrays"][k].get("max_abs"),
                                               "mean_abs": v10_vs_v9["arrays"][k].get("mean_abs")} for k in differing},
            "rebuild_deterministic": determinism["identical"],
            "rebuilds_equal_to_v10": [t for t, r in runs.items() if r["arrays_vs_v10_rebuild"]["identical"]],
            "rebuilds_equal_to_v9_g1": [t for t, r in runs.items() if r["arrays_vs_v9_g1_cache"]["identical"]],
            "node_has_X86_V4": runs["default_1"]["dispatch"]["X86_V4"],
            "v9_cache_written_in_original_sequence": provenance["v9_g1_cache"]["all_cache_mtimes"]["core_written_between_its_neighbours"],
            "v9_cache_inside_v9_g1_prepare_window": provenance["v9_g1_cache"]["all_cache_mtimes"].get("core_inside_window"),
            "v9_manifest_pins_cache_sha256": provenance["v9_g1_cache"]["manifest_pins_cache_sha256"]},
        "raw_members": raw, "provenance": provenance, "v10_vs_v9_g1": v10_vs_v9,
        "rebuilds": runs, "determinism_default_1_vs_default_2": determinism,
        "dispatch_effect_no_X86_V4_vs_default": dispatch_effect, "dispatch_effect_baseline_vs_default": baseline_effect,
        "environment": dispatch_info()}


# ------------------------------------------------------------------------------------------------------------ the job
def expected_hashes(rec: dict, core: str) -> dict:
    """The core's member sha256 in each frozen inventory (each inventory file verified against the manifest pin)."""
    t = rec["manifest"]["test"]
    row = next(r for r in t["raw"] if r["core"] == core)
    src = t["source"]
    csv_p = Path(src["v9_member_list"]["path"])
    C.require(C.sha(csv_p) == src["v9_member_list"]["sha256"], "V9-G1 member list differs from the manifest pin")
    with csv_p.open() as f:
        crow = next(r for r in csv.DictReader(f) if r["core"] == core)
    dm_p = C.REPO / src["data_manifest"]["path"]
    C.require(C.sha(dm_p) == src["data_manifest"]["sha256"], "V10 data manifest differs from the manifest pin")
    with dm_p.open() as f:
        drow = next(r for r in csv.DictReader(f) if r["dataset"] == "BF_SHG" and r["record_id"] == core)
    hx = {k: re.search(rf"\b{k}=([0-9a-f]{{64}})\b", drow["source_version"]) for k in ("sha256_fixed", "sha256_moving")}
    C.require(all(hx.values()), "the V10 data manifest row names no member sha256")      # values may carry a note
    return {"v10_test_repeat_manifest": (row["bf_sha256"], row["shg_sha256"]),
            "v9_g1_frozen_member_csv": (crow["bf_sha256"], crow["shg_sha256"]),
            "v10_data_manifest": (hx["sha256_fixed"].group(1), hx["sha256_moving"].group(1))}


def check_pins(core: str, job: str | None) -> dict:
    """Every submission pin, before any TEST-derived file is opened (no raw member, cache or prepared file is read here)."""
    act, commit, auth_pin = (os.environ.get("V10_TR_EXPECT_ACTIVATION"), os.environ.get("V10_TR_EXPECT_COMMIT"),
                             os.environ.get("V10_TR_EXPECT_DIAG"))
    C.require(act and commit and auth_pin, "activation, commit and diagnostic-authorization pins are required")
    rec = C.records(act)                                   # the v4 records, unchanged
    code = C.verify_code(rec["activation"], commit)        # the v4-pinned code files, unchanged at this commit
    ab = (C.BUNDLE / AUTHORIZATION).read_bytes()
    C.require(C.sha_bytes(ab) == auth_pin and ab == (C.REPO / C.RECORDS / AUTHORIZATION).read_bytes()
              == C.committed_bytes(f"{C.RECORDS}/{AUTHORIZATION}"), "diagnostic authorization differs from its pin")
    auth = json.loads(ab)
    C.require(auth["v4_activation_sha256"] == rec["activation_sha256"], "authorization names another activation")
    C.require(str(C.REPO) == auth["code"]["execution_root"], "diagnostic runs only from its execution worktree")
    for rel, value in auth["code"]["files"].items():
        C.require(C.sha(C.REPO / rel) == value and C.sha_bytes(C.committed_bytes(rel)) == value, f"diagnostic code changed: {rel}")
    C.require(C.git("status", "--porcelain", "--", *auth["code"]["files"]) == "", "diagnostic code has local changes")
    C.require(core in auth["cores"], f"{core!r} is outside the authorized diagnostic scope")
    lock_p = C.BUNDLE / "INPUT_LOCK.json"
    C.require(C.sha(lock_p) == auth["failed_input_lock_sha256"], "INPUT_LOCK.json differs from the failed lock named")
    lock = C.read(lock_p)
    C.require(lock["status"] == "FAIL_V9_CACHE_MISMATCH" and core in lock["v9_cache_comparison"]["differing"],
              "the core is not a recorded cache mismatch")
    out = C.BUNDLE / "diagnostics" / core / job if job else None
    C.require(out is None or not out.exists(), "diagnostic output exists; never overwritten")
    return {"rec": rec, "code": code, "auth": auth, "auth_pin": auth_pin, "lock": lock, "out": out}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--rebuild"]:
        return rebuild_child(json.loads(argv[1]))
    if argv[:1] == ["--check-pins"]:                       # login-node pre-submission check: pins only, no TEST file
        k = check_pins(argv[1] if len(argv) > 1 else "", None)
        print("PINS OK", argv[1], "activation", k["rec"]["activation_sha256"][:12], "authorization", k["auth_pin"][:12],
              "commit", k["code"]["commit"][:12], flush=True)
        return 0
    job = C.require_slurm()
    core = argv[0] if argv else ""
    k = check_pins(core, job)
    rec, auth, lock, out = k["rec"], k["auth"], k["lock"], k["out"]
    spec = rec["protocol"]["inputs"]
    v10_root = Path(spec["root"])
    lock_sha = next(f["sha256"] for f in lock["files"] if f["core"] == core)
    C.require(C.sha(v10_root / "cache" / f"{core}.npz") == lock_sha, "the V10-rebuilt file differs from its lock entry")
    t0 = time.time()
    res = diagnose(core, expected_hashes(rec, core), Path(spec["raw_root"]), v10_root, Path(spec["v9_prepared_root"]),
                   Path(os.environ.get("TMPDIR") or "/tmp"), job, Path(spec["host_source_root"]), spec["host_code_hash"],
                   auth.get("windows"))
    res.update(job=job, utc=C.utc(), seconds=round(time.time() - t0, 1), scope=auth["scope"], authorization_sha256=k["auth_pin"],
               activation_sha256=rec["activation_sha256"], code_commit=k["code"]["commit"],
               input_lock_sha256=auth["failed_input_lock_sha256"], v10_rebuild_matches_lock=True)
    C.write_once(out / "DIAG.json", res)
    s = res["summary"]
    print("DIAG", core, json.dumps({x: s[x] for x in ("raw_members_match", "v10_vs_v9_differing_arrays", "rebuild_deterministic",
                                                    "rebuilds_equal_to_v10", "rebuilds_equal_to_v9_g1", "node_has_X86_V4")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
