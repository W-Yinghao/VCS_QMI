"""V10 TEST_REPEAT134 inputs, v5 (Slurm CPU job only; the frozen v5 protocol names the variant).

    prepare_inputs.py                 the v5 inputs stage (cpu.sbatch inputs): variant A or B1 of inputs.v5.variant
    prepare_inputs.py --check-pins    login-node check of the submission pins only (no TEST file is opened)

v4 (prepare job 1051404) wrote INPUT_LOCK.json FAIL_V9_CACHE_MISMATCH: 133 of 134 cores equal V9-G1's prepared cache, 1B_D8
differs.  That lock and the v4 input root are kept as written.  The owner's v5 decision (OWNER_DECISION_V5.json), applied
mechanically to the input diagnostic 1051581, selects one variant:

A   the mismatch is a CPU-dispatch artefact of REG-P1's builder: REG-P1's frozen cmd_prepare rebuilds all 134 cores into a
    new root with numpy's CPU dispatch pinned to the kernels that reproduced V9-G1's file (checked in the build process
    before any raw member is read); the criterion is unchanged: every array equals V9-G1's prepared cache bitwise.
B1  otherwise: the raw-verified, deterministic V10 rebuild of the v4 root is accepted for the listed exception cores; the
    raw members are verified again, every prepared file must equal its v4 lock entry, and the cores whose arrays differ
    from V9-G1's cache must be exactly the accepted exceptions.
Either writes INPUT_LOCK_V5.json once (PASS or FAIL); an environment error (wrong CPU dispatch, missing file) writes no lock.
TestRepeatGuard grants reads only against INPUT_LOCK_V5.json.  v10.adapter is never imported here (it replaces cmd_prepare
with a tripwire).  host_modules / prepare / compare_cache are unchanged from v4."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402

ARRAY_KEYS = ("bf", "shg", "bf_rgb", "shg_log", "bf_rgb_n", "shg_log_n")


def host_modules(source_root: Path, expect_hash: str):
    sys.path.insert(0, str(source_root / "src"))
    from v9.reg import g1 as G
    files = (Path(G.__file__).resolve(), Path(G.P.__file__).resolve(), Path(G.Q.__file__).resolve())
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]
    C.require(digest == expect_hash, f"host code hash {digest} != {expect_hash}")
    return G.P, digest


def prepare(P, cores: dict, raw_root: Path, root: Path) -> dict:
    """cores: core -> {bf_sha256, shg_sha256}.  Verify the raw members, then REG-P1's cmd_prepare verbatim into root."""
    import numpy as np  # noqa: F401  (REG-P1 imports it as well)
    C.require(not (root / "manifest.json").exists(), f"{root} is already prepared; inputs are built once")
    files = {}
    for core, row in sorted(cores.items()):
        bf, shg = raw_root / f"{core}_BF.tif", raw_root / f"{core}_SHG.tif"
        C.require(C.sha(bf) == row["bf_sha256"] and C.sha(shg) == row["shg_sha256"], f"{core}: raw member sha256 differs")
        files[core] = ("test", bf, shg)
    orig = (P.core_files, P.roles)
    P.core_files, P.roles = (lambda: files), (lambda cs: {c: "TEST" for c in cs})
    try:
        P.cmd_prepare(argparse.Namespace(root=root, smoke=False))
    finally:
        P.core_files, P.roles = orig
    m = C.read(root / "manifest.json")
    bad = [c for c, r in cores.items()
           if (m["files"][c]["sha256_bf"], m["files"][c]["sha256_shg"]) != (r["bf_sha256"], r["shg_sha256"])]
    C.require(not bad and sorted(m["roles"]) == sorted(cores) and set(m["roles"].values()) == {"TEST"},
              f"prepared manifest differs from the frozen core list: {bad[:5]}")
    return m


def compare_cache(root: Path, reference_root: Path, cores) -> dict:
    import numpy as np
    differing = []
    for c in sorted(cores):
        a, b = np.load(root / "cache" / f"{c}.npz"), np.load(reference_root / "cache" / f"{c}.npz")
        if sorted(a.files) != sorted(ARRAY_KEYS) or sorted(b.files) != sorted(ARRAY_KEYS) or not all(
                a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in ARRAY_KEYS):
            differing.append(c)
    return {"reference_root": str(reference_root), "cores": len(list(cores)), "identical": not differing,
            "differing": differing}


def dispatch_info() -> dict:
    """numpy's CPU dispatch of the kernels REG-P1's builder uses (float64 log1p and add); other ufuncs it calls are exact."""
    import socket
    import numpy as np
    from numpy.lib import introspect
    from numpy._core._multiarray_umath import __cpu_features__ as F

    def cur(f, sig):
        return introspect.opt_func_info(func_name=f"^{f}$")[f][sig]["current"]
    model = next((s.split(":", 1)[1].strip() for s in open("/proc/cpuinfo") if s.startswith("model name")), None)
    return {"host": socket.gethostname(), "cpu_model": model, "numpy": np.__version__, "numpy_path": str(Path(np.__file__).parent),
            "X86_V3": bool(F.get("X86_V3")), "X86_V4": bool(F.get("X86_V4")), "log1p_float64": cur("log1p", "dd"),
            "add_float64": cur("add", "ddd"), "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES", "")}


KERNELS = ("log1p_float64", "add_float64")


def build_child(spec: dict) -> int:
    """Variant A build process: the pinned kernels are checked first; only then REG-P1's cmd_prepare reads the raw members."""
    job = os.environ.get("SLURM_JOB_ID")
    C.require(job and os.environ.get("V10_TR_INPUTS_PARENT") == job, "the v5 build runs only as a child of the inputs job")
    d = dispatch_info()
    C.require(all(d[k] == spec["kernels"][k] for k in KERNELS),
              f"numpy dispatch {[d[k] for k in KERNELS]} differs from the v5 pin {[spec['kernels'][k] for k in KERNELS]}; "
              "no raw member was read")
    P, host_hash = host_modules(Path(spec["host_source_root"]), spec["host_code_hash"])
    prepare(P, spec["cores"], Path(spec["raw_root"]), Path(spec["root"]))
    print("BUILD " + json.dumps({"dispatch": d, "host_code_hash": host_hash}), flush=True)
    return 0


def lock_files(root: Path, cores) -> list:
    return [{"core": c, "path": str(root / "cache" / f"{c}.npz"), "sha256": C.sha(root / "cache" / f"{c}.npz")} for c in sorted(cores)]


def stage_a(rec: dict, code: dict, job: str) -> dict:
    """Re-prepare into the v5 root under the pinned dispatch; PASS iff every array equals V9-G1's cache (unchanged criterion)."""
    man, spec = rec["manifest"], rec["protocol"]["inputs"]
    pin = spec["numpy_dispatch"]
    root = Path(spec["root"])
    C.require(str(root) != spec["v4_lock"]["root"], "variant A builds into a new root; the v4 inputs stay as written")
    cores = {c["core"]: {"bf_sha256": c["bf_sha256"], "shg_sha256": c["shg_sha256"]} for c in man["test"]["raw"]}
    C.require(sorted(cores) == sorted(man["test"]["cores"]), "raw member table does not cover the 134 cores")
    env = {k: v for k, v in os.environ.items() if k not in ("NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES")}
    env["V10_TR_INPUTS_PARENT"] = job
    if pin["NPY_DISABLE_CPU_FEATURES"]:
        env["NPY_DISABLE_CPU_FEATURES"] = pin["NPY_DISABLE_CPU_FEATURES"]
    child = {"kernels": {k: pin[k] for k in KERNELS}, "cores": cores, "raw_root": spec["raw_root"], "root": str(root),
             "host_source_root": spec["host_source_root"], "host_code_hash": spec["host_code_hash"]}
    r = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--build", json.dumps(child)], env=env,
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout[-4000:])
    C.require(r.returncode == 0, "variant A build failed (no lock written): " + " | ".join(r.stderr.strip().splitlines()[-3:]))
    built = json.loads([x for x in r.stdout.splitlines() if x.startswith("BUILD ")][-1][len("BUILD "):])
    m = C.read(root / "manifest.json")
    C.require(sorted(m["files"]) == sorted(cores) and all((m["files"][c]["sha256_bf"], m["files"][c]["sha256_shg"])
                                                          == (cores[c]["bf_sha256"], cores[c]["shg_sha256"]) for c in cores),
              "the v5 prepared manifest differs from the frozen core list")
    v9 = compare_cache(root, Path(spec["v9_prepared_root"]), cores)
    return {"status": "PASS" if v9["identical"] else "FAIL_V9_CACHE_MISMATCH", "root": str(root),
            "files": lock_files(root, cores), "prepared_manifest_sha256": C.sha(root / "manifest.json"),
            "raw_members_verified": 2 * len(cores), "v9_cache_comparison": v9, "build_dispatch": built["dispatch"],
            "host_code_hash": built["host_code_hash"], "numpy_dispatch_pin": pin, "omega": m.get("omega"),
            "size": m.get("size"), "down": m.get("down")}


def stage_b1(rec: dict, code: dict, job: str) -> dict:
    """Lock the v4-prepared root with the accepted exceptions; PASS iff raw members verify, every file equals its v4 lock
    entry, and the cores differing from V9-G1's cache are exactly the accepted exceptions (at their accepted sha256)."""
    man, spec = rec["manifest"], rec["protocol"]["inputs"]
    v4p = C.BUNDLE / C.V4_LOCK
    C.require(C.sha(v4p) == spec["v4_lock"]["sha256"], "INPUT_LOCK.json (v4) differs from the v5 pin")
    v4 = C.read(v4p)
    root = Path(spec["root"])
    C.require(str(root) == v4["root"] == spec["v4_lock"]["root"], "variant B1 locks the v4 input root")
    cores = {c["core"]: c for c in man["test"]["raw"]}
    C.require(sorted(cores) == sorted(man["test"]["cores"]), "raw member table does not cover the 134 cores")
    raw_root = Path(spec["raw_root"])
    raw_bad = [c for c, r in sorted(cores.items())
               if C.sha(raw_root / f"{c}_BF.tif") != r["bf_sha256"] or C.sha(raw_root / f"{c}_SHG.tif") != r["shg_sha256"]]
    v4files = {f["core"]: f for f in v4["files"]}
    files = lock_files(root, cores)
    changed = [f["core"] for f in files if (f["path"], f["sha256"]) != (v4files[f["core"]]["path"], v4files[f["core"]]["sha256"])]
    manifest_ok = C.sha(root / "manifest.json") == v4["prepared_manifest_sha256"]
    v9 = compare_cache(root, Path(spec["v9_prepared_root"]), cores)
    exc = spec["accepted_exceptions"]
    exceptions_ok = (sorted(v9["differing"]) == sorted(exc)
                     and all(v4files[c]["sha256"] == exc[c]["file_sha256"] for c in exc))
    ok = not raw_bad and not changed and manifest_ok and exceptions_ok
    return {"status": "PASS" if ok else "FAIL_V5_B1", "root": str(root), "files": files,
            "prepared_manifest_sha256": C.sha(root / "manifest.json"), "prepared_manifest_unchanged": manifest_ok,
            "raw_members_verified": 2 * len(cores) - 2 * len(raw_bad), "raw_member_mismatch": raw_bad,
            "changed_since_v4_lock": changed, "v9_cache_comparison": v9, "accepted_exceptions": exc,
            "exceptions_are_exactly_the_differing_cores": exceptions_ok,
            "omega": v4.get("omega"), "size": v4.get("size"), "down": v4.get("down")}


def pins() -> dict:
    act, commit = os.environ.get("V10_TR_EXPECT_ACTIVATION"), os.environ.get("V10_TR_EXPECT_COMMIT")
    C.require(act and commit, "activation and commit pins are required at submission")
    rec = C.records(act)
    code = C.verify_code(rec["activation"], commit)
    C.require(rec["protocol"]["inputs"]["v5"]["variant"] in ("A", "B1"), "unknown v5 input variant")
    return {"rec": rec, "code": code}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--build"]:
        return build_child(json.loads(argv[1]))
    if argv[:1] == ["--check-pins"]:                         # login-node pre-submission check: records and code only
        k = pins()
        print("PINS OK v5 inputs variant", k["rec"]["protocol"]["inputs"]["v5"]["variant"], "activation",
              k["rec"]["activation_sha256"][:12], "commit", k["code"]["commit"][:12], flush=True)
        return 0
    job = C.require_slurm()
    k = pins()
    rec, code = k["rec"], k["code"]
    C.require(not (C.BUNDLE / C.LOCK).exists(), "INPUT_LOCK_V5.json exists; v5 inputs are locked once")
    variant = rec["protocol"]["inputs"]["v5"]["variant"]
    res = (stage_a if variant == "A" else stage_b1)(rec, code, job)
    lock = {"schema": "v10.test_repeat134.input_lock.v5", "status": res.pop("status"), "variant": variant, "job": job,
            "utc": C.utc(), "activation_sha256": rec["activation_sha256"], "manifest_sha256": rec["shas"][C.MANIFEST],
            "code_commit": code["commit"], "v4_lock_sha256": rec["protocol"]["inputs"]["v4_lock"]["sha256"],
            "partition": os.environ.get("SLURM_JOB_PARTITION"), "nodelist": os.environ.get("SLURM_JOB_NODELIST"), **res}
    C.write_once(C.BUNDLE / C.LOCK, lock)
    print("INPUT_LOCK_V5", lock["status"], variant, len(lock["files"]), flush=True)
    return 0 if lock["status"] == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as e:
        if not isinstance(e, SystemExit) or e.code not in (0, None):
            if sys.argv[1:2] != ["--build"]:
                C.write(C.BUNDLE / "failures" / f"{os.environ.get('SLURM_JOB_ID', 'local')}.json",
                        {"utc": C.utc(), "stage": "inputs_v5", "exception": repr(e), "traceback": traceback.format_exc()})
        raise
