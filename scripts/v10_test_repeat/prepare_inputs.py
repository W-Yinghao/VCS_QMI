"""V10 TEST_REPEAT134 inputs (Slurm only; the first stage that opens TEST data, after the freeze and the L40S DEV gates).

1. (v4) the frozen v4 records and code (no DEV-gate precondition; each TEST job runs its own in-job DEV check);
2. the 268 sealed-test raw members (V9-G1 byte copies) match the sha256 values frozen in the manifest (V9-G1's core list);
3. REG-P1's frozen cmd_prepare builds the 512-px inputs (V9-G1's own preparation path: core discovery and role assignment
   pointed at the test files, role TEST) into the TEST_REPEAT input root;
4. the arrays must equal V9-G1's prepared TEST cache bitwise (same frozen code on the same members);
5. INPUT_LOCK.json records every prepared file's sha256; TestRepeatGuard grants reads only against it.
v10.adapter is never imported here (it replaces cmd_prepare with a tripwire)."""
from __future__ import annotations

import argparse
import hashlib
import os
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


def main() -> int:
    job = C.require_slurm()
    act, commit = os.environ.get("V10_TR_EXPECT_ACTIVATION"), os.environ.get("V10_TR_EXPECT_COMMIT")
    C.require(act and commit, "activation and commit pins are required at submission")
    rec = C.records(act)
    code = C.verify_code(rec["activation"], commit)
    man, prot = rec["manifest"], rec["protocol"]
    # v4: prepare may run as soon as v4 is frozen (TEST jobs carry their own in-job DEV check)
    spec = prot["inputs"]
    C.require(not (C.BUNDLE / "INPUT_LOCK.json").exists(), "INPUT_LOCK.json exists; inputs are prepared once")
    P, host_hash = host_modules(Path(spec["host_source_root"]), spec["host_code_hash"])
    root = Path(spec["root"])
    cores = {c["core"]: c for c in man["test"]["raw"]}
    C.require(sorted(cores) == sorted(man["test"]["cores"]), "raw member table does not cover the 134 cores")
    m = prepare(P, cores, Path(spec["raw_root"]), root)
    v9 = compare_cache(root, Path(spec["v9_prepared_root"]), cores)
    files = [{"core": c, "path": str(root / "cache" / f"{c}.npz"), "sha256": C.sha(root / "cache" / f"{c}.npz")}
             for c in sorted(cores)]
    status = "PASS" if v9["identical"] else "FAIL_V9_CACHE_MISMATCH"
    C.write(C.BUNDLE / "INPUT_LOCK.json", {
        "schema": "v10.test_repeat134.input_lock.v1", "status": status, "job": job, "utc": C.utc(),
        "activation_sha256": rec["activation_sha256"], "manifest_sha256": rec["shas"][C.MANIFEST],
        "code_commit": code["commit"], "host_code_hash": host_hash, "root": str(root),
        "prepared_manifest_sha256": C.sha(root / "manifest.json"), "raw_members_verified": 2 * len(cores),
        "files": files, "v9_cache_comparison": v9, "omega": m.get("omega"), "size": m.get("size"), "down": m.get("down")})
    print("INPUT_LOCK", status, len(files), flush=True)
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as e:
        if not isinstance(e, SystemExit) or e.code not in (0, None):
            C.write(C.BUNDLE / "failures" / f"{os.environ.get('SLURM_JOB_ID', 'local')}.json",
                    {"utc": C.utc(), "stage": "prepare", "exception": repr(e), "traceback": traceback.format_exc()})
        raise
