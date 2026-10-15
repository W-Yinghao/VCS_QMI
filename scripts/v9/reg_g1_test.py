"""V9-G1 sealed-test read CLI (src/v9/reg/g1_test.py; protocol reports/V9/V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml).

Before the freeze (each refuses once the protocol exists):
    inventory    verify the sealed archive against DATA_MANIFEST, list its members, sha256 per file, TIFF header facts -> configs/V9_G1_TEST_CORES.csv
    perturb      REG-P2's generator with the new seed tags, 10 per core -> configs/V9_G1_TEST_PERTURBATIONS.json
    predictors   every G1 unit's CAL choice with sha256 (+ MI / NMI) -> configs/V9_G1_TEST_PREDICTORS.csv
    hashes       sha256 of the three files (for the protocol's frozen_files)
After the freeze:
    fixture      non-test smoke of every formal function (REG-P1 DEV cores in a fixture zip; reproduction of their G1 / REG-P2 records)
    prepare      THE READ STARTS: archive + per-file hashes, extraction, REG-P1's builder -> the test root; ledger row
    run          --predictor <id> ...: REG-P2 registration of every test core (skips a predictor whose result exists)
    aggregate    g1_aggregate.main over the test records -> reports/V9/V9_G1_TEST_results/
    status       results present / missing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v9.reg import g1_test as T  # noqa: E402

P = T.P


def formal_inputs():
    fz = T.require_frozen(); preds = T.load_predictors(); cores = T.read_cores_csv()
    if not (T.TEST_ROOT / "manifest.json").exists():
        raise SystemExit(f"[guard] {T.TEST_ROOT} is not prepared (run `prepare` first)")
    m = P.load_manifest(T.TEST_ROOT, False)
    if sorted(m["roles"]) != sorted(cores) or any((m["files"][c]["sha256_bf"], m["files"][c]["sha256_shg"]) != (r["bf_sha256"], r["shg_sha256"]) for c, r in cores.items()):
        raise SystemExit("[guard] the prepared test root differs from the frozen core list")
    if any(not (T.TEST_ROOT / "cache" / f"{c}.npz").exists() for c in cores):
        raise SystemExit("[guard] cache files missing in the test root")
    pert = json.load(open(T.PERT_JSON))
    if sorted(pert) != sorted(cores) or any(len(v) != P.N_PERT for v in pert.values()):
        raise SystemExit("[guard] the frozen perturbations do not cover every test core with 10 perturbations")
    return fz, preds, pert


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("inventory", "perturb", "predictors", "hashes", "prepare", "status"):
        sub.add_parser(c)
    f = sub.add_parser("fixture"); f.add_argument("--n-cores", type=int, default=3); f.add_argument("--n-pert", type=int, default=1)
    r = sub.add_parser("run"); r.add_argument("--predictor", nargs="+", required=True)
    g = sub.add_parser("aggregate"); g.add_argument("--out", type=Path, default=REPO / "reports/V9/V9_G1_TEST_results")
    a = ap.parse_args()
    if a.cmd == "inventory":
        T.refuse_if_frozen(); h, md = hashlib.sha256(), hashlib.md5()
        with open(T.TEST_ZIP, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk); md.update(chunk)
        if (h.hexdigest(), md.hexdigest(), T.TEST_ZIP.stat().st_size) != (T.ZIP_FACTS["sha256"], T.ZIP_FACTS["md5"], T.ZIP_FACTS["bytes"]):
            raise SystemExit("[guard] the archive differs from DATA_MANIFEST bf_shg_test_closed")
        rows = T.inventory(T.TEST_ZIP); T.write_cores_csv(rows, T.CORES_CSV); known = set(P.load_manifest(T.P1_ROOT, False)["roles"])
        facts = {k: sorted({tuple(r[f"{k}_{x}"] for x in ("w", "h", "spp", "bits", "comp", "planar")) for r in rows}) for k in ("bf", "shg")}
        print(json.dumps({"archive_verified": True, "n_cores": len(rows), "slides": {s: sum(r["slide"] == s for r in rows) for s in sorted({r["slide"] for r in rows})},
                          "overlap_with_reg_p1_cores": sorted(set(r["core"] for r in rows) & known), "tiff_facts (w, h, spp, bits, comp, planar)": facts}, indent=1)); return 0
    if a.cmd == "perturb":
        T.refuse_if_frozen(); cores = T.read_cores_csv()
        T.PERT_JSON.write_text(json.dumps({c: T.perturbations_tagged(c, *T.TEST_TAGS) for c in sorted(cores)}, indent=1)); print(f"[perturb] {len(cores)} cores × {P.N_PERT} -> {T.PERT_JSON}"); return 0
    if a.cmd == "predictors":
        T.refuse_if_frozen(); rows = T.predictor_rows(); T.write_predictors_csv(rows, T.PRED_CSV); print(f"[predictors] {len(rows)} rows -> {T.PRED_CSV}"); return 0
    if a.cmd == "hashes":
        print(json.dumps({p.name: T.sha(p) for p in (T.CORES_CSV, T.PERT_JSON, T.PRED_CSV)}, indent=1)); return 0
    if a.cmd == "fixture":
        return T.fixture(a.n_cores, a.n_pert)
    if a.cmd == "prepare":
        fz = T.require_frozen(); cores = T.read_cores_csv()
        T.prepare(T.TEST_ROOT, T.TEST_ZIP, cores, formal=True, ledger=T.LEDGER)
        T.ledger_update(T.LEDGER, archive=str(T.TEST_ZIP), archive_sha256=T.ZIP_FACTS["sha256"], archive_md5=T.ZIP_FACTS["md5"], archive_bytes=T.ZIP_FACTS["bytes"], n_cores=len(cores),
                        cores_csv_sha256=fz["V9_G1_TEST_CORES.csv"], perturbations_sha256=fz["V9_G1_TEST_PERTURBATIONS.json"], predictors_sha256=fz["V9_G1_TEST_PREDICTORS.csv"],
                        protocol=str(T.PROTOCOL.relative_to(REPO)), protocol_commit=fz["protocol_commit"])
        print(f"[prepare] {len(cores)} test cores -> {T.TEST_ROOT} (ledger {T.LEDGER})"); return 0
    if a.cmd == "run":
        fz, preds, pert = formal_inputs(); rc = 0
        for pid in a.predictor:
            if pid not in preds:
                raise SystemExit(f"unknown predictor {pid}")
            if (T.TEST_ROOT / "results" / f"{pid}.json").exists():
                print(f"[skip] {pid}: result exists"); continue
            T.register(pid, T.TEST_ROOT, pert, preds, n_pert=P.N_PERT)
        return rc
    if a.cmd == "aggregate":
        T.require_frozen(); j = T.aggregate(T.TEST_ROOT, a.out); T.ledger_update(T.LEDGER, aggregate_job=os.environ.get("SLURM_JOB_ID", "local"))
        print(f"[aggregate] {j['n_results']} results, missing {j['missing']} -> {a.out}"); return 0 if not j["missing"] else 1
    preds = T.load_predictors(); have = [p for p in preds if (T.TEST_ROOT / "results" / f"{p}.json").exists()]
    print(json.dumps({"results": len(have), "of": len(preds), "missing": [p for p in preds if p not in have]}, indent=1)); return 0


if __name__ == "__main__":
    sys.exit(main())
