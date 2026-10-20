#!/usr/bin/env python3
"""Reproducible CPU/unit and optional production-size synthetic IO validation."""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.dont_write_bytecode = True

from v9n2m.common import read_json, sha256_file, utc_now, write_json


def full_shape_fixture():
    import numpy as np
    from v9n2m.controls import generate_controls, load_weights
    from v9n2m.offline import score_weights
    begin = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="v9n2m_full_shape_SYNTHETIC_") as temporary:
        base = Path(temporary)
        n_images, n_anchor_images, seed = 29000, 2900, 901
        images = np.repeat(np.arange(n_images), 5)
        observed_images = images.copy()
        u_idx = np.arange(n_anchor_images*5, n_images*5)
        rng = np.random.default_rng([seed,9004,400])
        chosen = rng.choice(u_idx, size=round(.4*len(u_idx)), replace=False)
        observed_images[chosen] = images[rng.permutation(chosen)]
        def role(i):
            if i < 1160: return "teacher_warmup"
            if i < 2030: return "critic_fit"
            if i < 2465: return "critic_select"
            if i < 2900: return "reference_moments"
            return "U"
        rows = [{"pair_id": f"p{i}", "caption_id": str(i), "observed_image_id": str(observed_images[i]), "role": role(int(images[i]))}
                for i in range(n_images*5)]
        labels = {f"p{i}": {"original_image_id": str(images[i]), "mismatched": bool(images[i] != observed_images[i])}
                  for i in range(len(rows))}
        source = base / "production_SYNTHETIC.csv"
        score = np.random.default_rng([seed,1]).uniform(0,1,len(u_idx))
        T = np.tanh(.8*score-.1)
        w = .6*(1+T)/(1+.2*T)
        with source.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pair_id","w","score","rho_hat","fit_hash","reference_hash"])
            for i, weight, s in zip(u_idx,w,score):
                writer.writerow([f"p{i}",f"{weight:.10f}",f"{s:.10f}","0.40000000","synthetic_fit","synthetic_ref"])
        lock = Path(str(source)+".lock.json")
        write_json(lock,{"file":str(source),"sha256":sha256_file(source),"n":len(u_idx),
                         "record":{"method":"VCS_Weight","rho_strategy":"estimated","deployable":True,
                                   "rho_used":.4,"rho":{"raw":.4,"projected":.4},
                                   "critic":{"class":"cos2 (a s + b, a = softplus)","a":.8,"b":-.1}}})
        obs, lab, noise = base/"pairs.observed_pairs.json",base/"eval_only/pairs.eval_only.json",base/"pairs.noise_manifest.json"
        write_json(obs, rows); write_json(lab, labels)
        write_json(noise, {"design_rho":.4,"seed":seed,"observed_sha256":sha256_file(obs),"eval_only_sha256":sha256_file(lab)})
        cm = generate_controls(obs,source,lock,seed,base/"controls")
        # Re-read independent values and compare without using stored summary statistics.
        vectors = {a: np.asarray(list(load_weights(rec["csv"],lock_path=rec["lock"]).values()),dtype=np.float64) for a,rec in cm["arms"].items()}
        mass = {a:float(v.sum(dtype=np.float64)) for a,v in vectors.items()}
        assert np.array_equal(np.sort(vectors["VCS"]),np.sort(vectors["ShuffleVCS"]))
        assert np.max(np.abs(vectors["ConstantU"]-vectors["ConstantU"][0])) == 0
        assert np.isclose(mass["VCS"],mass["ConstantU"],rtol=2e-12,atol=1e-12)
        assert len(np.load(base/"controls/permutation.npy",allow_pickle=False)) == 26100
        first_hash = sha256_file(base/"controls/manifest.json")
        generate_controls(obs,source,lock,seed,base/"controls")
        assert sha256_file(base/"controls/manifest.json") == first_hash
        scoring = score_weights(obs,lab,source,lock,base/"controls",base/"scoring",noise_manifest_path=noise)
        assert scoring["production_reconstruction"]["consistent"]
        assert abs(scoring["arms"]["ConstantU"]["auroc"]-.5)<1e-12
        return {"status":"PASS","purpose":"Synthetic IO/math exercise only; no real dataset, model fit, or research result",
                "seed":seed,"n_pool":145000,"n_A":14500,"n_U":130500,"five_row_blocks":26100,
                "checks":["production-size strict ID coverage","Constant mass","Shuffle exact multiset","26,100 saved blocks",
                          "idempotent regeneration","score/rho rounding-consistent posterior reconstruction","tie-safe constant AUROC"],
                "permutation_sha256":sha256_file(base/"controls/permutation.npy"),
                "mass_absolute_difference_constant":abs(mass["VCS"]-mass["ConstantU"]),
                "elapsed_s":time.monotonic()-begin,"temporary_fixture_removed_on_return":True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-source",help="Original BASE/source, enabling actual-source structural checks")
    ap.add_argument("--full-shape",action="store_true",help="Also exercise full 145000-row synthetic controls and scoring")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    if args.baseline_source: os.environ["V9N2M_BASELINE_ROOT"]=str(Path(args.baseline_source).resolve())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO();begin=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(PACKAGE/"tests"))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/"unittest_output.txt").write_text(stream.getvalue(),encoding="utf-8")
    report={"created_utc":utc_now(),"status":"PASS" if result.wasSuccessful() else "FAIL",
            "unit_tests":{"run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
                          "skipped":len(result.skipped),"elapsed_s":time.monotonic()-begin},
            "baseline_source":args.baseline_source,"scientific_full_training":"NOT_RUN","CUDA":"NOT_RUN"}
    if result.wasSuccessful() and args.full_shape:
        report["full_shape_synthetic"]=full_shape_fixture()
    write_json(out/"LOCAL_VALIDATION.json",report)
    print(report)
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__": raise SystemExit(main())
