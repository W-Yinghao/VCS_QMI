"""Independent population identities plus actual artifact/sampling checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v10_synthetic import (exact_truth, generate_suite, load_samples,
    load_evaluation_truth, population_j, role_seed_sequence, sample_joint)


def run_checks(suite_dir: Path) -> dict:
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    rows = []

    def check(name: str, passed: bool, **details):
        row = {"name": name, "passed": bool(passed), **details}
        rows.append(row)
        if not passed:
            raise AssertionError(json.dumps(row, ensure_ascii=False))

    max_probability_error = max_gap_error = max_formula_error = max_sample_z = 0.0
    fingerprints = set()
    namespaces = set()
    total_base_observations = 0
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(910271)))
    for entry in manifest["cases"]:
        cid, family, m, rho = (entry[key] for key in ("case_id", "family", "m", "rho"))
        truth = load_evaluation_truth(suite_dir / cid)
        p, q, eta = (truth[key] for key in ("p", "q", "eta"))
        s = float(truth["S"])
        error = max(abs(float(p.sum()) - 1.0), abs(float(q.sum()) - 1.0),
                    float(np.max(np.abs(q - np.outer(p.sum(1), p.sum(0))))),
                    float(np.max(np.abs(p.sum(0) - 1.0 / m))),
                    float(np.max(np.abs(p.sum(1) - 1.0 / m))))
        max_probability_error = max(max_probability_error, error)
        check(f"{cid}:population_probabilities", error < 2e-13 and
              p.shape == (m, m) and np.all(p >= 0) and np.all(q > 0),
              maximum_normalization_or_marginal_error=error)
        # The analytic S expressions are independent checks on the saved tables.
        s_expected = (rho**2 / (4 - rho**2) if family == "two_block" else
                      rho**2 * (m - 1) / ((2 + rho * (m - 1)) * (2 - rho)))
        formula_error = abs(s - s_expected)
        max_formula_error = max(max_formula_error, formula_error)
        check(f"{cid}:analytic_S", formula_error < 2e-13,
              table_S=s, analytic_S=s_expected, absolute_error=formula_error)
        probes = [np.zeros_like(p), eta, rng.uniform(-1, 1, p.shape)]
        gap_error = max(abs(s - population_j(p, q, t) -
                           float(np.sum(0.5 * (p + q) * (t - eta)**2))) for t in probes)
        gap_error = max(gap_error, abs(s - population_j(p, q, eta)))
        max_gap_error = max(max_gap_error, gap_error)
        check(f"{cid}:original_S_J_gap", gap_error < 2e-13,
              maximum_absolute_gap_identity_error=gap_error)
        if rho == 0:
            check(f"{cid}:rho_zero_independence", np.max(np.abs(p - q)) < 1e-14
                  and np.max(np.abs(eta)) < 1e-13 and abs(s) < 1e-14)
        else:
            rank = int(np.linalg.matrix_rank(p - q, tol=1e-11))
            check(f"{cid}:dependence_rank", rank == (1 if family == "two_block" else m - 1),
                  rank_of_p_minus_q=rank)
        for role in ("fit", "cal", "eval"):
            sample = load_samples(suite_dir / cid, role)
            metadata = json.loads((suite_dir / cid / role / "metadata.json").read_text())
            n = entry["roles"][role]["n_independent_base_observations"]
            total_base_observations += n
            check(f"{cid}:{role}:sample_only_payload", set(sample) == {"x", "y", "base_id"}
                  and len(sample["x"]) == n and np.all(sample["x"] >= 0)
                  and np.all(sample["x"] < m) and np.all(sample["y"] >= 0)
                  and np.all(sample["y"] < m) and
                  np.array_equal(sample["base_id"], np.arange(n)) and
                  metadata["contains_oracle_values"] is False)
            fingerprint = tuple(metadata["state_fingerprint_uint32"])
            namespace = metadata["base_id_namespace"]
            check(f"{cid}:{role}:separate_stream", fingerprint not in fingerprints and
                  namespace not in namespaces,
                  spawn_key=metadata["spawn_key"])
            fingerprints.add(fingerprint)
            namespaces.add(namespace)
        # A separate large draw checks an observable event with a known
        # Bernoulli probability; it does not inspect latent generation labels.
        ss = role_seed_sequence(770061, family, m, rho, "eval")
        sampled = sample_joint(p, 30000, ss)
        event = ((sampled["x"] // (m // 2) == sampled["y"] // (m // 2))
                 if family == "two_block" else sampled["x"] == sampled["y"])
        expected = (1 + rho) / 2 if family == "two_block" else rho + (1 - rho) / m
        se = np.sqrt(expected * (1 - expected) / len(event))
        z = abs(float(event.mean()) - expected) / se if se > 0 else float(event.mean() != expected)
        max_sample_z = max(max_sample_z, z)
        check(f"{cid}:sampling_law", z < 7.0,
              independent_draws=len(event), empirical_event_probability=float(event.mean()),
              expected_event_probability=expected, standardized_error=float(z))

    check("counts_are_independent_base_observations", total_base_observations ==
          manifest["n_independent_base_observations_total"],
          total_independent_base_observations=total_base_observations,
          role_streams=len(fingerprints),
          all_pairs_are_not_counted_as_independent=True)

    families = tuple(dict.fromkeys(entry["family"] for entry in manifest["cases"]))
    sizes = tuple(dict.fromkeys(entry["m"] for entry in manifest["cases"]))
    rhos = tuple(dict.fromkeys(entry["rho"] for entry in manifest["cases"]))
    first = manifest["cases"][0]
    counts = {role: first["roles"][role]["n_independent_base_observations"]
              for role in ("fit", "cal", "eval")}
    # Actually regenerate with all enumeration orders reversed. This catches
    # using one global RNG or an order-dependent spawn sequence in the writer.
    with tempfile.TemporaryDirectory(prefix="v10_synthetic_check_", dir=ROOT / "validation") as tmp:
        regenerated = Path(tmp) / "reversed"
        generate_suite(regenerated, root_seed=manifest["root_seed"],
                       families=families[::-1], sizes=sizes[::-1], rhos=rhos[::-1],
                       role_sizes=counts)
        exact_replay = True
        for entry in manifest["cases"]:
            for role in ("fit", "cal", "eval"):
                original = load_samples(suite_dir / entry["case_id"], role)
                replay = load_samples(regenerated / entry["case_id"], role)
                exact_replay &= all(np.array_equal(original[key], replay[key]) for key in original)
        check("bit_exact_sample_replay_after_case_reordering", exact_replay)
        # The sample reader must fail rather than silently deliver oracle fields.
        bad = Path(tmp) / "bad_case" / "fit"
        bad.mkdir(parents=True)
        np.savez_compressed(bad / "samples.npz", x=[0], y=[0], base_id=[0], S=[0.0])
        rejected = False
        try:
            load_samples(bad.parent, "fit")
        except ValueError:
            rejected = True
        check("fit_reader_rejects_oracle_field", rejected)
        refused = False
        try:
            generate_suite(regenerated)
        except FileExistsError:
            refused = True
        check("existing_output_is_not_silently_overwritten", refused)

    # Changing a stream's draw count preserves its prefix and does not consume
    # the CAL/EVAL streams. Duplicate discrete values across roles remain valid.
    case = manifest["cases"][0]
    p = exact_truth(case["family"], case["m"], case["rho"])["p"]
    def draw(n, role):
        return sample_joint(p, n, role_seed_sequence(manifest["root_seed"],
             case["family"], case["m"], case["rho"], role))
    short, long = draw(64, "fit"), draw(513, "fit")
    check("same_stream_count_change_preserves_prefix",
          all(np.array_equal(short[k], long[k][:64]) for k in short))
    cal_before = draw(127, "cal")
    draw(999, "fit")
    cal_after = draw(127, "cal")
    check("fit_draw_count_does_not_consume_cal_rng",
          all(np.array_equal(cal_before[k], cal_after[k]) for k in cal_before))

    return {"status": "passed", "n_checks": len(rows), "n_cases": manifest["n_cases"],
            "scope": "Finite observable distributions, independent base-pair generation, artifact roles and RNG replay. No GPU, neural estimator, natural data or generalization result.",
            "numpy_version": np.__version__, "root_seed": manifest["root_seed"],
            "independence_scope": "Named SeedSequence/PCG64 streams; structural independence under the pseudo-random sampling model. Distinct observed values are not required and PRNG independence is not statistically proved by this finite test.",
            "summary": {"max_probability_error": max_probability_error,
                        "max_S_closed_form_error": max_formula_error,
                        "max_S_J_gap_error": max_gap_error,
                        "max_sampling_event_standardized_error": max_sample_z,
                        "n_independent_base_observations": total_base_observations},
            "checks": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "CHECK_RESULTS.json")
    args = parser.parse_args()
    (ROOT / "validation").mkdir(parents=True, exist_ok=True)
    result = run_checks(args.suite_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
                           + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "n_checks": result["n_checks"],
                      "n_cases": result["n_cases"], "summary": result["summary"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
