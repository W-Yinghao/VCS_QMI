"""Observable finite-distribution generators for V10 T03.

NumPy only. Oracle tables are saved separately for evaluation. Every row of
fit/cal/eval samples is one independently generated joint observation (X, Y).
An all-pairs product formed from n rows does not create n*n independent data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
SEED_SCHEME_VERSION = 1
FAMILIES = {"two_block": 11, "diagonal_mixture": 23}
ROLES = {"fit": 101, "cal": 211, "eval": 307}


def _validate_case(family: str, m: int, rho: float) -> tuple[str, int, float]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; choose {tuple(FAMILIES)}")
    if isinstance(m, bool) or int(m) != m or m < 2:
        raise ValueError("m must be an integer >= 2")
    m = int(m)
    rho = float(rho)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be finite and in [0, 1]")
    if family == "two_block" and m % 2:
        raise ValueError("Balanced two_block requires even m")
    return family, m, 0.0 if rho == 0.0 else rho


def exact_truth(family: str, m: int, rho: float) -> dict[str, Any]:
    """Return the population law of the OBSERVED discrete variables.

    These are finite analytic population tables, evaluated in float64, not
    Monte Carlo truth and not the law of an unobserved latent variable.
    """
    family, m, rho = _validate_case(family, m, rho)
    uniform = np.full((m, m), 1.0 / (m * m), dtype=np.float64)
    if family == "two_block":
        signs = np.concatenate((np.ones(m // 2), -np.ones(m // 2)))
        p = uniform * (1.0 + rho * np.outer(signs, signs))
        s_closed_form = rho * rho / (4.0 - rho * rho)
    else:
        p = (1.0 - rho) * uniform + (rho / m) * np.eye(m)
        s_closed_form = rho * rho * (m - 1) / (
            (2.0 + rho * (m - 1)) * (2.0 - rho)
        )
    px = p.sum(axis=1)
    py = p.sum(axis=0)
    q = np.outer(px, py)
    mixture = 0.5 * (p + q)
    eta = np.divide(p - q, p + q, out=np.zeros_like(p), where=(p + q) > 0)
    s = float(np.sum(mixture * eta * eta))
    return {
        "p": p, "q": q, "px": px, "py": py,
        "eta": eta, "S": s, "S_closed_form": float(s_closed_form),
    }


def population_j(p: np.ndarray, q: np.ndarray, critic: np.ndarray) -> float:
    """Original unregularized J on a finite observed-state table."""
    p, q, critic = (np.asarray(a, dtype=np.float64) for a in (p, q, critic))
    if p.shape != q.shape or p.shape != critic.shape:
        raise ValueError("p, q and critic must have the same shape")
    return float(np.sum((p - q) * critic - 0.5 * (p + q) * critic * critic))


def case_id(family: str, m: int, rho: float) -> str:
    family, m, rho = _validate_case(family, m, rho)
    token = format(rho, ".17g").replace(".", "p")
    return f"{family}__m{m:03d}__rho{token}"


def role_seed_sequence(root_seed: int, family: str, m: int,
                       rho: float, role: str) -> np.random.SeedSequence:
    """Stable named streams: changing iteration order does not change samples.

    The binary64 representation of rho is split into two uint32 words. No
    Python hash(), shared mutable global RNG or order-dependent spawn() call
    participates in this mapping. Sample counts do not enter the stream key.
    """
    family, m, rho = _validate_case(family, m, rho)
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    if isinstance(root_seed, bool) or int(root_seed) != root_seed or root_seed < 0:
        raise ValueError("root_seed must be a nonnegative integer")
    rho_hi, rho_lo = struct.unpack(">II", struct.pack(">d", rho))
    key = (SEED_SCHEME_VERSION, FAMILIES[family], m, rho_hi, rho_lo, ROLES[role])
    return np.random.SeedSequence(int(root_seed), spawn_key=key)


def sample_joint(p: np.ndarray, n: int,
                 seed_sequence: np.random.SeedSequence) -> dict[str, np.ndarray]:
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("p must be a finite nonnegative matrix")
    if not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("p must sum to one")
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    cells = rng.choice(p.size, size=int(n), p=p.reshape(-1))
    return {
        "x": (cells // p.shape[1]).astype(np.int32),
        "y": (cells % p.shape[1]).astype(np.int32),
        "base_id": np.arange(int(n), dtype=np.int64),
    }


def load_samples(case_directory: str | Path, role: str) -> dict[str, np.ndarray]:
    """Sample-only reader. It never opens the evaluator_only directory."""
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    path = Path(case_directory) / role / "samples.npz"
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"x", "y", "base_id"}:
            raise ValueError("Unexpected sample payload; oracle fields are forbidden")
        result = {name: archive[name].copy() for name in archive.files}
    if any(a.ndim != 1 for a in result.values()):
        raise ValueError("Sample arrays must be one dimensional")
    if len({len(a) for a in result.values()}) != 1:
        raise ValueError("Sample array lengths differ")
    return result


def load_evaluation_truth(case_directory: str | Path) -> dict[str, np.ndarray]:
    """Explicit oracle reader for the evaluator, never called by sample loading."""
    with np.load(Path(case_directory) / "evaluator_only" / "truth.npz",
                 allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def generate_suite(output_dir: str | Path, *, root_seed: int = 20261016,
                   families: tuple[str, ...] = tuple(FAMILIES),
                   sizes: tuple[int, ...] = (8, 32),
                   rhos: tuple[float, ...] = (0.0, 0.25, 0.75),
                   role_sizes: dict[str, int] | None = None) -> dict[str, Any]:
    """Write a new complete suite; refuse to overwrite a nonempty directory."""
    counts = dict(role_sizes or {"fit": 4096, "cal": 2048, "eval": 8192})
    if set(counts) != set(ROLES):
        raise ValueError("role_sizes must have exactly fit, cal, eval")
    for role, n in counts.items():
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError(f"Invalid number of independent base observations for {role}")
        counts[role] = int(n)
    cases = [_validate_case(f, m, rho) for f in families for m in sizes for rho in rhos]
    ids = [case_id(*case) for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("Need a nonempty collection of unique cases")
    for f, m, rho in cases:
        role_seed_sequence(root_seed, f, m, rho, "fit")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "v10_observable_discrete",
        "root_seed": int(root_seed), "seed_scheme_version": SEED_SCHEME_VERSION,
        "bit_generator": "PCG64", "numpy_version": np.__version__,
        "role_names": {"fit": "train/FIT", "cal": "CAL", "eval": "EVAL"},
        "sampling_unit": "one independently drawn observable joint pair (X,Y)",
        "negative_sampling": "not materialized; full cross-pairs share base observations",
        "oracle_policy": "evaluator_only truth; no oracle arrays in any sample payload",
        "truth_kind": "finite analytic population, float64 evaluation; no Monte Carlo truth",
        "cases": [],
    }
    for family, m, rho in cases:
        identifier = case_id(family, m, rho)
        case_dir = output_dir / identifier
        truth = exact_truth(family, m, rho)
        entry = {"case_id": identifier, "family": family, "m": m, "rho": rho,
                 "roles": {}, "truth_directory": f"{identifier}/evaluator_only"}
        for role, n in counts.items():
            directory = case_dir / role
            directory.mkdir(parents=True, exist_ok=True)
            ss = role_seed_sequence(root_seed, family, m, rho, role)
            samples = sample_joint(truth["p"], n, ss)
            sample_path = directory / "samples.npz"
            np.savez_compressed(sample_path, **samples)
            metadata = {
                "schema_version": SCHEMA_VERSION, "case_id": identifier,
                "role": role, "n_independent_base_observations": n,
                "observed_support": {"x": [0, m - 1], "y": [0, m - 1]},
                "base_id_namespace": f"seed{root_seed}/{identifier}/{role}",
                "root_seed": int(root_seed), "spawn_key": list(ss.spawn_key),
                "state_fingerprint_uint32": ss.generate_state(4).tolist(),
                "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                "contains_oracle_values": False,
            }
            _write_json(directory / "metadata.json", metadata)
            entry["roles"][role] = {
                "samples": f"{identifier}/{role}/samples.npz",
                "metadata": f"{identifier}/{role}/metadata.json",
                "n_independent_base_observations": n,
            }
        oracle_dir = case_dir / "evaluator_only"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oracle_dir / "truth.npz", **truth)
        _write_json(oracle_dir / "truth.json", {
            "target": "P_XY versus P_X P_Y of the observed integer variables",
            "family": family, "m": m, "rho": rho,
            "S": truth["S"], "S_closed_form": truth["S_closed_form"],
            "S_formula": ("rho^2/(4-rho^2)" if family == "two_block" else
                          "rho^2*(m-1)/((2+rho*(m-1))*(2-rho))"),
            "truth_kind": "analytic finite population; float64 rounding only",
            "monte_carlo_standard_error": None,
            "use": "evaluation and truth checks only; not fitting or hyperparameter selection",
            "oracle_file_sha256": hashlib.sha256((oracle_dir / "truth.npz").read_bytes()).hexdigest(),
        })
        manifest["cases"].append(entry)
    manifest["n_cases"] = len(cases)
    manifest["n_independent_base_observations_total"] = len(cases) * sum(counts.values())
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--m", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--rho", nargs="+", type=float, default=[0.0, 0.25, 0.75])
    parser.add_argument("--n-fit", "--n-train", dest="n_fit", type=int, default=4096)
    parser.add_argument("--n-cal", type=int, default=2048)
    parser.add_argument("--n-eval", type=int, default=8192)
    args = parser.parse_args()
    result = generate_suite(args.output_dir, root_seed=args.seed,
        families=tuple(args.families), sizes=tuple(args.m), rhos=tuple(args.rho),
        role_sizes={"fit": args.n_fit, "cal": args.n_cal, "eval": args.n_eval})
    print(json.dumps({"status": "generated", "output_dir": str(args.output_dir.resolve()),
                      "n_cases": result["n_cases"],
                      "n_independent_base_observations_total": result["n_independent_base_observations_total"]},
                     indent=2))


if __name__ == "__main__":
    main()
