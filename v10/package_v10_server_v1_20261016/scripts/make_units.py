"""Create an explicit experiment ledger; this script never submits jobs."""
import argparse
import csv
import itertools
import json
from pathlib import Path


def make_units(protocol):
    rows = []
    for n, k, method in itertools.product(protocol["compute"]["positions"],
                                          protocol["compute"]["simplex_widths"],
                                          protocol["compute"]["methods"]):
        rows.append({"stage": "T02", "dataset": "fixed_simplex_features", "method": method,
                     "n_positions": n, "simplex_width": k, "seed": protocol["compute"]["seed"],
                     "fit_fraction": "", "status": "designed_not_launched"})
    for dataset, method, seed in itertools.product(protocol["registration"]["datasets"],
                                                  protocol["registration"]["learned_methods"],
                                                  protocol["registration"]["training_seeds"]):
        rows.append({"stage": "T04_T05", "dataset": dataset, "method": method,
                     "n_positions": "inherit_locked_host", "simplex_width": "select_on_CAL",
                     "seed": seed, "fit_fraction": 1.0, "status": "designed_not_launched"})
    for dataset, method in itertools.product(protocol["registration"]["datasets"],
                                            protocol["registration"]["nonlearned_methods"]):
        rows.append({"stage": "T05", "dataset": dataset, "method": method,
                     "n_positions": "inherit_locked_host", "simplex_width": "not_applicable",
                     "seed": "not_applicable", "fit_fraction": "not_applicable",
                     "status": "designed_not_launched"})
    for i, row in enumerate(rows):
        row["unit_id"] = f"V10_{i:04d}"
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = make_units(json.loads(a.protocol.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["unit_id", "stage", "dataset", "method", "n_positions", "simplex_width", "seed", "fit_fraction", "status"]
    with a.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"units": len(rows), "output": str(a.output), "submitted_jobs": 0}))


if __name__ == "__main__":
    main()
