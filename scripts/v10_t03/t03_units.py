"""V10 T03 unit table: v10/protocol/T03_UNITS.csv (written by the protocol agent, v10/tools/make_protocol_p2.py; sha256 pinned in
protocol P2 t03.units_file) and an INDEPENDENT derivation of the same rows from decision section 9 (expected_rows), used by the tests
and by `t03_run.py units --check` to confirm the file before any run.  1212 grid rows (252 CAL + 600 main + 120 algo_extra +
240 capacity) plus 7 smoke rows (landing choice, not part of the 1212).
"""
from __future__ import annotations

import collections
import csv
import io
from pathlib import Path

import t03_common as C
import t03_select as S

COLUMNS = ["unit_id", "block", "family", "m", "rho", "method", "K", "lr", "cap", "data_root_seed", "algo_seed", "steps", "ckpt_every",
           "selection_rule", "depends_on"]
BLOCK_COUNTS = {"smoke": 7, "CAL": 252, "main": 600, "algo_extra": 120, "capacity": 240}
SMOKE_CONDITION = ("diagonal_mixture", 32, 0.75)                # landing choice (protocol landing_choices "T03 smoke")
SMOKE_LR = 3e-4


def conditions() -> list[tuple[str, int, float]]:
    return [(f, m, r) for f in C.FAMILIES for m in C.MS for r in C.RHOS]


def _rho(r: float) -> str:
    return "0" if float(r) == 0 else repr(float(r))


def _row(uid, block, cond, method, K, lr, cap, data_seed, algo_seed, rule, deps) -> dict:
    f, m, r = cond
    return {"unit_id": uid, "block": block, "family": f, "m": str(m), "rho": _rho(r), "method": method,
            "K": "NA" if K is None else str(K), "lr": lr if isinstance(lr, str) else C.fmt_lr(lr),
            "cap": "NA" if cap is None else str(cap), "data_root_seed": str(data_seed), "algo_seed": str(algo_seed),
            "steps": str(C.STEPS), "ckpt_every": str(C.CKPT_EVERY), "selection_rule": rule, "depends_on": ";".join(deps)}


def expected_rows() -> list[dict]:
    """Decision 9.2-9.4 grid in the frozen orders of protocol P2 t03.units_file.order."""
    rows, cal_ids = [], {}
    smoke = []
    for method in C.METHODS:
        for _, lr, cap in ([(None, SMOKE_LR, c) for c in C.M1_CAPS] if method == "MF_VCS_M1" else [(None, SMOKE_LR, None)]):
            smoke.append((method, cap))
    smk_ids = []
    for i, (method, cap) in enumerate(smoke, 1):
        uid = f"T03_SMK_{i:03d}"
        smk_ids.append(uid)
        rows.append(_row(uid, "smoke", SMOKE_CONDITION, method, C.K_MAIN if method in C.MF_METHODS else None, SMOKE_LR, cap,
                         C.SMOKE_DATA_SEED, C.TUNING_SEED, "SMOKE_NONE", []))
    n = 0
    for cond in conditions():
        for method in C.METHODS:
            for _, lr, cap in S.candidates_for(method):
                n += 1
                uid = f"T03_CAL_{n:04d}"
                cal_ids.setdefault((cond, method), []).append(uid)
                rows.append(_row(uid, "CAL", cond, method, C.K_MAIN if method in C.MF_METHODS else None, lr, cap, C.TUNING_SEED,
                                 C.TUNING_SEED, "CAL_UJ_SELECT", smk_ids))
    n = 0
    for ds in C.MAIN_DATA_SEEDS:
        for cond in conditions():
            for method in C.METHODS:
                n += 1
                rows.append(_row(f"T03_MAIN_{n:04d}", "main", cond, method, C.K_MAIN if method in C.MF_METHODS else None, "CAL",
                                 "CAL" if method == "MF_VCS_M1" else None, ds, C.MAIN_ALGO_SEED, "OWN_CAL_UJ_CKPT", cal_ids[(cond, method)]))
    n = 0
    for a in C.EXTRA_ALGO_SEEDS:
        for cond in conditions():
            for method in C.METHODS:
                n += 1
                rows.append(_row(f"T03_ALGO_{n:04d}", "algo_extra", cond, method, C.K_MAIN if method in C.MF_METHODS else None, "CAL",
                                 "CAL" if method == "MF_VCS_M1" else None, C.MAIN_DATA_SEEDS[0], a, "OWN_CAL_UJ_CKPT", cal_ids[(cond, method)]))
    n = 0
    for ds in C.MAIN_DATA_SEEDS:
        for cond in C.CAPACITY_CONDITIONS:
            for method in C.MF_METHODS:
                for K in C.K_CAPACITY:
                    n += 1
                    rows.append(_row(f"T03_CAP_{n:04d}", "capacity", cond, method, K, "CAL_K16", "CAL_K16" if method == "MF_VCS_M1" else None,
                                     ds, C.MAIN_ALGO_SEED, "OWN_CAL_UJ_CKPT", cal_ids[(cond, method)]))
    return rows


def render_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def load_rows(path: Path = C.UNITS_CSV, *, verify_sha: bool = True) -> list[dict]:
    raw = Path(path).read_bytes()
    digest = C.sha256_bytes(raw)
    if verify_sha and digest != C.UNITS_CSV_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest} != protocol P2 pin {C.UNITS_CSV_SHA256}; refusing to run units from a changed table")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if reader.fieldnames != COLUMNS:
        raise RuntimeError(f"{path}: columns {reader.fieldnames} != {COLUMNS}")
    rows = list(reader)
    counts = dict(collections.Counter(r["block"] for r in rows))
    if counts != BLOCK_COUNTS or len({r["unit_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"{path}: block counts {counts} != {BLOCK_COUNTS} or duplicate unit ids")
    return rows


def check_against_derivation(path: Path = C.UNITS_CSV) -> dict:
    """Byte comparison of the protocol file with the independent derivation, plus a row-level diff summary."""
    raw = Path(path).read_bytes()
    exp = expected_rows()
    got = load_rows(path, verify_sha=False)
    diff = [(e["unit_id"], k, e[k], g[k]) for e, g in zip(exp, got) for k in COLUMNS if e[k] != g[k]]
    return {"file": str(Path(path)), "sha256": C.sha256_bytes(raw), "pinned_sha256": C.UNITS_CSV_SHA256,
            "rows_file": len(got), "rows_derived": len(exp), "bytes_identical": raw == render_csv(exp),
            "field_differences": diff[:50], "n_field_differences": len(diff),
            "status": "PASS" if raw == render_csv(exp) and C.sha256_bytes(raw) == C.UNITS_CSV_SHA256 else "FAIL"}


def parse(row: dict) -> dict:
    """Typed view: m / seeds / steps int, rho float, K int or None, lr float or 'CAL' / 'CAL_K16', cap int / None / placeholder."""
    def num_or(v, conv):
        return None if v == "NA" else (v if v in ("CAL", "CAL_K16") else conv(v))
    return {"unit_id": row["unit_id"], "block": row["block"], "family": row["family"], "m": int(row["m"]), "rho": float(row["rho"]),
            "method": row["method"], "K": num_or(row["K"], int), "lr": num_or(row["lr"], float), "cap": num_or(row["cap"], int),
            "data_root_seed": int(row["data_root_seed"]), "algo_seed": int(row["algo_seed"]), "steps": int(row["steps"]),
            "ckpt_every": int(row["ckpt_every"]), "selection_rule": row["selection_rule"],
            "depends_on": [d for d in row["depends_on"].split(";") if d]}
