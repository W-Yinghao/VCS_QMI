"""P117 runner (v5 NEXT-E-CAL).  One process runs a list of cells and writes one JSON per cell (cells already written are skipped).

    python scripts/p117_cal.py --cells 'C1_gauss_mid:4096:0;C3_xor:16384:2' --out-dir outputs/P117_cal [--smoke] [--cpu]
    python scripts/p117_cal.py --list            (print the full cell list, one token per line)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import p117  # noqa: E402
from vcs_estim.run import git_commit  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=""); ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-dir", default="/home/infres/yinwang/CS_QMI/outputs/P117_cal"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.list:
        for c, N, s in p117.cal_cells():
            print(f"{c}:{N}:{s}")
        return 0
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); meta = {"commit": git_commit(), "utc": utc_now(), "device": str(device), "smoke": a.smoke}
    for tok in [t for t in a.cells.split(";") if t]:
        c, N, s = tok.split(":"); dst = out / (f"P117_{c}_{N}_{s}" + ("_smoke" if a.smoke else "") + ".json")
        if dst.is_file():
            print(f"skip {tok} (exists)", flush=True); continue
        r = p117.cal_cell(c, int(N), int(s), device, a.smoke); r["meta"] = meta; atomic_write_json(dst, r)
        msg = []
        for k, v in r["per_kind"].items():
            m = v["mechanism_same_T0"]; d = v["decomposition_DIAG"][str(p117.M_BINS[-1])]
            msg.append(f"{k}: post {m['identity']['posterior_mse']:.4f}->{m['selected']['posterior_mse']:.4f} ({m['selected']['which']}) "
                       f"Jerr {m['identity']['J_err']:+.4f}->{m['selected']['J_err']:+.4f} A {d['A_score_loss']:.4f} "
                       f"B_id {d['per_calibrator']['identity']['B_calibration']:.4f} res {d['per_calibrator']['identity']['residual']:+.1e} "
                       f"e2e_id Jerr {v['end_to_end_equal_budget']['identity_on_FIT_plus_CAL']['J_err']:+.4f}")
        print(f"[{tok}] {r['wall_seconds']:.0f}s S={r['truth']['S']:.4f} " + " | ".join(msg), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
