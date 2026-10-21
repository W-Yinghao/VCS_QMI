"""Print the exact sbatch lines of the TEST_REPEAT134 stages from the frozen records (prints only; submits nothing).

    submit_lines.py --commit <execution commit>      v6: the TEST lines of the 3 CoMIR_native units
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402

SBATCH = "/home/infres/yinwang/CS_QMI/ssl_pilot_v10_test_repeat_exec_v6_20261020/slurm/v10_test_repeat/gpu.sbatch"


def short(family: str) -> str:
    return family.replace("InfoNCE_host[J_perm1]", "InfoNCE").replace("_native", "").replace("_2D_adapted", "").replace("_VCS", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    a = ap.parse_args(argv)
    rec = C.records()
    pins = f"--export=ALL,V10_TR_EXPECT_ACTIVATION={rec['activation_sha256']},V10_TR_EXPECT_COMMIT={a.commit}"
    prot, man, v6 = rec["protocol"], rec["manifest"], rec["activation"]["v6"]
    print("# v6: the 3 CoMIR_native units only (same-job DEV32 re-evaluation, in-job check, then TEST134; recorded GPU type).")
    print("# The inputs (INPUT_LOCK_V5) and the other 39 units stay under v5.")
    for t in prot["jobs"]["test_units"]:
        if t["unit"] not in v6["units"]:
            continue
        u = next(x for x in man["units"] if x["key"] == t["unit"])
        print(f"sbatch --job-name=v10t6_{short(u['family'])}_{u['seed']} --partition={t['gpu_type']} "
              f"--time={t['time_limit']} {pins} {SBATCH} test {u['key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
