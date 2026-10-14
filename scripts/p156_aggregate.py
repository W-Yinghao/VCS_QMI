"""P156 aggregate: python scripts/p156_aggregate.py --dir /home/infres/yinwang/CS_QMI/outputs/P156_sep --out reports/P156_SEP_results/p156_aggregate"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import sep_agg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cells, smoke = sep_agg.load(a.dir); res = sep_agg.aggregate(cells); res["n_cell_keys"] = len(cells); res["smoke_only"] = smoke
    res["n_files"] = sum(len(v) for v in cells.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=str)); md = sep_agg.markdown(res, smoke); Path(a.out + ".md").write_text(md)
    print(md); return 0


if __name__ == "__main__":
    raise SystemExit(main())
