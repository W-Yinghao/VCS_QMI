"""Print the draft complete-unit plan. No Torch imports, data reads or submissions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.supplement.contract import FAMILIES, candidates, schedule


def main():
    methods = []
    for family in FAMILIES:
        recipe = schedule(family)
        methods.append({"family": family, "CAL_candidates": candidates(family),
                        "FINAL_seeds": recipe["final_seeds"], "schedule": recipe,
                        "activation": "V9 closeout, new targeted T01/T02/runner PASS, versioned method freeze"})
    print(json.dumps({"status": "DRAFT_IMPLEMENTATION_TESTS_DEFERRED", "methods": methods,
                      "cal_candidates": sum(len(m["CAL_candidates"]) for m in methods),
                      "final_units": sum(len(m["FINAL_seeds"]) for m in methods),
                      "submitted_jobs": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
