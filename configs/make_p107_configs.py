"""P107 — package v4 module A, next batch (owner 2026-10-01): five full 800-epoch CIFAR-10 cells built around P104 G2 (fixed angular scorer
f = 2s − 1, i.e. a = 2, b = −1).  Every YAML is P104's G2 (or G2F) cell — itself the frozen 8x recipe
`cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml` plus the G2 / G2F critic fields — with ONLY the fields below changed:

  A-L1  G2  + objective.loss js_matched_logistic + objective.js_fixed_scorer true (marker) (fixed scorer, K8 cross-view, negative right detach) — matched JS, fixed geometry
  A-L2  G2F + objective.loss js_matched_logistic           (a, b learned from (2, −1), K8, right detach)       — matched JS, learned geometry
  A-P1  G2  + pairing.negative_detach false                 (K8, plain autodiff through the negatives)            — G2 + full negative gradient
  A-P2  G2  + pairing.pair_scope all_view_tokens (chunk 256), right detach                                        — G2 + complete pairing
  A-P3  G2  + all_view_tokens + negative_detach false                                                              — complete pairing + full gradient

The G2 / G2F critic fields come from `make_p104_configs.apply` (imported, not copied), so the scorer is byte-identical to P104's.

    python configs/make_p107_configs.py [--write] [--seeds 0] [--uids A-L1,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))
from make_p104_configs import BASE, apply as p104_apply, flat  # noqa: E402

STAGE = "P107_v4_A_batch"
ORDER = ["A-L1", "A-L2", "A-P1", "A-P2", "A-P3"]
PARENT = {"A-L1": "G2", "A-L2": "G2F", "A-P1": "G2", "A-P2": "G2", "A-P3": "G2",
          "A-P2F": "G2F", "A-P3F": "G2F"}  # addendum 1: same-initialisation learned controls of the selected A-P2 / A-P3


def apply(c: dict, uid: str) -> dict:
    c = p104_apply(c, PARENT[uid])
    pair, obj = c["pairing"], c["objective"]
    if uid in ("A-L1", "A-L2"):
        obj["loss"] = "js_matched_logistic"
        if uid == "A-L1":
            obj["js_fixed_scorer"] = True  # explicit marker required by the config policy for JS on the fixed scorer
    elif uid == "A-P1":
        pair["negative_detach"] = False
    elif uid in ("A-P2F", "A-P3F"):  # P107 addendum 1 (§4.1 control): a, b learned from (2, -1); pairing / routing exactly as A-P2 / A-P3
        pair["pair_scope"] = "all_view_tokens"; pair["all_view_chunk"] = 256
        if uid == "A-P3F":
            pair["negative_detach"] = False
    elif uid == "A-P2":
        pair["pair_scope"] = "all_view_tokens"; pair["all_view_chunk"] = 256
    elif uid == "A-P3":
        pair["pair_scope"] = "all_view_tokens"; pair["all_view_chunk"] = 256; pair["negative_detach"] = False
    else:
        raise ValueError(uid)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--uids", default="")
    a = ap.parse_args()
    rows, shas = [], {}
    for s in [int(x) for x in a.seeds.split(",")]:
        base = yaml.safe_load(open(ROOT / "configs" / BASE.format(s=s)))
        assert base["train"]["epochs"] == 800 and base["views"]["count"] == 4 and base["pairing"]["negative_detach"] is True
        for uid in (a.uids.split(",") if a.uids else ORDER):
            c = apply(json.loads(json.dumps(base)), uid)
            c["run"]["stage"] = STAGE
            ce = c["logging"]["checkpoint_epochs"]
            if 20 not in ce:
                c["logging"]["checkpoint_epochs"] = sorted(ce + [20])
            tag = uid.replace("-", "")
            name = f"cifar10_hpY_{tag}_views4_800ep_seed{s}.yaml"
            txt = yaml.safe_dump(c, sort_keys=False)
            sha = hashlib.sha256(txt.encode()).hexdigest()
            fb, fc = flat(base), flat(c)
            diff = {k: (fb.get(k, "<absent>"), fc.get(k, "<absent>")) for k in sorted(set(fb) | set(fc)) if fb.get(k, "<absent>") != fc.get(k, "<absent>")}
            run_id = f"P107_{tag}_views4_800ep_seed{s}"
            rows.append((run_id, f"configs/{name}", "epoch_800.pt"))
            shas[name] = {"sha256": sha, "logical_variant": uid, "parent_p104_cell": PARENT[uid], "seed": s, "base": BASE.format(s=s),
                          "diff_vs_base": {k: list(v) for k, v in diff.items()}}
            print(f"{run_id:34s} {sha[:16]}  " + "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in diff.items()))
            if a.write:
                (ROOT / "configs" / name).write_text(txt)
    if a.write:
        p = ROOT / "configs" / "P107_SHA256.json"
        j = json.load(open(p)) if p.is_file() else {"stage": STAGE, "configs": {}}
        j["configs"].update(shas); j.setdefault("generated_utc", []).append(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        p.write_text(json.dumps(j, indent=1))
        print("wrote configs + configs/P107_SHA256.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
