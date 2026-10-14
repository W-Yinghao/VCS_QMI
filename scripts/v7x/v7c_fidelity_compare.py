"""V7C-03 source fidelity, step 2 (label-free): compare the frozen local question generators with the authors' released question files, and
check the structure of the local DSG graphs on GenAI-Rank.  No human image rating is read.

    python -I scripts/v7x/v7c_fidelity_compare.py --fid /home/infres/yinwang/CS_QMI/outputs/V7_CLOSEOUT/fidelity \
        --rank-dsg /projects/EEG-foundation-model/yinghao/outputs_v7/x30/dsg_gen_0of2.jsonl /projects/EEG-foundation-model/yinghao/outputs_v7/x30/dsg_gen_1of2.jsonl \
        --out reports/V7_CLOSEOUT/V7C_03_fidelity

Metrics (per prompt, then mean / median): counts; DSG broad-category distribution and the share of local tuples outside the released taxonomy;
tuple overlap (same broad category and argument-token Jaccard >= 0.5, greedy one-to-one) as precision (local matched / local) and recall
(released matched / released); dependency validity (parents exist, acyclic, entities depend on 0 only); TIFA element overlap (token Jaccard >= 0.5)
and question / answer forms.  A blind A / B review table (local vs released in random order per prompt, seed 2718) is written for a human check;
its key is kept next to the outputs, not in the table.  Thresholds are descriptive: the reading is fixed in V7C_03 before the table is read.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics as st
from collections import Counter
from pathlib import Path

TUP = re.compile(r"^\s*(\d+)\s*\|\s*([a-zA-Z ]+?)\s*-\s*([a-zA-Z ]*?)\s*\((.*)\)\s*$")
YESNO = re.compile(r"^\s*(is|are|does|do|did|can|was|were|has|have|will)\b", re.I)


def toks(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in {"a", "an", "the", "of"}}


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def greedy_match(L: list, R: list, key) -> int:
    """Greedy one-to-one matches between local items L and released items R; key(l, r) -> similarity or None."""
    pairs = [(key(l, r), i, j) for i, l in enumerate(L) for j, r in enumerate(R)]
    pairs = sorted((p for p in pairs if p[0] is not None), reverse=True); used_l, used_r, n = set(), set(), 0
    for s, i, j in pairs:
        if s < 0.5 or i in used_l or j in used_r:
            continue
        used_l.add(i); used_r.add(j); n += 1
    return n


def parse_local_tuples(raw: dict) -> list[dict]:
    out = []
    for line in str(raw.get("tuple", "")).splitlines():
        m = TUP.match(line)
        if m:
            out.append({"id": int(m.group(1)), "broad": m.group(2).strip().lower(), "detailed": m.group(3).strip().lower(), "args": m.group(4)})
    return out


def dep_valid(qid2dep: dict, qid2tuple: dict) -> dict:
    ids = {int(k) for k in qid2dep}; bad_parent = cyc = ent_bad = 0
    for k, ps in qid2dep.items():
        k = int(k)
        if any(p != 0 and p not in ids for p in ps):
            bad_parent += 1
        if str(qid2tuple.get(str(k), "")).startswith("entity") and any(p != 0 for p in ps):
            ent_bad += 1
    G = {int(k): [p for p in ps if p != 0] for k, ps in qid2dep.items()}; state = {}

    def dfs(u):
        state[u] = 1
        for v in G.get(u, []):
            if state.get(v) == 1 or (state.get(v) is None and v in G and dfs(v)):
                return True
        state[u] = 2; return False
    cyc = int(any(state.get(u) is None and dfs(u) for u in G))
    return {"bad_parent": bad_parent, "cyclic": cyc, "entity_with_parent": ent_bad}


def summ(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "mean": round(st.mean(xs), 4) if xs else None, "median": st.median(xs) if xs else None}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--fid", required=True); ap.add_argument("--rank-dsg", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); F = Path(a.fid); res = {"dsg_fidelity": {}, "tifa_fidelity": {}, "rank_dsg_structure": {}}; rng = random.Random(2718); blind, key = [], []
    # ---------------- DSG fidelity
    rel = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(F / "dsg1k_50_released.jsonl")}
    loc = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(F / "local_dsg_gen.jsonl")}
    tax = Counter(t["category_broad"].strip().lower() for r in rel.values() for t in r["tuples"]); taxset = set(tax)
    rows = []
    for pid, R in rel.items():
        L = loc.get(pid); lt = parse_local_tuples(L["raw"]) if L and isinstance(L.get("raw"), dict) else []
        rt = [{"broad": t["category_broad"].strip().lower(), "args": re.sub(r"^.*?\(|\)\s*$", "", t["tuple"])} for t in R["tuples"]]
        m = greedy_match(lt, rt, lambda l, r: jac(toks(l["args"]), toks(r["args"])) if l["broad"] == r["broad"] else None)
        m_any = greedy_match(lt, rt, lambda l, r: jac(toks(l["args"]), toks(r["args"])))      # category-agnostic: a relabelled taxonomy is not a content miss
        dv = dep_valid(L["qid2dependency"], L["qid2tuple"]) if L and L.get("qid2dependency") else None
        rel_dep = {"entity_with_parent": sum(1 for t in R["tuples"] if t["category_broad"].strip().lower() == "entity" and any(d != 0 for d in t["dependency"])),
                   "entity": sum(1 for t in R["tuples"] if t["category_broad"].strip().lower() == "entity")}
        rows.append({"prompt_id": pid, "n_local": len(lt), "n_released": len(rt), "matched": m, "precision": m / len(lt) if lt else None,
                     "recall": m / len(rt) if rt else None, "precision_any_category": m_any / len(lt) if lt else None,
                     "recall_any_category": m_any / len(rt) if rt else None, "released_dep": rel_dep,
                     "local_entity": sum(1 for t in lt if t["broad"] == "entity"), "outside_taxonomy": sum(1 for t in lt if t["broad"] not in taxset),
                     "empty_detailed": sum(1 for t in lt if not t["detailed"]), "parse_error": (L or {}).get("parse_error"), "dep": dv})
        order = [("local", [q for _, q in sorted(((int(k), v) for k, v in ((L or {}).get("qid2question") or {}).items()))]),
                 ("released", [t["question"] for t in R["tuples"]])]; rng.shuffle(order)
        blind.append({"set": "DSG-1k", "prompt_id": pid, "prompt": R["prompt"], "A": " | ".join(order[0][1]), "B": " | ".join(order[1][1])})
        key.append({"set": "DSG-1k", "prompt_id": pid, "A": order[0][0], "B": order[1][0]})
    lc = Counter(t["broad"] for pid in rel for t in (parse_local_tuples(loc[pid]["raw"]) if pid in loc and isinstance(loc[pid].get("raw"), dict) else []))
    res["dsg_fidelity"] = {"n_prompts": len(rel), "n_local_present": sum(1 for p in rel if p in loc), "tuples_local": summ([r["n_local"] for r in rows]),
                           "tuples_released": summ([r["n_released"] for r in rows]), "precision": summ([r["precision"] for r in rows]),
                           "recall": summ([r["recall"] for r in rows]), "precision_any_category": summ([r["precision_any_category"] for r in rows]),
                           "recall_any_category": summ([r["recall_any_category"] for r in rows]),
                           "released_entity_with_parent": f"{sum(r['released_dep']['entity_with_parent'] for r in rows)} / {sum(r['released_dep']['entity'] for r in rows)}",
                           "local_entity_with_parent": f"{sum((r['dep'] or {}).get('entity_with_parent', 0) for r in rows)} / {sum(r['local_entity'] for r in rows)}",
                           "local_outside_released_taxonomy": sum(r["outside_taxonomy"] for r in rows),
                           "local_empty_detailed_category": sum(r["empty_detailed"] for r in rows), "released_broad": dict(tax), "local_broad": dict(lc),
                           "parse_errors": sum(1 for r in rows if r["parse_error"]),
                           "dependency": {k: sum((r["dep"] or {}).get(k, 0) for r in rows) for k in ("bad_parent", "cyclic", "entity_with_parent")},
                           "per_prompt": rows}
    # ---------------- TIFA fidelity
    relt = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(F / "tifa_v1_50_released.jsonl")}
    loct = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(F / "local_tifa_qa.jsonl")}
    trows = []
    for pid, R in relt.items():
        L = loct.get(pid) or {}; gen = L.get("qas_generated"); kept = L.get("qas") or []
        le = [q.get("element") or "" for q in kept]; re_ = [q.get("element") or "" for q in R["qas"]]
        m = greedy_match(le, re_, lambda x, y: jac(toks(x), toks(y)))
        trows.append({"prompt_id": pid, "n_generated": len(gen) if isinstance(gen, list) else None, "parse_failed": not isinstance(gen, list),
                      "n_kept": len(kept), "n_released": len(R["qas"]), "element_matched": m, "element_precision": m / len(le) if le else None,
                      "element_recall": m / len(re_) if re_ else None,
                      "yesno_local": sum(1 for q in kept if YESNO.match(q.get("question") or "")), "yesno_released": sum(1 for q in R["qas"] if YESNO.match(q["question"]))})
        order = [("local", [q.get("question") for q in kept]), ("released", [q["question"] for q in R["qas"]])]; rng.shuffle(order)
        blind.append({"set": "TIFA v1.0", "prompt_id": pid, "prompt": R["prompt"], "A": " | ".join(map(str, order[0][1])), "B": " | ".join(map(str, order[1][1]))})
        key.append({"set": "TIFA v1.0", "prompt_id": pid, "A": order[0][0], "B": order[1][0]})
    res["tifa_fidelity"] = {"n_prompts": len(relt), "parse_failed": sum(1 for r in trows if r["parse_failed"]), "generated": summ([r["n_generated"] for r in trows]),
                            "kept": summ([r["n_kept"] for r in trows]), "released": summ([r["n_released"] for r in trows]),
                            "no_kept_question": sum(1 for r in trows if r["n_kept"] == 0), "element_precision": summ([r["element_precision"] for r in trows]),
                            "element_recall": summ([r["element_recall"] for r in trows]),
                            "yesno_share_local": round(sum(r["yesno_local"] for r in trows) / max(1, sum(r["n_kept"] for r in trows)), 4),
                            "yesno_share_released": round(sum(r["yesno_released"] for r in trows) / max(1, sum(r["n_released"] for r in trows)), 4),
                            "per_prompt": trows}
    # ---------------- GenAI-Rank local DSG structure
    R2 = [json.loads(l) for f in a.rank_dsg for l in open(f)]; bc = Counter(); empty_det = outside = 0; dep_tot = Counter(); nq = []
    for r in R2:
        lt = parse_local_tuples(r["raw"]) if isinstance(r.get("raw"), dict) else []; bc.update(t["broad"] for t in lt)
        empty_det += sum(1 for t in lt if not t["detailed"]); outside += sum(1 for t in lt if t["broad"] not in taxset); nq.append(len(r.get("qid2question") or {}))
        if r.get("qid2dependency"):
            dep_tot.update(dep_valid(r["qid2dependency"], r["qid2tuple"]))
    res["rank_dsg_structure"] = {"n_prompts": len(R2), "parse_errors": sum(1 for r in R2 if r.get("parse_error")), "questions": summ(nq),
                                 "broad_categories": dict(bc), "tuples_outside_released_taxonomy": outside, "tuples_with_empty_detailed_category": empty_det,
                                 "dependency": dict(dep_tot)}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    with open(str(out) + "_blind_review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "prompt_id", "prompt", "A", "B"]); w.writeheader(); w.writerows(blind)
    Path(F / "blind_review_key.json").write_text(json.dumps(key, indent=1))
    d, t, g = res["dsg_fidelity"], res["tifa_fidelity"], res["rank_dsg_structure"]
    print(json.dumps({"dsg": {k: v for k, v in d.items() if k != "per_prompt"}, "tifa": {k: v for k, v in t.items() if k != "per_prompt"}, "rank_dsg": g}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
