#!/usr/bin/env python
"""V9-N2 overlap audit of the "independent batch" rho interval (review item U07).  Read-only on the precheck cells; writes only --out-dir.

What is audited.  pipeline.compute_scores (src/v9/n2/pipeline.py:71-73) holds out 20 % of the U images, drawn by OBSERVED (receiving) image id with
rng([roles seed, 9031]), as the "independent batch" of the rho interval; the point estimate uses the other 80 %.  weights.weights_for_method then
calls estimate_rho twice (src/v9/n2/weights.py:174 and :177) with the SAME reference rows (A-reference-moments P and their within-role Q
re-pairing).  This script reproduces both groups and the Q re-pairings exactly and counts the links between them:

* caption-source links: a U pair is (caption c, observed image o); the caption's source image is src(c) = c // 5 (data.load_precomp, data.py:45:
  caption_image = {i: i // 5}; noise.make_pair_table keeps it as image_id until inject_mismatch replaces the observed image).  A mismatched pair
  whose o and src(c) lie in different groups carries the source image's caption into the other group;
* Q donor links: weights.repair_within_role gives every caption of critic_fit / critic_select / reference_moments the observed image of another
  row of the same role (one rng([roles seed, 9030]) for the three roles, in that order: pipeline.py:63-69);
* the image graph: one node per train image; an edge joins o and src(c) of every mismatched U pair and src(c) and the donor of every Q row (the
  pair <-> image incidence graph read on its image nodes); components holding images of both groups are what the split leaves connected;
* provenance units: the package contract of reference core.role_disjointness (each underlying unit in ONE cluster) evaluated with image ids as
  units on the clusters the two estimate_rho calls use (P: reference images, Q: reference Q rows, R: observed U images of the call's group).

Inputs per cell <out_root>/<dataset>/rho<rho>/s<seed>/: roles.json, pairs.observed_pairs.json, pairs.noise_manifest.json; for the checks only:
weights/<method>.csv(.lock.json) and precheck/precheck_<dataset>_rho<rho>_s<seed>.json.  The eval-only label file is never opened.

    python scripts/v9/n2_overlap_audit.py --out-root /projects/EEG-foundation-model/yinghao/outputs_v9/n2 --dataset f30k --rhos 0.4 0 \
        --seeds 901 902 903 917 --out-dir reports/V9/V9_N2_precheck/overlap_audit_20261016 [--pipeline-src src] [--code-root .]

--pipeline-src (default: <repo>/src when it exists) also runs the pipeline's own compute_scores on each cell with the teacher stubbed out
(load_teacher / score_pairs replaced: no model, no GPU, nothing written) and compares its held-out set, estimation mask and Q rows with the
reproduction here.  Needs torch (the pipeline imports it); skipped and recorded otherwise.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANCHOR_ROLES = ("teacher_warmup", "critic_fit", "critic_select", "reference_moments")
Q_ROLES = ("critic_fit", "critic_select", "reference_moments")          # pipeline.compute_scores: ONE rng([seed, 9030]) for the three, this order
CAPTIONS_PER_IMAGE = 5                                                    # data.load_precomp: caption_image = {i: i // 5}
GROUPS = ("E", "H")                                                       # E = estimation 80 %, H = held-out "independent" 20 %
SCALAR_CRITICS = ("VCS_Weight", "BCE_Weight")                             # T = tanh(a s + b): c is recomputable from the locked CSV scores
INTERVAL_METHODS = ("VCS_Weight", "BCE_Weight", "RuLSIF_Weight", "Nystrom_Weight")   # locks that record rho.independent_batch
CODE_FILES = ("src/v9/n2/pipeline.py", "src/v9/n2/weights.py", "src/v9/n2/noise.py", "src/v9/n2/roles.py", "src/v9/n2/data.py")
CODE_FACTS = {
    "groups": "pipeline.py:72-73: held = rng([roles seed, 9031]).choice(sorted distinct observed U image ids, round(20 %), no replacement); "
              "U_est_mask = observed image not in held",
    "two_calls": "weights.py:174 (estimation batch) and :177 (independent batch) pass the same T_P_ref, T_Q_ref, ref_P_img, ref_Q_img; the critic T "
                 "is fitted once (critic_fit / critic_select) for both",
    "pipeline_disjointness_units": "weights.py:135 builds units [f'P:{u}'], [f'Q:{u}'], [f'R:{u}'] from the cluster ids themselves, so "
                                   "REF.role_disjointness cannot see an image shared by two clusters",
    "caption_source": "data.py:45: caption_image = {i: i // 5}; noise.py:33 row image_id = the caption's own image; noise.py:41-52 replaces the "
                      "observed image of the selected U pairs by a permutation of those pairs' own images",
    "q_rows": "weights.py:105-119 repair_within_role (donor = observed image of another row of the same role); pipeline.py:63-69 one rng for "
              "critic_fit, critic_select, reference_moments in that order",
}


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_ids(ids) -> str:                                                 # roles.sha256_ids
    return hashlib.sha256(("\n".join(str(i) for i in ids)).encode()).hexdigest()


def _id_key(s):
    s = str(s)
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


# ------------------------------------------------------------------------------------------------ exact reproductions of the pipeline rules
def caption_source(caption_id, per_image: int = CAPTIONS_PER_IMAGE) -> str:
    """Source image of a caption in the precomp layout (data.load_precomp: caption i belongs to image i // 5)."""
    return str(int(caption_id) // per_image)


def heldout_images(u_observed_images, roles_seed: int) -> set:
    """pipeline.compute_scores:72 verbatim: 20 % of the distinct observed U images (sorted ids), rng([roles seed, 9031]), no replacement."""
    u_imgs = sorted(set(u_observed_images)); rng_u = np.random.default_rng([roles_seed, 9_031])
    return set(rng_u.choice(u_imgs, size=max(1, int(round(0.2 * len(u_imgs)))), replace=False).tolist()) if len(u_imgs) >= 5 else set()


def repair_within_role(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """weights.repair_within_role verbatim (the Q of one role: donors are observed images of other rows of the same role)."""
    imgs = [r["observed_image_id"] for r in rows]; n = len(rows)
    if n < 2:
        raise ValueError("a role needs >= 2 pairs to build Q")
    perm = rng.permutation(n); shift = int(rng.integers(1, n))
    out = []
    for k, r in enumerate(rows):
        donor = imgs[perm[(int(np.where(perm == k)[0][0]) + shift) % n]]
        if donor == r["observed_image_id"]:
            donor = imgs[perm[(int(np.where(perm == k)[0][0]) + shift + 1) % n]]
        q = dict(r); q["observed_image_id"] = donor; q["q_of"] = r["pair_id"]; out.append(q)
    return out


def q_rows_by_role(observed: list[dict], roles_seed: int) -> dict:
    """pipeline.compute_scores:63-69: P = the role's rows in table order; Q = repair_within_role(P, rng), one rng([seed, 9030]) for the three roles."""
    rng = np.random.default_rng([roles_seed, 9_030])
    return {role: repair_within_role([x for x in observed if x["role"] == role], rng) for role in Q_ROLES}


# ------------------------------------------------------------------------------------------------ counting logic (pure; unit-tested)
class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n)); self.r = [0] * n

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1


def _unit_reuse(clusters: list) -> dict:
    """clusters: [(type, set of underlying units)].  Package contract (reference core.role_disjointness): every unit lies in ONE cluster."""
    cnt = Counter(u for _, us in clusters for u in us); reused = {u for u, c in cnt.items() if c > 1}
    n_t = Counter(t for t, _ in clusters); s_t = Counter(t for t, us in clusters if us & reused)
    return {"n_clusters": {t: n_t[t] for t in sorted(n_t)}, "units_in_more_than_one_cluster": len(reused),
            "clusters_sharing_a_unit": {t: s_t.get(t, 0) for t in sorted(n_t)}}


def count_overlap(u_pairs, held, q_links=(), anchor_role_of=None) -> dict:
    """u_pairs: [(pair_id, observed_image, caption_source_image)] for the U pairs.  held: the held-out ("independent") images; as in the pipeline a
    pair belongs to the group of its OBSERVED image (H if held, else E).  q_links: [(role, caption_source_image, donor_image)] for the Q rows
    (role in Q_ROLES; 'reference_moments' rows are the P / Q clusters of both estimate_rho calls).  anchor_role_of: {image: role} for anchor
    images (graph nodes and labels).  Returns plain counts (JSON-ready)."""
    u_pairs = list(u_pairs); q_links = list(q_links); held = set(held); anchor_role_of = dict(anchor_role_of or {})
    obs_imgs = {o for _, o, _ in u_pairs}
    grp = {i: ("H" if i in held else "E") for i in obs_imgs}

    def label(i):
        return grp.get(i) or anchor_role_of.get(i) or "unassigned"

    # groups and caption-source links ---------------------------------------------------------------------------------------------------
    groups = {g: {"n_pairs": 0, "n_images": sum(1 for i in obs_imgs if grp[i] == g), "n_mismatched_pairs": 0} for g in GROUPS}
    groups["held_images_not_observed_in_U"] = len(held - obs_imgs)
    links = Counter(); units = {g: set() for g in GROUPS}; recv = {g: set() for g in GROUPS}; sent = {g: set() for g in GROUPS}
    r_units = defaultdict(set)                                            # R cluster (observed image) -> underlying images (itself + caption sources)
    for _, o, s in u_pairs:
        g = grp[o]; gs = label(s); groups[g]["n_pairs"] += 1; units[g].update((o, s)); r_units[o].update((o, s))
        if o != s:
            groups[g]["n_mismatched_pairs"] += 1; links[f"{g}<-{gs}"] += 1
            if gs in GROUPS and gs != g:
                recv[g].add(o); sent[gs].add(s)
    n_mis = sum(groups[g]["n_mismatched_pairs"] for g in GROUPS); cross = links["H<-E"] + links["E<-H"]
    csl = {"H<-E": links["H<-E"], "E<-H": links["E<-H"], "H<-H": links["H<-H"], "E<-E": links["E<-E"],
           "source_outside_U_groups": sum(v for k, v in links.items() if k.split("<-")[1] not in GROUPS),
           "cross_group_total": cross, "n_mismatched_total": n_mis, "cross_share_of_mismatched": (cross / n_mis) if n_mis else 0.0,
           "notation": "X<-Y = pair observed with an image of group X whose caption comes from an image of group Y"}
    both = units["H"] & units["E"]
    ib = {"total": len(both), "H_images_with_a_caption_in_E": len(sent["H"]), "E_images_with_a_caption_in_H": len(sent["E"]),
          "H_clusters_holding_an_E_caption": len(recv["H"]), "E_clusters_holding_an_H_caption": len(recv["E"]),
          "images_on_any_cross_link": len(sent["H"] | sent["E"] | recv["H"] | recv["E"]),
          "definition": "an image is in a group when it is the observed image or the caption-source image of one of that group's pairs"}

    # Q donors ----------------------------------------------------------------------------------------------------------------------------
    qd, qsets = {}, {}
    for role, a, b in q_links:
        d = qd.setdefault(role, {"n_rows": 0, "donor_in_same_role": 0, "donor_in_other_anchor_role": 0, "donor_in_U": 0, "crossing_H_E": 0,
                                 "receiver_outside_role": 0, "self_donor": 0})
        donors, receivers = qsets.setdefault(role, (set(), set()))
        la, lb = label(a), label(b); d["n_rows"] += 1; donors.add(b); receivers.add(a); d["self_donor"] += int(a == b)
        d["receiver_outside_role"] += int(la != role)
        if lb == role:
            d["donor_in_same_role"] += 1
        elif lb in GROUPS:
            d["donor_in_U"] += 1
        else:
            d["donor_in_other_anchor_role"] += 1
        d["crossing_H_E"] += int(la in GROUPS and lb in GROUPS and la != lb)
    for role, (donors, receivers) in qsets.items():
        qd[role]["n_distinct_donor_images"] = len(donors); qd[role]["n_receiving_images"] = len(receivers)

    # image graph -------------------------------------------------------------------------------------------------------------------------
    nodes = sorted(obs_imgs | {s for _, _, s in u_pairs} | {x for _, a, b in q_links for x in (a, b)} | set(anchor_role_of), key=_id_key)
    idx = {v: k for k, v in enumerate(nodes)}; dsu = _DSU(len(nodes)); n_edges = 0
    for _, o, s in u_pairs:
        if o != s:
            dsu.union(idx[o], idx[s]); n_edges += 1
    for _, a, b in q_links:
        if a != b:
            dsu.union(idx[a], idx[b]); n_edges += 1
    root = [dsu.find(k) for k in range(len(nodes))]; size = Counter(root); first = {}
    for k, r in enumerate(root):
        first.setdefault(r, k)                                            # smallest node (ids sorted) -> deterministic tie-break
    nH = Counter(r for v, r in zip(nodes, root) if grp.get(v) == "H"); nE = Counter(r for v, r in zip(nodes, root) if grp.get(v) == "E")
    pin = {g: Counter() for g in GROUPS}
    for _, o, _ in u_pairs:
        pin[grp[o]][root[idx[o]]] += 1

    def comp(r):
        return {"n_images": size[r], "n_H_images": nH[r], "n_E_images": nE[r], "n_anchor_images": size[r] - nH[r] - nE[r],
                "n_U_pairs_H": pin["H"][r], "n_U_pairs_E": pin["E"][r]}

    u_roots = [r for r in size if nH[r] + nE[r] > 0]; span = [r for r in size if nH[r] and nE[r]]
    key = lambda r: (-size[r], first[r])  # noqa: E731
    graph = {"n_nodes": len(nodes), "n_link_edges": n_edges, "n_components": len(size), "n_components_with_U_images": len(u_roots),
             "n_singleton_U_images": sum(1 for r in u_roots if size[r] == 1),
             "largest_component": comp(min(size, key=key)) if size else None,
             "largest_component_with_U_images": comp(min(u_roots, key=key)) if u_roots else None,
             "n_components_spanning_both_groups": len(span),
             "spanning_components": {"n_H_images": sum(nH[r] for r in span), "n_E_images": sum(nE[r] for r in span),
                                     "n_anchor_images": sum(size[r] - nH[r] - nE[r] for r in span),
                                     "n_U_pairs_H": sum(pin["H"][r] for r in span), "n_U_pairs_E": sum(pin["E"][r] for r in span)},
             "H_images_in_a_component_with_another_H_image": sum(nH[r] for r in size if nH[r] >= 2),
             "n_components_with_2plus_H_images": sum(1 for r in size if nH[r] >= 2)}

    # package contract with image ids as units --------------------------------------------------------------------------------------------
    ref = [(a, b) for role, a, b in q_links if role == "reference_moments"]
    P = [("P", {i}) for i in sorted({a for a, _ in ref}, key=_id_key)]; Q = [("Q", {a, b}) for a, b in ref]
    prov = {f"call_{g}": _unit_reuse(P + Q + [("R", r_units[o]) for o in sorted(obs_imgs, key=_id_key) if grp[o] == g]) for g in GROUPS}
    prov["shared_between_calls"] = {"identical_P_clusters": len(P), "identical_Q_clusters": len(Q), "R_units_in_both_calls": len(both)}
    return {"groups": groups, "caption_source_links": csl, "images_in_both_groups": ib, "q_donors": qd, "graph": graph, "provenance_units": prov}


# ------------------------------------------------------------------------------------------------ checks against what the runs saved / ran
def _read_csv_scores(path: Path):
    ids, score = [], []
    with open(path, newline="") as f:
        rd = csv.reader(f); head = next(rd); i_id, i_s = head.index("pair_id"), head.index("score")
        for row in rd:
            ids.append(row[i_id]); score.append(float(row[i_s]))
    return ids, np.asarray(score, dtype=np.float64)


def saved_record_check(cell: Path, u_ids: list, est_mask: np.ndarray, n_held: int, n_ref_imgs: int, n_ref_q: int) -> dict:
    """No held-out list is saved by the pipeline (U_independent_images lives only in the in-memory scores dict).  What IS saved: the counts of
    both batches in every interval method's lock, and for the scalar critics the U scores (CSV) and (a, b), so c = mean T over each batch can be
    recomputed with the reproduced split and compared with the saved c."""
    out = {}
    for m in INTERVAL_METHODS:
        lp = cell / "weights" / f"{m}.csv.lock.json"
        if not lp.exists():
            out[m] = {"status": "lock not found"}; continue
        lock = json.loads(lp.read_text()); rho = lock["record"].get("rho", {}); ib = rho.get("independent_batch") or {}
        nce, nci = rho.get("n_clusters") or {}, ib.get("n_clusters") or {}
        r = {"saved_est_n_U": rho.get("n_U"), "saved_ind_n_pairs": ib.get("n_pairs"), "saved_ind_n_images": ib.get("n_images"),
             "saved_n_clusters_est": nce, "saved_n_clusters_ind": nci}
        r["counts_match"] = bool(rho.get("n_U") == int(est_mask.sum()) and ib.get("n_pairs") == int((~est_mask).sum()) and ib.get("n_images") == n_held
                                 and nci.get("R") == n_held and nce.get("P") == n_ref_imgs == nci.get("P") and nce.get("Q") == n_ref_q == nci.get("Q"))
        if m in SCALAR_CRITICS:
            cp = cell / "weights" / f"{m}.csv"; r["csv_sha256_matches_lock"] = _sha_file(cp) == lock["sha256"]
            ids, s = _read_csv_scores(cp); r["csv_order_is_U_order"] = ids == list(u_ids)
            a, b = float(lock["record"]["critic"]["a"]), float(lock["record"]["critic"]["b"]); T = np.tanh(a * s + b)
            c_est, c_ind = float(T[est_mask].mean()), float(T[~est_mask].mean()); gap = rho["a"] - rho["b"]
            r.update({"c_est_recomputed": c_est, "c_est_saved": rho["c"], "abs_diff_c_est": abs(c_est - rho["c"]),
                      "c_ind_recomputed": c_ind, "c_ind_saved": ib["c"], "abs_diff_c_ind": abs(c_ind - ib["c"]),
                      "raw_ind_recomputed": (rho["a"] - c_ind) / gap, "raw_ind_saved": ib["raw"], "abs_diff_raw_ind": abs((rho["a"] - c_ind) / gap - ib["raw"])})
            r["c_match_1e-7"] = bool(r["abs_diff_c_est"] <= 1e-7 and r["abs_diff_c_ind"] <= 1e-7)
        out[m] = r
    return out


def pipeline_crosscheck(src_dir, observed: list, roles: dict, held: set, est_mask: np.ndarray, q: dict) -> dict:
    """Run the pipeline's own compute_scores with the teacher stubbed out; its held-out set, estimation mask and Q rows must equal ours."""
    try:
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        PL = importlib.import_module("v9.n2.pipeline")
    except Exception as e:                                                # torch or the source tree unavailable
        return {"status": f"skipped: {type(e).__name__}: {e}"}
    captured = []; orig = (PL.load_teacher, PL.score_pairs)

    def _score(host, rows, collate, batch=128, device="cpu"):
        captured.append([(r["pair_id"], r.get("q_of"), r["observed_image_id"]) for r in rows]); return np.zeros(len(rows))
    PL.load_teacher = lambda path, device="cpu": (None, None); PL.score_pairs = _score
    try:
        s = PL.compute_scores({"teacher": None, "observed": observed, "roles": roles, "collate": None}, "cpu", 128)
    finally:
        PL.load_teacher, PL.score_pairs = orig
    mine = {role: [(r["pair_id"], r["q_of"], r["observed_image_id"]) for r in q[role]] for role in Q_ROLES}
    q_eq = {role: captured[2 * k + 1] == mine[role] for k, role in enumerate(Q_ROLES)}     # captured: fit P, fit Q, sel P, sel Q, ref P, ref Q, U
    out = {"status": "ran", "module": str(Path(PL.__file__).resolve()), "heldout_equal": set(s["U_independent_images"]) == set(held),
           "est_mask_equal": bool(np.array_equal(np.asarray(s["U_est_mask"], dtype=bool), est_mask)), "q_rows_equal": q_eq}
    out["all_equal"] = bool(out["heldout_equal"] and out["est_mask_equal"] and all(q_eq.values()))
    return out


# ------------------------------------------------------------------------------------------------ one cell
def audit_cell(out_root: Path, dataset: str, rho: float, seed: int, pipeline_src=None) -> dict:
    t0 = time.time(); cell = out_root / dataset / f"rho{rho:g}" / f"s{seed}"
    fp = {k: cell / k for k in ("roles.json", "pairs.observed_pairs.json", "pairs.noise_manifest.json")}
    roles = json.loads(fp["roles.json"].read_text()); ob_bytes = fp["pairs.observed_pairs.json"].read_bytes(); observed = json.loads(ob_bytes)
    man = json.loads(fp["pairs.noise_manifest.json"].read_text()); rseed = int(roles["seed"])
    role_of = {i: r for r in ANCHOR_ROLES for i in roles["A"][r]}; role_of.update({i: "U" for i in roles["U"]})
    src = {o["pair_id"]: caption_source(o["caption_id"]) for o in observed}
    U = [o for o in observed if o["role"] == "U"]; u_ids = [o["pair_id"] for o in U]; obs_count = Counter(o["observed_image_id"] for o in U)
    held = heldout_images([o["observed_image_id"] for o in U], rseed)
    est_mask = np.array([o["observed_image_id"] not in held for o in U], dtype=bool)
    q = q_rows_by_role(observed, rseed)
    n_mis = sum(o["observed_image_id"] != src[o["pair_id"]] for o in observed)
    checks = {
        "observed_sha256_matches_manifest": hashlib.sha256(ob_bytes).hexdigest() == man.get("observed_sha256"),
        "mapping_sha256_matches_manifest": hashlib.sha256("\n".join(f"{o['pair_id']}\t{o['observed_image_id']}" for o in observed).encode()).hexdigest() == man.get("mapping_sha256"),
        "roles_hashes_recomputed": all(_sha_ids(roles["A"][r]) == roles["hashes"][r] for r in ANCHOR_ROLES) and _sha_ids(roles["U"]) == roles["hashes"]["U"],
        "roles_seed_equals_cell_seed": rseed == int(seed),
        "row_role_equals_role_of_caption_source": all(o["role"] == role_of.get(src[o["pair_id"]]) for o in observed),
        "anchor_rows_clean": all(o["observed_image_id"] == src[o["pair_id"]] for o in observed if o["role"] != "U"),
        "U_observed_images_are_U_images": all(role_of.get(o["observed_image_id"]) == "U" for o in U),
        "n_U_pairs_matches_manifest": len(U) == man.get("n_U_pairs"),
        "n_mismatched_matches_manifest": n_mis == man.get("n_mismatched"),
        "images_with_any_mismatch_matches_manifest": len({src[o["pair_id"]] for o in U if o["observed_image_id"] != src[o["pair_id"]]}) == man.get("images_with_any_mismatch"),
        "every_U_image_observed_exactly_5_times": len(obs_count) == roles["n_U"] and set(obs_count.values()) == {CAPTIONS_PER_IMAGE},
        "heldout_size_is_round_20pct": len(held) == int(round(0.2 * len(obs_count))),
    }
    pc = cell / "precheck" / f"precheck_{dataset}_rho{rho:g}_s{seed}.json"
    if pc.exists():
        pcj = json.loads(pc.read_text())
        checks["precheck_roles_hashes_equal"] = pcj["roles"]["hashes"] == roles["hashes"]; checks["precheck_noise_manifest_equal"] = pcj["noise_manifest"] == man
    saved_lists = [str(p) for p in sorted(cell.rglob("*.json")) if p.stat().st_size < 50_000_000 and "eval_only" not in p.parts
                   and any(k in p.read_text() for k in ("U_independent_images", "U_est_mask"))]
    u_pairs = [(o["pair_id"], o["observed_image_id"], src[o["pair_id"]]) for o in U]
    q_links = [(role, caption_source(r["caption_id"]), r["observed_image_id"]) for role in Q_ROLES for r in q[role]]
    counts = count_overlap(u_pairs, held, q_links, {i: r for r in ANCHOR_ROLES for i in roles["A"][r]})
    ref_P = [o for o in observed if o["role"] == "reference_moments"]; ref_Q = q["reference_moments"]
    shared_ref = {"P_rows": len(ref_P), "P_clusters_images": len({o["observed_image_id"] for o in ref_P}), "Q_rows": len(ref_Q),
                  "Q_clusters_q_of": len({r["q_of"] for r in ref_Q}), "Q_distinct_donor_images": len({r["observed_image_id"] for r in ref_Q}),
                  "used_by": "both estimate_rho calls (weights.py:174 estimation batch, :177 independent batch)"}
    saved = saved_record_check(cell, u_ids, est_mask, len(held), shared_ref["P_clusters_images"], shared_ref["Q_clusters_q_of"])
    xc = pipeline_crosscheck(pipeline_src, observed, roles, held, est_mask, q) if pipeline_src else {"status": "not requested"}
    held_sorted = sorted(held, key=_id_key)
    rec = {"cell": f"rho{rho:g}/s{seed}", "dir": str(cell), "dataset": dataset, "rho": rho, "seed": seed, "roles_seed": rseed,
           "inputs_sha256": {k: (hashlib.sha256(ob_bytes).hexdigest() if k == "pairs.observed_pairs.json" else _sha_file(p)) for k, p in fp.items()},
           "noise_manifest": {k: man.get(k) for k in ("design_rho", "n_pairs", "n_U_pairs", "n_selected", "n_mismatched", "realized_rho", "images_with_any_mismatch", "mapping_sha256")},
           "roles_counts": {"n_images": roles["n_images"], "n_U_images": roles["n_U"], **roles["counts"]},
           "checks": checks, "all_input_checks_pass": all(checks.values()),
           "heldout": {"n_images": len(held), "n_pairs": int((~est_mask).sum()), "sha256_sorted_ids": _sha_ids(held_sorted),
                       "saved_list_files": saved_lists, "saved_list_note": "no file in the cell carries the held-out list (pipeline keeps U_independent_images in memory only)" if not saved_lists else "see files"},
           "saved_record_check": saved, "pipeline_crosscheck": xc, "reference_shared_by_both_calls": shared_ref, **counts,
           "seconds": round(time.time() - t0, 2)}
    return rec


# ------------------------------------------------------------------------------------------------ report
def _pct(a, b):
    return f"{100.0 * a / b:.1f} %" if b else "n/a"


def _span(xs, fmt="{}"):
    lo, hi = min(xs), max(xs)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


def write_md(res: dict, path: Path) -> None:
    cells = res["cells"]; meta = res["meta"]
    L = [f"# V9-N2 overlap audit of the \"independent batch\" rho interval (review item U07) — {meta['dataset']} precheck cells", "",
         f"Generated {meta['generated_utc']} by `scripts/v9/n2_overlap_audit.py` (python {meta['python']}, numpy {meta['numpy']}); numbers below are copied "
         f"from `overlap_audit.json` in this folder.  Read-only inputs per cell: `{meta['out_root']}/{meta['dataset']}/rho<rho>/s<seed>/` `roles.json`, "
         "`pairs.observed_pairs.json`, `pairs.noise_manifest.json`; checks also read `weights/*.csv.lock.json`, the VCS / BCE weight CSVs and the precheck JSON.  "
         "The eval-only label file was not opened.  Facts and counts only.", "",
         "Definitions (code references as in `meta.code_facts`; the sha256 of each cited file is in `meta.code_sha256`):", "",
         f"- **H** = held-out \"independent batch\", **E** = estimation batch: {meta['code_facts']['groups']}.  A U pair belongs to the group of its observed image.",
         f"- Caption-source image of caption c: c // 5 ({meta['code_facts']['caption_source']}).  `X<-Y` counts pairs observed with an image of group X whose caption comes from an image of group Y.",
         "- An image \"appears in both groups\" when it is the observed or the caption-source image of at least one H pair and of at least one E pair.",
         f"- Q rows: {meta['code_facts']['q_rows']}.",
         f"- Image graph: one node per train image ({next(iter(cells.values()))['roles_counts']['n_images']} images, all roles); an edge joins the observed and the caption-source image of every mismatched U pair, and the "
         "caption-source and donor image of every Q row of the three roles (the pair <-> image incidence graph read on its image nodes).",
         f"- Both rho calls: {meta['code_facts']['two_calls']}.", "",
         "## 1. Reproduction and input checks", "",
         "No cell stores the held-out list (`U_independent_images` exists only in the in-memory scores dict).  The split was reproduced with the exact rule and checked "
         "(a) against the pipeline's own `compute_scores` run with the teacher stubbed (held-out set, estimation mask, Q rows of the three roles), (b) against the "
         "counts every interval lock saved, and (c) by recomputing c = mean T over each batch from the locked VCS / BCE CSV scores with the saved critic (a, b); "
         "(c) ties the reproduced split to what the precheck runs used, whatever the code version.", "",
         "| cell | input checks | held-out = pipeline | est. mask = pipeline | Q rows = pipeline | saved counts (4 locks) | max abs diff of c (est / ind), VCS + BCE | held-out ids sha256 |",
         "|---|---|---|---|---|---|---|---|"]
    for k, c in cells.items():
        xc = c["pipeline_crosscheck"]; sv = c["saved_record_check"]
        cnt_ok = all(v.get("counts_match") for v in sv.values() if "counts_match" in v); n_locks = sum(1 for v in sv.values() if "counts_match" in v)
        dce = max([sv[m]["abs_diff_c_est"] for m in SCALAR_CRITICS if "abs_diff_c_est" in sv.get(m, {})], default=None)
        dci = max([sv[m]["abs_diff_c_ind"] for m in SCALAR_CRITICS if "abs_diff_c_ind" in sv.get(m, {})], default=None)
        dc = f"{dce:.2e} / {dci:.2e}" if dce is not None else "no locks"
        L.append(f"| {k} | {sum(c['checks'].values())}/{len(c['checks'])} pass | {xc.get('heldout_equal', xc['status'])} | {xc.get('est_mask_equal', '-')} | "
                 f"{all(xc['q_rows_equal'].values()) if 'q_rows_equal' in xc else '-'} | {('match' if cnt_ok else 'MISMATCH') if n_locks else 'no locks'} ({n_locks}) | {dc} | {c['heldout']['sha256_sorted_ids'][:12]} |")
    L += ["", "Held-out sets per seed are identical at rho 0.4 and rho 0: " + ", ".join(f"s{s}: {v}" for s, v in res["cross_cell"]["heldout_identical_across_rho"].items()) + ".", "",
          "## 2. Groups and caption-source links (U pairs)", "",
          "| cell | E pairs / images | H pairs / images | mismatched pairs E / H | H<-E | E<-H | H<-H | E<-E | cross-group share of mismatched | images in both groups (H images with a caption in E + E images with a caption in H) | clusters holding an other-group caption (H / E) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in cells.items():
        g, l, b = c["groups"], c["caption_source_links"], c["images_in_both_groups"]
        L.append(f"| {k} | {g['E']['n_pairs']} / {g['E']['n_images']} | {g['H']['n_pairs']} / {g['H']['n_images']} | {g['E']['n_mismatched_pairs']} / {g['H']['n_mismatched_pairs']} | "
                 f"{l['H<-E']} | {l['E<-H']} | {l['H<-H']} | {l['E<-E']} | {l['cross_group_total']} / {l['n_mismatched_total']} = {_pct(l['cross_group_total'], l['n_mismatched_total'])} | "
                 f"{b['total']} ({b['H_images_with_a_caption_in_E']} + {b['E_images_with_a_caption_in_H']}) | "
                 f"{b['H_clusters_holding_an_E_caption']} ({_pct(b['H_clusters_holding_an_E_caption'], g['H']['n_images'])}) / {b['E_clusters_holding_an_H_caption']} ({_pct(b['E_clusters_holding_an_H_caption'], g['E']['n_images'])}) |")
    neq = [k for k, c in cells.items() if c["caption_source_links"]["H<-E"] != c["caption_source_links"]["E<-H"]]
    L += ["", ("H<-E equals E<-H in every cell: `inject_mismatch` gives the selected pairs a permutation of their own images (noise.py:46-47), so the selected "
               "pairs observed with an H image are as many as the selected pairs whose caption comes from an H image." if not neq else f"H<-E differs from E<-H in: {neq}."), "",
          "## 3. Image graph (caption-source links + Q donor links)", "",
          "| cell | link edges | components | largest component: images (H / E / anchor), U pairs | largest component with U images: images (H / E), U pairs | components spanning H and E | images in spanning components H / E (share of H) | U pairs in spanning components H / E | H images sharing a component with another H image |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, c in cells.items():
        gr = c["graph"]; lc = gr["largest_component"]; lu = gr["largest_component_with_U_images"]; sp = gr["spanning_components"]; nh = c["groups"]["H"]["n_images"]
        L.append(f"| {k} | {gr['n_link_edges']} | {gr['n_components']} | {lc['n_images']} ({lc['n_H_images']} / {lc['n_E_images']} / {lc['n_anchor_images']}), {lc['n_U_pairs_H'] + lc['n_U_pairs_E']} | "
                 f"{lu['n_images']} ({lu['n_H_images']} / {lu['n_E_images']}), {lu['n_U_pairs_H'] + lu['n_U_pairs_E']} | {gr['n_components_spanning_both_groups']} | "
                 f"{sp['n_H_images']} / {sp['n_E_images']} ({_pct(sp['n_H_images'], nh)}) | {sp['n_U_pairs_H']} / {sp['n_U_pairs_E']} | {gr['H_images_in_a_component_with_another_H_image']} |")
    L += ["", "## 4. Q donors and the reference rows shared by both rho calls", "",
          "| cell | Q rows fit / select / reference | donors in the same role | donors in another anchor role / in U | Q links crossing H and E | self donors (fit / select / reference) | reference rows used by both calls: P rows / P clusters (images) / Q rows (clusters) / distinct Q donor images |",
          "|---|---|---|---|---|---|---|"]
    for k, c in cells.items():
        qd = c["q_donors"]; sr = c["reference_shared_by_both_calls"]
        L.append(f"| {k} | {' / '.join(str(qd[r]['n_rows']) for r in Q_ROLES)} | {sum(qd[r]['donor_in_same_role'] for r in Q_ROLES)} | "
                 f"{sum(qd[r]['donor_in_other_anchor_role'] for r in Q_ROLES)} / {sum(qd[r]['donor_in_U'] for r in Q_ROLES)} | {sum(qd[r]['crossing_H_E'] for r in Q_ROLES)} | "
                 f"{' / '.join(str(qd[r]['self_donor']) for r in Q_ROLES)} | {sr['P_rows']} / {sr['P_clusters_images']} / {sr['Q_rows']} ({sr['Q_clusters_q_of']}) / {sr['Q_distinct_donor_images']} |")
    L += ["", "## 5. The package's one-unit-one-cluster contract with image ids as units", "",
          f"{meta['code_facts']['pipeline_disjointness_units']}.  With image ids as units (P cluster = its reference image; Q cluster = its caption-source and donor image; "
          "R cluster = its observed image plus the caption-source images of its pairs):", "",
          "| cell | call | clusters P / Q / R | units in more than one cluster | clusters sharing a unit P / Q / R | shared between the calls: identical P / Q clusters, images in R units of both |",
          "|---|---|---|---|---|---|"]
    for k, c in cells.items():
        pv = c["provenance_units"]; sh = pv["shared_between_calls"]
        for g, name in (("H", "independent (H)"), ("E", "estimation (E)")):
            u = pv[f"call_{g}"]; n = u["n_clusters"]; s = u["clusters_sharing_a_unit"]
            L.append(f"| {k} | {name} | {n.get('P', 0)} / {n.get('Q', 0)} / {n.get('R', 0)} | {u['units_in_more_than_one_cluster']} | {s.get('P', 0)} / {s.get('Q', 0)} / {s.get('R', 0)} | "
                     f"{sh['identical_P_clusters']} / {sh['identical_Q_clusters']}, {sh['R_units_in_both_calls']} |")
    hi = [c for c in cells.values() if c["rho"] > 0]; lo = [c for c in cells.values() if c["rho"] == 0]
    if hi:
        xs = [c["caption_source_links"]["cross_group_total"] for c in hi]; sh_ = [c["caption_source_links"]["cross_share_of_mismatched"] for c in hi]
        spn = [c["graph"]["n_components_spanning_both_groups"] for c in hi]; hs = [c["graph"]["spanning_components"]["n_H_images"] / c["groups"]["H"]["n_images"] for c in hi]
        hc = [c["images_in_both_groups"]["H_clusters_holding_an_E_caption"] for c in hi]; nh = [c["groups"]["H"]["n_images"] for c in hi]
        rhos_hi = sorted({c["rho"] for c in hi}); one = min(spn) == max(spn) == 1
        line = (f"Implication: at rho {', '.join(f'{r:g}' for r in rhos_hi)} the held-out batch is separated from the estimation batch by receiving-image id only — "
                f"{_span(xs)} mismatched pairs ({_span([100 * v for v in sh_], '{:.1f}')} % of the mismatched) link the two groups, {_span(hc)} of the {_span(nh)} held-out "
                f"clusters hold a caption from an estimation image, {_span(spn)} graph component{'' if one else 's'} span{'s' if one else ''} both groups and "
                f"hold{'s' if one else ''} {_span([100 * v for v in hs], '{:.1f}')} % of the held-out images, "
                "and both rho calls use the same reference P / Q rows — so these counts do not support calling the interval \"independent\" of the point estimate")
        if lo:
            n0 = max(c["caption_source_links"]["cross_group_total"] for c in lo)
            line += ("; at rho 0 " + ("no caption-source link crosses the split (0 in every cell)" if n0 == 0 else f"up to {n0} caption-source links cross the split")
                     + ", and the two calls still use the same reference P / Q rows and the same fitted critic.")
        else:
            line += "."
        L += ["", line]
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", required=True); ap.add_argument("--dataset", default="f30k")
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.4, 0.0]); ap.add_argument("--seeds", nargs="+", type=int, default=[901, 902, 903, 917])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pipeline-src", default=str(ROOT / "src") if (ROOT / "src" / "v9" / "n2" / "pipeline.py").exists() else None)
    ap.add_argument("--no-pipeline-crosscheck", action="store_true"); ap.add_argument("--code-root", default=str(ROOT))
    a = ap.parse_args(argv)
    src = None if a.no_pipeline_crosscheck or not a.pipeline_src else Path(a.pipeline_src)
    out_root = Path(a.out_root); cells = {}
    for rho in a.rhos:
        for seed in a.seeds:
            rec = audit_cell(out_root, a.dataset, rho, seed, src); cells[rec["cell"]] = rec
            print(f"[audit] {rec['cell']}: checks {sum(rec['checks'].values())}/{len(rec['checks'])}, crosscheck {rec['pipeline_crosscheck'].get('all_equal', rec['pipeline_crosscheck']['status'])}, "
                  f"cross links {rec['caption_source_links']['cross_group_total']}, spanning comps {rec['graph']['n_components_spanning_both_groups']}, {rec['seconds']} s", flush=True)
    cross = {"heldout_identical_across_rho": {}}
    for seed in a.seeds:
        hs = {cells[f"rho{r:g}/s{seed}"]["heldout"]["sha256_sorted_ids"] for r in a.rhos if f"rho{r:g}/s{seed}" in cells}
        cross["heldout_identical_across_rho"][str(seed)] = len(hs) == 1
    code_root = Path(a.code_root)
    meta = {"script": "scripts/v9/n2_overlap_audit.py", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
            "numpy": np.__version__, "out_root": str(out_root), "dataset": a.dataset, "rhos": a.rhos, "seeds": a.seeds,
            "pipeline_src": str(src) if src else None, "code_root": str(code_root),
            "code_sha256": {f: (_sha_file(code_root / f) if (code_root / f).exists() else None) for f in CODE_FILES},
            "code_facts": CODE_FACTS, "eval_only_opened": False, "review_item": "U07"}
    if src:
        meta["pipeline_src_sha256"] = {f: (_sha_file(src.parent / f) if (src.parent / f).exists() else None) for f in CODE_FILES}
    res = {"meta": meta, "cross_cell": cross, "cells": cells}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "overlap_audit.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    write_md(res, out / "OVERLAP_AUDIT.md")
    ok = all(c["all_input_checks_pass"] for c in cells.values()) and all(c["pipeline_crosscheck"].get("all_equal", True) for c in cells.values())
    print(f"[audit] wrote {out / 'overlap_audit.json'} and {out / 'OVERLAP_AUDIT.md'}; all checks pass: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
