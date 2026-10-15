"""P159 (SEP-3): P157's scoring-rule comparison in parametric critic classes under mixture contamination, on FROZEN natural features with no
truth — truth-free readouts on a clean, image-disjoint EVAL set.  Protocol: reports/P159_SEP3_NATURAL_PREREG_*_20261015.md.

Sources (existing caches only; no extraction, no official test split):
  coco_clip   frozen CLIP ViT-B/32 (laion2b) image / caption features of COCO train2017 (outputs/P49_precheck_A; roles SRC-FIT 20 000 images,
              SRC-CAL 5 000, SRC-EVAL 5 000, disjoint by image; 5 captions per image).  P pair = (img_i, txt_i[c_i]) with one caption per image
              drawn per seed; Q pair = (img_i, txt_{pi(i)}[c_{pi(i)}]) with pi one non-zero cyclic shift inside the role (named: cyclic1).
  cifar_ap3   frozen A-P3 encoder (P107 seed 1, 4v/800ep) h features of 20 000 base images of the CIFAR-10 FIT split in seven versions
              (outputs/P150_features/P107_AP3_views4_800ep_seed1; fp16 h, 512-d).  P pair = (h_clean(i), h_blur1(i)) of the same base image;
              Q pair = (h_clean(i), h_blur1(pi(i))), cyclic1.  View blur_s1 (1 px Gaussian blur) chosen after a descriptive cosine check on 4 000
              base images (no fit): blur_s0.5 / colour versions are nearly identical to clean (positive cos >= 0.985, perfectly separable from the
              product pairs: the gate's timing cell gave J 0.9995 for both arms), blur_s1 gives overlapping pair laws (positive cos median 0.71,
              q10 0.46; product cos median 0.17, q90 0.31).  Disclosed as a post-gate, pre-formal-run design change in the prereg.
Roles per seed (draw): FIT (N pairs), SELECT (N / 4, capped by the pool), EVAL (4 096, clean, disjoint by base image / COCO image).
Contamination of FIT and SELECT only (the practitioner's data; EVAL is never contaminated), exactly round(eps n) pairs per role, indices recorded:
  independent     P_eps = (1 - eps) P + eps Q:  the y of eps n P pairs replaced by the y of another base unit of the same role (a product pair as a positive)
  false_negative  Q_eps = (1 - eps) Q + eps P:  eps n Q pairs replaced by true pairs of another base unit of the same role (COCO: a caption not used
                  by that unit's P pair, i.e. a fresh true pair; CIFAR: the unit's only pair, i.e. a duplicate positive among the negatives, as in SSL).
Critics / rules / trainer: sep.make_critic (cos2, quad3 on the raw 512-d features; d = 512), sep.train_arm (arms js / vcs, P157's lr grid and
budget, native-risk selection primary, common-J selection sensitivity), unchanged.  sep.py and p157.py are imported, never edited.
Readouts on clean EVAL (both arms, T = arm_T(f)): common squared score J, the arm's native risk, AUROC of f for P-vs-Q (rank based), ECE of
(1 + T) / 2 against the pair label (15 equal-mass bins), gate mean and |T| > 0.95 mass.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from . import sep
from .benchmark import Role, _dev, _blocks
from .run import git_commit
from .synthetic import _gen

PROTOCOL = "P159"
OUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
SRC = {
    "coco_clip": {"dir": OUT_ROOT / "P49_precheck_A", "files": {"FIT": "features/SRC-FIT.pt", "SELECT": "features/SRC-CAL.pt", "EVAL": "features/SRC-EVAL.pt"},
                  "manifests": ["manifest.json", "splits.json"], "d": 512, "NS": (4096, 16384)},
    "cifar_ap3": {"dir": OUT_ROOT / "P150_features/P107_AP3_views4_800ep_seed1", "files": {"ALL": "features.pt"}, "manifests": ["manifest.json"],
                  "d": 512, "NS": (4096, 8192), "view_x": "clean", "view_y": "blur_s1", "layer": "h"},
}
SOURCES = tuple(SRC)
CLASSES, CTYPES, EPS = ("cos2", "quad3"), ("independent", "false_negative"), (0.1, 0.3)
SEEDS = tuple(range(300, 308))
LRS = sep.LRS_PAR + (3e-1, 1.0)                       # = P157's grid (1e-3 ... 1.0)
UPDATES, B = 2000, sep.B_DEFAULT
EVAL_N = 4096
_CACHE: dict = {}


# ------------------------------------------------------------------------------------------------------------ data
def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_source(name: str) -> dict:
    """Loads (and caches) the feature tensors of a source as float64 CPU tensors plus the provenance record (file sha256, manifest sha256)."""
    if name in _CACHE:
        return _CACHE[name]
    s = SRC[name]; prov = {"dir": str(s["dir"]), "files": {}, "manifests": {}}
    for m in s["manifests"]:
        prov["manifests"][m] = sha256_file(s["dir"] / m)
    data = {}
    if name == "coco_clip":
        for role, rel in s["files"].items():
            p = s["dir"] / rel; d = torch.load(p, map_location="cpu", weights_only=False)
            data[role] = {"x": d["img"].double(), "y_all": d["txt"].double(), "ids": d["image_ids"].clone()}      # y_all [n, 5, 512]
            prov["files"][role] = {"path": str(p), "sha256": sha256_file(p), "n": int(len(d["img"]))}
    elif name == "cifar_ap3":
        p = s["dir"] / s["files"]["ALL"]; d = torch.load(p, map_location="cpu", weights_only=False); F = d["features"]
        data["ALL"] = {"x": F[s["view_x"]][s["layer"]].double(), "y": F[s["view_y"]][s["layer"]].double(), "ids": d["ids"].clone(), "part": d["part"].clone()}
        prov["files"]["ALL"] = {"path": str(p), "sha256": sha256_file(p), "n": int(len(d["ids"])), "run": d["run"], "views": [s["view_x"], s["view_y"]], "layer": s["layer"]}
    else:
        raise ValueError(name)
    _CACHE[name] = {"data": data, "prov": prov}
    return _CACHE[name]


def cyclic1_shift(n: int, gen: torch.Generator) -> int:
    return int(torch.randint(1, n, (1,), generator=gen))


def _pairs_coco(block: dict, idx: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """P pairs (x_i, y_i) with one caption per image (drawn per seed); returns x, y, the caption index per unit and the cyclic shift."""
    cap = torch.randint(0, block["y_all"].shape[1], (len(idx),), generator=gen)
    x = block["x"][idx]; y = block["y_all"][idx, cap]; s = cyclic1_shift(len(idx), gen)
    return x, y, cap, s


def draw_roles(name: str, N: int, seed: int, eval_n: int = EVAL_N) -> tuple[dict, dict]:
    """Roles of one draw: FIT (N), SELECT (N / 4), EVAL (eval_n), disjoint by base unit; Q = cyclic1 inside each role.  Returns (roles, record)."""
    S = load_source(name); D = S["data"]; sizes = {"FIT": N, "SELECT": max(64, N // 4), "EVAL": eval_n}; roles, rec = {}, {"sizes": sizes, "shift": {}}
    if name == "coco_clip":
        for role in ("FIT", "SELECT", "EVAL"):
            blk = D[role]; g = _gen(("P159", name, role, N, seed)); n_pool = len(blk["x"])
            if sizes[role] > n_pool:
                raise ValueError(f"{name} {role}: {sizes[role]} > pool {n_pool}")
            idx = torch.randperm(n_pool, generator=g)[:sizes[role]]
            x, y, cap, s = _pairs_coco(blk, idx, g); yq = torch.roll(y, s, 0)
            roles[role] = Role(x, y, x.clone(), yq, base=None, pad_p=None); rec["shift"][role] = s
            rec[role] = {"idx_sha256": hashlib.sha256(repr(idx.tolist()).encode()).hexdigest(), "cap_sha256": hashlib.sha256(repr(cap.tolist()).encode()).hexdigest(),
                         "ids_head": blk["ids"][idx][:8].tolist(), "pool": n_pool, "caption_per_unit": cap}
            rec[role]["_idx"] = idx                                                    # kept for contamination (not serialised)
    elif name == "cifar_ap3":
        blk = D["ALL"]; n_pool = len(blk["x"]); g = _gen(("P159", name, "perm", N, seed)); perm = torch.randperm(n_pool, generator=g)
        need = sizes["FIT"] + sizes["SELECT"] + sizes["EVAL"]
        if need > n_pool:
            raise ValueError(f"{name}: FIT + SELECT + EVAL = {need} > pool {n_pool}")
        off = 0
        for role in ("FIT", "SELECT", "EVAL"):
            idx = perm[off:off + sizes[role]]; off += sizes[role]; gr = _gen(("P159", name, role, N, seed))
            x, y = blk["x"][idx], blk["y"][idx]; s = cyclic1_shift(len(idx), gr); yq = torch.roll(y, s, 0)
            roles[role] = Role(x, y, x.clone(), yq, base=None, pad_p=None); rec["shift"][role] = s
            rec[role] = {"idx_sha256": hashlib.sha256(repr(idx.tolist()).encode()).hexdigest(), "ids_head": blk["ids"][idx][:8].tolist(), "pool": n_pool,
                         "parts": torch.bincount(blk["part"][idx], minlength=3).tolist(), "_idx": idx}
    else:
        raise ValueError(name)
    if name == "cifar_ap3":                                                                    # one shared pool: positional disjointness
        ev = set(rec["EVAL"]["_idx"].tolist())
        for role in ("FIT", "SELECT"):
            assert not (set(rec[role]["_idx"].tolist()) & ev), "EVAL overlaps a fitting role"
    if name == "coco_clip":                                                                     # three P49 pools: disjoint by image id (asserted on ids)
        ids = {r: set(D[r]["ids"][rec[r]["_idx"]].tolist()) for r in ("FIT", "SELECT", "EVAL")}
        assert not (ids["EVAL"] & ids["FIT"]) and not (ids["EVAL"] & ids["SELECT"]), "EVAL image overlap"
    return roles, rec


# ------------------------------------------------------------------------------------------------------------ contamination (natural data)
def contaminate(name: str, roles: dict, rec: dict, ctype: str, eps: float, seed: int, skip=("EVAL",)) -> tuple[dict, dict]:
    """Mirror of sep.contaminate_fn_roles / r1.contaminate_roles for natural pairs (see the module docstring).  Exactly round(eps n) units per
    role; EVAL untouched; indices recorded with a SHA-256 per role."""
    out, recs = {}, {}
    S = load_source(name); D = S["data"]
    for role, r in roles.items():
        if role in skip or eps == 0.0 or ctype == "clean":
            out[role] = r; continue
        n = r.n; k = int(round(eps * n)); gi = _gen(("P159-CONT-IDX", name, ctype, eps, role, seed)); gp = _gen(("P159-CONT-PARTNER", name, ctype, eps, role, seed))
        sel = torch.randperm(n, generator=gi)[:k]
        partner = (sel + torch.randint(1, n, (k,), generator=gp)) % n                           # another unit of the same role (never itself)
        if ctype == "independent":                                                             # product pairs injected as positives
            yp = r.yp.clone(); yp[sel] = r.yp[partner]; out[role] = Role(r.xp, yp, r.xq, r.yq, r.base, r.pad_p)
        elif ctype == "false_negative":                                                        # true pairs injected among the negatives
            xq, yq = r.xq.clone(), r.yq.clone(); xq[sel] = r.xp[partner]
            if name == "coco_clip":                                                            # a caption the partner's P pair does not use: fresh true pair
                blk = D[role]; idx = rec[role]["_idx"][partner]; cap0 = rec[role]["caption_per_unit"][partner]; m = blk["y_all"].shape[1]
                cap = (cap0 + torch.randint(1, m, (k,), generator=gp)) % m; yq[sel] = blk["y_all"][idx, cap]
            else:                                                                              # CIFAR: the partner's only pair (a duplicate positive)
                yq[sel] = r.yp[partner]
            out[role] = Role(r.xp, r.yp, xq, yq, r.base, r.pad_p)
        else:
            raise ValueError(ctype)
        s_idx = sorted(sel.tolist())
        recs[role] = {"n": n, "k": k, "idx_sha256": hashlib.sha256(repr(s_idx).encode()).hexdigest(), "idx_head": s_idx[:16],
                      "partner_sha256": hashlib.sha256(repr(partner.tolist()).encode()).hexdigest()}
    return out, recs


# ------------------------------------------------------------------------------------------------------------ readouts (truth-free)
def auroc(score: torch.Tensor, y: torch.Tensor) -> float:
    """Rank AUROC (ties by average rank) of score for y in {0, 1}."""
    s = score.double(); o = torch.argsort(s); r = torch.empty(len(s), dtype=torch.float64); r[o] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    ss = s[o]; same = torch.ones(len(s), dtype=torch.bool); same[1:] = ss[1:] == ss[:-1]
    if same[1:].any():                                                                        # average ranks within ties
        grp = torch.cumsum(~same, 0); sums = torch.zeros(int(grp.max()) + 1, dtype=torch.float64).scatter_add_(0, grp, r[o]); cnt = torch.bincount(grp).double()
        r[o] = (sums / cnt)[grp]
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1.0))


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins (the precheck-A helper, probabilities p in [0, 1], labels y in {0, 1})."""
    o = torch.argsort(p); p, y = p[o].double(), y[o].double(); edges = torch.linspace(0, len(p), bins + 1).long(); e = 0.0
    for i in range(bins):
        a, b = int(edges[i]), int(edges[i + 1])
        if b > a:
            e += (b - a) / len(p) * abs(float(p[a:b].mean()) - float(y[a:b].mean()))
    return float(e)


def readouts(critic, arm: str, E: Role, device) -> dict:
    fp = sep._forward_role(critic, E.xp, E.yp, device); fq = sep._forward_role(critic, E.xq, E.yq, device)
    tp, tq = sep.arm_T(arm, fp), sep.arm_T(arm, fq); f = torch.cat([fp, fq]); t = torch.cat([tp, tq]); y = torch.cat([torch.ones(len(fp)), torch.zeros(len(fq))])
    risk = float(sep.arm_loss(arm, fp, fq))
    return {"J_eval": sep.common_J(tp, tq), "native_risk": risk if math.isfinite(risk) else None, "auroc": auroc(f, y), "ece": ece((1 + t) / 2, y),
            "gate_mean": float(0.5 * (1 - tp ** 2).mean() + 0.5 * (1 - tq ** 2).mean()),
            "abs_T_gt_0.95": float(0.5 * (tp.abs() > 0.95).double().mean() + 0.5 * (tq.abs() > 0.95).double().mean()),
            "T_pos_mean": float(tp.mean()), "T_neg_mean": float(tq.mean()), "n_eval": int(E.n)}


# ------------------------------------------------------------------------------------------------------------ cells
def cell(source: str, cls: str, ctype: str, eps: float, N: int, seed: int) -> dict:
    return {"block": "S3", "source": source, "cls": cls, "variant": "product", "ctype": ctype, "eps": float(eps), "d": SRC[source]["d"], "N": int(N),
            "updates": UPDATES, "B": B, "arms": list(sep.ARMS_PAIR), "lrs": list(LRS), "seed": int(seed), "tag": "p159"}


def cell_name(c: dict) -> str:
    return f"P159_{c['block']}_{c['source']}_{c['cls']}_{c['ctype']}_e{c['eps']:g}_N{c['N']}_U{c['updates']}_s{c['seed']}"


def cells_for_seed(seed: int, sources=SOURCES) -> list[dict]:
    """Per seed: sources x classes x N x ({clean} + types x eps) = 2 x 2 x 2 x 5 = 40 cells."""
    out = []
    for src in sources:
        for N in SRC[src]["NS"]:
            for cls in CLASSES:
                out.append(cell(src, cls, "clean", 0.0, N, seed))
                for ct in CTYPES:
                    for e in EPS:
                        out.append(cell(src, cls, ct, e, N, seed))
    return out


def run_cell(c: dict, device, smoke: bool = False) -> dict:
    t_all = time.time(); seed = c["seed"]; src = c["source"]
    N = min(256, c["N"]) if smoke else c["N"]; U = 40 if smoke else c["updates"]; lrs = c["lrs"][:1] if smoke else c["lrs"]; every = 20 if smoke else 100
    roles, rec = draw_roles(src, N, seed, eval_n=512 if smoke else EVAL_N); clean_roles = roles
    roles, crec = contaminate(src, roles, rec, c["ctype"], c["eps"], seed)
    E = clean_roles["EVAL"]; rows = []
    for arm in c["arms"]:
        group = []
        for lr in lrs:
            critic, stateJ, info = sep.train_arm(arm, c["cls"], "product", lr, roles, c["d"], c["B"], U, seed, device, None, every)
            ro = readouts(critic, arm, E, device)
            critic.load_state_dict(stateJ); critic.eval(); roJ = readouts(critic, arm, E, device)
            group.append({"arm": arm, "gamma": sep.ARMS[arm], "lr": lr, "select": {k: v for k, v in info.items() if k != "select_curve"},
                          "select_curve": info["select_curve"], "readout": ro, "readout_commonJ_selected": roJ, "selected": False, "selected_commonJ": False})
        rows += group
    R = {"cell": {**c, "name": cell_name(c), "protocol_id": PROTOCOL, "smoke": smoke, "N_run": N, "updates_run": U, "lrs_run": list(lrs),
                  "roles": {k: {kk: vv for kk, vv in rec[k].items() if not kk.startswith("_") and kk != "caption_per_unit"} for k in ("FIT", "SELECT", "EVAL")},
                  "sizes": rec["sizes"], "shift": rec["shift"], "contamination_records": crec, "provenance": load_source(src)["prov"],
                  "code_commit": git_commit(), "device": str(device), "torch": torch.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "data_authorisation": {"image_data": "frozen feature caches only", "official_test_accessible": False,
                                         "note": "COCO train2017 CLIP features (P49 roles) / CIFAR-10 FIT-split A-P3 features (P150 cache); no test split"}},
         "rows": rows, "wall_seconds": time.time() - t_all}
    return reselect_finite(R)


def reselect_finite(R: dict) -> dict:
    """P157's finite-guarded selection (native risk primary, common J sensitivity; a non-finite score is never selected)."""
    rows = R["rows"]; grid = sorted({r["lr"] for r in rows}); fin = lambda v, bad: v if (v is not None and math.isfinite(v)) else bad
    for arm in {r["arm"] for r in rows}:
        g = [r for r in rows if r["arm"] == arm]
        for r in g:
            r["selected"] = False; r["selected_commonJ"] = False
            r["lr_edge"] = ("low" if r["lr"] == grid[0] else "high" if r["lr"] == grid[-1] else None) if len(grid) > 1 else None
        min(g, key=lambda r: fin(r["select"]["select_risk"], math.inf))["selected"] = True
        max(g, key=lambda r: fin(r["select"]["commonJ_select"], -math.inf))["selected_commonJ"] = True
    R["cell"]["reselected_finite"] = True
    return R
