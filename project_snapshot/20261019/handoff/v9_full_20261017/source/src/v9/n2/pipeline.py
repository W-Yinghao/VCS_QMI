"""V9-N2 end-to-end driver for one (dataset, noise, method, seed) unit on an in-memory dataset object (fixture or the real precomp data from
data.py): roles -> noise -> teacher (per dataset x noise x seed, shared by all methods) -> teacher scores -> critic / weights (locked file) ->
weighted student -> dev selection -> test evaluation written separately.  Shared stages are cached under <out_root>/<dataset>/rho<rho>/s<seed>/
so the eight methods of a cell reuse one teacher and one noise draw; every stage writes its hash into the unit manifest.
"""
from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np

from .eval import evaluate_split, similarity_matrix
from .fixtures import collate_pairs
from .noise import inject_mismatch, load_observed, make_pair_table, write_noise
from .precheck import loss_coefficients, role_isolation, teacher_separability, weight_tables, write_precheck
from .roles import assign_roles, role_of_image, write_roles
from .teacher import HostConfig, load_teacher, score_pairs, train_teacher
from .train import WeightedStudent
from .weights import read_weight_lock, repair_within_role, weights_for_method, write_weight_lock

ANCHOR_ROLES = ("teacher_warmup", "critic_fit", "critic_select", "reference_moments")


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def prepare_shared(ds: dict, cfg: HostConfig, out_root: Path, dataset: str, rho: float, seed: int, device: str = "cpu", anchor_fraction: float = 0.1, min_sizes=None) -> dict:
    """Roles, noise and the teacher for one (dataset, rho, seed); idempotent."""
    base = out_root / dataset / f"rho{rho:g}" / f"s{seed}"; base.mkdir(parents=True, exist_ok=True)
    roles_p = base / "roles.json"
    if not roles_p.exists():
        roles = assign_roles(sorted(set(str(i) for i in ds["caption_image"].values())), seed, anchor_fraction, min_sizes=min_sizes); write_roles(roles, roles_p)
    roles = json.loads(roles_p.read_text()); rmap = role_of_image(roles)
    tag = "pairs"
    if not (base / f"{tag}.observed_pairs.json").exists():
        pairs = make_pair_table(ds["caption_image"], rmap); observed, labels, manifest = inject_mismatch(pairs, rho, seed); write_noise(observed, labels, manifest, base, tag)
    observed = load_observed(base / f"{tag}.observed_pairs.json"); manifest = json.loads((base / f"{tag}.noise_manifest.json").read_text())
    collate = partial(collate_pairs, ds["images"], ds["captions"])
    if not (base / "teacher" / "teacher.pt").exists():
        donor = None
        for other in sorted((out_root / dataset).glob("rho*")):              # same (dataset, seed): A is clean, roles are seeded identically -> identical teacher
            cand = other / f"s{seed}"
            if cand != base and (cand / "teacher" / "teacher.pt").exists() and (cand / "roles.json").exists() and json.loads((cand / "roles.json").read_text())["hashes"] == roles["hashes"]:
                donor = cand; break
        if donor is not None:
            import shutil
            (base / "teacher").mkdir(exist_ok=True)
            for f in ("teacher.pt", "teacher_log.json"):
                shutil.copy(donor / "teacher" / f, base / "teacher" / f)
            (base / "teacher" / "teacher_shared_from.json").write_text(json.dumps({"from": str(donor / "teacher" / "teacher.pt"), "sha256": hashlib.sha256((base / "teacher" / "teacher.pt").read_bytes()).hexdigest(),
                                                                                    "reason": "identical roles hashes and clean anchor pairs: the teacher depends on (dataset, seed) only"}, indent=1))
        else:
            train_teacher(cfg, [r for r in observed if r["role"] == "teacher_warmup"], collate, base / "teacher", seed, device)
    return {"base": base, "roles": roles, "observed": observed, "noise_manifest": manifest, "collate": collate, "teacher": base / "teacher" / "teacher.pt"}


def compute_scores(shared: dict, device: str = "cpu", batch: int = 128) -> dict:
    """Teacher scores for every role's P pairs, the role-isolated Q re-pairings of critic-fit / select / reference, and the observed U pairs."""
    host, _ = load_teacher(shared["teacher"], device); obs = shared["observed"]; rng = np.random.default_rng([shared["roles"]["seed"], 9_030])
    by_role = {r: [x for x in obs if x["role"] == r] for r in ANCHOR_ROLES + ("U",)}
    s = {}
    for role, key in (("critic_fit", "fit"), ("critic_select", "sel"), ("reference_moments", "ref")):
        P = by_role[role]; Q = repair_within_role(P, rng)
        s[f"{key}_P"], s[f"{key}_Q"] = score_pairs(host, P, shared["collate"], batch, device), score_pairs(host, Q, shared["collate"], batch, device)
        s[f"{key}_P_img"], s[f"{key}_Q_img"] = [r["observed_image_id"] for r in P], [r["q_of"] for r in Q]
    U = by_role["U"]; s["U"] = score_pairs(host, U, shared["collate"], batch, device); s["U_img"] = [r["observed_image_id"] for r in U]; s["U_pair_ids"] = [r["pair_id"] for r in U]
    # protocol §05.3 step 3: hold out 20 % of U images (by observed image, seeded) as the independent batch for the rho interval; the point estimate uses the rest
    u_imgs = sorted(set(s["U_img"])); rng_u = np.random.default_rng([shared["roles"]["seed"], 9_031]); held = set(rng_u.choice(u_imgs, size=max(1, int(round(0.2 * len(u_imgs)))), replace=False).tolist()) if len(u_imgs) >= 5 else set()
    s["U_est_mask"] = np.array([img not in held for img in s["U_img"]], dtype=bool); s["U_independent_images"] = sorted(held)
    s["hash"] = _h({k: (np.asarray(v).astype(int).tolist() if isinstance(v, np.ndarray) and v.dtype == bool else np.asarray(v).round(8).tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if k != "hash"})
    return s


def make_weights(shared: dict, scores: dict, method: str, seed: int, steps: int = 400, n3_mass: float | None = None) -> dict:
    base = shared["base"]; wdir = base / "weights"; wpath = wdir / f"{method}.csv"
    if (wpath.exists() and Path(str(wpath) + ".lock.json").exists()):
        return {"path": wpath, "lock": json.loads(Path(str(wpath) + ".lock.json").read_text()), "w": read_weight_lock(wpath)}
    s = dict(scores)
    if method == "RankMass":
        if n3_mass is None:
            raise ValueError("RankMass needs N3's training-side total mass (run VCS_Weight first)")
        s["total_mass"] = n3_mass
    res = weights_for_method(method, s, seed=seed, steps=steps)
    rec = res["record"]; rho_hat = rec.get("rho_used")
    lock = write_weight_lock(wpath, scores["U_pair_ids"], res["w"], scores["U"], rho_hat, fit_hash=_h(rec.get("critic", method)), reference_hash=_h({"a": rec.get("rho", {}).get("a"), "b": rec.get("rho", {}).get("b"), "scores": scores["hash"]}), record=rec)
    return {"path": wpath, "lock": lock, "w": read_weight_lock(wpath)}


WEIGHT_METHODS = ("Unweighted_A", "VCS_Weight", "BCE_Weight", "RuLSIF_Weight", "Nystrom_Weight", "RankMass", "PlainFilter")


def run_precheck(ds: dict, cfg: HostConfig, out_root: Path, dataset: str, rho: float, seed: int, device: str = "cpu", critic_steps: int = 400, anchor_fraction: float = 0.1,
                 min_sizes=None, methods=WEIGHT_METHODS, split_files: dict | None = None) -> dict:
    """The pre-registered gate (owner 2026-10-15): teacher separability on A-critic-select, every method's weights, loss coefficients, roles."""
    shared = prepare_shared(ds, cfg, out_root, dataset, rho, seed, device, anchor_fraction, min_sizes); scores = compute_scores(shared, device, cfg.batch_size)
    sep = teacher_separability(scores["sel_P"], scores["sel_Q"]); wmap = {}
    n3 = None
    for m in methods:
        try:
            if m == "RankMass":
                n3 = n3 if n3 is not None else make_weights(shared, scores, "VCS_Weight", seed, critic_steps)["lock"]["record"]["total_mass"]
            wmap[m] = make_weights(shared, scores, m, seed, critic_steps, n3_mass=n3)["w"]
        except NotImplementedError as e:
            wmap[m] = None; print(f"[precheck] {m}: {e}")
    wmap = {m: w for m, w in wmap.items() if w is not None}
    wt = weight_tables({m: np.array([w[p] for p in scores["U_pair_ids"]]) for m, w in wmap.items()})
    coef = loss_coefficients(shared["observed"], wmap, seed); iso = role_isolation(shared["roles"], split_files)
    return write_precheck(shared["base"] / "precheck", dataset, rho, seed, sep, wt, coef, iso, shared["noise_manifest"])


def precheck_allows(out_root: Path, dataset: str, rho: float, seed: int) -> dict:
    p = out_root / dataset / f"rho{rho:g}" / f"s{seed}" / "precheck" / f"precheck_{dataset}_rho{rho:g}_s{seed}.json"
    if not p.exists():
        raise ValueError(f"formal unit refused: no precheck at {p} (run `n2_run.py precheck` first; it is the pre-registered gate)")
    rec = json.loads(p.read_text())
    if not rec["formal_block_allowed"]:
        raise ValueError(f"formal unit refused by the precheck: {rec['refusal_reason']}")
    return rec


def unit_members(out_root: Path, dataset: str, rho: float, seed: int, method: str) -> list:
    """Members of the merged group that contains `method` in this cell's precheck (pre-registered merge rule); [method] when no precheck exists."""
    p = out_root / dataset / f"rho{rho:g}" / f"s{seed}" / "precheck" / f"precheck_{dataset}_rho{rho:g}_s{seed}.json"
    if not p.exists():
        return [method]
    groups = json.loads(p.read_text())["loss_coefficients"].get("groups", [])
    return next((sorted(g) for g in groups if method in g), [method])


def run_unit(ds: dict, cfg: HostConfig, out_root: Path, dataset: str, rho: float, method: str, seed: int, device: str = "cpu", epochs: int | None = None,
             lr_mult: float = 1.0, critic_steps: int = 400, formal: bool = False, anchor_fraction: float = 0.1, min_sizes=None, test: dict | None = None,
             members: list | None = None, loss_scale: str = "mean", grad_log_window: tuple | None = None, stop_epoch_steps: int | None = None,
             unit_tag: str = "") -> dict:
    group = unit_members(out_root, dataset, rho, seed, method)
    if members is None:
        members = group
    else:                                                       # explicit attribution (release orchestrator): must lie inside the precheck coefficient group
        members = sorted(set(members))
        if method not in members or any(m not in group for m in members):
            raise ValueError(f"members {members} are not all in the precheck coefficient group of {method} ({group}); one run cannot be attributed to them")
    shared = prepare_shared(ds, cfg, out_root, dataset, rho, seed, device, anchor_fraction, min_sizes)
    scores = compute_scores(shared, device, cfg.batch_size)
    precheck = precheck_allows(out_root, dataset, rho, seed) if formal else None
    n3 = None
    if method == "RankMass":
        n3 = make_weights(shared, scores, "VCS_Weight", seed, critic_steps)["lock"]["record"]["total_mass"]
    wres = make_weights(shared, scores, method, seed, critic_steps, n3_mass=n3)
    if unit_tag and formal:
        raise ValueError("a tagged (diagnostic) unit is never formal")
    unit_dir = out_root / dataset / f"rho{rho:g}" / f"s{seed}" / f"unit_{method}_lr{lr_mult:g}{unit_tag}"
    student = WeightedStudent(cfg, shared["observed"], wres["w"], shared["collate"], ds["dev_images"], ds["dev_captions"], unit_dir, seed, lr_mult, device, wres["lock"], formal,
                              loss_scale=loss_scale, grad_log_window=grad_log_window, stop_epoch_steps=stop_epoch_steps)
    done = student.fit(epochs)
    result = {"unit": unit_dir.name, "dataset": dataset, "rho": rho, "method": method, "seed": seed, "lr_mult": lr_mult, "formal": formal, "noise_manifest": shared["noise_manifest"],
              "weights_record": wres["lock"]["record"], "scores_hash": scores["hash"], "done": done,
              "members": members, "members_note": "methods whose loss coefficients are identical to this run's (precheck merge rule): ONE training, reported for every member; not independent replicates" if len(members) > 1 else "no merge",
              "precheck": None if precheck is None else {"formal_block_allowed": precheck["formal_block_allowed"], "merged": precheck["loss_coefficients"]["merged"].get(method)}}
    if test is not None:                                        # test evaluation: a separate, explicit step on the selected (best dev) checkpoint
        import torch
        from .teacher import build_host
        host = build_host(cfg).to(device); ck = torch.load(unit_dir / "best.pt", map_location="cpu", weights_only=False); host.load_state_dict(ck["model"]); host.eval()
        sims = similarity_matrix(host, test["images"], test["captions"], device=device); result["test"] = evaluate_split(sims, dataset)
    (unit_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result
