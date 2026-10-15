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
    s["hash"] = _h({k: (np.asarray(v).round(8).tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if k != "hash"})
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


def run_unit(ds: dict, cfg: HostConfig, out_root: Path, dataset: str, rho: float, method: str, seed: int, device: str = "cpu", epochs: int | None = None,
             lr_mult: float = 1.0, critic_steps: int = 400, formal: bool = False, anchor_fraction: float = 0.1, min_sizes=None, test: dict | None = None) -> dict:
    shared = prepare_shared(ds, cfg, out_root, dataset, rho, seed, device, anchor_fraction, min_sizes)
    scores = compute_scores(shared, device, cfg.batch_size)
    n3 = None
    if method == "RankMass":
        n3 = make_weights(shared, scores, "VCS_Weight", seed, critic_steps)["lock"]["record"]["total_mass"]
    wres = make_weights(shared, scores, method, seed, critic_steps, n3_mass=n3)
    unit_dir = out_root / dataset / f"rho{rho:g}" / f"s{seed}" / f"unit_{method}_lr{lr_mult:g}"
    student = WeightedStudent(cfg, shared["observed"], wres["w"], shared["collate"], ds["dev_images"], ds["dev_captions"], unit_dir, seed, lr_mult, device, wres["lock"], formal)
    done = student.fit(epochs)
    result = {"unit": unit_dir.name, "dataset": dataset, "rho": rho, "method": method, "seed": seed, "noise_manifest": shared["noise_manifest"], "weights_record": wres["lock"]["record"],
              "scores_hash": scores["hash"], "done": done}
    if test is not None:                                        # test evaluation: a separate, explicit step on the selected (best dev) checkpoint
        import torch
        from .teacher import build_host
        host = build_host(cfg).to(device); ck = torch.load(unit_dir / "best.pt", map_location="cpu", weights_only=False); host.load_state_dict(ck["model"]); host.eval()
        sims = similarity_matrix(host, test["images"], test["captions"], device=device); result["test"] = evaluate_split(sims, dataset)
    (unit_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result
