"""V10 method set on the EPFL RGB-NIR host (decision 2026-10-17 section 6.2 item 5).  Torch.

The registration host is R6 REG-P3 (src/v9/reg/r6_reg_p3.py, verbatim copy, sha256 pinned below), i.e. REG-P2's rigid machinery on one
centred 512 x 512 NATIVE window per kept EPFL pair (down = 1): reference U = RGB, moving V = NIR.  Differences from the BF-SHG host that
the V10 BF code would get wrong if it were pointed at the EPFL cache unchanged (each one is handled here and tested):
  moving image   P3.Inputs3: the raw NIR window ('nir' = NIR / 255) is perturbed (V o T*^-1, bilinear, reflection) and robust-normalised
                 LINEARLY for both the intensity and the learned path (REG-P3 coordinator rule (d)); the BF Q.Inputs path would apply
                 expm1 / log1p to a key that holds linear NIR
  training input P.learned_inputs on the REG-P3 cache: 'bf_rgb_n' = RGB robust-normalised [3, 512, 512], 'shg_log_n' = LINEAR NIR
                 robust-normalised [1, 512, 512] (key names kept by REG-P3 so the R6 trainers run unchanged)
  FIT set        the 336 FIT windows of 330 units (frac 1.0: every window, the V9-G2 sampler at fraction 1); BF code asserts FIT32
  pixel unit     err_full = Q.corner_error_full = 4 x the 512-frame px; on EPFL the frame is native, so err_full = 4 x native px and the
                 thresholds 42 / 9 are 10.5 / 2.25 NATIVE px (BF: 10.5 / 2.25 px of the x4-downsampled frame = 42 / 9 full-resolution px)
  perturbations  Q.perturbations_p2(window id), identical to REG-P3's perturbations.json (pinned); translations up to 51 native px
  data gate      EPFLDataGuard (v10.epfl.guard), admission by a pinned activation record; DEV only at stage ACTIVATED
Every method keeps its validated BF implementation (src/v10/adapter.py, src/v10/supplement/*, unchanged); only the data path, the FIT
set and, for CoMIR / RuLSIF, the FIT-size-dependent constructor are EPFL-specific here.  The RuLSIF centre rule cannot be the BF rule
(4 centres per FIT core of 32) on 336 FIT windows: EPFL_RULSIF_CANDIDATES is a DRAFT (1 paired centre per FIT window = 336 centres) that
needs the owner's decision before any EPFL CAL run.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v10 import adapter as A
from v10.supplement.comir import CoMIRConfig, CoMIRTrainer, verify_official_source
from v10.supplement.contract import FAMILIES as SUPPLEMENT_FAMILIES, HOST_METHODS, candidates as supplement_candidates, schedule as supplement_schedule
from v10.supplement.registration import FiniteEnergy, RegistrationSession, patches
from v10.supplement.rulsif import ExactRuLSIF
from v10.supplement.training import FixedTrainer, KernelTrainer

from .guard import EPFLDataGuard, P3_PERTURBATIONS_SHA256

P3_FILE_SHA256 = "5bbc3ebaf11b0cdca37663e0f0205bf00ab797fb523b3f8c7a38dd48b2e9c7a2"
_p3_path = Path(A.G.__file__).resolve().parent / "r6_reg_p3.py"
if hashlib.sha256(_p3_path.read_bytes()).hexdigest() != P3_FILE_SHA256:
    raise ImportError("r6_reg_p3.py differs from the pinned REG-P3 copy")
import r6_reg_p3 as P3  # noqa: E402  (src/v9/reg is on sys.path through the adapter's host import)

if P3.P is not A.P or P3.Q is not A.Q:
    raise ImportError("r6_reg_p3 must share the adapter's r6_reg_pilot / r6_reg_p2 module objects")
for _n in ("cmd_prepare", "cmd_perturb", "cmd_train", "cmd_comir", "cmd_register", "register_one", "prepare_epfl", "prepare_zurich",
           "write_cache", "main"):
    setattr(P3, _n, A._tripwire(f"P3.{_n}"))          # REG-P3 writers (outputs/R6_REG_P3) can never run from V10

BF1_METHODS = {"MF_VCS_perm1": "MF_perm1", "MF_VCS_M0": "MF_M0", "MF_VCS_M1": "MF_M1", "MLP_VCS_perm1": "MLP_perm1"}
FAMILIES = tuple(BF1_METHODS) + tuple(SUPPLEMENT_FAMILIES)
EPFL_FIT_WINDOWS, EPFL_CAL_WINDOWS, EPFL_DEV_WINDOWS = 336, 10, 74
ERR_UNIT = "err_full = 4 x native px on EPFL (512-frame px x 4); success < 42 / < 9 err_full = < 10.5 / < 2.25 native px"
EPFL_RULSIF_CANDIDATES = [{"order": i, "bandwidth_multiplier": w, "ridge": r, "alpha": 0.5, "centres_per_fit_window": 1,
                           "fit_positions_per_window": 2048, "patch_radius": 4, "status": "DRAFT_REQUIRES_OWNER_DECISION"}
                          for i, (w, r) in enumerate((w, r) for w in (0.5, 1.0, 2.0) for r in (1e-4, 1e-3, 1e-2))]


def candidates(family):
    """Frozen BF candidate grids reused on EPFL (decision 8.1 order); RuLSIF's EPFL centre rule is a draft (module doc)."""
    if family in BF1_METHODS:
        return [{"order": i, "lr": lr, "cap": cap} for i, (lr, cap) in enumerate(
            (lr, cap) for lr in (1e-4, 3e-4, 1e-3) for cap in ((10, 30, 100) if family == "MF_VCS_M1" else (None,)))]
    if family == "RuLSIF_exact":
        return [dict(c) for c in EPFL_RULSIF_CANDIDATES]
    return supplement_candidates(family)


def schedule(family):
    if family in BF1_METHODS:
        return {"kind": "gradient_training", "steps": 3000, "save_steps": list(range(250, 3001, 250)), "cal_steps": list(range(250, 3001, 250)),
                "final_seeds": [1001, 1002, 1003]}
    return supplement_schedule(family)


# ------------------------------------------------------------------------------------------------------------ data path
class EPFLGuardedInputs(P3.Inputs3):
    """REG-P3's Inputs3 (reference: luminance [1, S, S], RGB [3, S, S]; moving: perturbed raw NIR, linear robust-norm on both paths) behind
    the EPFL gate; every first access of a (window) / (window, perturbation) is a logged guarded read."""

    def __init__(self, guard: EPFLDataGuard, allowed_roles, purpose: str, pert: dict, device):
        super().__init__(torch, F, guard.r6_root, pert, torch.device(device))
        self.guard, self.allowed, self.purpose = guard, set(allowed_roles), purpose

    def reference(self, c):
        if c not in self.ref:
            self.guard.check(c, self.allowed, self.purpose + ":reference")
        return super().reference(c)

    def moving(self, c, i):
        if (c, i) not in self.mov:
            self.guard.check(c, self.allowed, self.purpose + f":moving#{i}")
        return super().moving(c, i)


def fit_windows(guard: EPFLDataGuard) -> list[str]:
    """Every FIT window (frac 1.0; V9-G2's sampler returns all windows at fraction 1); must equal REG-P3's FIT window set."""
    fit = guard.included("FIT")
    if fit != sorted(w for w, r in guard.p3_roles.items() if r == "FIT") or len(fit) != EPFL_FIT_WINDOWS:
        raise RuntimeError("EPFL FIT set differs from REG-P3's 336 FIT windows")
    return fit


def load_learned_inputs(guard, windows, allowed_roles, purpose, device) -> dict:
    out = {}
    for w in sorted(set(windows)):
        guard.check(w, allowed_roles, purpose)
        out[w] = A.P.learned_inputs(torch, guard.r6_root, w, torch.device(device))
    return out


_PERT_CACHE: dict = {}


def pinned_perturbations(guard) -> dict:
    if "p" not in _PERT_CACHE:
        raw = (guard.r6_root / "perturbations.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != P3_PERTURBATIONS_SHA256:
            raise RuntimeError("REG-P3 perturbations.json changed")
        import json
        _PERT_CACHE["p"] = json.loads(raw)
    return _PERT_CACHE["p"]


def perturbations_for(guard, windows, role, purpose) -> dict:
    """REG-P2's rigid family per window (Q.perturbations_p2(window id)); for CAL / DEV windows it must equal REG-P3's pinned
    perturbations.json (checked); FIT windows (geometry checks only) have no REG-P3 entry and use the same rule."""
    ref = pinned_perturbations(guard); out = {}
    for w in windows:
        guard.log_rule(w, {role}, purpose)
        out[w] = A.Q.perturbations_p2(w)
        if role in ("CAL", "DEV") and out[w] != ref.get(w):
            raise RuntimeError(f"{w}: perturbation rule differs from REG-P3's perturbations.json")
    return out


# ------------------------------------------------------------------------------------------------------------ trainers
class EPFLCoMIRTrainer(CoMIRTrainer):
    """The validated CoMIR recipe (supplement.comir, 23 x 32 x 32, A = moving NIR, B = reference RGB) on the 336 EPFL FIT windows."""

    def __init__(self, config, guard, device):   # noqa: D401 - same body as CoMIRTrainer.__init__ with the EPFL FIT set
        self.cfg, self.device = config, torch.device(device)
        self.source = verify_official_source()
        self.fit = fit_windows(guard)
        self.images = {}
        for w in self.fit:
            guard.check(w, {"FIT"}, "V10 EPFL CoMIR full FIT training")
            self.images[w] = A.P.load_img(guard.r6_root, w, ("shg_log_n", "bf_rgb_n"))
        self.moving_encoder, self.reference_encoder, self.architecture = A.P.comir_models(torch, config.seed, self.device)
        self.optimizers = [torch.optim.SGD(m.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
                           for m in self.modules()]
        self.g = np.random.default_rng(3000 + config.seed)
        self.n_steps = 0


class EPFLKernelTrainer(KernelTrainer):
    """Exact-moment RuLSIF on EPFL FIT windows, DRAFT centre rule: one paired centre per FIT window (336 centres), fit positions per window
    and patch geometry as BF; median bandwidth from the paired FIT centres only.  Not run in any CAL before the owner decides the rule."""

    def __init__(self, candidate, seed, guard, device):
        FixedTrainer.__init__(self, "RuLSIF_exact", candidate, seed, device)
        fit = fit_windows(guard); self.scenes = []
        g = np.random.default_rng(A.Q.seed_of(f"V10|RuLSIF|EPFL|FIT|{seed}"))
        for w in fit:
            guard.check(w, {"FIT"}, "V10 EPFL RuLSIF positive patches and exact product moments")
            left, right = A.P.load_img(guard.r6_root, w, ("bf", "shg"))
            il = torch.as_tensor(left, device=self.device, dtype=torch.float32)[None]; ir = torch.as_tensor(right, device=self.device, dtype=torch.float32)[None]
            pos = torch.as_tensor(g.uniform(5, A.P.SIZE - 6, (candidate["fit_positions_per_window"], 2)), device=self.device, dtype=torch.float32)
            self.scenes.append((patches(il, pos), patches(ir, pos)))
        count = candidate["centres_per_fit_window"]; cl, cr = [], []
        for left, right in self.scenes:
            ids = torch.as_tensor(g.choice(len(left), count, replace=False), device=self.device); cl.append(left[ids]); cr.append(right[ids])
        cx, cy = torch.cat(cl), torch.cat(cr)
        d = torch.pdist(torch.cat((cx, cy), -1).double()); d = d[d > 0]
        if d.numel() == 0:
            raise RuntimeError("Degenerate FIT-only kernel bandwidth")
        median = float(d.median())
        self.ratio = ExactRuLSIF(cx, cy, median * candidate["bandwidth_multiplier"], candidate["ridge"], candidate["alpha"])
        self.fit_metadata = {"fit_windows": fit, "bandwidth_median": median, "bandwidth": self.ratio.bandwidth, "centres": int(len(cx)),
                             "centres_selection": "DRAFT: paired, one per FIT window, without replacement",
                             "fit_positions_per_window": candidate["fit_positions_per_window"]}


def new_trainer(family, candidate, seed, guard, device):
    if family in BF1_METHODS or family in HOST_METHODS:
        fit = fit_windows(guard)
        imgs = load_learned_inputs(guard, fit, {"FIT"}, "V10 EPFL full FIT training", device)
        if family in BF1_METHODS:
            cfg = A.TrainConfig(method=BF1_METHODS[family], seed=seed, encoder_lr=candidate["lr"], head_lr=candidate["lr"], K=16, n_sc=16,
                                n_pos=128, crop=160, m1=A.M1Config(max_iterations=candidate.get("cap") or 30))
        else:
            cfg = A.TrainConfig(method=HOST_METHODS[family], seed=seed, encoder_lr=candidate["lr"], head_lr=candidate["lr"], n_sc=16, n_pos=128,
                                crop=160, mlp_perms=8)
        t = A.Trainer(cfg, imgs, fit, device); t.optimizers = [t.opt]
        return t
    if family == "CoMIR_native":
        return EPFLCoMIRTrainer(CoMIRConfig(seed=seed, epochs=candidate["epochs"], steps_per_epoch=candidate["steps_per_epoch"],
                                            batch_size=candidate["batch_size"], learning_rate=candidate["lr"]), guard, device)
    if family == "RuLSIF_exact":
        return EPFLKernelTrainer(candidate, seed, guard, device)
    if family in ("MI_native", "NMI_native", "MIND_2D_adapted"):
        return FixedTrainer(family, candidate, seed, device)
    raise ValueError("Unknown family: " + family)


# ------------------------------------------------------------------------------------------------------------ registration
class EPFLRegistrationSession(RegistrationSession):
    """The supplement RegistrationSession (common host geometry, role-gated inputs, per-window reference cache, no cross-perturbation
    moving cache, finite-energy wrapper) on EPFL inputs, extended to the four first-batch families with the BF1 runner's energies:
    MF -> A.MFRigidEnergy (perm1: the host permutation; M0 / M1: exact moments), MLP_VCS_perm1 -> Q.RigidEnergy 'critic'."""

    def __init__(self, family, trainer, guard, role, windows, device, candidate):
        if role not in ("CAL", "DEV"):
            raise PermissionError("Only CAL / DEV registration sessions exist")
        self.family, self.trainer, self.device, self.candidate = family, trainer, torch.device(device), candidate
        self.perturbations = perturbations_for(guard, windows, role, f"V10 EPFL frozen perturbation rule ({role})")
        self.inputs = EPFLGuardedInputs(guard, {role}, "V10 EPFL " + role, self.perturbations, self.device)
        self.core, self.reference = None, None

    def energy(self, core, perturbation):
        if self.family not in BF1_METHODS:
            return super().energy(core, perturbation)
        A.synchronize(self.device); started = time.perf_counter(); ref_seconds = 0.0
        if self.core != core:
            self.inputs.ref.clear(); self.inputs.mov.clear()
            self.reference = A.reference_descriptors(self.trainer.eU, self.inputs, core)
            self.core = core; A.synchronize(self.device); ref_seconds = time.perf_counter() - started
        mov = self.inputs.moving(core, perturbation)[1]
        rng = np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{core}|{perturbation}"))
        if self.trainer.mf:
            est = "perm1" if self.trainer.cfg.method == "MF_perm1" else "moments"
            energy = A.MFRigidEnergy(self.reference, mov, rng, eV=self.trainer.eV, head=self.trainer.head, estimator=est)
        else:
            energy = A.Q.RigidEnergy(torch, F, "critic", self.reference, mov, rng, eV=self.trainer.eV, critic=self.trainer.critic)
        A.synchronize(self.device); total = time.perf_counter() - started
        self.inputs.mov.pop((core, perturbation), None)
        return FiniteEnergy(energy), {"preparation_seconds": total, "reference_preparation_seconds": ref_seconds,
                                      "moving_and_energy_preparation_seconds": total - ref_seconds}


def native_px(err_full: float) -> float:
    """EPFL pixel unit: err_full / 4 (module doc)."""
    return float(err_full) / A.P.DOWN


def success_thresholds_native() -> dict:
    return {"primary_err_full": A.P.THR_FULL[0], "strict_err_full": A.P.THR_FULL[1], "primary_native_px": A.P.THR_FULL[0] / A.P.DOWN,
            "strict_native_px": A.P.THR_FULL[1] / A.P.DOWN, "interpretation": "recovery of the known imposed perturbation; not a bound on the "
            "true RGB-NIR alignment error (decision 6.2)"}


def mean_corner_error_native(th_hat_deg: float, t_hat, th_star_deg: float = 0.0, t_star=(0.0, 0.0)) -> float:
    """Mean corner error (native px on EPFL) between two rigid transforms about the window centre on REG-P2's corner patch (corners at
    +-HALF834 px of the 512 frame): mean_k |R(th_hat) c_k + t_hat - R(th_star) c_k - t_star|.  Written independently of
    Q.corner_error_full (explicit per-corner cos / sin) for the pixel-unit tests: Q.corner_error_full = 4 x this value."""
    a = A.Q.HALF834; out = []
    for y, x in ((-a, -a), (-a, a), (a, -a), (a, a)):
        ys = []
        for th, t in ((th_hat_deg, t_hat), (th_star_deg, t_star)):
            c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
            ys.append((c * y - s * x + t[0], s * y + c * x + t[1]))
        out.append(math.hypot(ys[0][0] - ys[1][0], ys[0][1] - ys[1][1]))
    return sum(out) / 4
