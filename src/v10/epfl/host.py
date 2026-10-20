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
    """Every FIT window (frac 1.0; V9-G2's sampler returns all windows at fraction 1): REG-P3's FIT windows minus the windows a frozen group
    record removed (production: 334 = 336 - water_0006 - street_0020).  A smoke guard may declare its own expected subset."""
    fit = guard.included("FIT")
    expected = guard.expected_fit_windows() if hasattr(guard, "expected_fit_windows") else \
        sorted(w for w, r in guard.p3_roles.items() if r == "FIT" and w not in getattr(guard, "removed", {}))
    if fit != expected or (not getattr(guard, "smoke", False) and len(fit) != EPFL_FIT_WINDOWS - len(getattr(guard, "removed", {}))):
        raise RuntimeError(f"EPFL FIT set ({len(fit)}) differs from REG-P3's FIT windows minus the group-record removals")
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


# ------------------------------------------------------------------------------------------------------------ production families (decision 2026-10-19)
PRODUCTION_FAMILIES = tuple(BF1_METHODS) + ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]", "MI_native", "NMI_native", "CoMIR_native",
                                            "RuLSIF_exact_v2", "MIND_2D_adapted")
RULSIF_V2 = "RuLSIF_exact_v2"


class RuLSIFV2Unavailable(RuntimeError):
    """EPFL RuLSIF waits for the frozen RuLSIF_exact_v2 (main-session decision 2026-10-19 no. 2): v1 is never adapted."""


def rulsif_v2_module():
    """The v2 module and the supplement contract of the frozen v2 (merged unchanged from v10/rulsif-exact-v2 2e6ea58)."""
    try:
        from v10.supplement import rulsif_v2 as V2
        from v10.supplement import contract as C
        C.rulsif_v2_eligible  # noqa: B018 - the amended (A1) contract must be present
    except (ImportError, AttributeError) as e:
        raise RuLSIFV2Unavailable("RuLSIF_exact_v2 is not in this source tree; EPFL RuLSIF waits for the frozen v2 (decision 2)") from e
    return V2, C


# EPFL mapping of v2's rules (decision 2026-10-19 no. 2, frozen in EPFL_PROTOCOL_FROZEN.json).  The centre rule was revised before any EPFL v2
# number existed: revision 2 (main-session cost decision: 1024 centres in total) and revision 3 (owner approval of 1024; seed-independent
# pseudo-random extra-centre order; provenance for every fit; a pre-registered 2048-centre capacity sensitivity) - EPFL_RULSIF_V2_ADDENDUM_REV3.json.
BF_FIT_CORES = 32                      # BF v2's FIT32: the function-class size is BF_FIT_CORES x centres_per_core = 32 x 32 = 1024 centres
RULSIF_V2_CENTRES = {1024: "primary", 2048: "capacity_sensitivity"}     # the only admissible run-identity values of the total centre count
EXTRA_ORDER_TAG = "V10|EPFL|RuLSIF_exact_v2|EXTRA_CENTRE_ORDER"


def epfl_centre_counts(n_windows: int, total: int) -> list:
    """REVISION 2 (superseded by revision 3, kept for its record): floor(total / n) per window plus one more for the first total mod n windows
    in sorted order.  Production uses centre_counts()."""
    if n_windows < 1 or total < n_windows:
        raise ValueError("every FIT window needs at least one centre")
    base, extra = divmod(total, n_windows)
    return [base + (1 if i < extra else 0) for i in range(n_windows)]


def extra_centre_order(windows) -> list:
    """Revision 3: the fixed pseudo-random order of the sorted FIT windows in which extra centres are given; a dedicated stream that does not
    depend on the seed and never touches v2's per-seed stream."""
    windows = list(windows)
    if windows != sorted(windows) or len(set(windows)) != len(windows):
        raise ValueError("FIT windows must be distinct and sorted")
    perm = np.random.default_rng(A.Q.seed_of(EXTRA_ORDER_TAG)).permutation(len(windows))
    return [windows[i] for i in perm]


def centre_counts(windows, total: int) -> list:
    """Revision 3: floor(total / n) centres per window (sorted window order) plus one more for each of the first total mod n windows of
    extra_centre_order(windows); deterministic and seed-independent."""
    windows = list(windows); n = len(windows)
    base, extra = divmod(total, n) if n else (0, 0)
    if n < 1 or base < 1:
        raise ValueError("every FIT window needs at least one centre")
    plus = set(extra_centre_order(windows)[:extra])
    return [base + (1 if w in plus else 0) for w in windows]


RULSIF_V2_EPFL_MAPPING = {
    "fit_unit": "one EPFL FIT window (centred 512 x 512 native window, REG-P3) takes the place of one BF FIT core",
    "fit_windows": "the 334 FIT windows of the frozen group record v2 (BF: FIT32)",
    "a_constant_basis": "unchanged: g = theta_0 + sum_l theta_l a_l(x) b_l(y), theta_0 outside the ridge",
    "b_fit_positions": "4096 per FIT window on the registration support: the disk of radius Q.R_OMEGA = 129 px about Q.CEN = 255.5 of the 512 window, Q.RigidEnergy's law (the EPFL host is REG-P2's machinery on 512 native windows, so the support is identical)",
    "c_kernel_widths": "per-modality medians of the EPFL FIT centres: h_x from the reference (RGB luminance, cache 'bf'), h_y from the moving (linear NIR, cache 'shg'), each times the candidate multiplier",
    "e_input_transform": "per-image mid-rank empirical CDF of both modalities (every FIT window, the reference and each moving image as given)",
    "f_centres": "T paired centres in total, T = 1024 (primary; BF v2's 32 FIT cores x 32) or T = 2048 (pre-registered capacity sensitivity); every FIT window gets floor(T / 334) and the T mod 334 extra centres go to the first windows of a fixed, seed-independent pseudo-random permutation of the sorted FIT windows, np.random.default_rng(Q.seed_of('V10|EPFL|RuLSIF_exact_v2|EXTRA_CENTRE_ORDER')).permutation(334) (T = 1024: 22 windows with 4, 312 with 3; T = 2048: 44 with 7, 290 with 6, the 44 include the 22); the same windows for every seed; drawn with v2's sampler on the per-seed stream below (all positions window by window, then the paired centres window by window, without replacement within each window); T + 1 coefficients with the constant; per-modality medians over the selected centres; FIT positions 334 x 4096 = 1368064 (identical for T = 1024 and 2048 at one seed); provenance recorded for every fit.  Addendum revision 3; revision 2 gave the extras to the first 22 windows in sorted order, revision 1 used 32 per window (10688 centres)",
    "shg_input": "candidate value 'intensity' (all eligible candidates): EPFL cache 'shg' = linearly robust-normalised NIR, and at registration the REG-P3 moving image (perturbed raw NIR, linear robust-norm); REG-P3 has no log path",
    "stream_tag": "V10|EPFL|RuLSIF_exact_v2|FIT|<seed> (draw order as v2: all positions window by window, then the centres window by window)",
    "candidates": "the frozen v2 grid's amended eligible subset (amendment A1: orders 0-7) in the frozen order; order 8 excluded",
}


def array_sha256(t) -> str:
    """sha256 of a float32 tensor's C-order little-endian bytes."""
    a = t.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(a.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def json_sha256(obj) -> str:
    import json as _json
    return hashlib.sha256(_json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def epfl_fit_scenes(guard, windows, counts, *, support, positions_per_window, stream_tag, device, purpose, shg_input="intensity",
                    input_transform="none", extra_windows=None):
    """v2's fit_scenes (src/v10/supplement/rulsif_v2.py, unchanged) with a per-window centre count: the same RNG stream, the same draw order
    (all positions window by window, then the paired centres window by window, g.choice without replacement), the same patches, transform
    and FIT-only medians over the selected centres.  With equal counts it reproduces v2.fit_scenes bit for bit (tested).
    Returns (scenes, cx, cy, medians, metadata, provenance): metadata = v2's keys + the provenance summary and hashes (-> resolved_kernel),
    provenance = the full per-window lists (-> the sidecar CENTRE_PROVENANCE.json)."""
    V2, _ = rulsif_v2_module()
    if len(windows) == 0 or len(set(windows)) != len(windows):
        raise ValueError("FIT windows must be distinct and nonempty")
    if len(counts) != len(windows) or any(c < 1 or c > positions_per_window for c in counts):
        raise ValueError("Invalid centres per window")
    device = torch.device(device)
    g = np.random.default_rng(A.Q.seed_of(stream_tag))
    scenes, all_positions = [], []
    for w in windows:
        guard.check(w, {"FIT"}, purpose)
        key = V2.SHG_INPUTS[shg_input]
        left, right = A.P.load_img(guard.r6_root, w, ("bf", key))
        image_left = V2.transformed(torch.as_tensor(left, device=device, dtype=torch.float32)[None], input_transform)
        image_right = V2.transformed(torch.as_tensor(right, device=device, dtype=torch.float32)[None], input_transform)
        positions = torch.as_tensor(V2.support_positions(g, positions_per_window, support), device=device, dtype=torch.float32)
        all_positions.append(positions)
        scenes.append((patches(image_left, positions), patches(image_right, positions)))
    centre_left, centre_right, indices, centre_yx = [], [], [], []
    for (left, right), count, positions in zip(scenes, counts, all_positions):
        chosen = g.choice(len(left), count, replace=False)
        ids = torch.as_tensor(chosen, device=device)
        centre_left.append(left[ids]); centre_right.append(right[ids])
        indices.append([int(i) for i in chosen]); centre_yx.append(positions[ids].detach().cpu().tolist())
    cx, cy = torch.cat(centre_left), torch.cat(centre_right)
    medians, pairs = {}, {}
    for key, values in (("joint", torch.cat((cx, cy), -1)), ("left", cx), ("right", cy)):
        distances = torch.pdist(values.double())
        positive = distances[distances > 0]
        if positive.numel() == 0:
            raise RuntimeError("Degenerate FIT-only kernel bandwidth")
        medians[key] = float(positive.median()); pairs[key] = int(positive.numel())
    spec = {"dtype": "float32", "byte_order": "little-endian", "layout": "C-order", "shape": [int(cx.shape[0]), int(cx.shape[1])]}
    hashes = {"cx_sha256": array_sha256(cx), "cy_sha256": array_sha256(cy), "indices_sha256": json_sha256(indices),
              "centre_yx_sha256": json_sha256(centre_yx), "counts_sha256": json_sha256(list(counts)), "array_spec": spec,
              "fit_patches_sha256": hashlib.sha256(b"".join(hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).digest()
                                                             for sc in scenes for t in sc)).hexdigest()}
    histogram = {str(c): list(counts).count(c) for c in sorted(set(counts))}
    metadata = {"fit_cores": list(windows), "bandwidth_median": medians["joint"],
                "bandwidth_median_left": medians["left"], "bandwidth_median_right": medians["right"],
                "median_pairs": pairs,
                "centres_selection": "paired, per FIT window (EPFL revision 3: floor(T / n) each, the extras by the fixed seed-independent order), "
                                     "without replacement",
                "centres_per_core": None, "centres_per_window_histogram": histogram, "centres": len(cx), "total_centres": len(cx),
                "extra_windows": list(extra_windows) if extra_windows is not None else None, "extra_order_tag": EXTRA_ORDER_TAG,
                "fit_positions_per_core": positions_per_window, "fit_support": support,
                "shg_input": shg_input, "input_transform": input_transform, "stream_tag": stream_tag, "provenance": hashes}
    provenance = {"schema": "v10.epfl.rulsif_v2_centre_provenance.v1", "stream_tag": stream_tag, "extra_order_tag": EXTRA_ORDER_TAG,
                  "total_centres": len(cx), "windows": list(windows), "counts": list(counts),
                  "extra_windows": list(extra_windows) if extra_windows is not None else None,
                  "indices": indices, "centre_yx": centre_yx, "medians": medians, "median_pairs": pairs, **hashes,
                  "notes": "indices index each window's 4096 drawn positions (the order of the draw); centre_yx = (y, x) on the 512 window "
                           "(float32 positions); cx = reference (RGB luminance) patches, cy = moving (linear NIR) patches, both after the "
                           "candidate's input transform; hashes of float32 C-order little-endian bytes"}
    return scenes, cx, cy, medians, metadata, provenance


def epfl_fit_kernel_model(guard, windows, candidate, *, stream_tag, device, purpose, total=None, counts=None):
    """v2's fit_kernel_model on EPFL with the revision-3 allocation of `total` centres (or explicit counts, for the equivalence test).
    Returns (model, scenes, metadata, provenance)."""
    V2, _ = rulsif_v2_module()
    extra = None
    if counts is None:
        if total not in RULSIF_V2_CENTRES:
            raise ValueError(f"total centres must be one of {sorted(RULSIF_V2_CENTRES)}")
        counts = centre_counts(windows, total)
        extra = extra_centre_order(windows)[:total % len(windows)]
    scenes, cx, cy, medians, metadata, provenance = epfl_fit_scenes(
        guard, windows, counts, support=candidate["fit_support"], positions_per_window=candidate["fit_positions_per_core"], stream_tag=stream_tag,
        device=device, purpose=purpose, shg_input=candidate["shg_input"], input_transform=candidate["input_transform"], extra_windows=extra)
    model = V2.kernel_model(cx, cy, medians, constant=True, bandwidth_rule=candidate["bandwidth_rule"],
                            bandwidth_multiplier=candidate["bandwidth_multiplier"], ridge=candidate["ridge"], alpha=candidate["alpha"],
                            penalize_constant=candidate["penalize_constant"], input_transform=candidate["input_transform"])
    metadata.update(constant_basis=True, bandwidth_rule=candidate["bandwidth_rule"], penalize_constant=bool(candidate["penalize_constant"]),
                    bandwidth_left=model.bandwidth_left, bandwidth_right=model.bandwidth_right,
                    kernel_dictionary={"total_centres": len(cx), "role": RULSIF_V2_CENTRES.get(len(cx), "explicit_counts")},
                    centre_rule="EPFL revision 3: T paired centres in total, floor(T / n) per window, the extras by the fixed seed-independent order")
    provenance.update(bandwidth_left=model.bandwidth_left, bandwidth_right=model.bandwidth_right,
                      bandwidth_multiplier=candidate["bandwidth_multiplier"], kernel_dictionary=metadata["kernel_dictionary"])
    return model, scenes, metadata, provenance


def verify_centre_provenance(provenance: dict, resolved: dict, total: int) -> None:
    """Sidecar vs resolved_kernel vs the revision-3 rule (no tensors): counts, extra windows, list hashes, summary hashes."""
    rk = resolved.get("provenance", {})
    if provenance.get("total_centres") != total or resolved.get("total_centres") != total or total not in RULSIF_V2_CENTRES:
        raise RuntimeError("Centre provenance: total centres differ from the run identity")
    windows = provenance["windows"]
    if provenance["counts"] != centre_counts(windows, total) or sum(provenance["counts"]) != total:
        raise RuntimeError("Centre provenance: counts differ from the revision-3 allocation")
    if provenance.get("extra_windows") != extra_centre_order(windows)[:total % len(windows)]:
        raise RuntimeError("Centre provenance: extra windows differ from the fixed order")
    if [len(i) for i in provenance["indices"]] != provenance["counts"] or [len(y) for y in provenance["centre_yx"]] != provenance["counts"]:
        raise RuntimeError("Centre provenance: per-window lists do not match the counts")
    if json_sha256(provenance["indices"]) != provenance["indices_sha256"] or json_sha256(provenance["centre_yx"]) != provenance["centre_yx_sha256"] \
            or json_sha256(provenance["counts"]) != provenance["counts_sha256"]:
        raise RuntimeError("Centre provenance: list hashes differ from their contents")
    for k in ("cx_sha256", "cy_sha256", "indices_sha256", "centre_yx_sha256", "counts_sha256", "fit_patches_sha256"):
        if rk.get(k) != provenance.get(k):
            raise RuntimeError("Centre provenance: resolved_kernel and the sidecar differ in " + k)
    if resolved.get("stream_tag") != provenance.get("stream_tag") or resolved.get("extra_order_tag") != EXTRA_ORDER_TAG:
        raise RuntimeError("Centre provenance: stream tags differ")


def _epfl_kernel_trainer_v2():
    """EPFLKernelTrainerV2: v2's KernelTrainerV2 (checks, modules(), step(): unchanged) with the EPFL FIT windows and the revision-3 centres."""
    V2, _ = rulsif_v2_module()

    class EPFLKernelTrainerV2(V2.KernelTrainerV2):
        def __init__(self, candidate, seed, guard, device, *, windows, tag, total):
            FixedTrainer.__init__(self, V2.FAMILY, candidate, seed, device)
            if candidate.get("constant_basis") is not True or candidate.get("fit_support") != "disk_r129":
                raise ValueError("RuLSIF_exact_v2 requires the registered constant basis and disk support")
            if candidate.get("alpha") != 0.5 or candidate.get("patch_radius") != 4:
                raise ValueError("RuLSIF_exact_v2 keeps alpha = 1/2 and 9 x 9 patches")
            if candidate.get("input_transform") not in V2.INPUT_TRANSFORMS or candidate.get("shg_input") not in V2.SHG_INPUTS:
                raise ValueError("RuLSIF_exact_v2 candidate lacks its registered input transform / SHG input path")
            if candidate.get("bandwidth_rule") not in V2.BANDWIDTH_RULES or not isinstance(candidate.get("penalize_constant"), bool):
                raise ValueError("RuLSIF_exact_v2 candidate lacks its registered bandwidth rule / constant ridge choice")
            if candidate.get("centres_per_core") != 32 or candidate.get("fit_positions_per_core") != 4096:
                raise ValueError("EPFL revisions 2-3 start from the BF definition (32 per core x 32 cores, 4096 positions)")
            if total not in RULSIF_V2_CENTRES:
                raise ValueError(f"total centres must be one of {sorted(RULSIF_V2_CENTRES)} (run identity)")
            self.centres_total = total
            self.ratio, self.scenes, self.fit_metadata, self.provenance = epfl_fit_kernel_model(
                guard, list(windows), candidate, stream_tag=tag, device=self.device, total=total,
                purpose="V10 EPFL RuLSIF_exact_v2 positive patches and exact product moments")
    return EPFLKernelTrainerV2


def production_candidates(family):
    """The BF V10 candidate definitions, unchanged (decision 2026-10-19): BF1 18 (lr ascending, M1 cap ascending within lr) and the
    supplemental ones; RuLSIF_exact_v2 = the frozen grid's amended eligible subset from the merged v2 contract (A1: orders 0-7)."""
    if family == RULSIF_V2:
        _, C = rulsif_v2_module()
        return C.candidates(RULSIF_V2)
    if family not in PRODUCTION_FAMILIES:
        raise ValueError("Not an EPFL production family: " + family)
    return candidates(family)


def production_schedule(family):
    if family == RULSIF_V2:
        _, C = rulsif_v2_module()
        return C.schedule(RULSIF_V2)
    if family not in PRODUCTION_FAMILIES:
        raise ValueError("Not an EPFL production family: " + family)
    return schedule(family)


def rulsif_v2_stream_tag(seed):
    return f"V10|EPFL|{RULSIF_V2}|FIT|{seed}"


def new_trainer(family, candidate, seed, guard, device, *, centres_total=None):
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
    if family == RULSIF_V2:   # v2 on EPFL (revision 3): 4096 disk positions per FIT window, T in {1024, 2048} centres, EPFL stream
        return _epfl_kernel_trainer_v2()(candidate, seed, guard, device, windows=fit_windows(guard), tag=rulsif_v2_stream_tag(seed),
                                         total=centres_total)
    if centres_total is not None:
        raise ValueError("centres_total applies to RuLSIF_exact_v2 only")
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
