"""RuLSIF_exact_v2: a separately registered variant designed after RuLSIF_exact (v1) failed CAL.

v1 (rulsif.py, KernelTrainer, RuLSIFRigidEnergy) is kept unchanged and is reported as a failed
adaptation. v2 keeps the verified v1 algebra -- exact within-scene V (or U) product moments,
H = 1/2 H_P + 1/2 H_Q, the float64 linear solve for h_P, alpha = 1/2, readout
J(g - 1) = 2 E_P g - (E_P g^2 + E_Q g^2)/2 - 1, no clipping of theta, g or T, float32 features and
registration -- and changes only what v10/supplement/RULSIF_V2_PREREGISTRATION.json names, each
from FIT-only evidence (4-fold core cross-fit, seed-1117 streams):

(a) constant basis g = theta_0 + sum_l theta_l a_l(x) b_l(y): the paired "centre" whose two kernels
    are identically one, so kernel_moments() is unchanged; theta_0 is left out of the ridge
    ((H + lambda D) theta = h_P, D = diag(0, 1, ..., 1)), hence E_{(P+Q)/2}[g] = 1 exactly on FIT;
(b) FIT positions on the registration support (radius-129 disk, Q.RigidEnergy law), 4096 per core
    as in one registration energy;
(c) separate per-modality kernel widths h_x, h_y = multiplier x each side's FIT-centre median
    distance (v1's joint median is dominated by SHG, flattening the BF kernel);
(e) a per-image mid-rank empirical-CDF transform of both modalities before patch extraction
    (FIT images, the reference and each moving image as given).
Options that FIT evidence did not support (joint width, square support, log1p SHG, penalised
theta_0, no constant) stay available only for design evidence and the v1-equivalence checks.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from v10 import adapter as A
from .registration import patches
from .rulsif import ExactRuLSIF, gaussian_features, kernel_moments
from .training import FixedTrainer

FAMILY = "RuLSIF_exact_v2"
SUPPORTS = ("disk_r129", "square_v1")


INPUT_TRANSFORMS = ("none", "copula")


def copula_map(image):
    """Per-image empirical CDF (mid-rank, ties averaged) of a [C, H, W] map, values in (0, 1).

    A fixed monotone transform of each whole image (FIT image, reference, or the moving image of
    one registration as given); it removes per-core marginal intensity differences.
    """
    flat = image.reshape(-1)
    ordered = flat.sort().values
    low = torch.searchsorted(ordered, flat, side="left")
    high = torch.searchsorted(ordered, flat, side="right")
    return ((low + high).to(image.dtype) / (2 * flat.numel())).reshape(image.shape)


def transformed(image, transform):
    if transform == "none":
        return image
    if transform == "copula":
        return copula_map(image)
    raise ValueError("Unknown input transform: " + str(transform))


def with_constant(features):
    """Prepend the constant basis column (value 1) to [N, M] kernel features."""
    return torch.cat((features.new_ones(len(features), 1), features), 1)


class ExactRuLSIFV2(ExactRuLSIF):
    """g(x, y) = theta_0 + sum_l theta_l k(x, cx_l; h_x) k(y, cy_l; h_y); coefficient 0 is the constant.

    The constant is the paired "centre" whose two kernels are identically one, so the inherited
    kernel_moments() hold the constant terms exactly (V and U) and common_j()/forward() need no
    new formula. Options (fixed per candidate by the pre-registration record):
      bandwidth_left/right  separate kernel widths per modality (h_x, h_y);
      penalize_constant     True: (H + lambda I) theta = h_P over all coefficients (v1's system);
                            False: (H + lambda D) theta = h_P, D = diag(0, 1, ..., 1), i.e. the ridge
                            shrinks the kernel part towards g = 1 (T = 0), not g towards 0 (T = -1);
      constant              False only for design evidence (v1 class with two widths).
    """

    def __init__(self, centres_left, centres_right, bandwidth_left, bandwidth_right, ridge, alpha=0.5, *,
                 penalize_constant=False, constant=True, input_transform="none"):
        super().__init__(centres_left, centres_right, bandwidth_left, ridge, alpha)
        if input_transform not in INPUT_TRANSFORMS:
            raise ValueError("Unknown input transform")
        self.input_transform = input_transform
        if not math.isfinite(bandwidth_right) or bandwidth_right <= 0:
            raise ValueError("Positive finite bandwidth required")
        self.bandwidth_left, self.bandwidth_right = float(bandwidth_left), float(bandwidth_right)
        self.bandwidth = None  # v1's single width does not apply; use bandwidth_left/right
        self.constant, self.penalize_constant = bool(constant), bool(penalize_constant)
        if self.constant:
            self.coefficients = centres_left.new_zeros(len(centres_left) + 1)

    def _augment(self, features):
        return with_constant(features) if self.constant else features

    def left_features(self, values):
        return self._augment(gaussian_features(values, self.centres_left, self.bandwidth_left))

    def right_features(self, values):
        return self._augment(gaussian_features(values, self.centres_right, self.bandwidth_right))

    def features(self, left, right):
        return self.left_features(left), self.right_features(right)

    @torch.no_grad()
    def fit(self, scenes, *, q_mode="v"):
        """v1's equal-scene-weight fp64 accumulation; ridge matrix lambda I or lambda D."""
        hp = h = None
        count = 0
        for left, right in scenes:
            a, b = self.features(left, right)
            p, p2, _, q2 = kernel_moments(a.double(), b.double(), q_mode)
            local_h = self.alpha * p2 + (1 - self.alpha) * q2
            hp = p.clone() if hp is None else hp + p
            h = local_h.clone() if h is None else h + local_h
            count += 1
        if count == 0:
            raise ValueError("No FIT scenes for RuLSIF")
        hp, h = hp / count, h / count
        penalty = torch.ones(len(hp), device=h.device, dtype=h.dtype)
        if self.constant and not self.penalize_constant:
            penalty[0] = 0
        system = h + self.ridge * torch.diag(penalty)
        raw = torch.linalg.solve(system, hp)
        if not torch.isfinite(raw).all():
            raise RuntimeError("Non-finite RuLSIF coefficients")
        self.coefficients.copy_(raw.to(self.coefficients))
        self.fitted.fill_(True)
        fit_j = float(2 * hp @ raw - raw @ h @ raw - 1)
        info = {"scene_count": count, "q_mode": q_mode, "alpha": self.alpha, "linear_solve_dtype": "float64",
                "negative_coefficients": int((raw < 0).sum()),
                "linear_solve_residual": float(torch.linalg.vector_norm(system @ raw - hp)),
                "coefficient_postprocessing": "none; unconstrained RuLSIF; no pointwise clipping of g",
                "basis": ("constant + " if self.constant else "") + "paired product Gaussians",
                "coefficient_count": len(raw), "ridge_applies_to_constant": self.penalize_constant if self.constant else None,
                "bandwidth_left": self.bandwidth_left, "bandwidth_right": self.bandwidth_right,
                "J_fit_moments": fit_j}
        if self.constant:
            info["constant_coefficient"] = float(raw[0])
            # E_{(P+Q)/2}[g] = 1 holds exactly when theta_0 is unpenalised (normal equation row 0).
            info["mixture_mean_of_g"] = float(h[0] @ raw)
        return info


def support_positions(rng, n, support):
    """FIT positions [n, 2] (y, x) in 512-px coordinates.

    disk_r129: the exact law and draw order of Q.RigidEnergy (r = R sqrt(U), angle U(0, 2 pi)),
               i.e. the support every registration energy evaluates; 9 x 9 patches stay inside.
    square_v1: v1's law, uniform on [5, SIZE - 6) per axis (design evidence / v1 gate only).
    """
    if support == "disk_r129":
        r = A.Q.R_OMEGA * np.sqrt(rng.uniform(0.0, 1.0, n))
        angle = rng.uniform(0.0, 2 * math.pi, n)
        return np.stack([A.Q.CEN + r * np.sin(angle), A.Q.CEN + r * np.cos(angle)], -1)
    if support == "square_v1":
        return rng.uniform(5, A.P.SIZE - 6, (n, 2))
    raise ValueError("Unknown FIT support: " + str(support))


SHG_INPUTS = {"intensity": "shg", "log1p": "shg_log_n"}   # host intensity path / host log1p path


def fit_scenes(guard, cores, *, support, centres_per_core, positions_per_core, stream_tag, device,
               purpose, loader=None, shg_input="intensity", input_transform="none"):
    """FIT patch scenes, paired centres and the FIT-only median distance.

    The draw order follows v1's KernelTrainer exactly: all positions core by core, then the
    centres core by core, then the median over the preselected paired centres. With
    support='square_v1', 4 centres per core, 2048 positions and v1's stream tag it reproduces
    v1's scenes, centres and median bit for bit, so v1 and v2 gates share one code path.
    loader(core, key) -> (bf, shg array) may serve cached arrays; the role check is always made here.
    shg_input: 'intensity' = robust_norm(SHG) (v1, MI/NMI path) or 'log1p' = robust_norm(log1p SHG)
    (the host's learned-method path); BF is always the robust-normalised luminance.
    """
    if len(cores) == 0 or len(set(cores)) != len(cores):
        raise ValueError("FIT cores must be distinct and nonempty")
    if centres_per_core < 1 or centres_per_core > positions_per_core:
        raise ValueError("Invalid centres per core")
    device = torch.device(device)
    g = np.random.default_rng(A.Q.seed_of(stream_tag))
    scenes = []
    for core in cores:
        guard.check(core, {"FIT"}, purpose)
        key = SHG_INPUTS[shg_input]
        left, right = (loader(core, key) if loader is not None
                       else A.P.load_img(guard.r6_root, core, ("bf", key)))
        image_left = transformed(torch.as_tensor(left, device=device, dtype=torch.float32)[None], input_transform)
        image_right = transformed(torch.as_tensor(right, device=device, dtype=torch.float32)[None], input_transform)
        positions = torch.as_tensor(support_positions(g, positions_per_core, support),
                                    device=device, dtype=torch.float32)
        scenes.append((patches(image_left, positions), patches(image_right, positions)))
    centre_left, centre_right = [], []
    for left, right in scenes:
        ids = torch.as_tensor(g.choice(len(left), centres_per_core, replace=False), device=device)
        centre_left.append(left[ids]); centre_right.append(right[ids])
    cx, cy = torch.cat(centre_left), torch.cat(centre_right)
    medians = {}
    for key, values in (("joint", torch.cat((cx, cy), -1)), ("left", cx), ("right", cy)):
        distances = torch.pdist(values.double())
        positive = distances[distances > 0]
        if positive.numel() == 0:
            raise RuntimeError("Degenerate FIT-only kernel bandwidth")
        medians[key] = float(positive.median())
    metadata = {"fit_cores": list(cores), "bandwidth_median": medians["joint"],
                "bandwidth_median_left": medians["left"], "bandwidth_median_right": medians["right"],
                "centres_selection": "paired, equal count per FIT core, without replacement",
                "centres_per_core": centres_per_core, "centres": len(cx),
                "fit_positions_per_core": positions_per_core, "fit_support": support,
                "shg_input": shg_input, "input_transform": input_transform, "stream_tag": stream_tag}
    return scenes, cx, cy, medians, metadata


BANDWIDTH_RULES = ("joint_median", "per_modality_median")


def kernel_model(cx, cy, medians, *, constant, bandwidth_rule, bandwidth_multiplier, ridge, alpha,
                 penalize_constant=False, input_transform="none"):
    """joint_median: v1's single width (median joint distance); per_modality_median: h_x and h_y
    from each side's own centre distances. v1's class is returned for (no constant, joint)."""
    if bandwidth_rule == "joint_median":
        left = right = medians["joint"] * bandwidth_multiplier
    elif bandwidth_rule == "per_modality_median":
        left, right = medians["left"] * bandwidth_multiplier, medians["right"] * bandwidth_multiplier
    else:
        raise ValueError("Unknown bandwidth rule: " + str(bandwidth_rule))
    if not constant and bandwidth_rule == "joint_median" and input_transform == "none":
        return ExactRuLSIF(cx, cy, left, ridge, alpha)
    return ExactRuLSIFV2(cx, cy, left, right, ridge, alpha, penalize_constant=penalize_constant,
                         constant=constant, input_transform=input_transform)


def fit_kernel_model(guard, cores, *, constant, support, centres_per_core, positions_per_core,
                     bandwidth_rule, bandwidth_multiplier, ridge, alpha, stream_tag, device, purpose,
                     penalize_constant=False, loader=None, shg_input="intensity", input_transform="none"):
    """Unfitted model, its FIT scenes and metadata (fit with model.fit(scenes, q_mode='v'))."""
    scenes, cx, cy, medians, metadata = fit_scenes(
        guard, cores, support=support, centres_per_core=centres_per_core,
        positions_per_core=positions_per_core, stream_tag=stream_tag, device=device,
        purpose=purpose, loader=loader, shg_input=shg_input, input_transform=input_transform)
    model = kernel_model(cx, cy, medians, constant=constant, bandwidth_rule=bandwidth_rule,
                         bandwidth_multiplier=bandwidth_multiplier, ridge=ridge, alpha=alpha,
                         penalize_constant=penalize_constant, input_transform=input_transform)
    metadata.update(constant_basis=bool(constant), bandwidth_rule=bandwidth_rule,
                    penalize_constant=bool(penalize_constant) if constant else None,
                    bandwidth_left=getattr(model, "bandwidth_left", model.bandwidth),
                    bandwidth_right=getattr(model, "bandwidth_right", model.bandwidth))
    return model, scenes, metadata


def moving_input(candidate, intensity, learned):
    """Moving SHG map matching the FIT input: host intensity path or host log1p path (both [1, S, S])."""
    if candidate["shg_input"] == "intensity":
        return intensity
    if candidate["shg_input"] == "log1p":
        return learned
    raise ValueError("Unknown SHG input path")


def stream_tag(seed):
    return f"V10|{FAMILY}|FIT|{seed}"


class KernelTrainerV2(FixedTrainer):
    """One complete analytic FIT32 fit; checkpoint/registration interfaces as v1's KernelTrainer."""
    kernel_v2 = True

    def __init__(self, candidate, seed, guard, device, *, cores=None, tag=None, loader=None):
        super().__init__(FAMILY, candidate, seed, device)
        if candidate.get("constant_basis") is not True or candidate.get("fit_support") != "disk_r129":
            raise ValueError("RuLSIF_exact_v2 requires the registered constant basis and disk support")
        if candidate.get("alpha") != 0.5 or candidate.get("patch_radius") != 4:
            raise ValueError("RuLSIF_exact_v2 keeps alpha = 1/2 and 9 x 9 patches")
        if cores is None:
            cores = A.fit_cores(guard, 1.0, seed)
            if len(cores) != 32:
                raise RuntimeError("Kernel fitting requires the full frozen FIT32")
        if candidate.get("input_transform") not in INPUT_TRANSFORMS:
            raise ValueError("RuLSIF_exact_v2 candidate lacks its registered input transform")
        if candidate.get("shg_input") not in SHG_INPUTS:
            raise ValueError("RuLSIF_exact_v2 candidate lacks its registered SHG input path")
        if candidate.get("bandwidth_rule") not in BANDWIDTH_RULES or not isinstance(candidate.get("penalize_constant"), bool):
            raise ValueError("RuLSIF_exact_v2 candidate lacks its registered bandwidth rule / constant ridge choice")
        self.ratio, self.scenes, self.fit_metadata = fit_kernel_model(
            guard, list(cores), constant=True, support=candidate["fit_support"],
            centres_per_core=candidate["centres_per_core"],
            positions_per_core=candidate["fit_positions_per_core"],
            bandwidth_rule=candidate["bandwidth_rule"], bandwidth_multiplier=candidate["bandwidth_multiplier"],
            ridge=candidate["ridge"], alpha=candidate["alpha"], penalize_constant=candidate["penalize_constant"],
            stream_tag=tag or stream_tag(seed), device=self.device, shg_input=candidate["shg_input"],
            input_transform=candidate["input_transform"],
            purpose="V10 RuLSIF_exact_v2 positive patches and exact product moments", loader=loader)

    def modules(self):
        return (self.ratio,)

    def step(self):
        if self.n_steps != 0:
            raise RuntimeError("Kernel fit is one complete analytic fit")
        info = self.ratio.fit(self.scenes, q_mode="v")
        self.n_steps = 1
        return {"step": 1, "kind": "kernel_fit", **self.fit_metadata, **info}


class RuLSIFV2RigidEnergy(A.Q.RigidEnergy):
    """v1's warp-then-extract energy on the augmented (constant + kernel) features.

    Same positions/permutation RNG consumption as every method (host kind 'comir'), same 9 x 9
    offsets, same per-candidate loop and blocks of four; exact V moments of g - 1.
    """

    def __init__(self, reference, moving, rng, *, ratio):
        if not isinstance(ratio, ExactRuLSIFV2):
            raise TypeError("RuLSIF_exact_v2 energy needs the constant-basis model")
        A.assert_frozen(ratio)
        if not bool(ratio.fitted):
            raise ValueError("Unfitted RuLSIF model")
        reference = transformed(reference, ratio.input_transform)
        moving = transformed(moving, ratio.input_transform)
        super().__init__(torch, F, "comir", reference, moving, rng)
        self.ratio = ratio
        self.left = ratio.left_features(patches(reference, self.x))
        w = ratio.coefficients
        self.left_weighted = self.left * w
        self.q_quadratic = (self.left.T @ self.left / len(self.x)) * (w[:, None] * w[None, :])
        axis = torch.arange(-4, 5, device=self.x.device, dtype=self.x.dtype)
        dy, dx = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((dy.flatten(), dx.flatten()), -1)
        self.patch_positions = (self.x[:, None] + offsets[None]).reshape(-1, 2)

    def batch(self, theta, translation):
        rows = []
        for k in range(len(theta)):
            coords = A.Q.apply_T(torch, theta[k:k + 1], translation[k:k + 1], self.patch_positions)
            values = A.Q.sample_pts(torch, F, self.mov, coords)[0, :, 0].reshape(len(self.x), 81)
            right = self.ratio.right_features(values)
            gp = (self.left_weighted * right).sum(-1)
            gq2 = ((right @ self.q_quadratic) * right).sum(-1).mean()
            # alpha = 1/2: J(g - 1) = 2 E_P g - (E_P g^2 + E_Q g^2)/2 - 1, constant included in g.
            rows.append(2 * gp.mean() - 0.5 * (gp.square().mean() + gq2) - 1)
        return torch.stack(rows)

    def grid_values(self, theta_degrees, translations):
        theta = torch.full((1,), math.radians(theta_degrees), device=translations.device,
                           dtype=translations.dtype)
        return torch.cat([self.batch(theta.expand(len(block)), block) for block in translations.split(4)])
