"""R6 REG-P2 (scripts/r6_reg_p2.py): rigid warp identities, support validity for every allowed (θ, t) and (θ*, t*), agreement with REG-P1's
energies at the identity, grid-mode vs exact critic energy, CoMIR warp-then-encode vs encode-once, the shared rigid optimiser (oracle
energy; MI on a synthetic pair), readouts."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r6_reg_p2 as Q  # noqa: E402
import r6_reg_pilot as P  # noqa: E402

CPU = torch.device("cpu")


def smooth(seed, sigma=3.0):
    return gaussian_filter(np.random.default_rng(seed).normal(size=(P.SIZE, P.SIZE)), sigma)


def tens(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def frozen_models(seed=0):
    torch.manual_seed(seed); eU, eV, cr = P.build_models(torch.nn)
    for m in (eU, eV, cr):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return eU, eV, cr


class Warp(unittest.TestCase):
    def test_identity_quarter_turn_and_shift(self):
        M = tens(np.random.default_rng(0).normal(size=(2, P.SIZE, P.SIZE)))
        self.assertTrue(torch.allclose(Q.perturb_map(torch, F, M, 0.0, (0.0, 0.0)), M, atol=1e-4))                       # identity = pixel values
        self.assertTrue(torch.allclose(Q.perturb_map(torch, F, M, 90.0, (0.0, 0.0)), torch.rot90(M, 1, (1, 2)), atol=1e-3))   # exact quarter turn
        Mt = Q.perturb_map(torch, F, M, 0.0, (3.0, -5.0)); Mr = torch.roll(M, (3, -5), (1, 2))
        self.assertTrue(torch.allclose(Mt[:, 10:-10, 10:-10], Mr[:, 10:-10, 10:-10], atol=1e-4))                             # V_p(y) = V(y − t*)

    def test_round_trip_and_inverse(self):
        V = tens(smooth(1))[None]; th, t = 23.7, (31.2, -17.9); Vp = Q.perturb_map(torch, F, V, th, t); z = Q.grid_coords(torch, 200, 300, CPU)
        pts = Q.apply_T(torch, torch.tensor([math.radians(th)]), tens([t]), z); back = Q.sample_pts(torch, F, Vp, pts)[0, :, 0]; ref = V[0, 200:301, 200:301].reshape(-1)
        self.assertLess(float((back - ref).abs().mean() / ref.std()), 0.05)                                                 # V_p(T*(z)) = V(z) up to interpolation
        inv = Q.apply_T(torch, torch.tensor([-math.radians(th)]), torch.zeros(1, 2), pts[0] - tens(t))[0]; self.assertTrue(torch.allclose(inv, z, atol=1e-3))


class Support(unittest.TestCase):
    def test_every_sample_inside_for_every_allowed_transform(self):
        self.assertLessEqual(Q.R_OMEGA + Q.RF_BOX * math.sqrt(2) + 2 * Q.TMAX + math.sqrt(2), Q.CEN)                         # the analytic bound
        self.assertGreaterEqual(Q.BOX2[0], 0); self.assertLessEqual(Q.BOX2[1], P.SIZE - 1)
        self.assertLessEqual(Q.BOX1[0] + Q.RF_BOX, Q.CEN - Q.R_OMEGA); self.assertGreaterEqual(Q.BOX1[1] - Q.RF_BOX, Q.CEN + Q.R_OMEGA)
        self.assertLessEqual(Q.BOX2[0] + Q.RF_BOX, Q.CEN - Q.R_OMEGA - Q.TMAX); self.assertGreaterEqual(Q.BOX2[1] - Q.RF_BOX, Q.CEN + Q.R_OMEGA + Q.TMAX)
        ones = torch.ones(1, P.SIZE, P.SIZE)                                     # zero padding: exactly 1 only where every bilinear neighbour is inside V
        z1 = Q.grid_coords(torch, *Q.BOX1, CPU); z1 = z1[(z1 - Q.CEN).norm(dim=1) <= Q.R_OMEGA + Q.RF_BOX * math.sqrt(2)]          # exact critic box (Ω + field)
        z2 = Q.grid_coords(torch, *Q.BOX2, CPU); z2 = z2[(z2 - Q.CEN).norm(dim=1) <= Q.R_OMEGA + Q.RF_BOX * math.sqrt(2) + Q.TMAX]  # grid box
        dirs = [np.array([math.sin(a), math.cos(a)]) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]; g = np.random.default_rng(0)
        cases = [(ths, Q.TMAX * d, th, -Q.TMAX * d2) for ths in (-Q.THMAX, Q.THMAX) for d in dirs for th in (-Q.THMAX, 0.0, Q.THMAX) for d2 in dirs[::2]]
        cases += [(g.uniform(-30, 30), Q.TMAX * dirs[g.integers(8)], g.uniform(-30, 30), Q.TMAX * dirs[g.integers(8)]) for _ in range(20)]
        for ths, ts, th, t in cases:
            Vp = Q.perturb_map(torch, F, ones, ths, tuple(ts), padding="zeros"); r = torch.tensor([math.radians(th)])
            w1 = Q.sample_pts(torch, F, Vp, Q.apply_T(torch, r, tens([t]), z1)); w2 = Q.sample_pts(torch, F, Vp, Q.apply_T(torch, r, torch.zeros(1, 2), z2))
            self.assertTrue(bool((w1 > 1 - 1e-5).all()) and bool((w2 > 1 - 1e-5).all()), (ths, ts, th, t))


class AgreesWithRegP1(unittest.TestCase):
    def test_identity_energies_equal_reg_p1(self):
        U, V = tens(smooth(6))[None], tens(smooth(7))[None]; rng = lambda: np.random.default_rng(3)
        gt = torch.Generator().manual_seed(0); W = torch.randn(162, 64, generator=gt) / 10; b = torch.rand(64, generator=gt) * 6.28; w = torch.randn(64, generator=gt) / 8
        e2 = Q.RigidEnergy(torch, F, "kernel", U, V, rng(), W=W, b=b, w=w); e1 = P.Energy(torch, F, "kernel", U, V, (0.0, 0.0), rng(), W=W, b=b, w=w)
        e1.x, e1.perm = e2.x, e2.perm; e1.u = e1.patches(U, e2.x)
        self.assertAlmostEqual(float(e2.batch(torch.zeros(1), torch.zeros(1, 2))[0]), float(e1(torch.zeros(2))), places=4)
        eU, eV, cr = frozen_models(); Urgb = tens(np.stack([smooth(8), smooth(9), smooth(10)]))
        with torch.no_grad():
            DU, DV = eU(Urgb[None])[0], eV(V[None])[0]
            e2 = Q.RigidEnergy(torch, F, "critic", DU, V, rng(), eV=eV, critic=cr); e1 = P.Energy(torch, F, "critic", DU, DV, (0.0, 0.0), rng(), critic=cr)
            e1.x, e1.perm = e2.x, e2.perm; e1.u = P.sample_map(torch, F, DU, e2.x)
            self.assertAlmostEqual(float(e2.batch(torch.zeros(1), torch.zeros(1, 2))[0]), float(e1(torch.zeros(2))), places=4)   # warp-then-encode = P1 at the identity


class CriticGrid(unittest.TestCase):
    def test_grid_mode_equals_exact_where_both_are_exact(self):
        eU, eV, cr = frozen_models(1); Urgb = tens(np.stack([smooth(11), smooth(12), smooth(13)])); Vp = Q.perturb_map(torch, F, tens(smooth(14))[None], 12.0, (5.0, -7.0))
        with torch.no_grad():
            e = Q.RigidEnergy(torch, F, "critic", eU(Urgb[None])[0], Vp, np.random.default_rng(0), eV=eV, critic=cr)
            for th in (-9.0, 12.0):                                              # t = 0: the two warps take identical samples
                self.assertAlmostEqual(float(e.grid_values(th, torch.zeros(1, 2))[0]), float(e.batch(torch.tensor([math.radians(th)]), torch.zeros(1, 2))[0]), places=4)
            ts = tens([[4.0, -8.0], [12.0, 20.0]])                               # θ = 0, integer shifts: translation equivariance is exact
            self.assertTrue(torch.allclose(e.grid_values(0.0, ts), e.batch(torch.zeros(2), ts), atol=1e-4))
        p = torch.tensor([3.0, 1.0, -2.0], requires_grad=True); e(p).backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0)                               # gradients reach (u, t)


class CoMIRWarpThenEncode(unittest.TestCase):
    def test_identity_agrees_with_encode_once_and_grid_equals_exact(self):
        mA, mB, _ = P.comir_models(torch, 0, CPU); mA.eval(); mB.eval()
        for p in list(mA.parameters()) + list(mB.parameters()):
            p.requires_grad_(False)
        for mod in mA.modules():
            if hasattr(mod, "efficient"):
                mod.efficient = False
        Urgb = tens(np.stack([smooth(15), smooth(16), smooth(17)])); Vp = Q.perturb_map(torch, F, tens(smooth(18))[None], 14.0, (6.0, -3.0)); z = (torch.zeros(1), torch.zeros(1, 2))
        with torch.no_grad():
            LU, LV = mB(Urgb[None])[0], mA(Vp[None])[0]
            once = Q.RigidEnergy(torch, F, "comir", LU, LV, np.random.default_rng(0)); wte = Q.RigidEnergy(torch, F, "comir_wte", LU, Vp, np.random.default_rng(0), mA=mA)
            a, b = float(once.batch(*z)[0]), float(wte.batch(*z)[0]); self.assertLess(abs(a - b), 1e-4 * abs(a) + 1e-7)        # identity: same latents up to interpolation
            self.assertEqual(float(wte.grid_values(-6.0, torch.zeros(1, 2))[0]), float(wte.batch(torch.tensor([math.radians(-6.0)]), torch.zeros(1, 2))[0]))   # t = 0: same warp
        p = torch.tensor([2.0, 1.0, -1.0], requires_grad=True); wte(p).backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0)                               # exact refinement differentiable


class Optimiser(unittest.TestCase):
    def test_recovers_known_rigid_transform_oracle(self):
        th_s, t_s = 17.3, (12.4, -33.1)

        class Oracle:
            def batch(self, th, t):
                return -(((th - math.radians(th_s)) * Q.RHO) ** 2 + ((t - tens(t_s)) ** 2).sum(-1))

            def grid_values(self, th_deg, ts):
                return self.batch(torch.full((len(ts),), math.radians(th_deg)), ts)

            def __call__(self, p):
                return self.batch(p[0:1] / Q.RHO, p[1:3][None])[0]
        th, t, n, _ = Q.optimise_rigid(torch, Oracle(), CPU)
        self.assertLess(abs(th - th_s), 0.05); self.assertLess(float(np.hypot(*(t - np.array(t_s)))), 0.05); self.assertGreater(n, 21 * 400)
        self.assertLess(Q.corner_error_full(th, t, th_s, t_s), 1.0)

    def test_projection_keeps_the_allowed_set(self):
        class Push:                                                              # maximum outside the allowed set
            def batch(self, th, t):
                return th * 100 + t[:, 0] + t[:, 1]

            def grid_values(self, th_deg, ts):
                return self.batch(torch.full((len(ts),), math.radians(th_deg)), ts)

            def __call__(self, p):
                return self.batch(p[0:1] / Q.RHO, p[1:3][None])[0]
        th, t, _, _ = Q.optimise_rigid(torch, Push(), CPU)
        self.assertLessEqual(abs(th), Q.THMAX + 1e-4); self.assertLessEqual(float(np.hypot(*t)), Q.TMAX + 1e-4)

    def test_recovers_rigid_transform_on_images_with_mi(self):
        base = smooth(1); U = tens(base)[None]; V = tens(-base ** 2)[None]                                                # nonlinear relation
        th_s, t_s = -21.4, (17.4, -9.8); Vp = Q.perturb_map(torch, F, V, th_s, t_s); rg = lambda M: (float(M.min()), float(M.max()))
        e = Q.RigidEnergy(torch, F, "mi", U, Vp, np.random.default_rng(2), ranges=(rg(U), rg(Vp))); th, t, _, _ = Q.optimise_rigid(torch, e, CPU)
        self.assertLess(Q.corner_error_full(th, t, th_s, t_s), 9.0)


class Readouts(unittest.TestCase):
    def test_corner_error(self):
        self.assertAlmostEqual(Q.corner_error_full(10.0, (3.0, 4.0), 10.0, (3.0, 4.0)), 0.0, places=9)
        self.assertAlmostEqual(Q.corner_error_full(5.0, (3.0, 4.0), 5.0, (0.0, 0.0)), 5.0 * P.DOWN, places=6)             # translation: every corner moves |δt|
        d = 2.0; want = 2 * math.sin(math.radians(d) / 2) * Q.HALF834 * math.sqrt(2) * P.DOWN
        self.assertAlmostEqual(Q.corner_error_full(d, (0.0, 0.0), 0.0, (0.0, 0.0)), want, places=6)                         # rotation about the patch centre
        self.assertAlmostEqual(2 * Q.HALF834 * P.DOWN, 833.0)

    def test_soft_mi_batch_matches_reg_p1(self):
        g = torch.Generator().manual_seed(0); u = torch.rand(3000, generator=g); v = torch.rand(4, 3000, generator=g); rg = ((0.0, 1.0), (0.0, 1.0))
        for nmi in (False, True):
            b = Q.soft_mi_batch(torch, u, v, rg, nmi=nmi)
            for k in range(4):
                self.assertAlmostEqual(float(b[k]), float(P.soft_mi(torch, u, v[k], rg, nmi=nmi)), places=5)

    def test_perturbations(self):
        p = Q.perturbations_p2("1B_A2"); self.assertEqual(p, Q.perturbations_p2("1B_A2")); a = np.array(p)
        self.assertTrue(np.allclose(a[:, 1:], P.perturbations("1B_A2")))                                                    # REG-P1's translations, same index
        self.assertTrue((np.abs(a[:, 0]) <= Q.THMAX).all()); self.assertNotEqual(Q.perturbations_p2("1B_A3"), p)


if __name__ == "__main__":
    unittest.main()
