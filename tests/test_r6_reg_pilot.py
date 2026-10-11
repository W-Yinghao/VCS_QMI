"""R6 REG-P1 pilot (scripts/r6_reg_pilot.py): TIFF reader, P / Q warping, support / leakage guards, shared optimiser, readouts, roles."""
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r6_reg_pilot as P  # noqa: E402


def write_tiff(path, a):
    """Minimal uncompressed little-endian TIFF (chunky, one strip), the layout of the BF/SHG files: pixels first, IFD at the end."""
    h, w = a.shape[:2]; spp = 1 if a.ndim == 2 else a.shape[2]; data = a.astype(np.uint8).tobytes(); off_ifd = 8 + len(data)
    tags = [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 2 if spp == 3 else 1), (273, 4, 1, 8), (277, 3, 1, spp),
            (278, 4, 1, h), (279, 4, 1, len(data)), (284, 3, 1, 1)]
    ifd = struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHII", t, ty, c, v) if ty == 4 else struct.pack("<HHIHH", t, ty, c, v, 0) for t, ty, c, v in tags) + struct.pack("<I", 0)
    Path(path).write_bytes(b"II*\x00" + struct.pack("<I", off_ifd) + data + ifd)


class Reader(unittest.TestCase):
    def test_rgb_and_grey(self):
        g = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            for shape in ((5, 7, 3), (6, 4)):
                a = g.integers(0, 256, shape).astype(np.uint8); p = Path(d) / "x.tif"; write_tiff(p, a)
                self.assertTrue(np.array_equal(P.read_tiff(p), a))


class Warping(unittest.TestCase):
    def test_pixel_centres_and_shift(self):
        g = torch.Generator().manual_seed(0); M = torch.randn(2, 40, 40, generator=g)
        yx = torch.tensor([[5.0, 7.0], [20.0, 33.0]]); s = P.sample_map(torch, F, M, yx)
        self.assertTrue(torch.allclose(s, M[:, [5, 20], [7, 33]].T, atol=1e-6))                     # integer positions = pixel values
        Ms = torch.roll(M, shifts=(3, -2), dims=(1, 2)); x = torch.tensor([[10.5, 12.25], [15.0, 20.75]])
        self.assertTrue(torch.allclose(P.sample_map(torch, F, Ms, x), P.sample_map(torch, F, M, x - torch.tensor([3.0, -2.0])), atol=1e-5))

    def test_support_inside_for_every_allowed_shift(self):
        lo, hi = P.OMEGA; worst = 2 * P.TMAX
        self.assertGreaterEqual(lo - worst - 4, 0); self.assertLessEqual(hi + worst + 4 + 1, P.SIZE)      # 9 × 9 kernel patches included
        self.assertGreaterEqual(lo - worst, P.RF_HALF); self.assertLessEqual(hi + worst + 1, P.SIZE - P.RF_HALF)   # descriptor receptive field

    def test_observer_sees_images_only(self):
        eU, eV, cr = P.build_models(torch.nn)
        self.assertEqual(eU[0].in_channels, 3); self.assertEqual(eV[0].in_channels, 1); self.assertEqual(cr[0].in_features, 32)   # BF RGB, SHG log1p (as CoMIR)
        rf = 1 + sum(m.kernel_size[0] - 1 for m in eU if isinstance(m, torch.nn.Conv2d)); self.assertEqual(rf, 2 * P.RF_HALF + 1)


class Optimiser(unittest.TestCase):
    def test_recovers_known_shift(self):
        t = np.array([13.3, -27.6])
        energy = lambda phi: -((phi - torch.tensor(t, dtype=torch.float32)) ** 2).sum()
        phi, n = P.optimise(torch, energy, torch.device("cpu")); self.assertLess(float(np.hypot(*(phi - t))), 0.05); self.assertGreater(n, 400)

    def test_recovers_shift_on_images(self):
        g = np.random.default_rng(1); base = g.normal(size=(P.SIZE, P.SIZE)); from scipy.ndimage import gaussian_filter
        U = torch.tensor(gaussian_filter(base, 3), dtype=torch.float32)[None]; V = torch.tensor(-gaussian_filter(base, 3) ** 2, dtype=torch.float32)[None]   # nonlinear relation
        t = (17.4, -9.8); e = P.Energy(torch, F, "mi", U, V, t, np.random.default_rng(2), ranges=((float(U.min()), float(U.max())), (float(V.min()), float(V.max()))))
        phi, _ = P.optimise(torch, e, torch.device("cpu")); self.assertLess(float(np.hypot(*(phi - np.array(t)))), 1.0)


class Readouts(unittest.TestCase):
    def test_j_formula(self):
        tp, tq = torch.tensor([0.5, -0.2]), torch.tensor([0.1, -0.9])
        want = ((tp - tp ** 2 / 2).mean() + (-tq - tq ** 2 / 2).mean()); self.assertAlmostEqual(float(P.j_value(torch, tp, tq)), float(want), places=6)

    def test_soft_mi(self):
        g = torch.Generator().manual_seed(0); u = torch.rand(4000, generator=g); v = torch.rand(4000, generator=g); rg = ((0.0, 1.0), (0.0, 1.0))
        self.assertGreater(float(P.soft_mi(torch, u, u, rg)), float(P.soft_mi(torch, u, v, rg)) + 1.0)
        n = float(P.soft_mi(torch, u, u, rg, nmi=True)); self.assertTrue(1.0 <= n <= 2.0 + 1e-6)


class Roles(unittest.TestCase):
    def test_split_and_perturbations(self):
        cores = {f"c{i}": ("train", None, None) for i in range(40)}; cores.update({f"v{i}": ("validation1", None, None) for i in range(25)})
        r1, r2 = P.roles(cores), P.roles(cores); self.assertEqual(r1, r2)
        self.assertEqual(sum(v == "CAL" for v in r1.values()), 8); self.assertEqual(sum(v == "FIT" for v in r1.values()), 32); self.assertEqual(sum(v == "DEV" for v in r1.values()), 25)
        p = P.perturbations("1B_A2"); self.assertTrue(np.array_equal(p, P.perturbations("1B_A2"))); mag = np.hypot(p[:, 0], p[:, 1])
        self.assertTrue((mag <= P.TMAX + 1e-9).all())
        for i, m in enumerate(mag):
            lo, hi = P.STRATA[i % 3]; self.assertTrue(lo - 1e-9 <= m <= hi + 1e-9)


if __name__ == "__main__":
    unittest.main()
