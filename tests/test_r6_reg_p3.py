"""R6 REG-P3 (scripts/r6_reg_p3.py, scripts/r6_reg_p3_aggregate.py).  EPFL RGB-NIR (the pilot): the base-alignment check and its exclusion
rule, the near-duplicate links, scene units, the per-category role rule, centred windows, an end-to-end prepare on a synthetic EPFL-like folder,
and the data facts of the real copy (exclusion lists, links, units, roles) that the protocol states.  Shared machinery: the smoke subset (no DEV
window), REG-P2's perturbation family per window, the moving-image path, end-to-end registrations through REG-P2's machinery on a synthetic root
(reduced grid), unit pooling and the readings.  Zurich (kept for the record, not run): GeoTIFF footprints, adjacency units, roles, windows, band
order and its data facts.  The protocol file is parsed and checked against the code."""
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r6_reg_p2 as Q  # noqa: E402
import r6_reg_p3 as R  # noqa: E402
import r6_reg_p3_aggregate as A  # noqa: E402
import r6_reg_pilot as P  # noqa: E402

CPU = torch.device("cpu")


def write_tiff8(path, a):
    """Minimal uncompressed little-endian 8-bit TIFF (chunky, one strip; RGB or grey), the layout of the EPFL files."""
    h, w = a.shape[:2]; spp = 1 if a.ndim == 2 else a.shape[2]; data = a.astype(np.uint8).tobytes(); bps = struct.pack("<3H", 8, 8, 8); o_bps = 8 + len(data)
    o_ifd = o_bps + len(bps); bps_tag = (258, 3, 3, o_bps) if spp == 3 else (258, 3, 1, 8)
    tags = [(256, 4, 1, w), (257, 4, 1, h), bps_tag, (259, 3, 1, 1), (262, 3, 1, 2 if spp == 3 else 1), (273, 4, 1, 8), (277, 3, 1, spp), (278, 4, 1, h),
            (279, 4, 1, len(data)), (284, 3, 1, 1)]
    ifd = struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHIHH", t, ty, c, v, 0) if ty == 3 and c == 1 else struct.pack("<HHII", t, ty, c, v) for t, ty, c, v in tags)
    Path(path).write_bytes(b"II*\x00" + struct.pack("<I", o_ifd) + data + bps + ifd + struct.pack("<I", 0))


def write_geotiff(path, a, scale=(0.6, 0.6), tie=(0.0, 0.0, 0.0, 1000.0, 5000.0, 0.0)):
    """Minimal uncompressed little-endian 4-band uint16 GeoTIFF (chunky, one strip) with ModelPixelScale / ModelTiepoint, as the Zurich tiles."""
    h, w, spp = a.shape; data = a.astype("<u2").tobytes(); bps = struct.pack("<4H", 16, 16, 16, 16); sc = struct.pack("<3d", scale[0], scale[1], 0.0)
    tp = struct.pack("<6d", *tie); o_bps = 8 + len(data); o_sc = o_bps + len(bps); o_tp = o_sc + len(sc); o_ifd = o_tp + len(tp)
    tags = [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 4, o_bps), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 8), (277, 3, 1, spp), (278, 4, 1, h),
            (279, 4, 1, len(data)), (284, 3, 1, 1), (33550, 12, 3, o_sc), (33922, 12, 6, o_tp)]
    ifd = struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHIHH", t, ty, c, v, 0) if ty == 3 and c == 1 else struct.pack("<HHII", t, ty, c, v) for t, ty, c, v in tags)
    Path(path).write_bytes(b"II*\x00" + struct.pack("<I", o_ifd) + data + bps + sc + tp + ifd + struct.pack("<I", 0))


def synth_scene(seed, H=600, W=700, shift=(0, 0)):
    """8-bit RGB scene (smooth texture) and its NIR: a non-monotone function of the luminance, optionally displaced by `shift` native px."""
    g = np.random.default_rng(seed); b = gaussian_filter(g.normal(size=(H, W)), 3); b = (b - b.min()) / (b.max() - b.min())
    rgb = np.stack([b, 0.8 * b + 0.1, 0.6 * b + 0.2], -1) * 255; L = R.lum(rgb) / 255
    nir = 255 * np.abs(np.roll(L, shift, (0, 1)) - 0.45) * 1.6; return np.clip(rgb, 0, 255).astype(np.uint8), np.clip(nir, 0, 255).astype(np.uint8)


def synth_tile(seed, H=512, W=512):
    """Zurich-like B, G, R, NIR uint16 tile: smooth visible bands, NIR a non-monotone function of the red band plus its own smooth part."""
    g = np.random.default_rng(seed); base = gaussian_filter(g.normal(size=(H, W)), 4); base = base / base.std()
    own = gaussian_filter(g.normal(size=(H, W)), 6); own = own / own.std(); red = 1500 + 400 * base
    nir = 3200 - 0.004 * (red - 1500) ** 2 * 6 + 120 * own
    t = np.stack([0.8 * red + 50, 0.9 * red + 20, red, nir], -1); return np.clip(t, 0, 4095).astype(np.uint16)


# ------------------------------------------------------------------------------------------------------------ EPFL (the pilot)
class EpflChecks(unittest.TestCase):
    def test_alignment_check_finds_known_shifts_and_the_rule(self):
        for shift, want in (((0, 0), (0, 0)), ((2, 0), (1, 0)), ((4, -6), (2, -3)), ((-6, 2), (-3, 1))):
            rgb, nir = synth_scene(3, shift=shift); c = R.alignment_check(rgb, nir)
            self.assertEqual(tuple(c["best_shift_half_res"]), want, shift); self.assertEqual(R.misaligned(c), max(map(abs, want)) >= 2)
            self.assertGreaterEqual(c["gradcorr_best"], c["gradcorr_zero"] - 1e-12)
        self.assertEqual(c["thumb"].shape, ((600 // 2) // 16 * ((700 // 2) // 16),))

    def test_near_duplicate_links(self):
        t = lambda s, n=40: R.zscore(np.random.default_rng(s).normal(size=n))
        th = {"a/0001": t(1), "a/0002": t(1), "a/0003": t(2), "b/0001": t(1), "b/0002": t(3, 30), "a/0004": t(4)}
        th["a/0004"] = R.zscore(0.9 * th["a/0003"] + 0.1 * t(5))                                                # correlated, > 0.9
        links = R.near_duplicate_links(th); pairs = [(x, y) for x, y, _ in links]
        self.assertEqual(pairs, [("a/0001", "a/0002"), ("a/0003", "a/0004")])                                   # never across categories
        self.assertTrue(all(c > 0.9 for _, _, c in links))

    def test_units_merge_through_excluded_pairs(self):
        u = R.units_epfl(["a/0001", "a/0003", "a/0009"], [["a/0001", "a/0002", 0.95], ["a/0002", "a/0003", 0.93]])   # a/0002 excluded
        self.assertEqual(u, {"a/0001": "a/0001+a/0003", "a/0003": "a/0001+a/0003", "a/0009": "a/0009"})

    def test_roles_per_category(self):
        uc = {f"a/{i:04d}": "a" for i in range(12)}; uc.update({f"b/{i:04d}": "b" for i in range(5)}); r = R.roles_epfl(uc)
        self.assertEqual(r, R.roles_epfl(dict(reversed(list(uc.items())))))
        cnt = lambda cat, role: sum(1 for u, v in r.items() if uc[u] == cat and v == role)
        self.assertEqual((cnt("a", "DEV"), cnt("a", "CAL"), cnt("a", "FIT")), (8, 1, 3)); self.assertEqual((cnt("b", "DEV"), cnt("b", "CAL")), (5, 0))

    def test_centred_window(self):
        self.assertEqual(R.centred_window(680, 1024), (84, 256)); self.assertEqual(R.centred_window(512, 512), (0, 0))
        with self.assertRaises(ValueError):
            R.centred_window(768, 510)

    def test_prepare_on_a_synthetic_folder(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "nirscene1"
            for cat in ("catA", "catB"):
                (data / cat).mkdir(parents=True)
                for i in range(12):
                    W = 500 if (cat, i) == ("catB", 3) else 700; seed = 100 + (4 if (cat, i) == ("catA", 5) else i) + (50 if cat == "catB" else 0)
                    rgb, nir = synth_scene(seed, W=W, shift=(6, 0) if (cat, i) == ("catA", 7) else (0, 0))
                    write_tiff8(data / cat / f"{i:04d}_rgb.tiff", rgb); write_tiff8(data / cat / f"{i:04d}_nir.tiff", nir)
            a = type("A", (), {"data": data, "root": Path(d) / "root", "smoke": False})(); a.root.mkdir(); R.prepare_epfl(a)
            m = json.loads((a.root / "manifest.json").read_text())
            self.assertEqual(m["excluded_alignment"], ["catA/0007"]); self.assertEqual(m["excluded_size"], ["catB/0003"])
            self.assertEqual([x[:2] for x in m["near_duplicate_links"]], [["catA/0004", "catA/0005"]])               # a copy of a scene is linked
            self.assertEqual(m["windows"]["catA_0005"]["unit"], "catA/0004+catA/0005"); self.assertEqual(len(m["units"]), 21)
            for cat, fit in (("catA", 1), ("catB", 2)):
                roles = [v["role"] for u, v in m["units"].items() if v["category"] == cat]
                self.assertEqual((roles.count("DEV"), roles.count("CAL"), roles.count("FIT")), (8, 1, fit))
            self.assertEqual(sorted(p.stem for p in (a.root / "cache").glob("*.npz")), sorted(m["roles"])); self.assertEqual(len(m["roles"]), 22)
            z = np.load(a.root / "cache" / "catB_0005.npz"); self.assertEqual(sorted(z.files), ["bf", "bf_rgb_n", "nir", "shg", "shg_log_n"])
            rgb, nir = synth_scene(155, W=700); y0, x0 = R.centred_window(600, 700)                                 # catB/0005: seed 100 + 5 + 50
            self.assertTrue(np.allclose(z["nir"], nir[y0:y0 + 512, x0:x0 + 512] / 255.0, atol=1e-6)); self.assertTrue(np.array_equal(z["shg"], z["shg_log_n"]))
            self.assertTrue(np.allclose(z["bf_rgb_n"], P.robust_norm(rgb[y0:y0 + 512, x0:x0 + 512] / 255.0), atol=1e-5))
            self.assertEqual(m["pairs"]["catA/0007"]["status"], "excluded_alignment"); self.assertEqual(m["pairs"]["catA/0007"]["best_shift_half_res"], [3, 0])


@unittest.skipUnless(R.EPFL3.exists(), "EPFL copy not available")
class EpflDataFacts(unittest.TestCase):
    """The protocol's EPFL statements, recomputed from the files: 477 pairs, the 33 alignment and 24 size exclusions, the 13 near-duplicate links,
    411 units and the per-category roles (no registration)."""

    @classmethod
    def setUpClass(cls):
        cls.pairs = R.epfl_pairs(); cls.chk, cls.shape = {}, {}
        for k, (pr, pn) in cls.pairs.items():
            rgb, nir = P.read_tiff(pr), P.read_tiff(pn); cls.chk[k] = R.alignment_check(rgb, nir); cls.shape[k] = nir.shape
        cls.mis = sorted(k for k, c in cls.chk.items() if R.misaligned(c)); cls.small = sorted(k for k in cls.pairs if k not in cls.mis and min(cls.shape[k]) < R.WIN)
        cls.links = R.near_duplicate_links({k: c["thumb"] for k, c in cls.chk.items()}); kept = sorted(set(cls.pairs) - set(cls.mis) - set(cls.small))
        cls.unit = R.units_epfl(kept, cls.links); cls.role = R.roles_epfl({u: u.split("/")[0] for u in set(cls.unit.values())})

    def test_exclusions_and_links(self):
        self.assertEqual(len(self.pairs), 477); self.assertEqual(self.mis, EXPECTED_EPFL_MISALIGNED); self.assertEqual(self.small, EXPECTED_EPFL_SMALL)
        self.assertEqual([tuple(x[:2]) for x in self.links], EXPECTED_EPFL_LINKS); self.assertTrue(all(len(s) == 2 for s in self.shape.values()))

    def test_units_and_roles(self):
        self.assertEqual(len(self.unit), 420); self.assertEqual(len(self.role), 411)
        self.assertEqual({r: sum(v == r for v in self.role.values()) for r in ("DEV", "CAL", "FIT")}, {"DEV": 72, "CAL": 9, "FIT": 330})
        self.assertEqual(sorted(u for u, r in self.role.items() if r == "CAL"), EXPECTED_EPFL_CAL)
        self.assertEqual(sum(1 for k in self.unit if self.role[self.unit[k]] == "DEV"), EXPECTED_EPFL_DEV_WINDOWS)


# ------------------------------------------------------------------------------------------------------------ shared machinery
class SmokeAndPerturbations(unittest.TestCase):
    def test_smoke_subset_reads_no_dev_window(self):
        roles = {"zh1_w00": "FIT", "zh1_w01": "FIT", "zh2_w00": "FIT", "zh3_w00": "CAL", "zh3_w01": "CAL", "zh4_w00": "CAL", "zh4_w01": "CAL", "zh5_w00": "DEV"}
        wins = {w: {"unit": w.split("_")[0]} for w in roles}; s = R.smoke_subset(roles, wins)
        self.assertTrue(all(roles[w] in ("FIT", "CAL") for w in s)); self.assertEqual(sorted(w for w, r in s.items() if r == "DEV"), ["zh4_w00", "zh4_w01"])
        self.assertEqual(sorted(w for w, r in s.items() if r == "FIT"), ["zh1_w00", "zh1_w01", "zh2_w00"])           # the first 4 FIT windows

    def test_profile_windows(self):
        m = {"windows": {f"{c}_{i:04d}": {"category": c, "unit": f"{c}/{i:04d}"} for c in ("a", "b", "c", "d", "e", "f") for i in range(3)}}
        self.assertEqual(R.profile_windows(m, sorted(m["windows"])), ["a_0000", "b_0000", "c_0000", "d_0000", "e_0000"])   # one per category
        m2 = {"windows": {"zh3_w00": {"unit": "zh3+zh4"}, "zh4_w00": {"unit": "zh3+zh4"}, "zh5_w00": {"unit": "zh5"}}}
        self.assertEqual(R.profile_windows(m2, ["zh5_w00", "zh4_w00", "zh3_w00"]), ["zh3_w00", "zh5_w00"])            # one per unit

    def test_perturbation_family_per_window(self):
        p = Q.perturbations_p2("country_0001"); a = np.array(p); self.assertEqual(p, Q.perturbations_p2("country_0001")); self.assertNotEqual(p, Q.perturbations_p2("country_0002"))
        self.assertTrue((np.abs(a[:, 0]) <= 30.0).all()); self.assertTrue(np.allclose(a[:, 1:], P.perturbations("country_0001")))
        for i, (_, ty, tx) in enumerate(p):
            lo, hi = P.STRATA[i % 3]; self.assertTrue(lo - 1e-9 <= math.hypot(ty, tx) <= hi + 1e-9); self.assertEqual(A.stratum({"t": [ty, tx]}), i % 3)


def make_root(d, smoke=True, pert=None):
    """A synthetic P3 root: FIT unit zh1 (2 windows), CAL unit zh2 (1 window), DEV unit zh3+zh4 (windows of two tiles)."""
    root = Path(d) / ("smoke" if smoke else ""); (root / "cache").mkdir(parents=True); roles = {"zh1_w00": "FIT", "zh1_w01": "FIT", "zh2_w00": "CAL", "zh3_w00": "DEV", "zh4_w00": "DEV"}
    unit = {"zh1": "zh1", "zh2": "zh2", "zh3": "zh3+zh4", "zh4": "zh3+zh4"}; wins = {}
    for k, w in enumerate(sorted(roles)):
        t = w.split("_")[0]; np.savez_compressed(root / "cache" / f"{w}.npz", **R.window_inputs(synth_tile(10 + k), 0, 0)); wins[w] = {"tile": t, "unit": unit[t], "y0": 0, "x0": 0}
    man = {"dataset": "synthetic", "roles": roles, "windows": wins, "gsd_m": None}
    sfx = "_smoke" if smoke else ""; (root / f"manifest{sfx}.json").write_text(json.dumps(man))
    if pert is not None:
        (root / f"perturbations{sfx}.json").write_text(json.dumps(pert))
    return root


def args(root, method, seed=0, n_pert=1, smoke=True):
    return type("A", (), {"root": root, "smoke": smoke, "method": method, "seed": seed, "n_pert": n_pert})()


FAST = dict(N_POS=1024, GRID_STEP=12.0, N_ADAM=40)                                  # reduced grid / samples for CPU tests (restored afterwards)


class MovingAndEndToEnd(unittest.TestCase):
    def test_moving_identity_and_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_root(d); pert = {"zh2_w00": [[0.0, 0.0, 0.0], [17.0, 12.0, -20.0]]}; I = R.Inputs3(torch, F, root, pert, CPU)
            nir = P.load_img(root, "zh2_w00", ("nir",))[0]; m0 = I.moving("zh2_w00", 0)
            self.assertTrue(torch.allclose(m0[0][0], torch.tensor(P.robust_norm(nir.astype(np.float64))), atol=1e-4)); self.assertIs(m0[0], m0[1])
            V = torch.tensor(nir)[None]; Vp = Q.perturb_map(torch, F, V, 17.0, (12.0, -20.0)); z = Q.grid_coords(torch, 180, 330, CPU)
            back = Q.sample_pts(torch, F, Vp, Q.apply_T(torch, torch.tensor([math.radians(17.0)]), torch.tensor([[12.0, -20.0]]), z))[0, :, 0]
            ref = V[0, 180:331, 180:331].reshape(-1); self.assertLess(float((back - ref).abs().mean() / ref.std()), 0.05)   # V_p(T*(z)) = V(z)
            self.assertTrue(np.allclose(I.reference("zh2_w00")[0][0].numpy(), P.load_img(root, "zh2_w00", ("bf",))[0]))

    def test_perturb_writes_cal_dev_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_root(d); R.cmd_perturb(args(root, "mi")); pert = json.loads((root / "perturbations_smoke.json").read_text())
            self.assertEqual(sorted(pert), ["zh2_w00", "zh3_w00", "zh4_w00"]); self.assertEqual(pert["zh3_w00"], Q.perturbations_p2("zh3_w00"))

    def test_mi_registration_recovers_grid_aligned_transforms(self):
        pert = {w: [[10.0, 9.0, -3.0]] for w in ("zh2_w00", "zh3_w00", "zh4_w00")}                     # on the reduced grid: θ 10°, t (9, −3)
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(P, **FAST), mock.patch.object(Q, "TH_STEP", 10.0):
            root = make_root(d, pert=pert); R.register_one(args(root, "mi"), torch, torch.nn, F, CPU); res = json.loads((root / "results" / "mi_smoke.json").read_text())
            self.assertEqual(sorted(res["cal"]), ["zh2_w00"]); self.assertEqual(sorted(k for k in res["dev"] if "#" not in k), ["zh3_w00", "zh4_w00"])
            self.assertEqual(sorted(k for k in res["dev"] if "#" in k), ["zh3_w00#profile"])                   # one profile per DEV unit
            for part in ("cal", "dev"):
                for k, v in res[part].items():
                    if "#" not in k:
                        self.assertLess(v[0]["err_full"], 9.0, (k, v[0]))                                         # exact up to interpolation
            self.assertEqual(res["dev_success42_unit"], 1.0); self.assertEqual(res["err_unit"], R.ERR_UNIT); self.assertEqual(res["dataset"], "synthetic")

    def test_critic_and_comir_paths_run(self):
        pert = {w: [[-4.0, 5.0, 7.0]] for w in ("zh2_w00", "zh3_w00", "zh4_w00")}
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(P, N_POS=256, GRID_STEP=17.0, N_ADAM=3, N_STARTS=1), mock.patch.object(Q, "TH_STEP", 15.0):
            root = make_root(d, pert=pert); run = root / "runs" / "vcs_s0"; run.mkdir(parents=True); torch.manual_seed(0); eU, eV, cr = P.build_models(torch.nn)
            for step in (50, 100, 150):
                torch.save({"eU": eU.state_dict(), "eV": eV.state_dict(), "critic": cr.state_dict(), "step": step}, run / f"ckpt_{step:05d}.pt")
            (run / "train.json").write_text(json.dumps({"objective": "vcs", "seed": 0}))
            R.register_one(args(root, "vcs"), torch, torch.nn, F, CPU); res = json.loads((root / "results" / "vcs_s0_smoke.json").read_text())
            self.assertEqual(res["chosen_ckpt"], "ckpt_00050.pt")                                                 # identical models: earliest wins the tie
            self.assertEqual([s["ckpt"] for s in res["selection"]], ["ckpt_00050.pt", "ckpt_00100.pt", "ckpt_00150.pt"])
            crun = root / "runs" / "comir_s0"; crun.mkdir(parents=True); mA, mB, cargs = P.comir_models(torch, 0, CPU)
            torch.save({"A": mA.state_dict(), "B": mB.state_dict(), "args": cargs}, crun / "comir.pt"); (crun / "comir.json").write_text(json.dumps({"seed": 0}))
            R.register_one(args(root, "comir"), torch, torch.nn, F, CPU); res = json.loads((root / "results" / "comir_s0_smoke.json").read_text())
            self.assertEqual(len(res["dev"]["zh3_w00"]), 1); self.assertIn("dev_success9_unit", res); self.assertIn("F_true", res["dev"]["zh4_w00"][0])
            with mock.patch.object(sys, "argv", ["agg", "--smoke", "--root", d, "--out", str(Path(d) / "rep")]):
                A.main(); out = json.loads((Path(d) / "rep" / "R6_REG_P3_results_smoke.json").read_text())
            self.assertEqual(out["dev_units"], ["zh3+zh4"]); self.assertIn("vcs_minus_comir@42", out["contrasts"]); self.assertEqual(out["contrasts"]["vcs_minus_comir@42"]["n_images"], 1)
            self.assertIsNone(out["ceiling_readout"]["triggered"]); self.assertIn("mi_smoke.json", out["ceiling_readout"]["missing_primary_rows"])
            self.assertEqual(len(out["not_run"]), 3); self.assertFalse(any(f.startswith("comir_wte") for f in out["missing"])); self.assertIsNone(out["gsd_m"])


def rec(ok, t=(5.0, 0.0), th=3.0):
    return {"err_full": 1.0 if ok else 100.0, "t": list(t), "theta": th, "rot_err_deg": 0.1, "trans_err_full": 1.0, "F_hat": 0.0, "F_true": 1.0, "seconds": 0.5}


class Readings(unittest.TestCase):
    def test_unit_pooling_differs_from_window_macro(self):
        run = {"dev": {"a1": [rec(1)] * 10, "a2": [rec(1)] * 10, "a3": [rec(0)] * 10, "b1": [rec(0)] * 10, "a1#profile": {}}, "cal": {}}
        u = A.unit_view(run, {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}); self.assertEqual(sorted(u["dev"]), ["A", "B"])
        self.assertAlmostEqual(A.macro(u, 42.0), (2 / 3 + 0) / 2); self.assertAlmostEqual(A.macro(run, 42.0), 0.5)          # unit vs window macro
        self.assertEqual(R.pool_units(run["dev"], {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}), u["dev"])

    def test_strata_from_translation(self):
        run = {"dev": {"A": [rec(1, (3, 4)), rec(0, (20, 0)), rec(1, (0, 40))], "B": [rec(0, (10, 0)), rec(1, (0, 33.9))]}}
        self.assertEqual(A.by_stratum_units(run, 42.0), [0.5, 0.5, 1.0])

    def test_contrasts_share_bootstrap_indices(self):
        g = np.random.default_rng(0); mk = lambda p: {"dev": {f"U{i}": [rec(g.random() < p) for _ in range(10)] for i in range(8)}}
        runs = {("vcs", s): mk(0.9) for s in (0, 1, 2)}; runs.update({("logistic", s): mk(0.6) for s in (0, 1, 2)}); runs[("mi", None)] = mk(0.5)
        c1 = A.contrast(runs, "vcs", "mi", 42.0, np.random.default_rng(A.BOOT_SEED)); A.contrast(runs, "vcs", "logistic", 42.0, np.random.default_rng(A.BOOT_SEED))
        c2 = A.contrast(runs, "vcs", "mi", 42.0, np.random.default_rng(A.BOOT_SEED)); self.assertEqual(c1["image_boot95"], c2["image_boot95"])
        self.assertEqual(c1["pairs"], [[0, None], [1, None], [2, None]]); self.assertEqual(c1["n_images"], 8)


# ------------------------------------------------------------------------------------------------------------ Zurich (kept for the record; not run)
class ZurichGeoTiff(unittest.TestCase):
    def test_reader_and_footprint(self):
        a = np.random.default_rng(0).integers(0, 4096, (40, 50, 4)).astype(np.uint16)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "zh1.tif"; write_geotiff(p, a, tie=(10.0, 20.0, 0.0, 1000.0, 5000.0, 0.0))
            self.assertTrue(np.array_equal(P.read_tiff(p), a))                                               # REG-P1's strict reader, 4 × 16 bit
            x0, x1, y0, y1 = R.footprint(p, a.shape)
            self.assertAlmostEqual(x0, 1000.0 - 10 * 0.6); self.assertAlmostEqual(x1, x0 + 50 * 0.6)
            self.assertAlmostEqual(y1, 5000.0 + 20 * 0.6); self.assertAlmostEqual(y0, y1 - 40 * 0.6)

    def test_gap(self):
        self.assertEqual(R.gap_m((0, 10, 0, 10), (10, 20, 0, 10)), 0.0)                                          # touching
        self.assertEqual(R.gap_m((0, 10, 0, 10), (5, 20, 5, 30)), 0.0)                                           # overlapping
        self.assertAlmostEqual(R.gap_m((0, 10, 0, 10), (40, 50, 0, 10)), 30.0)
        self.assertAlmostEqual(R.gap_m((0, 10, 0, 10), (13, 20, 14, 30)), 5.0)                                   # diagonal: hypot(3, 4)


class ZurichUnitsRolesWindows(unittest.TestCase):
    def test_units_merge_transitively(self):
        foot = {"zh1": (0, 100, 0, 100), "zh2": (130, 200, 0, 100), "zh3": (230, 300, 0, 100), "zh4": (400, 500, 0, 100), "zh5": (0, 100, 150.0, 200)}
        u = R.units_of(foot)
        self.assertEqual({u[k] for k in ("zh1", "zh2", "zh3")}, {"zh1+zh2+zh3"}); self.assertEqual(u["zh4"], "zh4"); self.assertEqual(u["zh5"], "zh5")

    def test_role_rule(self):
        uw = {f"zh{i}": 1 + i % 4 for i in range(1, 18)}; r = R.roles_p3(uw); self.assertEqual(r, R.roles_p3(dict(reversed(list(uw.items())))))
        self.assertEqual(sum(v == "DEV" for v in r.values()), R.N_DEV_UNITS); ncal = sum(uw[u] for u, v in r.items() if v == "CAL")
        self.assertGreaterEqual(ncal, R.MIN_CAL_WIN); self.assertLess(ncal - max(uw[u] for u, v in r.items() if v == "CAL"), R.MIN_CAL_WIN)   # stops once >= 8
        self.assertEqual(set(r), set(uw)); self.assertTrue(set(r.values()) <= {"FIT", "CAL", "DEV"})

    def test_windows(self):
        o = R.windows_of(1295, 1364); self.assertEqual(o, [(135, 170), (135, 682), (647, 170), (647, 682)])         # centred, non-overlapping
        for H, W in ((622, 782), (1639, 1672), (1179, 1830), (512, 512)):
            c = R.windows_of(H, W, covering=True); cov = np.zeros((H, W), int)
            for y, x in c:
                self.assertTrue(0 <= y <= H - R.WIN and 0 <= x <= W - R.WIN); cov[y:y + R.WIN, x:x + R.WIN] += 1
            self.assertTrue((cov >= 1).all())                                                                     # FIT: every pixel used
            n = R.windows_of(H, W); self.assertEqual(len(n), (H // R.WIN) * (W // R.WIN)); cov = np.zeros((H, W), int)
            for y, x in n:
                cov[y:y + R.WIN, x:x + R.WIN] += 1
            self.assertLessEqual(int(cov.max()), 1)                                                               # CAL / DEV: no overlap
        with self.assertRaises(ValueError):
            R.windows_of(500, 900)

    def test_band_order_luminance_and_linear_nir(self):
        t = synth_tile(1, 600, 700); w = R.window_inputs(t, 30, 50); a = t[30:30 + 512, 50:50 + 512].astype(np.float64) / R.DN
        rgb = a[..., [2, 1, 0]]; lum = 0.299 * a[..., 2] + 0.587 * a[..., 1] + 0.114 * a[..., 0]
        self.assertTrue(np.allclose(w["bf"], P.robust_norm(lum), atol=1e-5)); self.assertTrue(np.allclose(w["bf_rgb_n"], P.robust_norm(rgb), atol=1e-5))
        self.assertTrue(np.allclose(w["nir"], a[..., 3], atol=1e-7)); self.assertTrue(np.array_equal(w["shg"], w["shg_log_n"]))
        self.assertTrue(np.allclose(w["shg"], P.robust_norm(a[..., 3]), atol=1e-5)); self.assertEqual(w["bf_rgb_n"].shape, (512, 512, 3))
        with self.assertRaises(ValueError):
            R.window_inputs(t, 100, 50)                                                                           # window leaves the tile


@unittest.skipUnless(R.DATA3.exists(), "Zurich copy not available")
class ZurichDataFacts(unittest.TestCase):
    """The Zurich (not run) statements: 20 tiles, the adjacency units, the role assignment and the window counts (no pixel statistics)."""

    @classmethod
    def setUpClass(cls):
        cls.files = R.tile_files(); cls.shapes = {}
        for t, p in cls.files.items():
            b = p.read_bytes(); off = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, off)[0]; tg = {}
            for i in range(n):
                tag, typ, cnt, val = struct.unpack_from("<HHII", b, off + 2 + 12 * i)
                if tag in (256, 257, 277) and cnt == 1:
                    tg[tag] = val if typ == 4 else val & 0xFFFF
            cls.shapes[t] = (tg[257], tg[256], tg[277])
        cls.foot = {t: R.footprint(p, cls.shapes[t]) for t, p in cls.files.items()}; cls.unit = R.units_of(cls.foot)

    def test_units(self):
        self.assertEqual(len(self.files), 20); self.assertTrue(all("groundtruth" not in str(p) for p in self.files.values()))
        self.assertEqual(len(set(self.unit.values())), 17); self.assertEqual(self.unit["zh10"], "zh8+zh10+zh15"); self.assertEqual(self.unit["zh16"], "zh12+zh16")
        self.assertTrue(all(s[2] == 4 for s in self.shapes.values()))

    def test_roles_and_windows(self):
        uw = {}
        for t, s in self.shapes.items():
            uw[self.unit[t]] = uw.get(self.unit[t], 0) + len(R.windows_of(s[0], s[1]))
        r = R.roles_p3(uw); self.assertEqual(sum(uw.values()), 61)
        by = {k: sorted((u for u, v in r.items() if v == k), key=R.nat) for k in ("FIT", "CAL", "DEV")}
        self.assertEqual(by, ZURICH_ROLES); self.assertEqual({k: sum(uw[u] for u in us) for k, us in by.items()}, ZURICH_WINDOWS)


class ProtocolFile(unittest.TestCase):
    def test_protocol_parses_and_matches_the_code(self):
        import yaml
        fs = sorted((Path(__file__).resolve().parents[1] / "reports" / "R6").glob("R6_REG_P3_PROTOCOL_*_20261012.yaml"))
        self.assertTrue(fs); text = fs[-1].read_text(); d = yaml.safe_load(text)
        self.assertEqual(d["protocol_id"], "R6-REG-P3"); self.assertIs(d["official_test_access"], False); self.assertEqual(len(d["primary_contrasts"]), 2)
        self.assertIn("10.5", d["metrics"]["primary"]); self.assertEqual(len(d["wording_fixed"]), 4); self.assertIn("not run", d["zurich_not_run"]["status"])
        ex = d["data"]["exclusions"]; self.assertEqual(sorted(ex["alignment"]), EXPECTED_EPFL_MISALIGNED); self.assertEqual(sorted(ex["size"]), EXPECTED_EPFL_SMALL)
        self.assertEqual([tuple(x.split(" ~ ")) for x in d["data"]["near_duplicate_links"]], EXPECTED_EPFL_LINKS)
        self.assertEqual(sorted(d["data"]["roles"]["CAL_units"]), EXPECTED_EPFL_CAL); self.assertEqual(len(d["data"]["roles"]["DEV_units"]), 72)
        self.assertNotIn("comir_wte", " ".join(d["methods"]))


EXPECTED_EPFL_MISALIGNED = [
    "country/0015", "country/0017", "country/0018", "country/0024", "country/0040", "country/0042", "field/0029", "field/0039", "field/0040", "field/0045",
    "field/0054", "forest/0054", "mountain/0028", "mountain/0030", "mountain/0051", "mountain/0053", "oldbuilding/0085", "oldbuilding/0101", "street/0016",
    "street/0028", "street/0050", "street/0051", "street/0052", "street/0054", "water/0009", "water/0024", "water/0043", "water/0045", "water/0058",
    "water/0060", "water/0062", "water/0065", "water/0083"]                                           # the audit's 33 (base-alignment check >= 2 half-res px)
EXPECTED_EPFL_SMALL = [
    "country/0007", "country/0011", "forest/0003", "mountain/0001", "mountain/0002", "mountain/0003", "oldbuilding/0001", "oldbuilding/0003", "oldbuilding/0004",
    "oldbuilding/0081", "oldbuilding/0084", "oldbuilding/0086", "oldbuilding/0087", "oldbuilding/0093", "oldbuilding/0096", "street/0031", "urban/0004",
    "urban/0008", "urban/0024", "urban/0025", "urban/0045", "urban/0086", "water/0038", "water/0053"]  # narrower than a 512-px window (portrait images)
EXPECTED_EPFL_LINKS = [
    ("country/0001", "country/0025"), ("country/0001", "country/0032"), ("country/0016", "country/0018"), ("country/0016", "country/0044"),
    ("country/0017", "country/0025"), ("country/0017", "country/0032"), ("country/0023", "country/0027"), ("country/0025", "country/0032"),
    ("country/0036", "country/0045"), ("field/0011", "field/0042"), ("field/0013", "field/0014"), ("field/0020", "field/0038"),
    ("mountain/0039", "mountain/0041")]                                                               # the audit's 13 near-duplicate links
EXPECTED_EPFL_CAL = ["country/0023+country/0027", "field/0021", "forest/0028", "indoor/0000", "mountain/0050", "oldbuilding/0095", "street/0049",
                     "urban/0001", "water/0031"]                                                       # the role rule's CAL units (one per category)
EXPECTED_EPFL_DEV_WINDOWS = 74                                                                        # 72 DEV units; two of them hold 2 scenes
ZURICH_ROLES = {"FIT": ["zh1", "zh4", "zh18", "zh20", "zh12+zh16"], "CAL": ["zh3", "zh5", "zh13", "zh19"],
                "DEV": ["zh2", "zh6", "zh7", "zh9", "zh11", "zh14", "zh17", "zh8+zh10+zh15"]}             # Zurich (not run): rule output
ZURICH_WINDOWS = {"FIT": 14, "CAL": 9, "DEV": 38}


if __name__ == "__main__":
    unittest.main()
