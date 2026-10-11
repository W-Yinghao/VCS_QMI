"""r6 GAIN-P1 CREMA-D: actor roles (single split and 5-fold cross-fitting), audio / video front ends, U-only policy inputs, costs
(scripts/r6_gain_crema.py)."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r6_gain_crema as C  # noqa: E402
import r6_gain_pilot as P  # noqa: E402

HAVE_ROLES = P.roles_path("cremad").exists()
HAVE_MEDIA = (C.MEDIA / "AudioWAV").exists() and (C.MEDIA / "VideoFlash").exists()
HAVE_AV = importlib.util.find_spec("av") is not None


@unittest.skipUnless(HAVE_ROLES, "roles.json not built")
class Roles(unittest.TestCase):
    def setUp(self):
        self.stems, self.y, self.g = C.stems_and_labels()

    def check(self, roles):
        owner = {}
        for k, v in roles.items():
            for a in np.unique(self.g[v]):
                self.assertNotIn(a, owner); owner[a] = k                                # actor-disjoint
            self.assertEqual(len(np.unique(self.y[v])), 6, k)                             # all six emotions in every role
        allidx = np.concatenate(list(roles.values())); self.assertEqual(len(allidx), len(np.unique(allidx)))
        return allidx

    def test_single_split(self):
        allidx = self.check(C.crema_roles(-1)); self.assertEqual(len(allidx), len(self.stems))

    def test_folds_cover_every_actor_once_as_dev(self):
        dev_actors = []
        for k in range(C.N_FOLDS):
            r = C.crema_roles(k); self.assertEqual(len(self.check(r)), len(self.stems)); dev_actors += list(np.unique(self.g[r["DEV"]]))
            self.assertTrue(np.array_equal(r["DEV"], C.crema_roles(k)["DEV"]))            # deterministic
        self.assertEqual(sorted(dev_actors), sorted(np.unique(self.g).tolist()))

    def test_labels_from_stems(self):
        for s, lab in zip(self.stems[:50], self.y[:50]):
            self.assertEqual(C.EMOTIONS[lab], s.split("_")[2])


class FrontEnds(unittest.TestCase):
    def test_frame_index(self):
        idx = C.frame_index(75); self.assertEqual(len(idx), C.K_FRAMES); self.assertEqual(idx[0], 0); self.assertEqual(idx[-1], 74)
        self.assertTrue(np.all(np.diff(idx) > 0)); self.assertEqual(len(C.frame_index(3)), C.K_FRAMES)       # short clips repeat frames

    def test_crop_resize_central_square(self):
        sel = np.zeros((C.K_FRAMES, 360, 480, 3), np.uint8); sel[:, :, :60] = 255; sel[:, :, 420:] = 255  # bars outside the central square
        out = C.crop_resize(sel)
        self.assertEqual(out.shape, (C.K_FRAMES, 3, C.SIDE, C.SIDE)); self.assertEqual(out.dtype, np.uint8); self.assertEqual(int(out.max()), 0)

    def test_logmel_tone(self):
        m = C.build_crema("scratch").eval(); t = np.arange(C.N_SAMPLES) / C.SR
        wave = torch.from_numpy((8000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16))[None]
        lm = m.logmel(wave); self.assertEqual(tuple(lm.shape), (1, 1, C.N_MELS, C.N_MEL_FRAMES))
        hz = 700.0 * (10 ** (np.linspace(0, 2595.0 * np.log10(1 + C.SR / 2 / 700.0), C.N_MELS + 2) / 2595.0) - 1.0)
        peak = int(lm[0, 0, :, 100].argmax()); self.assertLess(abs(hz[peak + 1] - 1000), 150)

    def test_wav_pad(self):
        if not HAVE_MEDIA:
            self.skipTest("media missing")
        w, n = C.read_wav(sorted((C.MEDIA / "AudioWAV").glob("*.wav"))[0]); self.assertEqual(w.shape, (C.N_SAMPLES,)); self.assertEqual(w.dtype, np.int16)
        if n < C.N_SAMPLES:
            self.assertTrue(np.all(w[n:] == 0))


class Policy(unittest.TestCase):
    def test_policy_inputs_do_not_depend_on_v(self):
        torch.manual_seed(0); m = C.build_crema("scratch").eval(); wave = torch.randint(-3000, 3000, (2, C.N_SAMPLES), dtype=torch.int16)
        f1 = torch.randint(0, 255, (2, C.K_FRAMES, 3, C.SIDE, C.SIDE), dtype=torch.uint8); f2 = torch.randint(0, 255, f1.shape, dtype=torch.uint8)
        with torch.no_grad():
            h1, la1 = m.cheap(wave); h2, la2 = m.cheap(wave); b1, b2 = m.full(h1, f1), m.full(h2, f2)
        X1 = P.policy_features(h1.numpy(), la1.softmax(1).numpy()); X2 = P.policy_features(h2.numpy(), la2.softmax(1).numpy())
        self.assertTrue(np.array_equal(X1, X2)); self.assertFalse(torch.allclose(b1, b2)); self.assertEqual(X1.shape[1], 512 + 6)

    def test_policy_cost(self):
        c = {"t_u": np.array([1.0, 2.0]), "t_pol_mlp": np.array([0.5, 0.5]), "t_pol_entropy": np.array([0.1, 0.1]), "t_v": np.array([10.0, 20.0])}
        self.assertAlmostEqual(C.policy_cost(c, "never", 0.0), 1.5); self.assertAlmostEqual(C.policy_cost(c, "always", 1.0), 16.5)
        self.assertAlmostEqual(C.policy_cost(c, "gain_h", np.array([0, 1])), 1.5 + 0.5 + 10.0); self.assertAlmostEqual(C.policy_cost(c, "random", 0.5), 1.5 + 7.5)


@unittest.skipUnless(HAVE_MEDIA and HAVE_AV, "media or PyAV missing")
class Decoders(unittest.TestCase):
    def test_pyav_matches_ffmpeg_cli(self):
        for p in sorted((C.MEDIA / "VideoFlash").glob("*.flv"))[:2]:
            a, na = C.decode_selected(p, "av"); b, nb = C.decode_selected(p, "cli")
            self.assertEqual(na, nb); self.assertEqual(a.shape, b.shape)
            d = np.abs(a.astype(int) - b.astype(int)); print(f"decoder diff {p.name}: mean {d.mean():.3f}, max {d.max()}")
            self.assertLess(float(d.mean()), 2.0, p.name)                                 # same codec; swscale flags may differ slightly


class NoTestSplit(unittest.TestCase):
    def test_only_media_root_read(self):
        src = (R / "scripts" / "r6_gain_crema.py").read_text()
        self.assertIn('crema_d/media"', src); self.assertNotIn("crema_d/repo", src)
        if HAVE_ROLES:
            self.assertEqual(json.load(open(P.roles_path("cremad")))["media_root"], str(C.MEDIA))


if __name__ == "__main__":
    unittest.main()
