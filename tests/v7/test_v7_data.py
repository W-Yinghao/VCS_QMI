"""V7-01 data rules (scripts/v7/v7_data.py): label codes, element keys, span rules (repeats, case, hyphens, plurals, regex characters, no bare
substring), negation cue, de-duplication (prompt Jaccard, dHash bands exact up to distance 7), component split, FIT weights / prior,
multi-part archive reader, GenAI prompt extraction never reading a label column.  CPU, synthetic data only."""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "v7"))
import v7_data as VD  # noqa: E402


class Labels(unittest.TestCase):
    def test_element_key(self):
        self.assertEqual(VD.parse_element_key("puffin (animal)"), ("puffin", "animal"))
        self.assertEqual(VD.parse_element_key("no category"), ("no category", None))
        self.assertEqual(VD.parse_element_key("a (b) (counting)"), ("a (b)", "counting"))

    def test_flags_keep_missing_apart(self):
        self.assertEqual(VD.flag_fraction([0, 0, 1]), (1 / 3, 3, 0, 0))
        self.assertEqual(VD.flag_fraction([-1, 1, 1]), (1.0, 2, 1, 0))
        self.assertEqual(VD.flag_fraction([]), (None, 0, 0, 0))
        self.assertEqual(VD.flag_fraction([2]), (None, 0, 0, 1))

    def test_votes_and_status(self):
        v = VD.votes_summary([1, 0, 1]); self.assertAlmostEqual(v["y"], 2 / 3); self.assertEqual((v["n_valid"], v["n_pos"]), (3, 2))
        v2 = VD.votes_summary([1, 2]); self.assertEqual((v2["n_valid"], v2["n_unknown_codes"], v2["y"]), (1, 1, 1.0))
        self.assertEqual(VD.element_status(v, True), "bad_decomposition"); self.assertEqual(VD.element_status(v, False), "valid")
        self.assertEqual(VD.element_status(VD.votes_summary([]), False), "missing"); self.assertEqual(VD.element_status(VD.votes_summary([7]), False), "unknown")


class Spans(unittest.TestCase):
    def test_every_occurrence_case_insensitive(self):
        p = "A cat and a Cat near a catalogue"; s, rule = VD.find_spans(p, "cat")
        self.assertEqual(rule, "exact_word"); self.assertEqual([p[a:b] for a, b in s], ["cat", "Cat"])

    def test_no_bare_substring(self):
        self.assertEqual(VD.find_spans("concatenate strings", "cat"), ([], "unmatched"))

    def test_space_hyphen_and_inflection(self):
        p = "a man in a t shirt walking two dogs"; s, rule = VD.find_spans(p, "t-shirt"); self.assertEqual((rule, p[s[0][0]:s[0][1]]), ("space_hyphen", "t shirt"))
        s, rule = VD.find_spans(p, "dog"); self.assertEqual((rule, p[s[0][0]:s[0][1]]), ("inflection", "dogs"))

    def test_regex_characters_escaped(self):
        p = "a book about C++ (programming)"; s, rule = VD.find_spans(p, "C++"); self.assertEqual((rule, p[s[0][0]:s[0][1]]), ("exact_word", "C++"))

    def test_negation_cue(self):
        p = "an empty room without any chairs"; s, _ = VD.find_spans(p, "chairs"); self.assertTrue(VD.negation_cue(p, s))
        p2 = "a red car"; s2, _ = VD.find_spans(p2, "car"); self.assertFalse(VD.negation_cue(p2, s2)); self.assertFalse(VD.negation_cue(p2, []))


class Duplicates(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(VD.normalize_prompt("  A  Cat. "), "a cat")

    def test_union_find(self):
        uf = VD.UnionFind(["a", "b", "c", "d"]); uf.union("b", "a"); uf.union("c", "d")
        self.assertEqual(sorted(uf.groups().values()), [["a", "b"], ["c", "d"]])

    def test_near_duplicate_prompts(self):
        n = {"1": VD.normalize_prompt("a cat sitting on a red mat in the sun"), "2": VD.normalize_prompt("the cat sitting on a red mat in the sun"),
             "3": VD.normalize_prompt("a cat sitting on a red mat in the sun."), "4": VD.normalize_prompt("two dogs playing chess at night")}
        self.assertEqual(VD.near_duplicate_prompts(n), [("1", "2"), ("2", "3")])          # 1 / 3 are exact duplicates (handled elsewhere)

    def test_dhash_bands_exact_up_to_7(self):
        base = 0x0123456789ABCDEF; spread4 = base ^ (1 | 1 << 16 | 1 << 32 | 1 << 48); spread7 = base ^ sum(1 << (8 * k) for k in range(7))
        far = base ^ sum(1 << (8 * k) for k in range(8))
        pairs = VD.near_duplicate_hashes({"a": base, "b": spread4, "c": far}); self.assertIn(("a", "b"), pairs); self.assertNotIn(("a", "c"), pairs)
        self.assertIn(("a", "x"), VD.near_duplicate_hashes({"a": base, "x": spread7}, max_hamming=7))
        with self.assertRaises(ValueError):
            VD.near_duplicate_hashes({"a": base}, max_hamming=8)

    def test_dhash_image(self):
        from PIL import Image
        a = np.tile(np.arange(64, dtype=np.uint8) * 4, (48, 1)); im = Image.fromarray(a)
        self.assertEqual(VD.dhash64(im), VD.dhash64(im.copy())); self.assertNotEqual(VD.dhash64(im), VD.dhash64(Image.fromarray(a[:, ::-1].copy())))


class Split(unittest.TestCase):
    def test_roles_deterministic_complete_and_disjoint(self):
        rng = np.random.default_rng(0); comps = {f"c{i:03d}": [f"img{i}_{j}" for j in range(int(rng.integers(1, 9)))] for i in range(300)}
        sizes = {c: len(m) for c, m in comps.items()}; r1 = VD.assign_roles(comps, sizes, excluded={"c000"}); r2 = VD.assign_roles(comps, sizes, excluded={"c000"})
        self.assertEqual(r1, r2); self.assertEqual(r1["c000"], "EXCLUDED_OVERLAP"); self.assertEqual(set(r1), set(comps))
        tot = sum(sizes[c] for c in comps if c != "c000"); frac = {k: sum(sizes[c] for c in comps if r1[c] == k) / tot for k in ("FIT", "CAL_SELECT", "CAL_PROB", "DEV")}
        self.assertAlmostEqual(frac["FIT"], 0.8, delta=0.03); self.assertAlmostEqual(frac["DEV"], 0.1, delta=0.03)
        self.assertAlmostEqual(frac["CAL_SELECT"], frac["CAL_PROB"], delta=0.03)

    def test_fit_weights_prompt_uniform(self):
        imgs = [{"image_id": "i1", "prompt_id": "p"}, {"image_id": "i2", "prompt_id": "p"}, {"image_id": "i3", "prompt_id": "q"}]
        w = VD.fit_weights(imgs); self.assertEqual(w, {"i1": 0.75, "i2": 0.75, "i3": 1.5}); self.assertAlmostEqual(sum(w.values()), 3)

    def test_fit_prior_element_uniform_within_image(self):
        el = [{"element_id": "a", "image_id": "i1", "y": 1.0}, {"element_id": "b", "image_id": "i1", "y": 0.0}, {"element_id": "c", "image_id": "i3", "y": 1.0}]
        pi, w = VD.fit_prior(el, {"i1": 1.0, "i3": 1.0}); self.assertAlmostEqual(pi, 0.75); self.assertEqual(w, {"a": 0.5, "b": 0.5, "c": 1.0})


class Archive(unittest.TestCase):
    def test_multi_part_reader(self):
        buf = io.BytesIO(); payload = {f"dataset/images/G{k}/{k:05d}.png": bytes(np.random.default_rng(k).integers(0, 256, 5000 + 97 * k, dtype=np.uint8)) for k in range(7)}
        with zipfile.ZipFile(buf, "w") as z:
            for i, (n, b) in enumerate(payload.items()):
                z.writestr(n, b, compress_type=zipfile.ZIP_STORED if i % 2 else zipfile.ZIP_DEFLATED)
        data = buf.getvalue(); cuts = [0, 1234, 20000, len(data)]
        with tempfile.TemporaryDirectory() as d:
            for k in range(3):
                Path(d, f"images.zip.part-a{'abc'[k]}").write_bytes(data[cuts[k]:cuts[k + 1]])
            z = VD.open_archive(Path(d))
            for n, b in payload.items():
                self.assertEqual(z.read(n), b)


class GenAIPrompts(unittest.TestCase):
    def test_label_columns_never_read(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as d:
            pq.write_table(pa.table({"id": [1, 0], "prompt": ["b", "a"], "human_score": [[1, 2], [3, 4]]}), Path(d, "x.parquet"))
            out = Path(d, "p.jsonl"); VD.main(["genai-prompts", "--parquet-dir", d, "--out", str(out)])
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual(rows, [{"source": "genai_bench", "id": "0", "prompt": "a"}, {"source": "genai_bench", "id": "1", "prompt": "b"}])


if __name__ == "__main__":
    unittest.main()
