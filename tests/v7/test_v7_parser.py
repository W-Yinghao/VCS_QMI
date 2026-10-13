"""V7 prompt-only parser post-processing (scripts/v7/v7_parser.py): the embedded system text equals the package template, JSON extraction,
word alignment, span keep / re-derive / drop, polarity guard.  CPU, no model."""
import sys
import unittest
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "v7"))
import v7_parser as VP  # noqa: E402

TPL = Path("/home/infres/yinwang/CS_QMI/outputs/V7/package_ref/templates/PROMPT_ONLY_ELEMENTS.txt")


class Parser(unittest.TestCase):
    def test_system_text_is_the_package_template(self):
        if not TPL.exists():
            self.skipTest("package template copy not available")
        t = TPL.read_text(); self.assertEqual(VP.SYSTEM, t.split("SYSTEM:\n", 1)[1].split("\nUSER:", 1)[0])

    def test_first_json_object(self):
        self.assertEqual(VP.first_json_object('blah {"a": "x}y", "b": {"c": 1}} tail {"z": 2}'), {"a": "x}y", "b": {"c": 1}, "_format": "object"})
        self.assertIsNone(VP.first_json_object("no json here")); self.assertEqual(VP.first_json_object('{bad} {"ok": 1}'), {"ok": 1, "_format": "object"})

    def test_top_level_array_is_the_element_list(self):
        raw = 'Here: [\n {"id": 1, "text": "a [red] cat", "spans": [[0, 1]]},\n {"id": 2, "text": "dog", "spans": [[2, 5]]}\n] done'
        obj = VP.first_json_object(raw); self.assertEqual(obj["_format"], "array"); self.assertEqual([e["id"] for e in obj["elements"]], [1, 2])
        els, unres, st = VP.validate("a red cat and a dog", obj); self.assertEqual([e["text"] for e in els], ["a [red] cat", "dog"])
        self.assertEqual((els[0]["spans"], els[0]["span_source"]), ([[0, 1]], "model")); self.assertEqual((els[1]["spans"], els[1]["span_source"]), ([[16, 19]], "rederived"))
        self.assertEqual(unres, []); self.assertFalse(st.get("no_elements_list", False))

    def test_word_aligned(self):
        p = "a red cat"; self.assertTrue(VP.word_aligned(p, 2, 5)); self.assertFalse(VP.word_aligned(p, 3, 5)); self.assertFalse(VP.word_aligned(p, 1, 2))

    def test_validate_keep_rederive_drop(self):
        p = "two dogs without a red ball"
        obj = {"elements": [{"id": "e1", "text": "two dogs", "spans": [[0, 8]], "type": "count", "polarity": "positive"},
                            {"id": "e2", "text": "red ball", "spans": [[3, 9]], "polarity": "negative"},          # misaligned -> re-derived from text
                            {"id": "e3", "text": "a unicorn", "spans": [[99, 120]]},                              # neither -> dropped
                            {"id": "e4", "text": "ball", "spans": [[23, 27]], "polarity": "maybe"}], "unresolved": ["style?"]}
        els, unres, st = VP.validate(p, obj)
        self.assertEqual([e["id"] for e in els], ["e1", "e2", "e4"]); self.assertEqual(els[1]["spans"], [[19, 27]]); self.assertEqual(els[1]["span_source"], "rederived")
        self.assertEqual(els[2]["polarity"], "positive"); self.assertEqual(st, {"dropped": 1, "rederived": 1, "polarity_fixed": 1}); self.assertEqual(unres, ["style?"])

    def test_stopword_overlap_is_not_enough(self):
        p = "a fleeting moment in a vibrant city street"; obj = {"elements": [{"id": "e", "text": "in a vibrant city street", "spans": [[0, 10]]}]}
        i = p.index("in a vibrant city street"); els, _, st = VP.validate(p, obj)
        self.assertEqual(els[0]["spans"], [[i, i + len("in a vibrant city street")]]); self.assertEqual(els[0]["span_source"], "rederived")

    def test_no_elements(self):
        els, _, st = VP.validate("x", {"foo": 1}); self.assertEqual(els, []); self.assertTrue(st["no_elements_list"])


if __name__ == "__main__":
    unittest.main()
