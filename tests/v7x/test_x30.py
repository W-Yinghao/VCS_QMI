"""V7X-30 runner logic with stubs (no model): PN-VQA parsers and scoring formula, DSG scoring through DSG's own calc_vqa_score (read-only tree),
the zero-question paths, and the file contract of every runner's output."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v7x"))
import x30_pnvqa as PN  # noqa: E402

DSG_ROOT = Path("/projects/EEG-foundation-model/yinghao/third_party/DSG@7330119")


def test_pnvqa_parsers():
    assert PN.first_list('Elements:["man (human)", "selfie (activity)"]\nmore') == ["man (human)", "selfie (activity)"]
    assert PN.first_list("no list here") is None and PN.first_list("[not json") is None
    assert PN.parse_qa("Q: Is this a man?\nChoices: yes, no\nA: yes") == ("Is this a man?", "yes")
    assert PN.parse_qa("Q: Are there nuns?\nChoices: yes, no\nA: No.") == ("Are there nuns?", "no")
    assert PN.parse_qa("garbage") is None
    assert "{prompt}" in PN.SPLIT and "{caption}" in PN.QUESTION and "{element}" in PN.QUESTION and PN.VQA.count("{") == 3


def test_pnvqa_score_formula(tmp_path, monkeypatch):
    class FakeYesNo:
        def __init__(self):
            pass

        def p_yes(self, image, texts):                 # P(Yes) 0.9 for the true answer substituted, 0.2 for the false one
            return [0.9 if i % 2 == 0 else 0.2 for i in range(len(texts))]
    monkeypatch.setattr(PN, "YesNo", FakeYesNo)
    gen = tmp_path / "g.jsonl"; gen.write_text(json.dumps({"prompt_id": "p1", "prompt": "a cat", "qas": [{"element": "cat (animal)", "question": "Is there a cat?", "answer": "yes", "parsed": True}]}) + "\n"
                                               + json.dumps({"prompt_id": "p2", "prompt": "x", "qas": [{"element": "x", "question": None, "answer": None, "parsed": False}]}) + "\n")
    inp = tmp_path / "i.jsonl"; inp.write_text("".join(json.dumps({"prompt_id": p, "generator_id": "G", "image_id": f"G/{p}.jpg", "candidate_index": 0}) + "\n" for p in ("p1", "p2")))
    class FakeZip:
        def __init__(self, *a):
            pass

        def read(self, name):
            return b"img"
    monkeypatch.setattr(PN.zipfile, "ZipFile", FakeZip); monkeypatch.setenv("V7X_LOCAL", str(tmp_path))
    assert PN.main(["score", "--gen", str(gen), "--inputs", str(inp), "--out", str(tmp_path / "s.jsonl")]) == 0
    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert rows[0]["score"] == pytest.approx((0.9 + 1 - 0.2) / 2) and rows[1]["no_questions"] and rows[1]["score"] == 0.0


@pytest.mark.skipif(not DSG_ROOT.exists(), reason="DSG tree not present")
def test_dsg_score_with_its_own_calc(tmp_path, monkeypatch):
    sys.modules.setdefault("openai", types.ModuleType("openai")); sys.path.insert(0, str(DSG_ROOT))
    import x30_dsg as DS
    from dsg import vqa_utils
    class FakeMPLUG:
        def __init__(self, *a):
            pass

        def vqa(self, image, question):
            return "no" if "dog" in question else "yes"
    monkeypatch.setattr(vqa_utils, "MPLUG", FakeMPLUG)
    class FakeZip:
        def __init__(self, *a):
            pass

        def read(self, name):
            return b"img"
    monkeypatch.setattr(DS.zipfile, "ZipFile", FakeZip); monkeypatch.setenv("V7X_LOCAL", str(tmp_path))
    g = {"prompt_id": "p1", "prompt": "a dog on a red mat", "qid2question": {"1": "Is there a dog?", "2": "Is the dog on a mat?", "3": "Is the mat red?"},
         "qid2dependency": {"1": [0], "2": [1], "3": [0]}}
    (tmp_path / "g.jsonl").write_text(json.dumps(g) + "\n" + json.dumps({"prompt_id": "p2", "prompt": "z", "qid2question": {}, "qid2dependency": {}}) + "\n")
    (tmp_path / "i.jsonl").write_text("".join(json.dumps({"prompt_id": p, "generator_id": "G", "image_id": f"G/{p}.jpg", "candidate_index": 0}) + "\n" for p in ("p1", "p2")))
    assert DS.main(["score", "--dsg", str(tmp_path / "g.jsonl"), "--inputs", str(tmp_path / "i.jsonl"), "--out", str(tmp_path / "s.jsonl")]) == 0
    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    # q1 'no' (dog), q2 answered 'no' (dog) and its parent is 'no' -> 0, q3 'yes': with dependency 1 / 3, without 1 / 3
    assert rows[0]["score"] == pytest.approx(1 / 3) and rows[0]["score_without_dependency"] == pytest.approx(1 / 3)
    assert rows[1]["no_questions"] and rows[1]["score"] == 0.0
