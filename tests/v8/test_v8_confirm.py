"""V8-05 confirmation tooling: metric definitions on synthetic data, the prompt-list rule, missing-row handling, bootstrap determinism, Holm,
and the guarantee that dry-run never opens a rating column (the only label reader is monkeypatched to raise)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v8"))
import v8_confirm as C  # noqa: E402


def test_prompt_list_rule_exact_text():
    bench = [{"prompt_id": "gb0000", "prompt": "a cat"}, {"prompt_id": "gb0001", "prompt": "a dog"}, {"prompt_id": "gb0002", "prompt": "a cat "}]
    rank = [{"prompt_id": "00001", "prompt": "a cat"}]
    pl = C.nonrank_prompts(bench, rank)
    assert pl["n_overlap"] == 1 and pl["prompt_ids"] == ["gb0001", "gb0002"]     # exact equality: the trailing-space prompt is NOT an overlap
    assert pl["sha256_ids"] == C.hashlib.sha256("gb0001\ngb0002".encode()).hexdigest()


def test_spearman_matches_scipy_with_ties():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0); a = rng.integers(1, 6, 300).astype(float); b = a + rng.normal(0, 2, 300); b[::7] = b[1::7][: len(b[::7])]
    assert abs(C.spearman(a, b) - scipy.spearmanr(a, b).correlation) < 1e-12
    assert np.isnan(C.spearman(np.ones(10), b[:10]))


def test_pairwise_prompt_definition():
    h = np.array([5.0, 3.0, 3.0, 1.0])                      # 4 candidates; pairs with equal human means (1-2) skipped -> 5 informative pairs
    s = np.array([4.0, 2.0, 2.0, 0.0]); assert C.pairwise_prompt(s, h) == 1.0
    s = np.array([0.0, 2.0, 2.0, 4.0]); assert C.pairwise_prompt(s, h) == 0.0
    s = np.array([1.0, 1.0, 1.0, 1.0]); assert C.pairwise_prompt(s, h) == 0.5   # all method ties count 0.5
    assert C.pairwise_prompt(np.array([1.0, 2.0]), np.array([3.0, 3.0])) is None


def _toy(seed=1, n_prompts=40):
    rng = np.random.default_rng(seed); prompt_of, human, good, bad, const = {}, {}, {}, {}, {}
    for p in range(n_prompts):
        for g in C.GENS:
            im = f"{g}/{p:05d}.png"; prompt_of[im] = f"gb{p:04d}"; human[im] = float(rng.integers(1, 6))
            good[im] = human[im] + rng.normal(0, 0.3); bad[im] = rng.normal(); const[im] = 1.0
    return prompt_of, human, good, bad, const


def test_panel_metrics_and_missing_rows_dropped_for_all():
    prompt_of, human, good, bad, const = _toy()
    bad2 = dict(bad); bad2.pop("DALLE_3/00003.png")              # one image missing in one method
    panel = C.Panel(prompt_of, {"U-CS": good, "B": bad2, "C": const}, human)
    assert len(panel.images) == 40 * 6 - 1 and "DALLE_3/00003.png" not in panel.images
    assert panel.spearman("U-CS") > 0.8 and abs(panel.spearman("B")) < 0.3 and np.isnan(panel.spearman("C"))
    pw = panel.pairwise_per_prompt("U-CS"); assert np.nanmean(pw) > 0.85
    assert abs(np.nanmean(panel.pairwise_per_prompt("C")) - 0.5) < 1e-12


def test_contrasts_bootstrap_deterministic_and_holm():
    prompt_of, human, good, bad, const = _toy()
    panel = C.Panel(prompt_of, {"U-CS": good, "B": bad, "C": const}, human)
    r1 = C.contrasts(panel, "U-CS", ["B", "C"], "both", reps=200, seed=7); r2 = C.contrasts(panel, "U-CS", ["B", "C"], "both", reps=200, seed=7)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)   # NaN-safe equality (the constant method has no Spearman)
    assert r1["B"]["spearman"]["delta"] > 0.5 and r1["B"]["spearman"]["ci95"][0] > 0 and r1["B"]["spearman"]["p"] <= 2 / 200 + 1e-12
    fam = C.add_holm(r1); assert fam["B"]["pairwise"]["p_holm"] >= fam["B"]["pairwise"]["p"]
    assert C.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]   # step-down with monotone enforcement


def test_dry_run_never_opens_ratings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(C, "read_human", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("label read in dry-run")))
    monkeypatch.setattr(C, "inventory", lambda **k: {"prompt_list": {"n_confirmation": 850, "n_overlap": 750}, "n_images_expected": 5100, "methods": {}, "missing_units": ["x"],
                                                     "external_files": {}, "parquet": [], "ready_for_run": False})
    rc = C.cmd_dry_run(type("A", (), {"out": str(tmp_path), "hash": False})())
    assert rc == 1 and not calls and (tmp_path / "dry_run.json").exists()


def test_run_refuses_without_confirmed_request(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REQUEST", tmp_path / "missing.md")
    with pytest.raises(SystemExit):
        C.cmd_run(type("A", (), {"out": str(tmp_path / "o"), "allow_scoring": True})())
    with pytest.raises(SystemExit):
        C.cmd_run(type("A", (), {"out": str(tmp_path / "o"), "allow_scoring": False})())


def test_markdown_from_json_roundtrip():
    prompt_of, human, good, bad, const = _toy(n_prompts=12)
    panel = C.Panel(prompt_of, {"U-CS": good, "B0_FGA_released_O": bad}, human)
    res = {"utc": "t", "n_images_scored": len(panel.images), "n_prompts_scored": 12, "dropped_images": [], "images_without_rating": [], "boot": {"reps": 50, "seed": 1, "unit": "prompt"},
           "levels": {"U-CS": {"spearman_pooled": panel.spearman("U-CS"), "pairwise": 0.9, "spearman_per_generator": {g: 0.5 for g in C.GENS}}}, "per_seed": {},
           "families": {"primary_spearman": C.add_holm(C.contrasts(panel, "U-CS", ["B0_FGA_released_O"], "spearman", reps=50, seed=1))}}
    md = C.markdown(json.loads(json.dumps(res))); assert "primary_spearman" in md and "U-CS − B0_FGA_released_O" in md


# ------------------------------------------------------------------------------------------------------------------ TIFA160 secondary (§B)
def test_tifa160_correlations_match_scipy():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3); a = rng.integers(1, 6, 200).astype(float); b = a + rng.normal(0, 1.5, 200); b[::9] = 3.0
    assert abs(C.pearson(a, b) - scipy.pearsonr(a, b)[0]) < 1e-12
    assert abs(C.kendall_tau_b(a, b) - scipy.kendalltau(a, b).correlation) < 1e-12      # scipy default = tau-b
    assert np.isnan(C.kendall_tau_b(np.ones(5), b[:5]))


def test_tifa160_dsg_csv_parser_public_schema(tmp_path):
    csv_text = ("item_id,source_id,t2i_model,worker_id,answer\n"
                "tifa160_0,coco_1,mini-dalle,w1,4\ntifa160_0,coco_1,mini-dalle,w2,2\ntifa160_0,coco_1,sd1dot5,w1,5\ntifa160_1,coco_2,vq-diffusion,w3,1\n")
    p = tmp_path / "anns.csv"; p.write_text(csv_text)
    image_of = {("coco_1", "mini_dalle"): "annotated_images/coco_1_mini_dalle.jpg", ("coco_1", "stable_diffusion_v1_5"): "annotated_images/coco_1_stable_diffusion_v1_5.jpg"}
    r = C.read_tifa160_dsg(p, image_of)
    assert r["human"] == {"annotated_images/coco_1_mini_dalle.jpg": 3.0, "annotated_images/coco_1_stable_diffusion_v1_5.jpg": 5.0}   # unmapped (coco_2) ignored
    assert r["n_raters"]["annotated_images/coco_1_mini_dalle.jpg"] == 2


def test_tifa160_tifa_json_parser_layouts(tmp_path):
    image_of = {("coco_1", "mini_dalle"): "annotated_images/coco_1_mini_dalle.jpg", ("coco_2", "vq_diffusion"): "annotated_images/coco_2_vq_diffusion.jpg"}
    p = tmp_path / "a.json"; p.write_text(json.dumps({"coco_1_mini_dalle.jpg": [4, 2], "coco_2_vq_diffusion": {"human_score": 5}, "unknown_x": 1}))
    r = C.read_tifa160_tifa(p, image_of); assert r["layout"] == "dict" and r["human"] == {"annotated_images/coco_1_mini_dalle.jpg": 3.0, "annotated_images/coco_2_vq_diffusion.jpg": 5.0} and r["unparsed_entries"] == 1
    p2 = tmp_path / "b.json"; p2.write_text(json.dumps({"coco_1": {"mini_dalle": [3, 3]}}))
    r2 = C.read_tifa160_tifa(p2, image_of); assert r2["layout"] == "dict[id][model]" and r2["human"] == {"annotated_images/coco_1_mini_dalle.jpg": 3.0}
    p3 = tmp_path / "c.json"; p3.write_text(json.dumps([{"image": "coco_1_mini_dalle.jpg", "score": 2}, {"text_id": "coco_2", "model": "vq_diffusion", "scores": [1, 5]}, {"x": 1}]))
    r3 = C.read_tifa160_tifa(p3, image_of); assert r3["layout"] == "list" and r3["human"]["annotated_images/coco_2_vq_diffusion.jpg"] == 3.0 and r3["unparsed_entries"] == 1


def test_tifa160_contrasts_bootstrap_deterministic_and_holm():
    rng = np.random.default_rng(5); prompts = [f"p{i}" for i in range(20)]; gens = ["g1", "g2", "g3"]
    images = [f"{g}/{p}.jpg" for p in prompts for g in gens]; prompt_of = {im: im.split("/")[1][:-4] for im in images}
    human = {im: float(rng.integers(1, 6)) for im in images}
    methods = {"U-CS": {im: human[im] + rng.normal(0, 1) for im in images}, "X": {im: rng.normal() for im in images}, "Y": {im: -human[im] for im in images}}
    r1 = C.tifa160_contrasts(images, prompt_of, methods, human, "U-CS", ["X", "Y"], reps=300, seed=7)
    r2 = C.tifa160_contrasts(images, prompt_of, methods, human, "U-CS", ["X", "Y"], reps=300, seed=7)
    assert json.dumps(r1) == json.dumps(r2) and r1["Y"]["spearman"]["delta"] > 0 and r1["Y"]["spearman"]["ci95"][0] > 0
    fam = C.add_holm_corr(r1); assert fam["Y"]["spearman"]["p_holm"] >= fam["Y"]["spearman"]["p"] and fam["X"]["spearman"]["p_holm"] <= 1.0


def test_tifa160_dry_run_never_opens_labels(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("label reader called during dry-run")
    monkeypatch.setattr(C, "read_tifa160_dsg", boom); monkeypatch.setattr(C, "read_tifa160_tifa", boom)
    monkeypatch.setattr(C, "TIFA160_INPUTS", tmp_path / "missing_inputs.jsonl"); monkeypatch.setattr(C, "UNITS", tmp_path / "units")
    monkeypatch.setattr(C, "TIFA160_SUMS", tmp_path / "no_sums.txt")
    monkeypatch.setattr(C, "TIFA160_LABELS", {"tifa_2rater": tmp_path / "no_a.json", "dsg_5rater": tmp_path / "no_b.csv"})
    monkeypatch.setattr(C, "TIFA160_EXTERNAL", {k: tmp_path / f"{k}.jsonl" for k in C.TIFA160_EXTERNAL})
    a = type("A", (), {"out": str(tmp_path / "dr"), "hash": True})()
    assert C.cmd_tifa160_dry_run(a) == 1 and (tmp_path / "dr" / "tifa160_dry_run.json").exists()


def test_tifa160_run_refuses_without_confirmed_line(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REQUEST", tmp_path / "req.md"); (tmp_path / "req.md").write_text("# request CONFIRMED for GenAI-Bench only\n")
    a = type("A", (), {"out": str(tmp_path / "o"), "allow_scoring": True})()
    with pytest.raises(SystemExit):
        C.cmd_tifa160_run(a)


# ---- hardened readers / fast estimators (added before the TIFA160 read, 2026-10-15) --------------------------------------------------------
def _img(tid, tag):
    return f"annotated_images/{tid}_{tag}.jpg"


def test_tifa160_fast_estimators_equal_reference():
    rng = np.random.default_rng(11)
    for _ in range(5):
        a = rng.integers(1, 6, 300).astype(float); b = a + rng.normal(0, 1.2, 300); b[::7] = 2.0
        assert abs(C.spearman_tifa(a, b) - C.spearman(a, b)) < 1e-12 and abs(C.kendall_tifa(a, b) - C.kendall_tau_b(a, b)) < 1e-12
    assert np.isnan(C.kendall_tifa(np.ones(6), np.arange(6.0))) and np.isnan(C.spearman_tifa(np.ones(6), np.arange(6.0)))


def test_tifa160_contrasts_equal_reference_loop():
    """The restructured bootstrap (draws once, ours cached) gives exactly what the per-comparator reference loop gave."""
    rng = np.random.default_rng(9); prompts = [f"p{i}" for i in range(15)]; gens = ["g1", "g2", "g3", "g4"]
    images = [f"{g}/{p}.jpg" for p in prompts for g in gens]; prompt_of = {im: im.split("/")[1][:-4] for im in images}
    human = {im: float(rng.integers(1, 6)) for im in images}
    methods = {"U-CS": {im: human[im] + rng.normal(0, 1) for im in images}, "X": {im: rng.normal() for im in images}, "Y": {im: float(rng.integers(0, 3)) for im in images}}
    got = C.tifa160_contrasts(images, prompt_of, methods, human, "U-CS", ["X", "Y"], reps=200, seed=4)
    ps = sorted({prompt_of[im] for im in images}); idx = {p: np.array([i for i, im in enumerate(images) if prompt_of[im] == p]) for p in ps}
    h = np.array([human[im] for im in images]); m = {k: np.array([methods[k][im] for im in images]) for k in methods}
    rng2 = np.random.default_rng(4); draws = [rng2.integers(0, len(ps), len(ps)) for _ in range(200)]
    ref_fn = {"spearman": C.spearman, "pearson": C.pearson, "kendall_tau_b": C.kendall_tau_b}
    for comp in ("X", "Y"):
        for cn, fn in ref_fn.items():
            bs = np.array([fn(m["U-CS"][s], h[s]) - fn(m[comp][s], h[s]) for s in (np.concatenate([idx[ps[i]] for i in dr]) for dr in draws)]); bs = bs[~np.isnan(bs)]
            g = got[comp][cn]
            assert abs(g["delta"] - (fn(m["U-CS"], h) - fn(m[comp], h))) < 1e-12 and abs(g["p"] - C.boot_p(bs)) < 1e-12
            assert abs(g["ci95"][0] - np.percentile(bs, 2.5)) < 1e-9 and abs(g["ci95"][1] - np.percentile(bs, 97.5)) < 1e-9


def test_tifa160_dsg_reader_id_column_fallback_and_bad_answers(tmp_path):
    image_of = {("coco_1", "mini_dalle"): _img("coco_1", "mini_dalle"), ("coco_1", "stable_diffusion_v2_1"): _img("coco_1", "stable_diffusion_v2_1")}
    p = tmp_path / "a.csv"
    p.write_text("item_id,source_id,t2i_model,worker_id,answer\ncoco_1,x,mini-dalle,w1,5\ncoco_1,x,mini-dalle,w2,N/A\ntifa160_0,coco_1,sd2dot1,w1,2\n"
                 "tifa160_0,coco_1,sd2dot1,w2,3\ntifa160_9,coco_9,sd2dot1,w1,1\n")
    r = C.read_tifa160_dsg(p, image_of)
    assert r["human"] == {_img("coco_1", "mini_dalle"): 5.0, _img("coco_1", "stable_diffusion_v2_1"): 2.5}
    c = r["meta"]["counts"]; assert c["answer_not_numeric"] == 1 and c["rows_unmatched"] == 1 and c["text_id_from_item_id"] == 1 and c["text_id_from_source_id"] == 2
    assert r["meta"]["raters_per_image"] == {1: 1, 2: 1}


def test_tifa160_tifa_reader_more_layouts(tmp_path):
    ids = ["coco_1", "coco_12", "partiprompt_3"]; tags = list(C.TIFA160_MODELS)
    image_of = {(t, g): _img(t, g) for t in ids for g in tags}

    def run(obj):
        p = tmp_path / f"x{abs(hash(json.dumps(obj, sort_keys=True))) % 10**8}.json"; p.write_text(json.dumps(obj)); return C.read_tifa160_tifa(p, image_of)
    # (a) per-rater records, worker ids, metric fields alongside
    recs = [{"image": "coco_12_vq_diffusion.jpg", "worker_id": "A1Z9", "score": 4}, {"image": "coco_12_vq_diffusion.jpg", "worker_id": "B7Q", "score": 1},
            {"image": "coco_1_vq_diffusion.jpg", "worker_id": "A1Z9", "score": 3}]
    r = run(recs); assert r["human"] == {_img("coco_12", "vq_diffusion"): 2.5, _img("coco_1", "vq_diffusion"): 3.0}   # no prefix confusion coco_1 / coco_12
    assert r["meta"]["ratings_per_image"] == {1: 1, 2: 1}
    # (b) file-name keys; raw list + avg + metrics + counts -> the raw list
    r = run({"coco_1_stable_diffusion_v1_5.jpg": {"text": "a cat", "human_scores": [5, 2], "human_avg": 3.5, "tifa_score": 0.9, "clip_score": 0.3, "n_annotators": 2}})
    assert r["human"] == {_img("coco_1", "stable_diffusion_v1_5"): 3.5} and "tier1:list:human_scores.[*]" in r["meta"]["rating_paths"]
    # (c) id -> {text, images -> model alias -> annotations[{worker, rating}]}
    r = run({"partiprompt_3": {"text": "a dog", "images": {"minDALL-E": {"annotations": [{"worker": "w1", "rating": 2}, {"worker": "w2", "rating": 4}]},
                                                            "SD v2.1": {"annotations": [{"worker": "w1", "rating": 5}]}}}})
    assert r["human"] == {_img("partiprompt_3", "mini_dalle"): 3.0, _img("partiprompt_3", "stable_diffusion_v2_1"): 5.0}
    # (d) DSG-style stems; scores keyed by worker; (e) metric-only entry -> located, no rating
    r = run({"coco_12_sd1dot1": {"scores": {"w1": 4, "w2": 3}}, "coco_1_sd1dot1": {"clip_score": 0.2}})
    assert r["human"] == {_img("coco_12", "stable_diffusion_v1_1"): 3.5} and r["meta"]["images_without_rating"] == 1 and r["unparsed_entries"] == 1
    # (f) list of records with id + model fields; bare per-image number under a wrapper key
    r = run({"data": [{"id": "coco_1", "model": "stable_diffusion_v2_1", "human_score": 1}], "coco_12_mini_dalle": 4})
    assert r["human"] == {_img("coco_1", "stable_diffusion_v2_1"): 1.0, _img("coco_12", "mini_dalle"): 4.0}
    # structure summary never carries a value
    def leaves(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)
        else:
            yield x
    r = run(recs); s = r["meta"]["structure"]
    assert all(isinstance(v, (str, int, type(None))) for v in leaves(s)) and "A1Z9" not in json.dumps(s) and "4" not in json.dumps(s).replace("list_len", "")
