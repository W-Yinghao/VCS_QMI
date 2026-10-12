"""r7 R7-30 D1-30 smoke on the R7-20 fixtures only (no DEV, TextVQA-val or ChartQA read).  One LLaVA host process:
  sets      : gqa_fixture, textvqa_fixture (200 q each, already read in R7-20) and chartqa_proxy (the GQA fixture with the ChartQA suffix and
              the relaxed-accuracy evaluator; no ChartQA data); sets64 = the first 64 items of each in runner order (full grid)
  selectors : the D1-20 smoke checkpoints (torch 2.9 main env; loaded here in torch 2.1.2), relabelled as the D1-30 unit tags for seeds 0-2
              (the same weights; A1 / A2 copies with the ablation's cfg flag) with dynamic tau calibrated on the fixture (smoke only)
  checks    : (a) user turn: user_segments(q + suffix) == LlavaPrefill.segments(q) token ids; (b) selector inputs: PCA'd visual tokens vs the
              D1-20 smoke cache (same images, same chunking) and question ids vs its question_ids.jsonl; (c) full-model reproduction: 'full' on
              the GQA fixture and 'full_ocr' on the TextVQA fixture against the R7-20 official-path answers
  grid      : every D1-30 configuration on sets64 (3 benchmarks); timing on sets64 (any GPU, labelled; B 1 / 4, plus 16 / 32 on GQA);
              DEV-risk code path on the D1-10 probe records (fixture binary questions)
SparseVLM rows and the scorer run afterwards (slurm/r7_d1_30_gpu.sbatch smoke).
    python scripts/r7_d1_30_smoke.py --out O
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_data as DD  # noqa: E402
import r7_d1_30_devrisk as DR  # noqa: E402
import r7_d1_30_plan as PL  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_timing as TM  # noqa: E402
import r7_vl_common as V  # noqa: E402

D1 = RUN.D1
LR = 3e-4
SMOKE_BENCHES = ("gqa_fixture", "textvqa_fixture", "chartqa_proxy")


def smoke_selectors(torch, src: Path, out: Path):
    """D1-20 smoke checkpoints relabelled as D1-30 unit tags (seeds 0-2 share weights; A1 / A2 = cfg flag copies).  Returns lr_choice."""
    out.mkdir(parents=True, exist_ok=True); choice = {o: {"chosen_lr": LR, "smoke": True} for o in RUN.OBJECTIVES}
    main, abl = RUN.unit_names(choice)
    for o in RUN.OBJECTIVES:
        ck = torch.load(str(src / f"smoke_{o}.pt"), map_location="cpu", weights_only=False)
        for u in main[o]:
            torch.save(ck, out / f"{u}.pt")
    for tag, o, flag in (("A1", "vcs", {"no_question": True}), ("A2", "direct", {"target": "d"})):
        ck = torch.load(str(src / f"smoke_{o}.pt"), map_location="cpu", weights_only=False); ck = dict(ck, cfg=dict(ck["cfg"], **flag))
        for u in abl[tag]:
            torch.save(ck, out / f"{u}.pt")
    (out / "lr_choice.json").write_text(json.dumps(choice, indent=1))
    return choice


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--train-smoke", default=str(D1 / "train_smoke" / "NVIDIA_L40S")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; t0 = time.time()
    rep["sets"] = DD.fixture_sets(out / "sets")
    import r7_d1_teacher as TE
    host = TE.LlavaPrefill(out / "pred"); torch = host.torch; rep["gpu"] = torch.cuda.get_device_name(0); rep["load_seconds"] = host.load_seconds
    choice = smoke_selectors(torch, Path(a.train_smoke), out / "selectors"); main_u, abl_u = RUN.unit_names(choice)
    units = [u for o in RUN.OBJECTIVES for u in main_u[o]] + [u for k in ("A1", "A2") for u in abl_u[k]]
    pca = D1 / "cache_smoke" / "pca.pt"
    # sets64 (runner order) for the full grid
    (out / "sets64").mkdir(exist_ok=True)
    for b in SMOKE_BENCHES:
        items = RUN.load_items(out / "sets", b)[: a.n]; (out / "sets64" / f"{b}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    # dynamic tau on the GQA fixture subset (smoke only)
    bank = RUN.SelectorBank(out / "selectors", pca, units); gq = RUN.load_items(out / "sets64", "gqa_fixture")
    feats = RUN.encode_unique(host, gq, lambda fp: __import__("PIL.Image", fromlist=["Image"]).open(fp).convert("RGB"))
    st = bank.prepare(feats, RUN.question_embeddings(host, [it["question"] for it in gq])); taus = {}
    for u in units:
        if u.endswith(("_noq", "_dtarget")):
            continue
        t = RUN.calibrate_tau(bank.risks(u, st)); taus[u] = t; bank.tau[u] = dict(t)
        (out / "selectors" / f"{u}.json").write_text(json.dumps({"dynamic": {str(f): {"tau": v, "smoke_fixture_calibration": True} for f, v in t.items()}}, indent=1))
    rep["smoke_tau"] = taus
    # (a) user turn equality
    fx = [json.loads(x) for x in (DD.FIXTURES / "gqa.jsonl").read_text().splitlines() if x.strip()]; eq = 0
    for r in fx:
        p1, q1 = RUN.user_segments(host, r["question"] + DD.SHORT); p2, q2 = host.segments(r["question"], True)
        eq += int(torch.equal(p1, p2) and torch.equal(q1, q2))
    rep["check_user_turn_equal"] = f"{eq}/{len(fx)}"
    # (b) selector inputs vs the D1-20 smoke cache
    from PIL import Image
    ids = json.loads((D1 / "cache_smoke" / "images_PROBE.json").read_text()); files = {r["image_id"]: r["image_file"] for r in fx}
    ref = np.load(D1 / "cache_smoke" / "feats_PROBE.npy").astype(np.float32); diffs = []
    for c in range(0, len(ids), 16):
        f = host.encode([Image.open(files[i]).convert("RGB") for i in ids[c:c + 16]]); v = bank.prepare(f, [torch.zeros(1, 4096, device="cuda")] * f.shape[0])["v"]
        diffs.append(np.abs(v.cpu().numpy() - ref[c:c + 16]))
    d = np.concatenate([x.reshape(-1) for x in diffs]); qref = {r["qid"]: r["ids"] for r in V.read_jsonl(D1 / "cache_smoke" / "question_ids.jsonl")}
    qeq = sum(host.tok(r["question"], add_special_tokens=False).input_ids == qref[r["qid"]] for r in fx if r["qid"] in qref)
    rep["check_selector_inputs"] = {"visual_max_abs_diff": float(d.max()), "visual_frac_exact": float(np.mean(d == 0)), "question_ids_equal": f"{qeq}/{len(qref)}"}
    # (c) full-model reproduction against R7-20 (official LLaVA path, batch 1, max_new_tokens 128)
    (out / "configs").mkdir(exist_ok=True); (out / "configs" / "repro_gqa.txt").write_text("full\n"); (out / "configs" / "repro_textvqa.txt").write_text("full_ocr\n")
    for b, cf in (("gqa_fixture", "repro_gqa.txt"), ("textvqa_fixture", "repro_textvqa.txt")):
        RUN.run(Namespace(sets_dir=str(out / "sets"), bench=b, configs_file=str(out / "configs" / cf), shard="0/1", out=str(out / "pred_repro"),
                          chunk=32, batch=32, limit=0), host=host, bank=bank)
    r720 = {}
    for x in (Path("/home/infres/yinwang/CS_QMI/outputs/R7_fixtures/d1/pred_full.jsonl")).read_text().splitlines():
        j = json.loads(x); r720[(j["fixture"], j["qid"])] = j["text"]
    for b, cfg, fxn in (("gqa_fixture", "full", "gqa"), ("textvqa_fixture", "full_ocr", "textvqa")):
        mine = {r["qid"]: r["pred"] for p in RUN.cfg_paths(out / "pred_repro" / b, cfg) for r in V.read_jsonl(p)}
        same = sum(mine[q] == r720[(fxn, q)] for q in mine if (fxn, q) in r720)
        rep[f"check_repro_{b}"] = f"{same}/{sum((fxn, q) in r720 for q in mine)} identical to R7-20 answers"
    # full grid on sets64
    cfgs = PL.write_configs(choice, out / "configs"); rep["configs"] = cfgs
    for b in SMOKE_BENCHES:
        src_b = "textvqa_val" if b.startswith("textvqa") else ("chartqa_val" if b.startswith("chartqa") else "gqa_dev")
        cf = out / "configs" / f"configs_{src_b}.txt"
        RUN.run(Namespace(sets_dir=str(out / "sets64"), bench=b, configs_file=str(cf), shard="0/1", out=str(out / "pred"), chunk=32, batch=32, limit=0),
                host=host, bank=bank)
    # timing (any GPU; labelled smoke)
    for b in SMOKE_BENCHES:
        src_b = "textvqa_val" if b.startswith("textvqa") else ("chartqa_val" if b.startswith("chartqa") else "gqa_dev")
        bs = ["1", "4", "16", "32"] if b == "gqa_fixture" else ["1", "4"]
        TM.main(["--sets-dir", str(out / "sets64"), "--bench", b, "--configs-file", str(out / "configs" / f"timing_{src_b}.txt"), "--out", str(out / "timing" / b),
                 "--n", str(a.n), "--n-big", str(a.n), "--n1", "16", "--batches", *bs, "--allow-any-gpu"], host=host, bank=bank)
    # DEV-risk code path on the probe records (fixture binary questions)
    DR.main(["--out", str(out / "devrisk"), "--records", str(D1 / "smoke" / "probe" / "probe_records.jsonl"), "--images-from-fixture", str(DD.FIXTURES / "gqa.jsonl"),
             "--selectors-dir", str(out / "selectors"), "--pca", str(pca), "--reps", "200", "--units", main_u["vcs"][0], main_u["logistic"][0], main_u["direct"][0],
             abl_u["A1"][0], abl_u["A2"][0]], host=host)
    rep["seconds"] = time.time() - t0; rep["env"] = V.env_info(); (out / "smoke_report_llava.json").write_text(V.jdump(rep, indent=1)); print(V.jdump(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
