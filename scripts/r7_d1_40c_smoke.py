"""r7 R7-40c smoke on NON-SEALED data only (code checks; no DEV / TextVQA-val / ChartQA-val / test file, no D1-30 / R7-40b result is read):
the R7-20 fixtures (gqa_fixture, textvqa_fixture: already read in R7-20, excluded from the D1-30 sets; chartqa_proxy = the GQA fixture with
the ChartQA suffix) and a ChartQA TRAIN subset (32 human + 32 augmented rows, images extracted) for chart-image token counts and costs.
Selectors: the R7-40b smoke checkpoints (trained on the fixture probe records, cache_smoke PCA) relabelled as unit tags for seeds 0-2; dynamic
taus calibrated on the fixture (smoke only).  Numbers are pipeline checks, never results.
  prep  (CPU, SparseVLM env: pyarrow)  : sets/{gqa_fixture,textvqa_fixture,chartqa_proxy,chartqa_train_smoke}.jsonl + sets64/ (first 64 items
                                         in runner order) + a smoke protocol file carrying the freeze key for the scorer smoke
  gpu   (vl_baselines env, one GPU)    : (a) prompt / host interface on every smoke item (spliced ids == the processor's input_ids, N == header
        estimate); (b) full row: production batched path (batch 32) vs the official HF path (processor + model.generate, batch 1, same GPU)
        and vs the R7-20 answers; (c) pruned rows (grid / S1 fixed / S1 dynamic): production batched path vs HF generate on the same spliced
        sequence with the spliced rotary ids injected (batch 1, HF's own decoding loop, penalty and cache) and vs QwenCore at batch 1;
        (d) batched selection (timing form) == per-item selection (runner form); (e) the configuration grid through the production runner
        (all 76 rows on GQA, the 50 core rows on TextVQA / ChartQA-train) + resume; (f) worst-case memory (batch 32 × 1,280-token images,
        128 forced new tokens; timing's one-call tower on 32 such images); (g) timing rows on fixture subsets (any GPU, labelled, stand-in B*);
        (h) the DEV-risk code path on the R7-40a probe records (fixture binary questions)
  throughput (one GPU, after gpu)      : the runner's cost with the core rows on the GPU class the job lands on (+ thread / CPU facts), the
                                         anchor's cost and a GPU check of the timing bundle (smoke paths)
  anchor (one GPU, after gpu)          : decision 15, the batch-1 official full-row anchor: production function, equality with the frozen
                                         official_answer, agreement with the batched full row on the same GPU, resume, cost
  report (CPU, SparseVLM env)          : scorer on the smoke predictions (latency basis with the stand-in B*, fraction basis with none) and a
                                         summary -> <smoke>/smoke_summary.json
    python -I scripts/r7_d1_40c_smoke.py prep|gpu|throughput|anchor|report --out O
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_40c_plan as PL  # noqa: E402
import r7_vl_common as V  # noqa: E402

SM = PL.R40 / "smoke_40c"
FIXTURES = Path("/projects/EEG-foundation-model/yinghao/data/r7_fixtures")
CHART_TRAIN = Path("/projects/EEG-foundation-model/yinghao/data/chartqa/data/train-00000-of-00003.parquet")
R720_QWEN = Path("/home/infres/yinwang/CS_QMI/outputs/R7_fixtures/d1/pred_qwen25vl7b.jsonl")
TRAIN_SMOKE = PL.R40 / "train_smoke" / "NVIDIA_L40S"                  # R7-40b smoke selectors (fixture probe records; not trained units)
PCA_SMOKE = PL.R40 / "cache_smoke" / "pca.pt"
PROBE_RECORDS = PL.R40 / "probe" / "probe_records.jsonl"
SMOKE_LR = 3e-4
BENCH_ROWS = {"gqa_fixture": "all", "textvqa_fixture": "core", "chartqa_train_smoke": "core"}
SMOKE_B_STAR = 16                                                      # stand-in (the indicative L40S value of R7-40a); smoke only
TP_ITEMS = 32                                                          # items per benchmark in the consolidated L40S job (50-min limit, >= 2x margin)


# ------------------------------------------------------------------------------------------------------------------------------ prep (CPU)
def chartqa_train_subset(out: Path, per_split=32):
    """The first per_split human and per_split augmented rows of ChartQA TRAIN shard 0 in file order -> D1-30 item format, images extracted
    once (original bytes) by the frozen D1-30 helpers; ids chartqa_train_smoke_<i>.  Never val / test."""
    import pyarrow.parquet as pq
    import r7_d1_30_data as DD
    take = {"human": [], "augmented": []}
    for b in pq.ParquetFile(CHART_TRAIN).iter_batches(batch_size=64, columns=["imgname", "query", "label", "type", "image"]):
        for r in b.to_pylist():
            t = str(r["type"]).lower(); k = "human" if "human" in t else ("augmented" if "augmented" in t else None)
            if k and len(take[k]) < per_split:
                take[k].append(r)
        if all(len(v) >= per_split for v in take.values()):
            break
    rows = take["human"] + take["augmented"]; paths = {}
    for r in rows:
        DD.write_image("chartqa_train_smoke", r["imgname"], r.pop("image"), out, paths)
    items = DD.chartqa_items(rows)
    for i, it in enumerate(items):
        it.update(bench="chartqa_train_smoke", qid=f"chartqa_train_smoke_{i:05d}", image_file=paths[it["image_id"]])
    (out / "chartqa_train_smoke.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    return len(items)


def cmd_prep(a):
    import r7_d1_30_data as DD
    out = Path(a.out); sets = out / "sets"; sets.mkdir(parents=True, exist_ok=True); rep = {"sets": DD.fixture_sets(sets)}
    rep["sets"]["chartqa_train_smoke"] = chartqa_train_subset(sets)
    (out / "sets64").mkdir(exist_ok=True)
    for b in ("gqa_fixture", "textvqa_fixture", "chartqa_proxy", "chartqa_train_smoke"):
        items = RUN.load_items(sets, b)[: a.n]; (out / "sets64" / f"{b}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    (out / "smoke_protocol.yaml").write_text("# smoke only: lets the R7-40c scorer run on the smoke predictions (never the production protocol)\n"
                                             "r7_40c_staged_freeze: smoke\n")
    (out / "prep.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ gpu helpers
def smoke_selectors(torch, out: Path):
    """R7-40b smoke checkpoints relabelled as the unit tags of seeds 0-2 (the same weights) + a smoke lr_choice.json."""
    out.mkdir(parents=True, exist_ok=True); choice = {o: {"chosen_lr": SMOKE_LR, "smoke": True} for o in PL.OBJECTIVES}; main = PL.unit_names(choice)
    for o in PL.OBJECTIVES:
        for u in main[o]:
            shutil.copyfile(TRAIN_SMOKE / f"smoke_{o}.pt", out / f"{u}.pt")
    (out / "lr_choice.json").write_text(json.dumps(choice, indent=1))
    return choice, main


def calibrate_tau(R, op):
    """Smoke only: the bisection of r7_d1_40_selectors.dynamic_tau40 on risks R [n, 5] (nominal fractions)."""
    fr = np.asarray(PL.DYN_GRID)

    def mean_frac(t):
        ok = R <= t; ok[:, -1] = True
        return float(fr[np.argmax(ok, 1)].mean())
    lo, hi = float(R.min()) - 1.0, float(R.max()) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if mean_frac(mid) > op else (lo, mid)
    return hi, mean_frac(hi)


def hf_generate_spliced(host, seq, pos, pid, max_new):
    """HF generate (its own decoding loop, logits processors incl. the shipped repetition penalty, stopping criteria and KV cache) on a spliced
    sequence at batch 1, with the spliced 3-D rotary ids injected through get_rope_index (test-only hook: HF generate cannot otherwise take ids
    for a sequence whose image tokens were pruned).  Decoding positions follow HF's own rope-delta rule (cache position + delta)."""
    torch = host.torch; m = host.model; L = int(seq.shape[0]); dev = seq.device
    pos3 = torch.as_tensor(pos, device=dev).view(3, 1, L); delta = torch.tensor([[int(pos.max()) + 1 - L]], device=dev)
    m.model.get_rope_index = lambda *args, **kw: (pos3, delta); m.model.rope_deltas = None
    try:
        with torch.inference_mode():
            out = m.generate(input_ids=torch.as_tensor(np.asarray(pid), dtype=torch.long, device=dev)[None], inputs_embeds=seq[None],
                             attention_mask=torch.ones(1, L, dtype=torch.long, device=dev), do_sample=False, num_beams=1, max_new_tokens=max_new, use_cache=True)
    finally:
        del m.model.get_rope_index; m.model.rope_deltas = None
    return host.decode([out[0, L:].tolist()])[0]


def states_for(host, items, opener):
    import r7_d1_40c_run as RN
    enc = {}
    for it in items:
        if it["image_file"] not in enc:
            enc[it["image_file"]] = host.encode([opener(it["image_file"])])[0]
    return [RN.ItemState(host, it, enc[it["image_file"]]) for it in items]


def batched_answers(host, states, cfg, bank, max_new, batch=32):
    import r7_d1_40c_run as RN
    sel = [RN.kept_for(cfg, s, host, bank) for s in states]; texts = []
    for b0 in range(0, len(states), batch):
        part = list(range(b0, min(b0 + batch, len(states)))); seqs, poss, pids = zip(*[RN.assemble(host, states[j], sel[j][0]) for j in part])
        new, _ = host.core.generate(list(seqs), list(poss), list(pids), max_new_tokens=max_new); texts += host.decode(new)
    return texts, sel


def check_interface(host, items, opener):
    """(a) the spliced prompt == the processor's input_ids (pre + N image tokens + post), N == the header estimate, for every item."""
    import r7_d1_40c_run as RN
    ok = {"ids_equal": 0, "n_equals_header_estimate": 0, "items": len(items)}
    for it in items:
        img = opener(it["image_file"]); text = host.chat_text(it["prompt"], True)
        ids = host.proc(text=[text], images=[img], return_tensors="pt")["input_ids"][0].numpy()
        pre, post = host.segments(it["prompt"], True); n = int((ids == host.image_token_id).sum())
        ok["ids_equal"] += int(np.array_equal(np.concatenate([pre, np.full(n, host.image_token_id), post]), ids))
        r, c = RN.merged_grid(it["image_file"]); ok["n_equals_header_estimate"] += int(r * c == n)
    return ok


def memprobe(host, bank, main):
    """(f) worst cases: the runner (per-image tower, generation batch 32 of 1,280-token images with a long prompt, 128 forced new tokens, plus
    the selector forwards of a dynamic row) and the timing's one-call tower over 32 such images + prefill + 8 decode steps."""
    from PIL import Image
    import r7_d1_40c_run as RN
    torch = host.torch; g = np.random.default_rng(0); res = {"gpu_total_gb": V.env_info().get("gpu_mem_gb")}
    imgs = [Image.fromarray(g.integers(0, 255, (896, 1120, 3), dtype=np.uint8)) for _ in range(32)]      # 32 × 40 merged grid = 1,280 tokens
    q = "Considering every labelled bar, line and legend entry in the chart, what is the difference between the largest and the smallest " \
        "value shown for the most recent year, and in which category does the second largest value appear across all of the groups shown?"
    items = [{"qid": f"mem{i}", "image_id": f"mem{i}", "image_file": "", "question": q, "prompt": q + "\nAnswer the question with a single word."} for i in range(32)]
    enc = states = seqs = poss = pids = None
    try:
        V.cuda_peak_reset(torch); enc = [host.encode([im])[0] for im in imgs]; states = [RN.ItemState(host, it, e) for it, e in zip(items, enc)]
        res["tokens_per_image"] = states[0].n
        for s in states:
            RN.kept_for(f"dyn:{main['vcs'][0]}:0.5", s, host, bank)
        seqs, poss, pids = zip(*[RN.assemble(host, s, None) for s in states])
        host.core.generate(list(seqs), list(poss), list(pids), max_new_tokens=128, min_new_tokens=128)
        res["runner_peak_gb"] = V.cuda_peak_gb(torch); res["runner_seq_tokens"] = int(seqs[0].shape[0]); res["runner_oom"] = False
    except torch.cuda.OutOfMemoryError as e:
        res["runner_oom"] = True; res["runner_oom_message"] = str(e).splitlines()[0][:300]; res["runner_peak_gb"] = V.cuda_peak_gb(torch)
    enc = states = seqs = poss = pids = None; torch.cuda.empty_cache()                   # free the runner probe's tensors before the timing probe
    try:
        V.cuda_peak_reset(torch); enc2 = host.encode(imgs); res["timing_tower_peak_gb"] = V.cuda_peak_gb(torch)
        states = [RN.ItemState(host, it, e) for it, e in zip(items, enc2)]; seqs, poss, pids = zip(*[RN.assemble(host, s, None) for s in states])
        host.core.last_logits(list(seqs), list(poss)); host.core.generate(list(seqs), list(poss), list(pids), max_new_tokens=8, min_new_tokens=8)
        res["timing_peak_gb"] = V.cuda_peak_gb(torch); res["timing_oom"] = False
    except torch.cuda.OutOfMemoryError as e:
        res["timing_oom"] = True; res["timing_oom_message"] = str(e).splitlines()[0][:300]; res["timing_peak_gb"] = V.cuda_peak_gb(torch)
    torch.cuda.empty_cache()
    return res


def partial(out: Path, rep):
    """The GPU smoke's report so far (a time-limit kill keeps the finished steps)."""
    (out / "smoke_gpu.partial.json").write_text(V.jdump(rep, indent=1))


def cmd_gpu(a):
    import r7_d1_40_host as QH
    import r7_d1_40c_devrisk as DRK
    import r7_d1_40c_run as RN
    import r7_d1_40c_timing as TMG
    out = Path(a.out); sets, sets64 = out / "sets", out / "sets64"; rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; t0 = time.time()
    host = QH.QwenPrefill(out); torch = host.torch; rep.update(gpu=torch.cuda.get_device_name(0), load_seconds=host.load_seconds, stage_notes=host.stage_notes, host=host.info)
    choice, main = smoke_selectors(torch, out / "selectors"); units = [u for o in PL.OBJECTIVES for u in main[o]]
    bank = RN.QwenSelectorBank(out / "selectors", PCA_SMOKE, units); opener = RN.open_image
    load = lambda d, b: [{k: v for k, v in it.items() if k not in RN.ANSWER_KEYS} for it in RUN.load_items(d, b)]  # noqa: E731
    gq = states_for(host, load(sets64, "gqa_fixture"), opener); taus = {}
    for u in units:                                                                    # smoke-only CAL stand-in: taus on the GQA fixture subset
        R = np.stack([bank.risks(u, s.selector_state(host, bank)) for s in gq]); taus[u] = {op: calibrate_tau(R, float(op)) for op in PL.OPS}
        (out / "selectors" / f"{u}.json").write_text(json.dumps({"dynamic": {op: {"tau": t, "mean_kept_fraction": m, "smoke_fixture_calibration": True}
                                                                              for op, (t, m) in taus[u].items()}}, indent=1))
        bank.tau[u] = {op: t for op, (t, _) in taus[u].items()}
    rep["smoke_tau"] = taus; t1 = time.time()
    # (a) host interface on every smoke item
    rep["a_interface"] = {b: check_interface(host, load(sets, b), opener) for b in ("gqa_fixture", "textvqa_fixture", "chartqa_proxy", "chartqa_train_smoke")}
    rep["seconds_a"] = time.time() - t1; t1 = time.time(); partial(out, rep)
    # (b) full row: production batched vs official HF path (batch 1, same GPU) and vs R7-20
    r720 = {(r["fixture"], r["qid"]): r["text"] for r in V.read_jsonl(R720_QWEN)}; rep["b_full_row"] = {}
    for b, fx in (("gqa_fixture", "gqa"), ("textvqa_fixture", "textvqa"), ("chartqa_train_smoke", None)):
        its = load(sets, b); st = states_for(host, its, opener); mine, _ = batched_answers(host, st, "full", bank, PL.MAX_NEW[b])
        off = [host.official_answer(it["prompt"], opener(it["image_file"]), max_new_tokens=PL.MAX_NEW[b])["text"] for it in its]
        rec = {"items": len(its), "batched_vs_official_same_gpu": float(np.mean([x == y for x, y in zip(mine, off)]))}
        if fx:
            rec["official_vs_r720"] = float(np.mean([y == r720[(fx, it["qid"])] for y, it in zip(off, its)]))
            rec["batched_vs_r720"] = float(np.mean([x == r720[(fx, it["qid"])] for x, it in zip(mine, its)]))
        rep["b_full_row"][b] = rec
    rep["seconds_b"] = time.time() - t1; t1 = time.time(); partial(out, rep)
    # (c) pruned rows: production batched vs HF generate with injected ids (batch 1) and vs QwenCore batch 1
    rep["c_pruned_rows"] = {}
    for b, n_items, cfgs in (("gqa_fixture", 200, ["static:grid:0.25", f"sel:{main['vcs'][0]}:0.25", f"dyn:{main['vcs'][0]}:0.5"]),
                             ("textvqa_fixture", 64, ["static:grid:0.25", f"sel:{main['vcs'][0]}:0.5"])):
        its = load(sets, b)[:n_items]; st = states_for(host, its, opener)
        for cfg in cfgs:
            mine, sel = batched_answers(host, st, cfg, bank, PL.MAX_NEW[b]); hf, one = [], []
            for s, (kk, _) in zip(st, sel):
                seq, pos, pid = RN.assemble(host, s, kk); hf.append(hf_generate_spliced(host, seq, pos, pid, PL.MAX_NEW[b]))
                new, _ = host.core.generate([seq], [pos], [pid], max_new_tokens=PL.MAX_NEW[b]); one.append(host.decode(new)[0])
            rep["c_pruned_rows"][f"{b}|{cfg}"] = {"items": len(its), "batched_vs_hf_injected_batch1": float(np.mean([x == y for x, y in zip(mine, hf)])),
                                                  "core_batch1_vs_hf_injected_batch1": float(np.mean([x == y for x, y in zip(one, hf)])),
                                                  "batched_vs_core_batch1": float(np.mean([x == y for x, y in zip(mine, one)])),
                                                  "mean_kept_fraction": float(np.mean([1.0 if kk is None else len(kk) / s.n for s, (kk, _) in zip(st, sel)]))}
        if b == "gqa_fixture":                                                         # the full row through the injected path as well
            hf = []; mine, _ = batched_answers(host, st, "full", bank, PL.MAX_NEW[b])
            for s in st:
                seq, pos, pid = RN.assemble(host, s, None); hf.append(hf_generate_spliced(host, seq, pos, pid, PL.MAX_NEW[b]))
            rep["c_pruned_rows"][f"{b}|full"] = {"items": len(st), "batched_vs_hf_injected_batch1": float(np.mean([x == y for x, y in zip(mine, hf)]))}
    rep["seconds_c"] = time.time() - t1; t1 = time.time(); partial(out, rep)
    # (d) batched selection (timing form) == per-item selection (runner form)
    d = {}
    for cfg in (f"sel:{main['vcs'][0]}:0.25", f"sel:{main['direct'][0]}:0.5", f"dyn:{main['vcs'][0]}:0.5", f"dyn:{main['logistic'][1]}:0.25"):
        kind, u, f = PL.parse(cfg); agree = 0
        for c0 in range(0, len(gq), 16):
            part = gq[c0:c0 + 16]; kb, fb = bank.select_batch(u, [s.selector_state(host, bank) for s in part], f=f if kind == "sel" else None, op=f if kind == "dyn" else None)
            for s, k1, f1 in zip(part, kb, fb):
                k0, f0 = RN.kept_for(cfg, s, host, bank); k0 = np.arange(s.n) if k0 is None else k0
                agree += int(f0 == f1 and np.array_equal(k0, k1))
        d[cfg] = f"{agree}/{len(gq)}"
    rep["d_batched_selection_equals_per_item"] = d; rep["seconds_d"] = time.time() - t1; t1 = time.time(); partial(out, rep)
    # (e) configuration grid through the production runner + resume
    cdir = out / "configs"; rep["e_grid"] = {}
    for b, rows in BENCH_ROWS.items():
        res = PL.write_configs(choice, cdir / rows, rows, benches=[b]); cf = cdir / rows / f"configs_{b}.txt"
        RN.run(Namespace(sets_dir=str(sets64), bench=b, configs_file=str(cf), shard="0/1", out=str(out / "pred"), chunk=32, batch=32, limit=0,
                         selectors_dir=str(out / "selectors"), pca=str(PCA_SMOKE)), host=host, bank=bank)
        mk = json.loads((out / "pred" / b / "done_shard0of1.json").read_text()); before = sum(1 for _ in (out / "pred" / b).glob("*.jsonl"))
        RN.run(Namespace(sets_dir=str(sets64), bench=b, configs_file=str(cf), shard="0/1", out=str(out / "pred"), chunk=32, batch=32, limit=0,
                         selectors_dir=str(out / "selectors"), pca=str(PCA_SMOKE)), host=host, bank=bank)
        lines = sum(len((out / "pred" / b / f"{RUN.cfg_file(c)}@s0of1.jsonl").read_text().splitlines()) for c in PL.read_configs(cf))
        rep["e_grid"][b] = {"rows": res[b]["configs"], "items": mk["items"], "answers": mk["answers_this_session"], "seconds": mk["seconds"],
                            "seconds_per_answer": mk["seconds_per_answer"], "resume_lines_total": lines, "expected_lines": res[b]["configs"] * mk["items"],
                            "files": before, "hit_max_answers": sum(json.loads(x).get("hit_max", False) for c in PL.read_configs(cf)
                                                                    for x in (out / "pred" / b / f"{RUN.cfg_file(c)}@s0of1.jsonl").read_text().splitlines())}
    rep["seconds_e"] = time.time() - t1; t1 = time.time(); partial(out, rep)
    # (f) worst-case memory
    rep["f_memory"] = memprobe(host, bank, main); rep["seconds_f"] = time.time() - t1; t1 = time.time(); partial(out, rep)
    # (g) timing rows on fixture subsets (not the reference GPU; stand-in B*)
    rep["g_timing"] = {}
    for b, bs, n in (("gqa_fixture", ["1", "4", "16", "32"], "64"), ("textvqa_fixture", ["4", "16"], "32")):
        cf = cdir / "core" / f"timing_{b}.txt"
        if not cf.exists():
            PL.write_configs(choice, cdir / "core", "core", benches=[b])
        TMG.main(["--sets-dir", str(sets64), "--bench", b, "--configs-file", str(cf), "--out", str(out / "timing" / b), "--n", n, "--n-big", n, "--n1", "16",
                  "--batches", *bs, "--allow-any-gpu", "--b-star", str(SMOKE_B_STAR), "--selectors-dir", str(out / "selectors"), "--pca", str(PCA_SMOKE)], host=host, bank=bank)
        j = json.loads((out / "timing" / b / f"timing_{b}.json").read_text())
        rep["g_timing"][b] = {"seconds": j["seconds"], "rows": [{k: r.get(k) for k in ("config", "B", "pool", "n_batches", "per_item_ms_median", "vision_share_of_per_item",
                                                                                        "mean_kept_fraction", "peak_mem_gb")} for r in j["rows"]]}
    rep["seconds_g"] = time.time() - t1; t1 = time.time(); partial(out, rep)
    # (h) DEV-risk code path on the probe records (fixture binary questions)
    DRK.main(["smoke", "--out", str(out / "devrisk"), "--records", str(PROBE_RECORDS), "--fixture", str(FIXTURES / "gqa.jsonl"), "--selectors-dir", str(out / "selectors"),
              "--pca", str(PCA_SMOKE), "--units", main["vcs"][0], main["logistic"][0], main["direct"][0]], host=host)
    dj = json.loads((out / "devrisk" / "r7_40c_devrisk_smoke.json").read_text())
    rep["h_devrisk"] = {"coverage": dj["coverage"], "units": {u: {k: v[k] for k in ("n_pairs", "n_images")} for u, v in dj["units"].items()}, "seconds": dj["seconds"]}
    rep["seconds_h"] = time.time() - t1; rep["seconds"] = time.time() - t0; rep["env"] = V.env_info()
    (out / "smoke_gpu.json").write_text(V.jdump(rep, indent=1)); print(V.jdump(rep, indent=1))
    return 0


def cmd_throughput(a):
    """On the GPU class this job lands on (L40S, the decided quality class): (1) the production runner with the CORE rows on the three smoke
    subsets (sets64; selectors and taus from the gpu step) into pred_<gpu>/, with the time inside QwenCore.generate and the CPU / thread facts;
    (2) a sorted batch-32 full-row prefill; (3) the batch-1 official full-row anchor (decision 15): the production function run_official, its
    answers against the frozen QwenPrefill.official_answer on every item, against the batched full row of (1) (same GPU), resume, cost per item;
    (4) the nodesumo01 timing bundle's code path in the decided BSTAR=report mode with a stand-in report (smoke paths only).  Code checks and
    costs only."""
    import os
    import platform
    import r7_d1_40_host as QH
    import r7_d1_40c_run as RN
    out = Path(a.out); host = QH.QwenPrefill(out); torch = host.torch; gpu = torch.cuda.get_device_name(0); tag = gpu.replace(" ", "_")
    rep = {"gpu": gpu, "load_seconds": host.load_seconds, "torch_threads": torch.get_num_threads(), "affinity_cpus": len(os.sched_getaffinity(0)),
           "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor()),
           "host": platform.node(), "benches": {}}
    choice = json.loads((out / "selectors" / "lr_choice.json").read_text()); main = PL.unit_names(choice); units = [u for o in PL.OBJECTIVES for u in main[o]]
    bank = RN.QwenSelectorBank(out / "selectors", PCA_SMOKE, units); gen_t = [0.0, 0]; orig = host.core.generate

    def timed(*args, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = orig(*args, **kw); torch.cuda.synchronize(); gen_t[0] += time.perf_counter() - t0; gen_t[1] += 1
        return r
    host.core.generate = timed
    for b in ("gqa_fixture", "textvqa_fixture", "chartqa_train_smoke"):
        cdir = out / "configs" / f"core_{tag}"; PL.write_configs(choice, cdir, "core", benches=[b]); gen_t[0], gen_t[1] = 0.0, 0; t0 = time.time()
        RN.run(Namespace(sets_dir=str(out / "sets64"), bench=b, configs_file=str(cdir / f"configs_{b}.txt"), shard="0/1", out=str(out / f"pred_{tag}"), chunk=32, batch=32,
                         limit=TP_ITEMS, selectors_dir=str(out / "selectors"), pca=str(PCA_SMOKE)), host=host, bank=bank)
        mk = json.loads((out / f"pred_{tag}" / b / "done_shard0of1.json").read_text())
        rep["benches"][b] = {"answers": mk["answers_this_session"], "seconds": time.time() - t0, "seconds_per_answer": mk["seconds_per_answer"],
                             "generate_seconds": gen_t[0], "generate_calls": gen_t[1]}
    host.core.generate = orig
    from PIL import Image
    its = sorted(RN.load_items(out / "sets64", "gqa_fixture"), key=lambda it: RN.merged_grid(it["image_file"])[0] * RN.merged_grid(it["image_file"])[1])[:32]
    enc = [host.encode([Image.open(it["image_file"]).convert("RGB")])[0] for it in its]; st = [RN.ItemState(host, it, e) for it, e in zip(its, enc)]
    seqs, poss, pids = zip(*[RN.assemble(host, x, None) for x in st]); ts = []
    for _ in range(4):
        torch.cuda.synchronize(); t0 = time.perf_counter(); host.core.last_logits(list(seqs), list(poss)); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    rep["prefill_b32_sorted_gqa"] = {"tokens": int(sum(x.shape[0] for x in seqs)), "max_len": int(max(x.shape[0] for x in seqs)), "seconds_median": float(np.median(ts[1:]))}
    (out / f"smoke_throughput_{tag}.json").write_text(V.jdump(rep, indent=1)); print(V.jdump(rep, indent=1))
    rep["official_anchor"] = {}                     # decision 15 on this GPU: production function, equality with the frozen official_answer, batching
    for b in ("gqa_fixture", "textvqa_fixture", "chartqa_train_smoke"):              # effect vs the batched full row just generated here, resume, cost
        its = [{k: v for k, v in it.items() if k not in RN.ANSWER_KEYS} for it in RUN.load_items(out / "sets64", b)][:TP_ITEMS]; od = out / f"pred_{tag}" / b
        t0 = time.time(); RN.run_official(Namespace(sets_dir=str(out / "sets64"), bench=b, shard="0/1", out=str(out / f"pred_{tag}"), limit=TP_ITEMS), host=host)
        sec = time.time() - t0; f = od / f"{RUN.cfg_file(RN.OFFICIAL)}@s0of1.jsonl"; rows = {r["qid"]: r for r in V.read_jsonl(f)}; n_lines = len(f.read_text().splitlines())
        RN.run_official(Namespace(sets_dir=str(out / "sets64"), bench=b, shard="0/1", out=str(out / f"pred_{tag}"), limit=TP_ITEMS), host=host)   # resume: nothing redone
        ref = [host.official_answer(it["prompt"], RN.open_image(it["image_file"]), max_new_tokens=PL.MAX_NEW[b])["text"] for it in its]
        full = {r["qid"]: r["pred"] for r in V.read_jsonl(od / "full@s0of1.jsonl")}
        rep["official_anchor"][b] = {"items": len(its), "marker": (od / "done_official_shard0of1.json").exists(), "seconds_per_item": sec / max(len(its), 1),
                                     "anchor_equals_frozen_official_answer": float(np.mean([rows[it["qid"]]["pred"] == r for it, r in zip(its, ref)])),
                                     "anchor_equals_batched_full_same_gpu": float(np.mean([rows[it["qid"]]["pred"] == full[it["qid"]] for it in its])),
                                     "resume_lines_unchanged": len(f.read_text().splitlines()) == n_lines == len(its),
                                     "hit_max": int(sum(r["hit_max"] for r in rows.values()))}
        (out / f"smoke_throughput_{tag}.json").write_text(V.jdump(rep, indent=1)); print(V.jdump({b: rep["official_anchor"][b]}), flush=True)
    # the nodesumo01 timing bundle's code path on this GPU in the decided mode (BSTAR=report): a stand-in report (smoke path, never the production
    # report) is reused, one benchmark's timing rows on a tiny pool, the bundle marker
    import r7_d1_40c_timing as TMG
    tb = out / f"timing_bundle_{tag}"; (tb / "bstar").mkdir(parents=True, exist_ok=True); t0 = time.time()
    (tb / "bstar" / "timing_report.json").write_text(json.dumps({"gpu": gpu, "host": rep["host"], "B_star": SMOKE_B_STAR, "rule": "smoke stand-in, not a measurement"}))
    TMG.bundle(["--sets-dir", str(out / "sets64"), "--configs-dir", str(out / "configs" / f"core_{tag}"), "--out", str(tb), "--timing-report",
                str(tb / "bstar" / "timing_report.json"), "--b-star-mode", "report", "--benches", "gqa_fixture", "--selectors-dir", str(out / "selectors"),
                "--pca", str(PCA_SMOKE), "--allow-any-gpu", "--n", "8", "--n-big", "8", "--n1", "4", "--batches", "1", "4"], host=host, bank=bank)
    mk = json.loads((tb / TMG.BUNDLE_MARK).read_text())
    rep["timing_bundle_smoke"] = {"seconds": time.time() - t0, "parts": [{k: p.get(k) for k in ("part", "status", "seconds")} for p in mk["parts"]]}
    (out / f"smoke_throughput_{tag}.json").write_text(V.jdump(rep, indent=1)); print(V.jdump(rep["timing_bundle_smoke"], indent=1))
    return 0


def cmd_anchor(a):
    """Decision 15 (the batch-1 official full-row anchor) on the three smoke subsets (sets64): the production function run_official into pred/
    (+ marker), its answers against the frozen R7-40a reference r7_d1_40_host.QwenPrefill.official_answer on every item (same call: expected
    identical), against the batched full row computed in this job on the same GPU (the batching effect), a resume that redoes nothing, and
    the seconds per item.  Code check and cost only."""
    import r7_d1_40_host as QH
    import r7_d1_40c_run as RN
    out = Path(a.out); host = QH.QwenPrefill(out); torch = host.torch; rep = {"gpu": torch.cuda.get_device_name(0), "load_seconds": host.load_seconds, "benches": {}}
    load = lambda b: [{k: v for k, v in it.items() if k not in RN.ANSWER_KEYS} for it in RUN.load_items(out / "sets64", b)]  # noqa: E731
    for b in ("gqa_fixture", "textvqa_fixture", "chartqa_train_smoke"):
        its = load(b); t0 = time.time()
        RN.run_official(Namespace(sets_dir=str(out / "sets64"), bench=b, shard="0/1", out=str(out / "pred"), limit=0), host=host)
        sec = time.time() - t0; od = out / "pred" / b; rows = {r["qid"]: r for r in V.read_jsonl(od / f"{RUN.cfg_file(RN.OFFICIAL)}@s0of1.jsonl")}
        n_lines = len((od / f"{RUN.cfg_file(RN.OFFICIAL)}@s0of1.jsonl").read_text().splitlines())
        RN.run_official(Namespace(sets_dir=str(out / "sets64"), bench=b, shard="0/1", out=str(out / "pred"), limit=0), host=host)      # resume
        ref = [host.official_answer(it["prompt"], RN.open_image(it["image_file"]), max_new_tokens=PL.MAX_NEW[b])["text"] for it in its]
        mine, _ = batched_answers(host, states_for(host, its, RN.open_image), "full", None, PL.MAX_NEW[b])
        rep["benches"][b] = {"items": len(its), "marker": (od / "done_official_shard0of1.json").exists(), "seconds_per_item": sec / max(len(its), 1),
                             "anchor_equals_frozen_official_answer": float(np.mean([rows[it["qid"]]["pred"] == r for it, r in zip(its, ref)])),
                             "anchor_equals_batched_full_same_gpu": float(np.mean([rows[it["qid"]]["pred"] == m for it, m in zip(its, mine)])),
                             "resume_lines_unchanged": len((od / f"{RUN.cfg_file(RN.OFFICIAL)}@s0of1.jsonl").read_text().splitlines()) == n_lines == len(its),
                             "n_vis_mean": float(np.mean([r["n_vis"] for r in rows.values()])), "hit_max": int(sum(r["hit_max"] for r in rows.values()))}
        print(V.jdump({b: rep["benches"][b]}), flush=True)
    (out / "smoke_anchor.json").write_text(V.jdump(rep, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ report (CPU)
def cmd_report(a):
    import r7_d1_40c_score as SCO
    out = Path(a.out); sm = json.loads((out / "smoke_gpu.json").read_text()); rep = {"gpu_smoke": sm}
    lr = out / "selectors" / "lr_choice.json"; proto = out / "smoke_protocol.yaml"
    for basis, bstar, benches in (("latency", str(SMOKE_B_STAR), ["gqa_fixture", "textvqa_fixture"]), ("fraction", "none", list(BENCH_ROWS))):
        res_dir = out / f"results_{basis}"
        if res_dir.exists():
            shutil.rmtree(res_dir)
        cfgdir = out / "configs" / "score_view"; cfgdir.mkdir(parents=True, exist_ok=True)
        for b in benches:
            shutil.copyfile(out / "configs" / BENCH_ROWS[b] / f"configs_{b}.txt", cfgdir / f"configs_{b}.txt")
        have_anchor = all(any((out / "pred" / b).glob("full_official@*.jsonl")) for b in benches)
        SCO.main(["--sets-dir", str(out / "sets64"), "--pred-dir", str(out / "pred"), "--timing-dir", str(out / "timing"), "--configs-dir", str(cfgdir),
                  "--lr-choice", str(lr), "--out", str(res_dir), "--benches", *benches, "--reps", "200", "--protocol", str(proto), "--b-star", bstar,
                  "--anchor", "full_official" if have_anchor else "none"])
        r = json.loads((res_dir / "r7_40c_results.json").read_text())
        rep[f"scorer_{basis}"] = {b: {"n_items": R["n_items"], "families": R["families"], "official_anchor_reported": "official_anchor" in R,
                                      "primary": {k: ("not reached" if v.get("not_reached") else {"has_ci": "ci95" in v, "holm_family_size": v.get("holm_family_size")})
                                                  for k, v in R["contrasts"].items() if v.get("role") == "primary"},
                                      "integrity": R["integrity"]["totals"]} for b, R in r["benches"].items()}
    refused = {}
    try:                                                                              # the scorer must refuse without the freeze key
        SCO.main(["--sets-dir", str(out / "sets64"), "--pred-dir", str(out / "pred"), "--configs-dir", str(out / "configs" / "score_view"), "--lr-choice", str(lr),
                  "--out", str(out / "refusal"), "--protocol", str(out / "prep.json"), "--b-star", "none"])
        refused["no_freeze_key"] = False
    except SystemExit as e:
        refused["no_freeze_key"] = "not frozen" in str(e)
    rep["scorer_refusals"] = refused
    (out / "smoke_summary.json").write_text(V.jdump(rep, indent=1)); print(V.jdump({k: v for k, v in rep.items() if k != "gpu_smoke"}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prep", "gpu", "throughput", "anchor", "report"]); ap.add_argument("--out", default=str(SM))
    ap.add_argument("--n", type=int, default=64); a = ap.parse_args()
    return {"prep": cmd_prep, "gpu": cmd_gpu, "throughput": cmd_throughput, "anchor": cmd_anchor, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
