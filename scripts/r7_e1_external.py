"""r7 E1 — external Qwen2.5-VL-7B-Instruct token-pruning baselines (MMTok, HiPrune) as a separate host column next to R7-40c, on validation data
only (coordinator decision reports/R7/R7_EXTENSIONS_DECISION_20261013.md, E1; protocol draft reports/R7/R7_E1_EXTERNAL_PROTOCOL_DRAFT_20261013.yaml).
I/O glue only: each method runs as its official code ships it, in its own pinned environment; this file supplies R7-40c's items, prompts (Qwen chat
template), images, pixel bounds, greedy generation call, sharding, prediction-file layout and markers.  No method logic is re-implemented or changed.
  mmtok    MMTok (ICLR 2026; github.com/Ironieser/MMTok @8e26868, Apache-2.0): question-conditioned coverage maximisation over the merged visual
           tokens (after the merger, before the LLM).  Applied with the official wrapper mmtok.qwen.mmtok_qwen2_5_vl(model, language_tokenizer,
           processor, retain_ratio=f) on transformers' Qwen2_5_VLForConditionalGeneration (4.52.4, SDPA as its Qwen example, bf16); the wrapper's
           patched processor.apply_chat_template hands the user text (the item's prompt) to the selector.  k = int(f·N)
           (mmtok/qwen/qwen2_5_VLmodel_mmtok.py: int(token_retain_ratio * image_embeds.shape[0])).
  hiprune  HiPrune (ACL 2026 Findings; github.com/Danielement321/HiPrune @8278100, MIT): question-agnostic, vision-encoder attention (anchor tokens
           of object layer 16 with their 4-neighbour buffers, register tokens of the last layer).  Applied by loading the repo's own modeling copy
           Qwen2_5_VL/qwen2_5_vl_HiPrune.py (transformers 4.52.0; flash_attention_2 as eval_scripts/bench_qwen.sh; bf16) with HIPRUNE_QWEN_RETENTION=f,
           HIPRUNE_ALPHA=0.1, HIPRUNE_OBJECT_LAYER=16 (README defaults, read at import).  k = round(f·N) (Python round: half to even).
Budget: f in {0.125, 0.25, 0.5, 0.75} of the image's N merged tokens through the method's own knob; optional in-env anchor f = 1 (the method's code
path, no token removed).  R7-40c's own rule is max(1, floor(f·N + 1/2)) (r7_d1_40_masks.budget): MMTok keeps one token fewer whenever frac(f·N) >= 1/2,
HiPrune one fewer when f·N = m + 1/2 with m even.
Per item, batch 1: the frozen R7-40c official call (r7_d1_40c_run.official_generate's arguments) — the Qwen chat template with the item's pinned prompt
after one image (shipped system prompt), AutoProcessor(use_fast=True: R7-40c's Qwen2VLImageProcessorFast; 4.52.x would default to the slow one)
with min / max pixels 256·28² / 1280·28² (checked on synthetic images at load), then model.generate(do_sample=False, num_beams=1,
max_new_tokens=MAX_NEW[bench], use_cache=True, output_logits=True, return_dict_in_generate=True) with the shipped generation config (repetition
penalty 1.05, eos [151645, 151643]).  The LLM's prefill length is observed by a forward pre-hook that changes
nothing: kept = prefill length − (input length − N); recorded next to the method's own rule k_rule.
Output (R7-40c layout, readable by r7_d1_30_score.load_config / r7_d1_40c_score.Bench40): <out>/<bench>/ext__<method>__<f>@s<i>of<n>.jsonl rows
{qid, image_id, pred, kept, n_vis, f, n_new, hit_max, k_rule, ms}; one file per (configuration, shard), single writer, one line written and flushed per
item; resumed per (configuration, qid) over every file of the configuration; per-session provenance in <out>/<bench>/_e1_meta/ (not matched by the
loaders' '<cfg>@*.jsonl' glob); marker done_e1_<method>_shard<i>of<n>.json only when every (configuration, item) of the shard is answered.
One GPU job per (method, benchmark, shard): the checkpoint is staged once to node-local disk; each fraction then runs in a fresh interpreter (HiPrune
reads its knob at import; MMTok's wrapper is applied once per process, as in its example), one after the other on the same GPU; cwd = this repo
(never the fetched tree, which is appended to sys.path read-only).
    python -I scripts/r7_e1_external.py run --method M --bench B --shard i/n [--anchor] [--sets-dir S] [--out O] [--limit N]
    python -I scripts/r7_e1_external.py selfcheck --method M --out O          (CPU: tiny random Qwen2.5-VL through the production path; HiPrune via
                                                                               'probe', since its shipped LLM path needs flash_attention_2 = GPU)
    python -I scripts/r7_e1_external.py smokecheck --method M --out O --benches gqa_dev@0/6 ... [--fractions F ...] [--limit N]   (after a smoke)
    python -I scripts/r7_e1_external.py plan --lines F [--root R] [--anchor]    (configuration lists + unit lines; nothing submitted)
    python -I scripts/r7_e1_external.py complete --pred-dir P --out O [--anchor] (stage 2: coverage + markers; prediction files read for qids only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402  (frozen D1-30: prediction-file layout, resume, D1-30 sharding)
import r7_d1_40_host as QH  # noqa: E402  (frozen R7-40a: pixel bounds and checkpoint directory only)
import r7_d1_40_masks as M4  # noqa: E402  (frozen R7-40a: R7-40c's budget rule, for the recorded comparison only)
import r7_d1_40c_plan as PL  # noqa: E402  (frozen R7-40c: benchmarks, generation lengths, sharding, fraction keys, node exclusion)
import r7_d1_40c_run as RN  # noqa: E402  (frozen R7-40c: items without answers, image opening, smart_resize, new-token statistics)
import r7_vl_common as V  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
THIRD = Path("/projects/EEG-foundation-model/yinghao/third_party")
ENVS = Path("/projects/EEG-foundation-model/yinghao/envs")
E1 = PL.R40 / "eval_e1"
SMOKE_SETS = PL.R40 / "smoke_40c" / "sets64"             # R7-40c smoke fixtures (R7-20 fixtures, ChartQA TRAIN subset): never val / test
FRACTIONS = tuple(PL.FRACTIONS)                          # (0.125, 0.25, 0.5, 0.75)
ANCHOR = 1.0                                             # optional in-env anchor: the method's code path with every token kept
MAX_NEW = {**PL.MAX_NEW, "e1_selfcheck": 4}              # R7-40c / D30-7 generation lengths (+ the CPU self-check's tiny set)
PARTITION = "L40S"                                       # R7-40c decision 7 (quality units on L40S only)
PROCESSOR_USE_FAST = True    # R7-40c's image processor class (transformers 4.57.6 resolves Qwen2VLImageProcessorFast); transformers 4.52.x, and so the
                             # methods' own harnesses, default to the slow PIL processor (same grids, slightly different resampling): open decision
CODE_FILES = ("scripts/r7_e1_external.py", "tests/test_r7_e1_external.py", "slurm/r7_e1_cpu.sbatch", "slurm/r7_e1_gpu.sbatch")
# ≈ 3 × the estimated unit time with the anchor (5 configurations): R7-40c's L40S batch-1 official rates (0.136 / 0.350 / 0.237 s per item) × a method
# factor (MMTok 1.1, HiPrune 1.4-1.6: attention weights of all 32 vision layers) × the largest shard (1,334 / 1,212 / 963 items) + ≈ 8 min of staging
# and 5 loads.  ESTIMATES: replace with the smoke's measured rates before the freeze.
TIME = {("mmtok", "gqa_dev"): "01:15:00", ("mmtok", "textvqa_val"): "02:30:00", ("mmtok", "chartqa_val"): "01:30:00",
        ("hiprune", "gqa_dev"): "01:30:00", ("hiprune", "textvqa_val"): "03:15:00", ("hiprune", "chartqa_val"): "01:45:00"}
METHODS = {
    "mmtok": {
        "dir": THIRD / "MMTok@8e26868", "commit": "8e2686836ed5bbe05d064f1d1c20e50c73c70b94", "licence": "Apache-2.0", "env": ENVS / "mmtok",
        "attn": "sdpa", "transformers": "4.52.4",
        "knob": "retain_ratio argument of mmtok.qwen.mmtok_qwen2_5_vl (stored as MMTokCore.retain_ratio, read first by the patched forward); "
                "its fallback TOKEN_RETAIN_RATIO is set to the same value; SELECTION_METHOD=mmtok",
        "k_rule": "min(int(f*N), N)",
        "files": ("LICENSE", "README.md", "install.md", "requirements.txt", "pyproject.toml", "mmtok/__init__.py", "mmtok/core/__init__.py",
                  "mmtok/core/mmtok_core.py", "mmtok/core/semantic_selector.py", "mmtok/core/text_processor.py", "mmtok/qwen/__init__.py",
                  "mmtok/qwen/qwen2_5_vl_mmtok.py", "mmtok/qwen/qwen2_5_VLmodel_mmtok.py", "mmtok/qwen/modeling_qwen2_5_vl_mmtok.py",
                  "mmtok/llava/__init__.py", "mmtok/llava/llava_inject.py", "mmtok/llava/patch_llava.py", "mmtok/llava/clip_encoder_mmtok.py",
                  "mmtok/llava/llava_arch_mmtok.py", "example/qwen_mmtok_example.py", "example/lmms_eval_qwen_mmtok.py")},
    "hiprune": {
        "dir": THIRD / "HiPrune@8278100", "commit": "82781005a7e72a6be9ede58fd77473efa72b5e4f", "licence": "MIT", "env": ENVS / "hiprune",
        "attn": "flash_attention_2", "transformers": "4.52.0",
        "knob": "HIPRUNE_QWEN_RETENTION (environment, read at import of Qwen2_5_VL/qwen2_5_vl_HiPrune.py) with HIPRUNE_ALPHA=0.1, HIPRUNE_OBJECT_LAYER=16",
        "k_rule": "round(N*f) (Python round, half to even)", "alpha": "0.1", "object_layer": "16",
        "files": ("LICENSE", "README.md", "setup_qwen.sh", "eval_scripts/bench_qwen.sh", "Qwen2_5_VL/qwen2_5_vl_HiPrune.py",
                  "lmms-eval/lmms_eval/models/qwen2_5_vl.py", "lmms-eval/pyproject.toml")},
}
KNOB_VARS = ("HIPRUNE_QWEN_RETENTION", "HIPRUNE_ALPHA", "HIPRUNE_OBJECT_LAYER", "TOKEN_RETAIN_RATIO", "SELECTION_METHOD",
             "MMTok_JIT_N_THRESHOLD", "MMTok_JIT_K_MAX_THRESHOLD")


# ------------------------------------------------------------------------------------------------------------------------------ grammar and budgets
def fractions(anchor: bool = False):
    return list(FRACTIONS) + ([ANCHOR] if anchor else [])


def run_fractions(a):
    """The fractions of a run: --fractions (a subset, smoke) or all four; + the anchor with --anchor."""
    fr = [float(x) for x in (getattr(a, "fractions", None) or FRACTIONS)]
    if getattr(a, "anchor", False) and ANCHOR not in fr:
        fr.append(ANCHOR)
    for f in fr:
        cfg_name(a.method if a.method in METHODS else "mmtok", f)                   # validates the fraction
    return fr


def cfg_name(method: str, f) -> str:
    """'ext:<method>:<f>' (f as R7-40c's fraction keys; the anchor is 'ext:<method>:1'); file stem 'ext__<method>__<f>' (RUN.cfg_file)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    k = PL.fkey(f)
    if k not in PL.FKEYS and float(f) != ANCHOR:
        raise ValueError(f"fraction {f} not in {PL.FKEYS} or the anchor 1")
    return f"ext:{method}:{k}"


def parse_cfg(cfg: str):
    kind, method, f = cfg.split(":")
    if kind != "ext":
        raise ValueError(f"not an E1 configuration: {cfg!r}")
    if cfg_name(method, float(f)) != cfg:
        raise ValueError(f"non-canonical E1 configuration {cfg!r}")
    return method, float(f)


def config_list(methods=tuple(METHODS), anchor: bool = False):
    return [cfg_name(m, f) for m in methods for f in fractions(anchor)]


def k_rule(method: str, f, n) -> int:
    """Visual tokens the method's own code keeps for an image with n merged tokens at knob value f.
    MMTok: target = int(ratio * N) (qwen2_5_VLmodel_mmtok.py); select_vision_tokens keeps every token when N <= target; k_max = min(target, N).
    HiPrune: visual_token_num = round(n_image_tokens * RETAIN) (qwen2_5_vl_HiPrune.py); anchors + buffers + registers = visual_token_num."""
    f, n = float(f), int(n)
    if method == "mmtok":
        return min(int(f * n), n)
    if method == "hiprune":
        return round(n * f)
    raise ValueError(f"unknown method {method!r}")


def r7_40c_k(f, n) -> int:
    """R7-40c's budget (r7_d1_40_masks.budget, frozen) for comparison; f = 1 is the full row."""
    return int(n) if float(f) >= 1.0 else M4.budget(float(f), int(n))


def marker_name(method: str, shard: str) -> str:
    return f"done_e1_{method}_shard{shard.replace('/', 'of')}.json"


def meta_path(out: Path, cfg: str, shard: str) -> Path:
    return Path(out) / "_e1_meta" / f"{RUN.cfg_file(cfg)}@{RUN.shard_tag(shard)}.json"


def child_env(method: str, f, base=None) -> dict:
    """Environment of one fraction's interpreter: inherited knob variables removed, the method's own knobs set (HiPrune reads them at import)."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in KNOB_VARS}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    if method == "hiprune":
        env.update(HIPRUNE_QWEN_RETENTION=repr(float(f)), HIPRUNE_ALPHA=METHODS["hiprune"]["alpha"], HIPRUNE_OBJECT_LAYER=METHODS["hiprune"]["object_layer"])
    elif method == "mmtok":
        env.update(TOKEN_RETAIN_RATIO=repr(float(f)), SELECTION_METHOD="mmtok")
    else:
        raise ValueError(method)
    return env


# ------------------------------------------------------------------------------------------------------------------------------ provenance
def sha_bytes(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        if Path(p).exists():
            h.update(Path(p).read_bytes())
    return h.hexdigest()


def code_sha() -> dict:
    return {"files": list(CODE_FILES), "sha256": sha_bytes([REPO / p for p in CODE_FILES]),
            "script_sha256": RN.sha_file(Path(__file__).resolve())}


def method_files_sha(method: str) -> dict:
    d = METHODS[method]["dir"]
    return {p: (RN.sha_file(d / p) if (d / p).exists() else None) for p in METHODS[method]["files"]}


def repo_state() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "scripts", "slurm", "tests"], capture_output=True, text=True,
                               timeout=120).stdout.splitlines()
        return {"commit": head or None, "dirty_files": len(dirty)}
    except Exception as e:                                                             # provenance only; never fails a unit
        return {"commit": None, "error": f"{type(e).__name__}: {e}"}


def shard_items(sets_dir, bench, shard="0/1", limit=0):
    """The shard's items in the D1-30 shard order, answers stripped (r7_d1_40c_run.load_items)."""
    items = RN.load_items(sets_dir, bench, shard)
    return items[:limit] if limit else items


def ensure_newline(path: Path):
    """A crash can leave a torn last line: terminate it before appending (the loaders drop and count it; the next line stays intact)."""
    p = Path(path)
    if p.exists() and p.stat().st_size:
        with open(p, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            with open(p, "ab") as fh:
                fh.write(b"\n")


def first_records(out: Path, cfg: str):
    """{qid: first complete record} over every file of a configuration (the R7-40c loaders' rule)."""
    recs = {}
    for p in RUN.cfg_paths(out, cfg):
        for r in V.read_jsonl(p):
            if isinstance(r, dict) and "qid" in r and "pred" in r and r["qid"] not in recs:
                recs[r["qid"]] = r
    return recs


def kept_check(out: Path, cfg: str, qids) -> dict:
    """Observed kept tokens vs the method's own rule (and R7-40c's rule, for the record) over the shard's first records."""
    method, f = parse_cfg(cfg); recs = [r for q, r in first_records(out, cfg).items() if q in qids]
    ok = [r.get("kept") == k_rule(method, f, r.get("n_vis", -1)) for r in recs]
    bad = [{"qid": r["qid"], "kept": r.get("kept"), "k_rule": k_rule(method, f, r.get("n_vis", -1)), "n_vis": r.get("n_vis")} for r, o in zip(recs, ok) if not o]
    return {"rows": len(recs), "kept_equals_method_rule": int(sum(ok)), "mismatch_examples": bad[:5],
            "kept_equals_r7_40c_rule": int(sum(r.get("kept") == r7_40c_k(f, r.get("n_vis", 1)) for r in recs)),
            "mean_kept": float(np.mean([r["kept"] for r in recs])) if recs else None,
            "mean_n_vis": float(np.mean([r["n_vis"] for r in recs])) if recs else None,
            "mean_kept_fraction": float(np.mean([r["kept"] / r["n_vis"] for r in recs])) if recs else None,
            "hit_max": int(sum(bool(r.get("hit_max")) for r in recs))}


# ------------------------------------------------------------------------------------------------------------------------------ runtime (method env)
def add_method_path(method: str):
    """The fetched official tree, appended (never prepended) to sys.path, read-only: imported as its harnesses import it."""
    d = str(METHODS[method]["dir"])
    if d not in sys.path:
        sys.path.append(d)


def check_bounds(proc) -> dict:
    """The frozen R7-20 / R7-40 pixel bounds must be in effect: the processor's own call on four synthetic sizes must give the merged grids of
    smart_resize(min_pixels 256·28², max_pixels 1280·28²) (r7_d1_40c_run.check_processor's probe, without its 4.57.6-specific attribute names)."""
    from PIL import Image
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}],
                                    tokenize=False, add_generation_prompt=True)
    probe = []
    for w, h in RN.PROBE_WH:
        thw = proc(text=[text], images=[Image.new("RGB", (w, h))], return_tensors="pt")["image_grid_thw"][0]
        thw = [int(x) for x in (thw.tolist() if hasattr(thw, "tolist") else thw)]
        probe.append({"wh": [w, h], "grid": [thw[1] // RN.MERGE, thw[2] // RN.MERGE],
                      "expected": [x // RN.FACTOR for x in RN.smart_resize(h, w, min_pixels=QH.MIN_PIXELS, max_pixels=QH.MAX_PIXELS)]})
    ip = proc.image_processor
    attrs = {"class": type(ip).__name__, "min_pixels": getattr(ip, "min_pixels", None), "max_pixels": getattr(ip, "max_pixels", None),
             "size": str(getattr(ip, "size", None)), "patch_size": getattr(ip, "patch_size", None), "merge_size": getattr(ip, "merge_size", None)}
    if any(p["grid"] != p["expected"] for p in probe) or (attrs["patch_size"], attrs["merge_size"]) != (RN.PATCH, RN.MERGE):
        raise SystemExit(f"processor does not apply the frozen bounds {QH.MIN_PIXELS} / {QH.MAX_PIXELS}: {attrs}; synthetic grids {probe}")
    return {"probe": probe, "image_processor": attrs}


def attn_classes(model) -> dict:
    """Attention classes actually built (vision tower block 0 and LLM layer 0) and the configured implementations."""
    out = {"config": getattr(model.config, "_attn_implementation", None),
           "vision_config": getattr(getattr(model.config, "vision_config", None), "_attn_implementation", None)}
    for name, path in (("vision_block0", ("model.visual.blocks", "visual.blocks")), ("llm_layer0", ("model.language_model.layers", "model.layers"))):
        for p in path:
            obj = model
            try:
                for part in p.split("."):
                    obj = getattr(obj, part)
                out[name] = type(obj[0].attn if name == "vision_block0" else obj[0].self_attn).__name__
                break
            except (AttributeError, IndexError, TypeError):
                continue
    return out


class Runtime:
    """One loaded method host: processor, model with the method applied, and a forward pre-hook on the module that receives the (pruned) LLM input
    (it records the prefill length and returns nothing, so the call is unchanged)."""

    def __init__(self, torch, proc, model, llm, device, info):
        self.torch, self.proc, self.model, self.device, self.info = torch, proc, model, device, info
        self.image_token_id = int(model.config.image_token_id)
        eos = model.generation_config.eos_token_id
        self.eos_ids = [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos])]
        self.seen, self.last = [], None
        self._hook = llm.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, module, args, kwargs):
        x = kwargs.get("inputs_embeds")
        if x is not None and x.shape[1] > 1:
            self.seen.append(int(x.shape[1]))

    def sync(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()


def load_runtime(method, f, model_dir, proc_dir=None, attn=None, device="cuda", dtype="bfloat16") -> Runtime:
    """The method's model exactly as its official code builds it (see the module docstring), with the frozen pixel bounds."""
    import torch
    import transformers
    from transformers import AutoProcessor
    spec = METHODS[method]; attn = attn or spec["attn"]; td = getattr(torch, dtype); t0 = time.time()
    if transformers.__version__ != spec["transformers"]:
        raise SystemExit(f"{method} pins transformers {spec['transformers']}, this env has {transformers.__version__}: stop")
    add_method_path(method)
    proc = AutoProcessor.from_pretrained(str(proc_dir or model_dir), min_pixels=QH.MIN_PIXELS, max_pixels=QH.MAX_PIXELS, use_fast=PROCESSOR_USE_FAST)
    bounds = check_bounds(proc)                                                        # before MMTok patches apply_chat_template
    if method == "mmtok":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=td, attn_implementation=attn).to(device).eval()
        from mmtok.qwen import mmtok_qwen2_5_vl
        tok = proc.tokenizer
        if not getattr(tok, "is_fast", False):
            raise SystemExit("MMTok's coverage text encoding needs the fast tokenizer (its lmms-eval adapter enforces it): stop")
        model, proc = mmtok_qwen2_5_vl(model, language_tokenizer=tok, processor=proc, retain_ratio=float(f))     # the official wrapper, once
        core = model.model._mmtok_core
        if float(core.retain_ratio) != float(f):
            raise SystemExit(f"MMTok retain_ratio {core.retain_ratio} != {f}: stop")
        knobs = {"retain_ratio": core.retain_ratio, "TOKEN_RETAIN_RATIO": os.environ.get("TOKEN_RETAIN_RATIO"),
                 "SELECTION_METHOD": os.environ.get("SELECTION_METHOD"), "alpha": core.alpha, "softmax_tv_temperature": core.softmax_tv_temperature,
                 "softmax_vv_temperature": core.softmax_vv_temperature, "remove_padding_indices": core.remove_padding_indices,
                 "language_tokenizer": type(tok).__name__}
        llm = model.model.language_model
    elif method == "hiprune":
        import Qwen2_5_VL.qwen2_5_vl_HiPrune as HP                                     # reads the three knobs at import (child_env)
        want = (float(f), float(spec["alpha"]), int(spec["object_layer"]))
        if (HP.RETAIN, HP.ALPHA, HP.OBJECT_LAYER) != want:
            raise SystemExit(f"HiPrune knobs read at import {(HP.RETAIN, HP.ALPHA, HP.OBJECT_LAYER)} != {want}: stop")
        model = HP.Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=td, attn_implementation=attn).to(device).eval()
        knobs = {"HIPRUNE_QWEN_RETENTION": HP.RETAIN, "HIPRUNE_ALPHA": HP.ALPHA, "HIPRUNE_OBJECT_LAYER": HP.OBJECT_LAYER, "module": HP.__file__}
        llm = model.model
    else:
        raise SystemExit(f"unknown method {method!r}")
    info = {"method": method, "fraction": float(f), "attn": attn, "attn_built": attn_classes(model), "dtype": str(model.dtype), "device": str(device),
            "transformers": transformers.__version__, "torch": torch.__version__, "knobs": knobs, "bounds": bounds,
            "generation": {"eos": model.generation_config.eos_token_id, "pad": model.generation_config.pad_token_id,
                           "repetition_penalty": model.generation_config.repetition_penalty},
            "load_seconds": time.time() - t0}
    return Runtime(torch, proc, model, llm, device, info)


def answer_item(rt: Runtime, method: str, f, it, image, max_new: int, expected_n=None) -> dict:
    """One item at batch 1 through the frozen R7-40c official call, the method applied inside the model's forward."""
    torch = rt.torch
    text = rt.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["prompt"]}]}],
                                       tokenize=False, add_generation_prompt=True)          # MMTok's patched hook reads the user text here
    inputs = rt.proc(text=[text], images=[image], return_tensors="pt").to(rt.device)
    ids = inputs["input_ids"]; L = int(ids.shape[1]); n_vis = int((ids == rt.image_token_id).sum())
    if expected_n is not None and n_vis != int(expected_n):
        raise SystemExit(f"{it['qid']}: the processor gives {n_vis} visual tokens, the image header {expected_n} under the frozen bounds: stop")
    rt.last = {"qid": it["qid"], "L": L, "n_vis": n_vis}; rt.seen.clear(); rt.sync(); t0 = time.perf_counter()
    with torch.inference_mode():
        out = rt.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new, use_cache=True, output_logits=True,
                                return_dict_in_generate=True)
    rt.sync(); ms = 1e3 * (time.perf_counter() - t0)
    if not rt.seen:
        raise SystemExit(f"{it['qid']}: the LLM prefill was not observed (hook): stop")
    new = out.sequences[:, L:]; toks = [int(x) for x in new[0].tolist()]
    nn, hit = RN.new_token_stats(toks, set(rt.eos_ids), max_new)
    return {"qid": it["qid"], "image_id": it["image_id"], "pred": rt.proc.batch_decode(new, skip_special_tokens=True)[0].strip(),
            "kept": int(rt.seen[0] - (L - n_vis)), "n_vis": n_vis, "f": float(f), "n_new": nn, "hit_max": hit, "k_rule": k_rule(method, f, n_vis),
            "ms": round(ms, 2)}


# ------------------------------------------------------------------------------------------------------------------------------ worker (one fraction)
def identity(info: dict) -> dict:
    """Fields that must not change between sessions of one (configuration, shard)."""
    return {k: info.get(k) for k in ("method", "fraction", "attn", "dtype", "transformers", "torch", "processor", "knobs_id", "method_commit", "method_files")}


def session_meta(rt_info: dict, method: str, f) -> dict:
    kn = {k: v for k, v in (rt_info.get("knobs") or {}).items() if k not in ("module", "language_tokenizer")}
    return {**rt_info, "knobs_id": json.dumps(kn, sort_keys=True, default=str), "method_commit": METHODS[method]["commit"],
            "method_files": method_files_sha(method), "processor": ((rt_info.get("bounds") or {}).get("image_processor") or {}).get("class")}


def check_meta(path: Path, meta: dict) -> dict:
    """Every session of a (configuration, shard) file runs the same method build: an existing record with another identity stops the session."""
    rec = json.loads(path.read_text()) if path.exists() else {"identity": identity(meta), "sessions": []}
    if rec["identity"] != identity(meta):
        diff = {k: (rec["identity"].get(k), v) for k, v in identity(meta).items() if rec["identity"].get(k) != v}
        raise SystemExit(f"{path.name}: this session differs from the earlier ones {diff}: stop (one build per configuration file)")
    return rec


def write_meta(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp")
    tmp.write_text(V.jdump(rec, indent=1)); os.replace(tmp, path)


def worker(a, rt=None, opener=None, grid_of=None) -> int:
    """One (method, fraction) over the shard's items: rows appended to the configuration's own shard file, one line per item, flushed."""
    method, f = a.method, float(a.fraction); cfg = cfg_name(method, f)
    if a.bench not in MAX_NEW:
        raise SystemExit(f"no generation length for benchmark {a.bench}")
    items = shard_items(a.sets_dir, a.bench, a.shard, a.limit); max_new = MAX_NEW[a.bench]
    out = Path(a.out) / a.bench; out.mkdir(parents=True, exist_ok=True)
    done = RUN.answered(out, cfg); todo = [it for it in items if it["qid"] not in done]
    if not todo:
        print(f"[r7_e1] {cfg} {a.bench} {a.shard}: complete ({len(items)} items)", flush=True); return 0
    opener = opener or RN.open_image; grid_of = grid_of or RN.merged_grid
    if rt is None:
        rt = load_runtime(method, f, a.model_dir, a.proc_dir, a.attn, a.device, a.dtype)
    meta = session_meta(rt.info, method, f); mp = meta_path(out, cfg, a.shard); rec = check_meta(mp, meta)
    torch = rt.torch
    if str(rt.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    path = out / f"{RUN.cfg_file(cfg)}@{RUN.shard_tag(a.shard)}.jsonl"; ensure_newline(path); t0 = time.time(); n = 0
    try:
        with open(path, "a") as fh:                                                    # this (configuration, shard)'s own file: single writer
            for it in todo:
                rows, cols = grid_of(it["image_file"])
                r = answer_item(rt, method, f, it, opener(it["image_file"]), max_new, expected_n=rows * cols)
                fh.write(V.jdump(r) + "\n"); fh.flush(); n += 1
                if a.limit and a.limit <= 64:                                          # smoke-sized runs: one line per item
                    print(f"[r7_e1] {cfg} {a.bench} {r['qid']}: N {r['n_vis']} kept {r['kept']} (rule {r['k_rule']}, R7-40c {r7_40c_k(f, r['n_vis'])}) "
                          f"n_new {r['n_new']} hit_max {r['hit_max']} {r['ms']:.0f} ms pred {r['pred'][:40]!r}", flush=True)
                if n % 100 == 0:
                    print(f"[r7_e1] {cfg} {a.bench} {a.shard}: {n}/{len(todo)} items, {time.time() - t0:.0f}s", flush=True)
    finally:
        rec["sessions"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": os.environ.get("SLURM_JOB_ID"),
                                "node": socket.gethostname(), "answers": n, "todo": len(todo), "seconds": time.time() - t0,
                                "seconds_per_answer": (time.time() - t0) / n if n else None, "load_seconds": rt.info.get("load_seconds"),
                                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if str(rt.device).startswith("cuda") else None,
                                "gpu": torch.cuda.get_device_name(0) if str(rt.device).startswith("cuda") else None,
                                "attn_built": rt.info.get("attn_built"), "bounds": rt.info.get("bounds"), "knobs": rt.info.get("knobs"),
                                "generation": rt.info.get("generation")})
        write_meta(mp, rec)
    print(f"[r7_e1] {cfg} {a.bench} {a.shard}: {n} answers in {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ unit (one shard)
def spawn_worker(a, f, model_dir) -> int:
    """One fraction in a fresh interpreter of the same env (python -I; cwd = this repo; the method's knobs in its environment)."""
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "worker", "--method", a.method, "--fraction", PL.fkey(f), "--bench", a.bench,
           "--shard", a.shard, "--sets-dir", str(a.sets_dir), "--out", str(a.out), "--model-dir", str(model_dir),
           "--proc-dir", str(a.proc_dir or model_dir), "--limit", str(a.limit or 0), "--attn", a.attn or METHODS[a.method]["attn"],
           "--device", a.device, "--dtype", a.dtype]
    print(f"[r7_e1] start {cfg_name(a.method, f)}: {' '.join(cmd[3:])}", flush=True)
    return subprocess.run(cmd, env=child_env(a.method, f), cwd=str(REPO)).returncode


def run(a, spawn=None, stage=None, grid_of=None) -> int:
    """Every fraction (and the anchor when asked) of one (method, benchmark, shard), then the shard marker."""
    if a.method not in METHODS:
        raise SystemExit(f"unknown method {a.method!r}")
    if a.bench not in MAX_NEW:
        raise SystemExit(f"no generation length for benchmark {a.bench}")
    fr = run_fractions(a); cfgs = [cfg_name(a.method, f) for f in fr]
    items = shard_items(a.sets_dir, a.bench, a.shard, a.limit); qids = {it["qid"] for it in items}
    out = Path(a.out) / a.bench; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    before = {c: len(RUN.answered(out, c) & qids) for c in cfgs}
    pending = [f for f, c in zip(fr, cfgs) if before[c] < len(qids)]
    note = "nothing to run"
    if pending:
        grid_of = grid_of or RN.merged_grid
        for it in items:                                                               # image headers only: a missing image stops before staging
            grid_of(it["image_file"])
        if a.model_dir:
            qdir, tmp, note = Path(a.model_dir), None, f"given {a.model_dir}"
        else:
            qdir, tmp, note = (stage or V.stage)(QH.QWEN_DIR)
        print(f"[r7_e1] {a.method} {a.bench} {a.shard}: {len(items)} items, pending {[cfg_name(a.method, f) for f in pending]}; checkpoint: {note}", flush=True)
        try:
            for f in pending:
                rc = (spawn or spawn_worker)(a, f, qdir)
                if rc != 0:
                    raise SystemExit(f"worker {cfg_name(a.method, f)} ({a.bench} {a.shard}) exited with {rc}: stop (no marker)")
        finally:
            V.unstage(tmp)
    after = {c: len(RUN.answered(out, c) & qids) for c in cfgs}
    mark_done(out, a, items, cfgs, seconds=time.time() - t0, answers=sum(after[c] - before[c] for c in cfgs), stage_note=note)
    return 0


def mark_done(out: Path, a, items, cfgs, seconds=None, answers=None, stage_note=None):
    """Shard marker (the driver's file gate), written only when every (configuration, item) of the shard is answered."""
    qids = {it["qid"] for it in items}
    for c in cfgs:
        have = RUN.answered(out, c)
        if not qids <= have:
            raise SystemExit(f"incomplete after run: {c} ({len(qids - have)} of {len(qids)} items unanswered)")
    sets_file = Path(a.sets_dir) / f"{a.bench}.jsonl"; spec = METHODS[a.method]
    metas = {c: (json.loads(meta_path(out, c, a.shard).read_text()) if meta_path(out, c, a.shard).exists() else None) for c in cfgs}
    gpus = sorted({s.get("gpu") for m in metas.values() if m for s in m["sessions"] if s.get("gpu")})
    (out / marker_name(a.method, a.shard)).write_text(V.jdump({
        "method": a.method, "bench": a.bench, "shard": a.shard, "configs": cfgs, "items": len(items), "seconds": seconds, "answers_this_job": answers,
        "seconds_per_answer": (seconds / answers) if seconds and answers else None, "gpus": gpus, "node": socket.gethostname(),
        "job": os.environ.get("SLURM_JOB_ID"), "env": V.env_info(), "env_dir": str(spec["env"]), "python": sys.executable, "repo": repo_state(),
        "code": code_sha(), "method_commit": spec["commit"], "method_dir": str(spec["dir"]), "method_licence": spec["licence"],
        "method_files_sha256": method_files_sha(a.method), "knob": spec["knob"], "k_rule": spec["k_rule"],
        "sets_file_sha256": RN.sha_file(sets_file) if sets_file.exists() else None, "limit": a.limit or 0, "checkpoint": stage_note,
        "kept_check": {c: kept_check(out, c, qids) for c in cfgs}, "sessions": metas,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))


# ------------------------------------------------------------------------------------------------------------------------------ plan / stage 2
def unit_lines(root: Path = E1, anchor=False, partition=PARTITION, methods=tuple(METHODS)):
    """(sbatch line, marker, gates): one GPU unit per (method, benchmark, shard) with R7-40c's sharding (GQA 6, TextVQA 4, ChartQA 2)."""
    L = []
    for m in methods:
        for b in PL.BENCHES:
            n = PL.QUALITY_SHARDS[b]
            for i in range(n):
                L.append((f"sbatch --parsable --job-name=r7e1_{m}_{b}_{i} {PL.gpu(partition)} --time={TIME[(m, b)]} slurm/r7_e1_gpu.sbatch run {m} {b} {i}/{n}"
                          + (" anchor" if anchor else ""), Path(root) / "pred" / b / marker_name(m, f"{i}/{n}"), []))
    return L


def plan(a) -> int:
    root = Path(a.root); cdir = root / "configs"; cdir.mkdir(parents=True, exist_ok=True); cfgs = config_list(anchor=a.anchor)
    for b in PL.BENCHES:
        (cdir / f"configs_e1_{b}.txt").write_text("\n".join(cfgs) + "\n")
    PL.write_lines(unit_lines(root, a.anchor, a.partition), Path(a.lines),
                   "r7 E1 units (external Qwen2.5-VL baselines; one job per method x benchmark x shard; every fraction of a shard in one job)")
    print(json.dumps({"configs": cfgs, "units": len(unit_lines(root, a.anchor, a.partition)), "lines": a.lines}))
    return 0


def complete(a) -> int:
    """Stage 2: every configuration covers exactly each benchmark's items and every shard marker exists -> <out>/completeness_e1.json."""
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "benches": {}, "problems": []}
    for b in a.benches:
        qids = {it["qid"] for it in RN.load_items(a.sets_dir, b)}; out = Path(a.pred_dir) / b; short = {}
        for c in config_list(tuple(a.methods), a.anchor):
            have = RUN.answered(out, c)
            if have != qids:
                short[c] = {"missing": len(qids - have), "unexpected": len(have - qids)}
        n = PL.QUALITY_SHARDS.get(b, 1); miss = [m for m in (marker_name(x, f"{i}/{n}") for x in a.methods for i in range(n)) if not (out / m).exists()]
        res["benches"][b] = {"items": len(qids), "configs_incomplete": len(short), "examples": dict(list(short.items())[:5]), "markers_missing": miss}
        if short or miss:
            res["problems"].append(f"{b}: {len(short)} incomplete configurations, {len(miss)} missing markers")
    res["complete"] = not res["problems"]
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "completeness_e1.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0 if res["complete"] else 7


def smokecheck(a) -> int:
    """After a smoke run: each (benchmark, configuration) read back with R7-40c's loader r7_d1_30_score.load_config (integrity, coverage of the
    smoke items, row fields, kept == the method's rule, f) and the shard marker's provenance (GPU, attention classes built, processor class, peak
    memory, seconds per answer).  No answer key is read and nothing is scored."""
    import r7_d1_30_score as SC
    res = {"method": a.method, "out": a.out, "benches": {}, "problems": []}; fields = {"qid", "image_id", "pred", "kept", "n_vis", "f", "n_new", "hit_max", "k_rule", "ms"}
    for spec in a.benches:
        b, shard = spec.split("@") if "@" in spec else (spec, "0/1")
        out = Path(a.out) / b; qids = {it["qid"] for it in shard_items(a.sets_dir, b, shard, a.limit)}; mk = out / marker_name(a.method, shard)
        R = res["benches"][b] = {"shard": shard, "items": len(qids), "marker": mk.exists(), "configs": {}}
        if not mk.exists():
            res["problems"].append(f"{b}: no marker {mk.name}"); continue
        m = json.loads(mk.read_text())
        for f in run_fractions(a):
            c = cfg_name(a.method, f); recs, integ = SC.load_config(out, c); rows = [recs[q] for q in qids if q in recs]
            ses = ((m.get("sessions") or {}).get(c) or {}).get("sessions") or [{}]
            R["configs"][c] = {"rows": len(rows), "integrity": {k: integ[k] for k in ("dropped_lines", "blank_lines", "duplicate_qids")}, "files": integ["files"],
                               "kept_equals_rule": sum(r.get("kept") == k_rule(a.method, f, r.get("n_vis", -1)) for r in rows),
                               "mean_kept_fraction": float(np.mean([r["kept"] / r["n_vis"] for r in rows])) if rows else None,
                               "hit_max": sum(bool(r.get("hit_max")) for r in rows), "ms_median": float(np.median([r["ms"] for r in rows])) if rows else None,
                               "gpu": ses[-1].get("gpu"), "peak_mem_gb": ses[-1].get("peak_mem_gb"), "seconds_per_answer": ses[-1].get("seconds_per_answer"),
                               "attn_built": ses[-1].get("attn_built"), "processor": ((ses[-1].get("bounds") or {}).get("image_processor") or {}).get("class")}
            if set(recs) != qids or any(integ[k] for k in ("dropped_lines", "blank_lines", "duplicate_qids")):
                res["problems"].append(f"{b} {c}: {len(set(recs) & qids)} / {len(qids)} items, integrity {integ}")
            if any(set(r) != fields or r["f"] != f for r in rows):
                res["problems"].append(f"{b} {c}: row fields / f differ from {sorted(fields)}")
            if R["configs"][c]["kept_equals_rule"] != len(rows):
                res["problems"].append(f"{b} {c}: kept != the method's rule on {len(rows) - R['configs'][c]['kept_equals_rule']} rows")
        R["env"] = m.get("env"); R["code"] = (m.get("code") or {}).get("sha256", "")[:16]; R["method_commit"] = m.get("method_commit")
    res["pass"] = not res["problems"]; print("SMOKECHECK " + V.jdump(res, indent=1), flush=True)
    return 0 if res["pass"] else 8


# ------------------------------------------------------------------------------------------------------------------------------ CPU self-check
SELF_WH = ((500, 375), (320, 240), (700, 300))       # merged grids 14 x 19, 14 x 19, 11 x 25 under the frozen bounds


def tiny_model_dir(method: str, d: Path, attn: str):
    """The checkpoint's own config.json with small widths (vision depth 32 kept: object layer 16 and full-attention blocks 7 / 15 / 23 / 31), random
    weights, the shipped generation config; saved with the method's model class so the production loader reads it like the checkpoint."""
    import torch
    from transformers import GenerationConfig, Qwen2_5_VLConfig
    cfgd = json.loads((QH.QWEN_DIR / "config.json").read_text())
    cfgd.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_window_layers=2,
                rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]})
    cfgd["vision_config"].update(hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    cfg = Qwen2_5_VLConfig.from_dict(cfgd); torch.manual_seed(0)
    if method == "hiprune":
        add_method_path(method)
        import Qwen2_5_VL.qwen2_5_vl_HiPrune as HP
        cls = HP.Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    model = cls._from_config(cfg, attn_implementation=attn, torch_dtype=torch.float32)
    model.save_pretrained(str(d)); GenerationConfig.from_pretrained(str(QH.QWEN_DIR)).save_pretrained(str(d))
    return d


def probe(a) -> int:
    """CPU, one fraction: load_runtime + every self-check item through answer_item; an exception is recorded, not raised; the LLM prefill length
    seen by the hook gives the kept count.  Used for HiPrune, whose shipped LLM path needs flash_attention_2 (GPU only): with eager attention the
    pruned prefill fails in its _prepare_4d_causal_attention_mask_with_cache_position (cache_position keeps the unpruned length), after the
    selection has run.  Prints one line 'PROBE <json>'."""
    rt = load_runtime(a.method, float(a.fraction), a.model_dir, a.proc_dir, a.attn, "cpu", "float32"); res = []
    for it in shard_items(a.sets_dir, a.bench, "0/1"):
        rows, cols = RN.merged_grid(it["image_file"]); err, pred = None, None
        try:
            pred = answer_item(rt, a.method, float(a.fraction), it, RN.open_image(it["image_file"]), MAX_NEW[a.bench], expected_n=rows * cols)["pred"]
        except Exception as e:                                                         # recorded: the self-check reports it
            err = f"{type(e).__name__}: {str(e)[:300]}"
        L, n = (rt.last or {}).get("L"), (rt.last or {}).get("n_vis")
        res.append({"qid": it["qid"], "n_vis": n, "header_n": rows * cols, "prefill": rt.seen[0] if rt.seen else None,
                    "kept": (rt.seen[0] - (L - n)) if (rt.seen and L is not None) else None, "k_rule": k_rule(a.method, float(a.fraction), rows * cols),
                    "pred": pred, "error": err})
    print("PROBE " + V.jdump({"method": a.method, "fraction": float(a.fraction), "attn": a.attn, "attn_built": rt.info["attn_built"],
                              "knobs": rt.info["knobs"], "bounds": rt.info["bounds"], "items": res}), flush=True)
    return 0


def selfcheck(a) -> int:
    """CPU, method env: a tiny random Qwen2.5-VL through the production path (run -> one interpreter per fraction -> load_runtime with the method's
    official code -> answer_item -> rows -> marker) on three synthetic images; checks rows, kept == the method's k rule, the frozen bounds, resume."""
    from PIL import Image
    method = a.method; attn = "eager" if method == "hiprune" else "sdpa"                # flash-attn needs a GPU; the GPU smoke runs FA2
    os.environ.update({k: v for k, v in child_env(method, ANCHOR).items() if k in KNOB_VARS})   # only for building the tiny model in this process
    res = {"method": method, "attn_cpu": attn, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "problems": []}
    with tempfile.TemporaryDirectory(prefix="r7e1_selfcheck_") as t:
        t = Path(t); sets = t / "sets"; (sets / "img").mkdir(parents=True); items = []
        for i, (w, h) in enumerate(SELF_WH):
            p = sets / "img" / f"img{i}.png"; Image.fromarray(np.random.default_rng(i).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(p)
            items.append({"bench": "e1_selfcheck", "qid": f"sc{i}", "image_id": f"img{i}", "image_file": str(p), "question": f"What is object {i}?",
                          "prompt": f"What is object {i}?" + V.SHORT})
        (sets / "e1_selfcheck.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
        mdir = tiny_model_dir(method, t / "model", attn)
        if method == "hiprune":                                                        # FA2 is GPU-only; eager fails after the selection (probe docstring)
            res["probe"] = {}
            for f in fractions(True):
                cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "probe", "--method", method, "--fraction", PL.fkey(f), "--bench", "e1_selfcheck",
                       "--sets-dir", str(sets), "--model-dir", str(mdir), "--proc-dir", str(QH.QWEN_DIR), "--attn", attn]
                r = subprocess.run(cmd, env=child_env(method, f), cwd=str(REPO), capture_output=True, text=True)
                line = [x for x in r.stdout.splitlines() if x.startswith("PROBE ")]
                if r.returncode != 0 or not line:
                    res["problems"].append(f"probe {f}: exit {r.returncode}: {r.stderr[-800:]}"); continue
                pr = json.loads(line[-1][6:]); res["probe"][PL.fkey(f)] = pr
                for x in pr["items"]:
                    if (x["kept"], x["n_vis"]) != (x["k_rule"], x["header_n"]):
                        res["problems"].append(f"probe {f} {x['qid']}: kept {x['kept']} n_vis {x['n_vis']} vs rule {x['k_rule']} header {x['header_n']}")
                if f == ANCHOR and any(x["error"] for x in pr["items"]):
                    res["problems"].append(f"anchor (nothing pruned) failed with eager: {[x['error'] for x in pr['items']]}")
            res["eager_llm_errors_pruned"] = sorted({x["error"].split(":")[0] for k, pr in res["probe"].items() if k != "1" for x in pr["items"] if x["error"]})
            res["pass"] = not res["problems"]
            Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / f"selfcheck_{method}.json").write_text(V.jdump(res, indent=1))
            print(V.jdump({k: v for k, v in res.items() if k != "probe"}, indent=1))
            for k, pr in res["probe"].items():
                print(f"probe f={k}: " + "; ".join(f"{x['qid']} N={x['n_vis']} kept={x['kept']} rule={x['k_rule']} err={x['error'] and x['error'][:120]}" for x in pr["items"]))
            return 0 if res["pass"] else 9
        args = argparse.Namespace(method=method, bench="e1_selfcheck", shard="0/1", sets_dir=str(sets), out=str(t / "pred"), anchor=True, limit=0,
                                  model_dir=str(mdir), proc_dir=str(QH.QWEN_DIR), attn=attn, device="cpu", dtype="float32")
        calls = []

        def counted(aa, f, qdir):
            calls.append(f); return spawn_worker(aa, f, qdir)
        run(args, spawn=counted); first = list(calls); calls.clear(); run(args, spawn=counted)
        res["first_run_fractions"] = first; res["second_run_fractions"] = list(calls)
        if calls:
            res["problems"].append(f"resume ran {calls} again")
        od = t / "pred" / "e1_selfcheck"; mk = json.loads((od / marker_name(method, "0/1")).read_text())
        res["kept_check"] = mk["kept_check"]; res["rows"] = {}
        for c in config_list((method,), anchor=True):
            recs = first_records(od, c); m_, f = parse_cfg(c); res["rows"][c] = list(recs.values())
            for it in items:
                r = recs.get(it["qid"])
                if r is None:
                    res["problems"].append(f"{c}: {it['qid']} missing"); continue
                n = int(np.prod(RN.merged_grid(it["image_file"])))
                if (r["n_vis"], r["kept"], r["k_rule"]) != (n, k_rule(method, f, n), k_rule(method, f, n)):
                    res["problems"].append(f"{c} {it['qid']}: n_vis {r['n_vis']} kept {r['kept']} k_rule {r['k_rule']} vs N {n} rule {k_rule(method, f, n)}")
        metas = mk["sessions"]; res["sessions"] = {c: (m or {}).get("sessions", [{}])[0] for c, m in metas.items()}
    res["pass"] = not res["problems"]
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / f"selfcheck_{method}.json").write_text(V.jdump(res, indent=1))
    print(V.jdump({k: v for k, v in res.items() if k != "rows"}, indent=1))
    return 0 if res["pass"] else 9


# ------------------------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "worker"):
        p = sub.add_parser(name); p.add_argument("--method", required=True, choices=sorted(METHODS)); p.add_argument("--bench", required=True)
        p.add_argument("--shard", default="0/1"); p.add_argument("--sets-dir", default=str(PL.D30_SETS)); p.add_argument("--out", default=str(E1 / "pred"))
        p.add_argument("--limit", type=int, default=0); p.add_argument("--model-dir", default=None); p.add_argument("--proc-dir", default=None)
        p.add_argument("--attn", default=None, choices=["sdpa", "eager", "flash_attention_2"]); p.add_argument("--device", default="cuda")
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
        if name == "run":
            p.add_argument("--anchor", action="store_true"); p.add_argument("--fractions", nargs="+", default=None)
        else:
            p.add_argument("--fraction", required=True)
    s = sub.add_parser("selfcheck"); s.add_argument("--method", required=True, choices=sorted(METHODS)); s.add_argument("--out", default=str(E1 / "checks"))
    sc = sub.add_parser("smokecheck"); sc.add_argument("--method", required=True, choices=sorted(METHODS)); sc.add_argument("--out", required=True)
    sc.add_argument("--sets-dir", default=str(PL.D30_SETS)); sc.add_argument("--benches", nargs="+", required=True, help="<bench>@<i/n> ...")
    sc.add_argument("--fractions", nargs="+", default=None); sc.add_argument("--anchor", action="store_true"); sc.add_argument("--limit", type=int, default=0)
    pr = sub.add_parser("probe"); pr.add_argument("--method", required=True, choices=sorted(METHODS)); pr.add_argument("--fraction", required=True)
    pr.add_argument("--bench", default="e1_selfcheck"); pr.add_argument("--sets-dir", required=True); pr.add_argument("--model-dir", required=True)
    pr.add_argument("--proc-dir", default=None); pr.add_argument("--attn", default="eager", choices=["sdpa", "eager", "flash_attention_2"])
    pl = sub.add_parser("plan"); pl.add_argument("--root", default=str(E1)); pl.add_argument("--lines", required=True); pl.add_argument("--anchor", action="store_true")
    pl.add_argument("--partition", default=PARTITION)
    k = sub.add_parser("complete"); k.add_argument("--sets-dir", default=str(PL.D30_SETS)); k.add_argument("--pred-dir", required=True)
    k.add_argument("--out", required=True); k.add_argument("--anchor", action="store_true"); k.add_argument("--benches", nargs="+", default=list(PL.BENCHES))
    k.add_argument("--methods", nargs="+", default=list(METHODS))
    a = ap.parse_args(argv)
    return {"run": run, "worker": worker, "selfcheck": selfcheck, "probe": probe, "smokecheck": smokecheck, "plan": plan, "complete": complete}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
