"""r7 R7-20 shared pieces for the D1 / D2 fixtures.
- Fixed fixtures (pre-registered by a hash rule, grouped by image) from the lmms-lab parquet releases fetched by the data steward:
  GQA val-balanced (never testdev), TextVQA validation, POPE (one configuration).
- The host: LLaVA-1.5-7B (liuhaotian/llava-v1.5-7b) through the official LLaVA code shipped in the SparseVLM repository
  (Gumpest/SparseVLMs a9e7142, Apache-2.0), optionally with SparseVLM's official sparsification (dynamic_sparse=True, RETAIN_TOKN, USE_VERSION).
- A per-item CUDA-synchronised timing split: vision (encode_images), selector (SparseVLM scoring / top-k / merge functions), prefill (first LM
  forward minus selector), decode (remaining LM forwards).
- Official answer metrics: GQA exact match after LLaVA's scripts/convert_gqa_for_eval.py normalisation; TextVQA soft accuracy with the m4c
  evaluator in llava/eval/m4c_evaluator.py; POPE yes/no parsing of llava/eval/eval_pope.py.
Runs in /projects/EEG-foundation-model/yinghao/envs/sparsevlm (torch 2.1.2, transformers 4.37) with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

DATA = Path("/projects/EEG-foundation-model/yinghao/data")
MODELS = Path("/projects/EEG-foundation-model/yinghao/models")
LLAVA_DIR = MODELS / "llava-v1.5-7b"
CLIP_DIR = MODELS / "clip-vit-large-patch14-336"
CLIP_HUB_ID = "openai/clip-vit-large-patch14-336"
REF_DIR = Path(__file__).resolve().parents[1] / "r7" / "package_r7_20261011" / "reference"
SHORT = "\nAnswer the question using a single word or phrase."      # LLaVA-1.5 short-answer suffix (GQA / TextVQA / POPE question files)


# ------------------------------------------------------------------------------------------------------------------------------ io
def _json_default(o):
    """torch / numpy scalars and arrays, Paths -> JSON (SparseVLM keeps some counters as 0-d tensors)."""
    if type(o).__module__.startswith("torch") and hasattr(o, "tolist"):
        return o.item() if getattr(o, "numel", lambda: 0)() == 1 else o.tolist()
    if type(o).__module__.startswith("numpy") and hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def jdump(obj, **kw) -> str:
    return json.dumps(obj, default=_json_default, **kw)


def read_jsonl(path: Path):
    """Complete records of a JSONL file; a torn last line (crash mid-write) is dropped."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def cuda_peak_reset(torch):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_gb(torch):
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None


# ------------------------------------------------------------------------------------------------------------------------------ fixtures
def _h(tag, key):
    return hashlib.sha256(f"r7-20/{tag}/{key}".encode()).hexdigest()


def _rows(files, columns, batch=512):
    import pyarrow.parquet as pq
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=batch, columns=columns):
            yield from b.to_pylist()


def _save_image(struct, stem: Path):
    """Write the stored image bytes unchanged; the extension follows the decoded format."""
    from PIL import Image
    data = struct["bytes"]; fmt = (Image.open(io.BytesIO(data)).format or "bin").lower().replace("jpeg", "jpg")
    path = stem.with_suffix("." + fmt); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return str(path)


def _pick(groups, tag, n, per_group):
    """Groups in hash order, up to per_group items each (hash order inside), until n items."""
    sel = []
    for g in sorted(groups, key=lambda x: _h(tag + "-group", x)):
        take = sorted(groups[g], key=lambda r: _h(tag + "-item", r["_key"]))[:per_group]
        sel.extend(take[: n - len(sel)])
        if len(sel) >= n:
            break
    return sel


def build_gqa(out: Path, n=200, per_image=2):
    groups = {}
    for r in _rows(sorted((DATA / "gqa_lmms/val_balanced_instructions").glob("*.parquet")), ["id", "imageId", "question", "answer", "types"]):
        r["_key"] = r["id"]; groups.setdefault(r["imageId"], []).append(r)
    sel = _pick(groups, "gqa", n, per_image); need = {r["imageId"] for r in sel}; files = {}
    for r in _rows(sorted((DATA / "gqa_lmms/val_balanced_images").glob("*.parquet")), ["id", "image"], batch=64):
        if r["id"] in need and r["id"] not in files:
            files[r["id"]] = _save_image(r["image"], out / "images" / "gqa" / r["id"])
            if len(files) == len(need):
                break
    return [{"fixture": "gqa", "qid": r["id"], "image_id": r["imageId"], "image_file": files[r["imageId"]], "question": r["question"],
             "prompt": r["question"] + SHORT, "answer": r["answer"], "binary": r["answer"] in ("yes", "no"),
             "label": {"yes": 1, "no": 0}.get(r["answer"]), "structural": (r["types"] or {}).get("structural"),
             "semantic": (r["types"] or {}).get("semantic")} for r in sel], {"source": "lmms-lab/GQA val_balanced (rev a6e72d6)", "groups": len(groups)}


def build_textvqa(out: Path, n=200, per_image=2):
    shards = sorted((DATA / "textvqa_lmms/data").glob("validation-*.parquet")); groups = {}
    for r in _rows(shards, ["question_id", "image_id", "question", "answers", "ocr_tokens"]):
        r["_key"] = str(r["question_id"]); groups.setdefault(r["image_id"], []).append(r)
    sel = _pick(groups, "textvqa", n, per_image); need = {r["image_id"] for r in sel}; files = {}
    for r in _rows(shards, ["image_id", "image"], batch=64):
        if r["image_id"] in need and r["image_id"] not in files:
            files[r["image_id"]] = _save_image(r["image"], out / "images" / "textvqa" / r["image_id"])
            if len(files) == len(need):
                break
    # LLaVA's llava_textvqa_val_v051_ocr.jsonl layout: question, "Reference OCR token: …", short-answer suffix (3 lines)
    return [{"fixture": "textvqa", "qid": str(r["question_id"]), "image_id": r["image_id"], "image_file": files[r["image_id"]], "question": r["question"],
             "prompt": f"{r['question']}\nReference OCR token: {', '.join(r['ocr_tokens'] or [])}{SHORT}", "answers": list(r["answers"]),
             "binary": False, "label": None} for r in sel], {"source": "lmms-lab/textvqa validation (rev 9c0699c)", "groups": len(groups)}


def build_pope(out: Path, category="adversarial", n_images=50):
    shards = sorted((DATA / "pope_lmms/data").glob("test-*.parquet")); groups = {}
    for r in _rows(shards, ["question_id", "question", "answer", "image_source", "category"]):
        if r["category"] == category:
            r["_key"] = str(r["question_id"]); groups.setdefault(r["image_source"], []).append(r)
    order = sorted(groups, key=lambda x: _h("pope-group", x))[:n_images]; need = set(order); files = {}
    for r in _rows(shards, ["image_source", "image", "category"], batch=64):
        if r["category"] == category and r["image_source"] in need and r["image_source"] not in files:
            files[r["image_source"]] = _save_image(r["image"], out / "images" / "pope" / r["image_source"])
            if len(files) == len(need):
                break
    sel = [r for g in order for r in sorted(groups[g], key=lambda r: _h("pope-item", r["_key"]))]
    return [{"fixture": "pope", "qid": str(r["question_id"]), "image_id": r["image_source"], "image_file": files[r["image_source"]],
             "question": r["question"], "prompt": r["question"] + SHORT, "answer": r["answer"], "binary": True,
             "label": {"yes": 1, "no": 0}[r["answer"].strip().lower()], "category": category} for r in sel], \
        {"source": f"lmms-lab/POPE {category} (rev 4db1276)", "groups": len(groups)}


def write_fixtures(fx_dir: Path, which=("gqa", "textvqa", "pope")):
    """Build every fixture once; ids are fixed by the hash rule before any model output exists."""
    fx_dir.mkdir(parents=True, exist_ok=True); man_path = fx_dir / "fixture_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"rule": "sha256('r7-20/<fixture>-group/<image>') order; up to k items per image "
                                                                            "in sha256('r7-20/<fixture>-item/<qid>') order", "fixtures": {}}
    for name in which:
        f = fx_dir / f"{name}.jsonl"
        if f.exists():
            continue
        items, meta = {"gqa": build_gqa, "textvqa": build_textvqa, "pope": build_pope}[name](fx_dir)
        f.write_text("".join(json.dumps(x) + "\n" for x in items))
        man["fixtures"][name] = {**meta, "items": len(items), "images": len({x["image_id"] for x in items}),
                                 "binary_items": sum(bool(x["binary"]) for x in items), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                 "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        man_path.write_text(json.dumps(man, indent=1))
    return man


def load_fixture(fx_dir: Path, name):
    return [json.loads(x) for x in (fx_dir / f"{name}.jsonl").read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------------------------------------ host
class Timing:
    """CUDA-synchronised per-item wall clock.  vision = encode_images (CLIP + projector); prefill = first LM forward of generate (SparseVLM's
    selection runs inside it); selector = SparseVLM's attn_postprocess_topk / select_attn_head_by_sum / cluster_and_merge / batch_index_select
    (timed inside the prefill; the eager attention it needs at the 3 pruning layers stays in prefill); decode = the remaining LM forwards."""

    def __init__(self, model, sparse, vision_owner=None, vision_attr="encode_images"):
        import torch
        self.torch = torch; self.reset()
        owner = model if vision_owner is None else vision_owner
        enc, fwd = getattr(owner, vision_attr), model.forward

        @functools.wraps(enc)
        def encode_images(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter(); r = enc(*a, **k); torch.cuda.synchronize(); self.vision += time.perf_counter() - t
            return r

        @functools.wraps(fwd)
        def forward(*a, **k):
            v0 = self.vision
            torch.cuda.synchronize(); t = time.perf_counter(); r = fwd(*a, **k); torch.cuda.synchronize()
            dt = time.perf_counter() - t - (self.vision - v0)                           # vision run inside the forward (Qwen) is not prefill
            if self.n_fwd == 0:
                self.prefill += dt
            else:
                self.decode += dt
            self.n_fwd += 1
            return r
        setattr(owner, vision_attr, encode_images); model.forward = forward
        if sparse:
            import llava.model.language_model.modelling_sparse_llama as msl
            for name in ("attn_postprocess_topk", "select_attn_head_by_sum", "cluster_and_merge", "batch_index_select"):
                if hasattr(msl, name):
                    setattr(msl, name, self._wrap(getattr(msl, name)))

    def _wrap(self, f):
        torch = self.torch

        @functools.wraps(f)
        def g(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter(); r = f(*a, **k); torch.cuda.synchronize(); self.selector += time.perf_counter() - t
            return r
        return g

    def reset(self):
        self.vision = self.prefill = self.selector = self.decode = 0.0; self.n_fwd = 0

    def item(self):
        return {"vision_ms": 1e3 * self.vision, "selector_ms": 1e3 * self.selector, "prefill_ms": 1e3 * (self.prefill - self.selector),
                "prefill_incl_selector_ms": 1e3 * self.prefill, "decode_ms": 1e3 * self.decode, "decode_steps": max(self.n_fwd - 1, 0)}


def stage(src: Path):
    """Copy a checkpoint directory to node-local scratch before loading: sequential reads (~60 MB/s measured on the shared NFS) beat the
    page-fault reads of a direct mmap load (~17 MB/s observed).  Falls back to the NFS path when local space is short or scratch is
    tmpfs / NFS.  Returns (path to load, directory to delete after loading or None, note)."""
    import shutil
    import tempfile
    base = os.environ.get("TMPDIR") or "/tmp"
    try:
        size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file() and ".cache" not in f.parts)
        st = os.statvfs(base); free = st.f_bavail * st.f_frsize; real = os.path.realpath(base); fstype, best = "", ""
        for line in open("/proc/mounts"):
            mp, fs = line.split()[1], line.split()[2]
            if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) > len(best):
                best, fstype = mp, fs
        if free < 1.3 * size or fstype in ("tmpfs", "ramfs", "nfs", "nfs4"):
            return src, None, f"not staged (free {free / 2 ** 30:.0f} GiB on {fstype or '?'})"
        tmp = Path(tempfile.mkdtemp(prefix=f"r7stage_{os.environ.get('SLURM_JOB_ID', 'x')}_", dir=base)); t0 = time.time()
        shutil.copytree(src, tmp / src.name, ignore=shutil.ignore_patterns(".cache", ".git*"))
        return tmp / src.name, tmp, f"staged {size / 2 ** 30:.1f} GiB to {fstype} in {time.time() - t0:.0f}s"
    except Exception as e:                                                             # never fail the run because of staging
        return src, None, f"not staged ({type(e).__name__}: {e})"


def unstage(tmp):
    import shutil
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)


class Host:
    """LLaVA-1.5-7B via the official builder (llava.model.builder.load_pretrained_model).  The checkpoint's config names the vision tower by
    its hub id; a symlink farm in the working directory resolves that id to the local copy (no network, no edit of the downloaded files).
    RETAIN_TOKN / USE_VERSION are read by SparseVLM at import time, so one configuration per process."""

    def __init__(self, workdir: Path, sparse: bool, retain: int = 192, version: str = "1_0"):
        os.environ["USE_VERSION"], os.environ["RETAIN_TOKN"] = version, str(retain)
        t0 = time.time(); llava_dir, tmp1, n1 = stage(LLAVA_DIR); clip_dir, tmp2, n2 = stage(CLIP_DIR); self.stage_notes = [n1, n2]
        farm = workdir / "hub_links" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"; link = farm / CLIP_HUB_ID
        link.parent.mkdir(parents=True, exist_ok=True); link.symlink_to(clip_dir, target_is_directory=True)
        os.chdir(farm)
        import torch
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        self.tok, self.model, self.proc, _ = load_pretrained_model(str(llava_dir), None, get_model_name_from_path(str(LLAVA_DIR)), dynamic_sparse=sparse)
        unstage(tmp1); unstage(tmp2)                                                   # weights now live on the GPU
        self.load_seconds = time.time() - t0; self.sparse = sparse; self.torch = torch; self.model.eval()
        self.timing = Timing(self.model, sparse)
        ids = {lab: sorted({i for i in self.tok.convert_tokens_to_ids(w) if i is not None and i != self.tok.unk_token_id})
               for lab, w in (("yes", ["▁Yes", "▁yes", "Yes", "yes", "▁YES"]), ("no", ["▁No", "▁no", "No", "no", "▁NO"]))}
        self.yes_ids, self.no_ids = ids["yes"], ids["no"]

    def reset_sparse_counters(self):
        m = self.model.get_model()
        for a in ("total_cuda_time", "num_forward", "num_token_pool", "all_FLOPs"):
            if hasattr(m, a):
                setattr(m, a, 0)

    def sparse_counters(self):
        m = self.model.get_model()
        if not self.sparse or not getattr(m, "num_forward", 0):
            return {}
        n = int(m.num_forward)                                                          # SparseVLM keeps some counters as 0-d tensors
        return {"equal_visual_tokens": float(m.num_token_pool) / n, "official_prefill_cuda_ms_mean": float(m.total_cuda_time) / n,
                "official_tflops_per_sample": float(m.all_FLOPs) / n * 1e-12, "official_forwards": n}

    def answer(self, prompt: str, image_file: str | None, max_new_tokens=128):
        """Greedy answer to one question (model_vqa_loader's prompt and image path); with image_file=None the prompt carries no image token
        (text-only control).  Returns the text, P(yes) / P(no) of the first generated token and the timing split."""
        torch = self.torch
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        qs = (DEFAULT_IMAGE_TOKEN + "\n" + prompt) if image_file else prompt
        conv = conv_templates["vicuna_v1"].copy(); conv.append_message(conv.roles[0], qs); conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        kw = {}
        if image_file:
            img = Image.open(image_file).convert("RGB")
            kw = {"images": process_images([img], self.proc, self.model.config)[0].unsqueeze(0).to(dtype=torch.float16, device="cuda"), "image_sizes": [img.size]}
        self.timing.reset(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(ids, **kw, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                                      output_scores=True, return_dict_in_generate=True)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        text = self.tok.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        p = torch.softmax(out.scores[0][0].float(), -1)
        return {"text": text, "p_yes": float(p[self.yes_ids].sum()), "p_no": float(p[self.no_ids].sum()), "new_tokens": int(out.sequences.shape[1]),
                "prompt_tokens": int(ids.shape[1]), "wall_ms": 1e3 * wall, **self.timing.item()}


QWEN_DIR = MODELS / "Qwen2.5-VL-7B-Instruct"


class QwenHost:
    """Qwen2.5-VL-7B-Instruct (D1 plan's named host) through transformers' Qwen2_5_VLForConditionalGeneration, bf16, SDPA attention,
    min_pixels 256·28², max_pixels 1280·28² (≤ 1280 visual tokens).  Greedy with the shipped generation config otherwise; P(yes) from the
    raw first-step logits.  Runs in /projects/EEG-foundation-model/yinghao/envs/vl_baselines."""
    sparse = False

    def __init__(self, workdir: Path):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        t0 = time.time(); qdir, tmp, note = stage(QWEN_DIR); self.stage_notes = [note]
        self.proc = AutoProcessor.from_pretrained(str(qdir), min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        unstage(tmp); self.load_seconds = time.time() - t0; self.torch = torch
        self.timing = Timing(self.model, False, vision_owner=self.model.model.visual, vision_attr="forward")
        tok = self.proc.tokenizer; self.image_token_id = self.model.config.image_token_id
        ids = {lab: sorted({i for i in tok.convert_tokens_to_ids(w) if i is not None and i != tok.unk_token_id})
               for lab, w in (("yes", ["Yes", "yes", "YES", "ĠYes", "Ġyes"]), ("no", ["No", "no", "NO", "ĠNo", "Ġno"]))}
        self.yes_ids, self.no_ids = ids["yes"], ids["no"]

    def reset_sparse_counters(self):
        pass

    def sparse_counters(self):
        return {}

    def answer(self, prompt: str, image_file: str | None, max_new_tokens=128):
        torch = self.torch
        from PIL import Image
        content = ([{"type": "image"}] if image_file else []) + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        imgs = [Image.open(image_file).convert("RGB")] if image_file else None
        inputs = self.proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        self.timing.reset(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                                      output_logits=True, return_dict_in_generate=True)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        n_in = inputs["input_ids"].shape[1]; new = out.sequences[:, n_in:]
        p = torch.softmax(out.logits[0][0].float(), -1)
        return {"text": self.proc.batch_decode(new, skip_special_tokens=True)[0].strip(), "p_yes": float(p[self.yes_ids].sum()),
                "p_no": float(p[self.no_ids].sum()), "new_tokens": int(new.shape[1]), "prompt_tokens": int(n_in),
                "visual_tokens": int((inputs["input_ids"] == self.image_token_id).sum()), "wall_ms": 1e3 * wall, **self.timing.item()}


# ------------------------------------------------------------------------------------------------------------------------------ metrics
def gqa_correct(pred: str, gold: str) -> float:
    """LLaVA scripts/convert_gqa_for_eval.py (rstrip('.'), lower) then GQA eval.py exact match."""
    return float(pred.rstrip(".").lower() == gold)


def pope_yes(text: str) -> int:
    """llava/eval/eval_pope.py: first sentence, commas removed; 'No' / 'not' / 'no' as a word -> no, else yes."""
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return 0 if ("No" in words or "not" in words or "no" in words) else 1


def pope_metrics(pred, label):
    tp = sum(p == 1 and y == 1 for p, y in zip(pred, label)); fp = sum(p == 1 and y == 0 for p, y in zip(pred, label))
    tn = sum(p == 0 and y == 0 for p, y in zip(pred, label)); fn = sum(p == 0 and y == 1 for p, y in zip(pred, label))
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(label), "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "yes_ratio": sum(pred) / len(pred), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def textvqa_scorer():
    """Per-item soft accuracy exactly as m4c TextVQAAccuracyEvaluator.eval_pred_list computes it."""
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    ev = TextVQAAccuracyEvaluator()
    return lambda pred, answers: float(ev._compute_answer_scores(answers).get(ev.answer_processor(pred), 0.0))


def cluster_bootstrap(values, groups, stat=None, reps=2000, seed=0):
    """95 % percentile interval of stat(values) resampling whole groups (images)."""
    import numpy as np
    stat = stat or (lambda v: float(np.mean(v)))
    v = np.asarray(values, float); g = np.asarray(groups); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    rng = np.random.default_rng(seed); bs = []
    for _ in range(reps):
        pick = rng.choice(ug, size=len(ug), replace=True)
        bs.append(stat(v[np.concatenate([idx[u] for u in pick])]))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def env_info():
    import importlib.metadata as md
    import torch
    d = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
    for pkg in ("transformers", "flash-attn"):
        try:
            d[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    if torch.cuda.is_available():
        d.update(gpu=torch.cuda.get_device_name(0), gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1))
    return d
