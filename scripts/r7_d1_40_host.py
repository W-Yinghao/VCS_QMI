"""r7 R7-40 — Qwen2.5-VL-7B-Instruct as the second D1 host, with selection AFTER the vision encoder's token merger (coordinator OPEN-1 (b)).

Host-interface facts (transformers 4.57.6 as installed in /projects/EEG-foundation-model/yinghao/envs/vl_baselines; read before running):
  - AutoProcessor(min_pixels 256·28², max_pixels 1280·28²) resolves Qwen2VLImageProcessor to the FAST (torchvision) processor (it is in
    FORCE_FAST_IMAGE_PROCESSOR) — the same class R7-20 used.  smart_resize gives H × W multiples of 28; patches are flattened in the order
    (t, H/28, W/28, 2, 2) so every 2 × 2 merge group is contiguous.
  - model.model.visual (Qwen2_5_VisionTransformerPretrainedModel): patch embed -> window reordering -> 32 blocks (full attention at 7 / 15 /
    23 / 31, window attention elsewhere; SDPA runs each window / image separately) -> PatchMerger (RMSNorm, 4 × 1280 -> 5120 -> 3584 MLP over
    each 2 × 2 group) -> inverse window reordering.  Output: N = (H/28)·(W/28) tokens × 3584 per image, raster order over the merged grid.
  - get_rope_index: text before the image t = h = w = position; image token (r, c) gets (t, h, w) = (s, s + r, s + c) with s the image's
    start position (images: temporal index 0); text after the image restarts at max + 1 = s + max(rows, cols).  Decoding continues at
    (current length + rope delta) = max position + 1, the same on the three axes.
  - The text model takes 3-row position ids (3, B, L) for the rotary embedding only; the causal mask comes from the 2-D attention mask and
    the cache positions (sequence order), never from position values, so pruned sequences with gaps in their position ids are well defined.
  - Pruning: kept visual tokens stay in raster order and keep their ORIGINAL (t, h, w) ids; every text token keeps its full-sequence position
    (positions of the full sequence from the official get_rope_index, pruned columns removed).  Text after the image therefore keeps the
    same positions as in the full sequence and decoding continues from the full sequence's rope delta.
  - Shipped generation config: do_sample true, temperature 1e-6, repetition_penalty 1.05, eos [151645, 151643], pad 151643.  R7-20 called
    generate(do_sample=False), i.e. greedy WITH the 1.05 repetition penalty over the prompt + generated ids.  The batched decode loop here
    reproduces exactly that (HF semantics: next = argmax(penalised fp32 logits); finished rows emit pad; stop when every row has emitted an
    eos id or at max_new_tokens; min_new_tokens masks the eos ids); a CPU test checks it token for token against HF generate on a tiny
    randomly initialised Qwen2.5-VL, single and left-padded batches.
P(yes) / P(no): softmax of the raw fp32 logits of the first generated token (as R7-20: output_logits are unprocessed), summed over the yes /
no token ids of R7-20.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_vl_common as V  # noqa: E402

QWEN_DIR = V.QWEN_DIR
MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 1280 * 28 * 28
SHORT = V.SHORT                                                       # "\nAnswer the question using a single word or phrase."
YES_WORDS = ["Yes", "yes", "YES", "ĠYes", "Ġyes"]                     # the R7-20 QwenHost lists
NO_WORDS = ["No", "no", "NO", "ĠNo", "Ġno"]


# ------------------------------------------------------------------------------------------------------------------------------ pure helpers
def split_image_segment(ids, image_token_id: int):
    """Token ids of a chat prompt with ONE image placeholder -> (ids before it, ids after it)."""
    ids = np.asarray(ids)
    where = np.flatnonzero(ids == image_token_id)
    if len(where) != 1:
        raise ValueError(f"expected exactly one image placeholder, found {len(where)}")
    return ids[: where[0]], ids[where[0] + 1:]


def kept_columns(n_pre: int, n_vis: int, n_post: int, kept_idx) -> np.ndarray:
    """Sequence columns of a pruned sequence inside the full one: every text column plus the kept visual columns (raster order)."""
    k = np.sort(np.asarray(kept_idx, dtype=np.int64))
    if len(k) and (k[0] < 0 or k[-1] >= n_vis or np.any(np.diff(k) == 0)):
        raise ValueError("kept indices must be distinct visual token indices")
    return np.concatenate([np.arange(n_pre), n_pre + k, n_pre + n_vis + np.arange(n_post)]).astype(np.int64)


def prune_positions(pos_full, n_pre: int, n_vis: int, kept_idx):
    """(3, L_full) position ids of the full sequence -> (3, L_kept) ids of the pruned sequence (original ids kept)."""
    n_post = pos_full.shape[-1] - n_pre - n_vis
    cols = kept_columns(n_pre, n_vis, n_post, kept_idx)
    return pos_full[..., cols]


def next_position(pos) -> int:
    """First decoding position: max position + 1 (= current length + rope delta of get_rope_index)."""
    return int(pos.max()) + 1


def repetition_penalty(scores, histories, penalty: float):
    """transformers RepetitionPenaltyLogitsProcessor on fp32 scores [n, V]: for every id in a row's history, logit < 0 -> × penalty,
    else ÷ penalty (ids counted once).  histories: list of 1-D LongTensors (prompt ids + generated ids)."""
    import torch
    out = scores.clone()
    for i, ids in enumerate(histories):
        if penalty == 1.0 or len(ids) == 0:
            continue
        ids = torch.as_tensor(ids, device=scores.device, dtype=torch.long)
        s = torch.gather(scores[i], 0, ids)
        out[i] = scores[i].scatter(0, ids, torch.where(s < 0, s * penalty, s / penalty))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ core
class QwenCore:
    """Prefill / decode on spliced embeddings with explicit 3-D position ids, around a loaded Qwen2_5_VLForConditionalGeneration (any size:
    the CPU tests use a tiny random one)."""

    def __init__(self, model, eos_ids, pad_id: int, penalty: float):
        import torch
        self.torch, self.model = torch, model
        self.lm, self.visual, self.lm_head = model.model.language_model, model.model.visual, model.lm_head
        self.embed = self.lm.embed_tokens
        self.image_token_id = int(model.config.image_token_id); self.merge = int(model.config.vision_config.spatial_merge_size)
        self.eos_ids = [int(e) for e in (eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])]
        self.pad_id, self.penalty = int(pad_id), float(penalty)
        self.dev = next(model.parameters()).device

    def visual_tokens(self, pixel_values, grid_thw):
        """Merged visual tokens per image (official path: pixel values cast to the tower dtype on the device)."""
        torch = self.torch
        with torch.inference_mode():
            emb = self.visual(pixel_values.to(self.dev).type(self.visual.dtype), grid_thw=grid_thw.to(self.dev))
        sizes = (grid_thw.prod(-1) // self.merge ** 2).tolist()
        return list(torch.split(emb, sizes))

    def grid_of(self, thw):
        return int(thw[1]) // self.merge, int(thw[2]) // self.merge

    def full_positions(self, pre, n_vis: int, post, thw):
        """Official get_rope_index on the full sequence pre + n_vis image tokens + post -> (3, L) long (CPU)."""
        torch = self.torch
        ids = torch.cat([torch.as_tensor(pre, dtype=torch.long), torch.full((n_vis,), self.image_token_id, dtype=torch.long),
                         torch.as_tensor(post, dtype=torch.long)])[None]
        pos, _ = self.model.model.get_rope_index(ids, image_grid_thw=torch.as_tensor(thw, dtype=torch.long).view(1, 3))
        return pos[:, 0].clone()

    def text_positions(self, L: int):
        return self.torch.arange(L, dtype=self.torch.long).view(1, -1).expand(3, -1).clone()

    def embed_ids(self, ids):
        with self.torch.inference_mode():
            return self.embed(self.torch.as_tensor(ids, dtype=self.torch.long, device=self.dev))

    def _pad(self, seqs, positions, left: bool):
        torch = self.torch; n, L = len(seqs), max(s.shape[0] for s in seqs); d = seqs[0].shape[1]
        x = torch.zeros(n, L, d, dtype=seqs[0].dtype, device=self.dev); am = torch.zeros(n, L, dtype=torch.long, device=self.dev)
        pid = torch.ones(3, n, L, dtype=torch.long, device=self.dev)                    # pads get position 1, as get_rope_index fills them
        for i, (s, p) in enumerate(zip(seqs, positions)):
            sl = slice(L - s.shape[0], L) if left else slice(0, s.shape[0])
            x[i, sl] = s; am[i, sl] = 1; pid[:, i, sl] = torch.as_tensor(p, device=self.dev)
        return x, am, pid

    def last_logits(self, seqs, positions):
        """Right-padded batched prefill; fp32 logits of each sequence's last real position."""
        torch = self.torch; x, am, pid = self._pad(seqs, positions, left=False)
        with torch.inference_mode():
            hs = self.lm(inputs_embeds=x, attention_mask=am, position_ids=pid, use_cache=False, return_dict=True).last_hidden_state
            last = hs[torch.arange(len(seqs), device=self.dev), am.sum(1) - 1]
            return self.lm_head(last[:, None, :])[:, -1].float()

    def generate(self, seqs, positions, prompt_ids, max_new_tokens=128, min_new_tokens=0):
        """Batched greedy decoding (left padding) with the shipped repetition penalty; returns (new token ids per row, first-step raw fp32
        logits [n, V]).  prompt_ids: the full prompt ids per row (the history the official processor sees)."""
        torch = self.torch; n = len(seqs); x, am, pid = self._pad(seqs, positions, left=True)
        nxt = torch.tensor([next_position(p) for p in positions], dtype=torch.long, device=self.dev)
        hist = [torch.as_tensor(p, dtype=torch.long, device=self.dev) for p in prompt_ids]
        eos = torch.tensor(self.eos_ids, device=self.dev); unfinished = torch.ones(n, dtype=torch.bool, device=self.dev)
        new = [[] for _ in range(n)]; first = None
        with torch.inference_mode():
            o = self.lm(inputs_embeds=x, attention_mask=am, position_ids=pid, use_cache=True, return_dict=True)
            cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
            for step in range(max_new_tokens):
                logits = self.lm_head(h)[:, -1].float()
                if first is None:
                    first = logits.clone()
                scores = repetition_penalty(logits, hist, self.penalty)
                if step < min_new_tokens:
                    scores[:, eos] = -float("inf")
                tok = scores.argmax(-1)
                tok = torch.where(unfinished, tok, torch.full_like(tok, self.pad_id))
                for i in range(n):
                    new[i].append(int(tok[i])); hist[i] = torch.cat([hist[i], tok[i:i + 1]])
                unfinished = unfinished & ~torch.isin(tok, eos)
                if not bool(unfinished.any()) or step == max_new_tokens - 1:
                    break
                am = torch.cat([am, torch.ones(n, 1, dtype=am.dtype, device=self.dev)], 1)
                p = (nxt + step).view(1, n, 1).expand(3, n, 1)
                o = self.lm(inputs_embeds=self.embed(tok)[:, None], attention_mask=am, position_ids=p, past_key_values=cache, use_cache=True,
                            return_dict=True)
                cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
        return new, first


# ------------------------------------------------------------------------------------------------------------------------------ host
class QwenPrefill:
    """Qwen2.5-VL-7B-Instruct loaded once per process (node-local staging as R7-20), bf16, SDPA, R7-20 pixel bounds; the pieces the masked
    teacher, the sentinels, the timing and the feature cache need."""

    def __init__(self, workdir: Path):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        t0 = time.time(); qdir, tmp, note = V.stage(QWEN_DIR); self.stage_notes = [note]
        self.proc = AutoProcessor.from_pretrained(str(qdir), min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        V.unstage(tmp); self.load_seconds = time.time() - t0; self.torch = torch; self.tok = self.proc.tokenizer
        gc = self.model.generation_config
        self.core = QwenCore(self.model, gc.eos_token_id, gc.pad_token_id, gc.repetition_penalty or 1.0)
        self.image_token_id = self.core.image_token_id
        ids = {lab: sorted({i for i in self.tok.convert_tokens_to_ids(w) if i is not None and i != self.tok.unk_token_id})
               for lab, w in (("yes", YES_WORDS), ("no", NO_WORDS))}
        self.yes_ids, self.no_ids = ids["yes"], ids["no"]
        self.info = {"image_processor": type(self.proc.image_processor).__name__, "attn_implementation": self.model.config._attn_implementation,
                     "dtype": str(self.model.dtype), "generation": {"eos": self.core.eos_ids, "pad": self.core.pad_id, "repetition_penalty": self.core.penalty},
                     "yes_ids": self.yes_ids, "no_ids": self.no_ids, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}

    # prompts ------------------------------------------------------------------------------------------------------------------------
    def chat_text(self, prompt: str, with_image: bool = True) -> str:
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": prompt}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def segments(self, prompt: str, with_image: bool = True):
        """Token ids before / after the image placeholder of the chat prompt (the processor expands that ONE placeholder into N image
        tokens; the text around it is tokenised separately because the placeholder is a special token — checked by sentinel S0)."""
        ids = self.tok(self.chat_text(prompt, with_image)).input_ids
        if not with_image:
            return np.asarray(ids), np.asarray(ids[:0])
        return split_image_segment(ids, self.image_token_id)

    def question_vector(self, question: str):
        ids = self.tok(question, add_special_tokens=False).input_ids
        return self.core.embed_ids(ids).float().mean(0)

    # vision -------------------------------------------------------------------------------------------------------------------------
    def preprocess(self, images):
        return self.proc.image_processor(images=images, return_tensors="pt")

    def encode(self, images):
        """PIL images -> [(merged visual tokens [N, 3584] bf16, (rows, cols), thw)] (official processor + tower)."""
        inp = self.preprocess(images); feats = self.core.visual_tokens(inp["pixel_values"], inp["image_grid_thw"])
        return [(f, self.core.grid_of(thw), thw) for f, thw in zip(feats, inp["image_grid_thw"])]

    # official reference path (R7-20 QwenHost.answer, same GPU) -------------------------------------------------------------------------
    def official_answer(self, prompt: str, image, max_new_tokens=128):
        torch = self.torch; text = self.chat_text(prompt, image is not None)
        inputs = self.proc(text=[text], images=[image] if image is not None else None, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                                      output_logits=True, return_dict_in_generate=True)
        n_in = inputs["input_ids"].shape[1]; new = out.sequences[:, n_in:]; p = torch.softmax(out.logits[0][0].float(), -1)
        return {"text": self.proc.batch_decode(new, skip_special_tokens=True)[0].strip(), "p_yes": float(p[self.yes_ids].sum()),
                "p_no": float(p[self.no_ids].sum()), "input_ids": inputs["input_ids"][0].cpu(),
                "pixel_values": inputs.get("pixel_values"), "image_grid_thw": inputs.get("image_grid_thw")}

    def decode(self, token_lists):
        return [t.strip() for t in self.proc.batch_decode([list(x) for x in token_lists], skip_special_tokens=True)]


def yes_no(logits, yes_ids, no_ids):
    p = logits.softmax(-1)
    return p[:, yes_ids].sum(1).cpu().numpy(), p[:, no_ids].sum(1).cpu().numpy()
