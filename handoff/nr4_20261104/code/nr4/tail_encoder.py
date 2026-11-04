"""NR4 TAIL_LORA8 two-tower encoder (site implementation).

SigLIP2 google/siglip2-base-patch16-224 @75de2d55 (eager attention, fp32, TF32 off) loaded from the NR1 HF snapshot;
LoRA (r=8, alpha=8, dropout 0) injected with the package reference nr4.ref_adapters.inject_siglip_tail_lora into exactly
the 8 modules {vision_model,text_model}.encoder.layers.{10,11}.self_attn.{q_proj,v_proj}. Every other parameter is frozen
(including post_layernorm / final_layer_norm / pooling heads), but gradients flow through them to the LoRA weights.

Forward = the HF forward split at block 10: embeddings + blocks 0-9 (no LoRA, no grad needed) then blocks 10-11 + final
LayerNorm + pooler (with grad). Identical ops to SiglipVisionTransformer/SiglipTextTransformer.forward (text: no attention
mask, last-token pooling, as the NR1 processor returns input_ids only). Every forward uses the NR1 fixed batch shape B=32
(pad by repeating the last item, discard padded rows), so per-item outputs do not depend on chunk composition.
Output = sample-wise L2-normalised pooler output (the NR1 global feature definition).
torch.nn.MultiheadAttention fast path is disabled so the no-grad pass and the grad replay pass use the same kernels.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from nr4.ref_adapters import inject_siglip_tail_lora, siglip_tail_targets

MODEL_ID = 'google/siglip2-base-patch16-224'; REV = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'; B = 32; SPLIT = 10


def set_precision():
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


def load_siglip(device='cuda', lora=True, seed=1001):
    from transformers import AutoModel
    set_precision()
    m = AutoModel.from_pretrained(MODEL_ID, revision=REV, local_files_only=True, attn_implementation='eager', dtype=torch.float32).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    info = inject_siglip_tail_lora(m, seed=seed) if lora else None
    m.to(device).eval()
    return m, info


def base_weight_digest(m):
    """sha256 over all frozen (non-LoRA) parameters, in name order."""
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        if n.endswith('.A') or n.endswith('.B'): continue
        h.update(n.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def lora_params(m):
    return [(n, p) for n, p in m.named_parameters() if p.requires_grad]


def _pad(t):
    n = len(t)
    return t if n == B else torch.cat([t, t[-1:].expand(B - n, *t.shape[1:])], 0), n


def vision_prefix(m, pv):
    vm = m.vision_model; h = vm.embeddings(pv)
    for layer in vm.encoder.layers[:SPLIT]: h = layer(h, None)
    return h


def vision_tail(m, h):
    vm = m.vision_model
    for layer in vm.encoder.layers[SPLIT:]: h = layer(h, None)
    return vm.head(vm.post_layernorm(h))


def text_prefix(m, ids):
    tm = m.text_model; h = tm.embeddings(input_ids=ids, position_ids=None)
    for layer in tm.encoder.layers[:SPLIT]: h = layer(h, None)
    return h


def text_tail(m, h):
    tm = m.text_model
    for layer in tm.encoder.layers[SPLIT:]: h = layer(h, None)
    return tm.head(tm.final_layer_norm(h)[:, -1, :])


class TailEncoder:
    """Encode opaque items (rows of the uint8 pixel / token arrays) into current global features."""
    def __init__(self, model, pix_u8, tokens, device='cuda'):
        self.m, self.pix, self.tok, self.dev = model, pix_u8, tokens, device
        self.mean = torch.tensor([.5, .5, .5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([.5, .5, .5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.item_forwards = dict(image=0, text=0)

    def pixel_values(self, rows):
        """HF SiglipImageProcessor rescale (float64 * 1/255 -> float32) + normalize ((x-mean)/std, float32) on the cached
        resized uint8 image (do_rescale/do_normalize off at caching). Verified bitwise vs the full processor in acceptance."""
        u8 = torch.from_numpy(np.ascontiguousarray(self.pix[np.asarray(rows)])).to(self.dev)
        x = (u8.double() * (1 / 255)).float()
        return (x - self.mean) / self.std

    def _chunks(self, rows):
        rows = np.asarray(rows)
        for s in range(0, len(rows), B): yield rows[s:s+B]

    def prefix(self, kind, rows):
        """Blocks 0-9 output for each chunk (no grad; frozen weights, no LoRA)."""
        out = []
        with torch.no_grad():
            for r in self._chunks(rows):
                if kind == 'image':
                    x, n = _pad(self.pixel_values(r)); out.append((vision_prefix(self.m, x), n))
                else:
                    ids = torch.from_numpy(np.ascontiguousarray(self.tok[r])).long().to(self.dev)
                    x, n = _pad(ids); out.append((text_prefix(self.m, x), n))
        return out

    def tail(self, kind, h, n):
        z = vision_tail(self.m, h) if kind == 'image' else text_tail(self.m, h)
        self.item_forwards[kind] += n
        return F.normalize(z[:n].float(), dim=-1)

    def encode(self, kind, rows, grad=False):
        pre = self.prefix(kind, rows); outs = []
        with torch.set_grad_enabled(grad):
            for h, n in pre: outs.append(self.tail(kind, h, n))
        return torch.cat(outs), pre
