"""CMIM feature/fusion graph with explicit changes and no bundled weights."""
from __future__ import annotations

import os
import re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .manifest import sha256_file


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        context_dim = context_dim or query_dim
        self.heads, self.dim_head = heads, dim_head
        inner = heads * dim_head
        self.q = nn.Linear(query_dim, inner, bias=False)
        self.k = nn.Linear(context_dim, inner, bias=False)
        self.v = nn.Linear(context_dim, inner, bias=False)
        self.out = nn.Linear(inner, query_dim)

    def forward(self, query, context, mask=None):
        b, nq = query.shape[:2]
        nk = context.shape[1]
        q = self.q(query).reshape(b, nq, self.heads, self.dim_head).transpose(1, 2)
        k = self.k(context).reshape(b, nk, self.heads, self.dim_head).transpose(1, 2)
        v = self.v(context).reshape(b, nk, self.heads, self.dim_head).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) * self.dim_head ** -.5
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(logits.dtype).min)
        result = (logits.softmax(-1) @ v).transpose(1, 2).reshape(b, nq, -1)
        return self.out(result)


class CMIMFusion(nn.Module):
    """Retain CLS-based reference fusion, including the single context token.

    A single text context token makes the second attention's softmax equal to 1;
    this graph is retained in every causal loss comparison, not marketed as new.
    """
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(1024, 768, 1)
        self.conv = nn.Sequential(nn.Conv2d(768, 512, 3, padding=1), nn.ReLU())
        self.text_proj = nn.Parameter(torch.randn(768, 512) * 768 ** -.5)
        self.visual_norm, self.text_norm = nn.LayerNorm(512), nn.LayerNorm(512)
        self.attn = CrossAttention(512)
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(1024, 256), nn.PReLU(), nn.Dropout(.1), nn.Linear(256, 1))

    def forward(self, visual, text):
        x = self.conv(self.down(visual[:, :, None, None])).flatten(2).transpose(1, 2)
        x = self.visual_norm(x)
        t = self.text_norm(text[:, None, :] @ self.text_proj)
        z = torch.cat([x, self.attn(x, t)], dim=-1).mean(dim=1)
        # Do not use squeeze() on the batch dimension; CAL/TEST may end with B=1.
        return z, self.head(z).squeeze(-1)


class ImageRewardCMIM(nn.Module):
    visual_dim, text_dim, fusion_dim = 1024, 768, 1024

    def __init__(self, cfg, device="cpu"):
        super().__init__()
        weights, med_config = Path(cfg["checkpoint"]), Path(cfg["med_config"])
        if not weights.is_file() or not med_config.is_file():
            raise FileNotFoundError("Stage a local ImageReward checkpoint and med_config.json before training")
        expected = cfg.get("checkpoint_sha256")
        if not expected or expected == "FILL_AFTER_STAGING":
            raise ValueError("checkpoint_sha256 must be frozen from staged bytes before formal training")
        actual = sha256_file(weights)
        if actual != expected:
            raise ValueError("ImageReward checkpoint SHA256 mismatch")
        expected_med = cfg.get("med_config_sha256")
        if not expected_med or expected_med == "FILL_AFTER_STAGING" or sha256_file(med_config) != expected_med:
            raise ValueError("med_config SHA256 missing or mismatched")
        # Required tokenizer/config files must already exist in the HF cache.
        # Explicit offline mode prevents silent downloads/revisions during jobs.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import ImageReward
        except ImportError as exc:
            raise ImportError("Install a staged/pinned image-reward environment; synthetic is never a fallback") from exc
        self.reward = ImageReward.load(str(weights), device=device, med_config=str(med_config))
        self.initial_checkpoint_sha256 = actual
        self._freeze_half()
        self.image_to_text = nn.Conv2d(1024, 768, 1)
        self.text_cross_attention = CrossAttention(768)
        self.fusion = CMIMFusion()

    def _freeze_half(self):
        # Deterministic layer-index implementation, replacing the reference's
        # order-sensitive break half way through the boundary layer's parameters.
        for p in self.reward.parameters():
            p.requires_grad_(False)
        text_count = visual_count = 0
        for name, p in self.reward.blip.text_encoder.named_parameters():
            match = re.search(r"(?:^|\.)layer\.(\d+)\.", name)
            if match and int(match.group(1)) >= 6:
                p.requires_grad_(True)
                text_count += p.numel()
        for name, p in self.reward.blip.visual_encoder.named_parameters():
            match = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
            if (match and int(match.group(1)) >= 12) or name.startswith("norm."):
                p.requires_grad_(True)
                visual_count += p.numel()
        if text_count == 0 or visual_count == 0:
            raise RuntimeError("BLIP layer names differ from the locked adapter; inspect before training")

    def forward(self, images, prompts):
        token = self.reward.blip.tokenizer(list(prompts), padding="max_length", truncation=True,
                                           max_length=35, return_tensors="pt").to(images.device)
        image_tokens = self.reward.blip.visual_encoder(images)
        image_mask = torch.ones(image_tokens.shape[:2], dtype=torch.long, device=images.device)
        text_output = self.reward.blip.text_encoder(token.input_ids, attention_mask=token.attention_mask,
                                                   encoder_hidden_states=image_tokens,
                                                   encoder_attention_mask=image_mask, return_dict=True)
        text_tokens = text_output.last_hidden_state
        image_context = self.image_to_text(image_tokens[:, 1:, :].transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        conditioned_text = text_tokens + self.text_cross_attention(text_tokens, image_context)
        visual, text = image_tokens[:, 0, :], conditioned_text[:, 0, :]
        if visual.shape[-1] != self.visual_dim or text.shape[-1] != self.text_dim:
            raise RuntimeError("Unexpected BLIP dimensions")
        z, prediction = self.fusion(visual, text)
        return {"visual": visual, "text": text, "fusion": z, "prediction": prediction}


class TinyCMIM(nn.Module):
    """Explicit CPU integration fixture, never a scientific benchmark model."""
    visual_dim, text_dim, fusion_dim = 24, 16, 32

    def __init__(self, cfg=None, device="cpu"):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(32, 24))
        self.embed = nn.Embedding(257, 16, padding_idx=0)
        self.condition = nn.Linear(24, 16)
        self.fuse = nn.Sequential(nn.Linear(40, 32), nn.Tanh())
        self.head = nn.Linear(32, 1)

    def forward(self, images, prompts):
        byte_rows = [list(str(p).encode("utf-8")[:64]) for p in prompts]
        tokens = torch.zeros((len(prompts), 64), device=images.device, dtype=torch.long)
        for i, row in enumerate(byte_rows):
            if row:
                tokens[i, :len(row)] = torch.tensor(row, device=images.device) + 1
        mask = tokens.ne(0).unsqueeze(-1)
        text = (self.embed(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1)
        visual = self.visual(images)
        text = text + self.condition(visual)
        fusion = self.fuse(torch.cat([visual, text], -1))
        return {"visual": visual, "text": text, "fusion": fusion, "prediction": self.head(fusion).squeeze(-1)}


def build_model(cfg, device="cpu"):
    name = cfg.get("name", "imagereward_cmim")
    if name == "tiny_synthetic":
        return TinyCMIM(cfg, device).to(device)
    if name == "imagereward_cmim":
        return ImageRewardCMIM(cfg, device).to(device)
    raise ValueError(f"Unknown model: {name}")
