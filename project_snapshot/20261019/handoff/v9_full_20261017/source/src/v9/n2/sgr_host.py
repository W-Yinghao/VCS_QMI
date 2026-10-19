"""V9-N2 host: the SGR single-model path of SGRAF (Diao et al., AAAI 2021), as used unchanged by NCR (NeurIPS 2021) and L2RM (CVPR 2024).

Vendored from the NCR copy of SGRAF's `model.py` (Apache 2.0; https://github.com/XLearning-SCU/2021-NeurIPS-NCR, commit 60256ba;
SGRAF upstream https://github.com/Paranioar/SGRAF, commit 50d0c6f).  Only the SGR path is kept (EncoderImage, EncoderText, VisualSA, TextSA,
GraphReasoning, EncoderSimilarity with module_name='SGR', SCAN_attention); the SAF module and the optimiser wrapper are not part of the host used here.
Two deliberate deviations, both recorded in V9_N2_ADAPTER_NOTES.md: (i) `pack_padded_sequence(..., enforce_sorted=False)` so a batch need not be
length-sorted (per-sample outputs are identical); (ii) the Bi-GRU halving uses integer division (the upstream `/ 2` is a Python-2 remnant that NCR
already fixed).  Everything else is the official computation; `tests/v9/test_n2.py::test_vendored_host_matches_official` compares against the
official file when a clone is available.  Dimensions are parameters so that CPU fixtures can run at reduced size; the formal units use HOST_LOCK.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def l1norm(X, dim, eps=1e-8):
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    return torch.div(X, norm)


def l2norm(X, dim=-1, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(X, norm)


def _xavier_uniform_(m):
    for layer in m.children():
        for sub in (layer if isinstance(layer, nn.Sequential) else [layer]):
            if isinstance(sub, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(sub.in_features + sub.out_features)
                sub.weight.data.uniform_(-r, r); sub.bias.data.fill_(0)
            elif isinstance(sub, nn.BatchNorm1d):
                sub.weight.data.fill_(1); sub.bias.data.zero_()


class EncoderImage(nn.Module):
    """Region features (B, n_region, img_dim) -> local embeddings (B, n_region, embed) with an FC layer and l2 normalisation."""

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super().__init__()
        self.embed_size, self.no_imgnorm = embed_size, no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r); self.fc.bias.data.fill_(0)

    def forward(self, images):
        img_emb = self.fc(images)
        return img_emb if self.no_imgnorm else l2norm(img_emb, dim=-1)


class EncoderText(nn.Module):
    """Word ids (B, L) -> local word embeddings (B, L, embed) with a (bi-)GRU and l2 normalisation."""

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, no_txtnorm=False):
        super().__init__()
        self.embed_size, self.no_txtnorm, self.use_bi_gru = embed_size, no_txtnorm, use_bi_gru
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)
        self.cap_rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        cap_emb = self.dropout(self.embed(captions))
        packed = pack_padded_sequence(cap_emb, torch.as_tensor(lengths, dtype=torch.int64).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.cap_rnn(packed)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)
        if self.use_bi_gru:
            h = cap_emb.size(2) // 2
            cap_emb = (cap_emb[:, :, :h] + cap_emb[:, :, h:]) / 2
        return cap_emb if self.no_txtnorm else l2norm(cap_emb, dim=-1)


class VisualSA(nn.Module):
    def __init__(self, embed_dim, dropout_rate, num_region):
        super().__init__()
        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(num_region), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        _xavier_uniform_(self); self.softmax = nn.Softmax(dim=1)

    def forward(self, local, raw_global):
        l_emb = self.embedding_local(local); g_emb = self.embedding_global(raw_global)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        weights = self.softmax(self.embedding_common(l_emb.mul(g_emb)).squeeze(2))
        return l2norm((weights.unsqueeze(2) * local).sum(dim=1), dim=-1)


class TextSA(nn.Module):
    def __init__(self, embed_dim, dropout_rate):
        super().__init__()
        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        _xavier_uniform_(self); self.softmax = nn.Softmax(dim=1)

    def forward(self, local, raw_global):
        l_emb = self.embedding_local(local); g_emb = self.embedding_global(raw_global)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        weights = self.softmax(self.embedding_common(l_emb.mul(g_emb)).squeeze(2))
        return l2norm((weights.unsqueeze(2) * local).sum(dim=1), dim=-1)


class GraphReasoning(nn.Module):
    def __init__(self, sim_dim):
        super().__init__()
        self.graph_query_w = nn.Linear(sim_dim, sim_dim); self.graph_key_w = nn.Linear(sim_dim, sim_dim); self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU(); _xavier_uniform_(self)

    def forward(self, sim_emb):
        sim_edge = torch.softmax(torch.bmm(self.graph_query_w(sim_emb), self.graph_key_w(sim_emb).permute(0, 2, 1)), dim=-1)
        return self.relu(self.sim_graph_w(torch.bmm(sim_edge, sim_emb)))


def SCAN_attention(query, context, smooth, eps=1e-8):
    attn = torch.bmm(context, torch.transpose(query, 1, 2))
    attn = nn.LeakyReLU(0.1)(attn); attn = l2norm(attn, 2)
    attn = torch.transpose(attn, 1, 2).contiguous(); attn = F.softmax(attn * smooth, dim=2)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    weightedContext = torch.bmm(torch.transpose(context, 1, 2), attnT)
    return l2norm(torch.transpose(weightedContext, 1, 2), dim=-1)


class EncoderSimilarity(nn.Module):
    """(B_img, n_region, embed), (B_cap, L, embed) -> similarities (B_img, B_cap) in (0, 1) through the SGR graph-reasoning module."""

    def __init__(self, embed_size, sim_dim, num_region=36, sgr_step=3):
        super().__init__()
        self.v_global_w = VisualSA(embed_size, 0.4, num_region); self.t_global_w = TextSA(embed_size, 0.4)
        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim); self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)
        self.sim_eval_w = nn.Linear(sim_dim, 1); self.sigmoid = nn.Sigmoid()
        self.SGR_module = nn.ModuleList([GraphReasoning(sim_dim) for _ in range(sgr_step)])
        _xavier_uniform_(self)

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []; n_image = img_emb.size(0)
        img_glo = self.v_global_w(img_emb, torch.mean(img_emb, 1))
        for i in range(cap_emb.size(0)):
            n_word = int(cap_lens[i]); cap_i = cap_emb[i, :n_word, :].unsqueeze(0); cap_i_expand = cap_i.repeat(n_image, 1, 1)
            cap_glo_i = self.t_global_w(cap_i, torch.mean(cap_i, 1))
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = l2norm(self.sim_tranloc_w(torch.pow(torch.sub(Context_img, cap_i_expand), 2)), dim=-1)
            sim_glo = l2norm(self.sim_tranglo_w(torch.pow(torch.sub(img_glo, cap_glo_i), 2)), dim=-1)
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)
            for module in self.SGR_module:
                sim_emb = module(sim_emb)
            sim_all.append(self.sigmoid(self.sim_eval_w(sim_emb[:, 0, :])))
        return torch.cat(sim_all, 1)


class SGRHost(nn.Module):
    """The three trainable parts of the SGR host (image projection, text encoder, similarity reasoning) as one module."""

    def __init__(self, vocab_size, img_dim=2048, word_dim=300, embed_size=1024, sim_dim=256, num_layers=1, bi_gru=True, num_region=36, sgr_step=3):
        super().__init__()
        self.img_enc = EncoderImage(img_dim, embed_size)
        self.txt_enc = EncoderText(vocab_size, word_dim, embed_size, num_layers, use_bi_gru=bi_gru)
        self.sim_enc = EncoderSimilarity(embed_size, sim_dim, num_region=num_region, sgr_step=sgr_step)

    def forward(self, images, captions, lengths):
        """Similarity matrix (B_img, B_cap); pair i of a batch of pairs is its diagonal entry."""
        return self.sim_enc(self.img_enc(images), self.txt_enc(captions, lengths), lengths)


def hinge_per_anchor(sims: torch.Tensor, margin: float, hard_negative: bool = True):
    """SGRAF / NCR ContrastiveLoss *before* aggregation: per anchor i the caption-retrieval term (row i, negatives = other captions) and the
    image-retrieval term (column i, negatives = other images), max over negatives when hard_negative ('max_violation'), mean otherwise.
    Returns (ell_row, ell_col) of shape (B,); the official batch loss is ell_row.sum() + ell_col.sum()."""
    diagonal = sims.diag().view(sims.size(0), 1)
    cost_s = (margin + sims - diagonal.expand_as(sims)).clamp(min=0)
    cost_im = (margin + sims - diagonal.t().expand_as(sims)).clamp(min=0)
    mask = torch.eye(sims.size(0), dtype=torch.bool, device=sims.device)
    cost_s = cost_s.masked_fill(mask, 0); cost_im = cost_im.masked_fill(mask, 0)
    if hard_negative:
        return cost_s.max(1)[0], cost_im.max(0)[0]
    return cost_s.mean(1), cost_im.mean(0)
