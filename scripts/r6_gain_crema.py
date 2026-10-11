"""r6 GAIN-P1 CREMA-D — on-demand video branch for emotion recognition (second natural dataset; protocol draft
reports/R6/R6_GAIN_P1_CREMAD_PROTOCOL_DRAFT_20261012.yaml).  Routing, policy and statistics come from scripts/r6_gain_pilot.py, which is
frozen with GAIN-P1 for AV-MNIST: imported here, never modified.

Data: CREMA-D media (LFS bytes verified against the pointers) under /projects/EEG-foundation-model/yinghao/data/crema_d/media/{AudioWAV,
VideoFlash}: 7442 clips, 91 actors, 6 intended emotions (all kept).  CREMA-D has no official test split.  Roles by actor: the single split
of outputs/R6_GAIN/cremad/roles.json (`r6_gain_pilot.py roles --dataset cremad`; --fold -1) or 5-fold actor cross-fitting (--fold 0..4:
DEV = one fifth of the actors, the other actors split 60 : 20 : 10 into FIT_TASK / FIT_POLICY / CAL).
U = audio: WAV (16 kHz) -> first 4 s (zero-padded) -> log-mel 64 x 400 on the device (cheap).  V = video: FLV (H.264 or VP6F video,
480 x 360, 29.97 fps; probe 2026-10-12) -> all frames decoded ->
8 uniformly spaced frames -> central square crop -> 112 x 112 (extra branch; decoding is part of its cost).  The cache (prep) and the cost
measurement use the same decoder and the same crop / resize code.
System: ResNet-18 audio encoder (1-channel conv1) -> a(U); per-frame ResNet-18 video encoder (mean over frames) + fusion MLP -> b(U, V);
both encoders start from the same ImageNet ResNet-18 checkpoint (--init imagenet) or from scratch; families brier / ce on L(a) + L(b), end
to end; epoch selected on CAL by Brier(a) + Brier(b).
Cost (primary-eligible): end-to-end wall-clock per clip at batch size 1, measured for every DEV clip (median of 3 warm repeats):
t_U = WAV read + log-mel + U branch + head a; t_pol = policy; t_V = FLV read + decode (in-process, one thread) + frame selection / crop / resize +
V branch + fusion.  Policy cost per clip = mean over clips of t_U + t_pol + 1[acquired] t_V.

    python scripts/r6_gain_crema.py probe
    python scripts/r6_gain_crema.py prep [--workers 32] [--decoder av|cli]
    python scripts/r6_gain_crema.py run --family brier --seed 0 [--fold -1] [--smoke] [--costs]
    python scripts/r6_gain_crema.py aggregate [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6_gain_pilot as P  # noqa: E402

MEDIA = Path("/projects/EEG-foundation-model/yinghao/data/crema_d/media")
OUTC = P.OUT / "cremad"
CACHE = OUTC / "cache"
FFMPEG = Path("/projects/share/apps/ffmpeg/8.1")                                  # module ffmpeg/8.1
RESNET18 = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"     # torchvision ImageNet ResNet-18 (IMAGENET1K_V1)
SR, N_SAMPLES, N_FFT, HOP, N_MELS = 16000, 64000, 400, 160, 64
N_MEL_FRAMES = N_SAMPLES // HOP                                                    # 400
K_FRAMES, SIDE = 8, 112
EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
EPOCHS, BATCH, LR, WD = 30, 32, 3e-4, 1e-4
N_FOLDS = 5
SMOKE_LIM = {"FIT_TASK": 128, "FIT_POLICY": 64, "CAL": 64, "DEV": 64}


# ---------------------------------------------------------------------------------------------------------------- roles
def stems_and_labels():
    with open(P.roles_path("cremad")) as fh:
        r = json.load(fh)
    stems = r["stems"]; y = np.array([EMOTIONS.index(s.split("_")[2]) for s in stems]); g = np.array(r["groups"])
    return stems, y, g


def crema_roles(fold, smoke=False, lim=None):
    """fold −1: the single actor split of roles.json.  fold k in 0..4: DEV = actors of fold k (fixed actor permutation), the remaining
    actors split 60 : 20 : 10 (FIT_TASK / FIT_POLICY / CAL) by a fold-specific permutation."""
    if fold < 0:
        roles = P.load_roles("cremad")
    else:
        _, _, g = stems_and_labels(); actors = np.unique(g); perm = actors[np.random.default_rng(P.SEED + 1).permutation(len(actors))]
        folds = np.array_split(perm, N_FOLDS); dev_a = folds[fold]; rest = np.setdiff1d(actors, dev_a)
        rest = rest[np.random.default_rng(P.SEED + 100 + fold).permutation(len(rest))]; n_fit = int(round(len(rest) * 2 / 3)); n_pol = int(round(len(rest) * 2 / 9))
        parts = {"FIT_TASK": rest[:n_fit], "FIT_POLICY": rest[n_fit:n_fit + n_pol], "CAL": rest[n_fit + n_pol:], "DEV": dev_a}
        roles = {k: np.sort(np.flatnonzero(np.isin(g, v))) for k, v in parts.items()}
    if smoke:
        lim = lim or SMOKE_LIM; roles = {k: v[:lim[k]] for k, v in roles.items()}
    return roles


# ---------------------------------------------------------------------------------------------------------------- decoding
def read_wav(path):
    """int16 mono at 16 kHz -> the first 4 s, zero-padded; returns (int16[64000], original length)."""
    from scipy.io import wavfile
    sr, x = wavfile.read(str(path))
    if sr != SR or x.dtype != np.int16:
        raise ValueError(f"{path}: sr {sr}, dtype {x.dtype}")
    if x.ndim > 1:
        x = x.mean(1).astype(np.int16)
    out = np.zeros(N_SAMPLES, np.int16); n = min(len(x), N_SAMPLES); out[:n] = x[:n]
    return out, len(x)


def ffmpeg_env():
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = f"{FFMPEG}/lib:" + env.get("LD_LIBRARY_PATH", ""); return env


def frame_index(n):
    return np.round(np.linspace(0, n - 1, K_FRAMES)).astype(np.int64)


def decode_selected(path, decoder="av"):
    """All frames are decoded (inter-coded H.264 / VP6F need them); the 8 uniformly spaced frames are returned as RGB uint8 [8, H, W, 3]."""
    if decoder == "av":
        import av
        with av.open(str(path)) as c:
            s = c.streams.video[0]; s.codec_context.thread_count = 1; frames = list(c.decode(s))
        idx = frame_index(len(frames)); return np.stack([frames[i].to_ndarray(format="rgb24") for i in idx]), len(frames)
    pr = json.loads(subprocess.run([str(FFMPEG / "bin/ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                    "-of", "json", str(path)], capture_output=True, text=True, env=ffmpeg_env(), check=True).stdout)["streams"][0]
    w, h = int(pr["width"]), int(pr["height"])
    raw = subprocess.run([str(FFMPEG / "bin/ffmpeg"), "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, env=ffmpeg_env(), check=True).stdout   # frames as decoded (no CFR duplication)
    a = np.frombuffer(raw, np.uint8); n = a.size // (w * h * 3); a = a[: n * w * h * 3].reshape(n, h, w, 3)
    return a[frame_index(n)].copy(), n


def video_codec(path):
    import av
    with av.open(str(path)) as c:
        return c.streams.video[0].codec_context.name


def crop_resize(sel):
    """RGB uint8 [K, H, W, 3] -> central square crop -> bilinear (antialiased) resize to 112 x 112 -> uint8 [K, 3, 112, 112]."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(sel)).permute(0, 3, 1, 2).float(); h, w = x.shape[2:]; s = min(h, w)
    x = x[:, :, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return F.interpolate(x, size=(SIDE, SIDE), mode="bilinear", antialias=True, align_corners=False).round().clamp(0, 255).to(torch.uint8).numpy()


def mel_fb():
    """HTK mel filterbank (64 triangles, 0–8 kHz) on the 201 rfft bins of a 400-point FFT."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1); hz = 700.0 * (10 ** (np.linspace(hz2mel(0.0), hz2mel(SR / 2), N_MELS + 2) / 2595.0) - 1.0)
    fb = np.zeros((N_FFT // 2 + 1, N_MELS))
    for i in range(N_MELS):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]; fb[:, i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------- model
def build_crema(init="imagenet", n_classes=6):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    class LogMel(nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer("fb", torch.from_numpy(mel_fb())); self.register_buffer("win", torch.hann_window(N_FFT))

        def forward(self, wave):                                                      # int16 [B, 64000] -> [B, 1, 64, 400]
            S = torch.stft(wave.float() / 32768.0, N_FFT, HOP, N_FFT, self.win, center=True, return_complex=True).abs() ** 2
            return torch.log(torch.matmul(S.transpose(1, 2), self.fb).transpose(1, 2)[:, :, :N_MEL_FRAMES] + 1e-6).unsqueeze(1)

    class System(nn.Module):
        def __init__(self):
            super().__init__()
            a, v = resnet18(weights=None), resnet18(weights=None)
            conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            if init == "imagenet":
                sd = torch.load(RESNET18, map_location="cpu", weights_only=True); a.load_state_dict(sd); v.load_state_dict(sd)
                with torch.no_grad():
                    conv.weight.copy_(a.conv1.weight.sum(1, keepdim=True))
            a.conv1 = conv; a.fc = nn.Identity(); v.fc = nn.Identity()
            self.logmel, self.u_enc, self.v_enc = LogMel(), a, v
            self.head_a = nn.Linear(512, n_classes); self.head_b = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, n_classes))
            self.register_buffer("mel_mu", torch.zeros(())); self.register_buffer("mel_sd", torch.ones(()))
            self.register_buffer("img_mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
            self.register_buffer("img_sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0)

        def cheap(self, wave):
            h = self.u_enc((self.logmel(wave) - self.mel_mu) / self.mel_sd); return h, self.head_a(h)

        def v_feat(self, frames):                                                     # uint8 [B, K, 3, 112, 112]
            b, k = frames.shape[:2]; return self.v_enc((frames.flatten(0, 1).float() - self.img_mu) / self.img_sd).view(b, k, -1).mean(1)

        def full(self, h, frames):
            return self.head_b(torch.cat([h, self.v_feat(frames)], 1))

    return System()


class Tensors:
    """Cached waveforms (int16) and frames (uint8) in host memory; batches are moved to the device on demand."""
    def __init__(self, dev):
        import torch
        self.dev, self.dataset = dev, "cremad"
        self.wave = torch.from_numpy(np.load(CACHE / "wave_i16.npy")); self.frames = torch.from_numpy(np.load(CACHE / "frames_u8.npy"))
        self.y = torch.as_tensor(np.load(CACHE / "labels.npy")).long()

    def u_batch(self, idx):
        return self.wave[idx].to(self.dev, non_blocking=True)

    def v_batch(self, idx):
        return self.frames[idx].to(self.dev, non_blocking=True)

    def y_batch(self, idx):
        return self.y[idx].to(self.dev, non_blocking=True)


def train_crema(T, roles, family, seed, epochs, dev, log, init):
    import copy
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_crema(init).to(dev); fit = torch.as_tensor(roles["FIT_TASK"])
    with torch.no_grad():                                                             # log-mel normalisation from FIT_TASK
        s1 = s2 = n = 0.0
        for i in range(0, len(fit), 256):
            lm = model.logmel(T.u_batch(fit[i:i + 256])).double(); s1 += float(lm.sum()); s2 += float((lm ** 2).sum()); n += lm.numel()
        model.mel_mu.fill_(s1 / n); model.mel_sd.fill_(math.sqrt(max(s2 / n - (s1 / n) ** 2, 0.0)) + 1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); g = torch.Generator().manual_seed(seed); best = (math.inf, 0, None); hist = []
    for ep in range(1, epochs + 1):
        model.train(); perm = fit[torch.randperm(len(fit), generator=g)]; t0 = time.time(); tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:                                                          # BatchNorm needs > 1 sample
                continue
            y = T.y_batch(idx); h, la = model.cheap(T.u_batch(idx)); lb = model.full(h, T.v_batch(idx))
            loss = P.family_loss(la, y, family) + P.family_loss(lb, y, family); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t0; _, pa, pb = P.predict(model, T, roles["CAL"], chunk=64); yc = T.y[roles["CAL"]].numpy()
        ra, rb = float(P.brier_vec(pa, yc).mean()), float(P.brier_vec(pb, yc).mean())
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "cal_brier_a": ra, "cal_brier_b": rb, "cal_acc_a": float((pa.argmax(1) == yc).mean()),
                     "cal_acc_b": float((pb.argmax(1) == yc).mean()), "seconds_train": t_train, "seconds": time.time() - t0})
        log(f"epoch {ep}: loss {tot / len(perm):.4f} CAL Brier a {ra:.4f} b {rb:.4f} acc a {hist[-1]['cal_acc_a']:.3f} b {hist[-1]['cal_acc_b']:.3f} ({time.time() - t0:.1f}s)")
        if ra + rb < best[0]:
            best = (ra + rb, ep, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2]); return model, hist, best[1]


# ---------------------------------------------------------------------------------------------------------------- costs
def clip_costs(model, policy_score, stems, idx, dev, decoder, repeats=3, log=print):
    """Per-clip end-to-end wall-clock (ms) at batch size 1 for the given clips: one warm-up pass (recorded as 'cold'), then `repeats`
    passes; the per-clip median of the repeats is the primary value.  CPU work single-threaded (torch threads 1, decoder thread 1)."""
    import torch
    cuda = dev.type == "cuda"; model.eval(); nthr = torch.get_num_threads(); torch.set_num_threads(1)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    keys = ("t_u", "t_pol_mlp", "t_pol_entropy", "t_v", "t_v_decode", "t_v_prep", "t_v_gpu"); runs = []
    codecs = [video_codec(MEDIA / "VideoFlash" / f"{stems[ci]}.flv") for ci in idx]
    with torch.no_grad():
        for rep in range(repeats + 1):
            rec = {k: np.zeros(len(idx)) for k in keys}
            for j, ci in enumerate(idx):
                stem = stems[ci]; sync(); t0 = time.perf_counter()
                wave, _ = read_wav(MEDIA / "AudioWAV" / f"{stem}.wav"); h, la = model.cheap(torch.from_numpy(wave)[None].to(dev)); pa = la.softmax(1); sync(); t1 = time.perf_counter()
                X = torch.cat([h, pa], 1).float().cpu().numpy(); policy_score(X); t2 = time.perf_counter()
                float(-(pa * pa.clamp_min(1e-12).log()).sum()); t3 = time.perf_counter()
                sel, _ = decode_selected(MEDIA / "VideoFlash" / f"{stem}.flv", decoder); t4 = time.perf_counter()
                fr = torch.from_numpy(crop_resize(sel))[None]; t5 = time.perf_counter()
                model.full(h, fr.to(dev)).softmax(1); sync(); t6 = time.perf_counter()
                for k, v in zip(keys, (t1 - t0, t2 - t1, t3 - t2, t6 - t3, t4 - t3, t5 - t4, t6 - t5)):
                    rec[k][j] = 1e3 * v
            runs.append(rec); log(f"cost pass {rep}: mean t_U {rec['t_u'].mean():.2f} ms, t_V {rec['t_v'].mean():.2f} ms (decode {rec['t_v_decode'].mean():.2f})")
    torch.set_num_threads(nthr)
    out = {k: np.median(np.stack([r[k] for r in runs[1:]]), 0) for k in keys}; out.update({f"cold_{k}": runs[0][k] for k in keys})
    return out, codecs


def policy_cost(c, pol, acq):
    """Mean ms / clip for a routing: t_U + policy + acquired x t_V; acq = 0 / 1 mask or acquisition probability."""
    pc = {"gain_z": c["t_pol_mlp"], "gain_h": c["t_pol_mlp"], "entropy": c["t_pol_entropy"], "maxprob": c["t_pol_entropy"]}.get(pol, 0.0)
    return float(np.mean(c["t_u"] + pc + np.asarray(acq, float) * c["t_v"]))


# ---------------------------------------------------------------------------------------------------------------- subcommands
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")                                                       # ffprobe reports "N/A" for some FLV fields


def cmd_probe(a):
    """Video stream properties (ffprobe) for a fixed sample of clips and every WAV header; duration statistics."""
    from scipy.io import wavfile
    stems, y, g = stems_and_labels(); rng = np.random.default_rng(P.SEED); sample = sorted(rng.choice(len(stems), min(60, len(stems)), replace=False))
    vids = []
    for i in sample:
        pr = json.loads(subprocess.run([str(FFMPEG / "bin/ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,pix_fmt", "-of", "json",
                                        str(MEDIA / "VideoFlash" / f"{stems[i]}.flv")], capture_output=True, text=True, env=ffmpeg_env(), check=True).stdout)["streams"][0]
        vids.append(pr)
    srs, lens, dtypes = [], [], set()
    for s in stems:
        sr, x = wavfile.read(str(MEDIA / "AudioWAV" / f"{s}.wav"), mmap=True); srs.append(sr); lens.append(x.shape[0] / sr); dtypes.add(str(x.dtype))
    lens = np.array(lens)
    res = {"n_clips": len(stems), "n_actors": int(len(np.unique(g))), "label_counts": dict(zip(EMOTIONS, np.bincount(y, minlength=6).tolist())),
           "video_sample": {"n": len(vids), "codecs": sorted({v.get("codec_name") for v in vids}), "sizes": sorted({f"{v.get('width')}x{v.get('height')}" for v in vids}),
                            "r_frame_rate": sorted({v.get("r_frame_rate") for v in vids}), "pix_fmt": sorted({v.get("pix_fmt") for v in vids}),
                            "duration_s_percentiles": {str(q): float(np.nanpercentile([_num(v.get("duration")) for v in vids], q)) for q in (0, 50, 100)},
                            "nb_frames_percentiles": {str(q): float(np.nanpercentile([_num(v.get("nb_frames")) for v in vids], q)) for q in (0, 50, 100)}},
           "audio": {"sample_rates": sorted(set(srs)), "dtypes": sorted(dtypes), "duration_s_percentiles": {str(q): float(np.percentile(lens, q)) for q in (0, 1, 50, 99, 100)},
                     "frac_longer_than_4s": float((lens > N_SAMPLES / SR).mean())}}
    OUTC.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUTC / "probe.json", "w"), indent=1); print(json.dumps(res, indent=1)); return 0


def cmd_cvroles(a):
    """Record the 5-fold actor cross-fitting split: per fold and role, clips, actors, label counts and the sha256 of the indices."""
    import hashlib
    stems, y, g = stems_and_labels(); rec = {"n_folds": N_FOLDS, "actor_permutation_seed": P.SEED + 1, "folds": {}}
    for k in range(N_FOLDS):
        r = crema_roles(k)
        rec["folds"][str(k)] = {role: {"n": int(len(v)), "n_actors": int(len(np.unique(g[v]))), "actors": np.unique(g[v]).tolist(),
                                       "label_counts": np.bincount(y[v], minlength=6).tolist(), "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for role, v in r.items()}
    with open(OUTC / "roles_cv5.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: {role: (v["n"], v["n_actors"]) for role, v in f.items()} for k, f in rec["folds"].items()})); return 0


def _codec_one(stem):
    return stem, video_codec(MEDIA / "VideoFlash" / f"{stem}.flv")


def cmd_codecs(a):
    """Video codec of every clip (PyAV container read; no decoding) with its distribution by emotion and actor."""
    from multiprocessing import Pool
    stems, y, g = stems_and_labels()
    with Pool(a.workers) as pool:
        cod = dict(pool.map(_codec_one, stems, chunksize=16))
    c = np.array([cod[s] for s in stems]); res = {"counts": {k: int((c == k).sum()) for k in sorted(set(c))},
           "by_emotion": {e: {k: int(((c == k) & (y == i)).sum()) for k in sorted(set(c))} for i, e in enumerate(EMOTIONS)},
           "actors_with_mixed_codecs": int(sum(len(set(c[g == act])) > 1 for act in np.unique(g)))}
    json.dump({**res, "per_clip": cod}, open(OUTC / "codecs.json", "w")); print(json.dumps(res, indent=1)); return 0


def _prep_one(args):
    i, stem, decoder = args
    import torch
    torch.set_num_threads(1)
    wave, n_audio = read_wav(MEDIA / "AudioWAV" / f"{stem}.wav"); sel, n_frames = decode_selected(MEDIA / "VideoFlash" / f"{stem}.flv", decoder)
    return i, wave, crop_resize(sel), n_audio, n_frames, list(sel.shape[1:3])


def cmd_prep(a):
    """Decode every clip once into the cache: wave_i16.npy [N, 64000], frames_u8.npy [N, 8, 3, 112, 112], labels.npy, meta.json."""
    from multiprocessing import Pool
    stems, y, g = stems_and_labels(); n = len(stems) if a.limit is None else min(a.limit, len(stems)); CACHE.mkdir(parents=True, exist_ok=True)
    wave = np.lib.format.open_memmap(CACHE / "wave_i16.npy.part", "w+", np.int16, (n, N_SAMPLES))
    frames = np.lib.format.open_memmap(CACHE / "frames_u8.npy.part", "w+", np.uint8, (n, K_FRAMES, 3, SIDE, SIDE))
    n_audio, n_frames, sizes = np.zeros(n, np.int64), np.zeros(n, np.int64), {}; t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (i, w, f, na, nf, hw) in enumerate(pool.imap_unordered(_prep_one, [(i, stems[i], a.decoder) for i in range(n)], chunksize=8)):
            wave[i], frames[i], n_audio[i], n_frames[i] = w, f, na, nf; sizes[f"{hw[1]}x{hw[0]}"] = sizes.get(f"{hw[1]}x{hw[0]}", 0) + 1
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{n} clips, {time.time() - t0:.0f}s", flush=True)
    wave.flush(); frames.flush(); del wave, frames
    os.replace(CACHE / "wave_i16.npy.part", CACHE / "wave_i16.npy"); os.replace(CACHE / "frames_u8.npy.part", CACHE / "frames_u8.npy"); np.save(CACHE / "labels.npy", y[:n])
    meta = {"n": n, "stems_sha": __import__("hashlib").sha256("\n".join(stems[:n]).encode()).hexdigest(), "decoder": a.decoder, "native_sizes": sizes,
            "n_frames_percentiles": {str(q): float(np.percentile(n_frames, q)) for q in (0, 1, 50, 99, 100)},
            "audio_seconds_percentiles": {str(q): float(np.percentile(n_audio / SR, q)) for q in (0, 1, 50, 99, 100)},
            "frac_audio_cropped": float((n_audio > N_SAMPLES).mean()), "seconds": time.time() - t0, "n_audio": n_audio.tolist(), "n_frames": n_frames.tolist()}
    json.dump(meta, open(CACHE / "meta.json", "w")); print(json.dumps({k: v for k, v in meta.items() if k not in ("n_audio", "n_frames")}, indent=1)); return 0


def run_dir(a):
    return P.OUT / ("smoke" if a.smoke else "") / "cremad" / (f"{a.family}_s{a.seed}" + ("" if a.fold < 0 else f"_f{a.fold}"))


def cmd_run(a):
    import torch
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu"); t_start = time.time(); base = run_dir(a); base.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(CACHE / "meta.json")); stems, y_all, groups = stems_and_labels()
    if meta["decoder"] != a.decoder or meta["n"] != len(stems):
        raise SystemExit(f"cache decoder {meta['decoder']} / n {meta['n']} does not match --decoder {a.decoder} / {len(stems)} clips")
    log_f = open(base / "log.txt", "a")

    def log(msg):
        line = f"[cremad {a.family} s{a.seed} f{a.fold}] {msg}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    lim = dict(zip(("FIT_TASK", "FIT_POLICY", "CAL", "DEV"), map(int, a.smoke_lim.split(",")))); roles = crema_roles(a.fold, a.smoke, lim); T = Tensors(dev)
    for r1 in roles:
        for r2 in roles:
            if r1 < r2 and np.intersect1d(groups[roles[r1]], groups[roles[r2]]).size:
                raise SystemExit(f"actor overlap between {r1} and {r2}")
    log(f"device {dev}; roles " + ", ".join(f"{k} {len(v)} ({len(np.unique(groups[v]))} actors)" for k, v in roles.items()))
    model, hist, ep_sel = train_crema(T, roles, a.family, a.seed, a.smoke_epochs if a.smoke else a.epochs, dev, log, a.init)
    torch.save({"state": model.state_dict(), "epoch": ep_sel, "family": a.family, "seed": a.seed, "fold": a.fold, "init": a.init}, base / "system.pt")
    pred = {r: P.predict(model, T, roles[r], chunk=64) for r in ("FIT_POLICY", "CAL", "DEV")}; ys = {r: T.y[roles[r]].numpy() for r in pred}
    tg = {r: P.gain_targets(pred[r][1], pred[r][2], ys[r]) for r in pred}; X = {r: P.policy_features(pred[r][0], pred[r][1]) for r in pred}
    scores, pinfo, fns = {}, {}, {}
    for tname, ti in (("gain_z", 0), ("gain_h", 1)):
        fn, info = P.train_policy(X["FIT_POLICY"], tg["FIT_POLICY"][ti], X["CAL"], tg["CAL"][ti], a.seed, dev, epochs=20 if a.smoke else P.POLICY_EPOCHS)
        scores[tname] = {"CAL": fn(X["CAL"]), "DEV": fn(X["DEV"])}; pinfo[tname] = info; fns[tname] = fn
    for r in ("CAL", "DEV"):
        pa = pred[r][1]
        scores.setdefault("entropy", {})[r] = P.entropy(pa); scores.setdefault("maxprob", {})[r] = 1 - pa.max(1)
        scores.setdefault("oracle_h", {})[r] = tg[r][1]; scores.setdefault("oracle_z", {})[r] = tg[r][0]
    pa, pb, y = pred["DEV"][1], pred["DEV"][2], ys["DEV"]; z, h = tg["DEV"]
    exact = {k: P.topk_curve(scores[k]["DEV"], pa, pb, y, tie_seed=P.SEED + a.seed) for k in P.SCORED + P.ORACLES}
    exact["random"] = P.random_curve(len(y), pa, pb, y, a.seed, repeats=3 if a.smoke else P.RANDOM_REPEATS)
    thr = {k: P.budget_curve(scores[k]["CAL"], scores[k]["DEV"], pa, pb, y) for k in P.SCORED}
    ends = {"never": P.route(np.zeros(len(y), bool), pa, pb, y), "always": P.route(np.ones(len(y), bool), pa, pb, y)}
    import resource
    memory = {"cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None,
              "cuda_max_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20 if dev.type == "cuda" else None,
              "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}

    def strip(cv):
        return {b: {kk: vv for kk, vv in c.items() if kk not in ("per_sample_brier", "acquire")} for b, c in cv.items()}

    res = {"dataset": "cremad", "family": a.family, "seed": a.seed, "fold": a.fold, "init": a.init, "decoder": a.decoder, "smoke": a.smoke, "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "selected_epoch": ep_sel, "history": hist, "policy": pinfo,
           "dev": {"n": int(len(y)), "n_actors": int(len(np.unique(groups[roles["DEV"]]))), "brier_never": float(ends["never"][0].mean()), "brier_always": float(ends["always"][0].mean()),
                   "acc_never": float(ends["never"][1].mean()), "acc_always": float(ends["always"][1].mean()),
                   "mean_h": float(h.mean()), "frac_h_positive": float((h > 0).mean()), "mean_z": float(z.mean()), "mean_h_minus_z": float((h - z).mean()),
                   "identity_max_abs_err": float(np.abs((h - z) - 2 * ((pb - pa) * (np.eye(pa.shape[1])[y] - pb)).sum(1)).max()),
                   "curves_exact": {k: strip(cv) for k, cv in exact.items()}, "curves_cal_threshold": {k: strip(cv) for k, cv in thr.items()},
                   "gain_bins": {k: P.gain_bins(scores[k]["DEV"], h, z) for k in ("gain_z", "gain_h", "entropy")},
                   "corr_pred_h": {k: float(np.corrcoef(scores[k]["DEV"], h)[0, 1]) for k in P.SCORED}},
           "memory": memory}
    arrays = {"dev_index": roles["DEV"], "y": y, "pa": pa.astype(np.float32), "pb": pb.astype(np.float32), "h": h, "z": z,
              "brier_never": ends["never"][0].astype(np.float32), "brier_always": ends["always"][0].astype(np.float32)}
    arrays.update({f"brier_{k}_{b}": c["per_sample_brier"].astype(np.float32) for k, cv in exact.items() for b, c in cv.items()})
    arrays.update({f"acq_{k}_{b}": c["acquire"].astype(np.uint8) for k, cv in exact.items() if k != "random" for b, c in cv.items()})
    arrays.update({f"thr_brier_{k}_{b}": c["per_sample_brier"].astype(np.float32) for k, cv in thr.items() for b, c in cv.items()})
    arrays.update({f"thr_acq_{k}_{b}": c["acquire"].astype(np.uint8) for k, cv in thr.items() for b, c in cv.items()})
    if a.costs:                                                                       # per-DEV-clip end-to-end timing (bs 1)
        c, codecs = clip_costs(model, fns["gain_h"], stems, roles["DEV"], dev, a.decoder, repeats=1 if a.smoke else 3, log=log)
        arrays.update({f"cost_{k}": v.astype(np.float32) for k, v in c.items()}); arrays["codec"] = np.array(codecs)
        res["costs"] = {"cpu": cpu_name(), "gpu": res["gpu"], "decoder": a.decoder, "threads": 1, "batch_size": 1,
                        "mean_ms": {k: float(np.mean(v)) for k, v in c.items()},
                        "policy_cost_ms_exact": {pol: {b: policy_cost(c, pol, exact[pol][b]["acquire"]) for b in exact[pol]} for pol in P.SCORED},
                        "never_ms": policy_cost(c, "never", np.zeros(len(y))), "always_ms": policy_cost(c, "always", np.ones(len(y))),
                        "t_v_by_codec": {cd: {"n": int(np.sum(np.array(codecs) == cd)), "mean_t_v_decode_ms": float(c["t_v_decode"][np.array(codecs) == cd].mean())} for cd in sorted(set(codecs))}}
        log(f"costs (ms/clip, bs 1): never {res['costs']['never_ms']:.2f}, always {res['costs']['always_ms']:.2f}; t_V decode {res['costs']['mean_ms']['t_v_decode']:.2f}")
    res["seconds_total"] = time.time() - t_start
    np.savez_compressed(base / "per_sample.npz", **arrays); json.dump(res, open(base / "result.json", "w"), indent=1)
    log(f"DEV Brier never {res['dev']['brier_never']:.4f} always {res['dev']['brier_always']:.4f} (acc {res['dev']['acc_never']:.3f} / {res['dev']['acc_always']:.3f}); "
        f"mean h {res['dev']['mean_h']:+.4f}; exact 25 %: gain_h {exact['gain_h']['0.25']['brier']:.4f} gain_z {exact['gain_z']['0.25']['brier']:.4f} "
        f"entropy {exact['entropy']['0.25']['brier']:.4f} random {exact['random']['0.25']['brier']:.4f}; {res['seconds_total']:.0f}s; memory {memory}")
    return 0


def cpu_name():
    try:
        return next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        return "unknown"


def cmd_aggregate(a):
    """Per family over the seeds (systems pooled over folds when cross-fitted).  Primary risk: exact-rate DEV Brier averaged over the four
    budgets; interval: two-way bootstrap over seeds x DEV actors.  Primary cost: end-to-end ms / clip at batch size 1 from the per-clip
    table measured once per family (the run with --costs), applied to each system's own acquisitions."""
    root = P.OUT / ("smoke" if a.smoke else "") / "cremad"; stems, y_all, groups = stems_and_labels(); rng = np.random.default_rng(P.SEED)
    runs, incomplete = {}, []
    for d in sorted(root.glob("*_s*")):
        if (d / "result.json").exists():
            r = json.load(open(d / "result.json"))
            if (r["fold"] < 0) == (a.mode == "single"):                                # never mix the single split with cross-fitting
                runs.setdefault((r["family"], r["seed"]), []).append((r, dict(np.load(d / "per_sample.npz"))))
    if a.mode == "cv5":
        for key in list(runs):
            if sorted(t[0]["fold"] for t in runs[key]) != list(range(N_FOLDS)):
                incomplete.append(f"{key[0]}_s{key[1]}"); del runs[key]
    pols = ("random",) + P.SCORED + P.ORACLES; keys = ["auc"] + [str(b) for b in P.BUDGETS]
    res = {"mode": a.mode, "systems": sorted(f"{f}_s{s}" for f, s in runs), "incomplete_dropped": incomplete, "families": {}}
    for f in P.FAMILIES:
        ss = sorted(s for (ff, s) in runs if ff == f)
        if not ss:
            continue
        pooled = []
        for s in ss:                                                                  # pool the folds of one (family, seed): DEV = union
            parts = sorted(runs[(f, s)], key=lambda t: t[0]["fold"]); idx = np.concatenate([p[1]["dev_index"] for p in parts]); order = np.argsort(idx)
            keys_ps = [k for k in parts[0][1] if k.startswith(("brier_", "thr_brier_", "acq_", "thr_acq_", "cost_"))]
            pooled.append({"dev_index": idx[order], "folds": [p[0]["fold"] for p in parts], **{k: np.concatenate([p[1][k] for p in parts])[order] for k in keys_ps if all(k in p[1] for p in parts)}})
        assert all(np.array_equal(p["dev_index"], pooled[0]["dev_index"]) for p in pooled), "DEV clips differ across seeds"
        gi = np.unique(groups[pooled[0]["dev_index"]], return_inverse=True)[1]

        def per(pol, key, prefix="brier"):
            if key == "auc":
                return np.mean([per(pol, str(b), prefix) for b in P.BUDGETS], 0)
            return np.stack([p[f"{prefix}_{pol}_{key}"].astype(np.float64) for p in pooled])

        def three(D):
            return {"two_way": P.two_way_boot(D, gi, rng), "seed_t": P.tint(D.mean(1)), "cluster_cond": P.cluster_boot(D.mean(0), gi, rng)}

        ends = {e: np.stack([p[f"brier_{e}"].astype(np.float64) for p in pooled]) for e in ("never", "always")}
        fam = {"seeds": ss, "folds": pooled[0]["folds"], "n_dev": int(len(gi)), "n_dev_actors": int(gi.max()) + 1,
               "endpoints": {e: P.tint(ends[e].mean(1)) for e in ends}, "always_minus_never": three(ends["always"] - ends["never"]),
               "exact_rate_brier": {pol: {k: P.tint(per(pol, k).mean(1)) for k in keys} for pol in pols}, "contrasts": {}, "cost_ms_bs1": None}
        for name, (x, yv) in P.CONTRASTS.items():
            fam["contrasts"][name] = {k: three(per(x, k) - per(yv, k)) for k in keys}
        cost_src = [p for p in pooled if "cost_t_u" in p]                             # per-clip cost table (measured once per family)
        if cost_src:
            c = {k[5:]: cost_src[0][k].astype(np.float64) for k in cost_src[0] if k.startswith("cost_") and not k.startswith("cost_cold_")}
            cm = {"never": policy_cost(c, "never", 0.0), "always": policy_cost(c, "always", 1.0), "mean_component_ms": {k: float(v.mean()) for k, v in c.items()}}
            for pol in P.SCORED:
                cm[pol] = {str(b): P.tint([policy_cost(c, pol, p[f"acq_{pol}_{b}"]) for p in pooled]) for b in P.BUDGETS}
            cm["random"] = {str(b): policy_cost(c, "random", np.full(len(gi), np.mean([P.n_acquire(b, len(p["dev_index"])) for p in [pooled[0]]]) / len(gi))) for b in P.BUDGETS}
            fam["cost_ms_bs1"] = cm
        res["families"][f] = fam
    json.dump(res, open(root / "aggregate.json", "w"), indent=1)

    def ci(c):
        return f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]" if c.get("ci95") else "n/a"

    L = [f"# r6 GAIN-P1 CREMA-D ({a.mode}): DEV results (exact-rate routing primary; Brier, lower is better)", "",
         f"systems {res['systems']}; incomplete (dropped) {res['incomplete_dropped'] or 'none'}", ""]
    for f, fam in res["families"].items():
        L += [f"## family {f} (seeds {fam['seeds']}, folds {fam['folds']}; DEV n = {fam['n_dev']}, actors = {fam['n_dev_actors']})", "",
              f"- never {fam['endpoints']['never']['mean']:.4f}; always {fam['endpoints']['always']['mean']:.4f}; always − never {fam['always_minus_never']['two_way']['mean']:+.4f} "
              f"two-way {ci(fam['always_minus_never']['two_way'])}", ""]
        cm = fam["cost_ms_bs1"]
        L += ["| policy (exact rate) | mean over budgets (primary) | 10 % | 25 % | 50 % | 75 % |", "|---|---|---|---|---|---|"]
        for pol in pols:
            L.append(f"| {pol} | " + " | ".join(f"{fam['exact_rate_brier'][pol][k]['mean']:.4f}" for k in keys) + " |")
        L += ["", "| contrast | budget | mean | two-way 95 % (seeds x actors) | seed t 95 % | actors given systems 95 % |", "|---|---|---|---|---|---|"]
        for name, cc in fam["contrasts"].items():
            for k in keys:
                c = cc[k]; L.append(f"| {name} | {'mean (primary)' if k == 'auc' else str(int(100 * float(k))) + ' %'} | {c['two_way']['mean']:+.4f} | {ci(c['two_way'])} | {ci(c['seed_t'])} | {ci(c['cluster_cond'])} |")
        if cm:
            L += ["", f"Cost (end-to-end ms / clip, batch size 1; primary cost metric): never {cm['never']:.2f}, always {cm['always']:.2f}; components "
                  + ", ".join(f"{k} {v:.2f}" for k, v in cm["mean_component_ms"].items()), "", "| policy | " + " | ".join(f"{int(100 * b)} %" for b in P.BUDGETS) + " |", "|---|" + "---|" * len(P.BUDGETS)]
            for pol in P.SCORED:
                L.append(f"| {pol} | " + " | ".join(f"{cm[pol][str(b)]['mean']:.2f}" for b in P.BUDGETS) + " |")
            L.append("| random | " + " | ".join(f"{cm['random'][str(b)]:.2f}" for b in P.BUDGETS) + " |")
        L.append("")
    (root / "aggregate.md").write_text("\n".join(L)); print("\n".join(L)); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe"); sub.add_parser("cvroles"); p = sub.add_parser("codecs"); p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("prep"); p.add_argument("--workers", type=int, default=32); p.add_argument("--decoder", default="av", choices=["av", "cli"]); p.add_argument("--limit", type=int)
    p = sub.add_parser("run"); p.add_argument("--family", required=True, choices=P.FAMILIES); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fold", type=int, default=-1); p.add_argument("--epochs", type=int, default=EPOCHS); p.add_argument("--init", default="imagenet", choices=["imagenet", "scratch"])
    p.add_argument("--decoder", default="av", choices=["av", "cli"]); p.add_argument("--smoke", action="store_true"); p.add_argument("--cpu", action="store_true"); p.add_argument("--costs", action="store_true")
    p.add_argument("--smoke-lim", default="128,64,64,64", help="smoke sizes FIT_TASK,FIT_POLICY,CAL,DEV"); p.add_argument("--smoke-epochs", type=int, default=1)
    p = sub.add_parser("aggregate"); p.add_argument("--smoke", action="store_true"); p.add_argument("--mode", default="single", choices=["single", "cv5"])
    a = ap.parse_args()
    return {"probe": cmd_probe, "cvroles": cmd_cvroles, "codecs": cmd_codecs, "prep": cmd_prep, "run": cmd_run, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
