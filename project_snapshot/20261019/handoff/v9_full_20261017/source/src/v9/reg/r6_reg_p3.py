# PROVENANCE (V9-G2): verbatim copy of ssl_pilot/scripts/r6_reg_p3.py at main commit 2d171a6 (sha256 bf2b9f6b5048f256); read-only reference for the V9 REG extension on EPFL RGB-NIR. Do not edit; the original stays unmodified in ssl_pilot.
"""R6 REG-P3 — the REG-P2 rigid comparison on a second modality pair (reports/R6/R6_REG_P3_PROTOCOL_*_20261012.yaml).

The pilot runs on EPFL RGB-NIR Scene (nirscene1; Brown & Süsstrunk, CVPR 2011): visible RGB vs near-infrared photographs of 477 natural and urban
scenes in 9 categories, taken as two separate exposures and aligned by the authors.  The Zurich Summer path (aerial RGB vs NIR, QuickBird
PC-pansharpened) is kept for the record only: it is NOT run (coordinator, 2026-10-12): the pansharpened bands share the panchromatic detail, so in
the smokes MI, NMI and the kernel registered at 0.2–0.4 px medians, a ceiling for the class comparison.

REG-P1 / REG-P2 code is reused read-only by import (neither file is modified):
  P = scripts/r6_reg_pilot.py   strict TIFF reader, robust normalisation, encoders + critic, critic training (cmd_train: VCS / logistic / InfoNCE,
                                3 000 steps, 12 checkpoints), CoMIR training (cmd_comir: official DenseUNet, REG-P1's train-biodata recipe), the
                                closed-form kernel fit, J
  Q = scripts/r6_reg_p2.py      rigid geometry (T_{θ,t}, V ∘ T*⁻¹), the rigid perturbation family (θ* U(±30°) + REG-P1's stratified translations),
                                every energy (MI / NMI, kernel, critic warp-then-encode, CoMIR encode-once), the shared rigid optimiser, the
                                registration loop, CoMIR's mean corner error, the lexicographic CAL selection, the run summaries
EPFL specifics (this file; rules fixed by the coordinator on 2026-10-12, before any EPFL result):
  data      nirscene1/<category>/<id>_rgb.tiff and <id>_nir.tiff (8-bit; RGB 3 channels, NIR 1 channel; 1024 × 620–759 or 501–1023 × 768 px).
            No licence text; the owner accepted the source for internal research (P60) — publication needs the owner's OK.  No official split
  checks    alignment_check(): the audit's base-alignment check — gradient magnitudes of RGB luminance and NIR at half resolution, the integer
            shift in [−6, 6]² that maximises their correlation over the interior (8-px margin).  Pairs whose best shift has
            max(|dy|, |dx|) >= 2 half-resolution px are EXCLUDED and listed (33 of 477).  Pairs narrower than a 512-px window are excluded and listed
            (24 portrait images).  near_duplicate_links(): within-category pairs whose standardised luminance thumbnails (÷ 32) correlate > 0.9
            (13 links) are merged into one unit (transitively, also through excluded pairs)
  units     the scene (an image pair), near-duplicates merged -> 411 units over the 420 kept pairs
  roles     per category, units ordered by sha256("R6-REG-P3|EPFL|unit|" + unit): 8 DEV, 1 CAL, the rest FIT (72 DEV / 9 CAL / 330 FIT units)
  windows   one 512 × 512 native-resolution window centred in each kept image (every role): REG-P2's working-image size, so its support (disk
            r = 129), search set (±30°, |t| <= 51 px, P2's translation strata), corner patch (±104.125 px about the centre) and thresholds carry over
  inputs    reference U = RGB, moving V = NIR.  Learned methods (critics, CoMIR): RGB (3 channels) and linear NIR, each robust-normalised (median /
            IQR) per window (REG-P1's normalisation).  MI / NMI / kernel: RGB luminance and NIR intensity, robust-normalised.  The cache keeps
            REG-P1's key names so its training code runs unchanged: bf = luminance, bf_rgb_n = RGB, shg = shg_log_n = linear NIR (no log),
            nir = raw NIR / 255 (the source of every moving image V ∘ T*⁻¹)
  error     err_full in every result = REG-P2's corner error, i.e. in REG-P2-equivalent px = 4 × native px; success < 42 / < 9 there is < 10.5 / < 2.25
            native px (REG-P2's working-resolution thresholds: 5 % / 1.08 % of the 208.25-px corner patch)
Zurich specifics (not run): 20 tiles, B, G, R, NIR uint16; units = tiles merged when closer than 50 m (17); roles by hash (8 DEV, CAL >= 8
windows, rest FIT); CAL / DEV windows = the non-overlapping 512 grid, FIT = a covering grid; NIR = DN / 2¹².

Subcommands (outputs under --root, default outputs/R6_REG_P3; --smoke uses <root>/smoke and a subset in which no DEV window is read):
  prepare   [--dataset epfl | zurich] checks -> units -> roles -> windows; per-window cache + manifest (hashes, checks, exclusions, units, roles)
  perturb   10 rigid perturbations per CAL / DEV window: REG-P2's family seeded by the window id (Q.perturbations_p2)
  train     REG-P1's critic training on the FIT windows (--objective may be a comma list, --seed too: run sequentially)
  comir     REG-P1's CoMIR training on the FIT windows (--seed may be a comma list)
  register  REG-P2's registration for one method (--method / --seed may be comma lists): CAL selection (critics: 12 checkpoints, kernel:
            {0.5, 1, 2} × median heuristic; lexicographic on the unit-pooled CAL records), DEV evaluation, profiles (the first DEV window of 5
            categories; of 5 units when the manifest has no categories)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6_reg_pilot as P  # noqa: E402  (REG-P1, frozen; read-only)
import r6_reg_p2 as Q  # noqa: E402  (REG-P2, frozen; read-only)

DATA3 = Path("/projects/EEG-foundation-model/yinghao/data/zurich_summer_5914759/Zurich_dataset_v1.0")      # Zurich (not run)
EPFL3 = Path("/projects/EEG-foundation-model/yinghao/data/epfl_rgb_nir/nirscene1")                         # EPFL RGB-NIR (the pilot)
DATA_DEFAULT = {"epfl": EPFL3, "zurich": DATA3}
ROOT3 = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3")
WIN, N_TILES, DN = P.SIZE, 20, 4096.0                                        # 512-px windows; Zurich: CoMIR's scaling img / 2**12
BANDS = ("B", "G", "R", "NIR")                                                # Zurich file band order (QuickBird bands 1–4)
NEAR_M, N_DEV_UNITS, MIN_CAL_WIN = 50.0, 8, 8                                 # Zurich units / roles
ALIGN_SEARCH, ALIGN_MARGIN, ALIGN_EXCLUDE, THUMB_DOWN, DUP_THR = 6, 8, 2, 16, 0.9   # EPFL checks (half-resolution px)
EPFL_DEV_PER_CAT, EPFL_CAL_PER_CAT = 8, 1
OBJECTIVES = ("vcs", "logistic", "infonce"); METHODS = OBJECTIVES + ("mi", "nmi", "kernel", "comir", "comir_wte")
ERR_UNIT = "err_full = CoMIR's mean corner error in REG-P2-equivalent px = 4 x native px (thresholds 42 / 9 = 10.5 / 2.25 native px)"


def nat(k: str):
    return (int("".join(ch for ch in k if ch.isdigit()) or 0), k)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cache_arrays(rgb: np.ndarray, nir: np.ndarray) -> dict:
    """Cache arrays of one 512 window from RGB [S, S, 3] (R, G, B order) and NIR [S, S], both linear in [0, 1] (keys as REG-P1's cache)."""
    if rgb.shape != (WIN, WIN, 3) or nir.shape != (WIN, WIN):
        raise ValueError(f"window RGB {rgb.shape} NIR {nir.shape}, expected {WIN}²")
    rgb = rgb.astype(np.float64); nir = nir.astype(np.float64); lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return {"bf": P.robust_norm(lum), "shg": P.robust_norm(nir), "bf_rgb_n": P.robust_norm(rgb), "shg_log_n": P.robust_norm(nir), "nir": nir.astype(np.float32)}


def smoke_subset(roles: dict, wins: dict) -> dict:
    """Smoke roles: FIT = the first 4 FIT windows; CAL = the first 2 CAL windows; DEV = the next 2 CAL windows relabelled, so no DEV window is
    read before the protocol is frozen (window ids sorted)."""
    fit = sorted(w for w, r in roles.items() if r == "FIT"); cal = sorted(w for w, r in roles.items() if r == "CAL")
    if not fit or len(cal) < 4:
        raise ValueError("smoke subset needs FIT windows and >= 4 CAL windows")
    return {**{w: "FIT" for w in fit[:4]}, **{w: "CAL" for w in cal[:2]}, **{w: "DEV" for w in cal[2:4]}}


def write_cache(a, sel: dict, crop) -> None:
    cache = a.root / "cache"; cache.mkdir(parents=True, exist_ok=True)
    for wid in sorted(sel):
        np.savez_compressed(cache / f"{wid}.npz", **cache_arrays(*crop(wid)))


# ------------------------------------------------------------------------------------------------------------ EPFL RGB-NIR: checks, units, roles
def epfl_pairs(data: Path = EPFL3) -> dict:
    """'<category>/<id>' -> (RGB path, NIR path) for every <id>_rgb.tiff (sorted categories, sorted files); a missing NIR partner raises."""
    out = {}
    for cat in sorted(d.name for d in Path(data).iterdir() if d.is_dir()):
        for p in sorted((Path(data) / cat).glob("*_rgb.tiff")):
            sid = p.name[: -len("_rgb.tiff")]; q = p.with_name(f"{sid}_nir.tiff")
            if not q.exists():
                raise FileNotFoundError(q)
            out[f"{cat}/{sid}"] = (p, q)
    return out


def lum(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def grad_mag(m: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(m); return np.hypot(gy, gx)


def zscore(c: np.ndarray) -> np.ndarray:
    return (c - c.mean()) / (c.std() + 1e-9)


def alignment_check(rgb: np.ndarray, nir: np.ndarray) -> dict:
    """The base-alignment check of the REG-P3 data audit (fixed before any EPFL result).  Gradient magnitudes of RGB luminance and of NIR at half
    resolution (2 × 2 area mean); for every integer shift (dy, dx) in [−6, 6]² the correlation of the z-scored interiors (8-px margin); the first
    maximum in scan order (dy outer, dx inner).  Also the z-scored luminance thumbnail (half resolution ÷ 16) used by the near-duplicate check."""
    L = P.area_down(lum(rgb), 2); N = P.area_down(nir.astype(np.float64), 2); gL, gN = grad_mag(L), grad_mag(N); h, w = gL.shape; m = ALIGN_MARGIN
    cL = zscore(gL[m:h - m, m:w - m]); best = None
    for dy in range(-ALIGN_SEARCH, ALIGN_SEARCH + 1):
        for dx in range(-ALIGN_SEARCH, ALIGN_SEARCH + 1):
            v = float((cL * zscore(gN[m + dy:h - m + dy, m + dx:w - m + dx])).mean())
            if best is None or v > best[0]:
                best = (v, dy, dx)
    return {"best_shift_half_res": [best[1], best[2]], "gradcorr_best": best[0], "gradcorr_zero": float((cL * zscore(gN[m:h - m, m:w - m])).mean()),
            "thumb": zscore(P.area_down(L, THUMB_DOWN)).ravel()}


def misaligned(chk: dict) -> bool:
    """Exclusion rule (coordinator, 2026-10-12): the check peaks >= 2 half-resolution px (>= 4 native px) from zero shift."""
    return max(abs(chk["best_shift_half_res"][0]), abs(chk["best_shift_half_res"][1])) >= ALIGN_EXCLUDE


def near_duplicate_links(thumbs: dict) -> list:
    """Within-category scene pairs whose thumbnails have the same shape and correlate > 0.9 (mean product of z-scores), in scan order."""
    by = {}
    for k in thumbs:
        by.setdefault(k.split("/")[0], []).append(k)
    out = []
    for ks in by.values():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = thumbs[ks[i]], thumbs[ks[j]]
                if a.shape == b.shape:
                    c = float(np.mean(a * b))
                    if c > DUP_THR:
                        out.append([ks[i], ks[j], c])
    return out


def units_epfl(kept: list, links: list) -> dict:
    """kept pair -> unit: near-duplicate links merged transitively (also through excluded pairs, so linked kept scenes stay together); a unit is
    named by its kept pairs joined with '+'."""
    parent = {}

    def find(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            k = parent[k]
        return k
    for k in kept:
        find(k)
    for a_, b_, _ in links:
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for k in sorted(kept):
        groups.setdefault(find(k), []).append(k)
    return {k: "+".join(g) for g in groups.values() for k in g}


def roles_epfl(unit_cat: dict) -> dict:
    """unit -> role, stratified by category: within each category units ordered by sha256("R6-REG-P3|EPFL|unit|" + unit); the first 8 DEV, the
    next 1 CAL, the rest FIT."""
    r = {}
    for cat in sorted(set(unit_cat.values())):
        us = sorted((u for u, c in unit_cat.items() if c == cat), key=lambda u: hashlib.sha256(f"R6-REG-P3|EPFL|unit|{u}".encode()).hexdigest())
        for i, u in enumerate(us):
            r[u] = "DEV" if i < EPFL_DEV_PER_CAT else ("CAL" if i < EPFL_DEV_PER_CAT + EPFL_CAL_PER_CAT else "FIT")
    return r


def centred_window(H: int, W: int) -> tuple:
    if min(H, W) < WIN:
        raise ValueError(f"image {H} × {W} smaller than a {WIN}-px window")
    return (H - WIN) // 2, (W - WIN) // 2


def wid_of(pair: str) -> str:
    return pair.replace("/", "_")


def prepare_epfl(a):
    pairs = epfl_pairs(a.data); info, chk, crops = {}, {}, {}
    for k, (pr, pn) in pairs.items():
        rgb = P.read_tiff(pr); nir = P.read_tiff(pn)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or nir.ndim != 2 or rgb.shape[:2] != nir.shape or rgb.dtype != np.uint8 or nir.dtype != np.uint8:
            raise ValueError(f"{k}: RGB {rgb.shape} {rgb.dtype}, NIR {nir.shape} {nir.dtype}")
        c = alignment_check(rgb, nir); chk[k] = c; H, W = nir.shape
        info[k] = {"shape": [H, W], "sha256_rgb": sha(pr), "sha256_nir": sha(pn), "best_shift_half_res": c["best_shift_half_res"],
                   "gradcorr_best": c["gradcorr_best"], "gradcorr_zero": c["gradcorr_zero"]}
        if misaligned(c):
            info[k]["status"] = "excluded_alignment"
        elif min(H, W) < WIN:
            info[k]["status"] = "excluded_size"
        else:
            y0, x0 = centred_window(H, W); info[k].update({"status": "kept", "y0": y0, "x0": x0})
            crops[k] = (rgb[y0:y0 + WIN, x0:x0 + WIN].copy(), nir[y0:y0 + WIN, x0:x0 + WIN].copy())
    links = near_duplicate_links({k: c["thumb"] for k, c in chk.items()}); kept = sorted(crops)
    unit = units_epfl(kept, links); unit_cat = {u: u.split("/")[0] for u in set(unit.values())}; urole = roles_epfl(unit_cat); roles, wins = {}, {}
    for k in kept:
        w = wid_of(k); roles[w] = urole[unit[k]]; wins[w] = {"pair": k, "unit": unit[k], "category": k.split("/")[0], "y0": info[k]["y0"], "x0": info[k]["x0"]}
        info[k].update({"unit": unit[k], "role": urole[unit[k]]})
    sel = smoke_subset(roles, wins) if a.smoke else roles
    write_cache(a, sel, lambda w: (crops[wins[w]["pair"]][0] / 255.0, crops[wins[w]["pair"]][1] / 255.0))
    units = {u: {"role": r, "category": unit_cat[u], "pairs": u.split("+")} for u, r in urole.items()}
    man = {"dataset": "EPFL RGB-NIR Scene (nirscene1; Brown & Süsstrunk, CVPR 2011): RGB reference, NIR moving", "roles": dict(sel), "windows": {w: wins[w] for w in sel},
           "pairs": info, "units": units, "near_duplicate_links": links, "excluded_alignment": sorted(k for k, v in info.items() if v["status"] == "excluded_alignment"),
           "excluded_size": sorted(k for k, v in info.items() if v["status"] == "excluded_size"),
           "check_rule": f"alignment_check: best integer shift in [-{ALIGN_SEARCH}, {ALIGN_SEARCH}]^2 at half resolution; excluded if max(|dy|, |dx|) >= {ALIGN_EXCLUDE}",
           "unit_rule": f"scene; near-duplicate links (thumbnail correlation > {DUP_THR}, same category and thumbnail shape) merged transitively",
           "role_rule": f"per category, units by sha256('R6-REG-P3|EPFL|unit|'+unit): {EPFL_DEV_PER_CAT} DEV, {EPFL_CAL_PER_CAT} CAL, rest FIT",
           "window_rule": "one 512 window centred in each kept image", "size": WIN, "down": 1, "dn_scale": 255.0, "gsd_m": None, "official_split": None,
           "smoke": bool(a.smoke), "cache_keys": "bf = RGB luminance, bf_rgb_n = RGB, shg = shg_log_n = linear NIR (no log), nir = raw NIR / 255; all robust-normalised per window except nir"}
    (a.root / ("manifest_smoke.json" if a.smoke else "manifest.json")).write_text(json.dumps(man, indent=1))
    print(f"[prepare epfl] {len(pairs)} pairs; excluded: alignment {len(man['excluded_alignment'])}, size {len(man['excluded_size'])}; kept {len(kept)}; "
          f"near-duplicate links {len(links)}; units {len(urole)}", flush=True)
    for r in ("FIT", "CAL", "DEV"):
        ws = sorted(w for w, v in sel.items() if v == r)
        print(f"[prepare epfl] {r}: {sum(v == r for v in urole.values())} units, {sum(v == r for v in roles.values())} windows" + (f"; smoke {r} windows {ws}" if a.smoke else ""), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------ Zurich (not run): units, roles, windows
def tile_files(data: Path = DATA3) -> dict:
    """zh1 … zh20 under images_tif/ (the label rasters in groundtruth/ are never read)."""
    out = {f"zh{i}": data / "images_tif" / f"zh{i}.tif" for i in range(1, N_TILES + 1)}; miss = [k for k, p in out.items() if not p.exists()]
    if miss:
        raise FileNotFoundError(f"missing tiles {miss} under {data / 'images_tif'}")
    return out


def geo_tags(path: Path) -> dict:
    """ModelPixelScale (33550) and ModelTiepoint (33922) of a little-endian classic GeoTIFF (DOUBLE arrays, always stored at an offset)."""
    b = Path(path).read_bytes()
    if b[:4] != b"II*\x00":
        raise ValueError(f"{Path(path).name}: not a little-endian classic TIFF")
    off = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, off)[0]; out = {}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack_from("<HHII", b, off + 2 + 12 * i)
        if tag in (33550, 33922) and typ == 12:
            out[tag] = list(struct.unpack_from("<" + "d" * cnt, b, val))
    if 33550 not in out or 33922 not in out:
        raise ValueError(f"{Path(path).name}: no ModelPixelScale / ModelTiepoint")
    return out


def footprint(path: Path, shape) -> tuple:
    """(x0, x1, y0, y1) in map metres: raster (I, J) at the tie point maps to model (X, Y); x grows with columns, y decreases with rows."""
    t = geo_tags(path); sx, sy = t[33550][:2]; I, J, _, X, Y, _ = t[33922][:6]; H, W = shape[:2]; x0, y1 = X - I * sx, Y + J * sy
    return (x0, x0 + W * sx, y1 - H * sy, y1)


def gap_m(a, b) -> float:
    """Distance between two footprints (0 when they touch or overlap)."""
    dx = max(a[0], b[0]) - min(a[1], b[1]); dy = max(a[2], b[2]) - min(a[3], b[3]); return math.hypot(max(dx, 0.0), max(dy, 0.0))


def units_of(foot: dict, near: float = NEAR_M) -> dict:
    """tile -> unit: tiles closer than `near` metres (or overlapping) are merged transitively; a unit is named by its tiles joined with '+'."""
    ks = sorted(foot, key=nat); parent = {k: k for k in ks}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if gap_m(foot[a], foot[b]) < near:
                parent[find(b)] = find(a)
    groups = {}
    for k in ks:
        groups.setdefault(find(k), []).append(k)
    return {k: "+".join(sorted(g, key=nat)) for g in groups.values() for k in g}


def roles_p3(unit_windows: dict) -> dict:
    """Zurich: unit -> role.  Units ordered by sha256("R6-REG-P3|unit|" + unit): the first N_DEV_UNITS are DEV, then CAL until CAL holds
    >= MIN_CAL_WIN non-overlapping windows, the rest FIT.  unit_windows: unit -> number of non-overlapping 512 windows (a data fact, no result)."""
    order = sorted(unit_windows, key=lambda u: hashlib.sha256(f"R6-REG-P3|unit|{u}".encode()).hexdigest()); r = {}; n_dev = n_cal = 0
    for u in order:
        if n_dev < N_DEV_UNITS:
            r[u] = "DEV"; n_dev += 1
        elif n_cal < MIN_CAL_WIN:
            r[u] = "CAL"; n_cal += unit_windows[u]
        else:
            r[u] = "FIT"
    return r


def windows_of(H: int, W: int, covering: bool = False) -> list:
    """512-px window origins (y0, x0) in an H × W tile.  Default (CAL / DEV): the non-overlapping grid centred in the tile.  covering (FIT):
    ceil(L / 512) evenly spaced origins from 0 to L − 512 per axis (overlapping), so every pixel lies in at least one window."""
    if min(H, W) < WIN:
        raise ValueError(f"tile {H} × {W} smaller than a {WIN}-px window")

    def axis(L):
        if covering:
            n = math.ceil(L / WIN); return [round(k * (L - WIN) / (n - 1)) for k in range(n)] if n > 1 else [0]
        n = L // WIN; m = (L - n * WIN) // 2; return [m + k * WIN for k in range(n)]
    return [(y, x) for y in axis(H) for x in axis(W)]


def window_inputs(tile: np.ndarray, y0: int, x0: int) -> dict:
    """Zurich: cache arrays of one 512 window of a B, G, R, NIR uint16 tile (DN / 2¹²)."""
    a = tile[y0:y0 + WIN, x0:x0 + WIN].astype(np.float64) / DN
    if a.shape != (WIN, WIN, 4):
        raise ValueError(f"window at ({y0}, {x0}) has shape {a.shape}")
    return cache_arrays(a[..., [2, 1, 0]], a[..., 3])


def prepare_zurich(a):
    files = tile_files(a.data); tiles, foot, info = {}, {}, {}
    for t, p in files.items():
        arr = P.read_tiff(p)
        if arr.ndim != 3 or arr.shape[2] != 4 or arr.dtype != np.dtype("<u2"):
            raise ValueError(f"{t}: {arr.shape} {arr.dtype}, expected H × W × 4 uint16")
        tiles[t], foot[t] = arr, footprint(p, arr.shape)
        g = geo_tags(p); info[t] = {"shape": list(arr.shape), "sha256": sha(p), "footprint_m": list(foot[t]), "pixel_scale_m": g[33550][:2],
                                    "max_dn": arr.reshape(-1, 4).max(0).tolist()}
    unit = units_of(foot); uw = {}
    for t, arr in tiles.items():
        uw[unit[t]] = uw.get(unit[t], 0) + len(windows_of(*arr.shape[:2]))
    urole = roles_p3(uw); roles, wins = {}, {}
    for t in sorted(tiles, key=nat):
        role = urole[unit[t]]; origins = windows_of(*tiles[t].shape[:2], covering=role == "FIT")
        info[t].update({"unit": unit[t], "role": role, "n_windows": len(origins)})
        for k, (y0, x0) in enumerate(origins):
            wid = f"{t}_w{k:02d}"; roles[wid] = role; wins[wid] = {"tile": t, "unit": unit[t], "y0": int(y0), "x0": int(x0)}
    sel = smoke_subset(roles, wins) if a.smoke else roles
    cache = a.root / "cache"; cache.mkdir(parents=True, exist_ok=True)
    for wid in sorted(sel):
        w = wins[wid]; np.savez_compressed(cache / f"{wid}.npz", **window_inputs(tiles[w["tile"]], w["y0"], w["x0"]))
    ks = sorted(foot, key=nat); gaps = sorted(((x, y, round(gap_m(foot[x], foot[y]), 1)) for i, x in enumerate(ks) for y in ks[i + 1:] if gap_m(foot[x], foot[y]) < 200.0), key=lambda z: z[2])
    units = {u: {"role": r, "tiles": u.split("+"), "n_windows_nonoverlapping": uw[u]} for u, r in urole.items()}
    man = {"dataset": "Zurich Summer v1.0 (Zenodo 5914759), images_tif only: RGB reference, NIR moving", "roles": dict(sel), "windows": {w: wins[w] for w in sel}, "tiles": info,
           "units": units, "unit_rule": f"tiles closer than {NEAR_M:g} m merged (transitively)", "role_rule": f"units by sha256('R6-REG-P3|unit|'+unit): "
           f"first {N_DEV_UNITS} DEV, then CAL until >= {MIN_CAL_WIN} windows, rest FIT", "window_rule": "CAL / DEV: non-overlapping 512 grid centred; "
           "FIT: covering grid", "tile_gaps_lt_200m": gaps, "size": WIN, "down": 1, "dn_scale": DN, "bands": BANDS, "official_split": None,
           "gsd_m": float(np.mean([v["pixel_scale_m"][0] for v in info.values()])), "smoke": bool(a.smoke),
           "cache_keys": "bf = RGB luminance, bf_rgb_n = RGB, shg = shg_log_n = linear NIR (no log), nir = raw NIR / 4096; all robust-normalised per window except nir"}
    (a.root / ("manifest_smoke.json" if a.smoke else "manifest.json")).write_text(json.dumps(man, indent=1))
    for r in ("FIT", "CAL", "DEV"):
        us = sorted((u for u, x in urole.items() if x == r), key=nat); ws = sorted(w for w, v in sel.items() if v == r)
        print(f"[prepare zurich] {r}: {len(us)} units {us}; {sum(v == r for v in roles.values())} windows" + (f"; smoke {r} windows {ws}" if a.smoke else ""), flush=True)
    return 0


def cmd_prepare(a):
    return prepare_epfl(a) if a.dataset == "epfl" else prepare_zurich(a)


# ------------------------------------------------------------------------------------------------------------ perturbations
def cmd_perturb(a):
    """REG-P2's rigid family per CAL / DEV window (θ* U(±30°) seeded by 'R6-REG-P2|' + window, REG-P1's stratified translations seeded by
    'R6-REG-P1|' + window)."""
    m = P.load_manifest(a.root, a.smoke)
    pert = {c: Q.perturbations_p2(c) for c, r in sorted(m["roles"].items()) if r in ("CAL", "DEV") and (a.root / "cache" / f"{c}.npz").exists()}
    (a.root / ("perturbations_smoke.json" if a.smoke else "perturbations.json")).write_text(json.dumps(pert, indent=1)); print(f"[perturb] {len(pert)} windows × {P.N_PERT}")
    return 0


# ------------------------------------------------------------------------------------------------------------ training (REG-P1's code, unchanged)
def cmd_train(a):
    for s in a.seeds:
        for o in a.objectives:
            b = argparse.Namespace(**vars(a)); b.objective, b.seed = o, s; P.cmd_train(b)
    return 0


def cmd_comir(a):
    for s in a.seeds:
        b = argparse.Namespace(**vars(a)); b.seed = s; P.cmd_comir(b)
    return 0


# ------------------------------------------------------------------------------------------------------------ registration (REG-P2's machinery)
class Inputs3(Q.Inputs):
    """REG-P2's Inputs on the P3 cache: reference() unchanged (luminance [1, S, S], RGB [3, S, S]); moving() perturbs the raw NIR window
    (V ∘ T*⁻¹, bilinear, reflection outside the window — REG-P2's perturb_map) and robust-normalises the image as given; NIR is linear on both
    paths, so the intensity and learned paths coincide."""

    def moving(self, c, i):
        if (c, i) not in self.mov:
            torch = self.torch; nir = P.load_img(self.root, c, ("nir",))[0]; th, ty, tx = self.pert[c][i]
            nirp = Q.perturb_map(torch, self.F, torch.tensor(nir, dtype=torch.float32, device=self.dev)[None], th, (ty, tx))[0].cpu().numpy().astype(np.float64)
            x = torch.tensor(P.robust_norm(nirp), device=self.dev)[None]; self.mov[(c, i)] = (x, x)
        return self.mov[(c, i)]


def pool_units(recs: dict, unit_of: dict) -> dict:
    """Window-keyed registration records -> unit-keyed (every registration of a unit's windows); '#profile' entries dropped."""
    out = {}
    for k in sorted(recs):
        if "#" not in k:
            out.setdefault(unit_of[k], []).extend(recs[k])
    return out


def profile_windows(m: dict, dv: list, n: int = 5) -> list:
    """The DEV windows whose energy profiles are recorded: the first window (sorted ids) of each of the first n categories (EPFL) — of n units
    when the manifest's windows carry no category."""
    grp = lambda c: m["windows"][c].get("category") or m["windows"][c]["unit"]; seen, out = set(), []
    for c in sorted(dv):
        if grp(c) not in seen:
            seen.add(grp(c)); out.append(c)
    return out[:n]


def register_one(a, torch, nn, F, dev):
    m = P.load_manifest(a.root, a.smoke); unit_of = {w: v["unit"] for w, v in m["windows"].items()}
    pert = json.loads((a.root / ("perturbations_smoke.json" if a.smoke else "perturbations.json")).read_text())
    cal = sorted(c for c in pert if m["roles"][c] == "CAL"); dv = sorted(c for c in pert if m["roles"][c] == "DEV")
    fit = sorted(c for c, r in m["roles"].items() if r == "FIT" and (a.root / "cache" / f"{c}.npz").exists())
    prof = profile_windows(m, dv)
    I = Inputs3(torch, F, a.root, pert, dev); out = a.root / "results"; out.mkdir(parents=True, exist_ok=True)
    reg = lambda cores, maps, kind, kw=None, profile=(): Q.register_images(torch, F, cores, pert, maps, kind, dev, kw, a.n_pert, profile)
    key = lambda recs, order: Q.sel_key(pool_units(recs, unit_of), order)        # REG-P2's lexicographic rule on the unit-pooled CAL records
    res = {"method": a.method, "seed": a.seed, "dataset": m.get("dataset"), "root": str(a.root), "err_unit": ERR_UNIT,
           "selection_rule": "lexicographic on unit-pooled CAL: success<42, success<9, lower median corner error, earliest checkpoint / smaller multiplier",
           "r_omega": Q.R_OMEGA, "rho": Q.RHO, "box_exact": Q.BOX1, "box_grid": Q.BOX2, "n_pos": P.N_POS,
           "grid": {"theta_step_deg": Q.TH_STEP, "theta_max_deg": Q.THMAX, "t_step": P.GRID_STEP, "t_max": Q.TMAX}, "starts": P.N_STARTS, "adam": [P.N_ADAM, P.ADAM_LR],
           "n_pert": a.n_pert, "corner_half_px512": Q.HALF834, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}; t0 = time.time()
    if a.method in ("mi", "nmi"):
        def maps(c, i):
            U, V = I.reference(c)[0], I.moving(c, i)[0]; rg = lambda M: tuple(float(x) for x in np.percentile(M.cpu().numpy(), [0.5, 99.5]))
            return U, V, {"ranges": (rg(U), rg(V))}
        res["cal"] = reg(cal, maps, a.method); res["dev"] = reg(dv, maps, a.method, profile=prof)
    elif a.method == "kernel":
        sel = []; maps = lambda c, i: (I.reference(c)[0], I.moving(c, i)[0], {})
        for mult in ((1.0,) if a.smoke else (0.5, 1.0, 2.0)):
            kw = P.fit_kernel(torch, F, a.root, fit, mult, a.seed, dev); kw_e = {k: kw[k] for k in ("W", "b", "w")}
            r = reg(cal, maps, "kernel", kw_e); sel.append((key(r, mult), mult, kw_e, r, kw["h"]))
        best = max(sel, key=lambda z: z[0]); res["selection"] = [{"mult": s[1], "h": s[4], **Q.sel_row(pool_units(s[3], unit_of))} for s in sel]
        res["chosen_mult"] = best[1]; res["cal"] = best[3]; res["dev"] = reg(dv, maps, "kernel", best[2], prof)
    elif a.method in ("comir", "comir_wte"):
        run = a.root / "runs" / f"comir_s{a.seed}"; st = torch.load(run / "comir.pt", map_location=dev); mA, mB, _ = P.comir_models(torch, a.seed, dev)
        mA.load_state_dict(st["A"]); mB.load_state_dict(st["B"]); mA.eval(); mB.eval(); LU = {}; res["comir_run"] = json.loads((run / "comir.json").read_text())
        for p in list(mA.parameters()) + list(mB.parameters()):
            p.requires_grad_(False)
        for mod in mA.modules():                                                 # no activation checkpointing in the refinement (same values)
            if hasattr(mod, "efficient"):
                mod.efficient = False

        def maps(c, i):                                                          # A = NIR (moving), B = RGB (reference), as REG-P1 / P2's SHG / BF
            with torch.no_grad():
                if c not in LU:
                    LU[c] = mB(I.reference(c)[1][None])[0]
                if a.method == "comir":                                          # encode-once: the moving image as given
                    return LU[c], mA(I.moving(c, i)[1][None])[0], {}
            return LU[c], I.moving(c, i)[1], {}                                  # warp-then-encode (REG-P2's sensitivity row; not run in REG-P3)
        kw = {"mA": mA} if a.method == "comir_wte" else None
        res["cal"] = reg(cal, maps, a.method, kw) if a.method == "comir" else {}
        res["dev"] = reg(dv, maps, a.method, kw, prof)
    else:                                                                        # learned critic: vcs | logistic | infonce — this pilot's FIT runs
        run = a.root / "runs" / f"{a.method}_s{a.seed}"; ckpts = sorted(run.glob("ckpt_*.pt")); res["train"] = json.loads((run / "train.json").read_text())
        if not a.smoke and len(ckpts) != 12:
            raise RuntimeError(f"{run}: {len(ckpts)} checkpoints, REG-P1's recipe saves 12")
        eU, eV, cr = (x.to(dev).eval() for x in P.build_models(nn)); DU = {}; sel = []

        def load(ck):
            st = torch.load(ck, map_location=dev); eU.load_state_dict(st["eU"]); eV.load_state_dict(st["eV"]); cr.load_state_dict(st["critic"]); DU.clear()
            for p in list(eU.parameters()) + list(eV.parameters()) + list(cr.parameters()):
                p.requires_grad_(False)

        def maps(c, i):
            if c not in DU:
                with torch.no_grad():
                    DU[c] = eU(I.reference(c)[1][None])[0]
            return DU[c], I.moving(c, i)[1], {}
        for ck in ckpts:
            load(ck); r = reg(cal, maps, "critic", {"eV": eV, "critic": cr}); sel.append((key(r, int(ck.stem.split("_")[1])), ck, r))
            print(f"[select {a.method} s{a.seed}] {ck.name} CAL success<42 {sel[-1][0][0]:.3f} <9 {sel[-1][0][1]:.3f} median {-sel[-1][0][2] / P.DOWN:.2f} native px ({time.time() - t0:.0f}s)", flush=True)
        best = max(sel, key=lambda z: z[0]); res["chosen_ckpt"] = best[1].name; res["cal"] = best[2]
        res["selection"] = [{"ckpt": s[1].name, **Q.sel_row(pool_units(s[2], unit_of))} for s in sel]
        load(best[1]); res["dev"] = reg(dv, maps, "critic", {"eV": eV, "critic": cr}, prof)
    for part in ("cal", "dev"):
        Q.summarise(res, part); pooled = pool_units(res[part], unit_of)            # window-macro (REG-P2's summary) and unit-macro success
        for thr in Q.THR_FULL:
            res[f"{part}_success{int(thr)}_unit"] = P.success(pooled, thr)
    res["seconds_total"] = time.time() - t0; res["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    name = f"{a.method}" + ("" if a.method in ("mi", "nmi") else f"_s{a.seed}") + ("_smoke" if a.smoke else ""); (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"[register-p3 {name}] CAL unit success<42 {res['cal_success42_unit']:.3f}  DEV unit success<42 {res['dev_success42_unit']:.3f} <9 {res['dev_success9_unit']:.3f} "
          f"median corner {res['dev_median_err'] / P.DOWN:.2f} native px, rot {res['dev_median_rot_err_deg']:.2f}°, failures {res['dev_failures42']} "
          f"(energy {res['dev_failures42_energy']}) ({res['seconds_total']:.0f}s, {res['dev_sec_per_registration']:.2f}s/reg, {res['device']})", flush=True)
    return 0


def cmd_register(a):
    torch, nn, F = P.torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in a.seeds:
        for mth in a.methods:
            b = argparse.Namespace(**vars(a)); b.method, b.seed = mth, s; register_one(b, torch, nn, F, dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prepare", "perturb", "train", "comir", "register"]); ap.add_argument("--root", type=Path, default=ROOT3)
    ap.add_argument("--dataset", default="epfl", choices=["epfl", "zurich"], help="prepare: the pilot's data (epfl); zurich is kept for the record (not run)")
    ap.add_argument("--data", type=Path, default=None, help="dataset directory (read only); default per --dataset")
    ap.add_argument("--objective", default="vcs", help="train: comma list of vcs / logistic / infonce, run sequentially")
    ap.add_argument("--method", default="vcs", help="register: comma list of " + " / ".join(METHODS) + ", run sequentially")
    ap.add_argument("--seed", default="0", help="comma list of seeds, run sequentially"); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--comir-epochs", type=int, default=0, help="check runs only: CoMIR epochs at 32 steps × 32 patches (0 = recipe)")
    ap.add_argument("--n-pert", type=int, default=P.N_PERT, help="perturbations registered per window (the pilot: all 10)"); a = ap.parse_args()
    a.objectives = [o for o in a.objective.split(",") if o]; a.methods = [x for x in a.method.split(",") if x]; a.seeds = [int(s) for s in a.seed.split(",") if s != ""]
    bad = [o for o in a.objectives if o not in OBJECTIVES] + [x for x in a.methods if x not in METHODS]
    if bad or not a.seeds:
        ap.error(f"unknown objective / method {bad} or no seed")
    a.objective, a.method = a.objectives[0], a.methods[0]; a.data = a.data or DATA_DEFAULT[a.dataset]
    if a.smoke:
        a.root = a.root / "smoke"
    a.root.mkdir(parents=True, exist_ok=True)
    return {"prepare": cmd_prepare, "perturb": cmd_perturb, "train": cmd_train, "comir": cmd_comir, "register": cmd_register}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
