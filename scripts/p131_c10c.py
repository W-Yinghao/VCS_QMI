"""P131 — CIFAR-10-C-style robustness of frozen SSL representations (corruptions regenerated locally; reporting only).

    generate : VCS_P131_C10C=1 python scripts/p131_c10c.py generate --owner-authorised-p131 [--workers 15] [--n-images N (smoke)]
               official CIFAR-10 test images (md5-verified loader of the final-round code, used ONLY as corruption inputs) ->
               outputs/P131_c10c_regen/<corruption>.npy  [5 x 10000, 32, 32, 3] uint8, severities 1..5 in order (CIFAR-10-C layout),
               labels.npy, manifest.json (parameters, block seeds, library versions, sha256 of every file).  Clean test images are
               never written and never scored.
    evaluate : python scripts/p131_c10c.py evaluate --runs R1,R2,... [--out outputs/P131_results] [--smoke-n 500]
               per run: frozen h of the final checkpoint; the pilot linear head (vcs_ssl.diagnostics.linear_probe, frozen probe config,
               trained on the 45k dev fit split) — its clean development-validation accuracy is recomputed and checked against the
               run's stored evaluation_epoch_800.json; then accuracy on every corruption x severity.  No selection, no tuning.
    aggregate: python scripts/p131_c10c.py aggregate [--in outputs/P131_results] [--out reports/P131_c10c_results]

Labels "CIFAR-10-C-style, regenerated locally" — never "CIFAR-10-C" (the released files are not used; see vcs_diag.c10c for what is
re-implemented, approximated and omitted).  Owner authorisation 2026-10-04 ("全部提交", CIFAR-10-C reporting only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs"))
DATA_ROOT = os.environ.get("DATA_ROOT", "/home/infres/yinwang/CS_QMI/data/cifar10")
REGEN = Path(os.environ.get("P131_REGEN_DIR", str(OUT_ROOT / "P131_c10c_regen")))   # smoke: P131_REGEN_DIR=.../P131_c10c_regen_smoke
AUTH_ENV = "VCS_P131_C10C"
BASE_SEED = 131
LABEL = "CIFAR-10-C-style, regenerated locally"

# method groups (seed-paired); names checked at aggregation
GROUPS = {
    "A-P3": {s: f"P107_AP3_views4_800ep_seed{s}" for s in range(5)},
    "SimCLR": {0: "P41_simclr_views4_800ep_seed0", 1: "P41_simclr_views4_800ep_seed1", 2: "P41_simclr_views4_800ep_seed2",
               3: "P111_simclr_views4_800ep_seed3", 4: "P111_simclr_views4_800ep_seed4"},
    "G2": {0: "P104_G2_views4_800ep_seed0", 1: "P104_G2_views4_800ep_seed1", 2: "P104_G2_views4_800ep_seed2",
           3: "P111_G2_views4_800ep_seed3", 4: "P111_G2_views4_800ep_seed4"},
    "JS-AP3": {s: f"P114_JSAP3_views4_800ep_seed{s}" for s in range(3)},
    "recipe VCS": {s: f"P35_vcs_a5_views4_800ep_seed{s}" for s in range(3)},
}
ALL_RUNS = [r for g in GROUPS.values() for r in g.values()]


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------------------------------------------------------
def _load_test_images_for_corruption(authorised: bool):
    if not (authorised and os.environ.get(AUTH_ENV, "") == "1"):
        raise PermissionError(f"P131 generation reads the official CIFAR-10 test images as corruption inputs; it needs --owner-authorised-p131 "
                              f"and {AUTH_ENV}=1 (owner authorisation 2026-10-04, reporting only)")
    from vcs_ssl.data.cifar import _load_cifar10_official_test  # noqa: PLC0415  (md5-verified loader of the final-round code)
    return _load_cifar10_official_test(DATA_ROOT)


def _gen_one(args):
    name, images, n = args
    from vcs_diag import c10c  # noqa: PLC0415
    t = time.time()
    blocks = [c10c.corrupt_block(images[:n], name, sev, c10c.block_seed(BASE_SEED, name, sev)) for sev in range(1, 6)]
    arr = np.concatenate(blocks, axis=0)
    out = REGEN / f"{name}.npy"
    tmp = out.with_name(out.stem + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(out)
    return name, time.time() - t, sha256_path(out), list(arr.shape)


def cmd_generate(a) -> int:
    from vcs_diag import c10c  # noqa: PLC0415
    test = _load_test_images_for_corruption(a.owner_authorised_p131)
    n = int(a.n_images or len(test.targets))
    REGEN.mkdir(parents=True, exist_ok=True)
    names = c10c.PRIMARY + c10c.EXTRA
    t0 = time.time()
    files = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, sec, sha, shape in ex.map(_gen_one, [(nm, test.data, n) for nm in names]):
            files[name] = {"sha256": sha, "shape": shape, "seconds": round(sec, 1), "block_seeds": [c10c.block_seed(BASE_SEED, name, s) for s in range(1, 6)]}
            print(f"[gen] {name:18s} {sec:7.1f}s {shape}", flush=True)
    labels = np.tile(np.asarray(test.targets[:n], dtype=np.int64), 5)
    np.save(REGEN / "labels.npy", labels)
    import scipy, PIL, torch  # noqa: PLC0415,E401
    man = {"label": LABEL, "unit": "P131", "authorisation": "owner 2026-10-04 '全部提交' — CIFAR-10-C reporting only; clean test images never stored or scored",
           "source": {"partition": "official CIFAR-10 test (test_batch), corruption input only", "test_batch_sha256": test.source.get("test_batch_sha256"),
                      "n_images": n},
           "layout": "per corruption: severities 1..5 concatenated, n_images each, in test_batch order; labels.npy = test labels tiled 5x",
           "reference": "Hendrycks & Dietterich 2019, hendrycks/robustness ImageNet-C/create_c/make_cifar_c.py (CIFAR severity parameters)",
           "primary": c10c.PRIMARY, "families": c10c.FAMILIES, "extra": c10c.EXTRA, "omitted": c10c.OMITTED,
           "approximations": {"elastic_transform": "cv2.warpAffine replaced by scipy.ndimage.affine_transform (bilinear, mirror); cv2 quantises to 1/32 px",
                              "all": "RandomState per (corruption, severity) block (original: legacy global state) -> same distribution, not the released pixels; "
                                     "scipy >= 1.6 boundary handling in zoom / map_coordinates"},
           "cast": "np.uint8(float output) as the original (truncation)", "base_seed": BASE_SEED, "files": files,
           "labels_sha256": sha256_path(REGEN / "labels.npy"),
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "PIL": PIL.__version__, "torch": torch.__version__},
           "seconds_total": round(time.time() - t0, 1), "smoke": n < len(test.targets)}
    (REGEN / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"-> {REGEN} ({len(files)} corruptions, n={n}, {man['seconds_total']} s)")
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------------------------------------------------
def cmd_evaluate(a) -> int:
    import torch  # noqa: PLC0415
    from vcs_ssl.data.cifar import load_cifar10_train  # noqa: PLC0415
    from vcs_ssl.data.transforms import build_clean_transform  # noqa: PLC0415
    from vcs_ssl.diagnostics import extract_features, linear_probe  # noqa: PLC0415
    from vcs_ssl.evaluate import _load_run_and_models  # noqa: PLC0415

    man = json.loads((REGEN / "manifest.json").read_text())
    if man.get("label") != LABEL:
        raise ValueError("unexpected regenerated-set manifest")
    labels_all = torch.from_numpy(np.load(REGEN / "labels.npy"))
    n_img = int(man["source"]["n_images"])
    names = man["primary"] + man["extra"]
    arrays = {nm: np.load(REGEN / f"{nm}.npy", mmap_mode="r") for nm in names}
    smoke_n = int(a.smoke_n) if a.smoke_n else n_img
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    data = load_cifar10_train(DATA_ROOT)
    for run in [r for r in a.runs.split(",") if r]:
        dest = out_dir / f"{run}.json"
        if dest.is_file() and not a.force:
            print(f"[skip] {run}: {dest} exists", flush=True)
            continue
        t0 = time.time()
        run_dir = OUT_ROOT / run
        cfg, manifest, ck, ckpt_sha, enc, proj = _load_run_and_models(run_dir, "epoch_800.pt", device)
        if cfg["data"]["name"] != "cifar10":
            raise ValueError(f"{run}: P131 is CIFAR-10 only")
        if manifest["raw_file_hashes"] != data.file_hashes:
            raise ValueError("manifest raw file hashes differ from the data on disk")
        clean = build_clean_transform(cfg["views"])
        fit_uids = np.asarray(manifest["fit_uids"], dtype=np.int64)
        sel_uids = np.asarray(manifest["selection_uids"], dtype=np.int64)
        with torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
            torch.manual_seed(0)
            fit = extract_features(enc, None, data.data, data.targets, fit_uids, clean, device=device, num_workers=a.num_workers, seed=11)
            sel = extract_features(enc, None, data.data, data.targets, sel_uids, clean, device=device, num_workers=a.num_workers, seed=12)
            lp = linear_probe(fit["h"], fit["labels"], sel["h"], sel["labels"], cfg["evaluation"]["linear"], device=device, return_head=True)
        head = lp.pop("head")
        stored = json.loads((run_dir / "evaluations" / "evaluation_epoch_800.json").read_text())["linear_val_top1_pct"]
        mean = torch.tensor(cfg["views"]["normalize_mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["views"]["normalize_std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)

        @torch.no_grad()
        def h_fast(arr_u8: np.ndarray) -> torch.Tensor:
            # == ToTensor (uint8 -> float32 / 255) + Normalize, on the GPU; batch 512 as extract_features
            hs = []
            for s in range(0, len(arr_u8), 512):
                x = torch.from_numpy(np.array(arr_u8[s:s + 512])).to(device).permute(0, 3, 1, 2).float().div(255)
                hs.append(enc((x - mean) / std).float())
            return torch.cat(hs)

        # fast path == pilot feature path on the 5k selection split
        h_sel_fast = h_fast(data.data[sel_uids])
        fast_err = float((h_sel_fast.cpu() - sel["h"]).abs().max())
        with torch.no_grad():
            fast_acc = float((head(h_sel_fast).argmax(1).cpu() == sel["labels"]).float().mean()) * 100
        res = {"label": LABEL, "run_id": run, "seed": int(cfg["run"]["seed"]), "method": cfg["run"]["method"], "checkpoint": "epoch_800.pt",
               "checkpoint_sha256": ckpt_sha, "regen_manifest_sha256": sha256_path(REGEN / "manifest.json"),
               "clean_dev_val_top1_pct": lp["linear_val_top1_pct"], "stored_linear_val_top1_pct": stored,
               "clean_val_matches_stored": abs(lp["linear_val_top1_pct"] - stored) < 0.05,
               "fast_path_max_abs_h_err": fast_err, "fast_path_dev_val_top1_pct": fast_acc, "probe": {k: v for k, v in lp.items() if k != "curve"},
               "n_per_block": smoke_n, "acc": {}}
        with torch.no_grad():
            for nm in names:
                res["acc"][nm] = []
                for sev in range(1, 6):
                    lo = (sev - 1) * n_img
                    blk = arrays[nm][lo:lo + smoke_n]
                    y = labels_all[lo:lo + smoke_n].to(device)
                    pred = head(h_fast(blk)).argmax(1)
                    res["acc"][nm].append(float((pred == y).float().mean()) * 100)
        prim = man["primary"]
        res["mean_corruption_acc_primary"] = float(np.mean([np.mean(res["acc"][nm]) for nm in prim]))
        res["family_mean_acc"] = {f: float(np.mean([np.mean(res["acc"][nm]) for nm in ms])) for f, ms in man["families"].items()}
        res["mean_corruption_acc_extra"] = float(np.mean([np.mean(res["acc"][nm]) for nm in man["extra"]]))
        res["drop_primary_vs_clean_dev_val"] = res["clean_dev_val_top1_pct"] - res["mean_corruption_acc_primary"]
        res["seconds"] = round(time.time() - t0, 1)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(dest)
        print(f"[eval] {run}: clean dev-val {res['clean_dev_val_top1_pct']:.2f} (stored {stored:.2f}) fast-path err {fast_err:.2e}  "
              f"mCA(primary) {res['mean_corruption_acc_primary']:.2f}  {res['seconds']} s", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------------------------------------------------------
def _paired(a_vals: dict, b_vals: dict) -> dict:
    from scipy import stats  # noqa: PLC0415
    seeds = sorted(set(a_vals) & set(b_vals))
    d = np.array([a_vals[s] - b_vals[s] for s in seeds])
    n = len(d)
    if n < 2:
        return {"n": n, "delta": d.tolist()}
    m = float(d.mean()); h = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
    lab = "close" if abs(m) < 0.3 else ("clear" if (m - h > 0 or m + h < 0) else "inconclusive")
    return {"n": n, "seeds": seeds, "delta": d.tolist(), "mean": m, "ci95": [m - h, m + h], "label": lab}


def cmd_aggregate(a) -> int:
    rows = {}
    for f in sorted(Path(a.inp).glob("*.json")):
        r = json.loads(f.read_text())
        rows[r["run_id"]] = r
    man = json.loads((REGEN / "manifest.json").read_text())
    def metric(r, key, fam=None):
        if key == "mca":
            return r["mean_corruption_acc_primary"]
        if key == "drop":
            return r["drop_primary_vs_clean_dev_val"]
        if key == "fam":
            return r["family_mean_acc"][fam]
        if key == "clean":
            return r["clean_dev_val_top1_pct"]
    per = {g: {s: rows[r] for s, r in m.items() if r in rows} for g, m in GROUPS.items()}
    out = {"label": LABEL, "n_runs": len(rows), "missing": [r for r in ALL_RUNS if r not in rows], "groups": {}, "contrasts": {}}
    lines = [f"# P131 — {LABEL}: results", "", f"{len(rows)} runs; missing: {out['missing'] or 'none'}.  Primary = mean accuracy over the "
             f"{len(man['primary'])} implemented standard corruptions x 5 severities (omitted: {', '.join(man['omitted'])}).", "",
             "| method | seeds | clean dev-val | mCA (primary) | drop | " + " | ".join(man["families"]) + " | extra |", "|" + "---|" * (6 + len(man["families"]))]
    for g, d in per.items():
        if not d:
            continue
        vals = lambda k, fam=None: [metric(r, k, fam) for r in d.values()]  # noqa: E731
        out["groups"][g] = {"seeds": sorted(d), "clean": vals("clean"), "mca": vals("mca"), "drop": vals("drop"),
                            "families": {f: vals("fam", f) for f in man["families"]}, "extra": [r["mean_corruption_acc_extra"] for r in d.values()]}
        lines.append(f"| {g} | {len(d)} | {np.mean(vals('clean')):.2f} | {np.mean(vals('mca')):.2f} | {np.mean(vals('drop')):.2f} | "
                     + " | ".join(f"{np.mean(vals('fam', f)):.2f}" for f in man["families"]) + f" | {np.mean(out['groups'][g]['extra']):.2f} |")
    lines += ["", "## Paired contrasts (by seed; labels as P114: close |Δ| < 0.3, clear |Δ| ≥ 0.3 with the 95 % t interval excluding 0, else inconclusive)", "",
              "| contrast | metric | n | mean Δ | 95 % CI | label |", "|---|---|---|---|---|---|"]
    for ga, gb in (("A-P3", "SimCLR"), ("A-P3", "JS-AP3"), ("G2", "SimCLR"), ("A-P3", "recipe VCS")):
        if not per.get(ga) or not per.get(gb):
            continue
        for key, fams in (("mca", [None]), ("drop", [None]), ("fam", list(man["families"]))):
            for fam in fams:
                c = _paired({s: metric(r, key, fam) for s, r in per[ga].items()}, {s: metric(r, key, fam) for s, r in per[gb].items()})
                nm = key if fam is None else f"family:{fam}"
                out["contrasts"][f"{ga} - {gb} | {nm}"] = c
                if "mean" in c:
                    lines.append(f"| {ga} − {gb} | {nm} | {c['n']} | {c['mean']:+.2f} | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | {c['label']} |")
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    print(f"-> {a.out}.md / .json ({len(rows)} runs)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--owner-authorised-p131", action="store_true"); g.add_argument("--workers", type=int, default=15)
    g.add_argument("--n-images", type=int, default=0, help="smoke: corrupt only the first N test images")
    e = sub.add_parser("evaluate"); e.add_argument("--runs", default=",".join(ALL_RUNS)); e.add_argument("--out", default=str(OUT_ROOT / "P131_results"))
    e.add_argument("--smoke-n", type=int, default=0); e.add_argument("--num-workers", type=int, default=6); e.add_argument("--force", action="store_true")
    s = sub.add_parser("aggregate"); s.add_argument("--in", dest="inp", default=str(OUT_ROOT / "P131_results")); s.add_argument("--out", default="reports/P131_c10c_results")
    a = ap.parse_args(argv)
    return {"generate": cmd_generate, "evaluate": cmd_evaluate, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
