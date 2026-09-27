"""Pre-check A, P57 addendum 1 — EXTERNAL target for A-S1 / A-T: Localized Narratives (Google, annotations CC BY 4.0) on the Open Images
VALIDATION split (images CC BY 2.0, per-image attribution CSV kept next to the data).

    python scripts/precheck_a_external.py features --data <LN dir> --out <features dir> [--limit N]
    python scripts/precheck_a_external.py run --shift-dirs animal=<P49 dir>,indoor_outdoor=<P57 dir> --ext <features dir> --out <prefix> [--smoke]

features: frozen CLIP ViT-B/32 (same tower/cache as P49) for the selected images and their narrative (first narrative per image; narratives
are long, the CLIP tokenizer truncates at 77 tokens and the truncation rate is recorded).  Saved as one split `EXT-EVAL` in the P49 format —
`txt` is [n, 5, 512] with the single narrative embedding repeated in all five slots so that every P49/P57 helper (which reads caption slot 4)
works unchanged; identity = Open Images image id (hex), sha256 of the id list recorded.
run: for each source shift (animal = P49 splits, indoor = P57 splits) re-train the P58 *topic-pairing* selection (lr / epochs read from the P58
JSON) for vcs / infonce / logistic, 3 seeds, and evaluate on EXT-EVAL: A-S1 quantities (held-out J, native ECE vs cosine + Platt fitted on the
respective SRC-CAL, R@1) and the A-T task (matched = own narrative, mismatched = a random other image's narrative — no topic labels exist on
this target) at every nominal precision in --nominals; raw CLIP as reference; the in-domain SRC-EVAL (topic) row is repeated for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_a_adapters import Adapters, brier, ece, heldout_J, load_split, pair_sets, platt, retrieval, train  # noqa: E402
from precheck_a_features import CLIP_CACHE, ImgDS  # noqa: E402
from precheck_a_wave2 import make_partners, oracle_platt, rule_metrics, task_pairs  # noqa: E402


def sha_hex_ids(ids):
    return hashlib.sha256(",".join(ids).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------- features
def build_features(a) -> int:
    import open_clip
    from torch.utils.data import DataLoader
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    data = Path(a.data); out = Path(a.out); (out / "features").mkdir(parents=True, exist_ok=True)
    sel = json.load(open(data / "selection.json")); ids = [i for i in sel["image_ids"] if (data / "images" / f"{i}.jpg").exists() and (data / "images" / f"{i}.jpg").stat().st_size > 0]
    missing = len(sel["image_ids"]) - len(ids)
    if a.limit:
        ids = ids[: a.limit]
    caps = [sel["captions"][i] for i in ids]
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=CLIP_CACHE); tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval(); t0 = time.time()
    dl = DataLoader(ImgDS([str(data / "images" / f"{i}.jpg") for i in ids], preprocess), batch_size=a.batch, num_workers=a.workers, shuffle=False)
    feats = torch.zeros(len(ids), 512); truncated = torch.zeros(len(ids), dtype=torch.bool); tf = torch.zeros(len(ids), 512)
    with torch.no_grad():
        for x, ii in dl:
            feats[ii] = model.encode_image(x.to(device)).float().cpu()
        for s in range(0, len(caps), 512):
            toks = tok(caps[s: s + 512]); truncated[s: s + 512] = toks[:, -1] != 0   # the context is full only when the narrative was cut
            tf[s: s + 512] = model.encode_text(toks.to(device)).float().cpu()
    torch.save({"image_ids": torch.arange(len(ids)), "image_ids_hex": ids, "img": feats, "txt": tf[:, None, :].repeat(1, 5, 1), "captions": [[c] * 5 for c in caps],
                "truncated": truncated}, out / "features" / "EXT-EVAL.pt")
    man = {"towers": "open_clip ViT-B-32 laion2b_s34b_b79k (see models/open_clip/PROVENANCE_*.json)", "device": str(device), "source": str(data), "n_images": len(ids),
           "n_missing_images": missing, "ids_sha256": sha_hex_ids(ids), "selection_ids_sha256": sel["ids_sha256"], "truncation_rate": float(truncated.float().mean()),
           "narrative_words_mean": float(np.mean([len(c.split()) for c in caps])), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": time.time() - t0}
    json.dump({"splits": {"EXT-EVAL": ids}, "sha256": {"EXT-EVAL": sha_hex_ids(ids)}, "rule": sel["rule"], "seed": sel["seed"]}, open(out / "splits.json", "w"))
    json.dump(man, open(out / "manifest.json", "w"), indent=2); print(json.dumps(man, indent=1)); return 0


# --------------------------------------------------------------------------------------------------------------------------------- run
def eval_ext(m, method, ext_img, ext_txt, cal_img, cal_txt, cal_topic, cal_coarse, device, nominals):
    """A-S1 quantities and the A-T task on EXT-EVAL (exact pairing: own narrative vs a random other narrative)."""
    n = len(ext_img); row = {}
    cal_s, cal_c, cal_y, _, _ = pair_sets(m, cal_img, cal_txt, device, 11, cal_topic)          # calibration pairs on SRC-CAL as in P58
    s_, c_, y_, u, v = pair_sets(m, ext_img, ext_txt, device, 13, None)
    probs = {}
    if method == "vcs": probs["native"] = (1 + s_) / 2
    if method == "logistic": probs["native"] = torch.sigmoid(s_)
    probs["cosine+Platt(CAL)"], _, _ = platt(cal_c, cal_y, c_)
    row["R1_R5"] = retrieval(u, v); row["calibration"] = {k: dict(zip(("ECE", "max_dev", "Brier"), (*ece(p, y_)[:2], brier(p, y_)))) for k, p in probs.items()}
    if method == "vcs": row["heldout_J"], row["saturation"] = heldout_J(m, ext_img, ext_txt, device, 17)
    # A-T: source rules fitted on SRC-CAL task pairs (easy = random mismatch, the kind available on the target; hard = coarse-topic mismatch)
    task = {}
    cal_easy = task_pairs(m, cal_img, cal_txt, device, cal_topic, None, 21); cal_hard = task_pairs(m, cal_img, cal_txt, device, cal_topic, cal_coarse, 21)
    t_s, t_c, t_y = task_pairs(m, ext_img, ext_txt, device, torch.arange(n), None, 23)
    rules = {}
    if method == "vcs": rules["native_(1+T)/2"] = (1 + t_s) / 2
    if method == "logistic": rules["native_sigmoid"] = torch.sigmoid(t_s)
    rules["cosine+Platt(SRC-CAL easy)"], _, _ = platt(cal_easy[1], cal_easy[2], t_c); rules["cosine+Platt(SRC-CAL hard)"], _, _ = platt(cal_hard[1], cal_hard[2], t_c)
    if method in ("vcs", "logistic"): rules["score+Platt(SRC-CAL easy)"], _, _ = platt(cal_easy[0], cal_easy[2], t_s)
    rules["cosine+Platt(EXT oracle, cross-fitted)"] = oracle_platt(t_c, t_y)
    for nom in nominals:
        task[str(nom)] = {"n_pairs": int(len(t_y)), "rules": {k: rule_metrics(p, t_y, t_c, nom) for k, p in rules.items()}}
    row["task"] = task; return row


def run(a) -> int:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    shift_dirs = dict(kv.split("=", 1) for kv in a.shift_dirs.split(",")); methods = a.methods.split(","); seeds = [int(x) for x in a.seeds.split(",")]
    nominals = [float(x) for x in a.nominals.split(",")]; selection = json.load(open(a.selection)) if Path(a.selection).exists() else None
    if a.smoke: methods, seeds = methods[:1], seeds[:1]
    ext = Path(a.ext); ext_img, ext_txt, _ = load_split(ext, "EXT-EVAL"); ext_man = json.load(open(ext / "manifest.json"))
    results = {"settings": vars(a), "device": str(device), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ext_manifest": ext_man, "shifts": {}}; t0 = time.time()
    for shift, fdir in shift_dirs.items():
        fd = Path(fdir); raw = {k: load_split(fd, k) for k in ("SRC-FIT", "SRC-CAL", "SRC-EVAL")}
        if a.smoke: raw = {k: (v[0][:2000], v[1][:2000], v[2][:2000]) for k, v in raw.items()}
        splits = {k: (v[0], v[1]) for k, v in raw.items()}
        topic = {k: make_partners(raw[k][2], a.index_dir, 20260927, "topic") for k in raw}; coarse = {k: make_partners(raw[k][2], a.index_dir, 20260927, "coarse") for k in raw}
        fit_img, fit_txt = splits["SRC-FIT"]; cal_img, cal_txt = splits["SRC-CAL"]; ev_img, ev_txt = splits["SRC-EVAL"]
        R = {"features": str(fd), "n": {k: len(v[0]) for k, v in raw.items()}, "methods": {}}
        ref = Adapters("infonce").to(device).eval()
        R["raw_clip"] = {"EXT-EVAL": eval_ext(ref, "infonce", ext_img, ext_txt, cal_img, cal_txt, topic["SRC-CAL"][1], coarse["SRC-CAL"][1], device, nominals)}
        for method in methods:
            sel = (selection or {}).get("shifts", {}).get(shift, {}).get("pairings", {}).get("topic", {}).get(method, {}).get("selected")
            lr, ep = (sel["lr"], sel["epochs"]) if sel else (1e-3, 15)
            if a.smoke: ep = 1
            res = {"selected": {"lr": lr, "epochs": ep, "from": a.selection if sel else "default"}, "seeds": {}}
            for seed in seeds:
                m, val = train(method, fit_img, fit_txt, cal_img, cal_txt, lr, ep, seed, device, fit_partners=topic["SRC-FIT"][0], cal_partners=topic["SRC-CAL"][0])
                r = {"cal_loss": val, "EXT-EVAL": eval_ext(m, method, ext_img, ext_txt, cal_img, cal_txt, topic["SRC-CAL"][1], coarse["SRC-CAL"][1], device, nominals)}
                # in-domain reference row (SRC-EVAL, topic partner), as in P58
                cal_s, cal_c, cal_y, _, _ = pair_sets(m, cal_img, cal_txt, device, 11, topic["SRC-CAL"][1]); s_, c_, y_, u, v = pair_sets(m, ev_img, ev_txt, device, 13, topic["SRC-EVAL"][1])
                probs = {}
                if method == "vcs": probs["native"] = (1 + s_) / 2
                if method == "logistic": probs["native"] = torch.sigmoid(s_)
                probs["cosine+Platt(CAL)"], _, _ = platt(cal_c, cal_y, c_)
                row = {"R1_R5": retrieval(u, v), "calibration": {k: dict(zip(("ECE", "max_dev", "Brier"), (*ece(p, y_)[:2], brier(p, y_)))) for k, p in probs.items()}}
                if method == "vcs": row["heldout_J"], row["saturation"] = heldout_J(m, ev_img, ev_txt, device, 17, fixed_partner=topic["SRC-EVAL"][1])
                r["SRC-EVAL"] = row; res["seeds"][str(seed)] = r
            R["methods"][method] = res
            e = res["seeds"][str(seeds[0])]["EXT-EVAL"]; print(f"[{shift}/{method}] lr={lr:g} ep={ep}: EXT ECE " + ", ".join(f"{k} {v['ECE']:.3f}" for k, v in e["calibration"].items()) + (f" | J {e['heldout_J']:.3f}" if method == "vcs" else "") + f" | R@1 {e['R1_R5'][0]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        results["shifts"][shift] = R
    json.dump(results, open(a.out + ".json", "w"), indent=1, default=float); write_markdown(results, a); print("->", a.out + ".md"); return 0


def write_markdown(results, a):
    man = results["ext_manifest"]; mean = lambda xs: float(np.mean(xs))
    L = [f"# Pre-check A, addendum 1 — external target: Localized Narratives on Open Images validation ({man['n_images']} images; narrative truncation rate {man['truncation_rate']:.3f}, mean {man['narrative_words_mean']:.0f} words) — {results['utc']}", "",
         "Adapters = the P58 topic-pairing selection re-trained per source (lr / epochs in the JSON); pairing on the external target is exact (own narrative); mismatches are random (no topic labels).", "",
         "## A-S1 quantities (mean over seeds)", "", "| source | method | split | J (vcs) | native ECE | cosine+Platt(SRC-CAL) ECE | native − Platt | R@1 |", "|---|---|---|---|---|---|---|---|"]
    for shift, R in results["shifts"].items():
        for method, res in R["methods"].items():
            for split in ("SRC-EVAL", "EXT-EVAL"):
                rows = [r[split] for r in res["seeds"].values()]; nat = mean([r["calibration"]["native"]["ECE"] for r in rows]) if "native" in rows[0]["calibration"] else float("nan")
                pl = mean([r["calibration"]["cosine+Platt(CAL)"]["ECE"] for r in rows]); J = f"{mean([r['heldout_J'] for r in rows]):.3f}" if method == "vcs" else "—"
                L.append(f"| {shift} | {method} | {split} | {J} | {nat:.4f} | {pl:.4f} | {nat - pl:+.4f} | {mean([r['R1_R5'][0] for r in rows]):.3f} |")
    for nom in [float(x) for x in a.nominals.split(",")]:
        L += ["", f"## A-T on EXT-EVAL — nominal precision {nom:g} (matched = own narrative, mismatched = random other narrative; mean over seeds)", "",
              "| source | method | rule | accept | precision | \\|prec − nominal\\| | FNR | FPR | bal.err | AUROC | ECE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for shift, R in results["shifts"].items():
            items = [("raw_clip", {"seeds": {"0": {"EXT-EVAL": R["raw_clip"]["EXT-EVAL"]}}})] + list(R["methods"].items())
            for method, res in items:
                rules = next(iter(res["seeds"].values()))["EXT-EVAL"]["task"][str(nom)]["rules"].keys()
                for rule in rules:
                    vs = [r["EXT-EVAL"]["task"][str(nom)]["rules"][rule] for r in res["seeds"].values()]
                    g = lambda k: mean([v[k] for v in vs])
                    L.append(f"| {shift} | {method} | {rule} | {g('accept_rate'):.3f} | {g('precision'):.3f} | {g('abs_dev_from_nominal'):.3f} | {g('FNR'):.3f} | {g('FPR'):.3f} | {g('balanced_err'):.3f} | {g('AUROC'):.3f} | {g('ECE'):.4f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features"); f.add_argument("--data", required=True); f.add_argument("--out", required=True); f.add_argument("--limit", type=int, default=None); f.add_argument("--workers", type=int, default=8); f.add_argument("--batch", type=int, default=256)
    r = sub.add_parser("run"); r.add_argument("--shift-dirs", required=True); r.add_argument("--ext", required=True); r.add_argument("--out", required=True)
    r.add_argument("--methods", default="vcs,infonce,logistic"); r.add_argument("--seeds", default="0,1,2"); r.add_argument("--nominals", default="0.8,0.7")
    r.add_argument("--selection", default="reports/P58_precheck_A_wave2.json"); r.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); r.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); return build_features(a) if a.cmd == "features" else run(a)


if __name__ == "__main__":
    sys.exit(main())
