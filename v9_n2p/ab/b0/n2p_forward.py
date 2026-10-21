"""N2P DEV-only forward for NCR_A / L2RM_A candidates.  Runs in the baseline's official env, under n2p_guard.py policy "dev".

For every candidate of the frozen map that the trainer saved: official model, official encode_data / shard_attn_scores (the
shard size of the trainer's own validation), eval mode, no gradient, on the full Flickr DEV (1014 images / 5070 captions with
explicit IDs).  Writes one float32 .npy per network and one score-input JSON in the package schema
(v9-common-dev-score-input-v1).  The same checkpoint also goes through the official native validation path (first 1000 captions,
200 images) for the native200 parity table.  No TEST loader, no training step.

    python n2p_forward.py --baseline ncr|l2rm --unit <unit dir> --frozen <B0_FROZEN.json> --schedule gate|full --seed S --rho R
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

RECALLS = ("i2t_R1", "i2t_R5", "i2t_R10", "t2i_R1", "t2i_R5", "t2i_R10")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
            n += len(chunk)
    return {"path": path, "sha256": h.hexdigest(), "bytes": n}


def order_hash(ids):
    return sha_bytes(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--baseline", "--unit", "--frozen", "--schedule"):
        ap.add_argument(k, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    frozen = json.load(open(a.frozen))
    fam = frozen["families"][a.baseline]
    unit = a.unit
    code = os.path.join(unit, "code", a.baseline, fam["code_subdir"])
    sys.path.insert(0, code)
    from vocab import deserialize_vocab
    from data import get_dataset, get_loader, PrecompDataset, collate_fn
    from evaluation import encode_data, shard_attn_scores, i2t, t2i
    SGRAF = __import__(fam["model_module"]).SGRAF

    data_view = os.path.join(unit, "data_view")
    real = os.path.realpath(os.path.join(data_view, "f30k_precomp"))
    vocab = deserialize_vocab(os.path.join(frozen["paths"]["vocab_dir"], "f30k_precomp_vocab.json"))
    captions, images = get_dataset(data_view, "f30k_precomp", "dev", vocab)        # official tokenisation; dev files only

    # explicit IDs: dev_ids.txt gives the image ID of every caption row; five consecutive rows per image, rows duplicated in dev_ims.npy
    dev_ids = open(os.path.join(real, "dev_ids.txt")).read().split()
    if len(dev_ids) != 5070 or len(captions) != 5070 or images.shape[0] != 5070:
        raise SystemExit("unexpected Flickr DEV size")
    blocks_ok = all(len(set(dev_ids[5 * i:5 * i + 5])) == 1 for i in range(1014))
    rows_ok = all(np.array_equal(images[5 * i], images[5 * i + j]) for i in range(1014) for j in range(1, 5))
    image_ids = [dev_ids[5 * i] for i in range(1014)]
    if not (blocks_ok and rows_ok and len(set(image_ids)) == 1014):
        raise SystemExit("DEV identity check failed: blocks_ok=%s rows_ok=%s" % (blocks_ok, rows_ok))
    caption_rows = [{"caption_id": str(r), "image_id": dev_ids[r]} for r in range(5070)]
    dev = os.path.join(unit, "dev")
    for sub in ("ids", "scores", "score_inputs"):
        os.makedirs(os.path.join(dev, sub), exist_ok=True)
    id_files = {}
    for name, obj in (("image_ids.json", image_ids), ("caption_rows.json", caption_rows)):
        p = os.path.join(dev, "ids", name)
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if os.path.exists(p) and open(p, "rb").read() != data:
            raise SystemExit("existing ID file differs: " + p)
        open(p, "wb").write(data)
        id_files[name] = {"path": p, "sha256": sha_bytes(data), "bytes": len(data)}
    img_hash, cap_hash = order_hash(image_ids), order_hash([r["caption_id"] for r in caption_rows])
    # native200: the official dev loader keeps caption rows 0..999 -> images rows 0,5,..,995 (verified against the full ID lists)
    native_img_ids, native_cap_ids = [dev_ids[5 * i] for i in range(200)], [str(r) for r in range(1000)]
    native_subset_ok = native_img_ids == image_ids[:200] and native_cap_ids == [r["caption_id"] for r in caption_rows[:1000]]
    data_reads = [dict(sha_file(os.path.join(real, f)), split="dev") for f in ("dev_ims.npy", "dev_caps.txt", "dev_ids.txt")]

    full_loader = torch.utils.data.DataLoader(PrecompDataset(captions, images, "dev_full"), batch_size=128, shuffle=False,
                                              pin_memory=True, collate_fn=collate_fn, num_workers=a.workers)
    native_loader = get_loader(captions, images, "dev", 128, a.workers)
    if len(full_loader.dataset) != 5070 or len(native_loader.dataset) != 1000:
        raise SystemExit("loader sizes differ from 5070 / 1000")

    cand_dir = os.path.join(unit, "candidates")
    shard = fam["validation_shard_size"]
    summary = {"schema": "n2p-forward-summary-v1", "baseline": a.baseline, "method": fam["method"], "seed": a.seed, "rho": a.rho,
               "schedule": a.schedule, "ids": id_files, "image_order_sha256": img_hash, "caption_order_sha256": cap_hash,
               "dev_identity": {"n_images": 1014, "n_captions": 5070, "five_rows_per_image_identical": rows_ok,
                                "five_captions_per_image": blocks_ok, "native200_is_exact_id_prefix": native_subset_ok},
               "data_reads": data_reads, "shard_size": shard, "networks": fam["networks"],
               "env": {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "cudnn_benchmark_after_model_build": None,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
               "candidates": [], "missing": []}
    for c in frozen["candidate_maps"][a.baseline][a.schedule]:
        path = os.path.join(cand_dir, c["file"])
        if not os.path.exists(path):
            summary["missing"].append(c["candidate_id"])
            continue
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        raw = open(path, "rb").read()
        ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)   # official checkpoints pickle argparse.Namespace
        if (ck["phase"], ck["epoch"], ck["candidate_id"]) != (c["phase"], c["epoch"], c["candidate_id"]):
            raise SystemExit("candidate file does not hold its declared candidate: " + path)
        opt = ck["opt"]
        full, native, specs = {}, {}, []
        for net_name in fam["networks"]:
            m = SGRAF(opt)
            if a.baseline == "ncr":
                m.load_state_dict(ck[net_name])
                m.val_start()
            else:
                m = m.cuda()
                m.load_state_dict(ck[net_name])
                m.eval()
            summary["env"]["cudnn_benchmark_after_model_build"] = torch.backends.cudnn.benchmark
            with torch.no_grad():
                im, ca, ln = encode_data(m, full_loader, 10 ** 9)
                full[net_name] = shard_attn_scores(m, im[::5], ca, ln, opt, shard_size=shard).astype(np.float32)
                im, ca, ln = encode_data(m, native_loader, 10 ** 9)
                native[net_name] = shard_attn_scores(m, im[::5], ca, ln, opt, shard_size=shard).astype(np.float32)
            del m
            S = full[net_name]
            if S.shape != (1014, 5070) or native[net_name].shape != (200, 1000) or not np.isfinite(S).all():
                raise SystemExit("bad similarity matrix for " + c["candidate_id"])
            sp = os.path.join(dev, "scores", "%s_%s.npy" % (c["file"].split(".")[0], net_name))
            np.save(sp, S)
            b = open(sp, "rb").read()
            specs.append({"path": sp, "sha256": sha_bytes(b), "bytes": len(b), "image_order_sha256": img_hash,
                          "caption_order_sha256": cap_hash, "model_id": net_name, "checkpoint_sha256": sha_bytes(raw),
                          "phase": c["phase"], "epoch": c["epoch"], "candidate_id": c["candidate_id"], "epoch_complete": True})
        score_input = {"schema": "v9-common-dev-score-input-v1", "split": "dev", "dataset": "flickr30k",
                       "evaluation_role": "N2P_common_full_DEV", "test_read": False, "method": fam["method"], "seed": a.seed,
                       "rho": a.rho, "phase": c["phase"], "epoch": c["epoch"], "candidate_id": c["candidate_id"],
                       "image_ids": id_files["image_ids.json"], "caption_rows": id_files["caption_rows.json"],
                       "data_reads": data_reads, "scores": specs}
        write_json(os.path.join(dev, "score_inputs", c["file"].split(".")[0] + ".json"), score_input)
        # native200 parity: official native path vs the ID-verified 200 x 1000 block of the full matrix, same checkpoint
        nat = sum(native.values()) / len(native)
        sub = sum(S[:200, :1000] for S in full.values()) / len(full)
        r_nat, (rk_nat, *_rest) = i2t(200, nat.astype(np.float64), 5, return_ranks=True)
        r_sub, (rk_sub, *_rest) = i2t(200, sub.astype(np.float64), 5, return_ranks=True)
        ri_nat, (rki_nat, *_rest) = t2i(200, nat.astype(np.float64), 5, return_ranks=True)
        ri_sub, (rki_sub, *_rest) = t2i(200, sub.astype(np.float64), 5, return_ranks=True)
        rec_nat = dict(zip(RECALLS, list(r_nat[:3]) + list(ri_nat[:3])))
        rec_sub = dict(zip(RECALLS, list(r_sub[:3]) + list(ri_sub[:3])))
        summary["candidates"].append({
            "candidate_id": c["candidate_id"], "file": c["file"], "checkpoint_sha256": sha_bytes(raw), "checkpoint_bytes": len(raw),
            "score_files": [s["path"] for s in specs],
            "native200": {"native_path": dict(rec_nat, rsum=sum(rec_nat.values())), "fulldev_subset": dict(rec_sub, rsum=sum(rec_sub.values())),
                          "max_abs_sim_diff": float(max(np.abs(native[n] - full[n][:200, :1000]).max() for n in fam["networks"])),
                          "i2t_rank_changes": [image_ids[i] for i in np.flatnonzero(rk_nat != rk_sub)],
                          "t2i_rank_changes": [str(j) for j in np.flatnonzero(rki_nat != rki_sub)]},
            "seconds": round(time.time() - t0, 1), "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None})
        print("[N2P forward] %s done in %.1f s" % (c["candidate_id"], time.time() - t0), flush=True)
    write_json(os.path.join(dev, "FORWARD_SUMMARY.json"), summary)
    print("[N2P forward] %d candidates, %d missing" % (len(summary["candidates"]), len(summary["missing"])))


if __name__ == "__main__":
    main()
