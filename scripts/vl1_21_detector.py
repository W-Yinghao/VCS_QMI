"""VL1-21 — clean detector proposer for Table C ("proposer + critic"): torchvision Faster R-CNN R50-FPN (v1, FrozenBN backbone, 3 trainable
stages), ImageNet-1k V1 backbone (no COCO-pretrained weights anywhere), trained on COCO-2017 train MINUS every image that is (a) in a referring
val / test split (RefCOCOg UMD val / test, RefCOCO and RefCOCO+ val / testA / testB) or (b) in our CAL / DEV roles of any of the three datasets.
Off-the-shelf COCO detectors have seen those images (COCO train2014 ⊃ RefCOCO*), which is why the proposer is rebuilt (MAttNet practice,
extended to our development roles).
    python scripts/vl1_21_detector.py exclusions                       (CPU; writes outputs/VL1_21/exclusions.json)
    torchrun --nproc_per_node 4 scripts/vl1_21_detector.py train       (resumes from last.pt; 12 epochs, 1x schedule)
    python scripts/vl1_21_detector.py eval                             (COCO val2017 bbox AP gate + proposals / recall on DEV, CAL, val)
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_21"); CK = Path("/projects/EEG-foundation-model/yinghao/models/vl1_21_detector")
COCO = Path("/projects/common/coco"); TRAIN_ANN = COCO / "annotations" / "instances_train2017.json"; VAL_ANN = COCO / "annotations" / "instances_val2017.json"
EPOCHS, LR_STEPS, BATCH_PER_GPU, BASE_LR, WD, WARMUP = 12, (8, 11), int(os.environ.get("VL_DET_BPG", 4)), 0.02, 1e-4, 1000
# VL_DET_BPG: images per GPU (default 4 = the frozen 4 x 4).  1 GPU x 16 keeps total batch 16 and lr 0.02; the backbone uses FrozenBatchNorm2d,
# so the per-GPU split does not change the computation (procedural note in reports/VL1/VL1_21_CLEAN_PROPOSER_FROZEN_20261009.md).
DATASETS = ("refcocog", "refcoco", "refcoco+")


def roles_of(dataset: str):
    os.environ["VL_DATASET"] = dataset
    from vcs_vl import refcocog as RG
    RG = importlib.reload(RG); scenes, _ = RG.load_scenes(resolve_paths=False); return RG, scenes, RG.dev_roles(scenes)


def exclusions(a) -> int:
    ex, per = set(), {}
    for d in DATASETS:
        _, scenes, role = roles_of(d)
        ids = {s.image_id for s in scenes if role[s.image_id] in ("CAL", "DEV", "VAL_OFFICIAL", "TEST_CLOSED")}
        per[d] = {r: sum(1 for s in scenes if role[s.image_id] == r) for r in ("CAL", "DEV", "VAL_OFFICIAL", "TEST_CLOSED")}; ex |= ids
    tr = json.load(open(TRAIN_ANN)); train_ids = {im["id"] for im in tr["images"]}
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"rule": "exclude every image in CAL / DEV / VAL_OFFICIAL / TEST_CLOSED of RefCOCOg-UMD, RefCOCO-UNC, RefCOCO+-UNC", "per_dataset_roles": per,
           "n_excluded": len(ex), "n_excluded_in_coco2017_train": len(ex & train_ids), "n_coco2017_train": len(train_ids), "excluded_ids": sorted(ex)}
    json.dump(res, open(OUT / "exclusions.json", "w")); print({k: v for k, v in res.items() if k != "excluded_ids"}); return 0


class CocoTrain(torch.utils.data.Dataset):
    def __init__(self, excluded: set[int]):
        d = json.load(open(TRAIN_ANN)); anns = {}
        for x in d["annotations"]:
            if not x["iscrowd"] and x["bbox"][2] > 1 and x["bbox"][3] > 1:
                anns.setdefault(x["image_id"], []).append(x)
        self.items = [(im["file_name"], anns[im["id"]]) for im in sorted(d["images"], key=lambda z: z["id"]) if im["id"] not in excluded and im["id"] in anns]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, xs = self.items[i]; im = Image.open(COCO / "train2017" / fn).convert("RGB"); W = im.width
        boxes = torch.tensor([[b["bbox"][0], b["bbox"][1], b["bbox"][0] + b["bbox"][2], b["bbox"][1] + b["bbox"][3]] for b in xs], dtype=torch.float32)
        labels = torch.tensor([b["category_id"] for b in xs], dtype=torch.int64); x = TF.to_tensor(im)
        if torch.rand(()) < 0.5:
            x = x.flip(-1); boxes = torch.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], 1)
        return x, {"boxes": boxes, "labels": labels}


def build_model():
    os.environ.setdefault("TORCH_HOME", "/projects/EEG-foundation-model/yinghao/models/torch_home")
    from torchvision.models import ResNet50_Weights
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, num_classes=91)


def train(a) -> int:
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_rank); dev = torch.device("cuda", lr_rank); torch.manual_seed(20261009 + rank)
    excluded = set(json.load(open(OUT / "exclusions.json"))["excluded_ids"]); ds = CocoTrain(excluded)
    ck_dir = CK / "smoke" if a.smoke else CK; n_epochs = 1 if a.smoke else EPOCHS
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=20261009)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_PER_GPU, sampler=sampler, num_workers=a.workers, collate_fn=lambda b: tuple(zip(*b)), pin_memory=True, persistent_workers=True)
    model = build_model().to(dev); model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_rank])
    lr = BASE_LR * BATCH_PER_GPU * world / 16
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=WD); scaler = torch.amp.GradScaler("cuda")
    it_per_epoch = len(dl); start, step = 0, 0; ck_dir.mkdir(parents=True, exist_ok=True)
    if (ck_dir / "last.pt").exists():
        ck = torch.load(ck_dir / "last.pt", map_location="cpu", weights_only=False)
        model.module.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"]); start, step = ck["epoch"], ck["step"]
        if rank == 0:
            print(f"resumed at epoch {start} step {step}", flush=True)
    def lr_at(s):
        e = s / it_per_epoch; f = 0.1 ** sum(e >= m for m in LR_STEPS)
        return lr * f * (min(1.0, 0.001 + (1 - 0.001) * s / WARMUP) if s < WARMUP else 1.0)
    if rank == 0:
        print(f"train images {len(ds)} (excluded {len(excluded)}), world {world}, batch {BATCH_PER_GPU * world}, lr {lr}, iters/epoch {it_per_epoch}", flush=True)
    for ep in range(start, n_epochs):
        sampler.set_epoch(ep); model.train(); t0 = time.time(); run = []
        for i, (xs, ts) in enumerate(dl):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            xs = [x.to(dev, non_blocking=True) for x in xs]; ts = [{k: v.to(dev) for k, v in t.items()} for t in ts]
            with torch.autocast("cuda", dtype=torch.float16):
                losses = model(xs, ts); loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep} iter {i}: { {k: float(v) for k, v in losses.items()} }")
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step += 1; run.append(float(loss))
            if a.smoke and i >= 30:
                break
            if rank == 0 and (i % 200 == 0 or a.smoke):
                print(f"ep {ep} it {i}/{it_per_epoch} loss {np.mean(run[-200:]):.4f} lr {lr_at(step):.5f} {time.time() - t0:.0f}s", flush=True)
        if rank == 0:
            torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": ep + 1, "step": step}, ck_dir / "last.pt.tmp")
            os.replace(ck_dir / "last.pt.tmp", ck_dir / "last.pt"); print(f"epoch {ep} done {time.time() - t0:.0f}s mean loss {np.mean(run):.4f}", flush=True)
        dist.barrier()
    if rank == 0:
        torch.save({"model": model.module.state_dict(), "epochs": n_epochs, "smoke": a.smoke}, ck_dir / "final.pt"); print("final saved", ck_dir, flush=True)
    dist.destroy_process_group(); return 0


@torch.no_grad()
def detect(model, paths, dev, bs=8):
    out = []
    for s in range(0, len(paths), bs):
        xs = [TF.to_tensor(Image.open(p).convert("RGB")).to(dev) for p in paths[s:s + bs]]
        with torch.autocast("cuda", dtype=torch.float16):
            r = model(xs)
        out += [torch.cat([d["boxes"].float(), d["scores"].float()[:, None], d["labels"].float()[:, None]], 1).cpu() for d in r]
    return out


def iou_one(b, boxes):
    x1 = np.maximum(b[0], boxes[:, 0]); y1 = np.maximum(b[1], boxes[:, 1]); x2 = np.minimum(b[2], boxes[:, 2]); y2 = np.minimum(b[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)


def evaluate(a) -> int:
    from pycocotools.coco import COCO as CC
    from pycocotools.cocoeval import COCOeval
    dev = torch.device("cuda"); model = build_model(); model.load_state_dict(torch.load(CK / "final.pt", map_location="cpu", weights_only=False)["model"])
    model = model.to(dev).eval(); res = {}
    gt = CC(str(VAL_ANN)); ids = sorted(gt.getImgIds()); dets = detect(model, [str(COCO / "val2017" / gt.loadImgs(i)[0]["file_name"]) for i in ids], dev); js = []
    for i, d in zip(ids, dets):
        for x1, y1, x2, y2, sc, lb in d.tolist():
            js.append({"image_id": i, "category_id": int(lb), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
    E = COCOeval(gt, gt.loadRes(js), "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    res["coco_val2017_bbox_AP"] = float(E.stats[0]); res["coco_val2017_bbox_AP50"] = float(E.stats[1]); res["gate_AP_ge_0.33"] = bool(E.stats[0] >= 0.33)
    prop_dir = OUT / "proposals"; prop_dir.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:
        RG, scenes, role = roles_of(d)
        for r in ("CAL", "DEV", "VAL_OFFICIAL"):
            sc = [s for s in scenes if role[s.image_id] == r and len(s.referred) >= 1]
            dd = detect(model, [RG.coco_path(s.image_id) for s in sc], dev)
            torch.save({s.image_id: x for s, x in zip(sc, dd)}, prop_dir / f"{d.replace('+', 'plus')}_{r}.pt")
            hits = {k: [] for k in (10, 20, 50, 100)}
            for s, x in zip(sc, dd):
                bx = x[:, :4].numpy()
                for o in s.referred:
                    t = [o.box_xywh[0], o.box_xywh[1], o.box_xywh[0] + o.box_xywh[2], o.box_xywh[1] + o.box_xywh[3]]
                    iou = iou_one(t, bx) if len(bx) else np.zeros(0)
                    for k in hits:
                        hits[k].append(float(len(iou[:k]) > 0 and iou[:k].max() >= 0.5))
            res[f"{d}/{r}"] = {"n_images": len(sc), "n_targets": len(hits[100]), **{f"recall@{k}_iou50": float(np.mean(v)) for k, v in hits.items()},
                               "mean_detections": float(np.mean([len(x) for x in dd]))}
            print(d, r, res[f"{d}/{r}"], flush=True)
    rep = REPO / "reports" / "VL1"; json.dump(res, open(rep / "VL1_21_detector_eval.json", "w"), indent=1); print(json.dumps(res, indent=1)); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exclusions"); t = sub.add_parser("train"); t.add_argument("--workers", type=int, default=8); t.add_argument("--smoke", action="store_true"); sub.add_parser("eval")
    a = ap.parse_args(); return {"exclusions": exclusions, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
