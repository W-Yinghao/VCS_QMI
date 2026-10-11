"""Data steward for the r6 / r7 packages.

Owner authorization, 2026-10-11: "这个包和上个包的数据全都下载到/projects/EEG-foundation-model" (download all data of both packages to
/projects/EEG-foundation-model), which covers datasets whose terms R6-03 marked "needs owner acceptance" (AV-MNIST: ESC-50 CC BY-NC; CREMA-D:
ODbL + registration request) — research / non-commercial use only, recorded per item.

Manifest-driven, resumable downloads with a free-space floor (never below FLOOR_GB free on the shared filesystem).  Published checksums are
verified when they exist (Zenodo md5, HF / LFS sha256); otherwise sha256 is recorded.  Closed items (official test / held-out splits) are stored
under closed/ and only hashed — never extracted, opened or parsed.  Archives are listed before extraction and refused if they contain absolute
or parent paths; code shipped inside downloads is never executed.

    python scripts/data_fetch.py manifest                 # write data/DATA_MANIFEST.json (keeps recorded status fields)
    python scripts/data_fetch.py run --group fixtures     # one priority group; groups: r6, fixtures, tierA, tierB
    python scripts/data_fetch.py status
"""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/projects/EEG-foundation-model/yinghao/data"); MODELS = Path("/projects/EEG-foundation-model/yinghao/models")
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "DATA_MANIFEST.json"; LOG = Path("/home/infres/yinwang/CS_QMI/slurm_logs/data_fetch.log")
FLOOR_GB = 1200; WGET = "/home/infres/yinwang/anaconda3/bin/wget"; LOCK = threading.Lock(); TOUCHED: set[str] = set()   # ids this process updates
OWNER_AUTH = "owner 2026-10-11: download all r6 / r7 package data to /projects/EEG-foundation-model"
NC_AUTH = OWNER_AUTH + "; covers terms marked 'needs owner acceptance' — research / non-commercial use only"
EVAL_ONLY = "evaluation-only benchmark: never used for training or selection"


def gdrive(fid: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def http(url, name, md5=None, sha256=None):
    return {"url": url, "name": name, "md5": md5, "sha256": sha256}


# ---------------------------------------------------------------------------------------------------------------- items
ITEMS = [
    # ---- r6 ------------------------------------------------------------------------------------------------------
    {"id": "zurich_summer", "group": "r6", "kind": "http", "dest": DATA / "zurich_summer_5914759", "extract": True, "licence": "CC BY 4.0",
     "files": [http("https://zenodo.org/api/records/5914759/files/Zurich_dataset_v1.0.zip/content", "Zurich_dataset_v1.0.zip", md5="c79a08ec33cb867d155931f5201e33a0")]},
    {"id": "bf_shg_trainval", "group": "r6", "kind": "local", "dest": DATA / "bf_shg_4550300", "licence": "CC BY 4.0",
     "note": "train / validation1 / validation2 downloaded 2026-10-11 by the main session (md5 verified); test is the separate closed item"},
    {"id": "bf_shg_test_closed", "group": "r6", "kind": "http", "dest": DATA / "bf_shg_4550300" / "closed", "closed": True, "extract": False, "licence": "CC BY 4.0",
     "files": [http("https://zenodo.org/api/records/4550300/files/test.zip/content", "test.zip", md5="caa9ae81837a41664ed7b61df81e96fa")]},
    {"id": "av_mnist", "group": "r6", "kind": "http", "dest": DATA / "av_mnist_multibench", "extract": True, "auth": NC_AUTH,
     "licence": "MNIST CC BY-SA 3.0; FSDD CC BY-SA 4.0; ESC-50 noise CC BY-NC 3.0 (non-commercial); MultiBench code MIT",
     "files": [http(gdrive("1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp"), "avmnist.tar.gz")]},
    {"id": "crema_d", "group": "r6", "kind": "lfs_media", "dest": DATA / "crema_d", "auth": NC_AUTH, "dirs": ["AudioWAV", "VideoFlash"],
     "repo": "https://github.com/CheyneyComputerScience/CREMA-D.git", "media": "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/",
     "licence": "ODbL 1.0 (database) + DbCL 1.0 (contents); authors request a registration form (owner-directed download)",
     "note": "AudioMP3 skipped (derived from AudioWAV); every file verified against its LFS pointer sha256"},
    {"id": "epfl_rgb_nir", "group": "r6", "kind": "local", "dest": DATA / "epfl_rgb_nir", "licence": "no licence text; source accepted by the owner (P60, 2026-09-27)",
     "note": "copied by the main session from /home/infres/yinwang/CS_QMI/data/rgb_nir (nirscene1.zip sha256 7c465a23…f746bc)"},
    # ---- r7: fixtures first (requested by the R7-20 fixtures agent) ----------------------------------------------
    {"id": "llava_v1_5_7b", "group": "fixtures", "kind": "hf", "repo": "liuhaotian/llava-v1.5-7b", "type": "model", "dest": MODELS / "llava-v1.5-7b", "licence": "Llama 2 community licence"},
    {"id": "clip_vit_l14_336", "group": "fixtures", "kind": "hf", "repo": "openai/clip-vit-large-patch14-336", "type": "model", "dest": MODELS / "clip-vit-large-patch14-336",
     "ignore": ["*.h5", "*.msgpack"], "licence": "MIT (OpenAI CLIP)"},
    {"id": "gqa_lmms_val", "group": "fixtures", "kind": "hf", "repo": "lmms-lab/GQA", "type": "dataset", "dest": DATA / "gqa_lmms",
     "allow": ["val_balanced_instructions/*", "val_balanced_images/*", "README.md"], "licence": "GQA CC BY 4.0", "note": "no testdev / test / challenge / submission files"},
    {"id": "textvqa_lmms_val", "group": "fixtures", "kind": "hf", "repo": "lmms-lab/textvqa", "type": "dataset", "dest": DATA / "textvqa_lmms",
     "allow": ["data/validation-*", "README.md"], "licence": "TextVQA CC BY 4.0 (annotations); images Open Images CC BY 2.0", "note": "no test shards"},
    {"id": "pope_lmms", "group": "fixtures", "kind": "hf", "repo": "lmms-lab/POPE", "type": "dataset", "dest": DATA / "pope_lmms", "allow": ["data/*", "README.md"],
     "licence": "POPE MIT; COCO images CC BY 4.0", "note": EVAL_ONLY},
    {"id": "mhal_detect", "group": "fixtures", "kind": "git", "repo": "https://github.com/hendryx-scale/mhal-detect.git", "dest": DATA / "mhal_detect",
     "licence": "CC BY-NC 4.0 (annotations); owner-directed download", "auth": NC_AUTH},
    # ---- r7 Tier A: shared images and annotations ------------------------------------------------------------------
    {"id": "coco2014", "group": "tierA", "kind": "http", "dest": DATA / "coco", "extract": True, "licence": "COCO annotations CC BY 4.0; images Flickr terms",
     "files": [http("http://images.cocodataset.org/zips/train2014.zip", "train2014.zip"), http("http://images.cocodataset.org/zips/val2014.zip", "val2014.zip"),
               http("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
               http("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", "annotations_trainval2014.zip"),
               http("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
               http("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip", "caption_datasets.zip")],
     "note": "test2014 / test2017 not downloaded (no r7 protocol uses them)"},
    {"id": "flickr30k", "group": "tierA", "kind": "hf", "repo": "nlphuji/flickr30k", "type": "dataset", "dest": DATA / "flickr30k",
     "licence": "Flickr30k: official distribution by request form (research use); HF mirror used under the owner's download authorization", "auth": OWNER_AUTH},
    {"id": "cub_200_2011", "group": "tierA", "kind": "http", "dest": DATA / "cub_200_2011", "extract": True, "licence": "CUB-200-2011 (Caltech; research use)",
     "files": [http("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1", "CUB_200_2011.tgz"),
               http(gdrive("0B0ywwgffWnLLZW9uVHNjb2JmNlE"), "cvpr2016_cub.tar.gz")], "note": "Reed et al. CVPR 2016 captions for PCME's protocol"},
    {"id": "gqa", "group": "tierA", "kind": "http", "dest": DATA / "gqa", "extract": True, "licence": "CC BY 4.0",
     "files": [http("https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip", "images.zip"), http("https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip", "questions1.2.zip"),
               http("https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip", "sceneGraphs.zip")],
     "close_after_extract": ["test_*", "testdev_*", "challenge_*", "submission_*"],
     "note": "test-dev (read once per pre-registration, atlas DS01) / test / challenge / submission question files moved to closed/ unopened"},
    {"id": "textvqa", "group": "tierA", "kind": "http", "dest": DATA / "textvqa", "extract": True, "licence": "CC BY 4.0 (annotations); Open Images CC BY 2.0",
     "files": [http("https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip", "train_val_images.zip"),
               http("https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json", "TextVQA_0.5.1_train.json"),
               http("https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json", "TextVQA_0.5.1_val.json")]},
    {"id": "textvqa_test_closed", "group": "tierA", "kind": "http", "dest": DATA / "textvqa" / "closed", "closed": True, "extract": False, "licence": "CC BY 4.0",
     "files": [http("https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip", "test_images.zip"),
               http("https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json", "TextVQA_0.5.1_test.json")]},
    {"id": "chartqa", "group": "tierA", "kind": "hf", "repo": "ahmed-masry/ChartQA", "type": "dataset", "dest": DATA / "chartqa", "licence": "GPL-3.0",
     "close_after_extract": ["*test*"]},
    {"id": "scienceqa", "group": "tierA", "kind": "hf", "repo": "derek-thomas/ScienceQA", "type": "dataset", "dest": DATA / "scienceqa", "licence": "CC BY-SA 4.0",
     "close_after_extract": ["*test*"]},
    {"id": "mmbench", "group": "tierA", "kind": "hf", "repo": "lmms-lab/MMBench", "type": "dataset", "dest": DATA / "mmbench", "licence": "MMBench (OpenCompass; research use)",
     "close_after_extract": ["*test*"]},
    {"id": "pope", "group": "tierA", "kind": "git", "repo": "https://github.com/AoiDragon/POPE.git", "dest": DATA / "pope", "licence": "MIT", "note": EVAL_ONLY},
    {"id": "hallusionbench", "group": "tierA", "kind": "git+http", "repo": "https://github.com/tianyi-lab/HallusionBench.git", "dest": DATA / "hallusionbench", "extract": True,
     "files": [http(gdrive("1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0"), "hallusion_bench.zip")], "licence": "BSD-3-Clause (repo)", "note": EVAL_ONLY},
    {"id": "amber", "group": "tierA", "kind": "git+http", "repo": "https://github.com/junyangwang0410/AMBER.git", "dest": DATA / "amber", "extract": True,
     "files": [http(gdrive("1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"), "AMBER_images.zip")], "licence": "Apache-2.0 (repo)", "note": EVAL_ONLY},
    {"id": "snli_ve", "group": "tierA", "kind": "git+http", "repo": "https://github.com/necla-ml/SNLI-VE.git", "dest": DATA / "snli_ve", "extract": True,
     "files": [http("https://nlp.stanford.edu/projects/snli/snli_1.0.zip", "snli_1.0.zip")], "licence": "SNLI CC BY-SA 4.0; SNLI-VE BSD-3-Clause",
     "note": "SNLI-VE jsonl is generated by the repo's script from SNLI + Flickr30k: review before running (not run here)"},
    {"id": "cxc", "group": "tierA", "kind": "git", "repo": "https://github.com/google-research-datasets/Crisscrossed-Captions.git", "dest": DATA / "cxc", "licence": "CC BY 4.0", "note": EVAL_ONLY},
    {"id": "eccv_caption", "group": "tierA", "kind": "git", "repo": "https://github.com/naver-ai/eccv-caption.git", "dest": DATA / "eccv_caption", "licence": "MIT",
     "note": EVAL_ONLY + "; the gdrive extras folder (pm_*.json, mturk_parsed.csv) needs gdown (tools env)"},
    {"id": "sugarcrepe_pp", "group": "tierA", "kind": "hf", "repo": "Aman-J/SugarCrepe_pp", "type": "dataset", "dest": DATA / "sugarcrepe_pp", "licence": "CC BY 4.0",
     "note": EVAL_ONLY + "; source confirmed from the authors' repo github.com/Sri-Harsha/scpp"},
    {"id": "pcmepp", "group": "tierA", "kind": "git+http", "repo": "https://github.com/naver-ai/pcmepp.git", "dest": DATA / "pcmepp", "extract": True,
     "files": [http("https://github.com/naver-ai/pcmepp/releases/download/v0.1.0/coco_annotations.tar.gz", "coco_annotations.tar.gz")], "licence": "MIT",
     "note": "official PCME++ COCO weights not released (README); trained by the D4 protocol if chosen"},
    {"id": "clevr_v1", "group": "tierA", "kind": "http", "dest": DATA / "clevr_v1", "extract": True, "licence": "CC BY 4.0",
     "files": [http("https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip", "CLEVR_v1.0.zip")], "close_after_extract": ["*test*"]},
    # ---- r7 Tier A: D3 prompt / evaluator sets ---------------------------------------------------------------------
    {"id": "geneval", "group": "tierA", "kind": "git+http", "repo": "https://github.com/djghosh13/geneval.git", "dest": DATA / "geneval", "extract": False, "licence": "MIT",
     "files": [http("https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth",
                    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth")], "note": EVAL_ONLY},
    {"id": "t2i_compbench", "group": "tierA", "kind": "git", "repo": "https://github.com/Karine-Huang/T2I-CompBench.git", "dest": DATA / "t2i_compbench", "licence": "MIT",
     "note": EVAL_ONLY + "; original and ++ prompt sets kept as separate versions"},
    {"id": "conceptmix", "group": "tierA", "kind": "git", "repo": "https://github.com/princetonvisualai/ConceptMix.git", "dest": DATA / "conceptmix", "licence": "see repo", "note": EVAL_ONLY},
    {"id": "partiprompts", "group": "tierA", "kind": "git", "repo": "https://github.com/google-research/parti.git", "dest": DATA / "parti", "licence": "Apache-2.0", "note": EVAL_ONLY},
    {"id": "genai_bench", "group": "tierA", "kind": "hf", "repo": "BaiqiL/GenAI-Bench", "type": "dataset", "dest": DATA / "genai_bench", "licence": "Apache-2.0", "note": EVAL_ONLY},
    {"id": "hpdv2_test_closed", "group": "tierA", "kind": "hf", "repo": "zhwang/HPDv2", "type": "dataset", "dest": DATA / "hpdv2" / "closed", "closed": True,
     "licence": "HPDv2 (see repo); official held-out human-preference test split", "note": "closed: hashed only until a frozen D3 protocol opens it"},
    # ---- r7 Tier A: model weights named in the D1–D4 plans --------------------------------------------------------
    {"id": "qwen2_5_vl_7b_instruct", "group": "tierA", "kind": "hf", "repo": "Qwen/Qwen2.5-VL-7B-Instruct", "type": "model", "dest": MODELS / "Qwen2.5-VL-7B-Instruct", "licence": "Apache-2.0"},
    {"id": "vqascore_clip_flant5_xxl", "group": "tierA", "kind": "hf", "repo": "zhiqiulin/clip-flant5-xxl", "type": "model", "dest": MODELS / "clip-flant5-xxl", "licence": "Apache-2.0"},
    {"id": "clip_vit_l14", "group": "tierA", "kind": "hf", "repo": "openai/clip-vit-large-patch14", "type": "model", "dest": MODELS / "clip-vit-large-patch14",
     "ignore": ["*.h5", "*.msgpack"], "licence": "MIT (OpenAI CLIP)"},
    {"id": "prolip_b16", "group": "tierA", "kind": "hf", "repo": "SanghyukChun/ProLIP-ViT-B-16-DC-1B-12_8B", "type": "model", "dest": MODELS / "ProLIP-ViT-B-16-DC-1B-12_8B", "licence": "MIT"},
    {"id": "hpsv2_ckpt", "group": "tierA", "kind": "hf", "repo": "xswu/HPSv2", "type": "space", "dest": MODELS / "hpsv2", "allow": ["HPS_v2_compressed.pt"], "licence": "Apache-2.0 (HPSv2 repo)"},
    # ---- r7 Tier B: D6 official features + annotations (after Tier A) ---------------------------------------------
    {"id": "qvhighlights", "group": "tierB", "kind": "git+http", "repo": "https://github.com/jayleicn/moment_detr.git", "dest": DATA / "qvhighlights", "extract": True,
     "files": [http(gdrive("1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9"), "moment_detr_features.tar.gz")], "licence": "annotations CC BY-NC-SA 4.0; code MIT", "auth": NC_AUTH,
     "note": "official SlowFast + CLIP features (Moment-DETR); raw videos are Tier C"},
]

TIER_C = [  # listed for the owner, never started here
    {"id": "datacomp_small", "size": "~12.8M samples; ~450 GB of images (img2dataset scrape) + ~3 GB metadata", "why": "web-scale scraping; > 100 GB"},
    {"id": "cc3m", "size": "~3.3M URLs; ~130–400 GB depending on resize", "why": "web-scale scraping; link rot"},
    {"id": "recap_datacomp_subset", "size": "captions small; images need the DataComp scrape", "why": "depends on DataComp images"},
    {"id": "howto100m", "size": "features ~1–2 TB; videos tens of TB", "why": "> 100 GB"},
    {"id": "raw_videos (QVHighlights, Charades, ActivityNet, HiREST, MomentSeeker)", "size": "~50 GB – >500 GB each", "why": "raw YouTube / video corpora"},
]
GATED = [
    {"id": "sp_docvqa", "action": "owner registers at https://rrc.cvc.uab.es (Document VQA, Task 1 single-page), accepts the terms, then shares the download links / credentials"},
    {"id": "ego4d_nlq", "action": "owner signs the Ego4D licence at https://ego4ddataset.com, receives AWS credentials by e-mail (~48 h), then the ego4d CLI downloads the NLQ annotations + features"},
]


# ---------------------------------------------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    with LOCK:
        with open(LOG, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def free_gb() -> float:
    return shutil.disk_usage(DATA).free / 1e9


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> int:
    """List first; refuse absolute / parent paths and links leaving dest; never run anything."""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if any(n.startswith("/") or ".." in Path(n).parts for n in names):
                raise RuntimeError(f"unsafe paths in {archive.name}")
            z.extractall(dest)
        return len(names)
    with tarfile.open(archive) as t:
        members = t.getmembers()
        for m in members:
            if m.name.startswith("/") or ".." in Path(m.name).parts or ((m.issym() or m.islnk()) and (m.linkname.startswith("/") or ".." in Path(m.linkname).parts)):
                raise RuntimeError(f"unsafe member {m.name} in {archive.name}")
        t.extractall(dest, filter="data") if sys.version_info >= (3, 12) else t.extractall(dest)
    return len(members)


def close_matching(root: Path, patterns: list[str]) -> list[str]:
    """Move files matching the patterns (by name) into root/closed/ without opening them."""
    moved = []; closed = root / "closed"
    for p in list(root.rglob("*")):
        if p.is_file() and "closed" not in p.relative_to(root).parts and any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
            target = closed / p.relative_to(root); target.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(p), target); moved.append(str(target.relative_to(root)))
    return moved


# ---------------------------------------------------------------------------------------------------------------- handlers
def do_http(item, rec) -> None:
    dest = Path(item["dest"]); zdir = dest if item.get("closed") else dest / "zips"; zdir.mkdir(parents=True, exist_ok=True); rec["files"] = []
    for f in item["files"]:
        out = zdir / f["name"]
        subprocess.run([WGET, "-c", "-q", "--tries=10", "--timeout=60", "-O", str(out), f["url"]], check=True)
        entry = {"name": f["name"], "url": f["url"], "bytes": out.stat().st_size, "sha256": digest(out, "sha256")}
        if f.get("md5"):
            got = digest(out, "md5"); entry["md5"] = got; entry["md5_published_ok"] = got == f["md5"]
            if got != f["md5"]:
                raise RuntimeError(f"md5 mismatch for {f['name']}")
        if item.get("extract") and not item.get("closed") and (f["name"].endswith((".zip", ".tgz", ".tar.gz", ".tar"))):
            entry["extracted_entries"] = safe_extract(out, dest)
        rec["files"].append(entry); log(f"  {item['id']}: {f['name']} {entry['bytes']} B sha256 {entry['sha256'][:16]}")
    if item.get("close_after_extract"):
        rec["moved_to_closed"] = close_matching(dest, item["close_after_extract"])


def do_hf(item, rec) -> None:
    from huggingface_hub import HfApi, snapshot_download
    os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
    api = HfApi(); kind = item["type"]
    info = {"model": api.model_info, "dataset": api.dataset_info, "space": api.space_info}[kind](item["repo"], files_metadata=True)
    dest = Path(item["dest"]); dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(item["repo"], repo_type=kind, revision=info.sha, local_dir=str(dest), allow_patterns=item.get("allow"), ignore_patterns=item.get("ignore"))
    files = []
    for s in info.siblings:
        p = dest / s.rfilename
        if p.exists():
            lfs = getattr(s, "lfs", None); pub = getattr(lfs, "sha256", None) if lfs else None
            files.append({"name": s.rfilename, "bytes": p.stat().st_size, "sha256_published": pub, "sha256_ok": (digest(p, "sha256") == pub) if pub else None})
    rec.update({"revision": info.sha, "files": files, "bytes": sum(f["bytes"] for f in files)})
    if any(f["sha256_ok"] is False for f in files):
        raise RuntimeError("HF sha256 mismatch")
    if item.get("close_after_extract"):
        rec["moved_to_closed"] = close_matching(dest, item["close_after_extract"])


def do_git(item, rec) -> None:
    dest = Path(item["dest"]) / "repo"
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", "-q", item["repo"], str(dest)], check=True)
    rec["commit"] = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    rec["bytes_repo"] = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())


def do_lfs_media(item, rec) -> None:
    dest = Path(item["dest"]); repo = dest / "repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", "-q", item["repo"], str(repo)], check=True)
    jobs = []
    for d in item["dirs"]:
        for ptr in sorted((repo / d).iterdir()):
            lines = dict(l.split(" ", 1) for l in ptr.read_text().splitlines() if " " in l)
            jobs.append((d, ptr.name, lines["oid"].split(":", 1)[1], int(lines["size"])))
    bad = []

    def one(job):
        d, name, oid, size = job; out = dest / "media" / d / name; out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == size and digest(out, "sha256") == oid:
            return True
        subprocess.run([WGET, "-c", "-q", "--tries=10", "--timeout=60", "-O", str(out), item["media"] + f"{d}/{name}"], check=False)
        ok = out.exists() and out.stat().st_size == size and digest(out, "sha256") == oid
        if not ok:
            bad.append(f"{d}/{name}")
        return ok
    with ThreadPoolExecutor(6) as ex:
        n_ok = sum(ex.map(one, jobs))
    rec.update({"files_expected": len(jobs), "files_verified": n_ok, "failed": bad[:50], "n_failed": len(bad),
                "bytes": sum(s for _, _, _, s in jobs), "commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()})
    if bad:
        raise RuntimeError(f"{len(bad)} CREMA-D files failed LFS verification (see failed list; GitLab mirror is the fallback)")


def do_local(item, rec) -> None:
    p = Path(item["dest"]); rec["present"] = p.exists(); rec["bytes"] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


HANDLERS = {"http": do_http, "hf": do_hf, "git": do_git, "lfs_media": do_lfs_media, "local": do_local}


def run_item(item, manifest) -> None:
    rec = manifest["items"].setdefault(item["id"], {}); TOUCHED.add(item["id"])
    if rec.get("status") == "done":
        return
    if free_gb() < FLOOR_GB + 10:
        rec["status"] = "deferred_free_space"; log(f"DEFER {item['id']}: free {free_gb():.0f} GB < floor"); return
    rec.update({"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}); save(manifest)
    log(f"START {item['id']} ({item['kind']})")
    try:
        if item["kind"] == "git+http":
            do_git(item, rec); do_http(item, rec)
        else:
            HANDLERS[item["kind"]](item, rec)
        rec["status"] = "done"; log(f"DONE {item['id']}")
    except Exception as e:  # noqa: BLE001 — recorded, never silently filled
        rec["status"] = "failed"; rec["error"] = f"{type(e).__name__}: {e}"[:400]; log(f"FAIL {item['id']}: {rec['error']}")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save(manifest)


def static_fields(item):
    out = {k: (str(v) if isinstance(v, Path) else v) for k, v in item.items() if k not in ("files",)}
    out["closed"] = bool(item.get("closed")); out["auth"] = item.get("auth", OWNER_AUTH)
    if "files" in item:
        out["sources"] = [{"url": f["url"], "name": f["name"], "md5_published": f.get("md5")} for f in item["files"]]
    return out


def load():
    return json.load(open(MANIFEST)) if MANIFEST.exists() else {"items": {}}


def save(manifest):
    """Merge this process's item records into the on-disk manifest under an exclusive file lock (several runner processes share it)."""
    with LOCK:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True); lockf = open(str(MANIFEST) + ".lock", "w"); fcntl.flock(lockf, fcntl.LOCK_EX)
        disk = load()
        for k in TOUCHED:
            disk["items"][k] = manifest["items"][k]
        manifest["items"] = {**disk["items"], **{k: manifest["items"][k] for k in TOUCHED}}
        manifest["policy"] = {"floor_free_gb": FLOOR_GB, "closed": "official test / held-out splits: stored under closed/, hashed only, never extracted / opened / parsed",
                              "evaluation_only": EVAL_ONLY, "authorization": OWNER_AUTH}
        manifest["catalogue"] = {i["id"]: static_fields(i) for i in ITEMS}; manifest["tier_c_not_started"] = TIER_C; manifest["account_gated_not_started"] = GATED
        tmp = MANIFEST.with_suffix(f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False)); tmp.replace(MANIFEST)
        fcntl.flock(lockf, fcntl.LOCK_UN); lockf.close()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["manifest", "run", "status"]); ap.add_argument("--group", default=None)
    ap.add_argument("--ids", nargs="*", default=None); ap.add_argument("--jobs", type=int, default=2); a = ap.parse_args(); manifest = load()
    if a.cmd == "manifest":
        save(manifest); print(f"wrote {MANIFEST} ({len(ITEMS)} items)"); return 0
    if a.cmd == "status":
        for i in ITEMS:
            r = manifest["items"].get(i["id"], {}); print(f"{i['group']:9s} {i['id']:28s} {r.get('status', 'pending'):22s} {r.get('error', '')[:80]}")
        print(f"free {free_gb():.0f} GB"); return 0
    todo = [i for i in ITEMS if (a.group is None or i["group"] == a.group) and (a.ids is None or i["id"] in a.ids)]
    log(f"RUN group={a.group} ids={a.ids} items={len(todo)} jobs={a.jobs} free={free_gb():.0f} GB")
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda it: run_item(it, manifest), todo))
    log(f"RUN END group={a.group} free={free_gb():.0f} GB"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
