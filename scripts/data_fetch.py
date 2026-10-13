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
     "files": [http(gdrive("1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp"), "avmnist.tar.gz")], "close_after_extract": ["test_*"],
     "note": "official MultiBench test arrays (image/audio test_data.npy, test_labels.npy, incl. the avmnist_temp copy) closed 2026-10-12; they had been left open by "
             "the first extraction (no close pattern); the data steward never read them and no repo script references them (the r6 GAIN pilot reads its own train-only extraction in extracted/avmnist)"},
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
     "note": "test2014 / test2017 not downloaded (no r7 protocol uses them); the Karpathy 5k test split sits inside dataset_coco.json and val2014 and cannot be "
             "separated without parsing: " + EVAL_ONLY},
    {"id": "flickr30k", "group": "tierA", "kind": "hf", "repo": "nlphuji/flickr30k", "type": "dataset", "dest": DATA / "flickr30k",
     "licence": "Flickr30k: official distribution by request form (research use); HF mirror used under the owner's download authorization", "auth": OWNER_AUTH,
     "note": "the Karpathy 1k test split sits inside flickr_annotations_30k.csv and the image zip and cannot be separated without parsing: " + EVAL_ONLY},
    {"id": "cub_200_2011", "group": "tierA", "kind": "http", "dest": DATA / "cub_200_2011", "extract": True, "licence": "CUB-200-2011 (Caltech; research use)",
     "files": [http("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1", "CUB_200_2011.tgz"),
               http(gdrive("0B0ywwgffWnLLZW9uVHNjb2JmNlE"), "cvpr2016_cub.tar.gz")],
     "note": "Reed et al. CVPR 2016 captions for PCME's protocol; the test split is defined inside mixed files (train_test_split.txt, testclasses.txt) "
             "and cannot be separated without parsing: " + EVAL_ONLY},
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
    {"id": "hallusionbench", "group": "tierA", "kind": "git", "repo": "https://github.com/tianyi-lab/HallusionBench.git", "dest": DATA / "hallusionbench",
     "licence": "BSD-3-Clause (repo)", "note": EVAL_ONLY + "; the README's Drive zip (1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0) returns 404 on 2026-10-11, images come from hallusionbench_images"},
    {"id": "hallusionbench_images", "group": "tierA", "kind": "hf", "repo": "rayguan/HallusionBench", "type": "dataset", "dest": DATA / "hallusionbench" / "hf",
     "licence": "BSD-3-Clause (repo)", "provenance": "first author's upload (Tianrui Guan, HF account 'rayguan', 2023-12-10); the README's Drive zip returns 404 "
     "(2026-10-11); source accepted by the coordinator 2026-10-12; its HallusionBench.json is byte-identical to the GitHub repo's copy at commit 744007c232c292942c7f80eb61edb2465482da31 "
     "(sha256 ca6e0fb677dd56e5fb0e2700d06c58d01f8a6ae66e0f1773b2ac48c0a095d051)", "note": EVAL_ONLY + "; same layout as hallusion_bench.zip"},
    {"id": "amber", "group": "tierA", "kind": "git+http", "repo": "https://github.com/junyangwang0410/AMBER.git", "dest": DATA / "amber", "extract": True,
     "files": [http(gdrive("1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"), "AMBER_images.zip")], "licence": "Apache-2.0 (repo)", "note": EVAL_ONLY},
    {"id": "snli_ve", "group": "tierA", "kind": "git+http", "repo": "https://github.com/necla-ml/SNLI-VE.git", "dest": DATA / "snli_ve", "extract": True,
     "files": [http("https://nlp.stanford.edu/projects/snli/snli_1.0.zip", "snli_1.0.zip")], "licence": "SNLI CC BY-SA 4.0; SNLI-VE BSD-3-Clause",
     "close_after_extract": ["*_test.*"],
     "note": "SNLI-VE jsonl is generated by the repo's script from SNLI + Flickr30k: review before running (not run here); SNLI test + Flickr30k test list closed"},
    {"id": "cxc", "group": "tierA", "kind": "git", "repo": "https://github.com/google-research-datasets/Crisscrossed-Captions.git", "dest": DATA / "cxc", "licence": "CC BY 4.0",
     "close_after_extract": ["*_test*"], "note": EVAL_ONLY + "; *_test*.csv (COCO Karpathy-test judgements) closed, *_val*.csv open"},
    {"id": "eccv_caption", "group": "tierA", "kind": "git", "repo": "https://github.com/naver-ai/eccv-caption.git", "dest": DATA / "eccv_caption", "licence": "MIT",
     "note": EVAL_ONLY + "; the gdrive extras folder (pm_*.json, mturk_parsed.csv) needs gdown (tools env)"},
    {"id": "sugarcrepe_pp", "group": "tierA", "kind": "hf", "repo": "Aman-J/SugarCrepe_pp", "type": "dataset", "dest": DATA / "sugarcrepe_pp", "licence": "CC BY 4.0",
     "note": EVAL_ONLY + "; source confirmed from the authors' repo github.com/Sri-Harsha/scpp"},
    {"id": "pcmepp", "group": "tierA", "kind": "git+http", "repo": "https://github.com/naver-ai/pcmepp.git", "dest": DATA / "pcmepp", "extract": True,
     "files": [http("https://github.com/naver-ai/pcmepp/releases/download/v0.1.0/coco_annotations.tar.gz", "coco_annotations.tar.gz")], "licence": "MIT",
     "close_after_extract": ["test_*", "testall_*"],
     "note": "official PCME++ COCO weights not released (README); trained by the D4 protocol if chosen; precomp test / testall (COCO Karpathy test) closed"},
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
    {"id": "llava_mix665k", "group": "tierA", "kind": "hf", "repo": "liuhaotian/LLaVA-Instruct-150K", "type": "dataset", "dest": DATA / "llava_mix665k",
     "allow": ["llava_v1_5_mix665k.json", "README.md"], "licence": "CC BY 4.0 (LLaVA-Instruct; GPT-generated parts also under OpenAI terms; source datasets keep their own licences)",
     "provenance": "used only to exclude LLaVA-seen GQA QA pairs before role assignment (r7 D1 manifest freeze; coordinator priority 2026-10-12)",
     "note": "LLaVA-1.5 training mixture, annotations only (no images); requested by the R7-30 D1 fork (decision OPEN-3)"},
    # ---- V7 (owner 2026-10-13: VCS_CVPR_V7_T2I_Alignment_Server package; first-round data, init and baseline weights) ---------------
    {"id": "evalmuse_train", "group": "v7", "kind": "hf", "repo": "DY-Evalab/EvalMuse", "type": "dataset", "dest": DATA / "evalmuse",
     "allow": ["README.md", "train_list.json", "prompt_t2i.json", "images.zip.part-*"], "licence": "BSD-3-Clause (dataset card)",
     "note": "public TRAIN annotations + the 6-part image archive (holds the images of every split; only TRAIN members are ever read); V7 FIT / CAL / DEV come from TRAIN only"},
    {"id": "evalmuse_test_closed", "group": "v7", "kind": "hf", "repo": "DY-Evalab/EvalMuse", "type": "dataset", "dest": DATA / "evalmuse" / "closed",
     "allow": ["test.json"], "closed": True, "licence": "BSD-3-Clause",
     "note": "closed: hashed only; whether it carries usable labels is unknown and opening it needs an owner decision (V7 package D01)"},
    {"id": "genai_rank_images", "group": "v7", "kind": "hf", "repo": "zhiqiulin/GenAI-Image-Ranking-800", "type": "dataset", "dest": DATA / "genai_rank",
     "allow": [".gitattributes", "README.md", "DALLE_3.zip", "SDXL_Base.zip", "genai_skills.json"], "licence": "MIT (dataset card)",
     "note": EVAL_ONLY + "; GenAI-Rank candidate images (800 prompts x 9 per generator) and skill tags; no human scores"},
    {"id": "genai_rank_labels_closed", "group": "v7", "kind": "hf", "repo": "zhiqiulin/GenAI-Image-Ranking-800", "type": "dataset",
     "dest": DATA / "genai_rank" / "closed", "allow": ["human_rating.json"], "closed": True, "licence": "MIT",
     "note": "terminal human ratings: hashed only until the V7 prediction / protocol freeze and an owner scoring approval"},
    {"id": "fga_blip2_released", "group": "v7", "kind": "hf", "repo": "hanshuhao/FGA-BLIP2", "type": "model", "dest": MODELS / "fga_blip2_released",
     "licence": "no licence stated on the model card; EvalMuse code Apache-2.0",
     "note": "released FGA-BLIP2 weights: V7 baseline B0 inference only (may have seen all public TRAIN incl. the local holdout); never a training init"},
    {"id": "blip2_lavis_init", "group": "v7", "kind": "http", "dest": MODELS / "torch_home" / "hub" / "checkpoints", "licence": "BSD-3-Clause (LAVIS)",
     "files": [http("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth", "eva_vit_g.pth", md5="c483374fc17b72dc61e016680345a0ff"),
               http("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_finetune_coco.pth", "blip2_finetune_coco.pth", md5="0dc9921f993dc99818a664bf5b38cabc")],
     "note": "the common V7 training init (FGA config model_type coco, load_finetuned True): EVA ViT-g then the BLIP-2 COCO-finetuned checkpoint, in LAVIS's torch-hub cache layout"},
    {"id": "bert_base_uncased", "group": "v7", "kind": "hf", "repo": "google-bert/bert-base-uncased", "type": "model", "dest": MODELS / "bert-base-uncased",
     "allow": ["config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "pytorch_model.bin"], "licence": "Apache-2.0",
     "note": "Q-Former init and tokenizer that LAVIS loads by the name 'bert-base-uncased' (served from a local directory of that name)"},
    # ---- V7X (owner 2026-10-14: v2 external-benchmark package; public scorers of reports/V7/V7X_30 on branch v7/t2i-alignment) ------------
    {"id": "tifa_llama2_qg", "group": "v7x", "kind": "hf", "repo": "tifa-benchmark/llama2_tifa_question_generation", "type": "model", "dest": MODELS / "tifa_llama2_qg",
     "licence": "Apache-2.0 (model card; LLaMA-2 fine-tune: the Llama 2 community licence also applies to the base weights)",
     "note": "TIFA's released local question generator (tifascore get_llama2_pipeline): baseline B4T"},
    {"id": "unifiedqa_v2_t5_large", "group": "v7x", "kind": "hf", "repo": "allenai/unifiedqa-v2-t5-large-1363200", "type": "model", "dest": MODELS / "unifiedqa-v2-t5-large-1363200",
     "allow": ["config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "README.md"],
     "licence": "Apache-2.0 (UnifiedQA repository; no licence field on the card)", "note": "TIFA question filter: B4T"},
    {"id": "sbert_all_mpnet_base_v2", "group": "v7x", "kind": "hf", "repo": "sentence-transformers/all-mpnet-base-v2", "type": "model", "dest": MODELS / "all-mpnet-base-v2",
     "allow": ["config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "modules.json",
               "sentence_bert_config.json", "config_sentence_transformers.json", "1_Pooling/*", "README.md"],
     "licence": "Apache-2.0", "note": "TIFA free-form -> multiple-choice answer mapping (tifascore SBERTModel): B4T"},
    {"id": "mplug_large_vqa", "group": "v7x", "kind": "modelscope", "repo": "damo/mplug_visual-question-answering_coco_large_en", "revision": "v1.0.0",
     "dest": MODELS / "mplug_vqa_coco_large_en", "allow": ["*.json", "*.yaml", "*.bin", "*.txt", "README.md", ".gitattributes"], "licence": "Apache-2.0 (ModelScope card)",
     "note": "the VQA model TIFA recommends and DSG's example uses (B4T / B4D); weights and configs identical at v1.0.0 and master (2026-10-14 listing)"},
    {"id": "qwen2_5_32b_instruct", "group": "v7x", "kind": "hf", "repo": "Qwen/Qwen2.5-32B-Instruct", "type": "model", "dest": MODELS / "Qwen2.5-32B-Instruct", "licence": "Apache-2.0",
     "note": "local LLM for DSG's tuple / dependency / question prompts and PN-VQA's element / question templates (substitutes PaLM 2 / GPT: reproduction adaptation): B4D / B3"},
    {"id": "qwen2_vl_7b_instruct", "group": "v7x", "kind": "hf", "repo": "Qwen/Qwen2-VL-7B-Instruct", "type": "model", "dest": MODELS / "Qwen2-VL-7B-Instruct", "licence": "Apache-2.0",
     "note": "PN-VQA's VQA model (the EvalMuse paper's best MLLM family): B3"},
    {"id": "hpsv3", "group": "v7x", "kind": "hf", "repo": "MizzenAI/HPSv3", "type": "model", "dest": MODELS / "HPSv3", "licence": "Apache-2.0 (model card; code MIT)",
     "note": "extended preference scorer B5 (runnability check, then GenAI-Rank scores)"},
    # ---- r7 Tier A: model weights named in the D1–D4 plans --------------------------------------------------------
    {"id": "qwen2_5_vl_7b_instruct", "group": "tierA", "kind": "hf", "repo": "Qwen/Qwen2.5-VL-7B-Instruct", "type": "model", "dest": MODELS / "Qwen2.5-VL-7B-Instruct", "licence": "Apache-2.0"},
    {"id": "vqascore_clip_flant5_xxl", "group": "tierA", "kind": "hf", "repo": "zhiqiulin/clip-flant5-xxl", "type": "model", "dest": MODELS / "clip-flant5-xxl", "licence": "Apache-2.0"},
    {"id": "clip_vit_l14", "group": "tierA", "kind": "hf", "repo": "openai/clip-vit-large-patch14", "type": "model", "dest": MODELS / "clip-vit-large-patch14",
     "ignore": ["*.h5", "*.msgpack"], "licence": "MIT (OpenAI CLIP)"},
    {"id": "prolip_b16", "group": "tierA", "kind": "hf", "repo": "SanghyukChun/ProLIP-ViT-B-16-DC-1B-12_8B", "type": "model", "dest": MODELS / "ProLIP-ViT-B-16-DC-1B-12_8B", "licence": "MIT"},
    {"id": "hpsv2_ckpt", "group": "tierA", "kind": "hf", "repo": "xswu/HPSv2", "type": "space", "dest": MODELS / "hpsv2", "allow": ["HPS_v2_compressed.pt"], "licence": "Apache-2.0 (HPSv2 repo)"},
    # ---- r7 Tier B: D6 official features + annotations (after Tier A) ---------------------------------------------
    {"id": "qvhighlights", "group": "tierB", "kind": "git", "repo": "https://github.com/jayleicn/moment_detr.git", "dest": DATA / "qvhighlights",
     "licence": "annotations CC BY-NC-SA 4.0; code MIT", "auth": NC_AUTH, "close_after_extract": ["highlight_test_*"],
     "note": "official annotations (data/highlight_*_release.jsonl); the repo also ships highlight_test_with_gt.jsonl (test labels): both test files closed; raw videos are Tier C"},
    {"id": "qvhighlights_features", "group": "tierB", "kind": "hf", "repo": "dpaul06/QVHighlights_preprocessed", "type": "dataset", "dest": DATA / "qvhighlights" / "features_hf_mirror",
     "allow": ["moment_detr_features.tar.gz", "README.md"], "extract": True, "auth": NC_AUTH,
     "licence": "features derived from QVHighlights (annotations CC BY-NC-SA 4.0); mirror tagged MIT",
     "flag": "unofficial mirror, no official checksum",
     "validation_required": "before any D6 use, reproduce Moment-DETR's published QVHighlights val numbers with these features (same code and config); if they do not match "
                            "within the paper's rounding, the features are not used for any claim.",
     "provenance": "HF mirror dpaul06/QVHighlights_preprocessed, approved by the coordinator 2026-10-12",
     "checks": "2026-10-12 structure check (file names only, no array loaded): features/clip_features and slowfast_features 12,562 .npz each, clip_text_features "
               "10,310 + args.json, clip_sub_features 329,746; every train (7,218 queries / 7,100 videos) and val (1,550 / 1,519) annotation has its video and "
               "query features; test annotations closed, not checked",
     "note": "mirror of the official Moment-DETR SlowFast + CLIP feature tarball: the official Drive link (1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9) returns 404 on 2026-10-11 "
             "and no official checksum exists, so identity with the official file is unverifiable; sha256 checked against the mirror's LFS hash; load the .npz with allow_pickle=False"},
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


def _present(dest: Path, names) -> set:
    """Member names already on disk, from one readdir per directory (a stat per file costs ~70 ms on the loaded shared NFS)."""
    have = set()
    for d in {str(Path(n).parent) for n in names}:
        if (dest / d).is_dir():
            have.update(x if d == "." else f"{d}/{x}" for x in os.listdir(dest / d))
    return have


def safe_extract(archive: Path, dest: Path, threads: int = 4) -> int:
    """List first; refuse absolute / parent paths and links leaving dest; never run anything.
    Resumable without a stat per member: zip members are written to <name>.part by a few threads and renamed, so a present name is
    complete; the last few present members in archive order are still size-checked (tar, and zips cut by the earlier single-thread code)."""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
        if any(i.filename.startswith("/") or ".." in Path(i.filename).parts for i in infos):
            raise RuntimeError(f"unsafe paths in {archive.name}")
        key = lambda n: str(Path(n))  # noqa: E731 — "./a/b" and "a/b" name the same file
        files = [i for i in infos if not i.is_dir()]; present = _present(dest, [i.filename for i in files])
        bad = {key(i.filename) for i in [i for i in files if key(i.filename) in present][-8:] if (dest / i.filename).stat().st_size != i.file_size}
        todo = [i for i in files if key(i.filename) not in present or key(i.filename) in bad]
        for d in sorted({(dest / i.filename).parent for i in todo}):
            d.mkdir(parents=True, exist_ok=True)

        def part(chunk):
            with zipfile.ZipFile(archive) as z:
                for i in chunk:
                    out = dest / i.filename; tmp = out.with_name(out.name + ".part")
                    with z.open(i) as src, open(tmp, "wb") as dst:  # ZipExtFile checks the CRC at EOF
                        shutil.copyfileobj(src, dst, 1 << 20)
                    os.replace(tmp, out)
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(part, [todo[k::threads] for k in range(threads)]))
        return len(infos)
    with tarfile.open(archive) as t:
        members = t.getmembers()
        for m in members:
            if m.name.startswith("/") or ".." in Path(m.name).parts or ((m.issym() or m.islnk()) and (m.linkname.startswith("/") or ".." in Path(m.linkname).parts)):
                raise RuntimeError(f"unsafe member {m.name} in {archive.name}")
        key = lambda n: str(Path(n))  # noqa: E731
        present = _present(dest, [m.name for m in members if not m.isdir()])
        bad = {key(m.name) for m in [m for m in members if m.isfile() and key(m.name) in present][-8:] if (dest / m.name).stat().st_size != m.size}
        todo = [m for m in members if m.isdir() or key(m.name) not in present or key(m.name) in bad]
        t.extractall(dest, members=todo, filter="data") if sys.version_info >= (3, 12) else t.extractall(dest, members=todo)
    return len(members)


def close_matching(root: Path, patterns: list[str]) -> list[str]:
    """Move files matching the patterns (by name) into root/closed/ without opening them."""
    moved = []; closed = root / "closed"
    for p in list(root.rglob("*")):
        if p.is_file() and not {"closed", ".git"} & set(p.relative_to(root).parts) and any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
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
    if item.get("extract") and not item.get("closed"):
        rec["extracted_entries"] = {f["name"]: safe_extract(dest / f["name"], dest) for f in files if f["name"].endswith((".zip", ".tgz", ".tar.gz", ".tar"))}
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
    lfs_checkout(item, rec)


def lfs_checkout(item, rec) -> None:
    """Equivalent of `git lfs checkout` for the dirs: each LFS pointer stub in repo/<dir> is replaced (atomically) by a hard link to its
    copy in media/<dir> after checking that copy's size and sha256 against the pointer; no extra space is used."""
    dest = Path(item["dest"]); repo = dest / "repo"; jobs = [(d, n) for d in item["dirs"] for n in sorted(os.listdir(repo / d))]
    bad = []

    def one(job):
        d, name = job; wt = repo / d / name; src = dest / "media" / d / name
        if wt.stat().st_ino == src.stat().st_ino:
            return True  # already checked out by an earlier pass
        lines = dict(l.split(" ", 1) for l in wt.read_text().splitlines() if " " in l)
        oid, size = lines["oid"].split(":", 1)[1], int(lines["size"])
        if src.stat().st_size != size or digest(src, "sha256") != oid:
            bad.append(f"{d}/{name}"); return False
        tmp = wt.with_name(name + ".lfs-tmp"); tmp.unlink(missing_ok=True); os.link(src, tmp); os.replace(tmp, wt)
        return wt.stat().st_ino == src.stat().st_ino and wt.stat().st_size == size
    with ThreadPoolExecutor(16) as ex:
        n_ok = sum(ex.map(one, jobs))
    rec["working_tree"] = {"path": str(repo), "method": "LFS pointer stubs replaced by hard links to the verified media/ copies (sha256 + size vs pointer)",
                           "files": len(jobs), "checked_out": n_ok, "failed": bad[:50], "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log(f"  {item['id']}: working tree checkout {n_ok}/{len(jobs)} files")
    if n_ok != len(jobs):
        raise RuntimeError(f"working-tree checkout incomplete: {n_ok}/{len(jobs)}")


def do_modelscope(item, rec) -> None:
    """ModelScope model repository at a pinned revision: the file list with published sha256 (API), each blob fetched with wget and verified."""
    import urllib.request
    base = "https://www.modelscope.cn/api/v1/models/" + item["repo"]; rev = item.get("revision", "master"); dest = Path(item["dest"]); files = []
    with urllib.request.urlopen(f"{base}/repo/files?Revision={rev}&Recursive=true", timeout=60) as r:
        listing = json.load(r)["Data"]["Files"]
    for f in listing:
        if f.get("Type") != "blob" or (item.get("allow") and not any(fnmatch.fnmatch(f["Path"], pat) for pat in item["allow"])):
            continue
        out = dest / f["Path"]; out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.exists() and out.stat().st_size == f["Size"]):
            subprocess.run([WGET, "-c", "-q", "--tries=10", "--timeout=60", "-O", str(out), f"{base}/repo?Revision={rev}&FilePath={f['Path']}"], check=True)
        got = digest(out, "sha256"); files.append({"name": f["Path"], "bytes": out.stat().st_size, "sha256_published": f.get("Sha256"), "sha256_ok": got == f.get("Sha256")})
        log(f"  {item['id']}: {f['Path']} {out.stat().st_size} B sha256 ok {got == f.get('Sha256')}")
    rec.update({"revision": rev, "files": files, "bytes": sum(x["bytes"] for x in files)})
    if any(x["sha256_ok"] is False for x in files):
        raise RuntimeError("ModelScope sha256 mismatch")


def do_local(item, rec) -> None:
    p = Path(item["dest"]); rec["present"] = p.exists(); rec["bytes"] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


HANDLERS = {"http": do_http, "hf": do_hf, "git": do_git, "lfs_media": do_lfs_media, "local": do_local, "modelscope": do_modelscope}


def close_item(item, rec) -> None:
    """Apply the item's close patterns (git items included) and record every file under <dest>/closed/ (names only, never opened)."""
    if item.get("close_after_extract"):
        root = Path(item["dest"]); close_matching(root, item["close_after_extract"])
        rec["moved_to_closed"] = sorted(str(q.relative_to(root)) for q in (root / "closed").rglob("*") if q.is_file() and ".cache" not in q.parts)


def run_item(item, manifest) -> None:
    rec = manifest["items"].setdefault(item["id"], {}); TOUCHED.add(item["id"])
    rec.update({k: item[k] for k in ("flag", "validation_required", "provenance", "checks") if k in item})
    if "error" in rec:  # a retried failure is kept as history, never left as a stale error next to a later success
        rec.setdefault("history", []).append({"status": "failed", "error": rec.pop("error"), "finished": rec.get("finished"),
                                              "resolution": "retried; see the current status" if rec.get("status") != "done" else "retried and verified"})
    if rec.get("status") == "done":
        save(manifest); return  # keeps provenance / validation flags added after the item finished
    if free_gb() < FLOOR_GB + 10:
        rec["status"] = "deferred_free_space"; log(f"DEFER {item['id']}: free {free_gb():.0f} GB < floor"); return
    rec.update({"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}); save(manifest)
    log(f"START {item['id']} ({item['kind']})")
    try:
        if item["kind"] == "git+http":
            do_git(item, rec); do_http(item, rec)
        else:
            HANDLERS[item["kind"]](item, rec)
        close_item(item, rec)
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
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["manifest", "run", "status", "checkout", "close"]); ap.add_argument("--group", default=None)
    ap.add_argument("--ids", nargs="*", default=None); ap.add_argument("--jobs", type=int, default=2); a = ap.parse_args(); manifest = load()
    if a.cmd == "manifest":
        save(manifest); print(f"wrote {MANIFEST} ({len(ITEMS)} items)"); return 0
    if a.cmd == "close":  # close patterns added after an item finished
        for i in [i for i in ITEMS if i.get("close_after_extract") and a.ids is not None and i["id"] in a.ids]:
            rec = manifest["items"].setdefault(i["id"], {}); TOUCHED.add(i["id"]); close_item(i, rec); save(manifest)
            log(f"  {i['id']}: closed {len(rec['moved_to_closed'])} files")
        return 0
    if a.cmd == "checkout":  # LFS working-tree checkout for already-downloaded lfs_media items
        for i in [i for i in ITEMS if i["kind"] == "lfs_media" and (a.ids is None or i["id"] in a.ids)]:
            rec = manifest["items"].setdefault(i["id"], {}); TOUCHED.add(i["id"])
            try:
                lfs_checkout(i, rec)
            finally:
                save(manifest)
        return 0
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
