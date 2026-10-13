"""V7 baseline B1 (t2v-metrics 3.0 CLIP-FlanT5-XXL) offline cache: the release loads 'zhiqiulin/clip-flant5-xxl' (cache_dir ./hf_cache),
'google/flan-t5-xxl' (tokenizer, default cache) and 'openai/clip-vit-large-patch14-336' (vision tower, default cache) by hub name.  This builds
one HF hub-layout cache: the two models already registered in data/DATA_MANIFEST.json are linked file by file at their recorded revisions
(no second 23 GB copy); only the flan-t5-xxl tokenizer files are downloaded.  Run with network once; jobs then use HF_HUB_OFFLINE=1.
    python -I scripts/v7/v7_hf_cache.py --cache /projects/EEG-foundation-model/yinghao/models/hf_cache_v7/hub
"""
import argparse
import json
import os
from pathlib import Path

MANIFEST = Path("/home/infres/yinwang/CS_QMI/ssl_pilot/data/DATA_MANIFEST.json"); MODELS = Path("/projects/EEG-foundation-model/yinghao/models")
LINKED = {"zhiqiulin/clip-flant5-xxl": ("vqascore_clip_flant5_xxl", MODELS / "clip-flant5-xxl"),
          "openai/clip-vit-large-patch14-336": ("clip_vit_l14_336", MODELS / "clip-vit-large-patch14-336")}
TOKENIZER = ("google/flan-t5-xxl", ["spiece.model", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "config.json"])


def link_snapshot(cache: Path, repo: str, rev: str, src: Path):
    d = cache / ("models--" + repo.replace("/", "--")); snap = d / "snapshots" / rev; snap.mkdir(parents=True, exist_ok=True); (d / "refs").mkdir(exist_ok=True)
    (d / "refs" / "main").write_text(rev); n = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            dst = snap / f.name
            if not dst.exists():
                dst.symlink_to(f); n += 1
    return {"repo": repo, "revision": rev, "linked_files": n, "source": str(src)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); a = ap.parse_args(); cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    items = json.loads(MANIFEST.read_text())["items"]; rec = {"linked": [], "downloaded": None}
    for repo, (mid, src) in LINKED.items():
        rev = items[mid]["revision"]; rec["linked"].append(link_snapshot(cache, repo, rev, src))
    from huggingface_hub import snapshot_download
    os.environ.pop("HF_HUB_OFFLINE", None)
    p = snapshot_download(TOKENIZER[0], cache_dir=str(cache), allow_patterns=TOKENIZER[1]); rec["downloaded"] = {"repo": TOKENIZER[0], "path": p, "files": sorted(os.listdir(p))}
    (cache.parent / "CACHE_PROVENANCE.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
