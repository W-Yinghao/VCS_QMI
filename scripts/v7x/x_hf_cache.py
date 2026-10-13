"""V7X offline HF cache for the hub ids that third-party code hard-codes (TIFA's SBERTModel('sentence-transformers/all-mpnet-base-v2')): the local
copies registered in data/DATA_MANIFEST.json are linked file by file at their recorded revisions (no second copy, no download).
    python -I scripts/v7x/x_hf_cache.py --cache /projects/EEG-foundation-model/yinghao/models/hf_cache_v7x/hub"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v7"))
from v7_hf_cache import MANIFEST, MODELS, link_snapshot  # noqa: E402

LINKED = {"sentence-transformers/all-mpnet-base-v2": ("sbert_all_mpnet_base_v2", MODELS / "all-mpnet-base-v2")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); a = ap.parse_args(); cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    items = json.loads(MANIFEST.read_text())["items"]; rec = {"linked": []}
    for repo, (mid, src) in LINKED.items():
        if items.get(mid, {}).get("status") != "done":
            raise SystemExit(f"{mid} is not a completed download in {MANIFEST}")
        rec["linked"].append(link_snapshot(cache, repo, items[mid]["revision"], src))
    (cache.parent / "CACHE_PROVENANCE.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
