"""r7 R7-30 D1-30 evaluation sets.  'build' runs ONLY after the coordinator's D1-30 staged freeze (it refuses unless the frozen protocol
carries a 'd1_30_staged_freeze' key); before that only 'fixtures' (the R7-20 fixtures, already read) and the synthetic unit tests exist.
  gqa_dev     : the 2,000 DEV images of the frozen D1 manifest; their questions in GQA train_all (all question types) minus every LLaVA-seen
                (image, normalised question) pair of llava_v1_5_mix665k.json (the frozen D1 match rule); per image the first 4 in
                sha256('r7-30/d1-q/<qid>') order (the frozen D1 question rule); images in the manifest's role order
  textvqa_val : TextVQA 0.5.1 validation (lmms-lab parquet) minus every question on the 127 images read in R7-20 (image-grouped exclusion);
                primary prompt question + short-answer suffix (no OCR tokens), anchor prompt = LLaVA's OCR-token layout
  chartqa_val : ChartQA val (1,920 rows: human + augmented, 'type' kept; every type must name exactly one of the two), one group per chart
                image (imgname); prompt query + '\\nAnswer the question with a single word.' (pinned lmms-eval chartqa.yaml)
Images of TextVQA / ChartQA are written once (original bytes) under <out>/images/<bench>/ on /home; GQA images are read in place.
    python scripts/r7_d1_30_data.py build --out O --protocol reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml [--sets gqa_dev textvqa_val chartqa_val]
    python scripts/r7_d1_30_data.py fixtures --out O        (smoke: gqa_fixture, textvqa_fixture, chartqa_proxy from the R7-20 fixtures)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_data as DA  # noqa: E402   (frozen D1 reader allow-list, norm_q, seen-pair rule, hash tags; read-only use)

DATA = Path("/projects/EEG-foundation-model/yinghao/data")
D1 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1")
FIXTURES = DATA / "r7_fixtures"
FREEZE_KEY = "d1_30_staged_freeze"
SHORT = "\nAnswer the question using a single word or phrase."
CHART_SUFFIX = "\nAnswer the question with a single word."


def require_freeze(protocol: Path):
    if FREEZE_KEY not in Path(protocol).read_text():
        raise SystemExit(f"D1-30 is not frozen ({FREEZE_KEY} absent in {protocol}); evaluation sets stay unread")


def ocr_prompt(question, tokens):
    return f"{question}\nReference OCR token: {', '.join(tokens or [])}{SHORT}"


def pick_gqa_dev(rows, dev_order, seen, per_image=4):
    """rows: train_all rows; dev_order: DEV image ids in manifest order.  Returns (items, stats)."""
    dev = set(dev_order); by_img, n_seen = {}, 0
    for r in rows:
        if r["image_id"] not in dev:
            continue
        if (r["image_id"], DA.norm_q(r["question"])) in seen:
            n_seen += 1; continue
        by_img.setdefault(r["image_id"], []).append(r)
    out = []
    for img in dev_order:
        for r in sorted(by_img.get(img, []), key=lambda r: DA.h("d1-q", r["qid"]))[:per_image]:
            a = str(r["answer"]).strip().lower()
            out.append({"bench": "gqa_dev", "qid": r["qid"], "image_id": img, "question": r["question"], "answer": r["answer"], "binary": a in ("yes", "no"),
                        "structural": r.get("structural"), "semantic": r.get("semantic"), "detailed": r.get("detailed"), "prompt": r["question"] + SHORT})
    stats = {"excluded_llava_seen_pairs_on_dev_images": n_seen, "dev_images": len(dev_order), "images_without_unseen_questions": len(dev - set(by_img)),
             "items": len(out), "binary_items": sum(o["binary"] for o in out)}
    return out, stats


def textvqa_items(rows, excluded_images):
    out = []
    for r in rows:
        if r["image_id"] in excluded_images:
            continue
        out.append({"bench": "textvqa_val", "qid": str(r["question_id"]), "image_id": r["image_id"], "question": r["question"], "answers": list(r["answers"]),
                    "prompt": r["question"] + SHORT, "prompt_ocr_anchor": ocr_prompt(r["question"], r["ocr_tokens"])})
    return out


def chartqa_items(rows):
    out = []
    for i, r in enumerate(rows):
        t = str(r["type"]).lower(); kinds = [k for k in ("human", "augmented") if k in t]
        if len(kinds) != 1:
            raise SystemExit(f"ChartQA row {i}: type {r['type']!r} does not name exactly one of human / augmented")
        out.append({"bench": "chartqa_val", "qid": f"chartqa_val_{i:05d}", "image_id": r["imgname"], "question": r["query"], "answer": r["label"],
                    "type": r["type"], "split": kinds[0], "prompt": r["query"] + CHART_SUFFIX})
    return out


def image_path(out: Path, bench, iid, data: bytes):
    from PIL import Image
    fmt = (Image.open(io.BytesIO(data)).format or "bin").lower().replace("jpeg", "jpg")
    return out / "images" / bench / f"{hashlib.sha1(iid.encode()).hexdigest()[:16]}.{fmt}"


def write_image(bench, iid, data: bytes, out: Path, paths: dict):
    """Write one image once (original bytes); paths: {image_id: path} updated in place."""
    if iid not in paths:
        p = image_path(out, bench, iid, data); p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(data)
        paths[iid] = str(p)


def fixture_sets(out: Path):
    """Smoke only: the R7-20 GQA / TextVQA fixtures (already read in R7-20) in D1-30 item format, plus chartqa_proxy = the GQA fixture with the
    ChartQA prompt suffix (exercises the ChartQA prompt / max_new_tokens / relaxed-accuracy path without reading any ChartQA data)."""
    out.mkdir(parents=True, exist_ok=True); res = {}
    gqa = [json.loads(x) for x in (FIXTURES / "gqa.jsonl").read_text().splitlines() if x.strip()]
    tv = [json.loads(x) for x in (FIXTURES / "textvqa.jsonl").read_text().splitlines() if x.strip()]
    sets = {"gqa_fixture": [{"bench": "gqa_fixture", "qid": r["qid"], "image_id": r["image_id"], "image_file": r["image_file"], "question": r["question"],
                             "answer": r["answer"], "binary": r["binary"], "structural": r.get("structural"), "prompt": r["question"] + SHORT} for r in gqa],
            "textvqa_fixture": [{"bench": "textvqa_fixture", "qid": r["qid"], "image_id": r["image_id"], "image_file": r["image_file"], "question": r["question"],
                                 "answers": r["answers"], "prompt": r["question"] + SHORT, "prompt_ocr_anchor": r["prompt"]} for r in tv],
            "chartqa_proxy": [{"bench": "chartqa_proxy", "qid": r["qid"], "image_id": r["image_id"], "image_file": r["image_file"], "question": r["question"],
                               "answer": r["answer"], "type": "human_proxy", "split": "human", "prompt": r["question"] + CHART_SUFFIX} for r in gqa]}
    for k, rows in sets.items():
        (out / f"{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); res[k] = len(rows)
    return res


def build(out: Path, protocol: Path, sets, data=DATA, d1=D1, fixtures=FIXTURES):
    """The three evaluation sets (after the freeze only).  Roots are parameters for the synthetic CPU test; production uses the defaults."""
    require_freeze(Path(protocol)); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    summary = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "protocol_sha256": DA.sha256_file(Path(protocol))}
    import pyarrow.parquet as pq
    if "gqa_dev" in sets:
        man = [json.loads(x) for x in (d1 / "data" / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
        meta = json.loads((d1 / "data" / "d1_manifest_meta.json").read_text())
        if meta.get("frozen") is not True:
            raise SystemExit("D1 manifest not frozen")
        dev_order = list(dict.fromkeys(r["image_id"] for r in man if r["role"] == "DEV"))
        seen = DA.llava_seen_pairs(data / "llava_mix665k" / "llava_v1_5_mix665k.json")
        items, st = pick_gqa_dev(DA.read_questions(data / "gqa" / "train_all_questions", "all"), dev_order, seen)
        for it in items:
            it["image_file"] = str(data / "gqa" / "images" / f"{it['image_id']}.jpg")
            if not Path(it["image_file"]).exists():
                raise SystemExit(f"missing GQA image {it['image_file']}")
        summary["gqa_dev_rule"] = st; summary["gqa_dev_manifest_sha256"] = meta["manifest_sha256"]
        (out / "gqa_dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    if "textvqa_val" in sets:
        excl = {json.loads(x)["image_id"] for x in (fixtures / "textvqa.jsonl").read_text().splitlines() if x.strip()}
        shards = sorted((data / "textvqa_lmms" / "data").glob("validation-*.parquet")); rows, paths = [], {}
        for f in shards:
            for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["question_id", "image_id", "question", "answers", "ocr_tokens", "image"]):
                for r in b.to_pylist():
                    img = r.pop("image")
                    if r["image_id"] not in excl:
                        write_image("textvqa_val", r["image_id"], img["bytes"], out, paths)
                    rows.append(r)
        items = textvqa_items(rows, excl)
        for it in items:
            it["image_file"] = paths[it["image_id"]]
        summary["textvqa_rule"] = {"val_questions": len(rows), "excluded_r7_20_images": len(excl),
                                   "excluded_questions": sum(r["image_id"] in excl for r in rows), "items": len(items)}
        (out / "textvqa_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    if "chartqa_val" in sets:
        f = data / "chartqa" / "data" / "val-00000-of-00001.parquet"; rows, paths = [], {}
        for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["imgname", "query", "label", "type", "image"]):
            for r in b.to_pylist():
                write_image("chartqa_val", r["imgname"], r.pop("image"), out, paths); rows.append(r)
        items = chartqa_items(rows)
        for it in items:
            it["image_file"] = paths[it["image_id"]]
        summary["chartqa_rule"] = {"rows": len(rows), "charts": len(paths), "human": sum(i["split"] == "human" for i in items),
                                   "augmented": sum(i["split"] == "augmented" for i in items)}
        (out / "chartqa_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    for k in ("gqa_dev", "textvqa_val", "chartqa_val"):
        p = out / f"{k}.jsonl"
        if p.exists():
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            summary[k] = {"items": len(rows), "groups": len({r["image_id"] for r in rows}), "sha256": DA.sha256_file(p)}
    summary["seconds"] = time.time() - t0; (out / "d1_30_sets.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "fixtures"]); ap.add_argument("--out", required=True); ap.add_argument("--protocol")
    ap.add_argument("--sets", nargs="+", default=["gqa_dev", "textvqa_val", "chartqa_val"])
    a = ap.parse_args(); out = Path(a.out)
    if a.cmd == "fixtures":
        print(json.dumps(fixture_sets(out))); return 0
    print(json.dumps(build(out, Path(a.protocol), a.sets), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
