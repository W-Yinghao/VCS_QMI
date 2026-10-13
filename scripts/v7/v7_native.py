"""V7 N0: the FGA-BLIP2 native recipe (EvalMuse@45c8bb4, code unmodified) retrained from the common init on V7 FIT only; native evaluation.
  prep   --work W                      FIT records in the raw train_list format -> W/dataset/train_list.json, then the native process_train.py and
                                       element2mask.py run unmodified inside W (their printed counts -> W/native_prep.json)
  yaml   --work W --images IMG --seed S --out-dir O [--max-epoch E]   the native config (lavis/projects/blip2/train/fga_blip2.yaml) with absolute
                                       annotation / image / output paths and the unit seed; every other key as shipped (E only for smokes)
  stage  --dest IMG --roles R ...      the TRAIN images of the given roles extracted from the archive (node-local), at IMG/<generator>/<file>
  evalset --roles R ... --out F        a native eval.py data file (prompt, img_path, element_score keys) for the given roles
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as VC  # noqa: E402
import v7_data as VD  # noqa: E402


def roles_of(data_dir: Path):
    return {i["image_id"]: i["role"] for i in VC.read_jsonl(data_dir / "manifest_images.jsonl")}


def cmd_prep(a):
    work = Path(a.work); (work / "dataset").mkdir(parents=True, exist_ok=True); VC.stage_workdir(work)
    roles = roles_of(Path(a.data_dir)); raw = json.loads((VD.EVALMUSE / "train_list.json").read_text())
    fit = [r for r in raw if roles.get(r["img_path"]) == "FIT"]
    (work / "dataset" / "train_list.json").write_text(json.dumps(fit, ensure_ascii=False))
    rec = {"fit_records_in": len(fit), "scripts": {}}
    for script in ("process/process_train.py", "process/element2mask.py"):
        buf = io.StringIO(); t0 = time.time()
        with contextlib.redirect_stdout(buf):
            sys.argv = [str(VC.FGA_REPO / script)]; sys.path.insert(0, str(VC.FGA_REPO)); runpy.run_path(str(VC.FGA_REPO / script), run_name="__main__")
        rec["scripts"][script] = {"stdout": buf.getvalue().strip().splitlines()[-5:], "seconds": time.time() - t0}
    lines = rec["scripts"]["process/element2mask.py"]["stdout"]
    try:
        rec["element2mask"] = {"prompts_over_32_tokens": int(lines[-3]), "records": int(lines[-2]), "unmatched_elements": int(lines[-1])}
    except (ValueError, IndexError):
        rec["element2mask"] = {"unparsed_stdout": lines}
    kept = json.loads((work / "dataset" / "train_mask.json").read_text()); rec["records_kept_by_native_mapping"] = len(kept)
    rec["records_dropped_by_native_mapping"] = len(fit) - len(kept)
    (work / "native_prep.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1)); return 0


def cmd_yaml(a):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(VC.FGA_REPO / "lavis/projects/blip2/train/fga_blip2.yaml"); ann = str(Path(a.work).resolve() / "dataset" / "train_mask.json")
    anns = cfg.datasets.alignment.build_info.annotations
    for split in ("train", "val", "test"):                                 # val / test as the shipped dataset defaults (= the train file; the task evaluation is a no-op)
        anns[split] = {"url": ann, "storage": ann}
    cfg.datasets.alignment.build_info.images.storage = str(Path(a.images).resolve())
    cfg.run.seed = a.seed; cfg.run.output_dir = str(Path(a.out_dir).resolve())
    if a.max_epoch:
        cfg.run.max_epoch = a.max_epoch
    out = Path(a.out_dir) / "fga_blip2_unit.yaml"; out.parent.mkdir(parents=True, exist_ok=True); OmegaConf.save(cfg, out); print(out); return 0


def cmd_stage(a):
    roles = roles_of(Path(a.data_dir)); want = sorted(i for i, r in roles.items() if r in set(a.roles)); dest = Path(a.dest); t0 = time.time()
    z = VD.open_archive(); n = 0
    for img in want:
        out = dest / img
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(out.suffix + ".part"); tmp.write_bytes(z.read(VD.ZIP_PREFIX + img)); os.replace(tmp, out); n += 1
    print(json.dumps({"dest": str(dest), "roles": a.roles, "images": len(want), "extracted_now": n, "seconds": time.time() - t0})); return 0


def cmd_stage_rank(a):
    """GenAI-Rank candidates (genai_rank_inputs.jsonl: candidate_index 0..8 only) extracted from DALLE_3.zip / SDXL_Base.zip to DEST/<generator>/<file>."""
    import zipfile
    rows = VC.read_jsonl(a.inputs); dest = Path(a.dest); zs = {}; t0 = time.time(); n = 0
    for r in rows:
        out = dest / r["image_id"]
        if out.exists():
            continue
        g = r["generator_id"]; z = zs.get(g) or zs.setdefault(g, zipfile.ZipFile(f"/projects/EEG-foundation-model/yinghao/data/genai_rank/{g}.zip"))
        out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(".part"); tmp.write_bytes(z.read(r["image_id"])); os.replace(tmp, out); n += 1
    print(json.dumps({"dest": str(dest), "candidates": len(rows), "extracted_now": n, "seconds": time.time() - t0})); return 0


def cmd_evalset_rank(a):
    """The native eval.py data file for GenAI-Rank: prompt + img_path, no element keys (overall score only; no human score is read)."""
    rows = VC.read_jsonl(a.inputs)
    if any(not r.get("prompt") for r in rows):
        raise SystemExit("GenAI-Rank prompt mapping incomplete")
    items = [{"prompt_id": r["prompt_id"], "generator_id": r["generator_id"], "candidate_index": r["candidate_index"], "prompt": r["prompt"],
              "img_path": r["image_id"], "element_score": {}} for r in rows]
    Path(a.out).write_text(json.dumps(items, ensure_ascii=False)); print(json.dumps({"items": len(items), "out": a.out})); return 0


def cmd_evalset(a):
    roles = roles_of(Path(a.data_dir)); raw = json.loads((VD.EVALMUSE / "train_list.json").read_text())
    items = [{"prompt_id": r["prompt_id"], "prompt": r["prompt"], "img_path": r["img_path"], "element_score": {k: 0 for k in r["element_score"]}}
             for r in raw if roles.get(r["img_path"]) in set(a.roles)]                 # element keys only: eval.py needs the names, never the votes
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(items, ensure_ascii=False)); print(json.dumps({"items": len(items), "out": a.out})); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--work", required=True); p.add_argument("--data-dir", default=str(VC.DATA_OUT))
    y = sub.add_parser("yaml"); y.add_argument("--work", required=True); y.add_argument("--images", required=True); y.add_argument("--seed", type=int, required=True)
    y.add_argument("--out-dir", required=True); y.add_argument("--max-epoch", type=int, default=0)
    s = sub.add_parser("stage"); s.add_argument("--dest", required=True); s.add_argument("--roles", nargs="+", required=True); s.add_argument("--data-dir", default=str(VC.DATA_OUT))
    e = sub.add_parser("evalset"); e.add_argument("--roles", nargs="+", required=True); e.add_argument("--out", required=True); e.add_argument("--data-dir", default=str(VC.DATA_OUT))
    sr = sub.add_parser("stage-rank"); sr.add_argument("--dest", required=True); sr.add_argument("--inputs", default=str(VC.DATA_OUT / "genai_rank_inputs.jsonl"))
    er = sub.add_parser("evalset-rank"); er.add_argument("--out", required=True); er.add_argument("--inputs", default=str(VC.DATA_OUT / "genai_rank_inputs.jsonl"))
    a = ap.parse_args(argv)
    return {"prep": cmd_prep, "yaml": cmd_yaml, "stage": cmd_stage, "evalset": cmd_evalset, "stage-rank": cmd_stage_rank, "evalset-rank": cmd_evalset_rank}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
