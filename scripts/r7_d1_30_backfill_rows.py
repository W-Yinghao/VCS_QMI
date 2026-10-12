"""Worker tool for execution note E3 (data record, not pipeline code): copy the per-row counter records that SparseVLM shards printed before
E3 (they stopped on the old exact check after writing every answer of the measured rows) into the shard's E3 rows file, with the E3 fields
recomputed (deviation, 0.5-token tolerance, float32 explanation).  Run only when the shard's job has ended (single writer).
    python backfill_sparsevlm_rows.py --repo ssl_pilot --pred-root outputs/R7_D1/d1_30/pred --log slurm_logs/r7d30v_gqa_dev_0_1035245.out [--dry-run]"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--pred-root", required=True); ap.add_argument("--log", nargs="+", required=True)
ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args(); sys.path.insert(0, str(Path(a.repo) / "scripts"))
import r7_d1_30_sparsevlm as SV  # noqa: E402

for log in a.log:
    recs = []
    for line in Path(log).read_text().splitlines():
        if line.startswith("{") and '"counter_check"' in line:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "equal_visual_tokens" not in r:
                continue
            cnt = {"equal_visual_tokens": r["equal_visual_tokens"], "official_forwards": r["official_forwards"]}
            r.update(SV.counter_fields(cnt, r["predicted_equal_tokens"])); r["backfilled_from"] = Path(log).name; recs.append(r)
    if not recs:
        print(f"{log}: no row records"); continue
    bench, shard = recs[0]["bench"], recs[0]["shard"]; f = SV.rows_file(Path(a.pred_root) / bench, bench, shard)
    print(f"{Path(log).name}: {len(recs)} rows -> {f} " + ", ".join(f"{r['config'].split(':')[1]}{':ocr' if r['config'].endswith('ocr') else ''} "
                                                                     f"dev {r['counter_deviation']:+.6f} {'ok' if r['counter_check'] else 'FAIL'}"
                                                                     f"{' f32' if r['deviation_is_float32_accumulation'] else ''}" for r in recs))
    if not a.dry_run:
        with open(f, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
