"""V7-01 data audit report (formatting only): every number comes from outputs/V7/data/audit.json and, when present, the N0 native preparation
record (outputs_v7/n0/work/native_prep.json).  No dataset text is written (the 20 manual-check records stay in outputs/, not in the repo).
    python -I scripts/v7/v7_audit_report.py --audit A [--native N] --out reports/V7/V7_01_DATA_AUDIT.md"""
import argparse
import json
from pathlib import Path


def kv_table(d, k1="value", k2="count"):
    return "\n".join([f"| {k1} | {k2} |", "|---|---|"] + [f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |" for k, v in d.items()])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--audit", required=True); ap.add_argument("--native"); ap.add_argument("--out", required=True); a = ap.parse_args()
    A = json.loads(Path(a.audit).read_text()); N = json.loads(Path(a.native).read_text()) if a.native and Path(a.native).exists() else None; c = A["counts"]; L = []
    L.append("# V7-01 data audit: EvalMuse public TRAIN (2026-10-13)\n")
    L.append(f"Source: `train_list.json` sha256 `{A['source']['train_list_sha256'][:16]}…`, {A['source']['n_records']:,} records; archive {A['archive']['members']:,} members, "
             f"{A['archive']['train_members_missing']} TRAIN members missing, {A['archive']['non_train_file_members_never_read']:,} non-TRAIN files never read.  "
             f"Rules: `scripts/v7/v7_data.py` (fixed before any model output).  Manifests: `outputs/V7/data/manifest_{{images,elements}}.jsonl` (sha256 "
             f"`{A['files_sha256']['manifest_images.jsonl'][:12]}…`, `{A['files_sha256']['manifest_elements.jsonl'][:12]}…`).\n")
    L.append("## 1. Raw fields and codes\n"); L.append("| field | present | types | codes (top) |"); L.append("|---|---|---|---|")
    for k, v in A["fields"].items():
        L.append(f"| `{k}` | {v['present']:,} | {', '.join(f'{t} {n:,}' for t, n in v['types'].items())} | {', '.join(f'{x}: {n:,}' for x, n in list(v['codes'].items())[:8]) or '—'} |")
    L.append("\nElement votes use only 0 / 1; the three image flags use 0 / 1 with −1 = no answer (kept as missing, never 0).\n")
    L.append("## 2. Images, elements and label status\n")
    L.append(f"{c['images']:,} images ({c['annotators_per_image']} annotators per image), {c['elements']:,} element events; {c['negation_cue_elements']:,} elements carry a "
             f"negation cue (heuristic flag), {c['multi_span_elements']:,} match more than once in their prompt.\n")
    L.append("Image status:\n\n" + kv_table(c["image_label_status"], "status") + "\n\nElement status:\n\n" + kv_table(c["element_label_status"], "status") +
             "\n\nElement scope (main = valid label, meaningful prompt):\n\n" + kv_table(c["element_scope"], "scope") + "\n")
    L.append("Span rule that located each element in its prompt:\n\n" + kv_table(c["element_match_rules"], "rule") + "\n")
    L.append("Generators:\n\n" + kv_table(c["generators"], "generator") + "\n\nElement categories:\n\n" + kv_table({str(k): v for k, v in c["element_categories"].items()}, "category") + "\n")
    if c["problems"]:
        L.append("Problems: " + ", ".join(f"{k} {v:,}" for k, v in c["problems"].items()) + "\n")
    P = A["prompts"]; I = A["images"]; C = A["components"]
    L.append("## 3. Prompt and image identity\n")
    L.append(f"- {P['prompt_ids']:,} prompt ids; {P['prompt_ids_with_several_texts']} ids carry more than one text; {P['normalized_texts']:,} normalized texts, "
             f"{P['normalized_texts_with_several_ids']} shared by several ids.\n- Near-duplicate prompt-id pairs (token Jaccard ≥ {A['rules']['near_dup_prompt_jaccard']}): "
             f"{P['near_duplicate_prompt_id_pairs']:,}.\n- Prompt ids overlapping GenAI-Bench prompts (exact or near-duplicate): {P['prompt_ids_overlapping_external']} "
             f"→ their components are excluded from every role.\n- Images: {I['exact_duplicate_groups']} groups of byte-identical images, {I['near_duplicate_pairs']} near-duplicate "
             f"pairs (dHash ≤ {A['rules']['dhash_max_hamming']}), {I['decode_errors']} decode errors.\n- Components after merging all links: {C['n']:,} (largest {C['largest']} images).\n")
    L.append("## 4. Roles (split seed 17071; components, so the fractions are not exact)\n"); L.append("| role | images | prompt ids | components | main elements |"); L.append("|---|---|---|---|---|")
    for r, v in A["roles"].items():
        L.append(f"| {r} | {v['images']:,} | {v['prompt_ids']:,} | {v['components']:,} | {v['elements_main']:,} |")
    F = A["fit"]
    L.append(f"\nFIT prior (prompt → image → element uniform): π = {F['pi']:.4f} over all main elements, π = {F['pi_spanned']:.4f} over main elements with a span "
             f"(the M training population; unweighted mean {F['pi_unweighted']:.4f}); image weights {F['weight_min']:.3f}–{F['weight_max']:.3f}; "
             f"{F['elements_main_with_span']:,} of {F['n_elements_main']:,} FIT main elements have a span.\n")
    if "tokens" in A:
        T = A["tokens"]
        L.append(f"## 5. Prompt length (FIT, uncased BERT with [CLS] / [SEP])\n\nmax {T['max']}, p99 {T['p99']:.0f}; over 32 tokens: {T['over_32']:,} of {T['fit_prompts']:,} "
                 f"(the native 32-token cut); over 128: {T['over_128']}; over 512: {T['over_512']}.  Fast vs model (slow) tokenizer id mismatches: {T['fast_vs_slow_id_mismatch']}.\n")
    if N:
        e2m = N.get("element2mask", {})
        L.append(f"## 6. N0 native preparation (FGA's own scripts on FIT)\n\nFIT records in: {N['fit_records_in']:,}; kept by the native element-to-token mapping: "
                 f"{N['records_kept_by_native_mapping']:,}; dropped (any element without an exact token match): {N['records_dropped_by_native_mapping']:,}; "
                 f"native counts: {json.dumps(e2m)}.\n")
    Path(a.out).write_text("\n".join(L) + "\n"); print("written", a.out)


if __name__ == "__main__":
    main()
