#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python - "$ROOT/docs/02_SERVER_PLAN_CN.md" "$TMP/print.md" <<'PYPRINT'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(encoding='utf-8')
s=s.replace('\n# 5. R5-30', '\n\\Needspace{23\\baselineskip}\n\n# 5. R5-30')
Path(sys.argv[2]).write_text(s,encoding='utf-8')
PYPRINT
pandoc "$TMP/print.md" --from markdown+tex_math_dollars --standalone \
  -V documentclass=article -V papersize=a4 -V fontsize=11pt -V geometry:margin=21mm \
  -V mainfont='DejaVu Serif' -V sansfont='DejaVu Sans' -V monofont='DejaVu Sans Mono' \
  -V linestretch=1.05 --include-in-header="$ROOT/reference/pdf_header.tex" \
  -o "$TMP/plan.tex"
(cd "$TMP"; xelatex -interaction=nonstopmode -halt-on-error plan.tex >/dev/null; xelatex -interaction=nonstopmode -halt-on-error plan.tex >/dev/null)
cp "$TMP/plan.pdf" "$ROOT/VCS_Server_Plan_r5_CN.pdf"
# Preserve relevant layout diagnostics rather than intermediate LaTeX artifacts.
grep -E 'Overfull|Underfull|Missing character|Warning' "$TMP/plan.log" > "$ROOT/validation/pdf_build_warnings.txt" || true
