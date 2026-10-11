#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pandoc "$ROOT/docs/02_EXECUTIVE_PLAN_CN.md" \
  --from markdown+tex_math_single_backslash --pdf-engine=xelatex \
  --include-in-header="$ROOT/reference/pdf_header.tex" \
  -V mainfont='DejaVu Sans' -V sansfont='DejaVu Sans' -V monofont='DejaVu Sans Mono' \
  -V fontsize=11pt -V geometry='a4paper,top=20mm,bottom=20mm,left=19mm,right=19mm' \
  -V linestretch=1.18 -V colorlinks=true -V urlcolor=vcsaccent \
  -o "$ROOT/VCS_VL_Research_Atlas_r7_Overview_CN.pdf"
