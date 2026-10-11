#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pandoc docs/02_SERVER_PLAN_CN.md --standalone --pdf-engine=xelatex \
  --include-in-header=reference/pdf_header.tex \
  -V mainfont='Liberation Serif' -V sansfont='Liberation Sans' \
  -V monofont='DejaVu Sans Mono' -V fontsize=11pt \
  -V geometry:a4paper -V geometry:margin=20mm \
  -o VCS_Server_Plan_r6_CN.pdf
