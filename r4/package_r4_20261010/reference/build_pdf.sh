#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python - "$ROOT/docs/02_SERVER_PLAN_CN.md" "$TMP/print.md" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(encoding='utf-8')
# Keep the short scope table together; content is unchanged.
s=s.replace('\n# 7. 暂缓与保持不变的内容','\n\\newpage\n\n# 7. 暂缓与保持不变的内容')
Path(sys.argv[2]).write_text(s,encoding='utf-8')
PY
pandoc "$TMP/print.md" --from markdown+tex_math_dollars --pdf-engine=xelatex \
  -V documentclass=article -V papersize=a4 -V fontsize=11pt -V geometry:margin=21mm \
  -V mainfont='DejaVu Serif' -V sansfont='DejaVu Sans' -V monofont='DejaVu Sans Mono' \
  -V linestretch=1.16 --include-in-header="$ROOT/reference/pdf_header.tex" --standalone \
  -o "$ROOT/VCS_Server_Plan_r4_CN.pdf"
