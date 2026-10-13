"""Diagnostic (2026-10-13): stage the Qwen2.5-VL checkpoint as the R7-40 host does and load AutoProcessor with the frozen pixel bounds;
print the resulting image-processor size, the staging note, TMPDIR and the staged preprocessor config.  No data read."""
import os, sys
sys.path.insert(0, "/home/infres/yinwang/CS_QMI/ssl_pilot/scripts")
import r7_vl_common as V, r7_d1_40_host as QH
from transformers import AutoProcessor, __version__
print("host", os.uname().nodename, "TMPDIR", os.environ.get("TMPDIR"), "transformers", __version__)
qdir, tmp, note = V.stage(QH.QWEN_DIR); print("stage note:", note, "| load dir:", qdir)
p = AutoProcessor.from_pretrained(str(qdir), min_pixels=QH.MIN_PIXELS, max_pixels=QH.MAX_PIXELS)
print("processor", type(p.image_processor).__name__, "size", p.image_processor.size, "expected", (QH.MIN_PIXELS, QH.MAX_PIXELS))
print("staged files:", sorted(os.listdir(qdir))[:20])
cfg = os.path.join(qdir, "preprocessor_config.json"); print("preprocessor_config:", open(cfg).read()[:400] if os.path.exists(cfg) else "MISSING")
V.unstage(tmp)
