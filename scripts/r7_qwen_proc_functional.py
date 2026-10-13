"""Diagnostic (2026-10-13): does the Qwen processor, loaded exactly as scripts/r7_d1_40_host.py loads it, APPLY the frozen pixel bounds at
call time even though image_processor.size shows the checkpoint values?  Synthetic images only (no data read); processor files only."""
import sys
sys.path.insert(0, "/home/infres/yinwang/CS_QMI/ssl_pilot/scripts")
from PIL import Image
from transformers import AutoProcessor
import r7_d1_40_host as QH, r7_d1_40c_run as RUN
p = AutoProcessor.from_pretrained(str(QH.QWEN_DIR), min_pixels=QH.MIN_PIXELS, max_pixels=QH.MAX_PIXELS)
ip = p.image_processor
print("image_processor.size:", ip.size, "| attrs min/max:", getattr(ip, "min_pixels", None), getattr(ip, "max_pixels", None))
print("tokenizer.init_kwargs min/max:", p.tokenizer.init_kwargs.get("min_pixels"), p.tokenizer.init_kwargs.get("max_pixels"))
text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Q<|im_end|>\n"
ok = True
for w, h in ((500, 375), (640, 480), (320, 240), (1600, 1200), (2400, 900)):
    thw = p(text=[text], images=[Image.new("RGB", (w, h))], return_tensors="pt")["image_grid_thw"][0].tolist()
    got = (thw[1] // 2, thw[2] // 2)
    exp = {k: tuple(x // RUN.FACTOR for x in RUN.smart_resize(h, w, min_pixels=a, max_pixels=b))
           for k, (a, b) in {"frozen": (QH.MIN_PIXELS, QH.MAX_PIXELS), "ckpt": (3136, 12845056)}.items()}
    match = [k for k, v in exp.items() if v == got]; ok &= "frozen" in match
    print(f"{w}x{h}: processor grid {got} | frozen {exp['frozen']} | ckpt {exp['ckpt']} | matches {match}")
print("FUNCTIONAL_FROZEN_BOUNDS_APPLIED", ok)
