"""NR2C global-only SigLIP2 export (architecture G needs only the global vectors).
Global path = nr1.extract_siglip2 unchanged (same model/revision, eager fp32, TF32 off, fixed forward batch B=32,
same processor calls, same official-logit reconstruction and single-vs-batch checks, incl. last_hidden_state).
Local token arrays are NOT stored: image_local/text_local are 1-wide zero placeholders and text_mask is all-True,
only so that nr1.io.FeatureBank shape contracts hold. Manifest field local_features marks the cache G-only;
any GL use of this cache is invalid.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps
from nr1.io import jsonl, sha256, write_json
from nr1.extract_siglip2 import vision, text, content_mask, _save_memmap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--texts', required=True)
    ap.add_argument('--split', choices=['fit', 'dev'], required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--revision', required=True); ap.add_argument('--model-id', default='google/siglip2-base-patch16-224')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--local-only', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision): raise ValueError('full revision required')
    from transformers import AutoModel, AutoProcessor
    import transformers
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False   # NR1 amendment A1 (1)
    model = AutoModel.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=False,
        local_files_only=args.local_only, attn_implementation='eager', dtype=torch.float32).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=False, local_files_only=args.local_only)
    if model.config.model_type != 'siglip': raise ValueError('FixRes SiglipModel required')
    for p in model.parameters(): p.requires_grad_(False)
    ims, txt = jsonl(args.images), jsonl(args.texts)
    for records, keys in [(ims, {'id', 'path'}), (txt, {'id', 'text'})]:
        if not records or any(set(r) != keys for r in records): raise ValueError(f'Expected schema {keys}')
        if len({r['id'] for r in records}) != len(records): raise ValueError('Duplicate item ID.')
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists(): raise FileExistsError('Cache exists; verify/reuse rather than overwrite.')
    write_json(root/'image_ids.json', [r['id'] for r in ims]); write_json(root/'text_ids.json', [r['id'] for r in txt])
    small_images = []
    for r in ims[:min(4, len(ims))]:
        with Image.open(r['path']) as im: small_images.append(ImageOps.exif_transpose(im).convert('RGB').copy())
    small_text = [r['text'].lower() for r in txt[:min(4, len(txt))]]
    inputs = processor(text=small_text, images=small_images, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    inputs = {k: v.to(args.device) for k, v in inputs.items()}; B = args.batch_size; sb = 0.0
    with torch.inference_mode():
        outputs = model(**inputs)
        vp, vh = vision(model, inputs['pixel_values'], B)
        tk = {k: inputs[k] for k in ('input_ids', 'attention_mask', 'position_ids') if k in inputs}
        tp_, th = text(model, tk, B)
        logits = (F.normalize(vp.float(), dim=-1) @ F.normalize(tp_.float(), dim=-1).T) * model.logit_scale.exp() + model.logit_bias
        if not torch.allclose(logits, outputs.logits_per_image, atol=2e-5, rtol=2e-5): raise RuntimeError('Official global logit reconstruction failed.')
        v1p, v1h = vision(model, inputs['pixel_values'][:1], B); t1p, t1h = text(model, {k: v[:1] for k, v in tk.items()}, B)
        for a, b in [(v1p, vp[:1]), (t1p, tp_[:1]), (v1h, vh[:1]), (t1h, th[:1])]:
            if not torch.allclose(a, b, atol=3e-5, rtol=3e-5): raise RuntimeError('Single/batch encoding mismatch.')
            sb = max(sb, float((a-b).abs().max()))
    dim = int(vp.shape[-1])
    ig = _save_memmap(root, 'image_global.npy', len(ims), (dim,), np.float32)
    tg = _save_memmap(root, 'text_global.npy', len(txt), (dim,), np.float32)
    with torch.inference_mode():
        for s in range(0, len(ims), B):
            images = []
            for r in ims[s:s+B]:
                with Image.open(r['path']) as im: images.append(ImageOps.exif_transpose(im).convert('RGB').copy())
            inp = processor(images=images, return_tensors='pt')
            if set(inp) != {'pixel_values'}: raise ValueError('Unexpected image processor contract.')
            po, _ = vision(model, inp['pixel_values'].to(args.device), B)
            ig[s:s+len(images)] = F.normalize(po.float(), dim=-1).cpu().numpy()
        for s in range(0, len(txt), B):
            texts = [r['text'].lower() for r in txt[s:s+B]]
            inp = processor(text=texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
            content_mask(processor.tokenizer, texts, inp['input_ids'])   # same empty-caption / tokenizer guard as NR1
            kw = {k: inp[k].to(args.device) for k in ('input_ids', 'attention_mask', 'position_ids') if k in inp}
            po, _ = text(model, kw, B)
            tg[s:s+len(texts)] = F.normalize(po.float(), dim=-1).cpu().numpy()
    ig.flush(); tg.flush()
    np.save(root/'image_local.npy', np.zeros((len(ims), 1, dim), np.float16))
    np.save(root/'text_local.npy', np.zeros((len(txt), 1, dim), np.float16))
    np.save(root/'text_mask.npy', np.ones((len(txt), 1), np.bool_))
    files = {p.name: sha256(p) for p in root.iterdir() if p.suffix in ('.npy', '.json')}
    write_json(root/'manifest.json', dict(protocol_id='NR1_SAMPLEWISE_L2_V1', split=args.split, normalization='sample_l2_no_center',
        model_id=args.model_id, model_revision=args.revision, transformers_version=transformers.__version__, torch_version=torch.__version__,
        images_manifest_sha256=sha256(args.images), texts_manifest_sha256=sha256(args.texts),
        local_features='NOT STORED (G-only cache): image_local/text_local are 1-wide zero placeholders, text_mask all True; invalid for GL',
        backbone_global_check='passed', samplewise_small_check='passed', synthetic=False,
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        fixed_forward_batch=B, single_vs_batch_max_abs=sb, extractor='nr2.extract_global (global path = nr1.extract_siglip2 functions)',
        gpu=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else None, files=files))


if __name__ == '__main__':
    main()
