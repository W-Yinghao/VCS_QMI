# TIFA160 images — data card (added 2026-10-15 for the V8-05 §B secondary evaluation)

| field | value |
|---|---|
| id | `tifa160_images` |
| source | TIFA official repository https://github.com/Yushi-Hu/tifa (local read-only clone `third_party/tifa@8e37050`), README "TIFA v1.0 Human Annotations" → Google Drive file id `16y87Wg1mYwPhfZnLeBHi0HH4RiwD3a5K` (`tifa_annotated_images.zip`) |
| download | 2026-10-15 ~19:34 UTC, `curl` from `drive.usercontent.google.com/download?id=…&export=download&confirm=t` (no account); `/projects/EEG-foundation-model/yinghao/data/tifa160/download/tifa_annotated_images.zip`, 35,333,425 bytes, sha256 `6a1e283a1e6bef5cf9b785ee449097c7a83e6e0918d7d9de4291ac30f494b231` |
| contents | `annotated_images/<text id>_<model>.jpg`: 800 JPEG files (35,290,602 bytes) = 160 TIFA v1.0 text ids (84 coco, 51 partiprompt, 15 paintskill, 10 drawbench) × 5 generators {`mini_dalle`, `vq_diffusion`, `stable_diffusion_v1_1`, `stable_diffusion_v1_5`, `stable_diffusion_v2_1`}; 22 `__MACOSX` resource forks discarded; no absolute / parent paths |
| extraction | `/projects/EEG-foundation-model/yinghao/data/tifa160/images/annotated_images/`; per-file sha256 in `/projects/EEG-foundation-model/yinghao/data/tifa160/SHA256SUMS_images.txt` |
| verification | every file name maps to exactly one (text id, model) with the 160 ids of DSG's public `dsg/data/tifa160-dev-anns.csv` (`item_id`) and the captions of `tifa_v1.0/tifa_v1.0_text_inputs.json`; 160 / 160 ids have all five models |
| licence / terms | TIFA repository code: Apache-2.0 (`LICENSE`). The images are the authors' release of generated images (minDALL-E, VQ-Diffusion, Stable Diffusion v1.1 / v1.5 / v2.1 outputs for COCO / DrawBench / PartiPrompts / PaintSkills captions) accompanying the human-annotation release; no separate licence text ships with the zip (checked: the archive holds only JPEGs). Use here: non-commercial research evaluation of frozen models; the generating models' and caption sources' own terms apply; the images are not redistributed from this repository |
| labels | NOT downloaded here: the TIFA human annotations and DSG's 5-rater Likert file are already sealed under `third_party/closed/` (hashes in `SHA256SUMS_v7x_20261014.txt`); nothing opened |
| authorisation | owner 2026-10-15: 「批准读取 TIFA160 标签，作为冻结 V8 模型的次级外部评价。」 (recorded in `reports/OWNER_DECISIONS_20261015.md` on main); the image download is the data-steward step of that approval |
| use | evaluation-only benchmark: never used for training or selection; V8 models frozen; predictions on the 800 images only |
