import numpy as np, json, sys
from pathlib import Path
out=Path(sys.argv[1])
for ds,seed in (('Flickr30K',1),('COCO',2)):
    r=np.random.default_rng(seed); n=6000; c=2; D=32
    X=r.normal(size=(n,D)); truth=np.repeat(np.arange(n),c); Y=X[truth]*0.6+r.normal(size=(n*c,D))
    d=out/'curator_only/banks'/ds; d.mkdir(parents=True,exist_ok=True)
    np.savez(d/'bank_clean.npz',image_features=X.astype(np.float32),text_features=Y.astype(np.float32),true_image_index=truth,
             image_ids=np.array([f'i{k}' for k in range(n)]),text_ids=np.array([f't{k}' for k in range(n*c)]))
    import hashlib
    h=hashlib.sha256(open(d/'bank_clean.npz','rb').read()).hexdigest()
    json.dump({'annotation_role':'benchmark_clean','has_task_adapter':False,'source_split':'FIT','encoder':'google/siglip2-base-patch16-224','revision':'SYNTHETIC_TEST','bank_sha256':h},open(d/'bank_meta.json','w'))
