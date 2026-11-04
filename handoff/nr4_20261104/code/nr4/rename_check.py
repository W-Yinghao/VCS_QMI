#!/usr/bin/env python3
"""Supplement to acceptance D: rename+reorder of the DEV gallery; check that every order difference (after mapping back) is a swap
among items with bitwise-identical features (exact-duplicate captions), i.e. only the ID tie-break changes. Uses the trained
acceptance TAIL checkpoint's head on the frozen DEV cache (the duplicate-tie mechanism is feature-independent)."""
import json, numpy as np, torch
from types import SimpleNamespace
from nr1.io import FeatureBank, write_json
from nr1.models import DirectScorer
from nr4.evaluate import score_and_order
from nr4.paths import paths, OUT
P = paths('Flickr60'); b = FeatureBank(P['dev'], 'dev'); ck = torch.load(OUT/'acceptance/Flickr60_J/TAIL_LORA8_straight/step_008250.pt', map_location='cpu', weights_only=False)
G = DirectScorer(**ck['binding']['model_config']); G.load_state_dict(ck['model'])
x, y = torch.from_numpy(np.array(b.x)), torch.from_numpy(np.array(b.y))
_, _, _, ni, nt, *_ = score_and_order(SimpleNamespace(x=b.x, y=b.y, image_ids=b.image_ids, text_ids=b.text_ids), x, y, G)
rng = np.random.default_rng(7); pim, ptx = rng.permutation(len(x)), rng.permutation(len(y))
rb = SimpleNamespace(x=np.asarray(b.x)[pim], y=np.asarray(b.y)[ptx], image_ids=[f'r{k}' for k in range(len(x))], text_ids=[f's{k}' for k in range(len(y))])
_, _, _, ni2, nt2, *_ = score_and_order(rb, x[pim], y[ptx], G)
bi = ptx[ni2][np.argsort(pim)]; bt = pim[nt2][np.argsort(ptx)]
Y = np.asarray(b.y); _, grp = np.unique(Y, axis=0, return_inverse=True); grp = grp.ravel()
X = np.asarray(b.x); _, gi = np.unique(X, axis=0, return_inverse=True); gi = gi.ravel()
res = dict(i2t_queries_order_differs=int((bi != ni).any(1).sum()), t2i_queries_order_differs=int((bt != nt).any(1).sum()),
           i2t_orders_equal_up_to_identical_feature_groups=bool(np.array_equal(grp[bi], grp[ni])), t2i_orders_equal_up_to_identical_feature_groups=bool(np.array_equal(gi[bt], gi[nt])),
           n_text_items_in_identical_feature_groups=int(np.bincount(grp)[np.bincount(grp) > 1].sum()), n_image_items_in_identical_feature_groups=int(np.bincount(gi)[np.bincount(gi) > 1].sum()))
write_json(OUT/'acceptance/Flickr60_J/RENAME_REORDER_CHECK.json', res); print(json.dumps(res))
