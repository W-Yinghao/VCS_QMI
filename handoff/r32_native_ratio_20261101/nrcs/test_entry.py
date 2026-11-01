"""NRCS-v1 standard-protocol TEST entry for the main session's standing tool (handoff/test_eval_20261101). Nothing here is run
by the NRCS route itself on TEST.

1) views:  test_entry.py views
   For every finished E2/C2 unit, writes $OUT/test_view/<cell>/<arm>/ (PRIMARY: the fixed epoch-20 checkpoint) and
   $OUT/test_view/<cell>/<arm>__selJ/ (SECONDARY: the SELECT-observed-J checkpoint, earliest tie), each with
   model_best.pth.tar (keys opt/epoch/best_rsum/model/...) and DONE.json (logged full-DEV recalls of that checkpoint).
   Both rules are declared in ROOTS.json before any TEST read; DEV/TEST never select anything here.
2) evaluator: class Nrcs(dp).evaluate(u, ckpt) -> {'dev_recomputed': ..., 'test': ...} in test_eval's recall_dict format
   (Flickr30K: DEV 1014/5070 and 1K TEST 1000/5000, both directions, six Recalls, upstream IN2R i2t/t2i, per_captions=5).
   Pointwise: eval mode, frozen BN, no batch centering; score = ratio inner product (ratio arms) or head cosine (cosine arms).
"""
import argparse, ast, glob, json, os, sys
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
for p in (HERE, f'{W}/upstream', f'{W}/package/src'):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101'
VOCAB_DIR = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab'
KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')
EVAL_FILES = {'nrcs/test_entry.py': f'{HERE}/test_entry.py', 'nrcs/gpo_adapter.py': f'{HERE}/gpo_adapter.py',
              'upstream/vse.py': f'{W}/upstream/vse.py', 'upstream/glove_shim.py': f'{W}/upstream/glove_shim.py',
              'upstream/evaluation_909e5329_orig.py': f'{W}/upstream/evaluation_909e5329_orig.py',
              'package/src/native_ratio/models.py': f'{W}/package/src/native_ratio/models.py'}


def upstream_metric_fns():
    tree = ast.parse(open(f'{W}/upstream/evaluation_909e5329_orig.py').read())
    ns = {'np': np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('i2t', 't2i'):
            exec(compile(ast.Module([node], []), 'upstream_eval', 'exec'), ns)
    return ns['i2t'], ns['t2i']


def cell_arm(unit_id):
    # nrcs_v1_e2_n60_s1002_ratio_J_b1_i0p1 -> ('f60_s1002', 'e2_ratio_J_b1_i0p1')
    parts = unit_id.split('_')
    stage, n, s = parts[2], parts[3], parts[4]
    return f'f{n[1:]}_{s}', f"{stage}_{'_'.join(parts[5:])}"


def build_views(runs=None, views=None):
    import torch
    made = []
    runs = runs or f'{OUT}/runs'; views = views or f'{OUT}/test_view'
    for dp in sorted(glob.glob(f'{runs}/nrcs_v1_e2_*/DONE.json') + glob.glob(f'{runs}/nrcs_v1_c2_*/DONE.json')):
        d = json.load(open(dp)); u = d['unit_id']; rd = os.path.dirname(dp)
        cell, arm = cell_arm(u)
        for tag, src, block, sha_key, rule in (
                ('', f"{rd}/epoch{d['config']['epochs']}.pt", d['last'], 'epoch20_checkpoint_sha256', 'PRIMARY fixed epoch-20 checkpoint'),
                ('__selJ', f'{rd}/selected.pt', d['selected'], 'selected_checkpoint_sha256', 'SECONDARY SELECT-observed-J checkpoint (earliest tie)')):
            v = f'{views}/{cell}/{arm}{tag}'
            if os.path.exists(f'{v}/DONE.json'):
                continue
            os.makedirs(v, exist_ok=True)
            ck = torch.load(src, map_location='cpu', weights_only=False)
            dev = {k: block['dev']['retrieval'][k] for k in KEYS + ('rsum',)}
            dev.update(n_images=block['dev']['retrieval']['n_image_queries'], n_captions=block['dev']['retrieval']['n_text_queries'])
            opt = SimpleNamespace(data_name='f30k_precomp', noise_ratio=d['noise_percent'] / 100, seed=d['seed'], batch_size=128,
                                  vocab_path=VOCAB_DIR, num_epochs=d['config']['epochs'], embed_size=1024, arm=d['arm'],
                                  module_name='nrcs_gpo_two_tower')
            torch.save({'opt': opt, 'epoch': int(block['epoch']), 'best_rsum': float(dev['rsum']), 'model': ck['model'],
                        'nrcs_arm': d['arm'], 'nrcs_unit': u, 'checkpoint_rule': rule,
                        'source_checkpoint': src, 'source_checkpoint_sha256': d[sha_key]}, f'{v}/model_best.pth.tar')
            json.dump({'status': 'COMPLETE', 'unit_id': u, 'best_rsum': float(dev['rsum']), 'epochs_done': d['config']['epochs'],
                       'checkpoint_epoch': int(block['epoch']), 'checkpoint_rule': rule, 'logged_dev': dev,
                       'logged_dev_source': 'NRCS DONE.json full-DEV readout of this checkpoint (package retrieval_metrics; stable ties)',
                       'source_checkpoint_sha256': d[sha_key], 'gpu_model': d['cost']['gpu_model']}, open(f'{v}/DONE.json', 'w'), indent=1)
            made.append(v)
    print(f'{len(made)} views written'); [print(' ', m) for m in made]


class Nrcs:
    """Evaluator for test_eval.py (host 'nrcs'). dp = staged data root containing f30k_precomp/{dev,test}_{caps.txt,ims.npy}."""

    def __init__(self, dp):
        import torch, nltk
        import gpo_adapter as ga
        self.torch, self.nltk, self.ga, self.dp = torch, nltk, ga, dp
        self.i2t, self.t2i = upstream_metric_fns()
        self.w2i = ga.load_vocab()
        self.test_split, self.nfolds = None, 5
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

    def _split(self, split):
        caps = open(f'{self.dp}/f30k_precomp/{split}_caps.txt', encoding='utf-8').read().splitlines()
        ims = np.load(f'{self.dp}/f30k_precomp/{split}_ims.npy', mmap_mode='r')
        assert ims.shape[0] == len(caps) and len(caps) % 5 == 0
        rows = np.arange(0, len(caps), 5)
        assert all(np.array_equal(ims[r], ims[r + k]) for r in rows[:50] for k in range(1, 5))
        unk = self.w2i['<unk>']
        toks, offs = [], [0]
        for c in caps:
            ids = [self.w2i['<start>']] + [self.w2i.get(t, unk) for t in self.nltk.tokenize.word_tokenize(c.lower())] + [self.w2i['<end>']]
            toks.extend(ids); offs.append(len(toks))
        pool = SimpleNamespace(feats=self.torch.from_numpy(np.ascontiguousarray(ims[rows])).cuda(), device='cuda',
                               tokens=np.asarray(toks, np.int64), offsets=np.asarray(offs, np.int64),
                               n_img=len(rows), n_cap=len(caps), mapping=np.arange(len(caps)) // 5)
        return pool

    def _scores(self, model, pool):
        h_i, h_t = self.ga.encode_pool(model, pool, len(self.w2i), self.w2i['<mask>'])
        a, b = model.critic.retrieval_embeddings(model.heads_in(h_i), model.heads_in(h_t))
        return (a @ b.T).float().cpu().numpy()

    def _recalls(self, s):
        n = s.shape[0]
        r, ri = self.i2t(n, s, 5), self.t2i(n, s, 5)
        d = dict(zip(KEYS, map(float, (r[0], r[1], r[2], ri[0], ri[1], ri[2]))))
        d['rsum'] = float(sum(d[k] for k in KEYS)); d['i2t_medr'], d['t2i_medr'] = float(r[3]), float(ri[3])
        d.update(n_images=int(s.shape[0]), n_captions=int(s.shape[1]))
        return d

    def evaluate(self, u, ckpt):
        torch = self.torch
        model = self.ga.GPOTwoTower(ckpt['nrcs_arm'], self.w2i)
        model.load_state_dict(ckpt['model']); model.cuda().eval()
        out = {}
        with torch.no_grad():
            out['dev_recomputed'] = self._recalls(self._scores(model, self._split('dev')))
            out['test'] = self._recalls(self._scores(model, self._split(self.test_split or 'test')))
        del model; torch.cuda.empty_cache()
        return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('views',))
    a = ap.parse_args()
    build_views()
