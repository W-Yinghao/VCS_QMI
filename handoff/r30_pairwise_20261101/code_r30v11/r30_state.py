"""R30 v1.1 training-state fingerprint (docs/STATE_AND_REUSE_CN.md §1-§2). Pure function of a full state dict in the
r25_co_train.save_full_state layout {'epoch','best_rsum','model_A','model_B','all_loss','history','opt','rng',...}:
per-component sha256 over every tensor (dtype, shape, bytes), array, scalar and container, recursively; 'opt' and 'history'
are metadata and excluded. A live state (models in memory) is fingerprinted through the same layout
(r25_co_train.model_full_state + rng_state), so a fork can be checked component by component against its parent file."""
import hashlib, json
import numpy as np
import torch

COMPONENTS_MODEL = ('img_enc', 'txt_enc', 'optimizer', 'Eiters', 'avg_clean_prob', 'i2t_refiner', 't2i_refiner', 'n_param_groups',
                    'queue', 'r27_queue_ids')


def _feed(h, x):
    if torch.is_tensor(x):
        t = x.detach().cpu().contiguous()
        h.update(f'T{t.dtype}{tuple(t.shape)}'.encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    elif isinstance(x, np.ndarray):
        h.update(f'A{x.dtype}{x.shape}'.encode()); h.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, dict):
        h.update(b'D')
        for k in sorted(x, key=lambda z: str(z)):
            h.update(str(k).encode()); _feed(h, x[k])
    elif isinstance(x, (list, tuple)):
        h.update(f'L{len(x)}'.encode())
        for v in x:
            _feed(h, v)
    elif isinstance(x, (np.floating, np.integer)):
        h.update(f'S{x.dtype}{x.item()!r}'.encode())
    else:
        h.update(f'P{type(x).__name__}{x!r}'.encode())


def digest(x):
    h = hashlib.sha256(); _feed(h, x)
    return h.hexdigest()


def fingerprint(st):
    """{'components': {name: sha256}, 'overall': sha256} of the training state (metadata excluded)."""
    comp = {}
    for key in ('model_A', 'model_B'):
        for c in COMPONENTS_MODEL:
            if c in st[key]:
                comp[f'{key}.{c}'] = digest(st[key][c])
    for r in ('python', 'numpy', 'torch', 'cuda'):
        comp[f'rng.{r}'] = digest(st['rng'][r])
    for k in ('epoch', 'best_rsum', 'all_loss'):
        comp[k] = digest(st[k])
    return dict(components=comp, overall=hashlib.sha256(json.dumps(comp, sort_keys=True).encode()).hexdigest())


def live_state(R, model_A, model_B, epoch, best_rsum, all_loss):
    return dict(model_A=R.model_full_state(model_A), model_B=R.model_full_state(model_B), rng=R.rng_state(),
                epoch=epoch, best_rsum=best_rsum, all_loss=all_loss)


def compare(a, b):
    ka, kb = a['components'], b['components']
    return dict(equal=(ka == kb), missing=sorted(set(ka) ^ set(kb)), differing=sorted(k for k in set(ka) & set(kb) if ka[k] != kb[k]))
