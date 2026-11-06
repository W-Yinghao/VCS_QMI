"""CR6-B TAIL_LORA8 student on a FIXED CR4 graph, trained with the CR4/CR5 COMPLETE_PC task loss.

Same as cr5_student COMPLETE_PC (same FixedRelation, same sampler stream default_rng(sampler_seed), same 64-edge draws,
same COMPLETE candidate build, same packed_loss(kind='PC'), same new G = GlobalScorer(dim, a, b, head_seed) from the
condition's student_common_initialization, AdamW wd 1e-4, cosine lr, global clip 2, 8000 updates, saves 0/250/.../8000)
EXCEPT the features: every image/text that appears in the step's unique (image, text) pairs is encoded with the CURRENT
SigLIP2 tail (blocks 10-11 + final LN + pooler, LoRA r8/alpha8 on q_proj/v_proj of both towers, NR4 injector, adapter seed),
reading blocks 0-9 from the verified NR4 hidden cache (bank index == NR1 FIT cache index == hidden cache index, checked).
Query and all candidates use the same current encoder; no stale features across steps; no positive truncation.
Backward = exact two-pass embedding-gradient replay (NR4 algorithm): no-grad encode -> loss.backward() to the embedding
leaves (+ G grads) -> per-32-chunk re-forward with grad and backward of the cached embedding gradient; then ONE global clip
(G + LoRA, max-norm 2) and ONE step. Head: AdamW lr 1e-4 -> 1e-5 cosine (CosineAnnealingLR, identical to CR1 optim_loop);
LoRA: AdamW lr 1e-5 -> 1e-6 cosine (separate AdamW instance = per-parameter identical to a second param group).
No NCE prefix, no HREF, no NR task head, no teacher, no candidate refresh. Truth is never read.
"""
from __future__ import annotations
import hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.sparse import load_npz
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                   # noqa: E402
import cr1_prod as C1                                                    # noqa: E402
from cr1.models import GlobalScorer                                      # noqa: E402
from reference.support_complete import FixedRelation                    # noqa: E402
from reference.positive_exclusive import packed_loss                    # noqa: E402
from nr4.tail_encoder import load_siglip, TailEncoder, base_weight_digest, _pad   # noqa: E402

HEAD_LR, HEAD_MIN, LORA_LR, LORA_MIN, WD, CLIP = 1e-4, 1e-5, 1e-5, 1e-6, 1e-4, 2.0


class Hidden:
    """Blocks 0-9 hidden cache (NR4 HIDDEN_MANIFEST PASS) -> current tail features."""
    def __init__(self, enc, cond, ram=None):
        d = K.NR4_RAW/K.DATASET[cond]/'fit'; self.manifest = K.read_json(d/'HIDDEN_MANIFEST.json')
        if self.manifest['status'] != 'PASS': raise ValueError('hidden cache not verified')
        ram = ram if ram is not None else os.environ.get('CR6_HIDDEN_RAM', 'none')
        t = time.perf_counter()
        self.H = dict(image=np.load(d/'img_h10_f32.npy', mmap_mode=None if ram == 'all' else 'r'),
                      text=np.load(d/'txt_h10_f32.npy', mmap_mode=None if ram == 'all' else 'r'))
        self.load_seconds = time.perf_counter() - t; self.enc = enc; self.dir = d; self.ram = ram
        self.manifest_sha256 = K.sha256(d/'HIDDEN_MANIFEST.json')

    def chunks(self, kind, rows):
        for s in range(0, len(rows), 32):
            r = rows[s:s+32]
            h, n = _pad(torch.from_numpy(np.ascontiguousarray(self.H[kind][r])).to(self.enc.dev)); yield h, n

    def encode(self, kind, rows, grad=False):
        with torch.set_grad_enabled(grad):
            return torch.cat([self.enc.tail(kind, h, n) for h, n in self.chunks(kind, rows)])


def unique_items(pairs):
    ui, ii = np.unique(pairs[:, 0], return_inverse=True); ut, it = np.unique(pairs[:, 1], return_inverse=True)
    return ui, ii, ut, it


def g_scores(G, U, V, ii, it, chunk=4096):
    a = torch.as_tensor(ii, device=U.device); b = torch.as_tensor(it, device=U.device)
    return torch.cat([G.pairs(U[a[k:k+chunk]], V[b[k:k+chunk]]) for k in range(0, len(a), chunk)])


def tail_loss_backward(G, hid, pb, mode='replay', chunk=4096):
    """Returns (loss, info). Accumulates grads into G and LoRA; caller clips + steps once."""
    ui, ii, ut, it = unique_items(pb.pairs)
    if mode == 'direct':
        U = hid.encode('image', ui, True); V = hid.encode('text', ut, True)
        loss = packed_loss(g_scores(G, U, V, ii, it, chunk), pb, kind='PC'); loss.backward()
        return loss.detach(), dict(unique_images=int(len(ui)), unique_texts=int(len(ut)), replay_max_abs=0.0)
    with torch.no_grad(): U0 = hid.encode('image', ui, False); V0 = hid.encode('text', ut, False)
    U = U0.clone().requires_grad_(True); V = V0.clone().requires_grad_(True)
    loss = packed_loss(g_scores(G, U, V, ii, it, chunk), pb, kind='PC'); loss.backward()
    err = 0.0
    for kind, rows, g, ref in (('image', ui, U.grad.detach(), U0), ('text', ut, V.grad.detach(), V0)):
        o = 0
        for h, n in hid.chunks(kind, rows):
            z = hid.enc.tail(kind, h, n); err = max(err, float((z.detach() - ref[o:o+n]).abs().max()))
            torch.autograd.backward(z, g[o:o+n]); o += n
    if err > 1e-6: raise RuntimeError(f'replay mismatch {err}')
    return loss.detach(), dict(unique_images=int(len(ui)), unique_texts=int(len(ut)), replay_max_abs=err)


class TailStudent:
    def __init__(self, cond, relation_path, head_seed, adapter_seed, device='cuda', hid=None, siglip=None):
        self.det = C1.setup_determinism(device); torch.backends.mha.set_fastpath_enabled(False)
        self.cond, self.dev = cond, device
        self.bank, self.prep, _ = C1.get_prepared(K.bank_path(cond), K.prep_dir(cond))
        self.relation_path = Path(relation_path); self.relation_sha0 = K.sha256(relation_path)
        mat = load_npz(relation_path).tocsr()
        if mat.shape != (len(self.bank.text_global), len(self.bank.image_global)): raise ValueError('shape')
        self.relation = C1.ExposedRelation(mat); self.fr = FixedRelation(mat)
        if not (np.array_equal(self.fr.text_edges, self.relation.text_edges) and np.array_equal(self.fr.image_edges, self.relation.image_edges)):
            raise AssertionError('edge enumeration differs')
        cal = self.prep['student_common_initialization']; self.cal = cal
        self.G = GlobalScorer(self.bank.dim, cal['a'], cal['b'], int(head_seed)).to(device)
        if siglip is None: siglip, self.linfo = load_siglip(device, lora=True, seed=int(adapter_seed))
        else: self.linfo = None
        self.siglip = siglip; self.enc = TailEncoder(siglip, None, None, device)
        self.hid = hid if hid is not None else Hidden(self.enc, cond)
        self.hid.enc = self.enc
        if self.hid.H['image'].shape[0] != len(self.bank.image_global) or self.hid.H['text'].shape[0] != len(self.bank.text_global): raise ValueError('hidden size')
        self.lora = [(n, p) for n, p in siglip.named_parameters() if p.requires_grad]
        if sum(p.numel() for _, p in self.lora) != 98304: raise RuntimeError('LoRA parameter count')
        self.head_seed, self.adapter_seed = int(head_seed), int(adapter_seed)

    def adapter_state(self): return {n: p.detach().cpu().clone() for n, p in self.lora}

    def load_adapter(self, sd):
        for n, p in self.lora: p.data.copy_(sd[n].to(p.device))

    def frozen_features(self):
        return C1.features(self.bank, self.dev)


def train(cond, relation_path, out, unit, head_seed, sampler_seed, adapter_seed, device='cuda', steps=8000, max_steps=None,
          saves=K.SAVES, mode='replay', student=None, resume_every=500):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed TAIL student exists; do not overwrite')
    S = student or TailStudent(cond, relation_path, head_seed, adapter_seed, device)
    G, lora = S.G, [p for _, p in S.lora]
    optH = torch.optim.AdamW(G.parameters(), lr=HEAD_LR, weight_decay=WD); schH = torch.optim.lr_scheduler.CosineAnnealingLR(optH, T_max=steps, eta_min=HEAD_MIN)
    optA = torch.optim.AdamW(lora, lr=LORA_LR, weight_decay=WD); schA = torch.optim.lr_scheduler.CosineAnnealingLR(optA, T_max=steps, eta_min=LORA_MIN)
    trainable = list(G.parameters()) + lora
    opt_ids = {id(p) for o in (optH, optA) for g in o.param_groups for p in g['params']}
    req = {id(p) for p in G.parameters() if p.requires_grad} | {id(p) for n, p in S.siglip.named_parameters() if p.requires_grad}
    if opt_ids != req: raise RuntimeError('optimizer coverage != trainable set')
    rng = np.random.default_rng(int(sampler_seed))
    digest0 = base_weight_digest(S.siglip)
    binding = dict(unit=unit, protocol=K.PROTOCOL_ID, scope='TAIL_LORA8', consumer='COMPLETE_PC', condition=cond, bank_sha256=S.bank.digest,
                   prep_sha256=K.sha256(K.prep_dir(cond)/'PREP.json'), relation_sha256=S.relation_sha0, init=S.cal, head_seed=int(head_seed),
                   sampler_seed=int(sampler_seed), adapter_seed=int(adapter_seed), steps=steps, hidden_manifest_sha256=S.hid.manifest_sha256,
                   lr=dict(head=[HEAD_LR, HEAD_MIN], lora=[LORA_LR, LORA_MIN]), weight_decay=WD, clip=CLIP, backward=mode,
                   lora=dict(rank=8, alpha=8, blocks=[10, 11], targets=['q_proj', 'v_proj'], towers=['image', 'text'], params=98304))
    bsha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    saves = sorted({s for s in saves if s <= steps} | {0, steps})

    def state(step):
        return dict(model=G.state_dict(), adapter=S.adapter_state(), step=step, dimension=G.global_residual[0].in_features//4, architecture='G',
                    scope='TAIL_LORA8', optH=optH.state_dict(), optA=optA.state_dict(), schH=schH.state_dict(), schA=schA.state_dict(),
                    rng_numpy=rng.bit_generator.state, rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None, binding=binding, binding_sha256=bsha)

    def save(step, name=None):
        p = out/(name or f'step_{step:06d}.pt'); tmp = p.with_suffix('.tmp'); torch.save(state(step), tmp); tmp.replace(p)
    cands = sorted(out.glob('step_*.pt')) + ([out/'RESUME.pt'] if (out/'RESUME.pt').exists() else []); start = 0; resumed = None
    if cands:
        cks = [torch.load(c, map_location=device, weights_only=False) for c in cands]; ck = max(cks, key=lambda c: c['step'])
        if ck['binding_sha256'] != bsha: raise ValueError('Resume binding mismatch')
        G.load_state_dict(ck['model']); S.load_adapter(ck['adapter']); optH.load_state_dict(ck['optH']); optA.load_state_dict(ck['optA'])
        schH.load_state_dict(ck['schH']); schA.load_state_dict(ck['schA']); rng.bit_generator.state = ck['rng_numpy']
        torch.set_rng_state(ck['rng_torch'].cpu())
        if ck['rng_cuda'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ck['rng_cuda']])
        start = resumed = ck['step']; del cks
    else:
        save(0)
    log = out/'TRAIN.jsonl'
    if log.exists():
        keep = [l for l in log.read_text().splitlines() if l.strip() and json.loads(l)['step'] <= start]; log.write_text(''.join(k+'\n' for k in keep))
    init_head = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}; init_ad = S.adapter_state()
    edge_hits = np.zeros(len(S.relation.edge_mass), np.int64); st = dict(n=0, pairs=0, ui=0, ut=0, build=0.0, maxui=0, maxut=0)
    stop = steps if max_steps is None else int(max_steps); step0 = None
    if str(device).startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with open(log, 'a') as f:
        for step in range(start+1, stop+1):
            optH.zero_grad(set_to_none=True); optA.zero_grad(set_to_none=True)
            b = S.relation.batch_edges(rng, 64); np.add.at(edge_hits, b['edges'], 1); st['n'] += 1
            tb = time.perf_counter(); d = S.fr.from_edge_indices(b['edges'])
            if not (np.array_equal(d.images, b['images']) and np.array_equal(d.texts, b['texts'])): raise AssertionError('query set differs')
            pb = S.fr.build(d, 'COMPLETE'); st['build'] += time.perf_counter()-tb; st['pairs'] += len(pb.pairs)
            if step == 1 and start == 0:   # step-0 contract: zero-LoRA TAIL scores vs FROZEN (cached bank globals), same head, same batch
                X, Y = S.frozen_features()
                with torch.no_grad():
                    ui, ii, ut, it = unique_items(pb.pairs)
                    U = S.hid.encode('image', ui); V = S.hid.encode('text', ut)
                    sf = g_scores(G, X[torch.as_tensor(ui, device=X.device)], Y[torch.as_tensor(ut, device=X.device)], ii, it)
                    st_ = g_scores(G, U, V, ii, it)
                    step0 = dict(score_max_abs=float((sf-st_).abs().max()), image_feat_max_abs=float((U - X[torch.as_tensor(ui, device=X.device)]).abs().max()),
                                 text_feat_max_abs=float((V - Y[torch.as_tensor(ut, device=X.device)]).abs().max()), pairs=int(len(pb.pairs)))
                del X, Y
                step0['score_tolerance'] = 1e-5*max(1.0, float(G.log_scale.exp()))   # scores are a*cos+b+r: tolerance scales with a
                if max(step0['image_feat_max_abs'], step0['text_feat_max_abs']) > 1e-5 or step0['score_max_abs'] > step0['score_tolerance']:
                    raise RuntimeError(f'STEP0 TAIL != FROZEN start {step0}')
            loss, info = tail_loss_backward(G, S.hid, pb, mode)
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss at {step}')
            st['ui'] += info['unique_images']; st['ut'] += info['unique_texts']; st['maxui'] = max(st['maxui'], info['unique_images']); st['maxut'] = max(st['maxut'], info['unique_texts'])
            if step == 1 and start == 0:
                gb = {n: (float(p.grad.abs().max()) if p.grad is not None else None) for n, p in S.lora}
                if not any((v or 0) > 0 for k, v in gb.items() if k.endswith('.B')): raise RuntimeError('STEP0 LoRA B gradients all zero')
                K.write_json(out/'STEP0_CHECK.json', dict(step0_vs_frozen=step0, loss=float(loss), lora_grad_absmax=gb,
                             head_grad_absmax={n: float(p.grad.abs().max()) for n, p in G.named_parameters()}))
            gn = torch.nn.utils.clip_grad_norm_(trainable, CLIP, error_if_nonfinite=True)
            optH.step(); optA.step(); schH.step(); schA.step()
            if step == 1 or step % 100 == 0 or step == steps:
                ag = math.sqrt(sum(float((p.grad.double()**2).sum()) for p in lora if p.grad is not None))
                rec = dict(step=step, loss=float(loss), grad_norm=float(gn), adapter_grad_norm=ag, lr_head=schH.get_last_lr()[0], lr_lora=schA.get_last_lr()[0],
                           pairs=int(len(pb.pairs)), seconds=time.perf_counter()-t0, B_absmax=max(float(p.detach().abs().max()) for n, p in S.lora if n.endswith('.B')), **info)
                f.write(json.dumps(rec)+'\n'); f.flush()
            if step in saves: save(step)
            elif step % resume_every == 0: save(step, 'RESUME.pt')
    if stop < steps:
        if stop > start and stop not in saves: save(stop, 'RESUME.pt')
        return dict(partial=True, step=stop, resumed_from=resumed)
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    secs = time.perf_counter() - t0; n = max(st['n'], 1)
    head_change = {k: float((v.detach().cpu() - init_head[k]).abs().max()) for k, v in G.state_dict().items()}
    ad_change = {k: float((v - init_ad[k]).abs().max()) for k, v in S.adapter_state().items()}
    man = dict(scope='TAIL_LORA8', head_trainable_count=sum(p.numel() for p in G.parameters()), adapter_trainable_count=sum(p.numel() for p in lora),
               optimizer_covers_exactly_trainable=opt_ids == req, head_change_max_abs=head_change, adapter_change_max_abs=ad_change,
               all_trainable_changed=bool(all(v > 0 for v in head_change.values()) and all(v > 0 for v in ad_change.values())) if resumed is None else None,
               base_weight_digest_before=digest0, base_weight_digest_after=base_weight_digest(S.siglip),
               relation_sha256_before=S.relation_sha0, relation_sha256_after=K.sha256(relation_path), tail_targets=(S.linfo or {}).get('target_paths'))
    man['base_weights_unchanged'] = man['base_weight_digest_before'] == man['base_weight_digest_after']
    man['relation_unchanged'] = man['relation_sha256_before'] == man['relation_sha256_after']
    if not (man['base_weights_unchanged'] and man['relation_unchanged']): raise RuntimeError('base weights or CSR changed')
    K.write_json(out/'MODULE_AND_GRAD_MANIFEST.json', man)
    receipt = dict(stage='student', unit=unit, scope='TAIL_LORA8', consumer='COMPLETE_PC', condition=cond, head_seed=int(head_seed), sampler_seed=int(sampler_seed),
                   adapter_seed=int(adapter_seed), relation_sha256=S.relation_sha0, updates=steps, updates_this_attempt=steps-start, resumed_from_step=resumed,
                   binding_sha256=bsha, saved_steps=saves, checkpoint='step_008000.pt' if steps == 8000 else f'step_{steps:06d}.pt',
                   checkpoint_sha256=K.sha256(out/f'step_{steps:06d}.pt'), determinism=S.det, hidden_cache=dict(manifest_sha256=S.hid.manifest_sha256, ram=S.hid.ram),
                   same_draw_receipt={'edge_hits_sha256': hashlib.sha256(edge_hits.astype(np.int64).tobytes()).hexdigest()} if resumed is None else None,
                   cost=dict(synchronized_train_seconds=secs, mean_pair_scores_per_update=st['pairs']/n, mean_unique_images=st['ui']/n, mean_unique_texts=st['ut']/n,
                             max_unique_images=st['maxui'], max_unique_texts=st['maxut'], complete_build_cpu_seconds=st['build'],
                             peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()) if str(device).startswith('cuda') else None),
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, step0_vs_frozen=step0)
    np.save(out/'EDGE_EXPOSURE.npy', edge_hits.astype(np.int32))
    K.write_json(out/'DONE.json', receipt)
    return receipt
