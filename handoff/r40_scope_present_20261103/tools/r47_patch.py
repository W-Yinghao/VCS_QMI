"""R47 production patch (layout-free centring of the JOINT student; applied once to r39ext.py and train_unit.py of r47f / r47o / r47c; anchors asserted).
JointInputs.mode = JOINT_PREP_MANIFEST 'centring_mode' (default 'native_batch' = the R39 parent per-native-batch means, unchanged behaviour):
  'global'       variant G: u = l2norm(e - mu), mu = ONE parent mean per (split, modality, peer) over the full split (center_* files of shape [1, 1024]);
                 the layout-free parent cache of the condition is built with the same means, so the step-0 student reproduces the cache
  'train_batch'  variant S: training (grad) encodings centred by the mean of the student's own outputs over the unique items of the current batch
                 (per modality / peer); evaluation tables centred by the mean of the student's own outputs over fixed random batches of 512 items
                 (np.random.default_rng([4701, split, modality, peer]); independent for images and captions -> no layout grouping). The R39 step-0
                 student-vs-cache equivalence is not defined for this mode (the cache is the global-centred parent) and is recorded as skipped;
                 the step-0 two-stage DEV still has to reproduce the frozen-H ranking (head residual zero at init).
Usage: python3 tools/r47_patch.py <tree_dir>"""
import sys
from pathlib import Path
T = Path(sys.argv[1])


def patch(fn, reps):
    s = (T / fn).read_text()
    for a, b in reps:
        assert s.count(a) == 1, (fn, a[:80], s.count(a)); s = s.replace(a, b)
    (T / fn).write_text(s)


patch('r39ext.py', [
    ("""        self.n = dict(fit=dict(img=N_TRAIN_IMG, cap=N_TRAIN_CAP), dev=dict(img=N_DEV_IMG, cap=N_DEV_CAP))""",
     """        self.n = dict(fit=dict(img=N_TRAIN_IMG, cap=N_TRAIN_CAP), dev=dict(img=N_DEV_IMG, cap=N_DEV_CAP))
        self.mode = man.get('centring_mode', 'native_batch')   # R47: native_batch (R39) | global (G) | train_batch (S)
        assert self.mode in ('native_batch', 'global', 'train_batch'), self.mode"""),
    ("""                e = self.raw_encode(stud_p, split, kind, c)
                mu = self.mu[(split, kind, p)][batch_index(kind, c.to(self.dev))]
                outs.append(self.l2norm(e - mu))""",
     """                e = self.raw_encode(stud_p, split, kind, c)
                if self.mode == 'native_batch':
                    mu = self.mu[(split, kind, p)][batch_index(kind, c.to(self.dev))]
                elif self.mode == 'global':   # R47 G: one fixed parent mean per split / modality / peer
                    mu = self.mu[(split, kind, p)][:1]
                else:   # R47 S: mean of the student's own outputs over the unique items of this (training) batch
                    assert grad and len(c) == len(ids_t), 'R47 S: non-table encodings are only defined for whole training batches'
                    mu = e.mean(0, keepdim=True)
                outs.append(self.l2norm(e - mu))"""),
    ("""        if key not in self._eval:
            self._eval[key] = self.encode(stud_p, p, split, kind, np.arange(self.n[split][kind]), grad=False)
        return self._eval[key]""",
     """        if key not in self._eval:
            if self.mode == 'train_batch':   # R47 S evaluation: fixed random batches of 512 items (independent per modality), own-output means
                n = self.n[split][kind]; ids = torch.arange(n); ch = 512 if kind == 'img' else 2048
                with torch.no_grad():
                    e = torch.cat([self.raw_encode(stud_p, split, kind, ids[i:i + ch]) for i in range(0, n, ch)])
                    perm = np.random.default_rng([4701, ('fit', 'dev').index(split), ('img', 'cap').index(kind), PEERS.index(p)]).permutation(n)
                    b = np.empty(n, np.int64); b[perm] = np.arange(n) // 512; bt = torch.as_tensor(b, device=e.device); nb = int(b.max()) + 1
                    mu = torch.zeros(nb, e.shape[1], device=e.device, dtype=torch.float64).index_add_(0, bt, e.double())
                    mu = (mu / torch.bincount(bt, minlength=nb).double()[:, None]).float()
                    self._eval[key] = self.l2norm(e - mu[bt])
            else:
                self._eval[key] = self.encode(stud_p, p, split, kind, np.arange(self.n[split][kind]), grad=False)
        return self._eval[key]"""),
])
patch('train_unit.py', [
    ("""        if step == 0 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 fail-fast: student forward (grad on, training batch) == native cache""",
     """        if step == 0 and cfg.get('phase', 1) == 1 and stud is not None and ctx.joint.mode == 'train_batch':   # R47 S: equivalence to the (global-centred) cache not defined
            hist['r39']['step0_cache_equivalence'] = dict(passed=True, skipped='R47 S (train_batch centring): the student is centred by its own batch means, the cache by the global parent means')
        if step == 0 and cfg.get('phase', 1) == 1 and stud is not None and ctx.joint.mode != 'train_batch':   # R39 fail-fast: student forward (grad on, training batch) == native cache"""),
])
print('R47 patched', T)
