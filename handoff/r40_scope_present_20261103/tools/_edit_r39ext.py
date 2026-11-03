from pathlib import Path
R = Path(__file__).resolve().parents[1]
for t in ('r40f', 'r40c'):
    p = R / t / 'r39ext.py'; s = p.read_text()
    rep = [
     ('"""R39 additions,', '"""R40: R39 r39ext.py + update scope (Student(modalities): the frozen modality keeps the parent weights with requires_grad=False, so it is\nnot in trainable() / the optimizer / the clip, and its whole state is part of frozen_state(); param_distance split per modality).\nR39 additions,'),
     ("    def __init__(self, img_state, txt_state, opt):\n        super().__init__()",
      "    def __init__(self, img_state, txt_state, opt, modalities=('img', 'txt')):\n        super().__init__()\n        assert set(modalities) <= {'img', 'txt'} and len(modalities) > 0, modalities\n        self.modalities = tuple(m for m in ('img', 'txt') if m in modalities)   # R40: updated modalities (the rest stays the parent)"),
     ("        self.l2norm = V.l2norm\n        self.train()",
      "        for m in ('img', 'txt'):   # R40: frozen modality = parent parameters, requires_grad False (never in the optimizer)\n            if m not in self.modalities:\n                for q in getattr(self, m).parameters():\n                    q.requires_grad_(False)\n        self.l2norm = V.l2norm\n        self.train()"),
     ("        return {n: t for n, t in self.state_dict().items() if n == 'txt.embed.weight' or 'running_' in n or 'num_batches_tracked' in n or '.mlp.bns.1.' in n}",
      "        fm = tuple(m + '.' for m in ('img', 'txt') if m not in self.modalities)   # R40: + every tensor of the frozen modality\n        return {n: t for n, t in self.state_dict().items() if n == 'txt.embed.weight' or 'running_' in n or 'num_batches_tracked' in n or '.mlp.bns.1.' in n or n.startswith(fm)}"),
     ("def build_students(ck, device):\n    s = nn.ModuleDict({p: Student(ck[f'model_{p}'][0], ck[f'model_{p}'][1], ck['opt']) for p in PEERS}).to(device)",
      "def build_students(ck, device, modalities=('img', 'txt')):\n    s = nn.ModuleDict({p: Student(ck[f'model_{p}'][0], ck[f'model_{p}'][1], ck['opt'], modalities) for p in PEERS}).to(device)"),
     ("            sq = 0.0; sq0 = 0.0\n            for n, q in stud[p].named_parameters():\n                if q.requires_grad:\n                    r = ref_states[p][n]\n                    sq += float(((q.detach() - r) ** 2).sum()); sq0 += float((r ** 2).sum())\n            out[p] = dict(l2_from_parent=sq ** .5, rel_l2_from_parent=(sq / sq0) ** .5)",
      "            sq = 0.0; sq0 = 0.0; mod = {m: [0.0, 0.0] for m in ('img', 'txt')}\n            for n, q in stud[p].named_parameters():\n                if q.requires_grad:\n                    r = ref_states[p][n]\n                    a_, b_ = float(((q.detach() - r) ** 2).sum()), float((r ** 2).sum())\n                    sq += a_; sq0 += b_; mod[n.split('.')[0]][0] += a_; mod[n.split('.')[0]][1] += b_\n            out[p] = dict(l2_from_parent=sq ** .5, rel_l2_from_parent=(sq / sq0) ** .5,\n                          per_modality={m: dict(l2_from_parent=v[0] ** .5, rel_l2_from_parent=(v[0] / v[1]) ** .5 if v[1] > 0 else None) for m, v in mod.items()})"),
    ]
    for a, b in rep:
        assert s.count(a) == 1, (t, a[:60])
        s = s.replace(a, b)
    p.write_text(s); print(t, 'ok')
