"""R46 production patch (applied once to the copied train_unit.py of r46f (from r42f) and r46o (from r43o); every anchor asserted unique).
  R46-A (FIT-pool restriction): pool_of returns the condition's fixed record pool d['record_pool'] (records whose observed image is in the
       fixed image subset; loaded and hash-checked by data35 from the prep dir) for non-TRUST supervision; None for full-pool conditions
       (unchanged behaviour). Candidate tables / locks / calibration of the subset conditions are built within the subset (prep).
  R46-B (iterated N supervision, round 2):
    cfg['init_from'] = round-1 unit / step-8000 checkpoint (hash-bound): heads, nulls and student are loaded from it after the fresh build;
       fresh AdamW, schedule restarted (R39 JOINT cosine over the 8000 new updates); the R40 / R39 one-step and parameter-distance references are
       the loaded round-1 state; frozen student tensors (embedding / BN statistics) asserted equal to the parent's.
    cfg['candidate_file'] (REBUILT only) = round-2 natural top-32 tables built from the round-1 student (hash-bound): replace the parent tables
       in the condition context (Ctx keyed by the table hash); N labels follow from the observed relation as always.
    step-0 fail-fast for round 2: the step-0 two-stage DEV ranks (i2t and t2i) must equal the round-1 step-8000 ranks exactly and rsum equal
       (replaces the native-H step-0 check, which applies to fresh heads only); the R39 student-vs-native-cache step-0 check is replaced by
       this (the student starts from round 1, not from the parent).
Usage: python3 tools/r46_patch.py <path/to/train_unit.py>"""
import sys
p = sys.argv[1]; s = open(p).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a)); s = s.replace(a, b)


rep("    def __init__(self, cond, device, need_local, fit_maps=None):\n        self.cond, self.dev = cond, device\n        self.d = load_cond(cond, device)\n",
    """    def __init__(self, cond, device, need_local, fit_maps=None, cand_file=None):
        self.cond, self.dev = cond, device
        self.d = load_cond(cond, device)
        if cand_file:   # R46-B REBUILT: round-2 natural top-32 tables (round-1 student, same rule), hash-bound in the config
            assert sha256(cand_file['path']) == cand_file['sha256'], 'R46: round-2 candidate table changed'
            Z = np.load(cand_file['path']); assert Z['i2t_top32'].shape == self.d['cand_i2t'].shape and Z['t2i_top32'].shape == self.d['cand_t2i'].shape
            self.d = dict(self.d); self.d['cand_i2t'] = Z['i2t_top32']; self.d['cand_t2i'] = Z['t2i_top32']; self.d['cand_sha256'] = cand_file['sha256']
""")
rep("""            key = (cfg['cond'], cfg['structure'] == 'L')
            if key not in ctxs:
                ctxs[key] = Ctx(cfg['cond'], device, need_local=key[1])""",
    """            key = (cfg['cond'], cfg['structure'] == 'L', (cfg.get('candidate_file') or {}).get('sha256'))   # R46: + round-2 table
            if key not in ctxs:
                ctxs = {}   # R46: one context at a time (memory)
                ctxs[key] = Ctx(cfg['cond'], device, need_local=key[1], cand_file=cfg.get('candidate_file'))""")
rep("""    return ctx.d['trust_subset'] if cfg['supervision'] in D36.TRUST_SUPS else None""",
    """    return ctx.d['trust_subset'] if cfg['supervision'] in D36.TRUST_SUPS else ctx.d.get('record_pool')   # R46-A: fixed FIT record pool (None = full pool)""")
rep("""    boundary = None
    if cfg.get('parent_ckpt'):""", """    if cfg.get('init_from'):   # R46-B round 2: start from the round-1 step-8000 state (heads, nulls, student); fresh AdamW
        inf = cfg['init_from']; assert stud is not None and nulls is not None and sha256(inf['ckpt']) == inf['ckpt_sha256'], 'R46: round-1 checkpoint changed'
        par = torch.load(inf['ckpt'], map_location=dev, weights_only=False)
        assert par['step'] == inf['step'] and par['unit_id'] == inf['unit_id'] and par['config']['cond'] == cfg['cond'] and par['config']['objective'] == cfg['objective']
        heads.load_state_dict(par['heads']); nulls.load_state_dict(par['nulls']); stud.load_state_dict(par['student'])
        assert all(R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS), 'R46: round-1 frozen student tensors differ from the parent'
        stud_ref = {p: {n: q.detach().clone() for n, q in stud[p].named_parameters()} for p in PEERS}
        lineage['init_from'] = dict(inf, head_state_sha256=state_hash(heads), null_state_sha256=state_hash(nulls), student_state_sha256=state_hash(stud),
                                    optimizer='fresh AdamW (round-1 optimizer state not loaded); common cosine restarted over the round-2 updates')
        del par
    boundary = None
    if cfg.get('parent_ckpt'):""")
rep("""            if rk is not None and step == 0 and cfg.get('phase', 1) == 1:
                # fail-fast step-0 check""", """            if rk is not None and step == 0 and cfg.get('init_from'):   # R46-B: step 0 must reproduce the round-1 step-8000 DEV exactly
                R1 = np.load(cfg['init_from']['dev_ranks'])
                ni = int((rk['ranks_i'] != R1['ranks_i']).sum()); nt = int((rk['ranks_t'] != R1['ranks_t']).sum())
                rec['step0_init_from_check'] = dict(rsum=rec['dev_retrieval']['rsum'], round1_rsum=cfg['init_from']['eval_rsum'], n_i2t_rank_diff=ni, n_t2i_rank_diff=nt,
                                                    passed=bool(ni == 0 and nt == 0 and rec['dev_retrieval']['rsum'] == cfg['init_from']['eval_rsum']))
                write_json(out / 'eval_step0.json', dict(rec, step=0))
                assert rec['step0_init_from_check']['passed'], f"R46 step 0 does not reproduce round 1: {rec['step0_init_from_check']}"
            if rk is not None and step == 0 and cfg.get('phase', 1) == 1 and not cfg.get('init_from'):
                # fail-fast step-0 check""")
rep("""        if step == 0 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 fail-fast: student forward (grad on, training batch) == native cache""",
    """        if step == 0 and cfg.get('init_from'):   # R46-B: replaced by the step-0 round-1 DEV rank equality
            hist['r39']['step0_cache_equivalence'] = dict(passed=True, replaced='R46 round 2: the student starts from the round-1 step-8000 state; step-0 DEV ranks == round-1 ranks asserted instead')
        if step == 0 and cfg.get('phase', 1) == 1 and stud is not None and not cfg.get('init_from'):   # R39 fail-fast: student forward (grad on, training batch) == native cache""")
open(p, 'w').write(s)
print('R46 patched', p)
