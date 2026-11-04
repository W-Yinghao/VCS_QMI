"""R50 configs (user 2026-11-04 「查看这个包，准备相关实验，提交」; package VCS_Layout_Audit_R50_20261104, MATRIX.csv 24 units, protocol FIT_FIXED_CENTER_V1).
Each unit = the N reference config of the same parent / encoder / objective (frozen: R48 N-frozen lfG; both_joint: R47 G N-JOINT) with the cond swapped to
the R50 FIT-fixed cond <base>_lfF and the supervision replaced (A: N_PRESENT_ONLY, R40 B rule; B: ORACLE_MISSING, R39/R40 rule in the oracle runner) -
same init seed, sampler stream, 8000 updates, readouts 0/250/500/1000/2000/4000/8000, optimizer / schedule / clip / 16 queries per direction / two peers.
Also writes the 18 replay configs R50_REPLAY_<reference> (the reference config with only the cond / tree swapped; gate mode only, never formal) used to
show that the reference training is unchanged under the R50 FIT side. Writes configs/R50_*.json, configs/UNITS_R50.json, configs/UNITS_R50_REPLAY.json,
RESOLVED_MATRIX_R50.csv. Usage: python -I -B tools/make_configs_r50.py"""
import copy, csv, hashlib, json
from pathlib import Path
W = Path(__file__).resolve().parents[1]
PKG = Path('/home/infres/yinwang/CS_QMI/r50_layout_audit/VCS_Layout_Audit_R50_20261104')
QUOTE = '查看这个包，准备相关实验，提交'
rd = lambda p: json.loads(Path(p).read_text())
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PARENT = {'F60_R30H1001': ('F60_R30_H1001', 'n60', 'r50f', 'Flickr30K 60% injected noise'),
          'CC_R31H1001': ('CC31_H1001', 'cc', 'r50c', 'CC152K (real noise)'),
          'F80_IN2RH1001': ('F80_IN2R_H1001', 'f80in', 'r50f', 'Flickr30K 80% injected noise')}
REF = lambda pk, enc, o: f'R48_{pk}_N_frozen_lfG_{o}' if enc == 'frozen' else f'R47_G_{pk}_N_joint_{o}'
CENT = ('R50 FIT_FIXED_CENTER_V1: frozen-H cache, critic inputs and (JOINT) student = normalize(e - mu_FIT), ONE fixed parent FIT mean per peer / modality '
        '(the R47 G FIT means) on FIT and DEV; DEV only applies the FIT statistic (no DEV mean, no batch / layout grouping); readout = DEV two-stage (H top-100 '
        'reranked by peer-mean g) on the R50 cache; primary endpoint step 8000')
SUP = {'PRESENT_ONLY': ('N_PRESENT_ONLY', 'R50 A N_PRESENT_ONLY (R40 B rule on the R50 candidate tables): N lists unchanged (natural top-32 + null, 33 slots, all permuted, '
                        'Q = Uniform(33)); a row enters the loss only if >= 1 real slot carries observed-positive mass (present); absent rows (A_null = 1) masked entirely, '
                        'never redrawn; per direction loss = sum(mask * row_loss) / (n_rows * p_d), p_d fixed from the full FIT observed relation (data36.presence_probs, '
                        'recomputed in-unit); 0.5 per direction; head and (JOINT) encoder get the same masked loss (no R44 routing)'),
       'ORACLE_MISSING': ('ORACLE_MISSING', 'R50 B ORACLE_MISSING (R39/R40 rule, diagnostic, separate oracle process / dir): rows whose N label is all-null: if the row\'s natural '
                          'top-32 list (R50 candidate tables) holds >= 1 true slot (pre-injection pairing), A = 0.5 uniform(true slots) + 0.5 null, else unchanged; '
                          'observed-present rows unchanged; no insertion; not deployable')}
units, replay, rows = [], [], []
for r in csv.DictReader(open(PKG / 'MATRIX.csv')):
    pk, base, tree, ds = PARENT[r['parent']]; o = r['objective']; enc = r['encoder']; oracle = r['oracle'] == 'True'
    ref = REF(pk, enc, o); t = rd(W / 'configs' / f'{ref}.json')
    assert t['cond'] == f'{base}_lfG' and t['steps'] == 8000 == int(r['steps']) and t['objective'] == o and t['supervision'] == 'OBS_MISSING_NULL'
    assert t['init_seed'] == int(r['training_seed']) and t['sampler_seed'][0] == int(r['training_seed'])
    assert (t['adaptation'] == 'frozen') == (enc == 'frozen') and (enc == 'frozen' or t['adaptation'] == 'both_joint')
    sup, rule = SUP[r['supervision']]
    cfg = copy.deepcopy(t)
    for k in ('paired_joint_unit', 'frozen_template', 'reference_note'):
        cfg.pop(k, None)
    cfg.update(unit_id=r['unit'], line='R50', block=r['block'], cond=f'{base}_lfF', code_tree=tree, runner=f"{tree}/{'train_oracle.py' if oracle else 'train_unit.py'}",
               supervision=sup, supervision_label=r['supervision'], supervision_rule=rule, oracle=oracle, protocol_id=r['protocol_id'], centring_variant='F', centring_rule=CENT,
               reference_unit=None, reference_joint=None, dataset=ds, test_read=False,
               n_reference=dict(unit_id=ref, config_sha256=sh(W / 'configs' / f'{ref}.json'), reread='/home/infres/yinwang/CS_QMI/outputs_v9_r50/r50_rereads/' + ref,
                                note='same init / sampler stream / recipe; its step-8000 weights are reused and re-read under FIT_FIXED_CENTER_V1 (REUSE_RECEIPT_R50.json)'),
               package=dict(name='VCS_Layout_Audit_R50_20261104', matrix_row=r, matrix_sha256=sh(PKG / 'MATRIX.csv'), plan_sha256=sh(PKG / 'docs' / 'SERVER_PLAN_CN.md')),
               label='development experiment (DEV only; no TEST read in R50)' + ('; oracle diagnostic, not deployable' if oracle else ''),
               authorisation=dict(user_quote=QUOTE, relayed_by='main session', note='formal submission right after the CPU check and the GPU smokes pass'))
    p = W / 'configs' / f"{r['unit']}.json"; assert not p.exists() or rd(p) == cfg, f'{p} exists with different content'
    p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); units.append(r['unit'])
    rows.append(dict(unit_id=r['unit'], block=r['block'], tree=tree, runner=cfg['runner'], cond=cfg['cond'], supervision=sup, encoder=enc, objective=o, oracle=oracle, steps=8000,
                     n_reference=ref, config_sha256=sh(p)))
for pk, base, tree, ds in PARENT.values():
    for enc in ('frozen', 'both_joint'):
        for o in ('J', 'BCE', 'NCE'):
            ref = REF(pk, enc, o)
            if not (W / 'configs' / f'{ref}.json').exists():
                continue
            t = rd(W / 'configs' / f'{ref}.json'); cfg = copy.deepcopy(t)
            cfg.update(unit_id=f'R50_REPLAY_{ref}', line='R50 replay (gate mode only)', block='REPLAY', cond=f'{base}_lfF', code_tree=tree, runner=f'{tree}/train_unit.py',
                       replay_of=dict(unit_id=ref, config_sha256=sh(W / 'configs' / f'{ref}.json'), runs='runs_r48' if enc == 'frozen' else 'runs_r47'),
                       label='R50 reuse evidence: the reference config on the R50 FIT-fixed cond (training must reproduce the reference); never a formal unit')
            p = W / 'configs' / f"R50_REPLAY_{ref}.json"; p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); replay.append(cfg['unit_id'])
(W / 'configs' / 'UNITS_R50.json').write_text(json.dumps(units, indent=1))
(W / 'configs' / 'UNITS_R50_REPLAY.json').write_text(json.dumps(replay, indent=1))
with open(W / 'RESOLVED_MATRIX_R50.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R50 configs', len(units), 'replay configs', len(replay))
