"""R36 M-oracle runner (separate process; SHIELD_ORACLE / RELABEL_AVAILABLE_ORACLE only). Sets the oracle-process
flag, imports the truth module, refuses every non-oracle unit, then runs the shared trainer with the oracle
supervision hook; outputs go to OUT/oracle_runs/<unit> (never read by T/N/V). Writes ORACLE_DIAGNOSTICS.json.
Usage: python -B train_oracle.py --units r36_M_n60_G_SHIELD_ORACLE_J,..."""
import os
import sys
from pathlib import Path

os.environ['R36_ORACLE_PROCESS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import ORACLE_RUNS, CONDS, PREP, install_guards, write_json, utc
import oracle_truth
import train_unit


def selftest(pkg):
    """CPU: oracle_truth.shield/relabel vs the package reference (oracle_shield / oracle_relabel_available /
    loss_rows), toy rows incl. i2t multi-true and rows without true slots; masked true-slot gradient = 0; wrong
    observed positive keeps its mass. Prints one JSON line."""
    import importlib.util, json, torch
    from objectives import list_rows_pq, list_rows
    spec = importlib.util.spec_from_file_location('ref_objectives_r36', str(Path(pkg) / 'reference' / 'objectives.py'))
    RO = importlib.util.module_from_spec(spec); spec.loader.exec_module(RO)
    torch.set_default_dtype(torch.float64); rng = np.random.default_rng(5)
    a = np.zeros((6, 8)); a[0, 0] = 1; a[1, [0, 3]] = .5; a[2, 2] = 1; a[3, 1] = 1; a[4, 5] = 1; a[5, 7] = 1
    truth = np.zeros((6, 8), bool); truth[0, [2, 4]] = True; truth[1, 3] = True; truth[2, 2] = True; truth[4, [0, 1, 6]] = True
    out = {}
    a1, q, v, _ = oracle_truth.shield(a, truth)
    rq, rv = RO.oracle_shield(torch.tensor(a), torch.tensor(truth))
    out['shield_q_equal'] = bool(np.allclose(q, rq.numpy(), atol=0)) and bool((v == rv.numpy()).all())
    out['shield_wrong_positive_kept'] = bool(a1[0, 0] == 1 and a1[4, 5] == 1)
    g = torch.tensor(rng.normal(size=(6, 8)) * 2, requires_grad=True)
    for obj in ('J', 'BCE', 'NCE'):
        mine = list_rows_pq(g, torch.tensor(a1), torch.tensor(q), torch.tensor(v), obj)
        ref = RO.loss_rows(g, torch.tensor(a1), torch.tensor(q), obj, torch.tensor(v))
        mine_loss = -mine if obj == 'J' else mine
        out[f'shield_{obj}_abs_err_vs_reference'] = float((mine_loss - ref).abs().max())
        gr = torch.autograd.grad(mine.sum(), g)[0].numpy()
        out[f'shield_{obj}_masked_true_grad_zero'] = bool((gr[truth & (a == 0)] == 0).all())
        out[f'shield_{obj}_wrong_pos_grad_nonzero'] = bool(gr[0, 0] != 0)
        noF = ~(truth & (a == 0)).any(1)
        base = list_rows(g, torch.tensor(a), obj)
        out[f'shield_{obj}_rows_without_F_equal_unshielded'] = bool(torch.allclose(mine[torch.tensor(noF)], base[torch.tensor(noF)], atol=1e-12))
    a2, c = oracle_truth.relabel(a, truth)
    out['relabel_equal_reference'] = bool(np.allclose(a2, RO.oracle_relabel_available(torch.tensor(a), torch.tensor(truth)).numpy(), atol=0))
    out['relabel_counts'] = c
    caps = np.array([[10, 11, 12, 99]]); imgs = np.array([[2, 3, 4, 5]])
    ti, tt = oracle_truth.truth_masks(np.array([2]), caps, np.array([17]), imgs)
    out['truth_masks_i2t_multi_t2i_single'] = bool((ti == [[True, True, True, False]]).all() and (tt == [[False, True, False, False]]).all())
    out['ALL_PASS'] = all(v for k, v in out.items() if isinstance(v, bool)) and all(v < 1e-12 for k, v in out.items() if k.endswith('abs_err_vs_reference'))
    print('R36_ORACLE_SELFTEST ' + json.dumps(out, default=int))


if __name__ == '__main__' and '--selftest' in sys.argv:
    install_guards(role='oracle')
    selftest(sys.argv[sys.argv.index('--selftest') + 1])
elif __name__ == '__main__':
    install_guards(role='oracle')
    truth = oracle_truth.truth_map()
    obs = np.load(CONDS['n60']['noise_file']).astype(np.int64)
    ORACLE_RUNS.mkdir(parents=True, exist_ok=True)
    write_json(ORACLE_RUNS / 'ORACLE_DIAGNOSTICS.json', dict(utc=utc(), observed_mismatch_fraction_full_pool=float((obs != truth).mean()),
               note='evaluator statistic computed in the oracle process only'))
    train_unit.main(oracle=oracle_truth)
