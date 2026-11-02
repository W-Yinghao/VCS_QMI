"""R35 B-ORACLE runner (separate process). Sets the oracle-process flag, imports the truth module, refuses any
non-ORACLE unit, then runs the shared trainer with the truth-repaired FIT record->image map (P and Q both from the
repaired relation; same H60 features, same caption records, same common native calibration init).
Also writes ORACLE diagnostics: realised mismatch of the observed map and the truth-clean fraction of the B-TRUST
subset (evaluator statistics, computed only here).
Usage: python -B train_oracle.py --units r35_B_n60_s1001_ORACLE_J,..."""
import os
import sys
from pathlib import Path

os.environ['R35_ORACLE_PROCESS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import PREP, RUNS, install_guards, write_json, utc, read_json, CONDS
import oracle_truth
import train_unit


def diagnostics(truth):
    obs = np.load(CONDS['n60']['noise_file']).astype(np.int64)
    sub = np.load(PREP / 'trust_n60.npz')['subset'].astype(np.int64)
    return dict(utc=utc(), observed_mismatch_fraction_full_pool=float((obs != truth).mean()),
                trust_subset_size=int(len(sub)), trust_subset_truth_clean_fraction=float((obs[sub] == truth[sub]).mean()),
                trust_subset_retention=float(len(sub) / len(obs)),
                complement_truth_clean_fraction=float((obs[np.setdiff1d(np.arange(len(obs)), sub)] == truth[np.setdiff1d(np.arange(len(obs)), sub)]).mean()),
                note='evaluator statistics computed in the oracle process only; not used by any non-oracle arm')


if __name__ == '__main__':
    install_guards()
    truth = oracle_truth.truth_map()
    write_json(RUNS / 'ORACLE_DIAGNOSTICS.json', diagnostics(truth))
    train_unit.main(oracle_imap=truth)
