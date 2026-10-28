"""Explicit TEST entry for R25 (NOT run during P0/P1). Requires --confirm-locked and an existing
SELECTION_LOCK.json that lists the run's model_best.pth.tar; evaluates with the native evalrank
(peer-averaged similarity). Flickr TEST was read in earlier work: report as a repeat benchmark.
"""
import argparse, json, os, sys

sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/code')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('--confirm-locked', action='store_true')
    ap.add_argument('--split', default='test')
    a = ap.parse_args()
    lock = '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/SELECTION_LOCK.json'
    if not a.confirm_locked or not os.path.exists(lock):
        raise SystemExit('TEST is gated: requires --confirm-locked and SELECTION_LOCK.json (configs + selection rules locked)')
    ck = os.path.join(a.run_dir, 'model_best.pth.tar')
    assert ck in json.load(open(lock))['test_checkpoints'], 'checkpoint not in the selection lock'
    import functools, torch
    torch.load = functools.partial(torch.load, weights_only=False)  # own checkpoints (argparse.Namespace inside); torch>=2.6 default
    from evaluation import evalrank
    evalrank(ck, split=a.split, fold5=False)


if __name__ == '__main__':
    main()
