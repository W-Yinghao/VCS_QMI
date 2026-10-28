"""Write the R26 feeder lines (DESIGN §3). Usage: make_lines.py STAGE TIME_FLICKR TIME_COCO TIME_CC  (HH:MM:SS each).
Stage A: Flickr 40/60 x seed1001 x 5 arms; B: Flickr 40/60 x seed1002/1003 x 5 arms;
C: COCO40 / CC152K x {d_vcs, d_bce, m_vcs} x 3 seeds + native H0 x 3 seeds x 2 datasets. No result gate anywhere."""
import sys, json

D = '/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027'
ARMS5 = ('d_vcs', 'd_bce', 'm_vcs', 'n_vcs', 'd_snce')
SHORT = {'d_vcs': 'dvcs', 'd_bce': 'dbce', 'm_vcs': 'mvcs', 'n_vcs': 'nvcs', 'd_snce': 'dsnce', 'H0': 'H0'}


def line(kind, cell, arm, t, nice=40):
    name = f"r26_{SHORT[arm]}_{cell}"
    tail = f"{kind} {cell}" + (f" {arm}" if kind == 'arm' else '')
    return f"sbatch --job-name={name} --time={t} --nice={nice} {D}/sbatch/r26_unit.sbatch {tail}"


def main():
    stage, tf, tc, tcc = sys.argv[1:5]
    out = []
    if stage == 'A':
        for cell in ('f40_s1001', 'f60_s1001'):
            out += [line('arm', cell, a, tf) for a in ARMS5]
    elif stage == 'B':
        for cell in ('f40_s1002', 'f60_s1002', 'f40_s1003', 'f60_s1003'):
            out += [line('arm', cell, a, tf) for a in ARMS5]
    elif stage == 'C':
        for s in (1001, 1002, 1003):
            for cell, t in ((f'c40_s{s}', tc), (f'cc_s{s}', tcc)):
                out.append(line('native', cell, 'H0', t, nice=60))
                out += [line('arm', cell, a, t, nice=60) for a in ('d_vcs', 'd_bce', 'm_vcs')]
    else:
        raise SystemExit('stage A|B|C')
    path = f'{D}/LINES_{stage}.txt'
    open(path, 'w').write('\n'.join(out) + '\n')
    print(path, len(out))


if __name__ == '__main__':
    main()
