"""Write/submit R25 IN2R jobs. Usage:
  jobs.py prefix  NOISE SEED TIME           -> warmup + co-train epoch 0 -> prefix_state.pt
  jobs.py calib   NOISE SEED TIME           -> CALIBRATION.json from that prefix
  jobs.py arm     NOISE SEED ARM TIME       -> ARM in H0,V0,V1,B0,B1 resumed from the group prefix (epochs 1..39)
Constants for V*/B* are read from the group's CALIBRATION.json (frozen; never edited by hand).
"""
import json, os, subprocess, sys

ROOT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027'
PKG = '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027'
PY = '/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u'
ARMS = {'H0': ('sce', 'c0', None), 'V0': ('vcs', 'c0', 'vcs_c0'), 'V1': ('vcs', 'c1', 'vcs_c1'),
        'B0': ('bce', 'c0', 'bce_c0'), 'B1': ('bce', 'c1', 'bce_c1')}


def group(noise, seed):
    return f'f{int(round(float(noise) * 100))}_s{seed}'


def header(name, time, mem='48G'):
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --nice=30
#SBATCH --output={PKG}/logs/%x_%j.out
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version --format=csv
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/r25_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/r25_$SLURM_JOB_ID' EXIT
for f in train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
echo staged $(du -sh $LOC)
cd {PKG}/code
"""


def common(noise, seed):
    # data_path = node-local copy of the identical NCR-data f30k_precomp files (I/O only; no TEST files staged)
    return f"--data_path $TMPDIR/r25_$SLURM_JOB_ID --noise_file {ROOT}/noise/f30k_precomp_{noise}.npy --noise_ratio {noise} --seed {seed}"


def main():
    kind, noise, seed = sys.argv[1], sys.argv[2], sys.argv[3]
    g = group(noise, seed)
    gdir = f'{ROOT}/groups/{g}'
    if kind == 'prefix':
        name, time = f'r25_pfx_{g}', sys.argv[4]
        body = f"{PY} r25_run.py {common(noise, seed)} --output_dir {gdir}/prefix --stop_after_prefix\n"
    elif kind == 'calib':
        name, time = f'r25_cal_{g}', sys.argv[4]
        body = f"{PY} r25_calibrate.py {common(noise, seed)} --output_dir {gdir}/calib --resume {gdir}/prefix/prefix_state.pt\n"
    elif kind in ('arm', 'resume'):  # resume: continue a timed-out arm in place from its own state_last.pt
        arm, time = sys.argv[4], sys.argv[5]
        obj, link, key = ARMS[arm]
        name = f'r25_{arm}_{g}'
        extra = f"--noisy_obj {obj} --link {link}"
        if key:
            cal = json.load(open(f'{gdir}/calib/CALIBRATION.json'))
            c = cal['c_D'].get(key)
            assert c is not None and c > 0, (key, c)
            extra += f" --c_D {c!r}"
            if link == 'c1':
                assert cal['C1_available'], 'C1 unavailable for this group'
                extra += f" --c1_a {cal['a']!r} --c1_b {cal['b']!r}"
        src = f'{gdir}/{arm}/state_last.pt' if kind == 'resume' else f'{gdir}/prefix/prefix_state.pt'
        assert os.path.exists(src), src
        if kind == 'resume':
            name += '_resume'
        body = f"{PY} r25_run.py {common(noise, seed)} --output_dir {gdir}/{arm} --resume {src} {extra}\n"
    else:
        raise SystemExit(kind)
    os.makedirs(f'{PKG}/sbatch', exist_ok=True)
    path = f'{PKG}/sbatch/{name}.sbatch'
    open(path, 'w').write(header(name, time) + body + f'echo {name}_DONE\n')
    out = subprocess.run(['sbatch', path], capture_output=True, text=True)
    print(out.stdout.strip(), out.stderr.strip())
    with open(f'{PKG}/SUBMITTED.tsv', 'a') as f:
        f.write(f"{name}\t{out.stdout.strip().split()[-1] if out.returncode == 0 else 'FAILED'}\t{time}\t{body.strip()}\n")


if __name__ == '__main__':
    main()
