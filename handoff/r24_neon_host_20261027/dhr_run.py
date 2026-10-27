"""R24 S0.4: DeeperHistReg (Wodzinski et al., pip deeperhistreg 1.0.1, MagicLeap SuperPoint/SuperGlue weights) as an independent
strong reference on the same CoMIR-protocol problems as neon_run.py (fixed SHG, moving BF, exact synthetic truth).

Library pipeline default_initial_nonrigid (basic preprocessing: gray, normalisation, CLAHE, intensity flip; multi_feature initial
alignment = SIFT-RANSAC + SuperPoint-SuperGlue over rotations; instance-optimisation non-rigid NCC + diffusion) called on tensors.
Adaptations for 834-px patches (recorded, not tuned on DEV): no initial resampling, non-rigid registration size 832 with 5 levels.
The only configuration choice, whether the SHG is intensity-inverted before the library's own flip (SHG has a dark background,
BF a white one), is made on CAL only (--shg invert|raw)."""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
import deeperhistreg  # noqa: E402
sys.path.insert(0, os.path.dirname(deeperhistreg.__file__))
import torch as tc  # noqa: E402
from dhr_pipeline import registration_params as rp  # noqa: E402
from dhr_preprocessing import preprocessing as pre  # noqa: E402
from dhr_registration import initial_alignment_methods as ia, nonrigid_registration_methods as nr  # noqa: E402
from dhr_utils import warping as w  # noqa: E402
sys.path.insert(0, str(HERE))
from problems import C, build_problem, truth_map, GRID, CORNERS, W  # noqa: E402  (same problems and truth as neon_run.py)


def params():
    p = rp.default_initial_nonrigid(); p['device'] = 'cuda:0'
    p['preprocessing_params']['initial_resampling'] = False
    p['initial_registration_params']['device'] = 'cuda:0'
    n = p['nonrigid_registration_params']; n['device'] = 'cuda:0'; n['registration_size'] = W; n['num_levels'] = n['used_levels'] = 5
    n['iterations'] = [100, 100, 100, 100, 200]; n['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0015]; n['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.8]
    return p


def to_pix(df, pts):
    """DHR displacement field [1,H,W,2] (normalised x,y offsets, align_corners=False): fixed pixel (row, col) -> moving pixel."""
    H, Wd = df.shape[1:3]; out = []
    for r, c in pts:
        gx, gy = (2 * c + 1) / Wd - 1, (2 * r + 1) / H - 1
        g = tc.tensor([[[[gx, gy]]]], dtype=df.dtype, device=df.device)
        d = tc.nn.functional.grid_sample(df.permute(0, 3, 1, 2), g, align_corners=False)[0, :, 0, 0]
        mx, my = gx + float(d[0]), gy + float(d[1]); out.append([((my + 1) * H - 1) / 2, ((mx + 1) * Wd - 1) / 2])
    return np.array(out)


def register(mov, fix, p):
    src = tc.tensor(mov[:W, :W], dtype=tc.float32).permute(2, 0, 1)[None].cuda() / 255.0
    tgt = tc.tensor(fix[:W, :W], dtype=tc.float32)[None, None].repeat(1, 3, 1, 1).cuda() / 255.0
    with tc.no_grad():
        ps, pt, _, _, post = pre.basic_preprocessing(src, tgt, None, None, p['preprocessing_params'])
        T = ia.get_function(p['initial_registration_params']['initial_registration_function'])(ps, pt, p['initial_registration_params'])
        df0 = w.tc_transform_to_tc_df(T, ps.size())
    df = nr.get_function(p['nonrigid_registration_params']['nonrigid_registration_function'])(ps, pt, df0, p['nonrigid_registration_params'])
    assert ps.shape[-2:] == (W, W), ps.shape
    return df0.detach(), df.detach()


def run(role, shard, nshard, out, shg, smoke, kmax=C.K_DRAWS):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); g = C.guard(); p = params()
    probs = [C.draw('comir', c, k) for c in sorted(g.included(role)) for k in range(kmax)][shard::nshard][:2 if smoke else None]
    path = out / f'{role}_{shg}_s{shard}.jsonl'
    done = {(r['core'], r['k']) for r in map(json.loads, open(path))} if path.exists() else set()
    for prob in probs:
        if (prob['core'], prob['k']) in done: continue
        t0 = time.time(); pb = build_problem(g, role, prob)
        fix = 255 - pb['fix'] if shg == 'invert' else pb['fix']
        if smoke:   # convention check: a DHR field built from the truth maps the interior grid onto the truth
            q = np.stack(np.meshgrid(np.arange(W), np.arange(W), indexing='ij'), -1).astype(np.float64); m = truth_map(prob, q.reshape(-1, 2)).reshape(W, W, 2)
            gx, gy = (2 * q[..., 1] + 1) / W - 1, (2 * q[..., 0] + 1) / W - 1; mx, my = (2 * m[..., 1] + 1) / W - 1, (2 * m[..., 0] + 1) / W - 1
            dft = tc.tensor(np.stack([mx - gx, my - gy], -1)[None], dtype=tc.float32, device='cuda')
            assert np.abs(to_pix(dft, GRID) - truth_map(prob, GRID)).max() < 1e-2
        try:
            df0, df = register(pb['mov'], fix, p); err = {}
            for name, d in (('initial', df0), ('final', df)):
                for mname, pts in (('grid', GRID), ('corner', CORNERS)):
                    err[f'{name}_{mname}'] = float(np.linalg.norm(to_pix(d, pts) - truth_map(prob, pts), axis=1).mean())
            status = 'ok'
        except Exception as e:      # recorded as a failure of the method on this problem (counts as error = inf)
            err, status = {}, f'error: {type(e).__name__}: {e}'
        row = dict(core=prob['core'], k=prob['k'], role=role, shg=shg, status=status, seconds=time.time() - t0, **err,
                   gpu=tc.cuda.get_device_name(), job=os.environ.get('SLURM_JOB_ID'))
        with open(path, 'a') as f: f.write(json.dumps(row) + '\n')
        print(prob['core'], prob['k'], status, {k: round(v, 2) for k, v in err.items()}, '%.0fs' % row['seconds'], flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--role', default='CAL'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1); ap.add_argument('--out', required=True); ap.add_argument('--shg', default='invert', choices=('invert', 'raw'))
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--kmax', type=int, default=C.K_DRAWS)
    a = ap.parse_args(); run(a.role, a.shard, a.nshard, a.out, a.shg, a.smoke, a.kmax)
