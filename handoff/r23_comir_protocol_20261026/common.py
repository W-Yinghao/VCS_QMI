"""R23 task B: CoMIR (NeurIPS 2020) and MultiRegEval (Lu et al. PLOS ONE 2022) BF-SHG patch protocols on our DEV/CAL cores.

Geometry (host conventions, 512-px frame of the 2048-px core, x_full = 4 x + 1.5, centre 255.5 <-> 1023.5):
  truth (theta*, t*) with moving V_p = V o T*^-1, T(z) = R(theta)(z - c) + c + t, (y, x) order, theta in degrees;
  the moving full image is warped first and then cropped (MultiRegEval make_eliceiri_patches: warp then crop), so the moving
  patch has real content everywhere; both methods see only the 834 x 834 full-res centre patch (512 frame: pixels 152..359).
Error: host Q.corner_error_full = mean displacement of the four corners of the 834-px centre patch, full-res px (the CoMIR /
MultiRegEval definition). Success: CoMIR <9 / <42 px (1 % / 5 %); MultiRegEval <2 % of 834 = 16.68 px.
"""
import hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(HERE / 'pydeps'))
_spec = importlib.util.spec_from_file_location('r22_run_unit', BASE / 'ssl_pilot_v10_r22c/v10/r22_pmi/run_unit.py')
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)        # verifies the pinned host files, sets sys.path
A, torch, F, CP = R.A, R.torch, R.F, R.CP
Q, P = A.Q, A.P
from v10.supplement import training as TR, registration as SREG, mind as MIND  # noqa: E402  (pinned host files)

S, CEN, DOWN = P.SIZE, (P.SIZE - 1) / 2.0, P.DOWN
WIN = (152, 359)                     # inclusive 512-frame pixel window of the 834 full-res centre patch (pixel centres +-103.5)
HALF_PIX = (WIN[1] - WIN[0]) / 2.0   # 103.5
FULL_WIN = (607, 1440)               # inclusive full-res window, 834 px, centre 1023.5
FULL_CEN = 1023.5
REF_HALF = HALF_PIX - 17.0           # positions: reference receptive field (15 px + bilinear) inside the reference patch
MOV_MARGIN = {'learned': 15 * math.sqrt(2) + 1.0, 'mind': 4.0, 'comir': 1.0, 'mi': 1.0}
N_POS = P.N_POS
THETAS = np.arange(-30.0, 30.0 + 1e-9, 3.0)
TS = np.array([(y, x) for y in range(-28, 29, 4) for x in range(-28, 29, 4)], dtype=np.float64)   # 512-frame px
T_CLAMP, TH_CLAMP = 28.0, 30.0
PROTOCOLS = ('comir', 'mre')
K_DRAWS = 10
THR = {'comir': (9.0, 42.0), 'mre': (834 * 0.02,)}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def write(p, d):
    p = Path(p); tmp = p.with_name(p.name + f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + '\n'); tmp.replace(p)


def append(p, d):
    with open(p, 'a') as f: f.write(json.dumps(d, allow_nan=False) + '\n')


def guard():
    """BF V10 DataGuard (refuses the held TEST cores); roles from the V10 manifest, verified against the R6 manifest."""
    return A.DataGuard()


# ------------------------------------------------------------------------------------------------------------ problems
def draw(protocol, core, k):
    """One rigid transform; full-res parameters and the 512-frame truth (theta deg, ty, tx)."""
    if protocol == 'comir':           # Zenodo 3874362: rotation up to +-30 deg, x and y translation up to 100 px
        g = np.random.default_rng(Q.seed_of(f'R23-COMIR-PROTO|{core}|{k}'))
        th, ty, tx = g.uniform(-30, 30), g.uniform(-100, 100), g.uniform(-100, 100); level = None
    elif protocol == 'mre':           # MultiRegEval make_eliceiri_patches: level L, sign coin then uniform magnitude, per parameter
        level = k % 4 + 1; g = np.random.default_rng(Q.seed_of(f'R23-MRE-PROTO|{core}|{k}'))
        mag = lambda lo, hi: (1 if g.integers(2) else -1) * g.uniform(lo, hi)
        th = mag(5 * (level - 1), 5 * level); tx = mag(20 * (level - 1), 20 * level); ty = mag(20 * (level - 1), 20 * level)
    else:
        raise ValueError(protocol)
    return dict(protocol=protocol, core=core, k=k, level=level, theta=float(th), ty_full=float(ty), tx_full=float(tx),
                truth=[float(th), float(ty) / DOWN, float(tx) / DOWN])


def problems(g, role, protocol):
    return [draw(protocol, c, k) for c in g.included(role) for k in range(K_DRAWS)]


def corner_error(est, truth):
    """est/truth = (theta deg, ty, tx) in the 512 frame -> full-res px (host definition)."""
    return float(Q.corner_error_full(est[0], np.asarray(est[1:]), truth[0], np.asarray(truth[1:])))


def corner_error_direct(est, truth):
    """Independent implementation from the corner coordinates (MultiRegEval info_test.csv definition), full-res px."""
    a = 833 / 2.0; cs = np.array([[-a, -a], [-a, a], [a, -a], [a, a]])
    def mv(th, t):
        r = math.radians(th); Rm = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        return cs @ Rm.T + np.asarray(t) * DOWN
    return float(np.linalg.norm(mv(est[0], est[1:]) - mv(truth[0], truth[1:]), axis=1).mean())


# ------------------------------------------------------------------------------------------------------------ rendering
def _crop(a):
    return a[..., WIN[0]:WIN[1] + 1, WIN[0]:WIN[1] + 1]


def _canvas(patch, channels):
    out = np.zeros((channels, S, S), np.float32); out[:, WIN[0]:WIN[1] + 1, WIN[0]:WIN[1] + 1] = patch; return out


def render_quarter(g, role, prob, device, norm='patch', purpose='R23-B CoMIR/MultiRegEval patch protocol'):
    """512-frame inputs of one problem as host-shaped canvases (zero outside the patch window).
    Returns dict(ref_int [1,S,S], ref_rgb [3,S,S], mov_int [1,S,S], mov_log [1,S,S]) torch tensors on device.
    norm='patch': robust (median/IQR) normalisation on the patch only (the method sees only the patches);
    norm='core': the host normalisation over the full core / full perturbed image, then cropped (sensitivity)."""
    core = prob['core']; g.check(core, {role}, purpose)
    bf, rgb, shl = P.load_img(g.r6_root, core, ('bf', 'bf_rgb_n', 'shg_log'))
    sh = np.expm1(shl.astype(np.float64))
    th, ty, tx = prob['truth']
    shp = Q.perturb_map(torch, F, torch.tensor(sh, dtype=torch.float32, device=device)[None], th, (ty, tx))[0].cpu().numpy().astype(np.float64)
    rgb = rgb.transpose(2, 0, 1)
    if norm == 'patch':
        ref_int, ref_rgb = P.robust_norm(_crop(bf)), P.robust_norm(_crop(rgb))
        p = _crop(shp); mov_int, mov_log = P.robust_norm(p), P.robust_norm(np.log1p(p))
    elif norm == 'core':
        ref_int, ref_rgb = _crop(bf), _crop(rgb)
        mov_int, mov_log = _crop(P.robust_norm(shp)), _crop(P.robust_norm(np.log1p(shp)))
    else:
        raise ValueError(norm)
    T = lambda a, c: torch.tensor(_canvas(a, c), device=device)
    return dict(ref_int=T(ref_int, 1), ref_rgb=T(ref_rgb, 3), mov_int=T(mov_int, 1), mov_log=T(mov_log, 1),
                patch=dict(ref_int=ref_int, mov_int=mov_int))


def render_full(g, role, prob, purpose='R23-B CoMIR/MultiRegEval full-res patch'):
    """Full-res 834 x 834 patches from the raw TIFFs: BF luminance (host formula) and the warped-then-cropped SHG."""
    from scipy.ndimage import map_coordinates
    core = prob['core']; g.check(core, {role}, purpose)
    split, pw, pm = P.core_files()[core]
    bf = P.read_tiff(pw).astype(np.float64) / 255.0; shg = P.read_tiff(pm)
    sh = shg.astype(np.float64) / (255.0 if shg.dtype == np.uint8 else 65535.0)
    lum = 0.299 * bf[..., 0] + 0.587 * bf[..., 1] + 0.114 * bf[..., 2]
    ys, xs = np.mgrid[FULL_WIN[0]:FULL_WIN[1] + 1, FULL_WIN[0]:FULL_WIN[1] + 1].astype(np.float64)
    r = math.radians(prob['theta']); c, s = math.cos(-r), math.sin(-r)
    zy, zx = ys - FULL_CEN - prob['ty_full'], xs - FULL_CEN - prob['tx_full']     # V_p(y) = V(R(-theta)(y - c - t) + c)
    sy, sx = c * zy - s * zx + FULL_CEN, s * zy + c * zx + FULL_CEN
    mov = map_coordinates(sh, [sy, sx], order=1, mode='reflect')
    ref = lum[FULL_WIN[0]:FULL_WIN[1] + 1, FULL_WIN[0]:FULL_WIN[1] + 1]
    return ref.astype(np.float32), mov.astype(np.float32)


def positions(prob):
    """N_POS reference positions uniform in the reference-valid square of the patch (shared by every method of a problem)."""
    g = np.random.default_rng(Q.seed_of(f"R23-PATCH-POS|{prob['protocol']}|{prob['core']}|{prob['k']}"))
    x = CEN + g.uniform(-REF_HALF, REF_HALF, size=(N_POS, 2))
    perm = g.permutation(N_POS)
    extra = np.random.default_rng(Q.seed_of(f"R23-PATCH-PERMS|{prob['protocol']}|{prob['core']}|{prob['k']}"))
    perms = np.stack([perm] + [extra.permutation(N_POS) for _ in range(7)])
    return x, perms
