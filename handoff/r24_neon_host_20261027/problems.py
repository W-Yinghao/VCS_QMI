"""R24 shared problem construction (CoMIR-protocol draws, fixed SHG / moving BF, exact truth); no host imports."""
import math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'r23_comir_protocol_20261026'))
import common as C                                                                        # noqa: E402  (V10 DataGuard, draws, raw TIFF reader)

S = 834; W = 832; CF = 607; CEN = 1023.5; DOWN = 4


def area(a, f=DOWN):
    h, w = a.shape[:2]; a = a[:h - h % f, :w - w % f].astype(np.float64)
    return a.reshape(h // f, f, w // f, f, *a.shape[2:]).mean((1, 3))


def rq(a):
    q1, med, q3 = np.percentile(a, [25, 50, 75]); return med, max(q3 - q1, 1e-6)


def build_problem(g, role, prob):
    from scipy.ndimage import map_coordinates
    core = prob['core']; g.check(core, {role}, 'R24 Neon host problem')
    split, pw, pm = C.P.core_files()[core]
    bf = C.P.read_tiff(pw); shg = C.P.read_tiff(pm); assert bf.dtype == np.uint8 and shg.dtype == np.uint8
    ys, xs = np.mgrid[CF:CF + S, CF:CF + S].astype(np.float64)
    r = math.radians(prob['theta']); c, s = math.cos(-r), math.sin(-r)
    zy, zx = ys - CEN - prob['ty_full'], xs - CEN - prob['tx_full']                       # V_p(y) = V(R(-theta)(y - c - t) + c)
    sy, sx = c * zy - s * zx + CEN, s * zy + c * zx + CEN
    mov = np.stack([map_coordinates(bf[..., ch].astype(np.float64), [sy, sx], order=1, mode='reflect') for ch in range(3)], -1)
    mov = np.clip(np.rint(mov), 0, 255).astype(np.uint8)                                   # the host loads uint8 images
    fix = shg[CF:CF + S, CF:CF + S].copy()
    # core-level robust-normalisation constants of the encoders' training inputs (bf_rgb_n, shg_log_n of the full core)
    bfr = area(bf.astype(np.float64) / 255.0); shd = area(shg.astype(np.float64) / 255.0)
    return dict(fix=fix, mov=mov, bf_norm=rq(bfr), shg_norm=rq(np.log1p(shd)), orig_bf=bf[CF:CF + S, CF:CF + S].copy())


def truth_map(prob, q):
    """q: [..., 2] patch (row, col) of the fixed image -> true moving patch coordinates T(q + CF) - CF."""
    r = math.radians(prob['theta']); Rm = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    z = q + CF - CEN; return z @ Rm.T + CEN + np.array([prob['ty_full'], prob['tx_full']]) - CF


GRID = np.stack(np.meshgrid(np.linspace(26, W - 27, 16), np.linspace(26, W - 27, 16), indexing='ij'), -1).reshape(-1, 2)
CORNERS = np.array([[0, 0], [0, W - 1], [W - 1, 0], [W - 1, W - 1]], np.float64)


