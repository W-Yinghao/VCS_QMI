"""Dependency-free SVG plots (matplotlib is not installed in the project environment).  Scatter / line series with axes, ticks, a legend and an
optional identity line; good enough for the three N1 figures of package §04.5.  Every number plotted is also in the aggregate JSON."""
from __future__ import annotations

import math
from pathlib import Path

COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd"]


def _ticks(lo: float, hi: float, n: int = 5):
    if hi <= lo:
        hi = lo + 1.0
    step = (hi - lo) / n; mag = 10 ** math.floor(math.log10(step)); step = math.ceil(step / mag) * mag
    start = math.floor(lo / step) * step; return [start + i * step for i in range(int((hi - start) / step) + 2) if start + i * step <= hi + 1e-12]


def plot(series: list[dict], path: str | Path, title: str = "", xlabel: str = "", ylabel: str = "", identity: bool = False, width: int = 640,
         height: int = 440, xlim=None, ylim=None, logx: bool = False) -> Path:
    """series: [{name, x: [...], y: [...], line: bool, marker: bool, yerr: [...]|None}]."""
    L, R, T, B = 70, 180, 40, 55; pw, ph = width - L - R, height - T - B
    xs = [v for s in series for v in s["x"]]; ys = [v for s in series for v in s["y"]]
    if logx:
        xs = [math.log10(v) for v in xs if v > 0]
    x0, x1 = (xlim if xlim else (min(xs), max(xs))); y0, y1 = (ylim if ylim else (min(ys), max(ys)))
    if logx and xlim:
        x0, x1 = math.log10(xlim[0]), math.log10(xlim[1])
    if x1 - x0 < 1e-9:
        x0, x1 = x0 - 0.5, x1 + 0.5
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.5, y1 + 0.5
    padx, pady = 0.04 * (x1 - x0), 0.06 * (y1 - y0); x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady
    fx = lambda v: L + (((math.log10(v) if logx else v) - x0) / (x1 - x0)) * pw
    fy = lambda v: T + ph - ((v - y0) / (y1 - y0)) * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>', f'<text x="{L + pw / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
           f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>']
    for tx in _ticks(x0 + padx, x1 - padx):
        X = fx(10 ** tx if logx else tx); lab = f"{10 ** tx:g}" if logx else f"{tx:g}"
        if L - 1 <= X <= L + pw + 1:
            out.append(f'<line x1="{X:.1f}" y1="{T + ph}" x2="{X:.1f}" y2="{T + ph + 5}" stroke="#333"/><text x="{X:.1f}" y="{T + ph + 18}" text-anchor="middle">{lab}</text>')
    for ty in _ticks(y0 + pady, y1 - pady):
        Y = fy(ty)
        if T - 1 <= Y <= T + ph + 1:
            out.append(f'<line x1="{L - 5}" y1="{Y:.1f}" x2="{L}" y2="{Y:.1f}" stroke="#333"/><text x="{L - 8}" y="{Y + 4:.1f}" text-anchor="end">{ty:g}</text>')
    out.append(f'<text x="{L + pw / 2}" y="{height - 12}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text transform="translate(16,{T + ph / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    if identity:
        lo, hi = max(x0, y0), min(x1, y1); out.append(f'<line x1="{fx(10 ** lo if logx else lo):.1f}" y1="{fy(lo):.1f}" x2="{fx(10 ** hi if logx else hi):.1f}" y2="{fy(hi):.1f}" stroke="#999" stroke-dasharray="4 3"/>')
    for i, s in enumerate(series):
        c = COLOURS[i % len(COLOURS)]; pts = [(fx(x), fy(y)) for x, y in zip(s["x"], s["y"]) if (not logx or x > 0)]
        if s.get("line", True) and len(pts) > 1:
            out.append('<polyline fill="none" stroke="%s" stroke-width="1.5" points="%s"/>' % (c, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
        if s.get("marker", True):
            for (x, y) in pts:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{c}"/>')
        if s.get("yerr"):
            for (x, y), e in zip(pts, s["yerr"]):
                out.append(f'<line x1="{x:.1f}" y1="{fy(fy_inv(y, fy, y0, y1, T, ph) - e):.1f}" x2="{x:.1f}" y2="{fy(fy_inv(y, fy, y0, y1, T, ph) + e):.1f}" stroke="{c}"/>')
        out.append(f'<rect x="{L + pw + 12}" y="{T + 14 * i}" width="10" height="10" fill="{c}"/><text x="{L + pw + 26}" y="{T + 14 * i + 9}">{s["name"]}</text>')
    out.append("</svg>"); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(out)); return path


def fy_inv(Y, fy, y0, y1, T, ph):
    return y0 + (T + ph - Y) / ph * (y1 - y0)
