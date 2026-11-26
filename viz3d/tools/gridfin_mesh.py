"""Super Heavy V3 grid fin as a light polygon mesh for the 3-D viewer.

Same parameters as the CFD fin (rocketcfd/vehicles/grid_fin.py, photo-measured,
dimensions.md): 3.5 m wide, 4.2 m reach from the hull, 0.5 m deep, 8 diamonds
across, diagonal lattice plates with semi-elliptical notches between crossings
on the face toward the engines (the teeth), notched edge plates, a V-notched
cross plate, a solid root triangle and the neck.

Plates are single surfaces (the viewer draws them double-sided). Output: the
viewer's fin format, int16 millimetres, non-indexed triangles, fin-local axes
x along the booster axis (toothed face at -depth/2, toward the engines), y
around the hull, z radial from the hull surface.

  python viz3d/tools/gridfin_mesh.py viz3d/assets/gridfin_v3.i16
"""
import sys
from dataclasses import dataclass

import numpy as np

S2 = np.sqrt(2.0)


@dataclass
class Fin:
    width: float = 3.5
    depth: float = 0.5
    diamonds: int = 8
    neck: float = 0.30
    root_half: float = 0.475
    rows_outer: int = 6
    chamfer_rows: int = 2
    tooth_depth: float = 0.45
    tooth_land: float = 0.006
    neck_width: float = 0.6
    neck_depth: float = 0.4

    s = property(lambda self: self.width / self.diamonds)
    p = property(lambda self: self.s / S2)
    w = property(lambda self: 0.5 * self.width)
    z_root = property(lambda self: self.neck)
    z_cross = property(lambda self: self.neck + (self.w - self.root_half))
    z_tip = property(lambda self: self.z_cross + self.rows_outer * self.s)
    chamfer = property(lambda self: self.chamfer_rows * self.s)


def outline(F):
    w, c, r0 = F.w, F.chamfer, F.root_half
    return np.array([(-r0, F.z_root), (r0, F.z_root), (w, F.z_cross), (w, F.z_tip - c),
                     (w - c, F.z_tip), (-(w - c), F.z_tip), (-w, F.z_tip - c), (-w, F.z_cross)])


def clip_line(poly, p0, d):
    """Parameter range [t0, t1] of the line p0 + t d inside the convex polygon."""
    t0, t1 = -1e9, 1e9
    n = len(poly)
    cen = poly.mean(0)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]])
        if nrm @ (cen - a) > 0:
            nrm = -nrm                                  # outward normal
        den = nrm @ d
        num = nrm @ (a - p0)
        if abs(den) < 1e-12:
            if num < 0:
                return None
            continue
        t = num / den
        if den > 0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
    return (t0, t1) if t1 - t0 > 1e-6 else None


def notch(along, pitch_half, bdepth, land, shape="ellipse"):
    """Depth of the notch into the toothed face at `along` (crossings at
    multiples of 2*pitch_half)."""
    c = 2.0 * pitch_half * (np.floor(along / (2.0 * pitch_half)) + 0.5)
    half = pitch_half - 0.5 * land
    q = np.clip(np.abs(along - c) / half, 0.0, 1.0)
    return bdepth * (np.sqrt(1.0 - q * q) if shape == "ellipse" else (1.0 - q))


def strip(tris, P0, P1, along0, along1, F, pitch_half=None, shape="ellipse", per=10):
    """Plate between (y,z) points P0 -> P1, full depth, the -x face notched."""
    hd, bdepth = 0.5 * F.depth, F.tooth_depth * F.depth
    L = float(np.linalg.norm(P1 - P0))
    if L < 1e-4:
        return
    if pitch_half is None:
        n = 1
    else:
        n = max(2, int(np.ceil(abs(along1 - along0) / (2 * pitch_half) * per)))
        # put samples exactly on crossings and notch centres too
    ts = np.linspace(0.0, 1.0, n + 1)
    if pitch_half is not None:
        a = along0 + ts * (along1 - along0)
        k0, k1 = sorted((along0 / pitch_half, along1 / pitch_half))
        extra = [(m * pitch_half - along0) / (along1 - along0) for m in range(int(np.ceil(k0)), int(np.floor(k1)) + 1)]
        ts = np.unique(np.clip(np.concatenate([ts, extra]), 0, 1))
        a = along0 + ts * (along1 - along0)
        xf = -hd + notch(a, pitch_half, bdepth, F.tooth_land, shape)
    else:
        xf = np.full_like(ts, -hd)
    yz = P0[None] + ts[:, None] * (P1 - P0)[None]
    for i in range(len(ts) - 1):
        a0 = (xf[i], *yz[i]); a1 = (xf[i + 1], *yz[i + 1])
        b0 = (hd, *yz[i]); b1 = (hd, *yz[i + 1])
        tris += [(a0, a1, b1), (a0, b1, b0)]


def fin_triangles(F=Fin()):
    tris = []
    poly = outline(F)
    s, p, zc = F.s, F.p, F.z_cross
    span = 3 * (F.z_tip + F.width)
    # lattice: family A (y + (z - zc) = k s), along-coordinate v; family B mirrored
    for fam in (+1, -1):
        for k in range(-40, 41):
            p0 = np.array([0.0, zc]) + np.array([k * s * 0.5, k * s * 0.5]) * np.array([fam, 1.0])
            # family A: points with y + zp = k s -> p0 = (k s/2, zc + k s/2), dir (1,-1)
            d = np.array([fam, -1.0]) / S2
            r = clip_line(poly, p0, d)
            if r is None:
                continue
            P0, P1 = p0 + r[0] * d, p0 + r[1] * d
            # along-coordinate = the other family's u/v: crossings at multiples of p
            other = lambda P: (-fam * P[0] + (P[1] - zc)) / S2
            strip(tris, P0, P1, other(P0), other(P1), F, 0.5 * p)
    # edge plates
    for i in range(len(poly)):
        A, B = poly[i], poly[(i + 1) % len(poly)]
        if A[1] < zc - 1e-9 or B[1] < zc - 1e-9:                    # tapered root edges: smooth
            strip(tris, A, B, 0, 0, F, None)
            continue
        if abs(A[0] - B[0]) < 1e-9:                                  # sides
            strip(tris, A, B, A[1] - zc, B[1] - zc, F, 0.5 * s)
        elif abs(A[1] - B[1]) < 1e-9:                                # outer end
            off = 0.5 * s * (F.diamonds % 2)
            strip(tris, A, B, A[0] + off, B[0] + off, F, 0.5 * s)
        else:                                                        # cut corners, along a plate
            f = 1.0 if A[0] + B[0] > 0 else -1.0
            other = lambda P: (-f * P[0] + (P[1] - zc)) / S2
            strip(tris, A, B, other(A), other(B), F, 0.5 * p)
    # cross plate with V notches
    off = 0.5 * s * (F.diamonds % 2)
    strip(tris, np.array([-F.w, zc]), np.array([F.w, zc]), -F.w + off, F.w + off, F, 0.5 * s, "v")
    # solid root triangle (both faces) and the neck box
    hd = 0.5 * F.depth
    T = [(-F.root_half, F.z_root), (F.root_half, F.z_root), (0.0, zc)]
    for x in (-hd, hd):
        tris.append(tuple((x, y, z) for y, z in T))
    for (y0, z0), (y1, z1) in ((T[0], T[1]), (T[1], T[2]), (T[2], T[0])):
        tris += [((-hd, y0, z0), (-hd, y1, z1), (hd, y1, z1)), ((-hd, y0, z0), (hd, y1, z1), (hd, y0, z0))]
    nx, ny = 0.5 * F.neck_depth, 0.5 * F.neck_width
    z0, z1 = -0.05, F.z_root + 0.05
    C = [(x, y, z) for x in (-nx, nx) for y in (-ny, ny) for z in (z0, z1)]
    for f in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        q = [C[j] for j in f]
        tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return np.array(tris, float)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "viz3d/assets/gridfin_v3.i16"
    T = fin_triangles()
    np.round(T * 1000.0).astype("<i2").tofile(out)
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    print(f"{len(T)} triangles, bounds {np.round(lo, 3)} .. {np.round(hi, 3)} -> {out}")
