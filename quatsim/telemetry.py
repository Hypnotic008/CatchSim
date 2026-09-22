"""
Telemetry-style visuals.

ON USING SPACEX'S ASSETS: DON'T
-------------------------------
Their livestream overlay, engine-ring artwork, typography and logos are
SpaceX's intellectual property. A portfolio aimed at SpaceX is the worst
possible place to ship reproduced assets, and a recruiter who recognises the
artwork learns that you copied a screenshot rather than that you built a
telemetry system.

Everything drawn here is generated from simulation state. The ENGINE LAYOUT is
physical fact -- 33 Raptors in rings of 3, 10 and 20 -- not protected
expression, so drawing it from geometry is fine. The styling is deliberately
its own thing.

WHAT ACTUALLY MAKES THESE USEFUL
--------------------------------
Not the styling. Every plot here shows COMMANDED alongside ACHIEVED. A control
project that only plots what the vehicle did is showing an outcome; plotting
it against what the controller asked for is showing that the controller works,
and the gap between the two is the entire result. The reference signals come
from FlightLog, which is why the sequencer logs them.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Wedge

from . import quaternion as Q

# Instrument palette: dark ground, cyan for achieved, amber for commanded.
BG = "#0d1117"
FG = "#c9d1d9"
GRID = "#21262d"
C_ACHIEVED = "#2dd4bf"
C_COMMANDED = "#f0b429"
C_HOT = "#ff6b4a"
C_COLD = "#30363d"
C_TARGET = "#ff4d6d"


# ---------------------------------------------------------------------------
# engine layout -- physical geometry, drawn from first principles
# ---------------------------------------------------------------------------

def engine_positions(radius: float = 4.5):
    """
    Raptor positions on the thrust puck: 3 centre, 10 middle, 20 outer.

    Returns (xy array of shape (33,2), ring index per engine). The centre three
    are the ones that hot-stage and fly the final landing burn; the inner
    thirteen are the gimballing set; all 33 relight on Raptor 3.
    """
    pts, ring = [], []
    for n, r, idx in ((3, 0.18 * radius, 0), (10, 0.52 * radius, 1),
                      (20, 0.88 * radius, 2)):
        phase = np.pi / 2 if idx == 0 else 0.0
        for k in range(n):
            a = phase + 2 * np.pi * k / n
            pts.append((r * np.cos(a), r * np.sin(a)))
            ring.append(idx)
    return np.array(pts), np.array(ring)


def lit_engine_mask(n_lit: int) -> np.ndarray:
    """
    Which engines fire for a given count, chosen SYMMETRICALLY.

    Filling the rings in index order is wrong and not just cosmetically. Five
    engines taken as "the first five" puts two adjacent inner-ring engines on
    one side of the vehicle, which is a net side force and a net roll-plane
    torque the controller then has to trim out continuously. Real selections
    are balanced: opposed pairs so the thrust resultant stays on the
    centreline.

    3  -> the three centre engines
    5  -> centre three plus an OPPOSED inner pair (the flip set)
    13 -> centre three plus the full inner ring (the gimballing set)
    33 -> everything
    2  -> an opposed pair, the terminal hover set. V3 cannot hover on three:
          that needs 34% throttle against a ~40% floor.
    """
    n = max(0, min(int(n_lit), 33))
    _, ring = engine_positions()
    centre = np.where(ring == 0)[0]
    inner = np.where(ring == 1)[0]
    outer = np.where(ring == 2)[0]

    mask = np.zeros(33, dtype=bool)
    if n == 0:
        return mask

    if n <= 3:
        # take them spread around the centre cluster rather than adjacent
        pick = np.linspace(0, len(centre), n, endpoint=False).astype(int)
        mask[centre[pick]] = True
        return mask

    mask[centre] = True
    remaining = n - len(centre)

    if remaining > 0:
        take = min(remaining, len(inner))
        # opposed pairs: step halfway round the ring between successive picks
        half = len(inner) // 2
        order, used = [], set()
        for k in range(len(inner)):
            for cand in (k, (k + half) % len(inner)):
                if cand not in used:
                    used.add(cand)
                    order.append(cand)
            if len(order) >= take:
                break
        mask[inner[np.array(order[:take])]] = True
        remaining -= take

    if remaining > 0:
        half = len(outer) // 2
        order, used = [], set()
        for k in range(len(outer)):
            for cand in (k, (k + half) % len(outer)):
                if cand not in used:
                    used.add(cand)
                    order.append(cand)
            if len(order) >= remaining:
                break
        mask[outer[np.array(order[:remaining])]] = True

    return mask


def draw_engine_ring(ax, n_lit: int, radius: float = 4.5):
    """Engine status disc: hot engines filled, cold engines outlined."""
    pts, ring = engine_positions(radius)
    mask = lit_engine_mask(n_lit)

    ax.add_patch(Circle((0, 0), radius * 1.06, facecolor="#11161d",
                        edgecolor=GRID, lw=1.2))
    for (x, y), hot in zip(pts, mask):
        ax.add_patch(Circle((x, y), 0.085 * radius,
                            facecolor=C_HOT if hot else "none",
                            edgecolor=C_HOT if hot else C_COLD,
                            lw=1.1, alpha=1.0 if hot else 0.8))
    ax.set_xlim(-radius * 1.15, radius * 1.15)
    ax.set_ylim(-radius * 1.15, radius * 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -radius * 1.30, f"{int(mask.sum())} / 33 LIT", color=FG,
            ha="center", va="center", fontsize=9, family="monospace")


def draw_attitude_indicator(ax, q: np.ndarray, q_cmd: np.ndarray | None = None,
                            limit_deg: float = 6.0):
    """
    Tilt indicator: where the nose points, projected onto the horizontal plane.

    The dot's distance from centre is the tilt from vertical, its angle is the
    azimuth of the lean. This is a deliberately different instrument from an
    aircraft attitude ball, because the quantity that matters for a catch is a
    single scalar -- angle from vertical -- and the catch limit is a CIRCLE on
    this display. A pitch/roll ball would split that one number across two
    axes and hide it.

    Driven straight off the quaternion. No Euler conversion anywhere, which
    matters because the vehicle sits within a degree of vertical where an
    Euler representation is least trustworthy.
    """
    for rr, lab in ((limit_deg / 3, None), (2 * limit_deg / 3, None),
                    (limit_deg, None)):
        ax.add_patch(Circle((0, 0), rr, facecolor="none", edgecolor=GRID, lw=0.9))
    ax.add_patch(Circle((0, 0), 0.5, facecolor="none", edgecolor=C_TARGET,
                        lw=1.2, ls="--"))
    ax.plot([-limit_deg, limit_deg], [0, 0], color=GRID, lw=0.7)
    ax.plot([0, 0], [-limit_deg, limit_deg], color=GRID, lw=0.7)

    def project(qq):
        nose = Q.rotate(qq, np.array([1.0, 0.0, 0.0]))
        tilt = np.degrees(np.arccos(np.clip(nose[2], -1.0, 1.0)))
        h = np.hypot(nose[0], nose[1])
        if h < 1e-9:
            return 0.0, 0.0, tilt
        return tilt * nose[0] / h, tilt * nose[1] / h, tilt

    if q_cmd is not None:
        cx, cy, _ = project(q_cmd)
        ax.plot([cx], [cy], marker="+", color=C_COMMANDED, ms=11, mew=1.8)
    x, y, tilt = project(q)
    ax.plot([x], [y], marker="o", color=C_ACHIEVED, ms=7)

    ax.set_xlim(-limit_deg * 1.1, limit_deg * 1.1)
    ax.set_ylim(-limit_deg * 1.1, limit_deg * 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -limit_deg * 1.28, f"TILT {tilt:5.2f}\u00b0", color=FG,
            ha="center", va="center", fontsize=9, family="monospace")


# ---------------------------------------------------------------------------
# commanded vs achieved -- the plot that was missing
# ---------------------------------------------------------------------------

def tracking_3d(out: dict, target: np.ndarray,
                altitude_window: float | None = 120.0,
                path: str = "tracking_3d.png"):
    """
    Commanded path against achieved path, the way a controls paper shows it.

    This is the figure the project was missing. Plotting only the achieved
    trajectory shows an outcome; overlaying the reference shows that the
    controller tracks it, and the visible gap IS the result. Anyone reading a
    guidance and control project looks for this plot first.
    """
    r, r_ref = out["r"], out["r_ref"]
    if altitude_window is not None:
        m = r[:, 2] < target[2] + altitude_window
        if m.sum() > 10:
            r, r_ref = r[m], r_ref[m]

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(r_ref[:, 0], r_ref[:, 1], r_ref[:, 2], "--", color="k", lw=2.0,
            label="Path command", zorder=3)
    ax.plot(r[:, 0], r[:, 1], r[:, 2], "-", color="#d1495b", lw=1.4,
            label="Achieved (6-DOF)", zorder=4)
    ax.scatter(*target, color="#2e4057", s=90, marker="X",
               label="Catch point", depthshade=False)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.view_init(elev=20, azim=-60)
    ax.set_title("Divert tracking: commanded vs achieved")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def tracking_error(out: dict, path: str = "tracking_error.png"):
    """
    Position and attitude tracking error against time, per axis.

    Per-axis rather than as a magnitude, because the three axes have different
    causes: vertical error is throttle authority, lateral error is the tilt
    limit, and attitude error is the inner loop. A single norm hides which one
    is misbehaving.
    """
    t = out["t"]
    err = out["r"] - out["r_ref"]
    att = np.array([np.degrees(Q.angle_between(a, b))
                    for a, b in zip(out["q"], out["q_ref"])])

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for i, (lab, c) in enumerate((("x", "#d1495b"), ("y", "#00798c"),
                                  ("z", "#edae49"))):
        ax[0].plot(t, err[:, i], color=c, lw=1.3, label=f"{lab} error")
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[0].set_ylabel("position error (m)")
    ax[0].legend(fontsize=9, ncol=3)
    ax[0].grid(alpha=0.25)
    ax[0].set_title("Tracking error: reference minus achieved")

    ax[1].plot(t, att, color="#2e4057", lw=1.3)
    ax[1].set_ylabel("attitude error (deg)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_yscale("log")
    ax[1].grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# booster geometry -- an actual body, not a line
# ---------------------------------------------------------------------------

def booster_mesh(q: np.ndarray, r: np.ndarray, length: float = 72.3,
                 diameter: float = 9.0, n_sides: int = 20,
                 fin_azimuths=(0.0, 90.0, 180.0)):
    """
    Cylinder body plus three grid fins, positioned and oriented by quaternion.

    Returns (body_quads, fin_quads) as lists of vertex arrays, ready for
    Poly3DCollection.

    Three fins, not four, at 90/90/180 -- the V3 arrangement. They are drawn as
    fixed structure because that is what they are: unlike Falcon 9's, Super
    Heavy's fins never fold, so there is no stowed state to render.
    """
    R = 0.5 * diameter
    half = 0.5 * length
    ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    def to_world(p_body):
        return r + Q.rotate(q, np.asarray(p_body, dtype=float))

    body = []
    for k in range(n_sides):
        a0, a1 = ang[k], ang[(k + 1) % n_sides]
        quad = [(-half, R * np.cos(a0), R * np.sin(a0)),
                (-half, R * np.cos(a1), R * np.sin(a1)),
                (+half, R * np.cos(a1), R * np.sin(a1)),
                (+half, R * np.cos(a0), R * np.sin(a0))]
        body.append(np.array([to_world(p) for p in quad]))

    # Fins sit forward of the CoM, now on the methane tank rather than the
    # interstage, so they are drawn nearer mid-body than on earlier vehicles.
    fin_x = 0.28 * length
    fin_span, fin_chord = 0.9 * R, 0.10 * length
    fins = []
    for az in np.radians(fin_azimuths):
        c, s = np.cos(az), np.sin(az)
        quad = [(fin_x - fin_chord / 2, R * c, R * s),
                (fin_x + fin_chord / 2, R * c, R * s),
                (fin_x + fin_chord / 2, (R + fin_span) * c, (R + fin_span) * s),
                (fin_x - fin_chord / 2, (R + fin_span) * c, (R + fin_span) * s)]
        fins.append(np.array([to_world(p) for p in quad]))

    return body, fins


# ---------------------------------------------------------------------------
# the instrument panel
# ---------------------------------------------------------------------------

def telemetry_panel(out: dict, frame: int | None = None,
                    target: np.ndarray | None = None,
                    path: str = "telemetry_panel.png"):
    """
    Instrument panel at one instant: engine ring, tilt indicator, readouts,
    and the altitude/speed traces with commanded values overlaid.

    Original design. The layout convention -- engine status on the left,
    attitude beside it, numeric readouts on the right -- is functional and
    common to every launch telemetry display ever built; the artwork is not
    copied from anyone.
    """
    i = len(out["t"]) - 1 if frame is None else int(frame)
    t = out["t"]
    r, v = out["r"], out["v"]
    speed = np.linalg.norm(v, axis=1)

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.6, 1.6],
                          hspace=0.35, wspace=0.3)

    ax_eng = fig.add_subplot(gs[:, 0], facecolor=BG)
    draw_engine_ring(ax_eng, int(out["n_lit"][i]))

    ax_att = fig.add_subplot(gs[:, 1], facecolor=BG)
    draw_attitude_indicator(ax_att, out["q"][i], out["q_ref"][i])

    def trace(ax, y, yref, label, unit):
        ax.set_facecolor(BG)
        if yref is not None:
            ax.plot(t, yref, "--", color=C_COMMANDED, lw=1.4, label="commanded")
        ax.plot(t, y, color=C_ACHIEVED, lw=1.5, label="achieved")
        ax.axvline(t[i], color=FG, lw=0.9, alpha=0.6)
        ax.plot([t[i]], [y[i]], "o", color=C_ACHIEVED, ms=6)
        ax.set_ylabel(f"{label} ({unit})", color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(alpha=0.18, color=GRID)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)

    ax_alt = fig.add_subplot(gs[0, 2])
    trace(ax_alt, r[:, 2], out["r_ref"][:, 2], "ALTITUDE", "m")

    ax_spd = fig.add_subplot(gs[1, 2])
    trace(ax_spd, speed, None, "SPEED", "m/s")
    ax_spd.set_xlabel("time (s)", color=FG, fontsize=9)

    ax_lat = fig.add_subplot(gs[0, 3])
    lat = np.linalg.norm(r[:, :2] - (target[:2] if target is not None else 0.0),
                         axis=1)
    latref = np.linalg.norm(
        out["r_ref"][:, :2] - (target[:2] if target is not None else 0.0), axis=1)
    trace(ax_lat, lat, latref, "LATERAL", "m")

    ax_txt = fig.add_subplot(gs[1, 3], facecolor=BG)
    ax_txt.axis("off")
    rows = [
        ("T+", f"{t[i]:7.2f} s"),
        ("ALT", f"{r[i, 2]:7.1f} m"),
        ("VEL", f"{speed[i]:7.2f} m/s"),
        ("TILT", f"{out['tilt'][i]:7.3f} deg"),
        ("THR", f"{out['throttle'][i]:7.2f}"),
        ("PROP", f"{out['prop'][i] / 1000:7.2f} t"),
    ]
    for k, (lab, val) in enumerate(rows):
        ax_txt.text(0.02, 0.92 - k * 0.16, lab, color="#7d8590",
                    fontsize=10, family="monospace", va="top")
        ax_txt.text(0.45, 0.92 - k * 0.16, val, color=FG,
                    fontsize=12, family="monospace", va="top")

    fig.suptitle(f"SUPER HEAVY  \u00b7  TERMINAL PHASE  \u00b7  {out['segment'][i].upper()}",
                 color=FG, fontsize=13, family="monospace")
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# full-mission animation
# ---------------------------------------------------------------------------

def tower_geometry(catch_altitude: float = 105.0, height: float = 146.0,
                   width: float = 12.0, arm_length: float = 15.0,
                   arm_gap: float = 12.0, pad=(-30.0, 0.0),
                   r: np.ndarray | None = None,
                   target: np.ndarray | None = None):
    """Wireframe catch tower with tracked, hinged chopstick arms.

    Geometry mirrors :class:`quatsim.tower.CatchTower`: a rectangular tower
    lattice plus two arms hinged at the tower face.  When ``r`` and ``target``
    are supplied, the carriage/arms translate inside the same estimated
    tracking envelope used by the catch model.
    """
    px, py = pad
    hw = 0.5 * width
    corners = [(px-hw, py-hw), (px+hw, py-hw),
               (px+hw, py+hw), (px-hw, py+hw)]
    tower = []
    for i, (x0, y0) in enumerate(corners):
        x1, y1 = corners[(i + 1) % 4]
        tower.append(np.array([[x0, y0, 0.0], [x0, y0, height]]))
        tower.append(np.array([[x0, y0, 0.0], [x1, y1, 0.0]]))
        tower.append(np.array([[x0, y0, height], [x1, y1, height]]))
        for frac in (0.33, 0.66):
            z = height * frac
            tower.append(np.array([[x0, y0, z], [x1, y1, z]]))

    offset = np.zeros(3)
    if r is not None and target is not None:
        d = np.asarray(r, float) - np.asarray(target, float)
        offset = np.array([
            np.clip(d[0], -3.0, 3.0),
            np.clip(d[1], -5.0, 5.0),
            np.clip(d[2], -3.0, 3.0),
        ])

    arms = []
    hinge_x = px + hw + offset[0]
    z = catch_altitude + offset[2]
    for side in (+1, -1):
        y = py + side * 0.5 * arm_gap + offset[1]
        tip_x = hinge_x + arm_length
        tip_y = y - side * 3.0
        # Main arm, angled hinge-to-tip in plan view.
        arms.append(np.array([[hinge_x, y, z], [tip_x, tip_y, z]]))
        # Short closing tip.
        arms.append(np.array([[tip_x, tip_y, z],
                              [tip_x, tip_y - side * 2.0, z]]))
        # Hinge linkage back to the tower face.
        arms.append(np.array([[px + offset[0], py + offset[1], z],
                              [hinge_x, y, z]]))
    return tower, arms


def visual_catch_point(target: np.ndarray, pad=(-30.0, 0.0),
                       width: float = 12.0, arm_length: float = 15.0,
                       arm_gap: float = 12.0) -> np.ndarray:
    """Return the visual seating point between the two 15 m chopstick tips.

    This is a visualization-only coordinate.  The simulator's physical catch
    target remains ``target``; only the rendered booster is eased toward this
    point near the end of the animation.  Keeping this separate prevents the
    artwork from changing the guidance/catch physics.
    """
    px, py = pad
    hinge_x = px + 0.5 * width
    # Both arms extend in +x, so their tip midpoint is halfway along the arm.
    return np.asarray(target, dtype=float) + np.array([
        hinge_x + 0.5 * arm_length - target[0],
        py - target[1],
        0.0,
    ])


def _smooth_visual_position(p: np.ndarray, target: np.ndarray,
                            visual_target: np.ndarray,
                            altitude_start: float = 180.0) -> np.ndarray:
    """Smoothly move only the rendered booster onto the visual catch point."""
    p = np.asarray(p, dtype=float)
    target = np.asarray(target, dtype=float)
    vt = np.asarray(visual_target, dtype=float)
    span = max(float(altitude_start - target[2]), 1.0)
    u = np.clip((altitude_start - p[2]) / span, 0.0, 1.0)
    # smoothstep: zero slope at both ends, avoiding a visible snap.
    w = u * u * (3.0 - 2.0 * u)
    return p + w * (vt - target)


def _interp_quaternion(q0: np.ndarray, q1: np.ndarray, f: float) -> np.ndarray:
    """Short-arc normalized linear interpolation for closely spaced log samples."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - f) * q0 + f * q1
    return q / max(np.linalg.norm(q), 1e-15)


def _interp_log_state(out: dict, t: float):
    """Interpolate continuous state for animation; discrete telemetry is nearest-sample."""
    tt = np.asarray(out["t"], dtype=float)
    j = int(np.searchsorted(tt, t, side="right"))
    if j <= 0:
        return 0, 0, 0.0, np.asarray(out["r"][0]), np.asarray(out["q"][0])
    if j >= len(tt):
        k = len(tt) - 1
        return k, k, 0.0, np.asarray(out["r"][k]), np.asarray(out["q"][k])
    k = j - 1
    f = (t - tt[k]) / max(tt[j] - tt[k], 1e-15)
    r = (1.0 - f) * out["r"][k] + f * out["r"][j]
    q = _interp_quaternion(out["q"][k], out["q"][j], f)
    return k, j, f, r, q


def animate_mission(out: dict, target: np.ndarray, pad=(-30.0, 0.0),
                    n_frames: int = 200, fps: int = 25,
                    path: str = "mission.gif", hold_seconds: float = 2.0,
                    visual_align_altitude: float = 180.0):
    """
    Hot-stage separation to catch, in one animation.

    THE TIMESCALE PROBLEM, AND HOW IT IS HANDLED
    --------------------------------------------
    This flight spans an 11 s burn, a 213 s ballistic coast, and a 26 s divert.
    Sampling frames uniformly in TIME spends three quarters of the animation on
    a vehicle falling quietly through vacuum and blurs every burn into a couple
    of frames. Sampling uniformly in LOG INDEX is not much better, because the
    integrator uses a fixed step.

    So frames are distributed by arc length along the trajectory instead --
    dense where the vehicle is manoeuvring or moving fast, sparse through the
    quiet middle. The result shows the flip, the burn and the catch at a
    watchable rate without a 4-minute GIF.

    The inset axes hold a zoomed terminal view, because at a 100 km scale the
    40 m divert is smaller than a pixel.
    """
    r, qs = out["r"], out["q"]

    # Build a continuous frame-time schedule, then interpolate between the
    # simulator's ~0.2 s logged samples.  This keeps the intentionally
    # compressed summary animation while removing the visible 0.2 s stepping.
    seglen = np.linalg.norm(np.diff(r, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    n_base = max(n_frames - 30, 1)
    base_arc = np.linspace(0.0, arc[-1], n_base)
    base_idx = np.searchsorted(arc, base_arc, side="left")

    # Reserve extra frame times around engine transitions, but do NOT force
    # the animation to snap to integer log indices.
    seg_names = np.asarray(out['segment'])
    n_lit = np.asarray(out['n_lit'])
    forced_idx = {0, len(r) - 1}
    for name in np.unique(seg_names):
        m = np.flatnonzero(seg_names == name)
        if len(m):
            forced_idx.update(np.linspace(m[0], m[-1], min(5, len(m))).astype(int))
    m3 = n_lit == 3
    edges = np.flatnonzero(np.diff(np.r_[False, m3, False].astype(int)))
    for a, b in edges.reshape(-1, 2):
        forced_idx.update(np.linspace(a, b - 1, min(8, b - a)).astype(int))

    all_idx = np.unique(np.r_[base_idx, list(forced_idx)])
    if len(all_idx) > n_frames:
        fset = np.array(sorted(forced_idx), dtype=int)
        keep_extra = max(n_frames - len(fset), 0)
        extra = np.setdiff1d(all_idx, fset)
        if len(extra) > keep_extra:
            extra = extra[np.linspace(0, len(extra)-1, keep_extra).astype(int)]
        all_idx = np.unique(np.r_[fset, extra])

    # Convert the selected log locations to their actual timestamps.  We then
    # interpolate every rendered frame at a fractional time, so playback is
    # smooth even when the simulation log itself is sampled every 0.2 s.
    frame_times = out["t"][all_idx].astype(float)
    if hold_seconds > 0.0:
        frame_times = np.concatenate([frame_times,
                                      np.full(int(hold_seconds * fps), frame_times[-1])])
    fig = plt.figure(figsize=(11, 8), facecolor=BG)
    ax = fig.add_axes([0.02, 0.05, 0.62, 0.88], projection="3d",
                      facecolor=BG)
    axi = fig.add_axes([0.68, 0.46, 0.29, 0.44], facecolor=BG)
    axt = fig.add_axes([0.68, 0.06, 0.29, 0.32], facecolor=BG)

    xr = (min(r[:, 0].min(), 0) / 1000, max(r[:, 0].max(), 0) / 1000)
    zr = (0, r[:, 2].max() / 1000 * 1.1)

    def style3d(a):
        for pane in (a.xaxis, a.yaxis, a.zaxis):
            pane.set_pane_color((0.05, 0.07, 0.09, 1.0))
            pane._axinfo["grid"]["color"] = GRID
        a.tick_params(colors=FG, labelsize=7)
        a.set_xlabel("downrange (km)", color=FG, fontsize=8)
        a.set_ylabel("cross (km)", color=FG, fontsize=8)
        a.set_zlabel("altitude (km)", color=FG, fontsize=8)

    readout = fig.text(0.68, 0.955, "", color=FG, fontsize=9,
                       family="monospace")

    def draw(f):
        t_now = float(frame_times[f])
        i0, i1, frac, p_now, q_now = _interp_log_state(out, t_now)
        ax.clear(); axi.clear(); axt.clear()

        # Draw the logged trajectory plus the interpolated current point.
        ax.plot(r[:i0 + 1, 0] / 1000, r[:i0 + 1, 1] / 1000, r[:i0 + 1, 2] / 1000,
                color=C_ACHIEVED, lw=1.6)
        if i1 != i0:
            ax.plot([r[i0, 0] / 1000, p_now[0] / 1000],
                    [r[i0, 1] / 1000, p_now[1] / 1000],
                    [r[i0, 2] / 1000, p_now[2] / 1000],
                    color=C_ACHIEVED, lw=1.6)
        ax.scatter([pad[0]], [pad[1]], [0], color=C_TARGET, s=60, marker="^",
                   depthshade=False)
        # Vehicle stays on the simulated trajectory until the terminal
        # approach, then is smoothly eased toward the visual seating point.
        # This is renderer-only; out["r"] and all physics remain untouched.
        visual_target = visual_catch_point(target, pad=pad)
        p_vis_m = _smooth_visual_position(p_now, target, visual_target,
                                          altitude_start=visual_align_altitude)
        nose = Q.rotate(q_now, np.array([1.0, 0.0, 0.0])) * 4.0
        p = p_vis_m / 1000
        ax.plot([p[0] - nose[0], p[0] + nose[0]],
                [p[1] - nose[1], p[1] + nose[1]],
                [p[2] - nose[2], p[2] + nose[2]],
                color=C_HOT, lw=4.0, solid_capstyle="round")
        # Use the same wireframe/hinged-arm asset as the realtime animation.
        if p_now[2] < target[2] + 3000.0:
            tower, arms = tower_geometry(catch_altitude=float(target[2]),
                                          r=p_now, target=target, pad=pad,
                                          arm_length=15.0)
            for segd in tower:
                ax.plot(segd[:, 0] / 1000, segd[:, 1] / 1000, segd[:, 2] / 1000,
                        color="#8b949e", lw=1.0, alpha=0.9)
            for segd in arms:
                ax.plot(segd[:, 0] / 1000, segd[:, 1] / 1000, segd[:, 2] / 1000,
                        color=C_TARGET, lw=2.2, alpha=0.95)
        ax.set_xlim(*xr); ax.set_ylim(-20, 20); ax.set_zlim(*zr)
        ax.view_init(elev=16, azim=-72)
        style3d(ax)
        ax.set_title(f"T+{t_now:6.2f} s   {out['segment'][i0].upper()}",
                     color=FG, fontsize=11, family="monospace")

        # terminal inset
        # mask must be applied to the SAME slice it was built from
        rr = r[:i0 + 1]
        m = rr[:, 2] < target[2] + 400
        axi.plot(rr[m, 0], rr[m, 2], color=C_ACHIEVED, lw=1.4)
        vc = visual_catch_point(target, pad=pad)
        axi.scatter([vc[0]], [vc[2]], color=C_TARGET, s=50, marker="X")
        axi.set_xlim(-80, 80); axi.set_ylim(0, 400)
        axi.set_title("terminal (m)", color=FG, fontsize=8, family="monospace")
        for a in (axi, axt):
            a.tick_params(colors=FG, labelsize=7)
            for sp in a.spines.values():
                sp.set_color(GRID)
            a.grid(alpha=0.15, color=GRID)

        axt.plot(out["t"][:i0 + 1], out["r"][:i0 + 1, 2] / 1000,
                 color=C_ACHIEVED, lw=1.3)
        axt.set_xlim(0, out["t"][-1]); axt.set_ylim(0, zr[1])
        axt.set_xlabel("t (s)", color=FG, fontsize=8)
        axt.set_title("altitude (km)", color=FG, fontsize=8, family="monospace")

        # Update ONE persistent text artist rather than calling fig.text each
        # frame. ax.clear() only clears axes -- figure-level artists survive,
        # so a fresh fig.text per frame stacks every previous readout on top of
        # the last and the digits overlap into nonsense ("39/33" from a 3 and
        # a 5 drawn over each other). Create once, set_text thereafter.
        engines = int(out["n_lit"][i0])
        readout.set_text(f"ENGINES {engines:2d}/33    "
                         f"ALT {p_now[2]/1000:6.1f} km    "
                         f"V {np.linalg.norm(out['v'][i0]):6.0f} m/s")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frame_times), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path
