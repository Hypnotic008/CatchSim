"""
Visualization.

Three outputs, each answering a different question:

    flight_dashboard   did it fly correctly?      -> time histories
    trajectory_3d      what did it look like?     -> path + body axes
    animate            show someone who is not
                       going to read a plot        -> GIF

DESIGN NOTE ON ATTITUDE PLOTS
-----------------------------
Body orientation is drawn from the quaternion by rotating the body basis
vectors directly. Nothing here converts to Euler angles, which matters for the
same reason it has mattered all along: the booster spends the terminal phase
within a degree or two of vertical, and a pitch/yaw plot near vertical is
exactly where the Euler representation is least trustworthy. The tilt-from-
vertical scalar is unambiguous everywhere and is what the catch criteria use,
so it is what gets plotted.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import quaternion as Q

# Muted palette; the booster is drawn in near-black so the axes read against it.
C_BODY = "#22252a"
C_X = "#d1495b"      # body x, the nose / thrust direction
C_Y = "#00798c"
C_Z = "#edae49"
C_PATH = "#2e4057"
C_TARGET = "#d1495b"


# ---------------------------------------------------------------------------
# time histories
# ---------------------------------------------------------------------------

def flight_dashboard(out: dict, target: np.ndarray | None = None,
                     path: str = "flight_dashboard.png"):
    """
    Six-panel summary of a run.

    Segment boundaries are marked on every panel, because most of the
    interesting behaviour in this sim happens AT a handoff rather than inside
    a phase -- the landing-burn-to-hover transition is where a badly chosen
    control mode shows up.
    """
    t = out["t"]
    r, v = out["r"], out["v"]
    speed = np.linalg.norm(v, axis=1)
    lateral = np.linalg.norm(r[:, :2] - (target[:2] if target is not None else 0.0),
                             axis=1)

    # find where the segment label changes
    segs = out["segment"]
    bounds = [i for i in range(1, len(segs)) if segs[i] != segs[i - 1]]

    fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    ax = ax.ravel()

    def mark(a):
        for i in bounds:
            a.axvline(t[i], color="0.6", lw=0.8, ls="--", zorder=0)
        a.grid(alpha=0.25)

    ax[0].plot(t, r[:, 2], color=C_PATH)
    if target is not None:
        ax[0].axhline(target[2], color=C_TARGET, lw=1.0, ls=":")
    ax[0].set_ylabel("altitude (m)")
    ax[0].set_title("Altitude")

    ax[1].plot(t, speed, color=C_PATH)
    ax[1].set_ylabel("speed (m/s)")
    ax[1].set_title("Speed")

    ax[2].plot(t, lateral, color=C_PATH)
    ax[2].set_ylabel("lateral error (m)")
    ax[2].set_yscale("symlog", linthresh=0.1)
    ax[2].set_title("Lateral distance to target")

    ax[3].plot(t, out["tilt"], color=C_PATH)
    ax[3].axhline(0.5, color=C_TARGET, lw=1.0, ls=":")
    ax[3].set_ylabel("tilt from vertical (deg)")
    ax[3].set_yscale("symlog", linthresh=0.01)
    ax[3].set_title("Attitude error (catch limit 0.5 deg)")

    ax[4].plot(t, out["throttle"], color=C_PATH)
    ax[4].set_ylabel("throttle")
    ax[4].set_ylim(-0.05, 1.05)
    ax[4].set_xlabel("time (s)")
    ax[4].set_title("Throttle")

    ax[5].plot(t, out["prop"] / 1000.0, color=C_PATH)
    ax[5].set_ylabel("propellant (t)")
    ax[5].set_xlabel("time (s)")
    ax[5].set_title("Propellant remaining")

    for a in ax:
        mark(a)

    # label the phases along the top
    names, starts = [], [0] + bounds
    for i, s in enumerate(starts):
        names.append((t[s], segs[s]))
    for tt, nm in names:
        ax[0].annotate(nm, xy=(tt, ax[0].get_ylim()[1]), fontsize=8,
                       rotation=90, va="top", ha="right", color="0.35")

    fig.suptitle("Super Heavy terminal phase: landing burn to catch", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3D trajectory
# ---------------------------------------------------------------------------

def _booster_segments(r, q, length=71.0, scale=1.0):
    """
    Line segments drawing the booster body at one instant.

    The body is a line from the engine end to the nose, drawn along the body
    x-axis rotated by q -- so the picture is generated from the same quaternion
    the dynamics used, with no intermediate representation to get wrong.
    """
    nose = Q.rotate(q, np.array([1.0, 0.0, 0.0]))
    half = 0.5 * length * scale
    return np.array([r - nose * half, r + nose * half])


def trajectory_3d(out: dict, target: np.ndarray | None = None,
                  n_triads: int = 8, altitude_window: float | None = 140.0,
                  equal_aspect: bool = True,
                  path: str = "trajectory_3d.png"):
    """
    Descent path with the vehicle drawn at intervals.

    CROPPED BY DEFAULT, and that is not cosmetic. The full run drops 1100 m
    vertically while moving 40 m sideways, so a 3D view of all of it is a
    vertical line: the axis scales differ by more than 10x, the vehicle and its
    body triad get stretched into meaningless streaks, and the divert -- the
    only part with any 3D structure -- is invisible. Cropping to the last
    altitude_window metres and forcing equal aspect gives a picture whose
    geometry can be trusted. Pass altitude_window=None for the full path,
    knowing it will be distorted.

    The body axis triad is drawn by rotating the body basis vectors with the
    same quaternion the dynamics used, so nothing is re-derived through an
    intermediate representation. Red is body x, the nose and thrust direction.
    Watching it stay near-vertical while the vehicle translates sideways is the
    point of the figure: the booster leans by a couple of degrees, moves, and
    comes back upright.
    """
    r, qs = out["r"], out["q"]

    if altitude_window is not None and target is not None:
        mask = r[:, 2] < target[2] + altitude_window
        if mask.sum() < 10:
            mask = np.ones(len(r), dtype=bool)
    else:
        mask = np.ones(len(r), dtype=bool)
    rr, qq = r[mask], qs[mask]

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rr[:, 0], rr[:, 1], rr[:, 2], color=C_PATH, lw=1.6, label="CoM path")

    # scale the drawn vehicle to the plot, not to reality: at true 71 m the
    # booster is taller than the cropped window and hides the path.
    # NumPy 2 removed ndarray.ptp(); np.ptp() is the portable form.
    span = max(np.ptp(rr[:, 0]), np.ptp(rr[:, 2]), 1.0)
    body_len = 0.35 * span
    axis_len = 0.12 * span

    idx = np.linspace(0, len(rr) - 1, n_triads).astype(int)
    body_lines = []
    for i in idx:
        body_lines.append(_booster_segments(rr[i], qq[i], length=body_len))
        for vec, c in ((np.array([1.0, 0, 0]), C_X),
                       (np.array([0, 1.0, 0]), C_Y),
                       (np.array([0, 0, 1.0]), C_Z)):
            d = Q.rotate(qq[i], vec) * axis_len
            ax.plot(*zip(rr[i], rr[i] + d), color=c, lw=1.8, alpha=0.95)

    ax.add_collection3d(Line3DCollection(body_lines, colors=C_BODY, lw=4.0,
                                         alpha=0.5))

    if target is not None:
        ax.scatter(*target, color=C_TARGET, s=110, marker="X",
                   label="catch point", depthshade=False)
        arm = 0.25 * span
        ax.plot([target[0] - arm, target[0] + arm], [target[1], target[1]],
                [target[2], target[2]], color=C_TARGET, lw=2.5, alpha=0.6,
                label="chopstick plane")

    if equal_aspect:
        # Equal box aspect so a metre reads the same on every axis. Without
        # this the lean angles in the picture are lies.
        xs, ys, zs = rr[:, 0], rr[:, 1], rr[:, 2]
        cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
        half = 0.5 * max(np.ptp(xs), np.ptp(ys), np.ptp(zs), 1.0) * 1.25
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(max(cz - half, 0.0), cz + half)
        ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("downrange x (m)")
    ax.set_ylabel("crossrange y (m)")
    ax.set_zlabel("altitude z (m)")
    ax.set_title("Divert to the catch point\nred axis = body x (nose / thrust direction)")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=14, azim=-58)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def terminal_closeup(out: dict, target: np.ndarray,
                     altitude_window: float = 60.0,
                     path: str = "terminal_closeup.png"):
    """
    Plan view and side view of the final approach only.

    The full trajectory is dominated by a 1 km vertical drop, which squashes
    the 40 m divert into nothing. This crops to the part that decides whether
    the catch succeeds.
    """
    r = out["r"]
    mask = r[:, 2] < target[2] + altitude_window

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(r[mask, 0], r[mask, 1], color=C_PATH, lw=1.6)
    ax[0].scatter(*target[:2], color=C_TARGET, s=90, marker="X")
    ax[0].add_patch(plt.Circle(target[:2], 1.0, fill=False,
                               color=C_TARGET, ls="--", lw=1.0))
    ax[0].set_xlabel("downrange x (m)")
    ax[0].set_ylabel("crossrange y (m)")
    ax[0].set_title("Plan view (dashed circle = 1 m catch tolerance)")
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.25)

    ax[1].plot(r[mask, 0], r[mask, 2], color=C_PATH, lw=1.6)
    ax[1].scatter(target[0], target[2], color=C_TARGET, s=90, marker="X")
    ax[1].set_xlabel("downrange x (m)")
    ax[1].set_ylabel("altitude z (m)")
    ax[1].set_title("Side view")
    ax[1].grid(alpha=0.25)

    fig.suptitle("Terminal approach", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------

def animate(out: dict, target: np.ndarray | None = None,
            n_frames: int = 120, fps: int = 20, body_scale: float = 1.0,
            path: str = "flight.gif"):
    """
    Animated 3D view of the run, written to a GIF.

    Frames are subsampled evenly from the log rather than by time, so a run
    with a long hover does not spend most of its frames on a stationary
    vehicle.
    """
    r, qs = out["r"], out["q"]
    idx = np.linspace(0, len(r) - 1, n_frames).astype(int)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    pad = 60.0
    xlim = (min(r[:, 0].min(), 0) - pad, max(r[:, 0].max(), 0) + pad)
    ylim = (r[:, 1].min() - pad, r[:, 1].max() + pad)
    zlim = (0.0, r[:, 2].max() + pad)

    def draw(frame):
        ax.clear()
        i = idx[frame]
        ax.plot(r[:i + 1, 0], r[:i + 1, 1], r[:i + 1, 2],
                color=C_PATH, lw=1.2, alpha=0.8)

        seg = _booster_segments(r[i], qs[i], scale=body_scale)
        ax.plot(*zip(*seg), color=C_BODY, lw=5.0, solid_capstyle="round")

        L = 40.0
        for vec, c in ((np.array([1.0, 0, 0]), C_X),
                       (np.array([0, 1.0, 0]), C_Y),
                       (np.array([0, 0, 1.0]), C_Z)):
            d = Q.rotate(qs[i], vec) * L
            ax.plot(*zip(r[i], r[i] + d), color=c, lw=2.0)

        if target is not None:
            ax.scatter(*target, color=C_TARGET, s=80, marker="X",
                       depthshade=False)
            ax.plot([target[0] - 30, target[0] + 30], [target[1], target[1]],
                    [target[2], target[2]], color=C_TARGET, lw=2.0, alpha=0.6)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"t = {out['t'][i]:6.2f} s    alt {r[i, 2]:7.1f} m    "
                     f"tilt {out['tilt'][i]:5.2f} deg")
        ax.view_init(elev=16, azim=-60 + 25 * frame / n_frames)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path
