"""
Figures and animation for the full RTLS catch.

    mission_overview(out, target)   whole flight on one page
    landing_detail(out, target)     the landing burn, graded against the catch
    monte_carlo_report(results)     dispersion campaign summary
    animate_catch(out, target)      time-warped MP4 of the whole flight

Design notes
------------
* One dark theme everywhere, so stills and video read as one set. Series
  colours are the validated categorical dark steps, assigned to flight phases
  in flight order and ALWAYS direct-labelled, so identity never rests on colour
  alone. Status colours (good / critical) are reserved for pass / fail and
  are always paired with a marker shape or a word.
* One y-scale per panel. Quantities with different units get their own small
  multiple instead of a second axis.
* The animation draws the booster and the tower TO SCALE in the close-up. No
  "visual catch point" fudge: the drawn booster sits exactly where the
  simulation says it is, and the tower is placed so its chopstick arms close on
  the catch target.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import patches as mpatches          # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402

import numpy as np                                  # noqa: E402

from . import environment as ENV                    # noqa: E402
from . import quaternion as Q                       # noqa: E402

# ---------------------------------------------------------------------------
# theme

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"

PHASES = ["Flip", "Boostback", "Coast", "Landing burn (13)",
          "Final approach (3)"]
PHASE_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]

LIMITS = {                      # catch criteria (see phases.catch_report)
    "lateral_error_m": (1.0, "Lateral error", "m"),
    "vertical_error_m": (1.0, "Vertical error", "m"),
    "horizontal_speed_ms": (0.5, "Horizontal speed", "m/s"),
    "vertical_speed_ms": (1.0, "Vertical speed", "m/s"),
    "tilt_deg": (0.5, "Tilt", "deg"),
    "body_rate_deg_s": (1.0, "Body rate", "deg/s"),
    "fin_roll_deg": (10.0, "Fin roll", "deg"),
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": PAGE, "axes.facecolor": SURFACE,
        "savefig.facecolor": PAGE, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlecolor": INK, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "DejaVu Sans", "legend.frameon": False,
        "legend.fontsize": 8, "lines.linewidth": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# phase bookkeeping


def phase_index(out: dict) -> np.ndarray:
    """Per-sample flight-phase index into PHASES."""
    seg = np.asarray(out["segment"])
    n = np.asarray(out["n_lit"])
    thr = np.asarray(out["throttle"])
    idx = np.full(len(seg), 2, dtype=int)
    idx[np.isin(seg, ["ignition_5", "flip_33"])] = 0
    idx[np.char.startswith(seg.astype(str), "boostback")] = 1
    land = seg == "landing"
    idx[land & (n >= 13) & (thr > 0)] = 3
    idx[land & (n > 0) & (n < 13) & (thr > 0)] = 4
    # the final logged sample carries throttle 0; keep it in the last phase
    if land[-1] and idx[-2] == 4:
        idx[-1] = 4
    return idx


def phase_spans(out: dict):
    """[(phase, t0, t1), ...] contiguous runs."""
    idx = phase_index(out)
    t = np.asarray(out["t"])
    spans, k0 = [], 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or idx[k] != idx[k0]:
            spans.append((int(idx[k0]), float(t[k0]),
                          float(t[min(k, len(t) - 1)])))
            k0 = k
    return spans


def _phase_lines(ax, x, y, idx, lw=2.0, alpha=1.0, zorder=3):
    for p in range(len(PHASES)):
        m = idx == p
        if not m.any():
            continue
        # include one sample either side so the runs join up
        mm = m | np.roll(m, 1) & (np.arange(len(m)) > 0)
        yy = np.where(mm, y, np.nan)
        ax.plot(x, yy, color=PHASE_COLORS[p], lw=lw, alpha=alpha,
                zorder=zorder, solid_capstyle="round")


def _shade_phases(ax, spans, alpha=0.10):
    for p, t0, t1 in spans:
        ax.axvspan(t0, t1, color=PHASE_COLORS[p], alpha=alpha, lw=0,
                   zorder=0)


def _events(out: dict) -> dict:
    ev = {}
    for t, name in out["events"]:
        if name.startswith("begin boostback_33"):
            ev["boostback start"] = t
        if "predictive cutoff" in name:
            ev["33-engine cutoff"] = t
        if name == "begin boostback_3":
            ev.setdefault("boostback end", t + 3.0)
    le = out.get("landing_events", {})
    if le and np.isfinite(le.get("ignition_t", np.nan)):
        ev["landing ignition"] = le["ignition_t"]
    if le and np.isfinite(le.get("handover_t", np.nan)):
        ev["13 -> 3 engines"] = le["handover_t"]
    return ev


def _nearest(t_arr, t):
    return int(np.clip(np.searchsorted(t_arr, t), 0, len(t_arr) - 1))


# ---------------------------------------------------------------------------
# figure 1: mission overview


def mission_overview(out: dict, target: np.ndarray, report: dict | None = None,
                     path: str = "figs/mission_overview.png",
                     title: str = "Super Heavy V3 · return to launch site and tower catch"):
    _style()
    t = np.asarray(out["t"])
    r = np.asarray(out["r"])
    v = np.asarray(out["v"])
    idx = phase_index(out)
    spans = phase_spans(out)
    tgt = np.asarray(target, float)

    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(3, 4, left=0.085, right=0.985, top=0.885,
                          bottom=0.065, hspace=0.55, wspace=0.28,
                          width_ratios=[1.25, 1.25, 1, 1])
    fig.text(0.05, 0.955, title, fontsize=17, weight="bold", color=INK)
    caught = report["caught"] if report else None
    sub = (f"flight time {t[-1]:.1f} s  ·  apogee {r[:, 2].max() / 1e3:.1f} km  ·  "
           f"propellant at catch {out['prop'][-1] / 1e3:.1f} t")
    fig.text(0.05, 0.925, sub, fontsize=10.5, color=INK2)
    if caught is not None:
        fig.text(0.985, 0.95, ("✔ CAUGHT" if caught else "✖ NOT CAUGHT"),
                 ha="right", fontsize=17, weight="bold",
                 color=GOOD if caught else CRITICAL)

    # --- trajectory profile ------------------------------------------------
    ax = fig.add_subplot(gs[0:2, 0:2])
    x_km = (r[:, 0] - tgt[0]) / 1e3
    z_km = r[:, 2] / 1e3
    _phase_lines(ax, x_km, z_km, idx, lw=2.4)
    ax.set_xlabel("downrange from tower (km)")
    ax.set_ylabel("altitude (km)")
    ax.set_title("Trajectory — the RTLS hook")
    ax.set_ylim(-3, z_km.max() * 1.12)
    ax.set_xlim(min(x_km.min(), 0) - 6, x_km.max() + 6)
    ax.set_aspect("auto")
    # direct phase labels at each phase's midpoint
    placed = []
    for p in range(len(PHASES)):
        m = np.where(idx == p)[0]
        if len(m) == 0:
            continue
        k = m[len(m) // 2]
        if p >= 3:
            continue            # landing phases are invisible at this scale
        xy = (x_km[k], z_km[k])
        off = {0: (-60, 26), 1: (22, 12), 2: (-10, 22)}[p]
        ax.annotate(PHASES[p], xy, xytext=off, textcoords="offset points",
                    color=INK2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                    shrinkA=2, shrinkB=3))
        placed.append(p)
    ka = int(np.argmax(r[:, 2]))
    ax.plot(x_km[ka], z_km[ka], "o", ms=7, color=INK, mec=SURFACE, mew=2)
    ax.annotate(f"apogee {z_km[ka]:.1f} km", (x_km[ka], z_km[ka]),
                xytext=(8, 8), textcoords="offset points", color=INK,
                fontsize=9)
    ax.plot(x_km[0], z_km[0], "o", ms=7, color=PHASE_COLORS[0], mec=SURFACE,
            mew=2)
    ax.annotate("hot-stage separation", (x_km[0], z_km[0]), xytext=(8, -14),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.plot(0, tgt[2] / 1e3, marker="v", ms=10, color=INK, mec=SURFACE)
    ax.annotate("tower catch", (0, tgt[2] / 1e3), xytext=(10, 6),
                textcoords="offset points", color=INK, fontsize=9)
    # legend (always present for >=2 series)
    handles = [plt.Line2D([], [], color=c, lw=2.4) for c in PHASE_COLORS]
    ax.legend(handles, PHASES, loc="upper right", ncol=1)

    # --- scorecard ---------------------------------------------------------
    axs = fig.add_subplot(gs[2, 0:2])
    if report:
        _scorecard(axs, report)
    else:
        axs.axis("off")

    # --- small multiples ---------------------------------------------------
    speed = np.linalg.norm(v, axis=1)
    qdyn = np.array([ENV.dynamic_pressure(v[k], r[k, 2])
                     for k in range(len(t))]) / 1e3
    panels = [
        ("Altitude", r[:, 2] / 1e3, "km"),
        ("Speed", speed, "m/s"),
        ("Dynamic pressure", qdyn, "kPa"),
        ("Engines lit", np.asarray(out["n_lit"], float), "count"),
        ("Tilt from vertical", np.asarray(out["tilt"]), "deg"),
        ("Propellant", np.asarray(out["prop"]) / 1e3, "t"),
    ]
    for k, (name, y, unit) in enumerate(panels):
        a = fig.add_subplot(gs[k // 2, 2 + k % 2])
        _shade_phases(a, spans)
        if name == "Engines lit":
            a.step(t, y, where="post", color=INK2, lw=1.6)
            a.set_yticks([0, 3, 5, 13, 33])
        else:
            _phase_lines(a, t, y, idx, lw=1.8)
        a.set_title(f"{name} ({unit})")
        a.set_xlim(0, t[-1])
        if k >= 4:
            a.set_xlabel("mission elapsed time (s)")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _scorecard(ax, report: dict):
    """Catch criteria as 'fraction of limit used' bars."""
    names, frac, vals = [], [], []
    for key, (lim, label, unit) in LIMITS.items():
        if key not in report["checks"]:
            continue
        val = float(report["checks"][key][0])
        names.append(f"{label}")
        frac.append(val / lim)
        vals.append(f"{val:.3g} {unit}  / {lim:g}")
    y = np.arange(len(names))[::-1]
    cols = [GOOD if f <= 1.0 else CRITICAL for f in frac]
    ax.barh(y, np.minimum(frac, 1.5), height=0.55, color=cols,
            edgecolor=SURFACE, linewidth=2)
    ax.axvline(1.0, color=INK2, lw=1.2, ls="--")
    ax.text(1.0, len(names) - 0.35, " limit", color=INK2, fontsize=8,
            va="bottom")
    for yy, f, s in zip(y, frac, vals):
        mark = "✔" if f <= 1.0 else "✖"
        ax.text(min(f, 1.5) + 0.02, yy, f"{mark} {s}", va="center",
                color=INK2, fontsize=8.5)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.75)
    ax.set_xlabel("fraction of catch limit used")
    ax.set_title("Catch scorecard at the catch plane")
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# figure 2: landing burn detail


def _booster_outline_xz(r, q, length=72.3, radius=4.5, com_frac=0.40):
    """Side-view (x-z) polygon of the booster body plus its aft point and
    axis, from the full attitude quaternion. r is the CoM."""
    ax = Q.rotate(q, np.array([1.0, 0.0, 0.0]))
    a = np.array([ax[0], ax[2]])
    na = float(np.linalg.norm(a))
    a = a / na if na > 1e-9 else np.array([0.0, 1.0])
    # apparent length shortens when the axis leans out of the x-z plane
    L = length * na
    n = np.array([-a[1], a[0]])
    c = np.array([r[0], r[2]])
    aft = c - a * (com_frac * L)
    nose = c + a * ((1.0 - com_frac) * L)
    poly = np.array([aft + n * radius, nose + n * radius,
                     nose + n * radius * 0.2 + a * 1.0,
                     nose - n * radius * 0.2 + a * 1.0,
                     nose - n * radius, aft - n * radius])
    return poly, aft, nose, a, n


# The simulation's catch target is the booster's CENTRE OF MASS. The arms
# close under the hardpoints near the top of the booster, this far above it.
HARDPOINT_ABOVE_COM = (1.0 - 0.40) * 72.3 - 8.0


def tower_xz(target, height=146.0, width=12.0, arm_reach=15.0):
    """Tower structure placed so the chopstick arm tips close on the target.
    Returns (lattice segments, arm segments) in the x-z plane, metres. The
    arms are drawn at hardpoint height: target (CoM) + HARDPOINT_ABOVE_COM."""
    tx, tz = float(target[0]), float(target[2]) + HARDPOINT_ABOVE_COM
    face = tx - arm_reach                  # tower face, toward -x
    x0, x1 = face - width, face
    segs = [[(x0, 0), (x0, height)], [(x1, 0), (x1, height)],
            [(x0, height), (x1, height)]]
    for k in range(10):
        z0 = height * k / 10
        z1 = height * (k + 1) / 10
        segs.append([(x0, z0), (x1, z1)])
        segs.append([(x0, z1), (x1, z0)])
        segs.append([(x0, z1), (x1, z1)])
    arms = [[(x1, tz), (tx + 6.0, tz)], [(x1, tz + 2.5), (x1, tz - 2.5)]]
    return segs, arms


def landing_detail(out: dict, target: np.ndarray, report: dict | None = None,
                   path: str = "figs/landing_detail.png"):
    _style()
    t = np.asarray(out["t"])
    r = np.asarray(out["r"])
    v = np.asarray(out["v"])
    tgt = np.asarray(target, float)
    le = out.get("landing_events", {})
    t_ign = le.get("ignition_t", t[-1] - 40.0)
    t_hand = le.get("handover_t", np.nan)
    m = t >= t_ign - 2.0
    tt = t[m] - t[-1]                       # time to catch (negative)
    idx = phase_index(out)[m]
    rr, vv = r[m], v[m]
    G = out.get("guidance_log", [])
    gt = np.array([g["t"] for g in G]) - t[-1] if G else np.array([])

    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(2, 4, left=0.05, right=0.985, top=0.87,
                          bottom=0.07, hspace=0.38, wspace=0.30,
                          width_ratios=[1.15, 1, 1, 1])
    fig.text(0.05, 0.945, "Landing burn and catch", fontsize=17,
             weight="bold")
    sub = (f"ignition at {le.get('ignition_alt', np.nan):.0f} m, "
           f"{le.get('ignition_speed', np.nan):.0f} m/s  ·  13→3 engines at "
           f"{le.get('handover_alt', np.nan):.0f} m  ·  burn "
           f"{t[-1] - t_ign:.1f} s  ·  x-axis: seconds to catch")
    fig.text(0.05, 0.912, sub, fontsize=10.5, color=INK2)
    if report:
        ok = report["caught"]
        fig.text(0.985, 0.94, "✔ CAUGHT" if ok else "✖ NOT CAUGHT",
                 ha="right", fontsize=17, weight="bold",
                 color=GOOD if ok else CRITICAL)

    # --- side view with to-scale silhouettes -------------------------------
    ax = fig.add_subplot(gs[:, 0])
    _phase_lines(ax, rr[:, 0] - tgt[0], rr[:, 2], idx, lw=1.6, alpha=0.9)
    segs, arms = tower_xz(tgt - np.array([tgt[0], 0, 0]))
    ax.add_collection(LineCollection(segs, colors=MUTED, lw=0.8))
    ax.add_collection(LineCollection(arms, colors=INK, lw=2.4))
    q = np.asarray(out["q"])[m]
    times = np.arange(np.ceil(tt[0]), 0.01, 2.0)
    for tk in list(times) + [tt[-1]]:
        k = int(np.argmin(abs(tt - tk)))
        if rr[k, 2] > 560.0:
            continue
        pos = rr[k] - np.array([tgt[0], 0, 0])
        poly, *_ = _booster_outline_xz(pos, q[k])
        ax.add_patch(mpatches.Polygon(poly, closed=True, fc="none",
                                      ec=PHASE_COLORS[idx[k]], lw=0.9,
                                      alpha=0.85))
    ax.set_aspect("equal")
    zmax = min(rr[:, 2].max(), 520.0)
    xs = rr[rr[:, 2] < zmax, 0] - tgt[0]
    ax.set_xlim(min(xs.min(), -40) - 40, max(xs.max(), 40) + 40)
    ax.set_ylim(0, zmax + 60)
    ax.set_xlabel("downrange from catch point (m)")
    ax.set_ylabel("altitude (m)")
    ax.set_title("Side view below 500 m, booster to scale every 2 s")
    ax.fill_between([-1e4, 1e4], -50, 0, color="#24211c", zorder=0)

    def vline(a):
        if np.isfinite(t_hand):
            a.axvline(t_hand - t[-1], color=MUTED, lw=0.9, ls=":")

    # --- top view, final approach -------------------------------------------
    at = fig.add_subplot(gs[0, 1])
    low = rr[:, 2] < tgt[2] + 150.0
    ex, ey = rr[low, 0] - tgt[0], rr[low, 1] - tgt[1]
    pts = np.column_stack([ex, ey])
    if len(pts) > 1:
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap="viridis_r", lw=2.2)
        lc.set_array(rr[low, 2][:-1] - tgt[2])
        at.add_collection(lc)
        cb = fig.colorbar(lc, ax=at, pad=0.02, fraction=0.05)
        cb.set_label("height above catch (m)", color=INK2, fontsize=8)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
    at.add_patch(mpatches.Circle((0, 0), 1.0, fc="none", ec=INK2, lw=1.2,
                                 ls="--"))
    at.text(0.75, -1.45, "1 m limit", color=INK2, fontsize=8)
    at.plot(ex[-1:], ey[-1:], "o", ms=8, color=INK, mec=SURFACE, mew=2)
    lim = 5.0
    at.set_xlim(-lim, lim); at.set_ylim(-lim, lim)
    at.set_aspect("equal")
    at.set_title("Top view of the last 150 m, zoomed to ±5 m")
    at.set_xlabel("x (m)"); at.set_ylabel("y (m)")

    # --- lateral miss ------------------------------------------------------
    al = fig.add_subplot(gs[1, 1])
    d = np.linalg.norm(rr[:, :2] - tgt[:2], axis=1)
    _phase_lines(al, tt, np.maximum(d, 1e-3), idx)
    al.set_yscale("log")
    al.axhline(1.0, color=INK2, ls="--", lw=1.0)
    al.text(tt[0], 1.15, "limit 1 m", color=INK2, fontsize=8)
    vline(al)
    al.set_title("Horizontal distance to catch point (m)")
    al.set_xlabel("time to catch (s)")

    # --- vertical speed ----------------------------------------------------
    a1 = fig.add_subplot(gs[0, 2])
    _phase_lines(a1, tt, -vv[:, 2], idx)
    a1.set_yscale("symlog", linthresh=10)
    a1.axhline(1.0, color=INK2, ls="--", lw=1.0)
    a1.text(tt[0], 1.15, "catch limit 1 m/s", color=INK2, fontsize=8)
    vline(a1)
    a1.set_title("Descent rate (m/s)")

    # --- horizontal speed --------------------------------------------------
    a2 = fig.add_subplot(gs[0, 3])
    hs = np.linalg.norm(vv[:, :2], axis=1)
    _phase_lines(a2, tt, hs, idx)
    a2.set_yscale("symlog", linthresh=1)
    a2.axhline(0.5, color=INK2, ls="--", lw=1.0)
    a2.text(tt[0], 0.56, "limit 0.5 m/s", color=INK2, fontsize=8)
    vline(a2)
    a2.set_title("Horizontal speed (m/s)")

    # --- tilt vs envelope --------------------------------------------------
    a3 = fig.add_subplot(gs[1, 2])
    if len(gt):
        a3.plot(gt, [g["tilt_limit"] for g in G], color=MUTED, lw=1.2,
                ls="-", label="guidance tilt envelope")
    _phase_lines(a3, tt, np.asarray(out["tilt"])[m], idx)
    a3.axhline(0.5, color=INK2, ls="--", lw=1.0)
    a3.set_yscale("symlog", linthresh=1)
    a3.set_ylim(0, 60)
    vline(a3)
    a3.legend(loc="upper right")
    a3.set_title("Tilt from vertical (deg) — limit 0.5")
    a3.set_xlabel("time to catch (s)")

    # --- throttle ----------------------------------------------------------
    a4 = fig.add_subplot(gs[1, 3])
    thr = np.asarray(out["throttle"])[m]
    _phase_lines(a4, tt, thr * 100, idx)
    a4.set_ylim(0, 105)
    vline(a4)
    for p, lab in ((3, "13 engines"), (4, "3 engines")):
        k = np.where(idx == p)[0]
        if len(k):
            a4.text(tt[k[len(k) // 2]], 96, lab, color=INK2, fontsize=8,
                    ha="center")
    a4.set_title("Throttle (%)")
    a4.set_xlabel("time to catch (s)")

    for a in (al, a1, a2, a3, a4):
        a.set_xlim(tt[0], 0.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# figure 3: Monte Carlo


def _wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / den


def monte_carlo_report(results: list[dict], path: str = "figs/monte_carlo.png",
                       spec: dict | None = None):
    _style()
    rows = [r["row"] for r in results]
    ok = np.array([bool(r.get("caught")) for r in rows])
    n, k = len(rows), int(ok.sum())
    lb = _wilson_lower(k, n)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.86,
                          bottom=0.07, hspace=0.42, wspace=0.28)
    fig.text(0.055, 0.945, "Monte Carlo dispersion campaign", fontsize=17,
             weight="bold")
    fig.text(0.055, 0.905,
             "dispersed separation state, propellant, dry mass, thrust, Isp, "
             "drag, density, winds and body rates — guidance flies on the "
             "nominal models", fontsize=10.5, color=INK2)
    fig.text(0.985, 0.94, f"{k}/{n} caught", ha="right", fontsize=22,
             weight="bold", color=GOOD if k == n else WARNING)
    fig.text(0.985, 0.905, f"{100 * k / n:.1f}%  ·  95% lower bound "
             f"{100 * lb:.1f}% (Wilson)", ha="right", fontsize=10.5,
             color=INK2)

    def arr(key):
        return np.array([float(r.get(key, np.nan)) for r in rows])

    # --- (a) catch footprint ------------------------------------------------
    a = fig.add_subplot(gs[0, 0])
    fx, fy = [], []
    for res in results:
        tr = res.get("traj")
        if tr is None or len(tr["low_r"]) == 0:
            fx.append(np.nan); fy.append(np.nan)
            continue
        fx.append(tr["low_r"][-1, 0]); fy.append(tr["low_r"][-1, 1])
    fx, fy = np.array(fx), np.array(fy)
    a.add_patch(mpatches.Circle((0, 0), 1.0, fc="none", ec=INK2, ls="--",
                                lw=1.2))
    a.text(0.72, 0.75, "1 m limit", color=INK2, fontsize=8)
    a.scatter(fx[ok], fy[ok], s=36, color=GOOD, edgecolor=SURFACE, lw=1.2,
              marker="o", label=f"caught ({k})", zorder=3)
    if (~ok).any():
        a.scatter(fx[~ok], fy[~ok], s=60, color=CRITICAL, marker="X",
                  edgecolor=SURFACE, lw=1.0, label=f"missed ({n - k})",
                  zorder=4)
    lim = 1.4
    far = np.hypot(fx, fy)
    if np.nanmax(far) > lim:
        lim = min(float(np.nanmax(far)) * 1.1, 50.0)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    a.set_title("Catch footprint at the catch plane")
    a.set_xlabel("downrange error (m)"); a.set_ylabel("crossrange error (m)")
    a.legend(loc="lower left")

    # --- (b) what the landing burn had to fix ---------------------------------
    b = fig.add_subplot(gs[0, 1])
    ix, iy = arr("ignition_x"), arr("ignition_y")
    b.scatter(ix[ok], iy[ok], s=30, color="#3987e5", edgecolor=SURFACE,
              lw=1.0, label="caught")
    if (~ok).any():
        b.scatter(ix[~ok], iy[~ok], s=55, color=CRITICAL, marker="X",
                  label="missed")
    b.plot(0, 0, marker="v", ms=10, color=INK, mec=SURFACE)
    b.set_title("Position at landing-burn ignition")
    b.set_xlabel("downrange from tower (m)"); b.set_ylabel("crossrange (m)")
    b.legend(loc="lower left")

    # --- (c) margin strip ----------------------------------------------------
    c = fig.add_subplot(gs[0, 2])
    keys = list(LIMITS.keys())
    rng = np.random.default_rng(0)
    for j, key in enumerate(keys):
        lim_, label, unit = LIMITS[key]
        f = arr(key) / lim_
        y = len(keys) - 1 - j + rng.uniform(-0.18, 0.18, size=len(f))
        good = f <= 1.0
        c.scatter(np.minimum(f[good], 2.0), y[good], s=14, color=GOOD,
                  alpha=0.8, lw=0)
        c.scatter(np.minimum(f[~good], 2.0), y[~good], s=30, color=CRITICAL,
                  marker="X", lw=0)
        worst = np.nanmax(f) if np.isfinite(f).any() else np.nan
        c.text(2.05, len(keys) - 1 - j, f"worst {worst:.2f}", va="center",
               fontsize=8, color=INK2)
    c.axvline(1.0, color=INK2, ls="--", lw=1.0)
    c.set_yticks(range(len(keys))[::-1], [LIMITS[k_][1] for k_ in keys])
    c.set_xlim(0, 2.45)
    c.set_xlabel("value / catch limit (≤ 1 passes)")
    c.set_title("Margin on every catch criterion")
    c.grid(axis="y", visible=False)

    # --- (d) landing trajectories -------------------------------------------
    d = fig.add_subplot(gs[1, 0])
    for res, good in zip(results, ok):
        tr = res.get("traj")
        if tr is None or len(tr["low_r"]) == 0:
            continue
        rr = tr["low_r"]
        d.plot(rr[:, 0], rr[:, 2], color=GOOD if good else CRITICAL,
               lw=0.8 if good else 1.4, alpha=0.45 if good else 0.9)
    d.plot(0, 105, marker="v", ms=10, color=INK, mec=SURFACE)
    d.set_title("Landing trajectories below 1.5 km")
    d.set_xlabel("downrange from tower (m)"); d.set_ylabel("altitude (m)")

    # --- (e) propellant ------------------------------------------------------
    e = fig.add_subplot(gs[1, 1])
    p = arr("prop_left_t")
    bins = np.linspace(np.nanmin(p) - 1, np.nanmax(p) + 1, 22)
    e.hist(p[ok], bins=bins, color="#3987e5", edgecolor=SURFACE, lw=2,
           label="caught")
    if (~ok).any():
        e.hist(p[~ok], bins=bins, color=CRITICAL, edgecolor=SURFACE, lw=2,
               label="missed")
    if np.nanmin(p) < 5.0:
        e.axvline(0, color=CRITICAL, lw=1)
        e.text(0.3, 0.5, "tanks dry", color=CRITICAL, fontsize=8,
               transform=e.get_xaxis_transform())
    e.set_title("Propellant remaining at catch (t)")
    e.set_xlabel("t"); e.set_ylabel("cases")
    e.text(0.02, 0.92, f"min {np.nanmin(p):.1f} t · median "
           f"{np.nanmedian(p):.1f} t", transform=e.transAxes, color=INK2,
           fontsize=9)

    # --- (f) sensitivity -----------------------------------------------------
    f_ = fig.add_subplot(gs[1, 2])
    inputs = {
        "separation altitude": "d_alt", "separation downrange": "d_downrange",
        "separation crossrange": "d_crossrange", "separation speed": "d_speed",
        "flight-path angle": "d_fpa", "heading": "d_heading",
        "propellant load": "d_prop", "dry mass": "k_dry",
        "engine thrust": "k_thrust", "Isp": "k_isp", "drag coeff.": "k_cd",
        "air density": "k_rho", "surface wind": "wind_surface",
        "jet-stream wind": "wind_jet",
    }
    # worst margin across criteria (max value/limit) per case
    W = np.nanmax(np.column_stack([arr(k_) / LIMITS[k_][0] for k_ in keys]),
                  axis=1)
    corr = {}
    for lab, key in inputs.items():
        x = arr(key)
        if np.nanstd(x) == 0:
            continue
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(W))
        corr[lab] = float(np.corrcoef(rx, ry)[0, 1])
    corr["propellant (at catch)"] = float(np.corrcoef(
        np.argsort(np.argsort(p)), np.argsort(np.argsort(W)))[0, 1])
    items = sorted(corr.items(), key=lambda kv: abs(kv[1]))[-10:]
    yy = np.arange(len(items))
    f_.barh(yy, [v_ for _, v_ in items], height=0.6,
            color=["#e66767" if v_ > 0 else "#3987e5" for _, v_ in items],
            edgecolor=SURFACE, lw=2)
    f_.set_yticks(yy, [k_ for k_, _ in items])
    f_.axvline(0, color=AXIS, lw=1)
    f_.set_xlim(-1, 1)
    f_.set_title("What drives the worst margin (Spearman ρ)")
    f_.set_xlabel("← improves margin     ρ     erodes margin →")
    f_.grid(axis="y", visible=False)

    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# animation


def _ffmpeg():
    """Point matplotlib at an ffmpeg binary: PATH first, then imageio-ffmpeg."""
    import shutil
    from matplotlib import animation
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return animation.FFMpegWriter.isAvailable()
    except Exception:
        return False


def _playback_speed(ph: int, alt: float) -> float:
    """Sim seconds per video second, by phase. Slow where things happen."""
    if ph in (0, 1):
        return 2.5
    if ph == 2:
        return 14.0 if alt > 6000.0 else 3.0
    if ph == 3:
        return 0.8
    return 1.6


def _frame_times(out: dict, fps: int):
    t = np.asarray(out["t"])
    ph = phase_index(out)
    alt = np.asarray(out["r"])[:, 2]
    rate = np.array([_playback_speed(int(p), float(a))
                     for p, a in zip(ph, alt)])
    # smooth the playback rate so warp changes are eased, not stepped
    w = max(int(len(t) * 0.01), 1)
    kern = np.ones(2 * w + 1) / (2 * w + 1)
    rate = np.convolve(np.pad(rate, w, mode="edge"), kern, mode="valid")
    dv = np.diff(t) / rate[:-1]
    vt = np.concatenate([[0.0], np.cumsum(dv)])
    frames = np.arange(0.0, vt[-1], 1.0 / fps)
    return np.interp(frames, vt, t), np.interp(frames, vt, rate)


def _interp(out, tq):
    t = np.asarray(out["t"])
    j = int(np.clip(np.searchsorted(t, tq), 1, len(t) - 1))
    f = float(np.clip((tq - t[j - 1]) / max(t[j] - t[j - 1], 1e-9), 0, 1))
    r = (1 - f) * out["r"][j - 1] + f * out["r"][j]
    v = (1 - f) * out["v"][j - 1] + f * out["v"][j]
    q0, q1 = out["q"][j - 1], out["q"][j]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = Q.normalize((1 - f) * q0 + f * q1)
    k = j if f > 0.5 else j - 1
    return k, r, v, q


def _engine_layout():
    """Raptor positions: centre 3, middle ring 10, outer ring 20. The first
    3 / 13 / 33 entries are the engines lit on 3 / 13 / 33-engine burns."""
    inner3 = [(0.55 * np.cos(a), 0.55 * np.sin(a))
              for a in np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 2]
    mid10 = [(1.25 * np.cos(a), 1.25 * np.sin(a))
             for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
    outer20 = [(2.1 * np.cos(a), 2.1 * np.sin(a))
               for a in np.linspace(0, 2 * np.pi, 20, endpoint=False)]
    return np.array(inner3 + mid10 + outer20)


def animate_catch(out: dict, target: np.ndarray, path: str = "figs/catch.mp4",
                  report: dict | None = None, fps: int = 30,
                  hold_seconds: float = 4.0, dpi: int = 100,
                  max_frames: int | None = None, progress=None,
                  snapshots: list | None = None):
    """
    Time-warped animation of the whole flight.

    Left: side view whose camera follows the booster and zooms continuously
    from the full 100 km arc to a to-scale close-up of the tower. Right:
    live telemetry. Playback rate varies by phase (shown on screen) so the
    3.5 minute coast does not drown the 40 second landing.
    """
    from matplotlib import animation

    _style()
    tgt = np.asarray(target, float)
    t_all = np.asarray(out["t"])
    r_all = np.asarray(out["r"])
    v_all = np.asarray(out["v"])
    ph_all = phase_index(out)
    frames_t, frames_rate = _frame_times(out, fps)
    n_hold = int(hold_seconds * fps)
    if max_frames:
        frames_t = frames_t[:max_frames]
    n_frames = len(frames_t) + n_hold

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.patch.set_facecolor(PAGE)
    ax = fig.add_axes([0.0, 0.0, 0.64, 1.0])
    ax.set_facecolor("#08111f")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # ground and sea
    ax.fill_between([-3e5, tgt[0] + 400], -3e4, 0, color="#221f1a", zorder=1)
    ax.fill_between([tgt[0] + 400, 3e5], -3e4, 0, color="#0b2236", zorder=1)
    ax.plot([-3e5, 3e5], [0, 0], color="#3a352c", lw=1, zorder=2)

    # tower, to scale
    segs, arms = tower_xz(tgt)
    tower_lc = LineCollection(segs, colors="#9a968c", lw=0.8, zorder=4)
    arm_lc = LineCollection(arms, colors="#e8e6df", lw=2.0, zorder=6)
    ax.add_collection(tower_lc)
    ax.add_collection(arm_lc)

    # full planned path, faint, and the flown trail by phase
    ax.plot(r_all[:, 0], r_all[:, 2], color="#ffffff", alpha=0.12, lw=1.0,
            zorder=2, ls=(0, (4, 4)))
    trail = [ax.plot([], [], color=c, lw=2.0, zorder=3,
                     solid_capstyle="round")[0] for c in PHASE_COLORS]

    body = mpatches.Polygon(np.zeros((6, 2)), closed=True, fc="#d9d6cf",
                            ec="#ffffff", lw=0.6, zorder=8)
    plume_outer = mpatches.Polygon(np.zeros((3, 2)), closed=True,
                                   fc="#ff8a3d", ec="none", alpha=0.55,
                                   zorder=7)
    plume_inner = mpatches.Polygon(np.zeros((3, 2)), closed=True,
                                   fc="#fff1b8", ec="none", alpha=0.9,
                                   zorder=7)
    fins = [mpatches.Polygon(np.zeros((4, 2)), closed=True, fc="#8f8b82",
                             ec="none", zorder=9) for _ in range(2)]
    for pch in [plume_outer, plume_inner, body] + fins:
        ax.add_patch(pch)
    scale_txt = ax.text(0.02, 0.03, "", transform=ax.transAxes, color=INK2,
                        fontsize=11, family="monospace")
    warp_txt = ax.text(0.02, 0.965, "", transform=ax.transAxes, color=INK,
                       fontsize=13, family="monospace", va="top")
    phase_txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=20,
                        weight="bold", va="top")
    banner = ax.text(0.5, 0.86, "", transform=ax.transAxes, ha="center",
                     va="top", fontsize=34, weight="bold", color=GOOD,
                     alpha=0.0, zorder=20,
                     bbox=dict(boxstyle="round,pad=0.5", fc=PAGE, ec="none",
                               alpha=0.0))
    scale_bar, = ax.plot([], [], color=INK2, lw=2, transform=ax.transAxes)

    # top-view inset for the final approach
    axi = fig.add_axes([0.435, 0.06, 0.19, 0.30])
    axi.set_facecolor("#0d1522")
    axi.set_xlim(-6, 6); axi.set_ylim(-6, 6); axi.set_aspect("equal")
    axi.add_patch(mpatches.Circle((0, 0), 1.0, fc="none", ec=INK2, ls="--"))
    axi.set_title("top view, ±8 m around the catch point", fontsize=9,
                  color=INK2)
    axi.text(0.0, -1.9, "1 m", ha="center", color=INK2, fontsize=8)
    axi.tick_params(labelsize=7)
    inset_trail, = axi.plot([], [], color=PHASE_COLORS[4], lw=1.6)
    inset_dot, = axi.plot([], [], "o", color=INK, ms=6)
    axi.set_visible(False)

    # --- right panel ---------------------------------------------------------
    fig.text(0.665, 0.955, "SUPER HEAVY V3  ·  RTLS + TOWER CATCH",
             color=INK, fontsize=14, weight="bold")
    readout = fig.text(0.665, 0.915, "", color=INK, fontsize=12.5,
                       family="monospace", va="top", linespacing=1.55)
    ax_eng = fig.add_axes([0.865, 0.64, 0.12, 0.25])
    ax_eng.set_aspect("equal"); ax_eng.axis("off")
    ax_eng.set_xlim(-2.6, 2.6); ax_eng.set_ylim(-2.9, 2.6)
    ax_eng.add_patch(mpatches.Circle((0, 0), 2.5, fc="none", ec=AXIS))
    lay = _engine_layout()
    eng = ax_eng.scatter(lay[:, 0], lay[:, 1], s=90, c=[SURFACE] * 33,
                         edgecolors=MUTED, linewidths=0.8)
    eng_txt = ax_eng.text(0, -2.85, "", ha="center", color=INK2, fontsize=10,
                          family="monospace")

    strips = []
    spec = [("altitude (km)", r_all[:, 2] / 1e3, None),
            ("speed (m/s)", np.linalg.norm(v_all, axis=1), None),
            ("throttle (%)", np.asarray(out["throttle"]) * 100, (0, 105))]
    for k, (lab, y, yl) in enumerate(spec):
        a = fig.add_axes([0.69, 0.40 - k * 0.155, 0.285, 0.12])
        _phase_lines(a, t_all, y, ph_all, lw=1.2, alpha=0.25)
        live = [a.plot([], [], color=c, lw=1.8)[0] for c in PHASE_COLORS]
        cur = a.axvline(0, color=INK, lw=0.8)
        a.set_xlim(0, t_all[-1])
        if yl:
            a.set_ylim(*yl)
        a.set_title(lab, fontsize=9.5)
        a.tick_params(labelsize=7.5)
        if k == 2:
            a.set_xlabel("mission elapsed time (s)", fontsize=8.5)
        strips.append((a, y, live, cur))
    lg = fig.add_axes([0.665, 0.535, 0.33, 0.05]); lg.axis("off")
    for j, (nm, c) in enumerate(zip(PHASES, PHASE_COLORS)):
        x0, y0 = (j % 3) * 0.34, 0.75 - (j // 3) * 0.5
        lg.plot([x0, x0 + 0.04], [y0, y0], color=c, lw=3)
        lg.text(x0 + 0.05, y0, nm, va="center", fontsize=8.5, color=INK2)
    lg.set_xlim(0, 1); lg.set_ylim(0, 1)

    cam = {"cx": None, "cz": None, "hs": None}

    def camera(r, alt_agl, dtv):
        # Half-height of the view: generous far out, true scale near the
        # tower. Sized on height above GROUND so the whole booster (nose
        # ~43 m above the CoM) always fits once the ground is pinned in view.
        h = max(float(r[2]), 0.0)
        hs_t = float(np.clip(0.62 * h + 75.0, 110.0, 70_000.0))
        # Below ~3 km, slide the framing onto the tower: horizontally between
        # booster and tower, vertically with the ground near the bottom edge.
        blend = float(np.clip(1.0 - (h - 400.0) / 2600.0, 0.0, 1.0))
        blend = blend * blend * (3.0 - 2.0 * blend)
        cx_t = (1 - blend) * r[0] + blend * (0.5 * (r[0] + tgt[0]) - 12.0)
        cz_t = (1 - blend) * r[2] + blend * (0.88 * hs_t)
        if cam["hs"] is None:
            cam.update(cx=cx_t, cz=cz_t, hs=hs_t)
        a = 1.0 - np.exp(-dtv / 0.6)
        cam["hs"] = float(np.exp((1 - a) * np.log(cam["hs"])
                                 + a * np.log(hs_t)))
        cam["cx"] += a * (cx_t - cam["cx"])
        cam["cz"] += a * (cz_t - cam["cz"])
        return cam["cx"], cam["cz"], cam["hs"]

    aspect = (0.64 * 16) / 9.0
    lay_n = len(lay)
    order_lit = {0: [], 3: list(range(3)), 5: list(range(5)),
                 13: list(range(13)), 33: list(range(33))}

    def draw(fi):
        hold = fi >= len(frames_t)
        tq = frames_t[-1] if hold else frames_t[fi]
        rate = 0.0 if hold else frames_rate[fi]
        k, r, v, q = _interp(out, tq)
        if hold:
            k = len(t_all) - 1
        ph = int(ph_all[k])
        alt_agl = max(r[2] - tgt[2], 0.0)
        cx, cz, hs = camera(r, alt_agl, 1.0 / fps)
        ax.set_xlim(cx - hs * aspect, cx + hs * aspect)
        ax.set_ylim(cz - hs, cz + hs)

        # sky darkens with camera altitude
        s = float(np.clip(cz / 60_000.0, 0, 1))
        c0 = np.array([0x13, 0x2a, 0x4a]) / 255.0
        c1 = np.array([0x03, 0x05, 0x0a]) / 255.0
        ax.set_facecolor(tuple((1 - s) * c0 + s * c1))

        for p, ln in enumerate(trail):
            m = ph_all[:k + 1] == p
            if m.any():
                mm = m | np.concatenate([[False], m[:-1]])
                ln.set_data(np.where(mm, r_all[:k + 1, 0], np.nan),
                            np.where(mm, r_all[:k + 1, 2], np.nan))

        # booster: true scale when the view is tight, exaggerated when far
        exag = max(1.0, (hs * 0.16) / 72.3)
        L, R = 72.3 * exag, 4.5 * exag
        poly, aft, nose, a_, n_ = _booster_outline_xz(r, q, length=72.3 * exag,
                                                      radius=R)
        body.set_xy(poly)
        fin_c = nose - a_ * (0.07 * L)
        for j, sg in enumerate((+1, -1)):
            base = fin_c + sg * n_ * R
            fins[j].set_xy([base, base + sg * n_ * 0.55 * R,
                            base + sg * n_ * 0.55 * R - a_ * 0.10 * L,
                            base - a_ * 0.10 * L])
        thr = 0.0 if hold else float(out["throttle"][k])
        nl = 0 if hold else int(out["n_lit"][k])
        if thr > 0 and nl > 0:
            flick = 1.0 + 0.08 * np.sin(fi * 2.3) + 0.05 * np.sin(fi * 5.1)
            pl = L * (0.35 + 0.55 * thr) * (nl / 33.0) ** 0.35 * flick
            wdt = R * (0.55 + 0.45 * (nl / 33.0) ** 0.5)
            plume_outer.set_xy([aft + n_ * wdt, aft - n_ * wdt,
                                aft - a_ * pl])
            plume_inner.set_xy([aft + n_ * wdt * 0.5, aft - n_ * wdt * 0.5,
                                aft - a_ * pl * 0.55])
            plume_outer.set_visible(True); plume_inner.set_visible(True)
        else:
            plume_outer.set_visible(False); plume_inner.set_visible(False)

        # scale bar: a round number near 20% of the view width
        width = 2 * hs * aspect
        nice = 10 ** np.floor(np.log10(width * 0.2))
        for mlt in (5, 2, 1):
            if mlt * nice <= width * 0.2:
                nice *= mlt
                break
        frac = nice / width
        scale_bar.set_data([0.02, 0.02 + frac], [0.07, 0.07])
        lab = f"{nice / 1000:g} km" if nice >= 1000 else f"{nice:g} m"
        scale_txt.set_text(lab + ("   booster drawn to scale" if exag == 1.0
                                  else f"   booster ×{exag:.0f} for visibility"))
        warp_txt.set_text("■ HOLD" if hold else f"▶ {rate:4.1f}× real time")
        if hold:
            ok_ = report is None or report["caught"]
            phase_txt.set_text("IN THE CHOPSTICKS" if ok_ else "MISSED")
            phase_txt.set_color(GOOD if ok_ else CRITICAL)
        else:
            phase_txt.set_text(PHASES[ph].upper())
            phase_txt.set_color(PHASE_COLORS[ph])

        # final-approach inset
        if alt_agl < 350.0 or hold:
            axi.set_visible(True)
            mlow = (r_all[:k + 1, 2] - tgt[2]) < 350.0
            ex = r_all[:k + 1][mlow, 0] - tgt[0]
            ey = r_all[:k + 1][mlow, 1] - tgt[1]
            inset_trail.set_data(ex, ey)
            inset_dot.set_data([r[0] - tgt[0]], [r[1] - tgt[1]])
            axi.set_xlim(-8, 8); axi.set_ylim(-8, 8)
        else:
            axi.set_visible(False)

        spd = float(np.linalg.norm(v))
        hsp = float(np.hypot(v[0], v[1]))
        dist = float(np.hypot(r[0] - tgt[0], r[1] - tgt[1]))
        readout.set_text(
            f"T+ {tq:7.1f} s\n"
            f"ALTITUDE   {r[2] / 1000:9.3f} km\n"
            f"SPEED      {spd:9.1f} m/s\n"
            f"  vertical {v[2]:+9.1f} m/s\n"
            f"  horiz.   {hsp:9.2f} m/s\n"
            f"TO TOWER   {dist:9.1f} m\n"
            f"TILT       {out['tilt'][k]:9.2f} °\n"
            f"THROTTLE   {100 * thr:9.0f} %\n"
            f"PROPELLANT {out['prop'][k] / 1000:9.1f} t")
        lit = order_lit.get(nl, list(range(nl)))
        cols = [("#ff9d4d" if j in lit else SURFACE) for j in range(lay_n)]
        eng.set_facecolors(cols)
        eng_txt.set_text(f"{nl:2d} / 33 lit")

        for a, y, live, cur in strips:
            cur.set_xdata([tq, tq])
            for p, ln in enumerate(live):
                m = ph_all[:k + 1] == p
                if m.any():
                    mm = m | np.concatenate([[False], m[:-1]])
                    ln.set_data(np.where(mm, t_all[:k + 1], np.nan),
                                np.where(mm, y[:k + 1], np.nan))

        if hold and report is not None:
            ok = report["checks"]
            a_in = min((fi - len(frames_t) + 1) / (0.6 * fps), 1.0)
            banner.set_alpha(a_in)
            banner.get_bbox_patch().set_alpha(0.75 * a_in)
            banner.set_color(GOOD if report["caught"] else CRITICAL)
            lat = ok["lateral_error_m"][0]
            banner.set_text(("✔ CATCH" if report["caught"] else "✖ MISS")
                            + f"\n{lat:.2f} m · {ok['horizontal_speed_ms'][0]:.2f} m/s"
                            f" · {ok['tilt_deg'][0]:.2f}°")
            banner.set_fontsize(28)
        return []

    if snapshots:
        # Step every frame (the camera is a smoothed state) but only render
        # the requested mission times, as PNGs. For checking the look quickly.
        want = sorted(snapshots)
        paths, j = [], 0
        base = os.path.splitext(path)[0]
        for fi in range(n_frames):
            draw(fi)
            tq = frames_t[min(fi, len(frames_t) - 1)]
            while j < len(want) and (tq >= want[j] or fi == n_frames - 1):
                pth = f"{base}_t{want[j]:05.0f}.png"
                fig.savefig(pth, dpi=dpi, facecolor=PAGE)
                paths.append(pth)
                j += 1
        plt.close(fig)
        return paths

    have_ff = path.endswith(".mp4") and _ffmpeg()
    if path.endswith(".mp4") and not have_ff:
        path = path[:-4] + ".gif"
    writer = (animation.FFMpegWriter(fps=fps, bitrate=6000,
                                     extra_args=["-pix_fmt", "yuv420p"])
              if have_ff else animation.PillowWriter(fps=min(fps, 20)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with writer.saving(fig, path, dpi=dpi):
        for fi in range(n_frames):
            draw(fi)
            writer.grab_frame(facecolor=PAGE)
            if progress and fi % 150 == 0:
                progress(f"  frame {fi}/{n_frames}")
    plt.close(fig)
    return path
