"""
Real-time mission animation with live telemetry.

TWO ANIMATIONS, DIFFERENT JOBS
------------------------------
    animate_realtime   1.0x wall clock, every second of flight takes a second.
                       Written as MP4, because 250 s at 24 fps is 6000 frames
                       and a GIF of that is hundreds of megabytes.

    animate_summary    arc-length sampled, compressed to ~10 s. Good for a
                       portfolio page where nobody watches four minutes.

THE ZOOM PROBLEM
----------------
A fixed view cannot show both a 100 km ballistic arc and a 40 m divert. Held at
the wide scale, the entire terminal phase happens inside one pixel; held at the
close scale, the booster leaves frame in the first second. So the view tracks
the vehicle and the axis span follows its altitude, smoothly, on a log
interpolation between 120 km and 200 m. The camera is doing what a range
tracking telescope does.

TELEMETRY
---------
Engine ring, altitude and speed are read from the flight log every frame. The
engine ring is the honest check that the burns are happening: 5 lit for the
flip, 33 for boostback, 0 through the coast, 13 for the landing burn, 2 for the
hover. If the ring does not match the phase label, something is wrong with the
sequencer, not with the picture.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

from . import quaternion as Q
from .telemetry import BG, FG, GRID, C_ACHIEVED, C_HOT, C_COLD, C_TARGET
from .telemetry import (engine_positions, lit_engine_mask, tower_geometry,
                        visual_catch_point, _smooth_visual_position)


def ensure_ffmpeg(explicit_path: str | None = None) -> str | None:
    """
    Make matplotlib able to find ffmpeg, and return the path used.

    WHY THIS IS NEEDED ON WINDOWS
    -----------------------------
    `winget install Gyan.FFmpeg` puts the real binary deep under

        %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_*\\ffmpeg-*-full_build\\bin\\

    and only exposes it through a shim in ...\\WinGet\\Links. That shim
    directory is frequently missing from the PATH a child process inherits --
    so `shutil.which("ffmpeg")` returns None, matplotlib tries to launch
    "ffmpeg" anyway, and Popen raises FileNotFoundError from six frames deep
    with no indication of what is actually wrong.

    Searching the Packages tree directly sidesteps the whole PATH question,
    and globbing the version directory means it keeps working after an update
    instead of pinning a path that goes stale.
    """
    import glob
    import os
    import shutil

    import matplotlib

    if explicit_path and os.path.isfile(explicit_path):
        matplotlib.rcParams["animation.ffmpeg_path"] = explicit_path
        return explicit_path

    found = shutil.which("ffmpeg")
    if found:
        matplotlib.rcParams["animation.ffmpeg_path"] = found
        return found

    local = os.environ.get("LOCALAPPDATA", "")
    patterns = [
        os.path.join(local, "Microsoft", "WinGet", "Packages",
                     "Gyan.FFmpeg*", "**", "bin", "ffmpeg.exe"),
        os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
        r"C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe",
        r"C:\\ffmpeg\\bin\\ffmpeg.exe",
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            matplotlib.rcParams["animation.ffmpeg_path"] = hits[0]
            return hits[0]
    return None


def _resample(out: dict, times: np.ndarray) -> np.ndarray:
    """Return lower-bracketing indices; rendering interpolates between samples."""
    idx = np.searchsorted(out["t"], times, side="right") - 1
    return np.clip(idx, 0, len(out["t"]) - 1)


def _interp_quaternion(q0: np.ndarray, q1: np.ndarray, f: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float); q1 = np.asarray(q1, dtype=float)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - f) * q0 + f * q1
    return q / max(np.linalg.norm(q), 1e-15)


def _interp_state(out: dict, t: float):
    tt = np.asarray(out["t"], dtype=float)
    j = int(np.searchsorted(tt, t, side="right"))
    if j <= 0:
        return 0, np.asarray(out["r"][0]), np.asarray(out["q"][0])
    if j >= len(tt):
        k = len(tt) - 1
        return k, np.asarray(out["r"][k]), np.asarray(out["q"][k])
    k = j - 1
    f = (t - tt[k]) / max(tt[j] - tt[k], 1e-15)
    r = (1.0 - f) * out["r"][k] + f * out["r"][j]
    q = _interp_quaternion(out["q"][k], out["q"][j], f)
    return k, r, q


def _zoom_span(altitude: float, wide: float = 120_000.0,
               close: float = 200.0, h_hi: float = 60_000.0,
               h_lo: float = 300.0) -> float:
    """
    Half-span of the view, interpolated logarithmically on altitude.

    Log rather than linear because altitude itself spans three decades; a
    linear ramp would stay zoomed out until the last couple of seconds and
    then snap, which looks like a bug even though it isn't.
    """
    h = float(np.clip(altitude, h_lo, h_hi))
    f = (np.log(h) - np.log(h_lo)) / (np.log(h_hi) - np.log(h_lo))
    return float(np.exp(np.log(close) + f * (np.log(wide) - np.log(close))))


def _draw_engines(ax, n_lit: int):
    ax.clear()
    ax.set_facecolor(BG)
    pts, _ = engine_positions(1.0)
    mask = lit_engine_mask(n_lit)
    ax.add_patch(Circle((0, 0), 1.08, facecolor="#11161d", edgecolor=GRID, lw=1.0))
    for (x, y), hot in zip(pts, mask):
        ax.add_patch(Circle((x, y), 0.085,
                            facecolor=C_HOT if hot else "none",
                            edgecolor=C_HOT if hot else C_COLD, lw=1.0))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.35, 1.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, -1.27, f"{int(mask.sum()):2d} / 33", color=FG, ha="center",
            fontsize=11, family="monospace")


def animate_realtime(out: dict, target: np.ndarray, path: str = "mission_rt.mp4",
                     fps: int = 24, speed: float = 1.0,
                     body_frac: float = 0.10,
                     visual_align_altitude: float = 180.0,
                     ffmpeg_path: str | None = None,
                     hold_seconds: float = 3.0):
    """
    Animate at `speed` x wall clock. speed=1.0 is real time.

    Writes MP4 via ffmpeg. Set speed=6.0 for a faster-but-still-time-proportional version
    that still spends proportional time in each phase, unlike arc-length
    sampling which deliberately does not.
    """
    t_end = float(out["t"][-1])
    n_frames = int(t_end / speed * fps)
    times = np.linspace(0.0, t_end, n_frames)
    idx = _resample(out, times)

    # HOLD ON THE FINAL FRAME.
    #
    # The run ends the instant the catch criteria are evaluated, so without
    # this the video cuts to black mid-catch -- the viewer sees the booster
    # arrive and then nothing, with no chance to read the terminal numbers or
    # register that it actually stopped. Repeating the last logged state for a
    # couple of seconds costs almost nothing (the frames are identical) and
    # makes the ending legible.
    if hold_seconds > 0.0:
        idx = np.concatenate([idx, np.full(int(hold_seconds * fps), idx[-1])])
        n_frames = len(idx)

    r, qs = out["r"], out["q"]
    v = out["v"]
    speed_hist = np.linalg.norm(v, axis=1)

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    ax3 = fig.add_axes([0.00, 0.02, 0.66, 0.94], projection="3d", facecolor=BG)
    axe = fig.add_axes([0.69, 0.55, 0.13, 0.30], facecolor=BG)
    axa = fig.add_axes([0.85, 0.08, 0.13, 0.84], facecolor=BG)
    readout = fig.text(0.69, 0.94, "", color=FG, fontsize=13,
                       family="monospace", va="top")
    phase_txt = fig.text(0.69, 0.47, "", color=C_ACHIEVED, fontsize=12,
                         family="monospace", va="top")

    def draw(f):
        t_now = float(times[min(f, len(times) - 1)])
        i, p, q_now = _interp_state(out, t_now)
        span = _zoom_span(p[2])
        visual_target = visual_catch_point(target)
        p_vis = _smooth_visual_position(p, target, visual_target,
                                         altitude_start=visual_align_altitude)

        ax3.clear()
        ax3.set_facecolor(BG)
        for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            pane.set_pane_color((0.05, 0.07, 0.09, 1.0))
            pane._axinfo["grid"]["color"] = GRID

        ax3.plot(r[:i + 1, 0], r[:i + 1, 1], r[:i + 1, 2],
                 color=C_ACHIEVED, lw=1.3, alpha=0.85)

        # vehicle, scaled to the current view so it stays visible at every zoom
        nose = Q.rotate(q_now, np.array([1.0, 0.0, 0.0])) * (body_frac * span)
        ax3.plot([p_vis[0] - nose[0], p_vis[0] + nose[0]],
                 [p_vis[1] - nose[1], p_vis[1] + nose[1]],
                 [p_vis[2] - nose[2], p_vis[2] + nose[2]],
                 color=C_HOT, lw=5.0, solid_capstyle="round")

        # Tower and chopsticks, drawn only once the view is tight enough for
        # them to be more than a smudge. At a 100 km span a 146 m tower is
        # sub-pixel, so rendering it early just adds noise.
        if span < 3000.0:
            tower, arms = tower_geometry(catch_altitude=float(target[2]), r=p, target=target,
                                         pad=(-30.0, 0.0), arm_length=15.0)
            for segd in tower:
                ax3.plot(segd[:, 0], segd[:, 1], segd[:, 2],
                         color="#8b949e", lw=1.6, alpha=0.9)
            for segd in arms:
                ax3.plot(segd[:, 0], segd[:, 1], segd[:, 2],
                         color=C_TARGET, lw=3.0, alpha=0.95)
        ax3.scatter([target[0]], [target[1]], [target[2]], color=C_TARGET,
                    s=70, marker="X", depthshade=False)

        ax3.set_xlim(p[0] - span, p[0] + span)
        ax3.set_ylim(p[1] - span, p[1] + span)
        ax3.set_zlim(max(p[2] - span, 0.0), p[2] + span)
        ax3.set_box_aspect((1, 1, 1))
        ax3.tick_params(colors=FG, labelsize=6)
        ax3.set_xlabel("x (m)", color=FG, fontsize=7)
        ax3.set_ylabel("y (m)", color=FG, fontsize=7)
        ax3.set_zlabel("z (m)", color=FG, fontsize=7)
        # Camera stops rotating once the hold starts, so the final frames are
        # genuinely static rather than drifting past a stopped vehicle.
        sweep = min(f, n_frames - int(hold_seconds * fps) - 1)
        ax3.view_init(elev=14, azim=-70 + 20 * max(sweep, 0) / max(n_frames, 1))

        _draw_engines(axe, int(out["n_lit"][i]))

        axa.clear(); axa.set_facecolor(BG)
        axa.plot(r[:i + 1, 2] / 1000.0, out["t"][:i + 1], color=C_ACHIEVED, lw=1.4)
        axa.set_ylim(0, t_end); axa.set_xlim(0, r[:, 2].max() / 1000 * 1.05)
        axa.invert_yaxis()
        axa.tick_params(colors=FG, labelsize=7)
        axa.set_xlabel("alt (km)", color=FG, fontsize=8)
        axa.set_ylabel("T+ (s)", color=FG, fontsize=8)
        for sp in axa.spines.values():
            sp.set_color(GRID)
        axa.grid(alpha=0.15, color=GRID)

        tilt = out["tilt"][i]
        readout.set_text(
            f"T+{out['t'][i]:7.2f} s\n"
            f"ALT  {r[i, 2] / 1000:8.3f} km\n"
            f"VEL  {speed_hist[i]:8.1f} m/s\n"
            f"TILT {tilt:8.2f} deg\n"
            f"PROP {out['prop'][i] / 1000:8.2f} t"
        )
        phase_txt.set_text(str(out["segment"][i]).upper())
        return []

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, blit=False)

    # MP4 needs ffmpeg as an external executable. If it is not on PATH,
    # matplotlib raises FileNotFoundError from deep inside subprocess, which
    # is an unhelpful place to discover a missing dependency -- especially
    # after the frames have already been computed. Check first and fall back
    # to GIF rather than throwing away the work.
    want_mp4 = path.endswith(".mp4")
    if want_mp4:
        ensure_ffmpeg(ffmpeg_path)
    if want_mp4 and not animation.FFMpegWriter.isAvailable():
        path = path[:-4] + ".gif"
        print("ffmpeg not found (PATH or winget dirs) -- writing GIF: " + path)
        print("  pass ffmpeg_path=r'C:\\...\\ffmpeg.exe' to point at it directly")
        if n_frames > 600:
            print(f"  NOTE: {n_frames} frames as a GIF will be large and slow.")
            print("  Install ffmpeg (winget install Gyan.FFmpeg) for MP4, or")
            print("  pass a higher `speed` / lower `fps` to cut the frame count.")
        want_mp4 = False

    if want_mp4:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    else:
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path
