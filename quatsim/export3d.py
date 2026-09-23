"""
Export a flown mission for the 3-D renderer in ``viz3d/``.

FRAMES
------
The simulation is East-North-Up (x downrange/east, y north, z up). Three.js
is Y-up, so a point maps as

    three = (x, z, -y)            M = [[1,0,0],[0,0,1],[0,-1,0]], det M = +1

The booster model is built directly in BODY coordinates (nose +x, third grid
fin +z) and its world orientation is the rotation M * R(q_sim). That product
is exported as a quaternion, so the renderer does no frame bookkeeping.

PLAYBACK SCHEDULE
-----------------
Video time is not sim time. The rate (sim seconds per video second) is:

    flip + boostback                      1x   (real time)
    reorient + coast, above ~8 km        12x   (fast forward)
    last km before the landing burn       3x   (slower fast forward)
    landing burn + final approach + catch 1x   (real time)

with smoothed transitions, then a hold on the caught booster. The schedule is
computed here once, so the MP4 and the interactive viewer play identically.
"""

from __future__ import annotations

import json

import numpy as np

from . import quaternion as Q

M_ENU_TO_THREE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
Q_M = Q.from_dcm(M_ENU_TO_THREE)

RATE_BOOSTBACK = 1.0
RATE_COAST = 12.0
RATE_APPROACH = 3.0
RATE_LANDING = 1.0
APPROACH_ALTITUDE = 8_000.0


def to_three(p):
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _phase(out):
    """0 flip, 1 boostback, 2 coast/pre-ignition, 3 brake, 4 final approach."""
    from .visuals import phase_index
    return phase_index(out)


def playback_schedule(out, fps=30, hold=6.0):
    t = np.asarray(out["t"], float)
    alt = np.asarray(out["r"])[:, 2]
    ph = _phase(out)
    rate = np.where(ph <= 1, RATE_BOOSTBACK,
                    np.where(ph >= 3, RATE_LANDING,
                             np.where(alt > APPROACH_ALTITUDE, RATE_COAST,
                                      RATE_APPROACH)))
    # Smooth the rate in SIM time with a ~4 s window (log-rate, so a 12x->1x
    # change eases rather than lurches), but never let the landing burn start
    # fast: the smoothing is one-sided into the burn.
    lr = np.log(rate)
    dt_s = float(np.median(np.diff(t)))
    w = max(int(round(2.0 / dt_s)), 1)
    k = np.ones(2 * w + 1) / (2 * w + 1)
    lr_s = np.convolve(np.pad(lr, w, mode="edge"), k, mode="valid")
    lr_s = np.where(ph >= 3, lr, np.minimum(lr_s, lr + np.log(4.0)))
    rate_s = np.exp(lr_s)
    vt = np.concatenate([[0.0], np.cumsum(np.diff(t) / rate_s[:-1])])
    frames = np.arange(0.0, vt[-1], 1.0 / fps)
    ft = np.interp(frames, vt, t)
    fr = np.interp(frames, vt, rate_s)
    n_hold = int(hold * fps)
    ft = np.concatenate([ft, np.full(n_hold, t[-1])])
    fr = np.concatenate([fr, np.zeros(n_hold)])
    return ft, fr


def export_mission(out: dict, target, report: dict | None, path: str,
                   fps: int = 30, stride: int = 1):
    t = np.asarray(out["t"], float)[::stride]
    r = np.asarray(out["r"], float)[::stride]
    v = np.asarray(out["v"], float)[::stride]
    q = np.asarray(out["q"], float)[::stride]
    ph = _phase(out)[::stride]
    thr = np.asarray(out["throttle"], float)[::stride]
    nl = np.asarray(out["n_lit"], int)[::stride]
    prop = np.asarray(out["prop"], float)[::stride]
    tilt = np.asarray(out["tilt"], float)[::stride]

    qo = np.array([Q.multiply(Q_M, qq) for qq in q])       # world = M R(q)
    # keep quaternion sign continuous for interpolation
    for i in range(1, len(qo)):
        if np.dot(qo[i], qo[i - 1]) < 0:
            qo[i] = -qo[i]

    ft, fr = playback_schedule(out, fps=fps)
    le = out.get("landing_events", {})
    tgt = np.asarray(target, float)
    hist = out.get("boostback_history", [])
    events = [
        {"t": 0.0, "name": "STAGE SEP"},
        {"t": 1.0, "name": "BOOSTBACK STARTUP"},
    ]
    for tt, name in out["events"]:
        if "predictive cutoff" in name:
            events.append({"t": float(tt), "name": "33 → 13"})
        elif name == "begin boostback_3":
            events.append({"t": float(tt), "name": "13 → 3"})
        elif name == "begin reorient":
            events.append({"t": float(tt), "name": "BOOSTBACK SHUTDOWN"})
    ka = int(np.argmax(r[:, 2]))
    events.append({"t": float(t[ka]), "name": "APOGEE"})
    if le and np.isfinite(le.get("ignition_t", np.nan)):
        events.append({"t": float(le["ignition_t"]), "name": "LANDING BURN"})
    if le and np.isfinite(le.get("handover_t", np.nan)):
        events.append({"t": float(le["handover_t"]), "name": "13 → 3"})
    events.append({"t": float(t[-1]), "name": "CATCH"})
    events.sort(key=lambda e: e["t"])

    from .visuals import engine_layout
    lay, lit = engine_layout()

    data = {
        "t": np.round(t, 3).tolist(),
        "p": np.round(to_three(r), 2).ravel().tolist(),
        "v": np.round(to_three(v), 2).ravel().tolist(),
        "q": np.round(qo[:, [1, 2, 3, 0]], 6).ravel().tolist(),  # x,y,z,w
        "phase": ph.astype(int).tolist(),
        "throttle": np.round(thr, 3).tolist(),
        "n_lit": nl.tolist(),
        "prop": np.round(prop / 1000.0, 2).tolist(),
        "tilt": np.round(tilt, 3).tolist(),
        "frames": {"t": np.round(ft, 4).tolist(),
                   "rate": np.round(fr, 3).tolist(), "fps": fps},
        "target": to_three(tgt).tolist(),
        "target_enu": tgt.tolist(),
        "events": events,
        "engines": {"pos": np.round(lay, 4).tolist(),
                    "lit": {str(k): v for k, v in lit.items()}},
        "ignition_t": le.get("ignition_t"),
        "handover_t": le.get("handover_t"),
        "cutoff_t": (float(hist[-1][0] + hist[-1][1]) if hist else None),
        "report": None if report is None else {
            "caught": bool(report["caught"]),
            "checks": {k: [float(v[0]), bool(v[1])]
                       for k, v in report["checks"].items()},
            "prop_t": float(report["propellant_remaining_t"]),
        },
    }
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return path


def build_viewer(mission_json: str, out_html: str,
                 viz_dir: str | None = None,
                 three_version: str = "0.170.0") -> str:
    """
    Write a single self-contained interactive viewer: viz3d/index.html with
    the scene module and the mission data inlined, and Three.js loaded from
    jsDelivr. Open it in any browser: play / pause / scrub, free camera.
    """
    import os
    viz_dir = viz_dir or os.path.join(os.path.dirname(__file__), "..", "viz3d")
    html = open(os.path.join(viz_dir, "index.html")).read()
    js = open(os.path.join(viz_dir, "scene.js")).read()
    data = open(mission_json).read()
    import base64
    fin = base64.b64encode(open(os.path.join(viz_dir, "assets", "gridfin.i16"),
                                "rb").read()).decode()
    cdn = f"https://cdn.jsdelivr.net/npm/three@{three_version}"
    html = html.replace('"./node_modules/three/build/three.module.js"',
                        f'"{cdn}/build/three.module.js"')
    html = html.replace('"./node_modules/three/examples/jsm/"',
                        f'"{cdn}/examples/jsm/"')
    html = html.replace('<script type="module" src="./scene.js"></script>',
                        "<script>window.MISSION = " + data + ";\n"
                        'window.GRIDFIN_B64 = "' + fin + '";</script>\n'
                        '<script type="module">\n' + js + "\n</script>")
    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html
