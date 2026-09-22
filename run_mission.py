"""
Run the converged RTLS mission. No Jupyter required.

Place this next to the `quatsim/` package directory and run it -- green arrow
in PyCharm, or `python run_mission.py` from a terminal with the venv active.

    quatsim/                 <- project folder
    |-- quatsim/             <- the package
    |-- tests/
    +-- run_mission.py       <- this file

Flags:
    --resolve     re-solve the boostback duration from scratch (~3 min)
    --figures     write the static figures
    --animate     write the summary GIF and a real-time animation
    --speed 1.0   animation speed multiplier (1.0 = true real time, slow)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from quatsim.aero import AeroModel, GridFinModel
from quatsim.control import AttitudeController, ControlGains
from quatsim.mission import (SeparationState, build_mission, mission_report,
                             propagate_flip, solve_boostback)
from quatsim.entry import EntryConfig
from quatsim.landing import LandingConfig
from quatsim.phases import FlightSequencer, catch_report, solve_ignition_altitude
from quatsim.position import PositionController, PositionGains
from quatsim.vehicle import Vehicle

# --- converged solution, see SOLUTION.md -----------------------------------
FLIP_DURATION = 4.0
BB_THROTTLE = 0.87
T33_BURN = 9.1472         # s on 33 engines
BB_ELEVATION = 1.0        # deg above horizontal on the retrograde axis
ALT_BIAS = 0.0            # termination now lands on target without bias
TARGET = np.array([0.0, 0.0, 105.0])


def build():
    vehicle = Vehicle(prop_mass=500e3)
    aero = AeroModel()
    fins = GridFinModel()

    # Separation state calibrated against Flight 13: downrange fitted to the
    # 51 km splashdown, which then independently predicted the measured 11 s
    # boostback.
    sep = SeparationState(altitude=68e3, downrange=83e3, speed=1500.0,
                          flight_path_angle=25.0, prop_remaining=500e3)

    sf = propagate_flip(vehicle, aero, sep, FLIP_DURATION, n_lit=5)
    bb0 = solve_boostback(vehicle, aero, sep, start_state=sf,
                          throttle=BB_THROTTLE, bracket=(1.0, 40.0))
    land = solve_ignition_altitude(vehicle, aero, bb0["prop_after"], 105.0,
                                   13, 0.45, v_entry=bb0["arrival_speed"])

    seq = FlightSequencer(
        vehicle, aero,
        AttitudeController(vehicle, ControlGains(wn=1.5, zeta=0.8)),
        PositionController(vehicle, PositionGains(wn=0.35, zeta=0.95)),
        fins=fins)
    return vehicle, aero, sep, bb0, land, seq


# Boostback aim point: where the ballistic arc should cross 1200 m altitude,
# measured downrange of the tower. Deliberately OFFSHORE (+x): if the landing
# burn never lit, the booster would come down short of the tower rather than
# on it, and the landing burn performs the final divert in. The predictive
# boostback now hits this point to within a few metres, so it is a design
# choice rather than a calibration fudge.
AIM_X_1200 = 250.0


def landing_config() -> LandingConfig:
    return LandingConfig(target=TARGET.copy())


def fly(t33, vehicle, aero, sep, bb0, land, seq, dt=0.02, log_every=20,
        aim_x=AIM_X_1200, landing=None, steer=True):
    bb = dict(bb0)
    bb["duration"] = t33 + 6.0
    segs = build_mission(vehicle, aero, sep, bb, land, TARGET, np.zeros(3),
                         flip_duration=FLIP_DURATION, flip_engines=5,
                         boostback_elevation=np.radians(BB_ELEVATION))
    for s in segs:
        if s.name.startswith("boostback"):
            s.throttle = BB_THROTTLE
        if s.name == "boostback_33":
            s.predictive_boostback = True
            s.boostback_target_altitude = 1200.0
            s.boostback_target_x = aim_x
            s.boostback_target_y = float(TARGET[1])
            s.boostback_tail13 = 3.0
            s.boostback_tail3 = 3.0
            # Predictive mode owns the cutoff; this is only a safe maximum
            # window, not a commanded burn duration.
            s.duration = 14.0
        if s.name == "coast" and steer:
            s.entry = EntryConfig(aim=np.array([aim_x, float(TARGET[1])]),
                                  aim_altitude=1200.0)
        if s.name == "landing":
            s.landing = landing if landing is not None else landing_config()
            s.r_target = TARGET
            s.catch_altitude = float(TARGET[2])
    return seq.run(sep.state_vector(), segs, dt=dt, log_every=log_every)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve", action="store_true",
                    help="re-solve the boostback duration (~3 min)")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--speed", type=float, default=6.0,
                    help="animation speed; 1.0 is true real time")
    args = ap.parse_args(argv)

    t0 = time.time()
    vehicle, aero, sep, bb0, land, seq = build()
    print(vehicle.summary())
    print(mission_report(sep, bb0, land))
    print()

    t33 = T33_BURN
    if args.resolve:
        # Bisect on final downrange using the FULL 6-DOF simulator. Solving on
        # a simplified model and hoping it matches never worked -- gravity
        # model, flip thrust direction and timestep quantization each showed up
        # as tens of km of miss.
        print("re-solving boostback duration...")
        lo, hi = 11.0, 13.0
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            x = fly(mid, vehicle, aero, sep, bb0, land, seq)["r"][-1, 0]
            lo, hi = (mid, hi) if x > 0 else (lo, mid)
        t33 = 0.5 * (lo + hi)
        print(f"  converged t33 = {t33:.4f} s\n")

    out = fly(t33, vehicle, aero, sep, bb0, land, seq,
              log_every=(1 if args.animate else 20))
    r = out["r"]

    print(f"apogee        {r[:, 2].max() / 1000:8.1f} km")
    if seq._boostback_pred is not None:
        print(f"predictive 33-engine cutoff  {seq._boostback_pred['remaining_33']:8.4f} s")
        print(f"predicted target-x residual   {seq._boostback_pred['predicted_x']:8.2f} m")
    print(f"flight time   {out['t'][-1]:8.1f} s")
    print()
    rep = catch_report(out["state"][-1], TARGET)
    for name, (value, ok) in rep["checks"].items():
        print(f"  {name:20s} {value:10.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"  {'propellant left':20s} {rep['propellant_remaining_t']:10.2f} t")
    print(f"\n  CAUGHT: {rep['caught']}")
    print(f"\n({time.time() - t0:.0f} s)")

    if args.figures:
        from quatsim import telemetry as T
        print(T.tracking_error(out, path="tracking_error.png"))
        print(T.telemetry_panel(out, target=TARGET, path="telemetry_panel.png"))

    if args.animate:
        from quatsim import telemetry as T
        from quatsim.realtime import animate_realtime
        print(T.animate_mission(out, TARGET, n_frames=170, fps=24,
                                path="mission_summary.gif", hold_seconds=3.0,
                                visual_align_altitude=180.0))
        # MP4 needs ffmpeg on PATH; falls back to GIF automatically if absent.
        print(animate_realtime(out, TARGET, path="mission_rt.mp4",
                               speed=args.speed, fps=24, hold_seconds=3.0,
                               visual_align_altitude=180.0))

    return 0 if rep["caught"] else 1


if __name__ == "__main__":
    sys.exit(main())
