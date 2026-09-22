"""
Run the converged RTLS mission. No Jupyter required.

Place this next to the `quatsim/` package directory and run it -- green arrow
in PyCharm, or `python run_mission.py` from a terminal with the venv active.

    quatsim/                 <- project folder
    |-- quatsim/             <- the package
    |-- tests/
    +-- run_mission.py       <- this file

Flags:
    --figures         mission overview + landing detail figures (figs/)
    --animate         time-warped 2-D MP4 of the whole flight (figs/catch.mp4)
    --animate3d       3-D MP4 (figs/catch3d.mp4) + interactive viewer HTML
    --montecarlo N    N dispersed cases + report (figs/monte_carlo.*)
    --no-steer        fly without grid-fin entry steering, for comparison
"""

from __future__ import annotations

import argparse
import os
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
# Nominal 33-engine boostback time. Only a planning reference now: the
# closed-loop predictive boostback solves the real cutoff in flight.
T33_BURN = 9.1472         # s on 33 engines
BB_ELEVATION = 1.0        # deg above horizontal on the retrograde axis
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--figures", action="store_true",
                    help="write figs/mission_overview.png and figs/landing_detail.png")
    ap.add_argument("--animate", action="store_true",
                    help="write figs/catch.mp4 (time-warped, ~1 min)")
    ap.add_argument("--animate3d", action="store_true",
                    help="write figs/catch3d.mp4 (3-D, rendered in headless Chromium;\n"
                         "needs `npm install` in viz3d/) and figs/catch3d_viewer.html")
    ap.add_argument("--montecarlo", type=int, default=0, metavar="N",
                    help="run N dispersed cases and write figs/monte_carlo.png")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-steer", action="store_true",
                    help="disable grid-fin entry steering (for comparison)")
    args = ap.parse_args(argv)

    os.makedirs("figs", exist_ok=True)
    t0 = time.time()
    vehicle, aero, sep, bb0, land, seq = build()
    print(vehicle.summary())
    print(mission_report(sep, bb0, land))
    print()

    out = fly(T33_BURN, vehicle, aero, sep, bb0, land, seq,
              log_every=(2 if (args.animate or args.animate3d) else 5),
              steer=not args.no_steer)
    r = out["r"]
    le = out["landing_events"]

    print(f"apogee                 {r[:, 2].max() / 1000:8.1f} km")
    hist = out["boostback_history"]
    if hist:
        print(f"33-engine cutoff       {hist[-1][0] + hist[-1][1]:8.3f} s MET "
              f"({len(hist)} closed-loop re-targets, final yaw "
              f"{np.degrees(hist[-1][2]):+.3f} deg)")
    print(f"landing ignition       {le['ignition_alt']:8.0f} m at "
          f"{le['ignition_speed']:.0f} m/s")
    print(f"13 -> 3 engines        {le['handover_alt']:8.0f} m")
    print(f"flight time            {out['t'][-1]:8.1f} s")
    print()
    rep = catch_report(out["state"][-1], TARGET)
    for name, (value, ok) in rep["checks"].items():
        print(f"  {name:20s} {value:10.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"  {'propellant left':20s} {rep['propellant_remaining_t']:10.2f} t")
    print(f"\n  CAUGHT: {rep['caught']}")
    print(f"\n({time.time() - t0:.0f} s)")

    if args.figures or args.animate:
        from quatsim import visuals as VIS
    if args.figures:
        print(VIS.mission_overview(out, TARGET, rep,
                                   path="figs/mission_overview.png"))
        print(VIS.landing_detail(out, TARGET, rep,
                                 path="figs/landing_detail.png"))
    if args.animate:
        # MP4 via ffmpeg (PATH, else the imageio-ffmpeg wheel); GIF fallback.
        print(VIS.animate_catch(out, TARGET, path="figs/catch.mp4",
                                report=rep, progress=print))

    if args.animate3d:
        import subprocess
        from quatsim.export3d import build_viewer, export_mission
        here = os.path.dirname(os.path.abspath(__file__))
        data = export_mission(out, TARGET, rep,
                              os.path.join(here, "viz3d", "data", "mission.json"))
        print(build_viewer(data, "figs/catch3d_viewer.html"))
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = "ffmpeg"
        subprocess.run(["node", os.path.join(here, "viz3d", "render.mjs"),
                        "--out", os.path.abspath("figs/catch3d.mp4"),
                        "--workers", str(args.workers), "--ffmpeg", ff],
                       check=True)

    if args.montecarlo:
        import pickle
        from quatsim import visuals as VIS
        from quatsim.montecarlo import run_campaign, write_csv
        print(f"\nMonte Carlo: {args.montecarlo} dispersed cases + nominal, "
              f"{args.workers} workers")
        res = run_campaign(args.montecarlo, seed=args.seed,
                           workers=args.workers)
        with open("figs/monte_carlo.pkl", "wb") as fh:
            pickle.dump(res, fh)
        print(write_csv(res, "figs/monte_carlo.csv"))
        print(VIS.monte_carlo_report(res, path="figs/monte_carlo.png"))
        n_ok = sum(r["row"]["caught"] for r in res)
        print(f"  caught {n_ok}/{len(res)}")

    return 0 if rep["caught"] else 1


if __name__ == "__main__":
    sys.exit(main())
