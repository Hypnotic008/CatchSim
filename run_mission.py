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
    --relight-fail K  K of the 13 landing engines fail to relight
    --engine-shutdown T  one lit engine shuts down T s after landing ignition
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from quatsim.aero import AeroModel, GridFinModel, GridFinSet
from quatsim.control import AttitudeController, ControlGains
from quatsim.mission import (SeparationState, build_mission, mission_report,
                             propagate_flip, solve_boostback)
from quatsim.entry import EntryConfig
from quatsim.landing import LandingConfig
from quatsim.phases import FlightSequencer, catch_report, solve_ignition_altitude
from quatsim.position import PositionController, PositionGains
from quatsim.tower import CatchTower
from quatsim.vehicle import Vehicle

# --- converged solution, see SOLUTION.md -----------------------------------
FLIP_DURATION = 4.0
BB_THROTTLE = 0.87
# Nominal 33-engine boostback time. Only a planning reference now: the
# closed-loop predictive boostback solves the real cutoff in flight.
T33_BURN = 9.1472         # s on 33 engines
BB_ELEVATION = 1.0        # deg above horizontal on the retrograde axis
TARGET = np.array([0.0, 0.0, 105.0])

# Hot-stage separation (estimates; see build()).
SEPARATION = dict(altitude=68e3, downrange=83e3, speed=1500.0,
                  flight_path_angle=25.0, prop_remaining=500e3)

# Flight profiles (--flight). 13: the V2-era calibration this sim was built on
# (separation 68 km, CatchSim's estimated aerodynamics). 15: V3 as flown on
# Flight 15 -- separation ~60 km and booster apogee 92.2 km (flight-path angle
# fitted to the apogee), on the rocketcfd aerodynamic tables
# (data/aero_db_v3.json), with the boostback aim point further offshore: with
# the q-alpha limit the brake cannot lean hard at high dynamic pressure, so the
# booster carries its horizontal speed further and must start further out.
# Flight 15 ignited the landing burn at ~3 km; this profile ignites at ~2.1 km.
FLIGHTS = {
    13: dict(separation={}, aero=None, aim_x=200.0),
    15: dict(separation=dict(altitude=60e3, flight_path_angle=20.7),
             aero="data/aero_db_v3.json", aim_x=550.0),
}

# Which Starbase tower catches the booster: 2 (default; arms open to the
# south-southwest, so the booster coming home from the east approaches across
# the arms) or 1 (arms open to the ocean). CATCHSIM_TOWER lets worker
# processes inherit the choice.
TOWER_NUMBER = int(os.environ.get("CATCHSIM_TOWER", "2"))


def tower(n: int | None = None) -> CatchTower:
    return CatchTower.numbered(TOWER_NUMBER if n is None else n,
                               center=(float(TARGET[0]), float(TARGET[1])),
                               catch_altitude=float(TARGET[2]))


def report(out, n: int | None = None) -> dict:
    """Catch report for a flown mission against the selected tower."""
    tw = tower(n)
    return catch_report(out["state"][-1], TARGET, catch_axis=tw.facing,
                        corridor=tw, trajectory=out["r"])


_CFD_DB = {}


def make_aero(**kw):
    """The aerodynamic model: CatchSim's estimates, or with CATCHSIM_AERO_CFD
    (run_mission --aero-cfd) the rocketcfd table, for both truth and guidance."""
    path = os.environ.get("CATCHSIM_AERO_CFD")
    if not path:
        return AeroModel(**kw)
    from quatsim.aero_cfd import CFDAeroDatabase, CFDAeroModel
    if path not in _CFD_DB:
        _CFD_DB[path] = CFDAeroDatabase(path)
    return CFDAeroModel(db=_CFD_DB[path], **kw)


def build():
    vehicle = Vehicle(prop_mass=500e3)
    aero = make_aero()
    # Three grid fins with their own deflections (rudder + catch pair).
    # CATCHSIM_FIN_CFD = fin_control.json from rocketcfd for CFD derivatives.
    fins = GridFinSet()
    if os.environ.get("CATCHSIM_FIN_CFD"):
        fins.load_cfd(os.environ["CATCHSIM_FIN_CFD"])
    elif os.environ.get("CATCHSIM_AERO_CFD"):
        fins.load_aero_db(os.environ["CATCHSIM_AERO_CFD"])

    # Separation state calibrated against Flight 13: downrange fitted to the
    # 51 km splashdown, which then independently predicted the measured 11 s
    # boostback.
    sep = SeparationState(**SEPARATION)

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
AIM_X_1200 = float(os.environ.get("CATCHSIM_AIM_X", "200.0"))


def apply_flight(n: int):
    """Set the module's separation state and aim point for a flight profile
    (also called in Monte Carlo workers via CATCHSIM_FLIGHT)."""
    global AIM_X_1200
    prof = FLIGHTS[int(n)]
    SEPARATION.update(prof["separation"])
    AIM_X_1200 = prof["aim_x"]
    # the profile's aerodynamic table too, unless one was chosen explicitly: a Monte Carlo
    # worker started with only CATCHSIM_FLIGHT set otherwise flew the placeholder aero
    if prof.get("aero") and not os.environ.get("CATCHSIM_AERO_CFD"):
        here = os.path.dirname(os.path.abspath(__file__))
        os.environ["CATCHSIM_AERO_CFD"] = os.path.join(here, prof["aero"])


if os.environ.get("CATCHSIM_FLIGHT"):
    apply_flight(int(os.environ["CATCHSIM_FLIGHT"]))


def landing_config() -> LandingConfig:
    tw = tower()
    return LandingConfig(target=TARGET.copy(), catch_roll_axis=tw.facing,
                         corridor_across=tw.across[:2],
                         arm_travel_toward=tw.downrange_travel, arm_travel_across=tw.crossrange_travel,
                         arm_height=float(TARGET[2]) + 35.0)


def fly(t33, vehicle, aero, sep, bb0, land, seq, dt=0.02, log_every=20,
        aim_x=None, landing=None, steer=True):
    aim_x = AIM_X_1200 if aim_x is None else aim_x
    bb = dict(bb0)
    bb["duration"] = t33 + 6.0
    segs = build_mission(vehicle, aero, sep, bb, land, TARGET, np.zeros(3),
                         flip_duration=FLIP_DURATION, flip_engines=5,
                         boostback_elevation=np.radians(BB_ELEVATION),
                         catch_axis=tower().facing)
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
    global TOWER_NUMBER
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
    ap.add_argument("--relight-fail", type=int, default=0, metavar="K",
                    help="K of the 13 landing-burn engines fail to relight")
    ap.add_argument("--engine-shutdown", type=float, default=-1.0, metavar="T",
                    help="one lit engine shuts down T s after landing ignition")
    ap.add_argument("--tower", type=int, choices=(1, 2), default=TOWER_NUMBER,
                    help="catch tower: 2 = Tower 2, arms facing SSW (default);\n"
                         "1 = Tower 1, arms facing the ocean")
    ap.add_argument("--flight", type=int, choices=sorted(FLIGHTS), default=13,
                    help="flight profile: 13 = separation 68 km on the estimated\n"
                         "aerodynamics (default); 15 = V3 / Flight 15 (separation\n"
                         "60 km, apogee ~92 km) on the CFD aerodynamic tables")
    ap.add_argument("--aero-cfd", default=None, metavar="JSON",
                    help="body aerodynamics from rocketcfd (aero_db.json from\n"
                         "cases/aero_database.py) instead of the estimates")
    ap.add_argument("--fin-cfd", default=None, metavar="JSON",
                    help="grid fin control derivatives from rocketcfd\n"
                         "(fin_control.json from cases/fin_control_export.py)")
    ap.add_argument("--no-steer", action="store_true",
                    help="disable grid-fin entry steering (for comparison)")
    args = ap.parse_args(argv)
    TOWER_NUMBER = args.tower
    os.environ["CATCHSIM_TOWER"] = str(args.tower)
    if args.fin_cfd:
        os.environ["CATCHSIM_FIN_CFD"] = os.path.abspath(args.fin_cfd)
    prof = FLIGHTS[args.flight]
    os.environ["CATCHSIM_FLIGHT"] = str(args.flight)
    apply_flight(args.flight)
    if args.aero_cfd or prof["aero"]:
        here = os.path.dirname(os.path.abspath(__file__))
        os.environ["CATCHSIM_AERO_CFD"] = os.path.abspath(args.aero_cfd or os.path.join(here, prof["aero"]))

    os.makedirs("figs", exist_ok=True)
    t0 = time.time()
    vehicle, aero, sep, bb0, land, seq = build()
    vehicle.landing_relight_failures = args.relight_fail
    vehicle.landing_shutdown_t = args.engine_shutdown
    print(vehicle.summary())
    print(mission_report(sep, bb0, land))
    tw = tower()
    print(f"catch tower            Tower {TOWER_NUMBER}, arms facing "
          f"{tw.facing_deg:.0f} deg (compass)")
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
    if np.isfinite(le.get("mid_alt", np.nan)):
        print(f"13 -> 5 engines        {le['mid_alt']:8.0f} m")
    first = "5" if np.isfinite(le.get("mid_alt", np.nan)) else "13"
    print(f"{first + ' -> 3 engines':23s}{le['handover_alt']:8.0f} m")
    print(f"flight time            {out['t'][-1]:8.1f} s")
    print()
    rep = report(out)
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
        print(VIS.grid_fin_history(out, path="figs/grid_fins.png"))
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
