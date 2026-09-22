"""
Monte Carlo dispersion analysis for the full RTLS catch.

WHAT "ROBUST" MEANS HERE
------------------------
A single converged trajectory proves the guidance CAN catch the booster. It
says nothing about whether it WILL, because the real separation state, engine
performance, drag and wind are never the ones the mission was designed on.
This module answers the second question.

Every case:

  * PLANS on the nominal models: the pre-flight boostback/ignition solves in
    run_mission.build() use the nominal vehicle, aero and separation state.
  * FLIES on a dispersed truth plant: separation state, propellant load, dry
    mass, per-engine thrust, Isp, drag coefficient, atmospheric density, wind
    and initial body rates are all sampled.
  * GUIDES on the nominal models: the sequencer's guidance-side vehicle and
    aero stay nominal, so the boostback predictor and the landing guidance
    do not know the thrust, Isp, drag or wind dispersion. Feedback has to
    absorb all of it.

Dispersions are Gaussian with the listed 3-sigma values, truncated at 3 sigma.
They are ENGINEERING ESTIMATES for a stress test, not published SpaceX
figures -- widen them and re-run to find where the design breaks.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import dynamics6dof as D6
from .aero import AeroModel
from .phases import catch_report


# ---------------------------------------------------------------------------
# wind


@dataclass
class WindProfile:
    """
    Horizontal wind as a function of altitude.

    A log-law boundary layer near the surface blending into a Gaussian jet
    stream around ``jet_altitude``, fading to calm above ~25 km where the
    booster is in near-vacuum anyway. Direction veers linearly with altitude.
    Deterministic for a given parameter set; the Monte Carlo samples the
    parameters.
    """
    surface_speed: float = 6.0          # m/s at 10 m
    jet_speed: float = 35.0             # m/s peak
    jet_altitude: float = 11_000.0      # m
    jet_width: float = 4_000.0          # m (1-sigma of the jet bump)
    direction_surface: float = 0.0      # rad, direction the wind blows TOWARD
    veer: float = np.radians(30.0)      # rad of veer from surface to jet

    def __call__(self, altitude: float) -> np.ndarray:
        h = max(float(altitude), 0.5)
        bl = self.surface_speed * np.log(h / 0.1) / np.log(10.0 / 0.1)
        bl = min(bl, self.surface_speed * 1.6)
        jet = self.jet_speed * np.exp(-0.5 * ((h - self.jet_altitude)
                                              / self.jet_width) ** 2)
        fade = 1.0 if h < 20_000.0 else np.exp(-(h - 20_000.0) / 4_000.0)
        speed = max(bl, jet) * fade
        frac = min(h / self.jet_altitude, 1.0)
        ang = self.direction_surface + self.veer * frac
        return np.array([speed * np.cos(ang), speed * np.sin(ang), 0.0])


# ---------------------------------------------------------------------------
# dispersions


@dataclass
class DispersionSpec:
    """3-sigma values. Set any to zero to switch that dispersion off."""
    sep_altitude: float = 1_500.0       # m
    sep_downrange: float = 3_000.0      # m
    sep_crossrange: float = 500.0       # m
    sep_speed: float = 30.0             # m/s
    sep_fpa: float = 1.5                # deg
    sep_heading: float = 0.5            # deg (-> ~13 m/s crossrange velocity)
    sep_rate: float = 0.5               # deg/s, each body axis
    prop: float = 15_000.0              # kg at separation
    dry_mass: float = 0.015             # fraction
    thrust: float = 0.03                # fraction, per-engine thrust
    isp: float = 0.02                   # fraction, SL and vac together
    cd: float = 0.15                    # fraction
    density: float = 0.08               # fraction
    wind_surface_max: float = 12.0      # m/s, uniform 0..max
    wind_jet_max: float = 60.0          # m/s, uniform 0..max
    # Day-of-launch wind forecast error (balloon sounding -> guidance).
    wind_forecast_scale: float = 0.30   # fraction of magnitude
    wind_forecast_dir: float = 15.0     # deg


def _tn(rng, three_sigma):
    """Truncated normal with the given 3-sigma, clipped at 3 sigma."""
    if three_sigma == 0.0:
        return 0.0
    return float(np.clip(rng.normal(0.0, three_sigma / 3.0),
                         -three_sigma, three_sigma))


def sample_case(rng: np.random.Generator, spec: DispersionSpec) -> dict:
    return {
        "d_alt": _tn(rng, spec.sep_altitude),
        "d_downrange": _tn(rng, spec.sep_downrange),
        "d_crossrange": _tn(rng, spec.sep_crossrange),
        "d_speed": _tn(rng, spec.sep_speed),
        "d_fpa": _tn(rng, spec.sep_fpa),
        "d_heading": _tn(rng, spec.sep_heading),
        "d_rate": [_tn(rng, spec.sep_rate) for _ in range(3)],
        "d_prop": _tn(rng, spec.prop),
        "k_dry": 1.0 + _tn(rng, spec.dry_mass),
        "k_thrust": 1.0 + _tn(rng, spec.thrust),
        "k_isp": 1.0 + _tn(rng, spec.isp),
        "k_cd": 1.0 + _tn(rng, spec.cd),
        "k_rho": 1.0 + _tn(rng, spec.density),
        "wind_surface": float(rng.uniform(0.0, spec.wind_surface_max)),
        "wind_jet": float(rng.uniform(0.0, spec.wind_jet_max)),
        "wind_dir": float(rng.uniform(0.0, 2.0 * np.pi)),
        "wind_veer": float(rng.uniform(-np.pi / 4, np.pi / 4)),
        "wind_jet_alt": float(rng.uniform(9_000.0, 14_000.0)),
        "wind_fc_scale": 1.0 + _tn(rng, spec.wind_forecast_scale),
        "wind_fc_dir": _tn(rng, spec.wind_forecast_dir),
    }


def nominal_case() -> dict:
    return {"d_alt": 0.0, "d_downrange": 0.0, "d_crossrange": 0.0,
            "d_speed": 0.0, "d_fpa": 0.0, "d_heading": 0.0,
            "d_rate": [0.0, 0.0, 0.0], "d_prop": 0.0, "k_dry": 1.0,
            "k_thrust": 1.0, "k_isp": 1.0, "k_cd": 1.0, "k_rho": 1.0,
            "wind_surface": 0.0, "wind_jet": 0.0, "wind_dir": 0.0,
            "wind_veer": 0.0, "wind_jet_alt": 11_000.0,
            "wind_fc_scale": 1.0, "wind_fc_dir": 0.0}


# ---------------------------------------------------------------------------
# one case


def dispersed_separation_state(sep, case: dict) -> np.ndarray:
    """Truth separation state vector for a case."""
    s = copy.copy(sep)
    s.altitude = sep.altitude + case["d_alt"]
    s.downrange = sep.downrange + case["d_downrange"]
    s.speed = sep.speed + case["d_speed"]
    s.flight_path_angle = sep.flight_path_angle + case["d_fpa"]
    s.prop_remaining = sep.prop_remaining + case["d_prop"]
    x = s.state_vector()
    x[1] += case["d_crossrange"]
    # heading error: rotate the horizontal velocity about the vertical
    hd = np.radians(case["d_heading"])
    vx, vy = x[3], x[4]
    x[3] = vx * np.cos(hd) - vy * np.sin(hd)
    x[4] = vx * np.sin(hd) + vy * np.cos(hd)
    x[D6.W_SLICE] = np.radians(np.asarray(case["d_rate"], float))
    return x


def run_case(case: dict, dt: float = 0.02, keep_trajectory: bool = False,
             log_every: int = 10) -> dict:
    """Fly one dispersed mission and score it. Self-contained, so it can be
    mapped over a process pool."""
    import run_mission as RM   # the mission definition lives at top level

    t0 = time.time()
    vehicle, aero, sep, bb0, land, seq = RM.build()

    # --- truth plant ------------------------------------------------------
    truth_v = copy.deepcopy(vehicle)
    truth_v.dry_mass *= case["k_dry"]
    truth_v.thrust_per_engine *= case["k_thrust"]
    truth_v.isp_sl *= case["k_isp"]
    truth_v.isp_vac *= case["k_isp"]
    wind = WindProfile(surface_speed=case["wind_surface"],
                       jet_speed=case["wind_jet"],
                       jet_altitude=case["wind_jet_alt"],
                       direction_surface=case["wind_dir"],
                       veer=case["wind_veer"])
    truth_a = AeroModel(cd_scale=case["k_cd"], density_scale=case["k_rho"],
                        wind=wind)

    seq.vehicle = truth_v
    seq.aero = truth_a
    seq.attitude.vehicle = truth_v
    # Guidance stays on the nominal vehicle and drag model. Its wind is the
    # day-of-launch FORECAST: the truth profile with a magnitude and
    # direction error, as from a balloon sounding hours before flight.
    seq.gnc_vehicle = vehicle
    fc_k = case.get("wind_fc_scale", 1.0)
    fc_rot = np.radians(case.get("wind_fc_dir", 0.0))
    c_, s_ = np.cos(fc_rot), np.sin(fc_rot)

    def forecast(h, _w=wind):
        w = _w(h) * fc_k
        return np.array([c_ * w[0] - s_ * w[1], s_ * w[0] + c_ * w[1], 0.0])

    seq.gnc_aero = AeroModel(wind=forecast)

    x0 = dispersed_separation_state(sep, case)
    sep_truth = copy.copy(sep)
    sep_truth.state_vector = lambda x0=x0: x0.copy()

    out = RM.fly(RM.T33_BURN, vehicle, aero, sep_truth, bb0, land, seq,
                 dt=dt, log_every=log_every)
    rep = catch_report(out["state"][-1], RM.TARGET)

    row = {k: v for k, v in case.items() if k != "d_rate"}
    row.update({f"d_rate_{a}": case["d_rate"][i]
                for i, a in enumerate("xyz")})
    row.update({k: float(v[0]) for k, v in rep["checks"].items()})
    row.update({f"ok_{k}": bool(v[1]) for k, v in rep["checks"].items()})
    row["caught"] = bool(rep["caught"])
    row["prop_left_t"] = rep["propellant_remaining_t"]
    ev = out.get("landing_events", {})
    row.update({k: float(v) for k, v in ev.items()})
    row["flight_time"] = float(out["t"][-1])
    row["apogee_km"] = float(out["r"][:, 2].max() / 1000.0)
    ground = any("GROUND" in e[1] for e in out["events"])
    row["ground_contact"] = bool(ground)

    seg = np.array(out["segment"])
    land_mask = seg == "landing"
    if land_mask.any():
        row["max_tilt_landing"] = float(out["tilt"][land_mask].max())
        row["max_throttle_landing"] = float(out["throttle"][land_mask].max())
    # where the vehicle crossed the ignition altitude, relative to the tower
    if np.isfinite(row.get("ignition_t", np.nan)):
        i = int(np.searchsorted(out["t"], row["ignition_t"]))
        i = min(i, len(out["t"]) - 1)
        row["ignition_x"] = float(out["r"][i, 0] - RM.TARGET[0])
        row["ignition_y"] = float(out["r"][i, 1] - RM.TARGET[1])
    hist = out.get("boostback_history", [])
    if hist:
        row["bb_yaw_deg"] = float(np.degrees(hist[-1][2]))
        row["bb_resolves"] = len(hist)
    row["runtime_s"] = time.time() - t0

    res = {"row": row}
    if keep_trajectory:
        res["traj"] = {"t": out["t"], "r": out["r"], "v": out["v"],
                       "segment": out["segment"], "tilt": out["tilt"],
                       "throttle": out["throttle"], "n_lit": out["n_lit"],
                       "prop": out["prop"]}
    return res


def _worker(args):
    idx, case, dt = args
    try:
        res = run_case(case, dt=dt, keep_trajectory=True, log_every=10)
        res["row"]["case"] = idx
        # Keep only the last ~1500 m of trajectory for the overlay plots, plus
        # a coarse full-flight track.
        tr = res["traj"]
        r = tr["r"]
        full = slice(None, None, 10)
        low = r[:, 2] < 1500.0
        res["traj"] = {"full_r": r[full], "low_r": r[low],
                       "low_v": tr["v"][low], "low_t": tr["t"][low],
                       "low_tilt": tr["tilt"][low],
                       "low_throttle": tr["throttle"][low]}
        return res
    except Exception as exc:  # a crash is a failed case, not a lost campaign
        return {"row": {"case": idx, "caught": False, "error": repr(exc),
                        **{k: v for k, v in case.items() if k != "d_rate"}},
                "traj": None}


def run_campaign(n: int, seed: int = 1, spec: DispersionSpec | None = None,
                 workers: int = 4, dt: float = 0.02,
                 include_nominal: bool = True, progress=print) -> list[dict]:
    """Run ``n`` dispersed cases (plus the nominal, as case 0)."""
    import multiprocessing as mp

    spec = spec or DispersionSpec()
    rng = np.random.default_rng(seed)
    cases = [sample_case(rng, spec) for _ in range(n)]
    if include_nominal:
        cases = [nominal_case()] + cases
    jobs = [(i, c, dt) for i, c in enumerate(cases)]
    results = []
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
            results.append(res)
            row = res["row"]
            if progress:
                status = ("CAUGHT" if row.get("caught") else
                          ("ERROR " + row["error"] if "error" in row else "MISS"))
                progress(f"  [{k:3d}/{len(jobs)}] case {row['case']:3d} "
                         f"{status:7s} lat {row.get('lateral_error_m', np.nan):7.2f} m"
                         f"  vh {row.get('horizontal_speed_ms', np.nan):5.2f}"
                         f"  tilt {row.get('tilt_deg', np.nan):5.2f}"
                         f"  prop {row.get('prop_left_t', np.nan):5.1f} t"
                         f"  ({time.time() - t0:5.0f} s)")
    results.sort(key=lambda r: r["row"]["case"])
    return results


def spec_dict(spec: DispersionSpec) -> dict:
    return asdict(spec)
