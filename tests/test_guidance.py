"""Tests for the guidance additions: landing law, aero normal force, entry
steering geometry, boostback predictor continuity, wind profile."""

import numpy as np
import pytest

from quatsim import dynamics6dof as D6
from quatsim.aero import AeroModel
from quatsim.entry import EntryConfig, EntryGuidance
from quatsim.guidance import attitude_from_pointing
from quatsim.landing import LandingConfig, LandingGuidance, poly_accel
from quatsim.montecarlo import WindProfile
from quatsim.vehicle import Vehicle


def test_poly_accel_meets_boundary_conditions():
    """Integrating the quadratic profile that poly_accel starts must land on
    (r_f, v_f, a_f) at T."""
    r0, v0 = np.array([120.0, -40.0]), np.array([-15.0, 3.0])
    rf, vf, af = np.array([0.0, 0.0]), np.zeros(2), np.zeros(2)
    T = 12.0
    a0 = poly_accel(r0, v0, rf, vf, af, T)
    # recover the full profile a(t) = a0 + c1 t + c2 t^2 from the same BCs
    A = np.array([[T ** 2 / 2, T ** 3 / 3], [T ** 3 / 6, T ** 4 / 12],
                  [T, T ** 2]])
    for j in range(2):
        rhs = np.array([vf[j] - v0[j] - a0[j] * T,
                        rf[j] - r0[j] - v0[j] * T - a0[j] * T ** 2 / 2,
                        af[j] - a0[j]])
        c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        assert np.allclose(A @ c, rhs, atol=1e-9)


def test_normal_force_sign_and_zero_at_zero_alpha():
    aero = AeroModel()
    v = np.array([0.0, 0.0, -400.0])
    up = attitude_from_pointing([0.0, 0.0, 1.0])
    assert np.allclose(aero.normal_force(up, v, 5000.0), 0.0, atol=1e-6)
    tilt = np.radians(3.0)
    q = attitude_from_pointing([np.sin(tilt), 0.0, np.cos(tilt)])
    f = aero.normal_force(q, v, 5000.0)
    # nose toward +x while falling tail-first pushes the vehicle toward -x
    assert f[0] < 0.0 and abs(f[1]) < 1e-6


def test_wind_profile_shape():
    w = WindProfile(surface_speed=8.0, jet_speed=40.0, jet_altitude=11_000.0,
                    direction_surface=0.0, veer=0.0)
    assert np.linalg.norm(w(10.0)) == pytest.approx(8.0, rel=1e-6)
    assert np.linalg.norm(w(11_000.0)) == pytest.approx(40.0, rel=1e-3)
    assert np.linalg.norm(w(60_000.0)) < 1.0


def _landing(state_r, state_v, prop=120e3):
    veh, aero = Vehicle(prop_mass=500e3), AeroModel()
    lg = LandingGuidance(LandingConfig(), veh, aero)
    q = attitude_from_pointing(-np.asarray(state_v, float))
    return lg, lg.step(state_r, state_v, q, prop, 0.0)


def test_landing_holds_fire_high_and_ignites_on_energy():
    lg, cmd = _landing([0.0, 0.0, 4000.0], [0.0, 0.0, -300.0])
    assert cmd.phase == "coast" and cmd.throttle == 0.0
    lg, cmd = _landing([0.0, 0.0, 900.0], [0.0, 0.0, -300.0])
    assert cmd.phase == "brake" and cmd.n_lit == 13


def test_landing_lateral_command_points_at_target_and_respects_tilt():
    lg, cmd = _landing([300.0, -200.0, 900.0], [0.0, 0.0, -300.0])
    # thrust direction leans back toward the target (-x, +y)
    assert cmd.direction[0] < 0.0 and cmd.direction[1] > 0.0
    tilt = np.degrees(np.arccos(cmd.direction[2]))
    assert tilt <= np.degrees(lg.cfg.max_tilt_brake) + 1e-6
    assert lg.cfg.throttle_floor <= cmd.throttle <= 1.0


def test_landing_planar_state_gives_planar_command():
    _, cmd = _landing([250.0, 0.0, 900.0], [-50.0, 0.0, -300.0])
    assert abs(cmd.direction[1]) < 1e-12


def test_terminal_flare_does_not_hold_a_stopped_vehicle_up():
    """Regression: the flare feed-forward once held a vehicle 0.1 m above the
    catch plane until the tanks ran dry."""
    lg, _ = _landing([0.0, 0.0, 900.0], [0.0, 0.0, -300.0])
    lg.phase, lg.n_lit, lg.handover_alt, lg.handover_t = "terminal", 3, 250.0, 0.0
    q = attitude_from_pointing([0.0, 0.0, 1.0])
    cmd = lg.step([0.0, 0.0, 105.1], [0.0, 0.0, 0.0], q, 60e3, 1.0)
    assert cmd.info["a_z"] < 0.0            # asks to descend, not to climb


def test_entry_tilts_away_from_desired_force():
    veh, aero = Vehicle(prop_mass=500e3), AeroModel()
    eg = EntryGuidance(EntryConfig(aim=np.array([-500.0, 0.0])), veh, aero,
                       None)
    r, v = np.array([0.0, 0.0, 8000.0]), np.array([0.0, 0.0, -450.0])
    q_ref, info = eg.command(r, v, attitude_from_pointing([0, 0, 1.0]),
                             120e3, 0.0)
    assert q_ref is not None and info["alpha_deg"] > 0.0
    f = aero.normal_force(q_ref, v, r[2])
    assert f[0] < 0.0                       # force toward the aim point


def test_boostback_predictor_is_continuous_in_cutoff_time():
    """Regression: engine-stage switches were evaluated only at step starts,
    making the predicted arrival jump by ~4 km between neighbouring cutoffs."""
    from quatsim.mission import (SeparationState, predict_boostback_arrival_rk4,
                                 propagate_flip)
    veh, aero = Vehicle(prop_mass=500e3), AeroModel()
    sep = SeparationState(altitude=68e3, downrange=83e3, speed=1500.0,
                          flight_path_angle=25.0, prop_remaining=500e3)
    s0 = propagate_flip(veh, aero, sep, 4.0)
    q = s0[D6.Q_SLICE]
    xs = [predict_boostback_arrival_rk4(veh, aero, s0, q, rem,
                                        throttle=0.87)[0][0]
          for rem in (9.10, 9.13, 9.16)]
    d1, d2 = xs[1] - xs[0], xs[2] - xs[1]
    assert d1 < 0 and d2 < 0                          # monotone
    assert abs(d1 - d2) < 0.05 * abs(d1)              # and smooth


def test_engine_layout_matches_webcast_diagram():
    """Centre 3 as an inverted triangle; the 5-engine set is the centre 3
    plus the two middle-ring engines at 3 and 9 o'clock."""
    from quatsim.visuals import engine_layout
    pos, lit = engine_layout()
    assert pos.shape == (33, 2)
    c = pos[:3]
    assert sum(c[:, 1] > 0) == 2 and sum(c[:, 1] < 0) == 1     # two up, one down
    five = pos[lit[5]]
    extra = [p for p in five if np.hypot(*p) > 0.4]
    assert len(extra) == 2
    for p in extra:
        assert abs(p[1]) < 1e-9 and abs(abs(p[0]) - 0.6) < 1e-9  # 3 and 9 o'clock
    assert extra[0][0] * extra[1][0] < 0                         # opposite sides
    assert lit[3] == [0, 1, 2] and len(lit[13]) == 13 and len(lit[33]) == 33


def test_export3d_frame_mapping_and_schedule():
    from quatsim import quaternion as Q
    from quatsim.export3d import M_ENU_TO_THREE, Q_M, playback_schedule
    q = Q.normalize(np.array([0.3, 0.5, -0.2, 0.7]))
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(Q.rotate(Q.multiply(Q_M, q), v),
                       M_ENU_TO_THREE @ Q.rotate(q, v))
    # synthetic mission: boostback, long coast, approach, landing
    t = np.arange(0.0, 300.0, 0.1)
    alt = np.where(t < 20, 70e3, np.where(t < 250, 70e3 - (t - 20) * 290,
                                          np.maximum(4000 - (t - 250) * 150, 105)))
    seg = np.where(t < 20, "boostback_33", np.where(t < 262, "coast", "landing"))
    out = {"t": t, "r": np.column_stack([0 * t, 0 * t, alt]), "segment": list(seg),
           "n_lit": np.where(t < 20, 33, np.where(t < 262, 0, 13)),
           "throttle": np.where((t < 20) | (t >= 262), 0.8, 0.0)}
    ft, fr = playback_schedule(out, fps=30, hold=2.0)
    assert np.all(np.diff(ft) >= -1e-9)
    assert fr[np.searchsorted(ft, 10.0)] == pytest.approx(1.0, abs=0.05)   # boostback 1x
    assert fr[np.searchsorted(ft, 120.0)] == pytest.approx(12.0, rel=0.05)  # coast 12x
    assert fr[np.searchsorted(ft, 280.0)] == pytest.approx(1.0, abs=0.05)  # landing 1x
