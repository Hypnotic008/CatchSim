import json

import numpy as np
import pytest

from quatsim.aero import GridFinModel, GridFinSet, fin_radial, wind_angles
from quatsim import quaternion as Q


def test_three_fins_give_independent_pitch_yaw_roll():
    B = GridFinSet().matrix(2.0, 20e3)
    assert np.linalg.matrix_rank(B) == 3
    # rudder (+z): yaw and roll, no pitch; catch pair: pitch and roll, no yaw
    assert abs(B[1, 0]) < 1e-6 and abs(B[2, 1]) < 1e-6 and abs(B[2, 2]) < 1e-6
    assert fin_radial(0.0) == pytest.approx([0, 0, 1])
    assert fin_radial(90.0) == pytest.approx([0, -1, 0])


def test_placeholder_pitch_authority_matches_the_old_fin_model():
    for mach, q in ((0.6, 8e3), (1.0, 25e3), (2.5, 15e3)):
        assert GridFinSet().max_torque(mach, q) == pytest.approx(GridFinModel().max_torque(mach, q))


def test_allocation_hits_the_torque_and_respects_limits():
    fs = GridFinSet()
    tau = np.array([1e5, -4e5, 2e5])
    d, t = fs.allocate(tau, 1.5, 20e3, np.zeros(3), dt=10.0)
    assert np.all(np.abs(d) <= fs.deflection_limit + 1e-12)
    assert t == pytest.approx(tau, rel=1e-6)
    # too much: scaled back as a whole, same torque direction
    d, t = fs.allocate(100 * tau, 1.5, 20e3, np.zeros(3), dt=10.0)
    assert np.max(np.abs(d)) == pytest.approx(fs.deflection_limit)
    assert t / np.linalg.norm(t) == pytest.approx(tau / np.linalg.norm(tau), abs=1e-9)
    # rate limit
    d, _ = fs.allocate(100 * tau, 1.5, 20e3, np.zeros(3), dt=0.02)
    assert np.max(np.abs(d)) == pytest.approx(fs.rate_limit * 0.02)


def test_fins_hold_neutral_in_near_vacuum():
    fs = GridFinSet()
    d, t = fs.allocate([0, 1e6, 0], 3.0, 50.0, np.zeros(3), dt=10.0)
    assert not np.any(d) and not np.any(t)


def test_cfd_derivatives_load_and_drive_the_matrix(tmp_path):
    rows = lambda dM: [dict(mach=m, aoa=a, roll=r, dF=[0, 0, 0], dM=[v * m for v in dM], dC_h=None)
                       for m in (0.8, 2.0) for a in (0.0, 8.0) for r in (0.0, 90.0)]
    d = {"fins": {"rudder fin": rows([0.01, 0.0, -0.02]),
                  "catch fin 90": rows([0.01, 0.02, 0.0]),
                  "catch fin 270": rows([0.01, -0.02, 0.0])}}
    p = tmp_path / "fin_control.json"
    p.write_text(json.dumps(d))
    fs = GridFinSet().load_cfd(str(p))
    q = 10e3
    B = fs.matrix(1.4, q, aoa_deg=7.0, roll_deg=80.0)
    k = q * fs.S_ref * fs.L_ref * 1.4                    # dM scales with Mach in the table
    assert B[:, 0] == pytest.approx(k * np.array([0.01, 0.0, -0.02]))
    assert B[:, 2] == pytest.approx(k * np.array([0.01, -0.02, 0.0]))


def test_wind_angles_engines_first():
    q = Q.from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(-90.0))  # body x -> +z: nose up
    aoa, _ = wind_angles(q, np.array([0.0, 0.0, -300.0]))               # falling, engines first
    assert aoa == pytest.approx(0.0, abs=1e-6)
