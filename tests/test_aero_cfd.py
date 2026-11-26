"""The rocketcfd drop-in (quatsim/aero_cfd.py) wired into CatchSim, checked against a synthetic
aero_db.json in the exporter's format that mimics CatchSim's own estimates."""
import json

import numpy as np
import pytest

from quatsim import aero as AERO
from quatsim import quaternion as Q
from quatsim.aero_cfd import CFDAeroDatabase, CFDAeroModel


def synthetic_db(path, cn_alpha=2.0, cm_alpha=0.35, fin_factor=2.0):
    mach = [0.6, 0.8, 0.95, 1.1, 1.3, 1.6, 2.0, 2.6, 3.2, 4.0]
    aoa = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    roll = [0.0, 90.0, 180.0]
    T = {k: [] for k in ("CX", "CN", "CS", "Cl", "C_cross", "C_restore", "CD")}
    for m in mach:
        rows = {k: [] for k in T}
        cd = AERO.drag_coefficient(m) * fin_factor
        for a in aoa:
            sa, ca = np.sin(np.radians(a)), np.cos(np.radians(a))
            # CatchSim's estimate: drag CD along the wind (cos a ex + sin a cross) and a
            # normal force cn sin a normal to it (-sin a ex + cos a cross), in table axes
            cl = cn_alpha * sa
            vals = dict(CX=cd * ca - cl * sa, CN=cd * sa + cl * ca, CS=0.0, Cl=0.0, C_cross=0.0,
                        C_restore=cm_alpha * sa, CD=cd)
            for k in T:
                rows[k].append([vals[k]] * len(roll))
        for k in T:
            T[k].append(rows[k])
    d = dict(reference=dict(S_ref=63.6, L_ref=72.3, x_ref=[0.4 * 72.3, 0, 0]),
             fins=dict(names=[], mirror_swap={}), mach=mach, aoa_deg=aoa, roll_deg=roll,
             tables=T, mirror_sign=dict(CX=1, CN=1, CS=-1, Cl=-1, C_cross=-1, C_restore=1, CD=1),
             cm_q_quasi_steady=None)
    with open(path, "w") as fh:
        json.dump(d, fh)
    return path


@pytest.fixture
def model(tmp_path):
    return CFDAeroModel(db=CFDAeroDatabase(synthetic_db(tmp_path / "aero_db.json")))


def test_drag_matches_the_estimate_it_was_built_from(model):
    v = np.array([0.0, 0.0, -600.0])                 # falling at 600 m/s
    # equal up to the reference area (63.6 m^2 in the table, pi 4.5^2 = 63.617 in CatchSim)
    assert model.drag_force(v, 15_000.0) == pytest.approx(AERO.AeroModel().drag_force(v, 15_000.0), rel=1e-3)


def test_airframe_moment_matches_the_weathercocking_model(model):
    # engines-first with 8 deg of incidence in pitch: same restoring moment as the estimate
    q = Q.from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(-90.0 + 8.0))
    v = np.array([0.0, 0.0, -500.0])
    m_cfd = model.airframe_moment(q, v, 12_000.0)
    m_est = AERO.restoring_moment(q, v, 12_000.0, cm_q=0.0)
    assert np.linalg.norm(m_cfd) > 1e5
    assert m_cfd == pytest.approx(m_est, rel=0.02, abs=1e3)


def test_beyond_the_table_crossflow_grows_like_sin_alpha(tmp_path):
    db = CFDAeroDatabase(synthetic_db(tmp_path / "db.json"))
    c20 = db.coefficients(2.0, 20.0, 0.0)
    c30 = db.coefficients(2.0, 30.0, 0.0)
    k = np.sin(np.radians(30.0)) / np.sin(np.radians(20.0))
    assert c30["C_restore"] == pytest.approx(c20["C_restore"] * k)
    assert c30["CN"] == pytest.approx(c20["CN"] * k)
    assert c30["CX"] == pytest.approx(c20["CX"] * np.cos(np.radians(30)) / np.cos(np.radians(20)))
