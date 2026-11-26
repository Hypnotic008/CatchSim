import numpy as np
import pytest

from quatsim.tower import CatchTower, TOWER_FACING_DEG, current_tower


def test_tower2_stands_north_with_arms_facing_ssw():
    t = CatchTower.numbered(2)
    px, py = t.pad
    assert py > 25.0 and abs(px) < 10.0           # ~30 m north of the catch point
    f = t.facing
    assert f[1] < -0.95 and f[0] < 0.0             # arms open south-southwest
    assert TOWER_FACING_DEG[1] == pytest.approx(90.0)


def test_tower1_reproduces_the_old_ocean_facing_geometry():
    t = CatchTower.numbered(1)
    assert t.pad == pytest.approx((-30.0, 0.0), abs=1e-9)
    # envelope: 3 m toward/away from the tower (x), 5 m across (y)
    assert t.compensated_error([2.9, 4.9, 105.0], [0, 0, 105.0])[2]
    assert not t.compensated_error([3.2, 0.0, 105.0], [0, 0, 105.0])[2]


def test_envelope_rotates_with_the_tower():
    t = CatchTower.numbered(2)
    tgt = np.array([0.0, 0.0, 105.0])
    across = tgt + 4.9 * t.across                 # within the 5 m across travel
    along = tgt + 3.5 * t.facing                  # outside the 3 m along travel
    assert t.compensated_error(across, tgt)[2]
    assert not t.compensated_error(along, tgt)[2]
    d = t.deadband(np.array([0.0, 0.0]))
    assert np.allclose(d, 0.0)


def test_clearance_flags_a_booster_crossing_the_arms():
    t2, t1 = CatchTower.numbered(2), CatchTower.numbered(1)
    # 10 m east of the catch point at arm height: on Tower 2's open east arm,
    # but inside Tower 1's open mouth
    p = np.array([10.0, 0.0, 150.0])
    assert t2.clearance(p) < 0.0 < t1.clearance(p)
    # centred: clear of both, by the open half-gap less the hull radius
    assert t2.clearance([0.0, 0.0, 150.0]) > 4.0
    # far above the arms: nothing to hit
    assert t2.clearance([10.0, 0.0, 600.0]) == float("inf")


def test_current_tower_reads_the_environment(monkeypatch):
    monkeypatch.setenv("CATCHSIM_TOWER", "1")
    assert current_tower().facing_deg == 90.0
    monkeypatch.setenv("CATCHSIM_TOWER", "2")
    assert current_tower().facing_deg == 190.0


def test_catch_report_scores_roll_against_the_tower_axis():
    from quatsim import dynamics6dof as D6
    from quatsim.guidance import attitude_from_pointing
    from quatsim.phases import catch_report
    t = CatchTower.numbered(2)
    q = attitude_from_pointing(np.array([0.0, 0.0, 1.0]), roll_reference=t.facing)
    s = D6.make_state([0.0, 0.0, 105.0], [0.0, 0.0, -0.3], q, np.zeros(3), 60e3)
    good = catch_report(s, [0, 0, 105.0], catch_axis=t.facing)
    bad = catch_report(s, [0, 0, 105.0])          # old +x axis
    assert good["checks"]["fin_roll_deg"][0] < 1.0
    assert bad["checks"]["fin_roll_deg"][0] > 80.0
