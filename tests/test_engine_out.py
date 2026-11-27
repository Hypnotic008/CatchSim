"""Landing-burn engine failures: the truth engine count and what the guidance plans on."""
from types import SimpleNamespace

import numpy as np
import pytest

from quatsim.landing import LandingConfig, LandingGuidance
from quatsim.phases import FlightSequencer
from quatsim.vehicle import Vehicle
from quatsim.aero import AeroModel


def _seq(relight=0, shutdown=-1.0, t=0.0):
    veh = Vehicle()
    veh.landing_relight_failures = relight
    veh.landing_shutdown_t = shutdown
    return SimpleNamespace(vehicle=veh, _eng_cap=None, _eng_shutdown_done=False, _eng_events=[], _t_now=t)


def _lg(ign_t=0.0):
    lg = LandingGuidance(LandingConfig(), Vehicle(), AeroModel())
    lg.ignition_t = ign_t
    return lg


def test_relight_failures_cap_the_engines_and_inform_the_guidance():
    s, lg = _seq(relight=2), _lg()
    assert FlightSequencer._landing_engines(s, lg, 13) == 11
    assert lg.n_healthy == 11
    # the 5- and 3-engine stages use any healthy engine: unaffected by two failures
    assert FlightSequencer._landing_engines(s, lg, 5) == 5
    assert FlightSequencer._landing_engines(s, lg, 3) == 3
    assert any("failed to relight" in e[1] for e in s._eng_events)


def test_in_flight_shutdown_takes_out_one_lit_engine_for_good():
    s, lg = _seq(shutdown=4.0, t=1.0), _lg(ign_t=0.0)
    assert FlightSequencer._landing_engines(s, lg, 3) == 3       # before the failure
    s._t_now = 5.0
    assert FlightSequencer._landing_engines(s, lg, 3) == 2       # one of the three is gone
    s._t_now = 9.0
    assert FlightSequencer._landing_engines(s, lg, 5) == 2       # shut-down / unlit ones stay off
    assert lg.n_healthy == 2


def test_guidance_thrust_counts_only_available_engines():
    lg = _lg()
    full = lg.vehicle.axial_thrust(13, 1.0)
    assert lg._thrust(13) == pytest.approx(full)
    lg.n_healthy = 10
    assert lg._thrust(13) == pytest.approx(lg.vehicle.axial_thrust(10, 1.0))
    assert lg._thrust(3) == pytest.approx(lg.vehicle.axial_thrust(3, 1.0))
