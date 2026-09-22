"""
Flight sequencer -- runs a phase schedule end to end.

WHAT THIS MODULE IS FOR
-----------------------
Everything up to now has been a component: quaternion algebra, rigid-body
dynamics, an attitude loop, a position loop. This is the thing that decides
WHICH of them is running at any moment, what the engines are doing, and when
to hand off.

Three jobs that nothing else does:

  1. ENGINE SEQUENCING. Which engines are lit, how many gimbal, what throttle.
     Transitions are discrete events, and the inertia and control authority
     both jump across them.

  2. PROPELLANT EXHAUSTION. dynamics6dof deliberately does NOT cut thrust when
     the tanks run dry -- it is the sequencer's job. Getting this wrong
     produces a vehicle flying on an empty tank, which looks like a
     beautifully converged solution and is worthless. Ask how I know.

  3. CONTROL MODE. During burns the attitude loop tracks a guidance reference.
     During the terminal phase a position loop wraps around it and commands the
     attitude itself. During the unpowered coast there is no gimbal authority
     at all and the only actuators are the three grid fins -- which have no
     roll backup whatsoever, since roll authority scales with the body radius
     rather than the moment arm and is 4-5x weaker than pitch or yaw.

THE LANDING BURN IS SOLVED, NOT SCHEDULED
-----------------------------------------
Every other phase has a duration you specify. The landing burn does not: you
specify the terminal condition (zero velocity at the catch altitude) and
solve_ignition_altitude finds where to light the engines. That is the first
number in this project the simulation produces rather than consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import dynamics6dof as D6
from . import quaternion as Q
from . import environment as ENV
from . import aero as AERO
from .aero import AeroModel
from .control import AttitudeController
from .guidance import attitude_from_pointing
from .landing import LandingConfig, LandingGuidance
from .position import PositionController, divert_profile
from .vehicle import Vehicle


class Mode:
    """Control mode for a phase."""
    ATTITUDE = "attitude"     # inner loop only, tracking a guidance reference
    POSITION = "position"     # outer position loop commands the attitude
    COAST = "coast"           # no engines; RCS + grid fins
    LANDING = "landing"       # closed-loop suicide burn, ignition is TRIGGERED


@dataclass
class Segment:
    """One scheduled segment of flight."""
    name: str
    mode: str
    n_lit: int = 0
    n_gimballing: int = 0
    throttle: float = 0.0
    duration: float = 0.0
    q_ref: np.ndarray | None = None          # ATTITUDE mode: constant target
    # ATTITUDE mode: a guidance.Slew to TRACK instead of a constant to step to.
    # Prefer this for any large reorientation. A constant q_ref hands the
    # controller the whole error at once, which saturates the gimbals for the
    # entire manoeuvre and leaves the loop running open; a slew feeds a
    # feasible reference plus its rate and acceleration, keeping the error
    # small and the actuators off their stops.
    q_slew: object | None = None
    # COAST mode: hold engines-first, body +x opposite the velocity vector.
    # This is the descent attitude -- the booster falls engine end forward.
    retrograde_hold: bool = False
    # LANDING mode: altitude at which the burn must bring the vehicle to rest.
    catch_altitude: float = 105.0
    throttle_floor: float = 0.40
    # Engine counts the landing burn may downselect through, most to fewest.
    # 13 -> 3. No 2-engine stage: it would run for only a couple of seconds
    # and 3 engines can null the remaining velocity on their own.
    downselect: tuple = (13, 3)
    # Fractions of the ignition-to-catch altitude band at which to step down.
    # (0.25, 0.08) with downselect (13, 3, 2) means: 13 engines for the top
    # 75% of the descent, 3 below that, 2 for the last 8%.
    downselect_margin: float = 0.60   # demand must fit under the new count
    tower: object | None = None       # CatchTower; enables position deadband
    # Fraction of hover thrust held as a floor during the burn. 0 = no floor
    # (thrust can go to zero, killing lateral authority); 1 = full hover.
    # Terminal (3-engine) phase: constant descent rate, easing to v_touch
    # over the last terminal_ease metres.
    v_terminal_descent: float = 6.0   # m/s
    # Altitude at which the 13-engine brake hands over to the 3-engine final
    # approach. Flight 7 timing (~7 s on 13, ~18 s on 3) implies the handover
    # happens well above the catch with plenty of runway left.
    handover_altitude: float = 180.0
    # Enable the two-phase (brake / precision-approach) vertical profile.
    # The validated runnable configuration enables this explicitly; keeping the
    # field false here preserves a clean single-phase option for comparisons.
    two_phase: bool = False
    # Solve the braking deceleration from the terminal conditions instead of
    # taking a fraction of available thrust. Throttle becomes an output.
    solve_brake: bool = False
    # Optimal terminal guidance in the 3-engine phase: drives lateral position
    # AND velocity to zero together, instead of nulling position and arriving
    # leaning.
    terminal_guidance: bool = True
    # Fraction of available deceleration at which the brake lights. <1 leaves
    # margin, so the throttle starts below 100% and has room to push harder.
    brake_margin: float = 0.85
    terminal_ease: float = 40.0       # m
    # Terminal-only tilt taper. The main landing taper is for the long brake
    # and must not consume the lateral authority reserved for the 3-engine
    # precision phase. The terminal phase gets its own 180 -> catch envelope.
    terminal_taper_altitude: float = 75.0  # m
    terminal_upright_height: float = 5.0  # m above catch plane
    # The nominal 40% Raptor floor is retained for the main burn. Three
    # engines cannot quite hover at that floor in this V3 mass model, so the
    # precision phase may use a lower effective floor to represent the
    # unresolved real minimum-throttle behavior rather than bouncing above the
    # catch plane. This is an assumption, not a measured engine limit.
    terminal_throttle_floor: float = 0.34
    hover_floor: float = 0.0
    # LANDING guidance tuning
    # COAST: exit when the vehicle drops below this altitude, instead of when
    # a fixed duration expires. Every other duration in this simulation is
    # either solved or measured; a clocked coast was the last one still running
    # on a guessed number, and it was flying the vehicle into the ground before
    # the landing segment ever started.
    exit_altitude: float | None = None
    # COAST steering: bias the engines-first attitude to aim the impact point.
    steer_target: np.ndarray | None = None
    k_steer: float = 2e-5        # rad of offset per metre of lateral error
    steer_limit: float = 0.30    # max offset fraction
    design_frac: float = 0.20   # fraction of available decel the profile uses
    v_touch: float = 1.0        # m/s, terminal descent rate at the target
    k_v: float = 0.9            # velocity-error gain
    k_lat: float = 0.9          # lateral velocity-reference tracking gain
    k_pos: float = 0.02         # lateral POSITION gain (legacy, unused by
                                # the braking profile)
    # Terminal lateral velocity the guidance aims for. NOT zero: the catch
    # interface is a pair of rails, which can absorb some sliding. Demanding a
    # mathematically stationary vehicle at contact is a harder problem than the
    # hardware actually poses. THIS IS A SIMULATION ASSUMPTION, not a validated
    # structural limit -- establishing what the arms can really absorb is an
    # interface and structural-dynamics question.
    v_lat_terminal: float = 0.5  # m/s
    # Desired horizontal speed at the 13 -> 3 handover. Zero is the natural
    # target for the precision phase: the three-engine controller should not
    # inherit a large lateral velocity and then spend its short final runway
    # braking it.
    v_lat_handover: float = 0.5   # m/s
    use_altitude_ceiling: bool = False
    # Fraction of available lateral authority the braking profile designs to.
    # Too high and v_ref is large near the target: the vehicle arrives fast and
    # overshoots, then has to chase back. Measured at 0.35 the lateral error
    # converged to 80 m and then DIVERGED to 330 m.
    lat_design_frac: float = 0.15
    # Lean limit during the landing burn. 30 deg, not 15.
    #
    # THIS IS THE BINDING CONSTRAINT, and it was mistaken for a gain problem
    # for a long time. The booster returns to the pad at ~188 m/s horizontally
    # -- that is not an error, it is the return velocity -- and killing it over
    # the ~855 m available needs v^2/2d = 20.7 m/s^2 of lateral deceleration.
    # At a_vert ~ 30 m/s^2 that is arctan(20.7/30) = 34.6 deg of tilt.
    #
    # Capped at 15 deg the command saturated permanently, which is why sweeping
    # k_lat and lat_design_frac produced IDENTICAL results to a tenth of a
    # metre across 7x gain changes: the actuator was pegged and the gains were
    # not reaching it.
    #
    # Raising the cap turns the burn-duration basin from a knife edge into a
    # bowl: across +/-0.016 s the lateral error spread falls from ~800 m to
    # ~280 m, and the central durations land between 14 and 74 m instead of
    # between 8 and 533 m.
    max_tilt: float = np.radians(20)
    taper_altitude: float = 600.0       # m above target where lean fades out
    r_target: np.ndarray | None = None       # POSITION mode target
    divert_duration: float = 0.0
    # Live predictive boostback cutoff. The controller predicts the future
    # boostback+tail+coast with the same RK4 plant before choosing cutoff.
    predictive_boostback: bool = False
    boostback_target_altitude: float = 1_200.0
    boostback_target_x: float | None = None
    boostback_reference_33: float = 9.1472
    boostback_tail13: float = 3.0
    boostback_tail3: float = 3.0
    boostback_target_y: float = 0.0
    # LANDING mode: guidance configuration (landing.LandingConfig). None
    # builds a default one aimed at r_target.
    landing: object | None = None
    boostback_steer_crossrange: bool = True
    # Re-solve the cutoff this often during the 33-engine burn, while more
    # than `boostback_resolve_min_remaining` seconds of burn remain.
    boostback_resolve_period: float = 2.5
    boostback_resolve_min_remaining: float = 1.5


@dataclass
class FlightLog:
    """Time histories, grown as a list and converted once at the end."""
    t: list = field(default_factory=list)
    state: list = field(default_factory=list)
    segment: list = field(default_factory=list)
    throttle: list = field(default_factory=list)
    tilt: list = field(default_factory=list)
    gimbal: list = field(default_factory=list)
    events: list = field(default_factory=list)
    # Reference signals. Logging these is not optional for a control project:
    # without them you can plot what the vehicle DID but never what it was
    # ASKED to do, and tracking error -- the thing the controller exists to
    # minimise -- is invisible.
    r_ref: list = field(default_factory=list)
    q_ref: list = field(default_factory=list)
    n_lit: list = field(default_factory=list)

    def arrays(self) -> dict:
        S = np.array(self.state)
        return {
            "t": np.array(self.t),
            "state": S,
            "r": S[:, D6.R_SLICE],
            "v": S[:, D6.V_SLICE],
            "q": S[:, D6.Q_SLICE],
            "omega": S[:, D6.W_SLICE],
            "prop": S[:, D6.P_INDEX],
            "segment": self.segment,
            "throttle": np.array(self.throttle),
            "tilt": np.array(self.tilt),
            "gimbal": np.array(self.gimbal),
            "events": self.events,
            "r_ref": np.array(self.r_ref),
            "q_ref": np.array(self.q_ref),
            "n_lit": np.array(self.n_lit),
        }


# ---------------------------------------------------------------------------
# landing burn targeting
# ---------------------------------------------------------------------------

def suicide_burn_needed(altitude: float, v_down: float, catch_altitude: float,
                        thrust: float, mass: float,
                        margin: float = 1.02, drag_decel: float = 0.0) -> bool:
    """
    Closed-loop landing burn trigger.

    Ignite when the distance still available is no longer more than the
    distance required to stop:

        v^2 / (2 * (a_thrust - g))  >=  h - h_catch

    THIS REPLACES A SCHEDULE WITH A MEASUREMENT, and that is the whole point.
    Precomputing an ignition time from an ASSUMED arrival speed works only if
    the vehicle arrives exactly as assumed. It never does -- the boostback,
    coast and aerodynamics all shift the arrival state -- so a scheduled burn
    ignites at the wrong moment and under-brakes. The vehicle then arrives at
    the catch altitude still doing tens of metres per second.

    Evaluated from the CURRENT state every step, the trigger is immune to all
    of that. It is the same lesson that fixed the boostback three times over:
    solve against what is actually there, not against a model of it.

    margin ignites slightly early, because igniting late is unrecoverable
    while igniting early only costs propellant.
    """
    # Drag is DECELERATING the vehicle too, and at these speeds it is not a
    # small term. Ignoring it makes the trigger conservative: it fires high,
    # the vehicle over-brakes, and the engines then push it back UP because
    # thrust exceeds weight. Counting drag puts ignition where it belongs.
    a_net = thrust / mass - ENV.G0 + drag_decel
    if a_net <= 0.0:
        return True                      # cannot stop at all; burn now
    h_available = altitude - catch_altitude
    if h_available <= 0.0:
        return True
    h_required = (v_down * v_down) / (2.0 * a_net)
    return bool(h_required * margin >= h_available)


def solve_ignition_altitude(
    vehicle: Vehicle,
    aero: AeroModel,
    prop_available: float,
    target_altitude: float,
    n_lit: int,
    throttle: float,
    v_entry: float | None = None,
    bracket: tuple[float, float] = (150.0, 6000.0),
    tol: float = 0.05,
) -> dict:
    """
    Find the altitude at which to light the landing burn.

    A one-dimensional shooting problem: propagate a vertical descent from a
    trial ignition altitude and see where the velocity reaches zero. Too low
    and the vehicle hits the ground still moving; too high and it arrives with
    altitude to spare and starts climbing, because at these thrust levels it
    cannot hover on the landing-burn engine count.

    Monotonic in the trial altitude, so bisection is reliable and needs no
    derivative. The vertical-only model ignores the lateral divert, which is
    fine because the divert happens after the burn, at zero vertical velocity.
    """
    lo, hi = bracket

    def shoot(h_ign: float) -> tuple[float, float, float]:
        """Returns (altitude at v=0, propellant used, burn duration)."""
        h = float(h_ign)
        prop = float(prop_available)
        v = -(aero.terminal_velocity(vehicle.mass(prop), h_ign)
              if v_entry is None else abs(v_entry))
        t, dt = 0.0, 0.002
        mdot = vehicle.mass_flow(n_lit, throttle, vacuum=False)

        while h > 0.0 and t < 120.0:
            m = vehicle.mass(prop)
            thrust = vehicle.axial_thrust(n_lit, throttle) if prop > 0.0 else 0.0
            drag = aero.drag_force(np.array([0.0, 0.0, v]), h)[2]
            a = thrust / m - ENV.G0 + drag / m
            v += a * dt
            h += v * dt
            t += dt
            if prop > 0.0:
                prop = max(prop - mdot * dt, 0.0)
            if v >= 0.0:
                return h, float(prop_available) - prop, t
        return h, float(prop_available) - prop, t

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        h_stop, _, _ = shoot(mid)
        if h_stop > target_altitude:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    h_ign = 0.5 * (lo + hi)
    h_stop, prop_used, duration = shoot(h_ign)
    return {
        "ignition_altitude": h_ign,
        "stop_altitude": h_stop,
        "burn_duration": duration,
        "prop_used": prop_used,
        "entry_speed": (aero.terminal_velocity(vehicle.mass(prop_available), h_ign)
                        if v_entry is None else abs(v_entry)),
        "feasible": prop_used < prop_available,
    }


# ---------------------------------------------------------------------------
# the sequencer
# ---------------------------------------------------------------------------

class FlightSequencer:
    """
    Walks a list of Segments, running the appropriate controller for each.

    The controllers are called once per integrator step here for simplicity.
    A real vehicle runs them at a fixed, slower rate; Forces exists so that
    change is a small one when you want it.
    """

    def __init__(self, vehicle: Vehicle, aero: AeroModel,
                 attitude: AttitudeController, position: PositionController,
                 spherical_gravity: bool = True,
                 fins=None):
        self.vehicle = vehicle
        self.aero = aero
        self.attitude = attitude
        self.position = position
        # MUST match whatever the targeting solve uses. An RTLS arc reaches
        # nearly 100 km, where flat-earth gravity is ~3% strong; over a 200 s
        # coast that is kilometres of altitude and tens of kilometres of range.
        # Solving the boostback under one gravity model and flying it under
        # another produced an 18 km disagreement in the predicted arrival
        # point -- a discrepancy that looks like a guidance failure and is
        # actually two models of the Earth.
        self.spherical_gravity = spherical_gravity
        # GridFinModel or None. Fins are the only aerodynamic control surface
        # the booster has, and the only three-axis actuator with real authority
        # once engines are out.
        self.fins = fins
        # Guidance-side models. By default guidance knows the truth plant
        # exactly; the Monte Carlo sets these to the NOMINAL models so that
        # thrust, Isp, drag and wind dispersions are genuinely unknown to the
        # predictor and the landing guidance, as they would be in flight.
        self.gnc_vehicle = vehicle
        self.gnc_aero = aero
        self._ignited = False
        self._burn_done = False
        self._segment_done = False
        self._lat_log = []
        self._t_now = 0.0
        self._landing_n_lit = 0
        self._a_brake = None
        self._ignition_alt = float("nan")
        self._ignition_v = float("nan")
        self._boostback_cutoff = False
        self._boostback_pred = None

    def run(self, state0: np.ndarray, segments: list[Segment],
            dt: float = 0.004, log_every: int = 5) -> dict:
        s = state0.copy()
        t = 0.0
        log = FlightLog()
        dry_reported = False
        step = 0          # counts every integrator step, NOT logged samples
        self._lat_log = []
        self._ignition_alt = float("nan")
        self._ignition_v = float("nan")
        self._boostback_cutoff = False
        self._boostback_pred = None
        self._boostback_pred_t = -1e9
        self._boostback_cutoff_tseg = None
        self._boostback_history = []
        self._landing = None
        self._guid_log = []
        self._bb_q_base = None
        self._bb_yaw = 0.0
        self._landing_n_lit = 0
        self._a_brake = None
        self._ignited = False
        self._burn_done = False
        self._segment_done = False
        self._lat_log = []
        self._t_now = 0.0
        self._landing_n_lit = 0
        self._ignition_alt = float("nan")
        self._ignition_v = float("nan")
        self._terminal_dispersion_applied = False

        for seg in segments:
            t_seg = 0.0
            r_seg_start = s[D6.R_SLICE].copy()
            log.events.append((t, f"begin {seg.name}"))

            # Whole steps plus one SHORT final step, so the segment lasts
            # exactly seg.duration. Rounding up to whole steps quantizes every
            # burn to the timestep -- at ~30 km of range per second of
            # boostback, a 0.05 s step is 1.5 km of miss distance, and a
            # targeting loop cannot resolve below a granularity the integrator
            # imposes on it.
            self._ignited = False
            self._segment_done = False
            n_steps = int(seg.duration / dt) if seg.duration > 0 else 0
            remainder = seg.duration - n_steps * dt if seg.duration > 0 else 0.0
            step_sizes = [dt] * n_steps
            if remainder > 1e-12:
                step_sizes.append(remainder)
            for _dt in step_sizes:
                r, v, q, w, prop = D6.unpack(s)

                # --- tanks dry: this is the sequencer's job, not dynamics' ---
                if prop <= 0.0 and seg.n_lit > 0:
                    if not dry_reported:
                        log.events.append((t, "PROPELLANT EXHAUSTED"))
                        dry_reported = True
                    thrust, torque, mdot, throttle = 0.0, np.zeros(3), 0.0, 0.0
                    gim = 0.0
                    r_ref_now, q_ref_now = r.copy(), q.copy()
                else:
                    (thrust, torque, mdot, throttle, gim,
                     r_ref_now, q_ref_now) = self._actuate(
                        seg, s, t_seg, r_seg_start
                    )

                # Optional terminal-state dispersion injection for robustness
                # testing.  Applied exactly once, at the instant the landing
                # burn has actually ignited, so it represents an off-nominal
                # state entering the 3-engine precision guidance.  Disabled
                # unless the caller sets seq.terminal_dispersion.
                if (seg.mode == Mode.LANDING and self._ignited
                        and not self._terminal_dispersion_applied
                        and getattr(self, "terminal_dispersion", None) is not None):
                    disp = self.terminal_dispersion
                    dr = np.asarray(disp.get("dr", [0.0, 0.0, 0.0]), dtype=float)
                    dv = np.asarray(disp.get("dv", [0.0, 0.0, 0.0]), dtype=float)
                    s[D6.R_SLICE] += dr
                    s[D6.V_SLICE] += dv
                    self._terminal_dispersion_applied = True
                    r, v, q, w, prop = D6.unpack(s)

                # CLOSED-LOOP PREDICTIVE BOOSTBACK.
                #
                # Solve the remaining 33-engine time and the burn heading
                # against a live RK4 prediction of burn + tail + coast, and
                # RE-SOLVE it every `boostback_resolve_period` seconds while
                # enough burn remains. The one-shot version froze whatever the
                # prediction said at burn start, including the flip's
                # residual rates; re-solving closes the loop on them.
                if (seg.mode == Mode.ATTITUDE and seg.name == "boostback_33"
                        and getattr(seg, "predictive_boostback", False)
                        and not self._boostback_cutoff):
                    due = (self._boostback_pred is None or (
                        t_seg - self._boostback_pred_t
                        >= seg.boostback_resolve_period
                        and self._boostback_cutoff_tseg - t_seg
                        > seg.boostback_resolve_min_remaining))
                    if due:
                        from .mission import (solve_predictive_remaining_33,
                                              yawed_attitude)
                        if self._bb_q_base is None:
                            self._bb_q_base = {
                                sg.name: np.asarray(sg.q_ref, float).copy()
                                for sg in segments
                                if sg.name.startswith("boostback")
                                and sg.q_ref is not None}
                        guess = (None if self._boostback_cutoff_tseg is None
                                 else self._boostback_cutoff_tseg - t_seg)
                        tx = seg.boostback_target_x
                        pred = solve_predictive_remaining_33(
                            self.gnc_vehicle, self.gnc_aero, s,
                            self._bb_q_base["boostback_33"],
                            max(seg.duration - t_seg, 0.0),
                            throttle=seg.throttle,
                            tail13=seg.boostback_tail13,
                            tail3=seg.boostback_tail3,
                            target_altitude=seg.boostback_target_altitude,
                            target_x=0.0 if tx is None else tx,
                            target_y=seg.boostback_target_y,
                            dt=0.10, guess=guess,
                            yaw_guess=self._bb_yaw,
                            steer_crossrange=seg.boostback_steer_crossrange,
                        )
                        self._boostback_pred = pred
                        self._boostback_pred_t = t_seg
                        self._boostback_history.append(
                            (t, pred["remaining_33"], pred["yaw"],
                             pred["predicted_x"], pred["predicted_y"]))
                        self._boostback_cutoff_tseg = t_seg + pred["remaining_33"]
                        self._bb_yaw = pred["yaw"]
                        for sg in segments:
                            if sg.name in self._bb_q_base:
                                sg.q_ref = yawed_attitude(
                                    self._bb_q_base[sg.name], self._bb_yaw)
                        log.events.append((t, f"boostback re-target: "
                                           f"{pred['remaining_33']:.4f} s left, "
                                           f"yaw {np.degrees(self._bb_yaw):+.3f} deg"))
                        (thrust, torque, mdot, throttle, gim,
                         r_ref_now, q_ref_now) = self._actuate(
                            seg, s, t_seg, r_seg_start)

                # Apply the solved cutoff on every integration step, without
                # quantizing it to the flight timestep.
                if (seg.name == "boostback_33"
                        and getattr(seg, "predictive_boostback", False)
                        and self._boostback_cutoff_tseg is not None
                        and not self._boostback_cutoff):
                    cutoff_remaining = self._boostback_cutoff_tseg - t_seg
                    if cutoff_remaining <= _dt + 1e-9:
                        if cutoff_remaining > 1e-8:
                            _dt = cutoff_remaining
                        else:
                            _dt = 0.0
                        self._boostback_cutoff = True

                if _dt <= 0.0:
                    log.events.append((t, "boostback_33: predictive cutoff"))
                    break

                # No fins_deployed flag: Super Heavy's grid fins are fixed
                # structure and never fold, so fin drag is always present.
                drag = self.aero.drag_force(v, float(r[2]))
                v_air = self.aero.air_velocity(v, float(r[2]))
                # Weathercocking. Applies in EVERY phase, not just the coast:
                # it is a property of the airframe, not a control mode. This is
                # what rotates the booster continuously from its boostback
                # attitude to engines-first and then upright as it descends.
                torque = torque + AERO.restoring_moment(
                    s[D6.Q_SLICE], v_air, float(r[2]), omega=w,
                    density_scale=self.aero.density_scale
                )
                forces = D6.Forces(thrust=thrust, torque=torque,
                                   mass_flow=mdot, drag=drag)

                def deriv(tt, ss, _f=forces):
                    return D6.derivative(tt, ss, _f,
                                         self.vehicle.mass, self.vehicle.inertia,
                                         spherical_gravity=self.spherical_gravity)

                self._t_now = t
                s, _ = D6.rk4_step(t, s, _dt, deriv)
                t += _dt
                t_seg += _dt

                step += 1
                if step % log_every == 0:
                    log.t.append(t)
                    log.state.append(s.copy())
                    log.segment.append(seg.name)
                    log.throttle.append(throttle)
                    log.tilt.append(np.degrees(D6.tilt_from_vertical(s)))
                    log.gimbal.append(np.degrees(gim))
                    log.r_ref.append(r_ref_now)
                    log.q_ref.append(q_ref_now)
                    log.n_lit.append(getattr(self, "_landing_n_lit", seg.n_lit)
                                     if seg.mode == Mode.LANDING else
                                     (0 if seg.mode == Mode.COAST else seg.n_lit))

                if self._boostback_cutoff and seg.name == "boostback_33":
                    log.events.append((t, "boostback_33: predictive cutoff"))
                    break
                if self._segment_done and seg.mode == Mode.LANDING:
                    break
                if (seg.exit_altitude is not None
                        and D6.altitude(s) <= seg.exit_altitude):
                    log.events.append((t, f"{seg.name}: exit altitude reached"))
                    break
                if D6.altitude(s) < 0.0:
                    log.events.append((t, "GROUND CONTACT"))
                    log.t.append(t); log.state.append(s.copy())
                    log.segment.append(seg.name); log.throttle.append(throttle)
                    log.tilt.append(np.degrees(D6.tilt_from_vertical(s)))
                    log.gimbal.append(np.degrees(gim))
                    log.r_ref.append(r_ref_now); log.q_ref.append(q_ref_now)
                    log.n_lit.append(getattr(self, "_landing_n_lit", seg.n_lit)
                                     if seg.mode == Mode.LANDING else
                                     (0 if seg.mode == Mode.COAST else seg.n_lit))
                    return log.arrays()

        # always log the final state, whatever the sample stride
        log.t.append(t); log.state.append(s.copy())
        log.segment.append(segments[-1].name); log.throttle.append(0.0)
        # Final sample: report the engine count actually reached, not the
        # segment's nominal. Otherwise the last frame of every animation shows
        # the landing burn back on 13 engines after it has downselected to 2.
        log.tilt.append(np.degrees(D6.tilt_from_vertical(s)))
        log.gimbal.append(0.0)
        log.r_ref.append(s[D6.R_SLICE].copy()); log.q_ref.append(s[D6.Q_SLICE].copy())
        log.n_lit.append(self._landing_n_lit
                         if segments[-1].mode == Mode.LANDING
                         and self._landing_n_lit
                         else segments[-1].n_lit)
        log.events.append((t, "sequence complete"))
        out = log.arrays()
        out["lat_log"] = list(self._lat_log)
        out["ignition_alt"] = self._ignition_alt
        out["ignition_v"] = self._ignition_v
        out["boostback_history"] = list(self._boostback_history)
        out["guidance_log"] = list(self._guid_log)
        lg = self._landing
        out["landing_events"] = ({} if lg is None else {
            "ignition_t": lg.ignition_t, "ignition_alt": lg.ignition_alt,
            "ignition_speed": lg.ignition_speed,
            "handover_t": lg.handover_t, "handover_alt": lg.handover_alt})
        return out

    # ------------------------------------------------------------------

    def _actuate(self, seg: Segment, s: np.ndarray, t_seg: float,
                 r_seg_start: np.ndarray):
        """Run the controller for this segment and return the actuator state."""
        r, v, q, w, prop = D6.unpack(s)
        veh = self.vehicle

        if seg.mode == Mode.COAST or seg.n_lit == 0:
            # No lit engines means no gimbal authority whatsoever. Grid fins
            # are the only actuator here and the fin control loop is not built
            # yet, so the vehicle is genuinely uncontrolled through this
            # segment. That is a real gap, not a modelling convenience.
            return 0.0, np.zeros(3), 0.0, 0.0, 0.0, r.copy(), q.copy()

        if seg.mode == Mode.LANDING:
            # Closed-loop landing burn -- see landing.py for the guidance law.
            if self._landing is None:
                cfg = seg.landing if seg.landing is not None else LandingConfig(
                    target=(np.asarray(seg.r_target, float)
                            if seg.r_target is not None
                            else np.array([0.0, 0.0, seg.catch_altitude])))
                self._landing = LandingGuidance(cfg, self.gnc_vehicle,
                                                self.gnc_aero)
            lg = self._landing
            cmd = lg.step(r, v, q, prop, self._t_now)
            self._ignited = cmd.phase != "coast"
            self._ignition_alt = lg.ignition_alt
            self._ignition_v = lg.ignition_speed
            self._landing_n_lit = cmd.n_lit
            if cmd.phase == "done":
                self._segment_done = True
            q_cmd = attitude_from_pointing(cmd.direction,
                                           roll_reference=cmd.roll_ref)
            if cmd.phase in ("coast", "done"):
                ao = self.attitude.update(q, w, q_cmd, np.zeros(3), prop,
                                          0, 0.0)
                return (0.0, ao.torque_applied, 0.0, 0.0, 0.0, r.copy(),
                        q_cmd)
            if cmd.phase == "brake" or cmd.phase == "terminal":
                self._guid_log.append(dict(cmd.info, t=self._t_now,
                                           phase=cmd.phase,
                                           throttle=cmd.throttle,
                                           n_lit=cmd.n_lit))
            n_lit = cmd.n_lit
            throttle = cmd.throttle
            thrust = veh.axial_thrust(n_lit, throttle)       # TRUTH engines
            n_gim = min(seg.n_gimballing, n_lit)
            ao = self.attitude.update(q, w, q_cmd, np.zeros(3), prop,
                                      n_gim, throttle)
            return (thrust, ao.torque_applied,
                    veh.mass_flow(n_lit, throttle, vacuum=False),
                    throttle, ao.gimbal_pitch, r.copy(), q_cmd)

        if seg.mode == Mode.POSITION:
            r_ref, v_ref, a_ref = divert_profile(
                r_seg_start, seg.r_target, seg.divert_duration, t_seg
            )
            po = self.position.update(r, v, r_ref, v_ref, prop,
                                      seg.n_lit, a_ref=a_ref)
            ao = self.attitude.update(q, w, po.q_cmd, np.zeros(3), prop,
                                      seg.n_gimballing, po.throttle)
            return (veh.axial_thrust(seg.n_lit, po.throttle), ao.torque_applied,
                    veh.mass_flow(seg.n_lit, po.throttle, vacuum=False),
                    po.throttle, ao.gimbal_pitch, r_ref, po.q_cmd)

        # ATTITUDE mode
        if seg.q_slew is not None:
            q_ref, w_ref, a_ref = seg.q_slew.at(t_seg)
        else:
            q_ref, w_ref, a_ref = (seg.q_ref if seg.q_ref is not None else q), \
                                  np.zeros(3), None
        ao = self.attitude.update(q, w, q_ref, w_ref, prop,
                                  seg.n_gimballing, seg.throttle,
                                  alpha_ref=a_ref)
        return (veh.axial_thrust(seg.n_lit, seg.throttle), ao.torque_applied,
                veh.mass_flow(seg.n_lit, seg.throttle, vacuum=False),
                seg.throttle, ao.gimbal_pitch, r.copy(), q_ref)


# ---------------------------------------------------------------------------
# catch scoring
# ---------------------------------------------------------------------------

def catch_report(state: np.ndarray, target: np.ndarray,
                 tol_lateral: float = 1.0, tol_speed: float = 1.0,
                 tol_h_speed: float = 0.5,
                 tol_tilt_deg: float = 0.5, tol_rate_deg: float = 1.0,
                 tower=None) -> dict:
    """
    Score the terminal state against the catch criteria.

    Four numbers, all of which have to pass. Reporting them individually
    rather than as a single score is deliberate: when a run fails you want to
    know WHICH constraint it missed, because they have different causes.
    """
    r, v, q, w, prop = D6.unpack(state)
    tgt = np.asarray(target, dtype=float)

    lateral = float(np.linalg.norm((r - tgt)[:2]))
    vertical = float(r[2] - tgt[2])
    horizontal_speed = float(np.linalg.norm(v[:2]))
    vertical_speed = float(abs(v[2]))
    tilt = float(np.degrees(D6.tilt_from_vertical(state)))
    rate = float(np.degrees(np.linalg.norm(w)))

    # Fin roll alignment. The booster is caught by its grid fins, so the
    # opposed pair must lie across the chopstick arms. Body z carries the
    # third fin and should point away from the tower (+x); the error is how
    # far it has rotated off that.
    fin3 = Q.rotate(q, np.array([0.0, 0.0, 1.0]))
    fin3_h = np.array([fin3[0], fin3[1]])
    nh = float(np.linalg.norm(fin3_h))
    roll_err = (float(np.degrees(np.arccos(np.clip(fin3_h[0] / nh, -1.0, 1.0))))
                if nh > 1e-6 else 0.0)

    # Tower compensation. The arms track the booster, so position error inside
    # the travel envelope is absorbed by the tower rather than failing the
    # catch. Velocity is NOT absorbed -- the arms can be somewhere else, they
    # cannot be moving at the vehicle's speed. Scoring against a fixed point
    # models a harder problem than the real one, and spends vehicle authority
    # on accuracy the tower would have supplied.
    if tower is not None:
        res_lat, res_vert, _ = tower.compensated_error(r, tgt)
    else:
        res_lat, res_vert = lateral, abs(vertical)

    checks = {
        "lateral_error_m": (res_lat, res_lat <= tol_lateral),
        "fin_roll_deg": (roll_err, roll_err <= 10.0),
        "vertical_error_m": (res_vert, res_vert <= tol_lateral),
        # Split, because the two have different physical consequences and a
        # single norm hides which one is out. 0.5 m/s of crossrange is a
        # sliding contact the rails absorb; 0.5 m/s of descent is a different
        # matter entirely.
        "horizontal_speed_ms": (horizontal_speed,
                                horizontal_speed <= tol_h_speed),
        "vertical_speed_ms": (vertical_speed, vertical_speed <= tol_speed),
        "tilt_deg": (tilt, tilt <= tol_tilt_deg),
        "body_rate_deg_s": (rate, rate <= tol_rate_deg),
    }
    return {
        "checks": checks,
        "propellant_remaining_t": prop / 1000.0,
        "caught": all(ok for _, ok in checks.values()),
    }
