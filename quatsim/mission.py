"""
Full RTLS mission: hot-stage separation through catch.

THE TRAJECTORY SHAPE, AND WHY IT IS A HOOK
------------------------------------------
The obvious mental picture -- boostback, fall back down, land -- is wrong, and
getting it wrong hides the most distinctive part of the flight.

At separation the booster is around 70 km up, doing well over a kilometre per
second, and CLIMBING at a flight path angle of roughly 25 degrees. The flip and
boostback reverse the DOWNRANGE component of that velocity. They barely touch
the vertical component. So when the burn ends the vehicle is still going up --
it continues to an apogee HIGHER than the altitude it separated at, while
already travelling back toward the pad.

The path therefore looks like a hook: up and downrange, a tight reversal at the
top of the powered arc, then a long ballistic curve up over apogee and back
down to the launch site. Plotted in 3D it is the one genuinely interesting
shape in the whole flight, and a terminal-phase-only simulation never shows it.

WHAT IS SOLVED HERE RATHER THAN SPECIFIED
-----------------------------------------
The boostback burn duration is not a number anyone looks up. It is whatever
makes the subsequent ballistic arc terminate at the launch site, so it is found
by shooting: guess a duration, propagate the coast, measure where the vehicle
comes down, iterate. That is how preliminary trajectory design is actually
done, and it is the reason this module refuses to hardcode the value.

UNCERTAINTY, STATED PLAINLY
---------------------------
The separation state below is an estimate. Public figures put hot-staging in
the neighbourhood of 65-70 km at 1.6-1.8 km/s, but the exact state vector is
not published, and everything downstream inherits that uncertainty. The
boostback duration this module solves for is only as good as the separation
state it starts from -- treat the SHAPE of the result as robust and the
NUMBERS as indicative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dynamics6dof as D6
from . import environment as ENV
from . import quaternion as Q
from . import aero as AERO
from .aero import AeroModel
from .control import AttitudeController, ControlGains
from .guidance import Slew, attitude_from_pointing
from .phases import Mode, Segment
from .vehicle import Vehicle


@dataclass
class SeparationState:
    """
    Vehicle state at hot-stage separation.

    ESTIMATE, not data. Downrange is measured from the pad along +x; the pad
    sits at the origin, so the booster has to null this distance.
    """
    altitude: float = 68_000.0          # m
    downrange: float = 95_000.0         # m from the pad
    speed: float = 1_700.0              # m/s
    flight_path_angle: float = 25.0     # deg above horizontal, CLIMBING
    prop_remaining: float = 400_000.0   # kg

    def state_vector(self) -> np.ndarray:
        fpa = np.radians(self.flight_path_angle)
        v = np.array([self.speed * np.cos(fpa), 0.0, self.speed * np.sin(fpa)])
        r = np.array([self.downrange, 0.0, self.altitude])
        # Nose along the velocity vector: the booster leaves the stack still
        # pointed the way it was flying.
        q = attitude_from_pointing(v)
        return D6.make_state(r, v, q, np.zeros(3), self.prop_remaining)


# ---------------------------------------------------------------------------
# boostback targeting
# ---------------------------------------------------------------------------

def _propagate_ballistic(vehicle: Vehicle, aero: AeroModel,
                         state: np.ndarray, stop_altitude: float,
                         dt: float = 0.02, t_max: float = 900.0):
    """
    Unpowered coast down to a target altitude.

    Fixed-step Euler with drag. The step is small (0.02 s) rather than the
    convenient 0.25 s, and that matters more than it looks: a 200 second coast
    integrated coarsely accumulates enough error to move the landing point by
    kilometres, so the solve converges on a target the real RK4 run then
    misses. Cheap integration inside a targeting loop is a false economy when
    the arc is this long.

    Spherical gravity is on: over a ~100 km arc reaching nearly 100 km altitude,
    the flat-earth approximation is worth a couple of percent, which is enough
    to move the landing point by kilometres.
    """
    r = state[D6.R_SLICE].copy()
    v = state[D6.V_SLICE].copy()
    prop = float(state[D6.P_INDEX])
    m = vehicle.mass(prop)
    t = 0.0
    apogee = r[2]

    while t < t_max:
        a = ENV.gravity(r, spherical=True) + aero.drag_force(v, r[2]) / m
        v = v + a * dt
        r = r + v * dt
        t += dt
        apogee = max(apogee, r[2])
        if r[2] <= stop_altitude and v[2] < 0:
            break
    return r, v, t, apogee


def propagate_flip(vehicle: Vehicle, aero: AeroModel, sep: SeparationState,
                   duration: float, n_lit: int = 5, throttle: float = 0.40,
                   dt: float = 0.05) -> np.ndarray:
    """
    Propagate the flip, so the boostback solve starts from where the flip
    actually ends.

    THIS IS NOT BOOKKEEPING. The flip takes about twelve seconds, and the
    booster crosses roughly 18 km further downrange while it happens -- plus it
    burns propellant on the engines kept lit through the manoeuvre. Solving the
    boostback from the SEPARATION state instead of the post-flip state makes
    the targeting inconsistent with the trajectory actually flown, and the
    vehicle lands tens of kilometres short.

    Thrust follows the SAME SLEW the sequencer flies, not the velocity vector.
    That distinction is worth about 40 km of miss distance: the flip carries
    roughly 200 m/s of delta-v, and pointing it prograde for the whole
    manoeuvre instead of sweeping it through 180 degrees puts that entire
    budget in the wrong direction. Over a 200 second coast the error compounds
    into a miss larger than the downrange distance being nulled.

    The lesson generalises. Every targeting solve in this project has to model
    what is actually FLOWN, not a convenient approximation of it -- the same
    mistake produced the earlier 46 km miss when the solve ignored the flip
    entirely.
    """
    s = sep.state_vector()
    r = s[D6.R_SLICE].copy()
    v = s[D6.V_SLICE].copy()
    prop = float(s[D6.P_INDEX])
    mdot = vehicle.mass_flow(n_lit, throttle)
    t = 0.0

    q_ascent = attitude_from_pointing(s[D6.V_SLICE])
    vh0 = np.array([s[D6.V_SLICE][0], s[D6.V_SLICE][1], 0.0])
    q_retro = attitude_from_pointing(-vh0)
    slew = Slew(q_ascent, q_retro, 0.0, duration, smooth=True)

    while t < duration and prop > 0.0:
        m = vehicle.mass(prop)
        q_ref, _, _ = slew.at(t)
        # thrust acts along body +x of the REFERENCE attitude
        direction = Q.rotate(q_ref, np.array([1.0, 0.0, 0.0]))
        a = ((vehicle.axial_thrust(n_lit, throttle) / m) * direction
             + ENV.gravity(r, spherical=True)
             + aero.drag_force(v, r[2]) / m)
        v = v + a * dt
        r = r + v * dt
        prop = max(prop - mdot * dt, 0.0)
        t += dt

    return D6.make_state(r, v, attitude_from_pointing(-v), np.zeros(3), prop)


def solve_boostback(
    vehicle: Vehicle,
    aero: AeroModel,
    sep: SeparationState,
    start_state: np.ndarray | None = None,
    n_lit: int = 33,
    n_gimballing: int = 13,
    throttle: float = 1.0,
    target_altitude: float = 1_200.0,
    bracket: tuple[float, float] = (1.0, 40.0),
    tol: float = 1.0,
) -> dict:
    """
    Solve for the boostback burn duration that brings the booster home.

    Shooting method. For a trial duration the burn is modelled as thrust
    applied along the retrograde horizontal direction (pure downrange
    reversal), then the vehicle coasts ballistically and we measure the
    downrange position where it reaches the landing-burn altitude. Bisection
    on the miss distance.

    Monotonic in duration -- burn longer, land shorter -- so bisection is safe.

    Returns the duration, the propellant it costs, and the coast that follows,
    including apogee. That apogee is the number worth looking at: it comes out
    HIGHER than the separation altitude, which is the quantitative statement of
    why the trajectory hooks rather than turning around.
    """
    # Start from the post-flip state when one is supplied. Defaulting to the
    # separation state is only correct for a zero-duration flip.
    s0 = sep.state_vector() if start_state is None else start_state.copy()
    mdot = vehicle.mass_flow(n_lit, throttle)

    # FIXED thrust direction for the whole burn, set by the horizontal
    # velocity AT IGNITION.
    #
    # The tempting alternative -- point along the instantaneous retrograde each
    # step -- is a trap, and an instructive one. Roughly ten seconds in, the
    # horizontal velocity passes through zero. At that moment "retrograde" is
    # undefined, the computed direction flips sign every step, and the thrust
    # cancels itself: the burn then consumes propellant at 26 t/s while the
    # horizontal velocity sits pinned near -10 m/s no matter how long it runs.
    # The vehicle burns its entire load and never comes home.
    #
    # A real boostback holds a fixed inertial attitude, which is also what the
    # segment list commands via a constant q_retro. The solver has to match the
    # thing it is solving for.
    vh0 = np.array([s0[D6.V_SLICE][0], s0[D6.V_SLICE][1], 0.0])
    burn_direction = -vh0 / np.linalg.norm(vh0)

    def shoot(duration: float):
        r = s0[D6.R_SLICE].copy()
        v = s0[D6.V_SLICE].copy()
        prop = float(s0[D6.P_INDEX])
        dt, t = 0.05, 0.0

        # Mirror the flown downselect: the last 6 s run on 13 then 3 engines.
        tail13, tail3 = 3.0, 3.0
        while t < duration and prop > 0.0:
            m = vehicle.mass(prop)
            t_left = duration - t
            n_now = (3 if t_left <= tail3
                     else 13 if t_left <= tail3 + tail13
                     else n_lit)
            mdot_now = vehicle.mass_flow(n_now, throttle)
            # Thrust reverses the downrange component. It leaves the vertical
            # component almost untouched, which is exactly why the vehicle
            # keeps climbing after burnout and the trajectory hooks.
            a = (vehicle.axial_thrust(n_now, throttle) / m) * burn_direction
            a = a + ENV.gravity(r, spherical=True) + aero.drag_force(v, r[2]) / m
            v = v + a * dt
            r = r + v * dt
            prop = max(prop - mdot_now * dt, 0.0)
            t += dt

        burn_end = D6.make_state(r, v, attitude_from_pointing(-v), np.zeros(3), prop)
        r_end, v_end, t_coast, apogee = _propagate_ballistic(
            vehicle, aero, burn_end, target_altitude
        )
        return r, v, prop, r_end, v_end, t_coast, apogee

    lo, hi = bracket
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        *_, r_end, v_end, t_coast, apogee = shoot(mid)
        if r_end[0] > 0.0:        # still downrange of the pad: burn longer
            lo = mid
        else:                     # overshot past the pad: burn shorter
            hi = mid
        if hi - lo < tol / 100.0:
            break

    duration = 0.5 * (lo + hi)
    r_bb, v_bb, prop_bb, r_end, v_end, t_coast, apogee = shoot(duration)

    return {
        "duration": duration,
        "prop_used": float(s0[D6.P_INDEX]) - prop_bb,
        "prop_after": prop_bb,
        "burnout_position": r_bb,
        "burnout_velocity": v_bb,
        "apogee": apogee,
        "apogee_gain": apogee - sep.altitude,
        "start_prop": float(s0[D6.P_INDEX]),
        "coast_duration": t_coast,
        "arrival_position": r_end,
        "arrival_velocity": v_end,
        "arrival_speed": float(np.linalg.norm(v_end)),
        "miss_distance": float(np.linalg.norm(r_end[:2])),
    }


def predict_boostback_arrival_rk4(
    vehicle: Vehicle,
    aero: AeroModel,
    state: np.ndarray,
    q_ref: np.ndarray,
    remaining_33: float,
    tail13: float = 3.0,
    tail3: float = 3.0,
    throttle: float = 0.87,
    target_altitude: float = 1_200.0,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict the future flight with the same 6-DOF RK4 architecture.

    Unlike the earlier point-mass predictor, this version reproduces the
    actual sequencer's control convention: one controller/Forces evaluation per
    integrator step, then ``D6.rk4_step`` holds those forces through k1..k4.
    The attitude controller, weathercocking moment, spherical gravity, frozen
    drag vector, propellant-dependent mass/inertia, and 33/13/3 engine staging
    are all represented.  This is deliberately a shadow copy of the plant,
    not a lower-fidelity trajectory formula.
    """
    s = np.asarray(state, dtype=float).copy()
    q_ref = np.asarray(q_ref, dtype=float).copy()
    ctrl = AttitudeController(vehicle, ControlGains(wn=1.5, zeta=0.8))
    t = 0.0
    total = max(float(remaining_33), 0.0) + float(tail13) + float(tail3)

    # Stage boundaries. Steps are cut to land EXACTLY on them: evaluating
    # the engine count only at fixed step starts rounds the 33-engine time to
    # the prediction step, and at ~30 km of range per second of burn a 0.1 s
    # quantum made the predicted arrival jump by 4 km -- the root finder then
    # chattered between two branches and never converged.
    b33 = max(float(remaining_33), 0.0)
    b13 = b33 + float(tail13)
    while t < total - 1e-12:
        if t < b33 - 1e-12:
            n_now, nxt = 33, b33
        elif t < b13 - 1e-12:
            n_now, nxt = 13, b13
        else:
            n_now, nxt = 3, total
        h = min(dt, nxt - t)

        r, v, q, w, prop = D6.unpack(s)
        if prop > 0.0:
            thrust = vehicle.axial_thrust(n_now, throttle)
            mdot = vehicle.mass_flow(n_now, throttle, vacuum=False)
        else:
            thrust = 0.0
            mdot = 0.0

        # Match FlightSequencer._actuate(): controller and environmental drag
        # are evaluated once at the beginning of the RK4 step.
        q_control_ref = q_ref
        ao = ctrl.update(q, w, q_control_ref, np.zeros(3), prop,
                         min(13, n_now), throttle)
        drag = aero.drag_force(v, float(r[2]))
        torque = ao.torque_applied + AERO.restoring_moment(
            q, v, float(r[2]), omega=w)
        forces = D6.Forces(thrust=thrust, torque=torque,
                           mass_flow=mdot, drag=drag)

        def deriv(tt, ss, _f=forces):
            return D6.derivative(tt, ss, _f,
                                  vehicle.mass, vehicle.inertia,
                                  spherical_gravity=True)

        s, _ = D6.rk4_step(t, s, h, deriv)
        t += h

    # Same unpowered reorient+coast translational plant.  Reorient attitude is
    # irrelevant to translation because thrust is zero and drag is aligned
    # with velocity in this model, so a translational RK4 continuation is
    # sufficient and much cheaper than predicting the visual attitude slew.
    s_prev = s.copy()
    while t < 900.0:
        if s[D6.R_SLICE][2] <= target_altitude and s[D6.V_SLICE][2] < 0.0:
            # Interpolate the crossing so the reported arrival point is a
            # continuous function of the burn, not of the step grid.
            z0, z1 = s_prev[D6.R_SLICE][2], s[D6.R_SLICE][2]
            f = (z0 - target_altitude) / (z0 - z1) if z0 != z1 else 1.0
            f = float(np.clip(f, 0.0, 1.0))
            s = s_prev + f * (s - s_prev)
            t = t - (1.0 - f) * h
            break
        s_prev = s.copy()
        h = min(dt, 900.0 - t)
        r, v, q, w, prop = D6.unpack(s)
        drag = aero.drag_force(v, float(r[2]))

        def coast_deriv(tt, ss, _drag=drag):
            r = ss[D6.R_SLICE]
            prop = float(ss[D6.P_INDEX])
            m = vehicle.mass(prop)
            d = np.zeros_like(ss)
            d[D6.R_SLICE] = ss[D6.V_SLICE]
            d[D6.V_SLICE] = ENV.gravity(r, spherical=True) + _drag / m
            return d

        s, _ = D6.rk4_step(t, s, h, coast_deriv)
        t += h

    return s[D6.R_SLICE].copy(), s[D6.V_SLICE].copy(), t

def yawed_attitude(q_ref: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate an attitude about INERTIAL +z (local vertical) by ``yaw`` rad.

    Used to steer the boostback burn axis out of the separation plane by a
    small angle to null the predicted crossrange miss. Rotating about the
    local vertical changes only the heading of the burn, not its elevation,
    so the downrange solve and the crossrange solve stay nearly decoupled.
    """
    return Q.normalize(Q.multiply(Q.from_axis_angle([0.0, 0.0, 1.0], yaw),
                                  np.asarray(q_ref, dtype=float)))


def solve_predictive_remaining_33(
    vehicle: Vehicle,
    aero: AeroModel,
    state: np.ndarray,
    q_ref: np.ndarray,
    max_remaining: float,
    throttle: float = 0.87,
    tail13: float = 3.0,
    tail3: float = 3.0,
    target_altitude: float = 1_200.0,
    target_x: float = 0.0,
    target_y: float = 0.0,
    dt: float = 0.1,
    guess: float | None = None,
    yaw_guess: float = 0.0,
    steer_crossrange: bool = True,
    tol_x: float = 2.0,
) -> dict:
    """Closed-loop boostback targeting: cutoff time AND burn heading.

    Two unknowns, two conditions, solved against a live RK4 prediction of the
    remaining burn, the 13/3-engine tail and the ballistic coast:

        remaining 33-engine time  ->  predicted downrange x at target altitude
        yaw of the burn axis      ->  predicted crossrange y at target altitude

    WHY THIS REPLACED THE 7-STEP BISECTION
    --------------------------------------
    Downrange sensitivity is ~30 km per second of 33-engine burn. Seven
    bisection steps over a 12 s window resolve the cutoff to 0.094 s, i.e.
    ~3 km of arrival error. The nominal case happened to land inside that
    quantum; a 2 km separation-downrange dispersion arrived 2 km off and the
    catch failed. Here the cutoff is found by a bracketed secant (Illinois
    false-position) iteration to ``tol_x`` metres, typically in 4-6 shots.

    The crossrange channel uses one finite-difference shot for dy/dyaw and a
    Newton step. The x/y coupling is weak (yaw of a fraction of a degree), so
    one pass each is sufficient; the sequencer calls this repeatedly during
    the burn, which closes the loop on any prediction error.
    """
    q_ref = np.asarray(q_ref, dtype=float)

    def shoot(rem, yaw):
        r, v, _ = predict_boostback_arrival_rk4(
            vehicle, aero, state, yawed_attitude(q_ref, yaw), rem,
            tail13=tail13, tail3=tail3, throttle=throttle,
            target_altitude=target_altitude, dt=dt)
        return r, v

    n_shots = 0

    def solve_x(yaw, g):
        nonlocal n_shots
        cache = {}

        def f(rem):
            nonlocal n_shots
            if rem not in cache:
                n_shots += 1
                r, v = shoot(rem, yaw)
                cache[rem] = (float(r[0] - target_x), r, v)
            return cache[rem]

        # Build a bracket around the guess. More burn -> smaller x.
        g = float(np.clip(g, 0.0, max_remaining))
        h = 0.08
        a, b = max(g - h, 0.0), min(g + h, max_remaining)
        fa, fb = f(a)[0], f(b)[0]
        while fa < 0.0 and a > 0.0:
            a = max(a - 4.0 * h, 0.0); fa = f(a)[0]
        while fb > 0.0 and b < max_remaining:
            b = min(b + 4.0 * h, max_remaining); fb = f(b)[0]
        if fa <= 0.0:
            return a, f(a)
        if fb >= 0.0:
            return b, f(b)
        side = 0
        c, fc = a, fa
        for _ in range(30):
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)[0]
            if abs(fc) < tol_x or (b - a) < 1e-6:
                break
            if fc > 0.0:
                a, fa = c, fc
                if side == 1:
                    fb *= 0.5
                side = 1
            else:
                b, fb = c, fc
                if side == -1:
                    fa *= 0.5
                side = -1
        return c, f(c)

    g0 = (0.5 * max_remaining) if guess is None else guess
    yaw = float(yaw_guess)
    rem, (fx, r, v) = solve_x(yaw, g0)

    if steer_crossrange:
        dyaw = np.radians(0.05)
        r2, _ = shoot(rem, yaw + dyaw)
        n_shots += 1
        dy = (r2[1] - r[1]) / dyaw
        if abs(dy) > 1.0:
            yaw = float(np.clip(yaw - (r[1] - target_y) / dy,
                                -np.radians(5.0), np.radians(5.0)))
            rem, (fx, r, v) = solve_x(yaw, rem)

    return {
        "remaining_33": float(rem),
        "yaw": float(yaw),
        "predicted_x": float(fx),
        "predicted_y": float(r[1] - target_y),
        "predicted_position": r,
        "predicted_velocity": v,
        "shots": n_shots,
    }


# ---------------------------------------------------------------------------
# mission assembly
# ---------------------------------------------------------------------------

def build_mission(
    vehicle: Vehicle,
    aero: AeroModel,
    sep: SeparationState,
    boostback: dict,
    landing: dict,
    catch_target: np.ndarray,
    approach_offset: np.ndarray,
    flip_duration: float = 4.0,
    ignition_duration: float = 1.0,
    flip_engines: int = 5,
    flip_throttle: float = 0.40,
    use_slew: bool = True,
    reorient_duration: float = 100.0,
    landing_3_duration: float = 4.0,
    bb_13_duration: float = 3.0,
    bb_3_duration: float = 3.0,
    boostback_elevation: float = 0.0,   # rad above horizontal, +ve = lofted
) -> list[Segment]:
    """
    Assemble the full segment list from the solved burn parameters.

    The flip is a real slew from the ascent attitude to retrograde, flown by
    the attitude controller rather than teleported. It runs on 33 engines lit
    but only 13 gimballing -- the distinction that would overstate pitch
    authority by 2.5x if collapsed into one number.
    """
    v_sep = sep.state_vector()[D6.V_SLICE]
    q_ascent = attitude_from_pointing(v_sep)

    # PURE-PITCH FLIP. Build the retrograde attitude by rotating the ascent
    # attitude about its own BODY Y axis, rather than by independently pointing
    # at the retrograde direction.
    #
    # The difference is not cosmetic. attitude_from_pointing picks its own roll
    # reference at each end, and the rotation between the two endpoints came
    # out 21.6% roll -- so the flip axis was tilted, and rotating about a
    # tilted axis sweeps the nose around a CONE. Peak thrust direction left the
    # trajectory plane by 97.6%, and the flip's ~200 m/s of delta-v went
    # sideways: 20 km of crossrange after a 200 s coast, from a problem that
    # starts perfectly planar.
    #
    # A pure pitch rotation keeps thrust in the plane exactly (max |thrust_y|
    # is zero to machine precision) and needs no roll authority at all.
    vhat = v_sep / np.linalg.norm(v_sep)
    target = -np.array([v_sep[0], v_sep[1], 0.0])
    target = target / np.linalg.norm(target)
    body_y = Q.rotate(q_ascent, np.array([0.0, 1.0, 0.0]))
    pitch_angle = np.arctan2(float(np.dot(np.cross(vhat, target), body_y)),
                             float(np.dot(vhat, target)))
    # BURN POINTING ELEVATION -- the second targeting parameter.
    #
    # Burn duration alone sets WHERE the vehicle arrives; it cannot also set
    # HOW FAST. One knob, two terminal conditions. Elevating or depressing the
    # burn axis trades vertical against horizontal delta-v: raising it lofts
    # the arc and lengthens the coast, lowering it flattens the arc and brings
    # the vehicle back faster and shallower. Together with duration that is two
    # knobs for two conditions, which is what the problem actually needs.
    #
    # Applied as pure pitch, like the flip, so the burn stays in the
    # trajectory plane and generates no crossrange.
    q_retro = Q.multiply(q_ascent,
                         Q.from_axis_angle([0.0, 1.0, 0.0],
                                           pitch_angle - boostback_elevation))
    q_up = attitude_from_pointing([0.0, 0.0, 1.0])

    # Flight 13 flew FIVE engines for the flip, then upselected to 33 for the
    # boostback. Five sit inside the gimballing set, so all of them contribute
    # control torque -- roughly 67% more authority than three.
    # ENGINE SEQUENCE THROUGH THE FLIP.
    #
    # Flight 13: a brief 5-engine ignition, then the inner 13 light, then the
    # outer 20 roughly 0.3 s later -- and the flip happens DURING that ramp, on
    # all 33, not before it. The earlier model had it backwards: a long
    # 5-engine flip that completed before upselecting, which made the manoeuvre
    # slow for the obvious reason that it was being flown on a fraction of the
    # available authority.
    #
    # Note that lighting all 33 raises THRUST but not gimbal torque: only the
    # inner 13 have actuators. With a 30 m moment arm and a 15 deg limit the
    # bang-bang floor for a 155 deg flip is 3.6 s, so the ~1-2 s seen on the
    # livestream implies more authority than this model has -- a longer moment
    # arm, a wider gimbal limit, or outer-ring actuators. Flagged rather than
    # fudged.
    # Reorient endpoint: engines-first against the solved arrival velocity.
    # A fixed endpoint gives the slew something to aim at instead of chasing a
    # target that moves as the trajectory arcs over apogee.
    v_arr = np.asarray(boostback["arrival_velocity"], dtype=float)
    q_entry = attitude_from_pointing(-v_arr)
    reorient_seg = Segment("reorient", Mode.COAST, n_lit=0, n_gimballing=0,
                           duration=reorient_duration)
    reorient_seg.q_slew = Slew(q_retro, q_entry, 0.0, reorient_duration,
                               smooth=True)

    coast_duration = max(boostback["coast_duration"] - reorient_duration, 1.0)
    coast_seg = Segment("coast", Mode.COAST, n_lit=0, n_gimballing=0,
                        duration=coast_duration * 2.0)
    # Ends on ALTITUDE, not a clock, so the landing burn always gets its
    # chance to run. Grid-fin steering is enabled by setting coast_seg.entry
    # (see entry.py and run_mission.fly).
    coast_seg.q_slew = None
    coast_seg.exit_altitude = 4000.0

    ign = Segment("ignition_5", Mode.ATTITUDE, n_lit=5, n_gimballing=5,
                  throttle=1.00, duration=ignition_duration)
    ign.q_slew = Slew(q_ascent, q_retro, 0.0,
                      ignition_duration + flip_duration, smooth=True)

    flip_seg = Segment("flip_33", Mode.ATTITUDE, n_lit=33, n_gimballing=13,
                       throttle=1.00, duration=flip_duration)
    # continues the same slew, offset by the ignition phase already flown
    flip_seg.q_slew = Slew(q_ascent, q_retro, -ignition_duration,
                           ignition_duration + flip_duration, smooth=True)

    return [
        ign,
        flip_seg,
        # BOOSTBACK DOWNSELECT 33 -> 13 -> 3, as flown on Flight 13
        # (11 s / 3 s / 3 s from livestream timing).
        #
        # Not cosmetic. The vehicle sheds roughly 25 t/s on 33 engines, so by
        # the end of the burn it is far lighter and 33 engines would be piling
        # on acceleration precisely when cutoff velocity needs fine control.
        # Stepping down keeps the acceleration roughly bounded and makes the
        # final delta-v trimmable. The solver varies only the 33-engine
        # portion; the two tail stages are held at their observed durations.
        Segment("boostback_33", Mode.ATTITUDE, n_lit=33, n_gimballing=13,
                throttle=1.00, q_ref=q_retro,
                duration=max(boostback["duration"] - bb_13_duration
                             - bb_3_duration, 0.1)),
        Segment("boostback_13", Mode.ATTITUDE, n_lit=13, n_gimballing=13,
                throttle=1.00, duration=bb_13_duration, q_ref=q_retro),
        Segment("boostback_3", Mode.ATTITUDE, n_lit=3, n_gimballing=3,
                throttle=1.00, duration=bb_3_duration, q_ref=q_retro),

        # REORIENT. Boostback leaves the booster nose-first pointing back at
        # the pad, which is the wrong attitude for everything that follows. It
        # has to come round to engines-first -- body +x opposite the velocity
        # vector -- so the engine end leads through entry and the landing burn
        # can fire along the flight path. Leaving this out freezes the vehicle
        # in the boostback attitude for the whole descent, which is both wrong
        # and immediately obvious in an animation.
        #
        # Flown on RCS alone: no engines are lit, so there is no gimbal. RCS
        # manages roughly 0.07 deg/s^2 in pitch, a ~72 s bang-bang slew for
        # 90 degrees. Slow, which is why it gets a generous allocation.
        reorient_seg,

        # No entry burn. Super Heavy does not have one, so this coast is long
        # and unpowered. Attitude is held engines-first on RCS; grid fins add
        # drag but are not yet used for control.
        coast_seg,
        # LANDING BURN: one closed-loop segment, 13 engines braking then 3
        # for the final approach, ending when the CoM reaches the catch
        # altitude. Ignition, throttle, engine downselect and attitude are all
        # outputs of landing.LandingGuidance; the duration is only a bound.
        Segment("landing", Mode.LANDING, n_lit=13, n_gimballing=13,
                throttle=1.00, duration=400.0,
                catch_altitude=float(catch_target[2]), q_ref=q_up,
                r_target=catch_target),
    ]


def mission_report(sep: SeparationState, bb: dict, landing: dict) -> str:
    return "\n".join([
        "RTLS MISSION SUMMARY",
        f"  separation        {sep.altitude/1000:7.1f} km alt, "
        f"{sep.downrange/1000:6.1f} km downrange, {sep.speed:6.0f} m/s "
        f"@ {sep.flight_path_angle:.0f} deg climbing",
        f"  boostback burn    {bb['duration']:7.2f} s  ({bb['prop_used']/1000:.1f} t)",
        f"  apogee AFTER bb   {bb['apogee']/1000:7.1f} km  "
        f"(+{bb['apogee_gain']/1000:.1f} km above separation)",
        f"  coast             {bb['coast_duration']:7.1f} s",
        f"  arrival           {bb['miss_distance']:7.0f} m from pad, "
        f"{bb['arrival_speed']:6.0f} m/s",
        f"  landing ignition  {landing['ignition_altitude']:7.0f} m  "
        f"({landing['prop_used']/1000:.1f} t)",
    ])
