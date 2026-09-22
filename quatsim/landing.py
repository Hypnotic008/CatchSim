"""
Landing-burn guidance for the tower catch -- one law, three dimensions.

WHY THIS REPLACED THE OLD LANDING BLOCK
---------------------------------------
The previous landing guidance grew by accretion: an energy-matched vertical
axis, a per-axis lateral stopping law, a separate "forward-predicted" terminal
law, a crossrange axis that was only ever DAMPED (never steered to the target),
a tilt clamp whose taper was cancelled by a 20 deg floor, and a thrust
magnitude that treated drag as if it always pointed up. It flew the nominal
trajectory beautifully and nothing else: a 2 km arrival error (what a +/-2 km
separation dispersion produces) left the vehicle ~1 km from the tower, and a
5 m/s crossrange error at separation was never corrected at all.

This module replaces it with a single, standard construction used in both
engine phases:

  1. A VERTICAL reference that defines the time-to-go.
  2. A LATERAL polynomial-guidance law that drives position, velocity AND
     acceleration to their targets at that time-to-go. Constraining the final
     acceleration to zero means the commanded lean is zero at the catch --
     the vehicle arrives upright by construction instead of by a taper that
     fights the lateral law.
  3. A THRUST-VECTOR solve in 3-D: required specific force =
     desired acceleration - gravity - (estimated drag)/m. Its direction is the
     attitude command, its magnitude the throttle. Tilt and throttle limits
     are applied to that vector with an explicit priority: vertical first.

THE LATERAL LAW
---------------
For a quadratic-in-time acceleration profile with boundary conditions
r(T) = r_f, v(T) = v_f, a(T) = a_f, the acceleration to apply NOW is

    a0 = a_f - 6 (v_f - v) / T + 12 (r_f - r - v T) / T^2

(the Apollo lunar-descent "E-guidance" family). With a_f = 0 and v_f = 0 this
is just 12 * ZEM / T^2 + 6 * v / T ... applied to both horizontal axes at
once. There is no special-casing of the crossrange axis, so the planar
invariance the old code enforced by hand falls out automatically: a planar
state produces a planar command.

PHASES
------
    COAST     engines off, attitude held engines-first against the velocity.
              Ignition is tested every step (energy test, below).
    BRAKE     13 engines. Vertical: energy matching to the GATE state
              (h_gate, v_gate), recomputed every step from the actual state,
              so throttle is an output rather than a schedule. Lateral: the
              polynomial law to the gate point directly above the tower.
    TERMINAL  3 engines. Vertical: constant descent at v_desc, then a
              constant-deceleration flare to v_touch at the catch plane.
              Lateral: the same polynomial law to the catch point with the
              time-to-go read off the vertical profile. A tilt envelope that
              closes with time-to-go keeps the final attitude inside the catch
              tolerance even when the lateral law would like more.

WHAT GUIDANCE KNOWS
-------------------
Guidance is handed its OWN vehicle and aero models. In the Monte Carlo these
are the nominal models while the truth plant is dispersed (thrust, Isp, drag,
density, wind). Nothing here reads the truth models, so every dispersion has
to be absorbed by feedback -- which is the definition of robust we care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import environment as ENV
from . import quaternion as Q
from .aero import AeroModel
from .vehicle import Vehicle


COAST, BRAKE, TERMINAL, DONE = "coast", "brake", "terminal", "done"


@dataclass
class LandingConfig:
    """Tuning and targets for the landing burn. SI units, angles in rad."""

    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 105.0]))

    # --- engines ---------------------------------------------------------
    n_brake: int = 13
    n_terminal: int = 3
    throttle_floor: float = 0.40          # Raptor deep-throttle floor
    # Effective floor on the final 3-engine approach. Three engines at 40%
    # have T/W ~ 1 in this mass model, which cannot descend; see SOLUTION.md.
    # A modelling assumption, not a measured engine limit.
    terminal_throttle_floor: float = 0.34

    # --- ignition / brake ------------------------------------------------
    # Ignite when the deceleration needed to reach the gate reaches this
    # fraction of what 13 engines can deliver. <1 leaves throttle headroom
    # for dispersions and for leaning over to divert.
    brake_margin: float = 0.45
    gate_altitude: float = 250.0          # absolute altitude of the handover
    v_gate: float = 8.0                   # descent rate at the handover
    max_tilt_brake: float = np.radians(30.0)
    # Hand over to 3 engines once the brake has slowed below this multiple of
    # v_gate. Waiting for the gate altitude with 13 engines at their floor
    # would over-brake and climb.
    handover_speed_factor: float = 1.6

    # --- terminal --------------------------------------------------------
    v_desc: float = 8.0                   # constant-descent segment
    a_flare: float = 0.55                 # flare deceleration, m/s^2
    v_touch: float = 0.30                 # descent rate at the catch plane
    k_v: float = 1.5                      # vertical velocity-error gain, 1/s
    max_tilt_terminal: float = np.radians(12.0)
    # Tilt envelope: allowed lean grows with time-to-go at this rate, so the
    # attitude loop always has time to bring the vehicle upright.
    tilt_envelope_rate: float = np.radians(1.2)   # rad per second of t_go
    tilt_envelope_floor: float = np.radians(0.25)
    # Minimum time-to-go the lateral law is evaluated at. The polynomial gains
    # scale as 1/T^2; flooring T keeps them finite, and the envelope above
    # keeps the resulting lean small.
    t_go_floor: float = 2.5

    # --- roll ------------------------------------------------------------
    # Body z (third grid fin) must point away from the tower at the catch.
    catch_roll_axis: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    roll_rate: float = np.radians(12.0)   # mean roll rate budget
    roll_min_duration: float = 6.0        # s


@dataclass
class LandingCommand:
    phase: str
    n_lit: int
    throttle: float
    direction: np.ndarray          # inertial thrust (= nose) direction
    roll_ref: np.ndarray           # inertial vector body z should lean toward
    info: dict


def poly_accel(r, v, r_f, v_f, a_f, T):
    """Acceleration now for a quadratic accel profile hitting (r_f, v_f, a_f)
    at time T. Vectorised over axes."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    return (np.asarray(a_f, float) - 6.0 * (np.asarray(v_f, float) - v) / T
            + 12.0 * (np.asarray(r_f, float) - r - v * T) / (T * T))


def _limit_tilt(f: np.ndarray, max_tilt: float) -> np.ndarray:
    """Clamp the horizontal part of a specific-force vector to a lean limit,
    keeping its vertical part (vertical has priority)."""
    fz = max(float(f[2]), 1e-3)
    fh = f[:2]
    nh = float(np.linalg.norm(fh))
    lim = fz * np.tan(max_tilt)
    if nh > lim:
        fh = fh * (lim / nh)
    return np.array([fh[0], fh[1], fz])


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return Q.rotate(Q.from_axis_angle(axis, angle), v)


class LandingGuidance:
    """Stateful landing-burn guidance. Call ``step`` once per control tick."""

    def __init__(self, cfg: LandingConfig, vehicle: Vehicle, aero: AeroModel):
        self.cfg = cfg
        self.vehicle = vehicle
        self.aero = aero
        self.reset()

    def reset(self):
        self.phase = COAST
        self.n_lit = 0
        self.ignition_alt = float("nan")
        self.ignition_speed = float("nan")
        self.ignition_t = float("nan")
        self.handover_alt = float("nan")
        self.handover_t = float("nan")
        self.roll_start = None

    # ------------------------------------------------------------------
    # vertical references

    def _terminal_profile(self, h_left: float):
        """Descent-rate reference, its time derivative factor, and t_go."""
        c = self.cfg
        h_left = max(h_left, 0.0)
        h_flare = (c.v_desc ** 2 - c.v_touch ** 2) / (2.0 * c.a_flare)
        t_flare = (c.v_desc - c.v_touch) / c.a_flare
        if h_left > h_flare:
            v_ref = c.v_desc
            t_go = (h_left - h_flare) / c.v_desc + t_flare
            in_flare = False
        else:
            v_ref = float(np.sqrt(c.v_touch ** 2 + 2.0 * c.a_flare * h_left))
            t_go = (v_ref - c.v_touch) / c.a_flare
            in_flare = True
        return v_ref, t_go, in_flare

    def _a_max_vertical(self, n: int, m: float, drag_up: float) -> float:
        return self.vehicle.axial_thrust(n, 1.0) / m - ENV.G0 + drag_up / m

    # ------------------------------------------------------------------

    def step(self, r, v, q, prop, t) -> LandingCommand:
        c = self.cfg
        self._t = t
        r = np.asarray(r, float)
        v = np.asarray(v, float)
        h = float(r[2])
        vz = float(v[2])
        m = self.vehicle.mass(prop)
        drag = self.aero.drag_force(v, h)            # guidance's ESTIMATE
        g_vec = np.array([0.0, 0.0, -ENV.G0])
        tgt = np.asarray(c.target, float)
        h_left = h - tgt[2]
        info = {}

        # ---------------- ignition test --------------------------------
        if self.phase == COAST:
            dh = h - c.gate_altitude
            a_need = ((vz * vz - c.v_gate ** 2) / (2.0 * max(dh, 1.0))
                      if vz < -c.v_gate else 0.0)
            a_max = self._a_max_vertical(c.n_brake, m, float(drag[2]))
            info.update(a_need=a_need, a_max=a_max)
            if dh <= 0.0 or a_need >= c.brake_margin * a_max:
                self.phase = BRAKE
                self.n_lit = c.n_brake
                self.ignition_alt = h
                self.ignition_speed = float(np.linalg.norm(v))
                self.ignition_t = t
            else:
                # Engines-first against the velocity: the aerodynamic trim
                # attitude, so the coast hold costs no RCS and ignition
                # starts with thrust already retrograde.
                sp = float(np.linalg.norm(v))
                d = -v / sp if sp > 1.0 else np.array([0.0, 0.0, 1.0])
                return LandingCommand(COAST, 0, 0.0, d,
                                      self._roll_keep(q, d), info)

        # ---------------- brake -> terminal handover --------------------
        if self.phase == BRAKE:
            slow = vz > -c.handover_speed_factor * c.v_gate
            low = h <= c.gate_altitude
            a_stop = vz * vz / (2.0 * max(h_left, 1.0)) + ENV.G0
            capable = (self.vehicle.axial_thrust(c.n_terminal, 1.0) / m
                       * 0.85 >= a_stop)
            if (slow or low) and capable:
                self.phase = TERMINAL
                self.n_lit = c.n_terminal
                self.handover_alt = h
                self.handover_t = t

        if h_left <= 0.0:
            self.phase = DONE

        # ---------------- vertical + lateral commands -------------------
        if self.phase == BRAKE:
            dh = max(h - c.gate_altitude, 1.0)
            # Energy matching to the gate, recomputed from the actual state.
            a_z = max((vz * vz - c.v_gate ** 2) / (2.0 * dh), 0.0)
            t_brake = 2.0 * dh / max(abs(vz) + c.v_gate, 1.0)
            # ONE lateral plan to the catch point for the whole remaining
            # descent: time-to-go is the brake plus the terminal profile from
            # the gate. Planning to the gate alone made T collapse to ~1 s
            # near the handover, the 12/T^2 gain exploded, and the vehicle
            # arrived at the handover leaning 29 deg and 77 m past the tower.
            _, t_term, _ = self._terminal_profile(c.gate_altitude - tgt[2])
            t_go = max(t_brake + t_term, c.t_go_floor)
            a_xy = poly_accel(r[:2], v[:2], tgt[:2], np.zeros(2), np.zeros(2),
                              t_go)
            max_tilt = c.max_tilt_brake
            floor = c.throttle_floor
        else:
            v_ref, t_go_v, in_flare = self._terminal_profile(h_left)
            # Feed-forward is the profile's own deceleration in the flare.
            a_ff = c.a_flare if in_flare else 0.0
            a_z = a_ff + c.k_v * ((-v_ref) - vz)
            t_go = max(t_go_v, c.t_go_floor)
            a_xy = poly_accel(r[:2], v[:2], tgt[:2], np.zeros(2),
                              np.zeros(2), t_go)
            max_tilt = min(c.max_tilt_terminal,
                           max(c.tilt_envelope_floor,
                               c.tilt_envelope_rate * t_go_v))
            floor = c.terminal_throttle_floor

        a_des = np.array([a_xy[0], a_xy[1], a_z])
        f = a_des - g_vec - drag / m                 # required specific force
        f = _limit_tilt(f, max_tilt)

        # Throttle: vertical first. If the full vector exceeds what the
        # engines can give, shed horizontal demand rather than vertical.
        f_avail = self.vehicle.axial_thrust(self.n_lit, 1.0) / m
        fn = float(np.linalg.norm(f))
        if fn > f_avail:
            fz = min(float(f[2]), f_avail)
            fh_max = np.sqrt(max(f_avail ** 2 - fz ** 2, 0.0))
            fh = f[:2]
            nh = float(np.linalg.norm(fh))
            if nh > fh_max and nh > 0.0:
                fh = fh * (fh_max / nh)
            f = np.array([fh[0], fh[1], fz])
            fn = float(np.linalg.norm(f))
        throttle = float(np.clip(fn / f_avail, floor, 1.0))
        direction = f / max(fn, 1e-9)

        info.update(t_go=t_go, a_x=float(a_des[0]), a_y=float(a_des[1]),
                    a_z=float(a_des[2]),
                    tilt_cmd=float(np.degrees(np.arccos(direction[2]))),
                    tilt_limit=float(np.degrees(max_tilt)),
                    zem=float(np.linalg.norm(tgt[:2] - r[:2] - v[:2] * t_go)))

        return LandingCommand(self.phase, self.n_lit, throttle, direction,
                              self._roll_ref(q, direction, h), info)

    # ------------------------------------------------------------------
    # roll

    @staticmethod
    def _roll_keep(q, direction):
        """Current body z projected normal to the thrust direction: a roll
        reference that asks for no roll motion at all."""
        z = Q.rotate(q, np.array([0.0, 0.0, 1.0]))
        z = z - float(z @ direction) * direction
        n = float(np.linalg.norm(z))
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return z / n

    def _roll_ref(self, q, direction, h):
        """
        Hold the incoming roll through the brake, then roll to the catch
        orientation on a smooth, TIME-based profile during the terminal
        descent.

        The coast hands over with the fin roll anywhere -- frequently ~180 deg
        from the catch orientation, since weathercocking does not care about
        roll. Two traps:

          * Re-deciding the roll direction every step with atan2 flips sign
            at +/-180 deg, so the reference chatters and then snaps through
            the whole manoeuvre at ~50 deg/s. The direction and total angle
            are therefore decided ONCE, at the handover.
          * Blending the start and end VECTORS linearly passes through zero
            for a 180 deg roll. The reference is instead the start vector
            rotated about the thrust axis by a smoothstep fraction of the
            total angle.
        """
        c = self.cfg
        keep = self._roll_keep(q, direction)
        if self.phase != TERMINAL:
            return keep
        want = np.asarray(c.catch_roll_axis, float)
        if self.roll_start is None:
            start = keep.copy()
            w = want - float(want @ direction) * direction
            nw = float(np.linalg.norm(w))
            ang = 0.0
            if nw > 1e-9:
                w /= nw
                ang = float(np.arctan2(float(np.cross(start, w) @ direction),
                                       float(start @ w)))
                if abs(ang) > np.radians(179.0):
                    ang = abs(ang)          # pick one direction, once
            dur = max(c.roll_min_duration, 1.5 * abs(ang) / c.roll_rate)
            self.roll_start = (start, ang, self.handover_t, dur)
        start, ang, t0, dur = self.roll_start
        u = float(np.clip((self._t - t0) / dur, 0.0, 1.0))
        u = u * u * (3.0 - 2.0 * u)
        ref = _rotate_about(start, direction, u * ang)
        ref = ref - float(ref @ direction) * direction
        n = float(np.linalg.norm(ref))
        return ref / n if n > 1e-9 else keep
