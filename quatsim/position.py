"""
Outer-loop position control for the hover-and-divert terminal phase.

THE CASCADE
-----------
A booster has no lateral thrusters. The only way to accelerate sideways is to
tilt the whole vehicle so that some of the thrust points sideways. So position
control cannot command force directly -- it commands an ATTITUDE, and the inner
attitude loop flies to it:

    position error  ->  desired inertial acceleration
                    ->  required thrust VECTOR (add gravity back)
                    ->  commanded attitude + throttle
                    ->  inner attitude loop (control.py)
                    ->  gimbal deflections

Which means a position command IS an attitude command, and the quaternion sits
in the middle of the loop rather than off to the side. This is the same
structure as quadrotor position control, and the same structure Falcon and
Super Heavy use on final approach.

TIMESCALE SEPARATION -- THE THING THAT BREAKS CASCADES
------------------------------------------------------
The outer loop must be substantially slower than the inner one, or it commands
attitudes faster than the vehicle can achieve them and the two loops fight.
The usual rule is a factor of 3 to 5 in bandwidth. With the attitude loop at
wn = 0.45 rad/s the position loop wants roughly wn = 0.10 rad/s, and going
faster is not a tuning preference -- it is how you get an oscillation that
looks like a controller bug but is actually a design error.

WHY THE DIVERT EXISTS AT ALL
----------------------------
The descent track is deliberately offset from the tower so that a landing-burn
failure drops the booster beside the tower rather than onto it. The cost of
that safety margin is that the vehicle has to arrive next to the chopsticks and
translate in under power, which is what this module flies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import environment as ENV
from .guidance import attitude_from_pointing
from .vehicle import Vehicle


@dataclass
class PositionGains:
    """
    Outer-loop shape. Keep wn well below the attitude loop's wn.

    max_tilt caps how far the controller will lean the vehicle to chase a
    lateral error. It is a real limit, not a safety bolt-on: leaning further
    trades vertical thrust for lateral, and at large tilt there is not enough
    vertical component left to hold altitude. It also keeps the commanded
    attitude inside the range where the inner loop is well behaved.
    """
    wn: float = 0.10
    zeta: float = 0.9
    max_tilt: float = np.radians(10.0)

    @property
    def kp(self) -> float:
        return self.wn ** 2

    @property
    def kd(self) -> float:
        return 2.0 * self.zeta * self.wn


@dataclass
class PositionOutput:
    q_cmd: np.ndarray          # attitude for the inner loop to track
    throttle: float            # clamped to the engine band
    thrust_cmd: float          # N, before clamping
    tilt_cmd: float            # rad from vertical, after limiting
    accel_cmd: np.ndarray      # m/s^2, inertial, what the law wanted
    throttle_saturated: bool


class PositionController:
    """
    Translational controller that outputs an attitude and a throttle.

    Stateless per call, like the attitude controller: everything it needs is
    an argument, so it can run at a lower rate than the integrator without
    bookkeeping.
    """

    def __init__(self, vehicle: Vehicle, gains: PositionGains | None = None,
                 throttle_floor: float = 0.40):
        self.vehicle = vehicle
        self.gains = gains or PositionGains()
        # Raptor cannot throttle arbitrarily deep. This floor is ASSUMED at
        # 40% and it is load-bearing: it is the reason a V3 booster cannot
        # hover on three engines (it would need 34%) and must use two.
        self.throttle_floor = float(throttle_floor)

    def update(
        self,
        r: np.ndarray,
        v: np.ndarray,
        r_ref: np.ndarray,
        v_ref: np.ndarray,
        prop_remaining: float,
        n_lit: int,
        a_ref: np.ndarray | None = None,
    ) -> PositionOutput:
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        g = self.gains

        # --- desired inertial acceleration -----------------------------
        a_des = (g.kp * (np.asarray(r_ref, float) - r)
                 + g.kd * (np.asarray(v_ref, float) - v))
        if a_ref is not None:
            a_des = a_des + np.asarray(a_ref, dtype=float)

        # --- convert to a required thrust vector -----------------------
        # Thrust must both produce a_des AND hold the vehicle up, so gravity
        # is added back here. Forgetting this term is the classic bug: the
        # vehicle tracks lateral position beautifully and falls out of the sky.
        m = self.vehicle.mass(prop_remaining)
        a_total = a_des - ENV.gravity(r)
        thrust_vec = m * a_total

        # --- limit the tilt --------------------------------------------
        mag = float(np.linalg.norm(thrust_vec))
        if mag < 1e-9:
            thrust_vec = np.array([0.0, 0.0, 1.0])
            mag = 1.0

        direction = thrust_vec / mag
        tilt = float(np.arccos(np.clip(direction[2], -1.0, 1.0)))

        if tilt > g.max_tilt:
            # Rotate the commanded direction back toward vertical about the
            # axis perpendicular to both, preserving the azimuth of the lean.
            up = np.array([0.0, 0.0, 1.0])
            axis = np.cross(up, direction)
            n = np.linalg.norm(axis)
            if n > 1e-9:
                axis /= n
                c, s = np.cos(g.max_tilt), np.sin(g.max_tilt)
                # Rodrigues, rotating `up` by max_tilt about `axis`
                direction = (up * c + np.cross(axis, up) * s
                             + axis * float(axis @ up) * (1 - c))
                direction /= np.linalg.norm(direction)
            tilt = g.max_tilt

        # --- attitude and throttle -------------------------------------
        # Body +x is the thrust direction, so pointing the nose along the
        # required thrust vector is the whole conversion.
        q_cmd = attitude_from_pointing(direction)

        # Only the component along the (possibly limited) thrust direction
        # counts toward what we actually need.
        needed = float(thrust_vec @ direction)
        full = self.vehicle.axial_thrust(n_lit, 1.0)
        raw = needed / full if full > 0 else 0.0
        throttle = float(np.clip(raw, self.throttle_floor, 1.0))
        saturated = bool(raw < self.throttle_floor or raw > 1.0)

        return PositionOutput(
            q_cmd=q_cmd,
            throttle=throttle,
            thrust_cmd=needed,
            tilt_cmd=tilt,
            accel_cmd=a_des,
            throttle_saturated=saturated,
        )


def divert_profile(r_start: np.ndarray, r_target: np.ndarray,
                   duration: float, t: float):
    """
    Smoothstep translation reference from one position to another.

    Same shape as the attitude slews: zero velocity AND zero acceleration at
    both ends, so the vehicle eases into the divert and settles out of it
    rather than stepping. Returns (r_ref, v_ref, a_ref).
    """
    r0 = np.asarray(r_start, dtype=float)
    r1 = np.asarray(r_target, dtype=float)
    d = r1 - r0

    if duration <= 0:
        return r1.copy(), np.zeros(3), np.zeros(3)

    u = float(np.clip(t / duration, 0.0, 1.0))
    inside = 0.0 < u < 1.0

    s = u * u * (3.0 - 2.0 * u)
    ds = 6.0 * u * (1.0 - u) / duration if inside else 0.0
    d2s = (6.0 - 12.0 * u) / (duration ** 2) if inside else 0.0

    return r0 + d * s, d * ds, d * d2s
