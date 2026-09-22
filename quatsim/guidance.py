"""
Attitude guidance -- generating the reference the controller tracks.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
The textbook argument for reference tracking is that step-commanding the
destination saturates the actuators. MEASURED ON THIS VEHICLE, THAT ARGUMENT
IS FALSE, and it is worth recording why rather than repeating it.

Super Heavy has enormous control authority: 13 gimballing Raptor 3s on a 30 m
moment arm give about 54 deg/s^2, so a bang-bang 180 deg flip has a floor of
only 3.7 seconds. Step-commanding the full flip at wn = 0.45 peaks at 6.4 deg
of gimbal against a 15 deg limit and never saturates at all. The flip is not
an authority-limited maneuver.

The real reasons to track a reference here are different, and better:

  1. TRAJECTORY SHAPE. A step command flies whatever path the closed-loop
     dynamics produce. A reference flies the path you chose, at the timing you
     chose, which is what lets the flip finish when boostback ignition needs it
     to rather than whenever the loop happens to settle.

  2. SMALL-SIGNAL VALIDITY. The 2*dq_vec error is a small-angle approximation
     to the rotation vector. At pi radians of error it is not small, the
     effective gain is not what you designed, and the second-order analysis
     behind Kp and Kd stops describing the system.

  3. HEADROOM FOR DISTURBANCES. Spending 6.4 of 15 degrees on the nominal
     maneuver leaves less for the thing you actually need gimbal authority
     for -- an engine-out, an ignition asymmetry, aerodynamic torque.

  4. BANDWIDTH. Step-commanding DOES saturate once wn rises: at wn = 0.9 the
     gimbals hit their stops 1.8% of the time. Reference tracking stays clean.

So the split -- guidance makes a feasible plan, control corrects deviations
from it -- is still right, for reasons this vehicle's numbers actually support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quaternion as Q


@dataclass
class Slew:
    """
    A constant-rate rotation from one attitude to another.

    SLERP gives constant angular VELOCITY, which means the rate steps
    discontinuously at both ends -- infinite angular acceleration on paper.
    Set smooth=True to ease in and out with a smoothstep in the interpolation
    parameter, which keeps the commanded acceleration finite and is much
    kinder to the actuators.
    """

    q_start: np.ndarray
    q_end: np.ndarray
    t_start: float
    duration: float
    smooth: bool = True

    @property
    def t_end(self) -> float:
        return self.t_start + self.duration

    @property
    def angle(self) -> float:
        """Total rotation, radians, along the short arc."""
        return Q.angle_between(self.q_start, self.q_end)

    @property
    def mean_rate(self) -> float:
        """Average angular rate, rad/s. Peak is 1.5x this when smooth=True."""
        return self.angle / self.duration if self.duration > 0 else 0.0

    def _s(self, t: float) -> tuple[float, float, float]:
        """Interpolation parameter and its first two time derivatives."""
        if self.duration <= 0:
            return 1.0, 0.0, 0.0
        u = float(np.clip((t - self.t_start) / self.duration, 0.0, 1.0))
        inside = 0.0 < u < 1.0
        if not self.smooth:
            du = 1.0 / self.duration if inside else 0.0
            return u, du, 0.0
        # smoothstep: 3u^2 - 2u^3, zero slope at both ends
        s = u * u * (3.0 - 2.0 * u)
        ds = 6.0 * u * (1.0 - u) / self.duration
        d2s = (6.0 - 12.0 * u) / (self.duration ** 2) if inside else 0.0
        return s, ds, d2s

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reference attitude, BODY-frame angular rate, and BODY-frame angular
        ACCELERATION at time t.

        The rate is the derivative of the slerp path. Because a slerp rotates
        about an axis fixed in the inertial frame, the body rate is that axis
        pulled back into body coordinates and scaled by the current rate.

        The acceleration term matters more than it looks. Rate feedforward
        alone leaves a steady tracking error of roughly alpha_ref / wn^2 --
        for an 18 s, 180 deg slew at wn = 0.45 that is about 16 degrees, which
        is not a subtle lag. The controller was being asked to manufacture the
        reference acceleration out of position error, and position error is
        exactly what it is supposed to be driving to zero.

        Conveniently the body-frame acceleration is just the same axis scaled
        by d2s: the extra term from the rotating frame is -omega_b x omega_b,
        which vanishes identically.
        """
        s, ds, d2s = self._s(t)
        q_ref = Q.slerp(self.q_start, self.q_end, s)

        axis, total = Q.to_axis_angle(
            Q.multiply(Q.conjugate(self.q_start), self.q_end)
        )
        # axis is expressed in q_start's body frame; rotate into the current
        # reference body frame, which differs from q_start by the slerp so far.
        dq = Q.multiply(Q.conjugate(q_ref), self.q_start)
        axis_now = Q.rotate(dq, axis)

        omega_ref = axis_now * (total * ds)
        alpha_ref = axis_now * (total * d2s)
        return q_ref, omega_ref, alpha_ref


class AttitudeSchedule:
    """
    An ordered list of slews with holds in between.

    Before the first slew the reference is its start attitude; after the last
    it is that slew's end attitude. Gaps between slews hold the previous
    endpoint at zero rate. That hold behaviour matters during the unpowered
    coast, where there is no gimbal authority and the reference should not be
    demanding motion the vehicle cannot produce.
    """

    def __init__(self, slews: list[Slew]):
        self.slews = sorted(slews, key=lambda s: s.t_start)
        if not self.slews:
            raise ValueError("schedule needs at least one slew")

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if t <= self.slews[0].t_start:
            return Q.normalize(self.slews[0].q_start), np.zeros(3), np.zeros(3)

        for s in self.slews:
            if s.t_start <= t <= s.t_end:
                return s.at(t)

        prev = [s for s in self.slews if s.t_end < t]
        if prev:
            return Q.normalize(prev[-1].q_end), np.zeros(3), np.zeros(3)
        return Q.normalize(self.slews[0].q_start), np.zeros(3), np.zeros(3)

    @property
    def t_end(self) -> float:
        return max(s.t_end for s in self.slews)


# ---------------------------------------------------------------------------
# helpers for building booster maneuvers
# ---------------------------------------------------------------------------

def attitude_from_pointing(x_body_in_inertial: np.ndarray,
                           roll_reference: np.ndarray | None = None) -> np.ndarray:
    """
    Build an attitude that points the body x-axis (the nose) along a given
    inertial direction.

    Pointing fixes only two of three degrees of freedom -- the rotation about
    the pointing axis is still free -- so a roll reference is needed to pin it
    down. Defaults to inertial +z, which fails if the nose points along z, so
    the fallback switches to +x in that case.
    """
    x = np.asarray(x_body_in_inertial, dtype=float)
    n = np.linalg.norm(x)
    if n < 1e-12:
        raise ValueError("pointing direction must be nonzero")
    x = x / n

    ref = np.array([0.0, 0.0, 1.0]) if roll_reference is None \
        else np.asarray(roll_reference, dtype=float)
    if abs(float(x @ ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])

    z = ref - float(ref @ x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)

    # columns are the body axes expressed in inertial coordinates
    return Q.from_dcm(np.column_stack([x, y, z]))


def retrograde_attitude(velocity_inertial: np.ndarray) -> np.ndarray:
    """
    Attitude with the nose pointed opposite the velocity vector.

    This is the boostback pointing condition: engines are aft, so pointing the
    nose retrograde puts the thrust along the velocity vector and kills
    downrange speed.
    """
    return attitude_from_pointing(-np.asarray(velocity_inertial, dtype=float))
