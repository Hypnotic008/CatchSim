"""
Attitude control: quaternion-error PD, gain-scheduled on instantaneous inertia.

THE CONTROL LAW
---------------
    dq  = q_actual* (X) q_cmd          (error, in BODY axes, canonicalized)
    tau = Kp * (2 * dq_vec) + Kd * (omega_ref - omega)

2*dq_vec is the small-angle rotation-vector error, so for small errors the
closed loop is a second-order system per axis:

    I thetaddot + Kd thetadot + Kp theta = 0

WHY THE GAINS ARE NOT NUMBERS
-----------------------------
Hand-tuning Kp and Kd is how most student projects do this, and it produces
gains that are meaningless outside the one condition they were tuned at. Here
they are DERIVED from two physically meaningful quantities:

    Kp = I * wn^2                 wn    = natural frequency, rad/s
    Kd = 2 * zeta * wn * I        zeta  = damping ratio

Specify how fast you want the loop (wn) and how damped (zeta), and the gains
follow from whatever I happens to be at that instant. Because Super Heavy's
transverse inertia falls by roughly 2.4x between separation and catch, a fixed
Kp that is correctly damped at separation is badly UNDER-damped at the catch --
the vehicle would oscillate exactly when precision matters most. Recomputing
from I(t) each step makes that failure mode structurally impossible.

This is textbook gain scheduling, and the booster is an unusually clean case
for it because the scheduling variable (propellant remaining) is known exactly.

WHAT THIS MODULE CANNOT DO
--------------------------
Gimballing the engine cluster as modelled produces pitch and yaw torque only.
Roll requires either differential tangential deflection across the ring or a
separate cold-gas system, neither of which is modelled here. Roll torque
commands are reported in the diagnostics as unallocated rather than silently
dropped -- pretending an actuator exists is worse than admitting it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quaternion as Q
from .actuators import RollAuthority
from .vehicle import Vehicle


@dataclass
class ControlGains:
    """
    Loop shape, specified physically rather than as raw gains.

    wn      natural frequency, rad/s. Bounded above by actuator bandwidth and
            by structural bending modes -- a real vehicle must keep the control
            loop well below its first bending frequency or it will cheerfully
            excite the airframe. Not modelled here, but it is the reason wn
            cannot simply be raised until tracking looks good.
    zeta    damping ratio. 0.7 is the usual compromise between rise time and
            overshoot; 1.0 is critically damped, slower but no overshoot.
    """

    wn: float = 0.45
    zeta: float = 0.75

    def kp(self, inertia: np.ndarray) -> np.ndarray:
        return inertia * (self.wn ** 2)

    def kd(self, inertia: np.ndarray) -> np.ndarray:
        return 2.0 * self.zeta * self.wn * inertia


@dataclass
class ControlOutput:
    """Everything the controller decided, for logging and diagnosis."""
    torque_cmd: np.ndarray        # what the law asked for, body axes
    torque_applied: np.ndarray    # what the actuators can actually deliver
    gimbal_pitch: float           # rad, after clamping
    gimbal_yaw: float             # rad, after clamping
    saturated: bool
    unallocated_roll: float       # N m of roll torque NO actuator could supply
    error_angle: float            # rad, geodesic distance to the reference
    roll_deflection: float = 0.0  # rad of tangential engine cant
    rcs_fraction: float = 0.0     # signed RCS duty, -1..1


class AttitudeController:
    """
    Gain-scheduled quaternion PD with gimbal allocation and saturation.

    Call update() once per control tick. The controller does NOT integrate
    anything; it is a pure function of the current state, which keeps it
    trivially testable and means it can be called at a lower rate than the
    dynamics propagator without any bookkeeping.
    """

    def __init__(self, vehicle: Vehicle, gains: ControlGains | None = None,
                 roll: RollAuthority | None = None):
        self.vehicle = vehicle
        self.gains = gains or ControlGains()
        # Roll actuation is SEPARATE from the pitch/yaw gimbal because it is
        # physically separate hardware: tangential engine cant and RCS, not
        # cluster deflection. Without it a manoeuvre with any roll content --
        # the post-separation flip is 21.6% roll -- simply cannot be flown.
        self.roll = roll or RollAuthority()

    # ------------------------------------------------------------------

    def torque_command(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_ref: np.ndarray,
        omega_ref: np.ndarray,
        inertia: np.ndarray,
        alpha_ref: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        The PD law with feedforward. Returns (torque, error_angle).

        Three terms, and all three earn their place:

          Kp * err        pulls the attitude toward the reference
          Kd * rate_err   damps it, using (omega_ref - omega) rather than
                          -omega so a constant-rate slew is not fought
          I * alpha_ref   supplies the torque the reference MANEUVER needs,
                          independent of any error

        Dropping the third term costs about alpha_ref / wn^2 of steady lag --
        16 degrees on an 18 s, 180 deg slew at wn = 0.45. Without it the loop
        can only produce the reference acceleration by first allowing the
        error that generates it, which defeats the point of having a reference.
        """
        dq = Q.attitude_error(q, q_ref)
        err_vec = 2.0 * dq[1:]
        err_angle = float(np.linalg.norm(Q.to_rotvec(dq)))

        kp = self.gains.kp(inertia)
        kd = self.gains.kd(inertia)

        tau = kp @ err_vec + kd @ (np.asarray(omega_ref, dtype=float)
                                   - np.asarray(omega, dtype=float))
        if alpha_ref is not None:
            tau = tau + inertia @ np.asarray(alpha_ref, dtype=float)
        return tau, err_angle

    # ------------------------------------------------------------------

    def allocate(
        self,
        torque_cmd: np.ndarray,
        n_gimballing: int,
        throttle: float,
    ) -> tuple[np.ndarray, float, float, bool, float]:
        """
        Turn a desired body torque into gimbal deflections.

        Inverting the forward model in vehicle.gimbal_torque:

            tau_y = +arm * T * sin(pitch_deflection)
            tau_z = -arm * T * sin(yaw_deflection)

        so the deflections come straight from arcsin, clamped to the mechanical
        limit. If no engines are lit there is no thrust and therefore no
        authority at all -- that is the unpowered coast, and it is handled by
        returning zero rather than by dividing by zero.
        """
        v = self.vehicle
        n = min(n_gimballing, v.n_gimballing_max)
        thrust = v.axial_thrust(n, throttle)

        # ---- roll, via differential gimbal then RCS ----
        roll_applied, roll_defl, rcs_frac, roll_sat = self.roll.allocate_roll(
            v, float(torque_cmd[0]), n_gimballing, throttle
        )
        unallocated_roll = float(torque_cmd[0]) - roll_applied

        if thrust <= 0.0 or n <= 0:
            # Engines out. No gimbal at all, so pitch and yaw fall back to RCS.
            # This is the coast: the vehicle is not uncontrollable, it is
            # controllable WEAKLY, and the difference matters. RCS can hold an
            # attitude against nothing much; it cannot fly a fast slew.
            tau_py = self.roll.max_rcs_pitch_yaw()
            py = np.clip(torque_cmd[1:], -tau_py, tau_py)
            sat = bool(np.any(np.abs(torque_cmd[1:]) > tau_py))
            return (np.array([roll_applied, py[0], py[1]]), 0.0, 0.0,
                    sat, unallocated_roll, roll_defl, rcs_frac)

        denom = v.gimbal_arm * thrust
        s_pitch = torque_cmd[1] / denom
        s_yaw = -torque_cmd[2] / denom

        saturated = bool(abs(s_pitch) > np.sin(v.gimbal_limit)
                         or abs(s_yaw) > np.sin(v.gimbal_limit))

        pitch = np.arcsin(np.clip(s_pitch, -1.0, 1.0))
        yaw = np.arcsin(np.clip(s_yaw, -1.0, 1.0))
        pitch = float(np.clip(pitch, -v.gimbal_limit, v.gimbal_limit))
        yaw = float(np.clip(yaw, -v.gimbal_limit, v.gimbal_limit))

        applied = v.gimbal_torque(n, pitch, yaw, throttle)
        applied[0] += roll_applied
        return (applied, pitch, yaw, saturated or roll_sat, unallocated_roll,
                roll_defl, rcs_frac)

    # ------------------------------------------------------------------

    def update(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_ref: np.ndarray,
        omega_ref: np.ndarray,
        prop_remaining: float,
        n_gimballing: int,
        throttle: float = 1.0,
        alpha_ref: np.ndarray | None = None,
        wn_override: float | None = None,
    ) -> ControlOutput:
        """
        One control tick.

        Inertia is recomputed from prop_remaining every call -- that IS the
        gain scheduling. There is no separate scheduling table, no
        interpolation, and no way for the gains to go stale.
        """
        inertia = self.vehicle.inertia(prop_remaining)

        # Bandwidth must be scheduled on the ACTUATOR, not just the inertia.
        # Gimbals support wn above 1 rad/s; RCS supports roughly 0.03. Running
        # the gimbal bandwidth on RCS commands torque two orders of magnitude
        # beyond what exists, saturating continuously and leaving the loop
        # effectively open -- the same failure that broke the flip, in a
        # different costume.
        gains = self.gains
        if wn_override is not None:
            gains = ControlGains(wn=wn_override, zeta=self.gains.zeta)
        elif n_gimballing <= 0 or throttle <= 0.0:
            tau = max(self.roll.max_rcs_pitch_yaw(), 1.0)
            gains = ControlGains(wn=float(np.sqrt(tau / inertia[1, 1] / 0.5)),
                                 zeta=self.gains.zeta)

        saved, self.gains = self.gains, gains
        try:
            tau_cmd, err = self.torque_command(q, omega, q_ref, omega_ref,
                                              inertia, alpha_ref)
        finally:
            self.gains = saved
        (applied, pitch, yaw, sat, roll,
         roll_defl, rcs_frac) = self.allocate(tau_cmd, n_gimballing, throttle)
        return ControlOutput(
            torque_cmd=tau_cmd,
            torque_applied=applied,
            gimbal_pitch=pitch,
            gimbal_yaw=yaw,
            saturated=sat,
            unallocated_roll=roll,
            error_angle=err,
            roll_deflection=roll_defl,
            rcs_fraction=rcs_frac,
        )


# ---------------------------------------------------------------------------
# authority analysis -- do this BEFORE tuning, not after
# ---------------------------------------------------------------------------

def max_feasible_wn(vehicle: Vehicle, prop_remaining: float,
                    n_gimballing: int, throttle: float,
                    design_error: float) -> float:
    """
    Largest natural frequency that will not saturate at a given error.

    At error theta the law asks for tau = I * wn^2 * theta, and the gimbals can
    deliver at most tau_max. Setting them equal:

        wn_max = sqrt(tau_max / (I * theta))

    Running this first tells you what loop bandwidth the vehicle can actually
    support, which beats picking gains, watching the response, and guessing why
    it looks wrong. For large slews the answer is uncomfortably small -- which
    is the quantitative reason guidance must supply a feasible reference
    instead of step-commanding the destination.
    """
    tau_max = float(np.linalg.norm(
        vehicle.gimbal_torque(n_gimballing, vehicle.gimbal_limit, 0.0, throttle)
    ))
    iyy = vehicle.inertia(prop_remaining)[1, 1]
    if design_error <= 0 or iyy <= 0:
        return float("inf")
    return float(np.sqrt(tau_max / (iyy * design_error)))


def slew_time_estimate(vehicle: Vehicle, prop_remaining: float,
                       n_gimballing: int, throttle: float,
                       angle: float) -> float:
    """
    Bang-bang lower bound on slew duration: accelerate at full gimbal for half
    the angle, decelerate for the other half.

        t = 2 * sqrt(angle / alpha_max)

    No controller can beat this with these actuators, so it is the floor on any
    guidance profile you design. If your commanded slew duration is anywhere
    near this number, the gimbals will be saturated for most of the maneuver
    and the loop is effectively open.
    """
    alpha = vehicle.max_angular_accel(n_gimballing, prop_remaining, throttle)
    if alpha <= 0:
        return float("inf")
    return float(2.0 * np.sqrt(angle / alpha))
