"""
Aerodynamics.

HONESTY NOTICE -- READ BEFORE TRUSTING ANY RESULT FROM THIS FILE
----------------------------------------------------------------
This is the weakest module in the project and it is not close. Nothing here is
measured. The drag coefficients are engineering guesses for a blunt cylinder
descending base-first, and the grid fin model is a placeholder with invented
authority.

Everything else in quatsim is either exact (quaternion algebra, rigid-body
dynamics), derived from a public figure, or derived from a measurement with the
derivation written down. This file is none of those. Results that depend
strongly on it -- terminal velocity, the aerodynamic control margin during the
unpowered coast, landing burn ignition altitude -- carry that uncertainty and
should be quoted with it.

THE INTERFACE IS THE POINT
--------------------------
GridFinModel exists so that a real aero database can replace it without
touching anything downstream. Swap the bodies of normal_force() and
hinge_moment(), keep the signatures, and the sequencer and controllers do not
change. CFD on the fin lattice is the natural way to produce that database.

If you do generate one, two things are worth checking in the results:

  1. Grid fins CHOKE transonically. Effectiveness drops sharply approaching
     Mach 1 and recovers supersonically. If your Cn-versus-Mach curve comes out
     monotonic, something is wrong with the run, not with the fins.
  2. Hinge moment matters as much as normal force. It sizes the actuators, and
     it is what saturates first in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import environment as ENV
from . import quaternion as Q


# ---------------------------------------------------------------------------
# body drag
# ---------------------------------------------------------------------------

def drag_coefficient(mach: float) -> float:
    """
    Axial drag coefficient for the booster descending base-first.

    GUESS, not data. Shape follows the usual pattern -- roughly flat
    subsonically, a transonic rise, a supersonic peak, then slow decay -- with
    magnitudes typical of a blunt cylinder. A real vehicle's curve would come
    from CFD or wind tunnel.
    """
    m = abs(float(mach))
    if m < 0.8:
        return 0.85
    if m < 1.2:
        return 0.85 + 0.75 * (m - 0.8) / 0.4      # transonic rise
    if m < 3.0:
        return 1.60 - 0.35 * (m - 1.2) / 1.8
    return 1.25


@dataclass
class AeroModel:
    """
    Axial drag on the vehicle body.

    reference_area is the base area of the cylinder. Grid fin drag is folded
    in crudely via fin_drag_factor rather than modelled separately, which is
    another reason not to read much into absolute numbers.
    """

    diameter: float = 9.0
    # CALIBRATED against Flight 9: landing-burn ignition at ~361 m/s
    # (1300 km/h). At the previous guess of 1.15 the vehicle arrived at 712 m/s
    # -- nearly double -- which saturated every terminal actuator and turned
    # the burn-duration basin into a knife edge. 2.0 reproduces the observed
    # ignition velocity to within 1%.
    #
    # Physically: the booster sheds far more energy on descent than a 15% fin
    # penalty implies -- three 50%-larger grid fins at high angle of attack,
    # plus body drag effects this axial-only model does not capture.
    fin_drag_factor: float = 2.00
    # Dispersion hooks. Nominal values reproduce the original model exactly.
    cd_scale: float = 1.0          # multiplies the whole drag coefficient
    density_scale: float = 1.0     # multiplies atmospheric density
    # Wind: callable altitude -> inertial wind vector (m/s), or None. Drag
    # acts on the AIR-relative velocity.
    wind: object | None = None

    def air_velocity(self, velocity: np.ndarray, altitude: float) -> np.ndarray:
        v = np.asarray(velocity, dtype=float)
        if self.wind is None:
            return v
        return v - np.asarray(self.wind(float(altitude)), dtype=float)

    @property
    def reference_area(self) -> float:
        return np.pi * (0.5 * self.diameter) ** 2

    def drag_force(self, velocity: np.ndarray, altitude: float,
                   fins_deployed: bool = True) -> np.ndarray:
        """
        Drag force vector, inertial frame, N.

        Opposes the velocity vector. Assumes the vehicle is aligned close to
        its velocity vector, which holds during the coast and fails badly at
        high angle of attack -- another limitation of treating drag as purely
        axial.
        """
        v = self.air_velocity(velocity, altitude)
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return np.zeros(3)

        mach = ENV.mach(v, altitude)
        cd = drag_coefficient(mach) * self.cd_scale
        if fins_deployed:
            cd *= self.fin_drag_factor

        q_dyn = (0.5 * ENV.density(altitude) * self.density_scale
                 * speed * speed)
        return -(q_dyn * cd * self.reference_area) * (v / speed)

    def terminal_velocity(self, mass: float, altitude: float = 0.0) -> float:
        """
        Steady-state descent speed where drag balances weight.

        Solved by fixed-point iteration because Cd depends on Mach, which
        depends on the speed being solved for.
        """
        v = 200.0
        for _ in range(60):
            cd = drag_coefficient(ENV.mach(np.array([0.0, 0.0, v]), altitude))
            cd *= self.fin_drag_factor
            denom = 0.5 * ENV.density(altitude) * cd * self.reference_area
            if denom <= 0:
                return float("inf")
            v_new = np.sqrt(mass * ENV.G0 / denom)
            if abs(v_new - v) < 1e-6:
                return float(v_new)
            v = 0.5 * (v + v_new)
        return float(v)


# ---------------------------------------------------------------------------
# grid fins -- interface, not a model
# ---------------------------------------------------------------------------

@dataclass
class GridFinModel:
    """
    Grid fin normal force and hinge moment.

    PLACEHOLDER. cn_alpha and cn_delta below are invented. Replace the bodies
    of the two methods with a real lookup and nothing else needs to change.

    Four fins in an X configuration. Two oppose pitch, two oppose yaw, and
    differential deflection gives roll -- which is the ONLY roll authority the
    vehicle has during the unpowered coast, since gimbals need lit engines.
    """

    n_fins: int = 4
    area_per_fin: float = 4.0          # m^2, planform, ESTIMATE
    moment_arm: float = 30.0           # m, fin plane to CoM, ESTIMATE
    deflection_limit: float = np.radians(20.0)

    cn_alpha: float = 2.0              # per rad, INVENTED
    cn_delta: float = 1.2              # per rad, INVENTED

    def effectiveness(self, mach: float) -> float:
        """
        Multiplier capturing transonic choking.

        The dip near Mach 1 is real physics, though its depth here is a guess.
        A monotonic curve would be the wrong shape entirely, so an approximate
        dip beats no dip.
        """
        m = abs(float(mach))
        if m < 0.7:
            return 1.0
        if m < 1.1:
            return 1.0 - 0.45 * (m - 0.7) / 0.4     # choking
        if m < 2.0:
            return 0.55 + 0.35 * (m - 1.1) / 0.9    # recovery
        return 0.90

    def normal_force(self, mach: float, alpha: float, deflection: float,
                     q_dyn: float) -> float:
        """Normal force from one fin pair, N. PLACEHOLDER."""
        d = float(np.clip(deflection, -self.deflection_limit, self.deflection_limit))
        cn = (self.cn_alpha * alpha + self.cn_delta * d) * self.effectiveness(mach)
        return float(cn * q_dyn * self.area_per_fin * 2.0)

    def control_torque(self, mach: float, q_dyn: float,
                       pitch_deflection: float, yaw_deflection: float) -> np.ndarray:
        """
        Body-frame torque from fin deflection, N m. PLACEHOLDER.

        Returns zero at zero dynamic pressure, which is the correct and
        important behaviour: high in the coast there is neither gimbal nor fin
        authority, and the vehicle is genuinely uncontrollable until it hits
        thicker air. That window is real and any honest sim should show it.
        """
        f_pitch = self.normal_force(mach, 0.0, pitch_deflection, q_dyn)
        f_yaw = self.normal_force(mach, 0.0, yaw_deflection, q_dyn)
        return np.array([0.0, f_pitch * self.moment_arm, -f_yaw * self.moment_arm])

    def max_torque(self, mach: float, q_dyn: float) -> float:
        """Peak available fin torque at full deflection, N m."""
        f = self.normal_force(mach, 0.0, self.deflection_limit, q_dyn)
        return float(abs(f * self.moment_arm))


def restoring_moment(q: np.ndarray, velocity: np.ndarray, altitude: float,
                     omega: np.ndarray | None = None,
                     cm_alpha: float = 0.35, cm_q: float = 12.0,
                     ref_area: float = 63.6,
                     ref_length: float = 72.3,
                     density_scale: float = 1.0) -> np.ndarray:
    """
    Aerodynamic weathercocking moment, body frame, N m.

    THE PHYSICS THAT WAS MISSING
    ----------------------------
    Grid fins sit forward of the centre of mass, so the vehicle's aerodynamic
    centre is behind the CoM relative to the direction of travel when falling
    engines-first. That makes engines-first a STABLE trim: any angle of attack
    generates a moment pushing back toward it, exactly like the feathers on an
    arrow.

    Without this term the booster has no reason to orient itself at all. RCS
    alone manages 0.07 deg/s^2 in pitch -- a 153 degree slew would take 94
    seconds of continuous bang-bang against a moving target, which is why the
    coast previously left the vehicle frozen in whatever attitude the boostback
    handed it. Real boosters barely steer during entry; the air does the work
    and the thrusters only trim.

    The moment is modelled as proportional to sin(AoA) about the axis that
    rotates the tail into the wind, which is the standard small-disturbance
    form and has the right sign and equilibrium everywhere. cm_alpha is an
    ESTIMATE -- this is the coefficient CFD on the fin lattice would replace.
    """
    v = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed < 1.0:
        return np.zeros(3)

    q_dyn = 0.5 * ENV.density(altitude) * density_scale * speed * speed
    if q_dyn < 1e-6:
        return np.zeros(3)

    # Body-frame velocity, and the tail direction we want aligned with it.
    v_body = Q.rotate_inverse(q, v / speed)
    tail = np.array([-1.0, 0.0, 0.0])          # engines lead

    axis = np.cross(tail, v_body)              # rotates tail toward velocity
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.zeros(3)

    moment = (cm_alpha * q_dyn * ref_area * ref_length * n) * (axis / n)

    # ---- PITCH DAMPING (Cm_q) ----
    # A restoring moment ALONE is a pendulum: it converts attitude error into
    # angular rate, overshoots, and oscillates forever. That is exactly what
    # showed up as the booster swinging back and forth through the descent
    # instead of settling upright.
    #
    # Real airframes are damped. Rotating at rate omega gives the nose and tail
    # different local incidence, and the resulting force difference opposes the
    # rotation. The standard form is proportional to omega * (L / 2V), so the
    # damping grows with dynamic pressure just as the restoring moment does --
    # which is why the oscillation was worst low down, where q_dyn is highest
    # and my model had stiffness but no damping.
    #
    # Only the component of omega PERPENDICULAR to the body axis is damped;
    # roll is not aerodynamically damped in this model.
    if omega is not None:
        w = np.asarray(omega, dtype=float)
        w_perp = np.array([0.0, w[1], w[2]])
        moment = moment - (cm_q * q_dyn * ref_area * ref_length
                           * ref_length / (2.0 * speed)) * w_perp

    return moment


def angle_of_attack(q: np.ndarray, velocity: np.ndarray) -> float:
    """
    Angle between the body x-axis and the velocity vector, radians.

    Computed from the quaternion directly rather than from Euler angles, which
    matters because the booster descends near 180 deg of angle of attack
    (engines first) -- precisely where an Euler-based calculation is worst.
    """
    v = np.asarray(velocity, dtype=float)
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        return 0.0
    nose = Q.rotate(q, np.array([1.0, 0.0, 0.0]))
    return float(np.arccos(np.clip(float(nose @ v) / s, -1.0, 1.0)))
