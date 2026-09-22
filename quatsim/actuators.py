"""
Roll authority -- the axis the main gimbal model cannot reach.

THE PROBLEM THIS SOLVES
-----------------------
Deflecting the whole engine cluster in pitch and yaw produces pitch and yaw
torque and NOTHING about the roll axis. That is fine until you notice that the
post-separation flip is a 180 degree rotation whose body axis is 21.6% roll --
about 39 degrees of pure roll buried inside the manoeuvre. With no roll
actuator the controller commands roll torque every step, none of it is applied,
and the flip settles roughly 48 degrees off target no matter how many engines
are lit. Adding authority in pitch and yaw does not help, because pitch and yaw
were never the problem.

FOUR ACTUATORS, AND WHEN EACH ONE WORKS
---------------------------------------
    DIFFERENTIAL GIMBAL   engines lit    strongest by far
    RCS / cold gas        always         weak but altitude-independent
    GRID FINS             q_dyn > 0      strong low down, useless in vacuum
    CHINES                q_dyn > 0      body lift; not modelled here

The flip happens above 70 km where there is effectively no dynamic pressure, so
fins and chines contribute nothing and are deliberately excluded rather than
faked. That leaves differential gimbal while engines are lit, and RCS when they
are not -- which is also the real division of labour on a booster.

DIFFERENTIAL GIMBAL IS THE ONE THAT MATTERS
-------------------------------------------
Canting the engines TANGENTIALLY rather than all in the same direction turns
the ring into a roll couple. The torque is (number of engines) x (thrust each)
x sin(deflection) x (ring radius), and because it is multiplied by the main
engine thrust it is two orders of magnitude stronger than any cold gas system.
It costs nothing extra: the actuators are already there, this is just a
different pattern of commands to them.

SIZING HONESTY
--------------
RCS thrust and the ring radius below are estimates. SpaceX does not publish
them. The differential-gimbal numbers follow from engine thrust and geometry
that are public, so they are firmer. Treat the RCS figure as a parameter to
tune rather than a specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vehicle import Vehicle


@dataclass
class RollAuthority:
    """
    Combined roll actuation: differential gimbal plus RCS.

    ring_radius is where the gimballing engines sit, not the body radius --
    the inner 13 are well inboard of the skin, which is why differential gimbal
    has a shorter arm than grid fins but vastly more force behind it.
    """

    ring_radius: float = 2.3            # m, inner-ring engine circle, ESTIMATE
    differential_limit: float = np.radians(5.0)   # tangential cant available

    rcs_thrust: float = 7_000.0         # N per couple, ESTIMATE
    rcs_arm: float = 4.5                # m, body radius (roll couple)
    pitch_yaw_arm: float = 25.0         # m, CoM to thruster pod, ESTIMATE
    rcs_available: bool = True

    # ------------------------------------------------------------------

    def max_differential_roll(self, vehicle: Vehicle, n_gimballing: int,
                              throttle: float) -> float:
        """
        Peak roll torque from tangential engine cant, N m.

        Zero when nothing is lit, which is correct and important: through the
        unpowered coast this actuator simply does not exist.
        """
        n = min(n_gimballing, vehicle.n_gimballing_max)
        if n <= 0 or throttle <= 0.0:
            return 0.0
        thrust = vehicle.axial_thrust(n, throttle)
        return float(thrust * np.sin(self.differential_limit) * self.ring_radius)

    def max_rcs_roll(self) -> float:
        """Peak roll torque from the RCS couple, N m. Altitude-independent."""
        return float(self.rcs_thrust * self.rcs_arm) if self.rcs_available else 0.0

    def max_rcs_pitch_yaw(self) -> float:
        """
        Peak RCS pitch/yaw torque, N m.

        Longer arm than roll: a roll couple works at the body radius, but
        pitch and yaw thrusters act at the distance from the CoM to the
        thruster pod, tens of metres away. So RCS is considerably better at
        pitch and yaw than at roll -- the opposite of the gimbal, which does
        pitch and yaw well and roll not at all.

        This is the ONLY three-axis actuator that works in vacuum with the
        engines out, which makes it the only thing holding attitude through
        the coast. Without it the booster tumbles, or worse, sits frozen in
        whatever attitude the boostback left it in.
        """
        if not self.rcs_available:
            return 0.0
        return float(self.rcs_thrust * self.pitch_yaw_arm)

    def allocate_roll(self, vehicle: Vehicle, roll_cmd: float,
                      n_gimballing: int, throttle: float) -> tuple:
        """
        Satisfy a roll torque command, differential gimbal first.

        Returns (applied_torque, differential_deflection, rcs_fraction,
        saturated).

        Gimbal first because it is free -- the engines are already burning and
        the actuators are already there -- and because it is roughly two orders
        of magnitude stronger. RCS picks up the remainder and is the only
        option when the engines are out.
        """
        cmd = float(roll_cmd)
        want = abs(cmd)
        sign = np.sign(cmd) if cmd != 0.0 else 0.0

        tau_diff_max = self.max_differential_roll(vehicle, n_gimballing, throttle)
        from_diff = min(want, tau_diff_max)
        deflection = 0.0
        if tau_diff_max > 0.0:
            deflection = sign * self.differential_limit * (from_diff / tau_diff_max)

        remaining = want - from_diff
        tau_rcs_max = self.max_rcs_roll()
        from_rcs = min(remaining, tau_rcs_max)
        rcs_fraction = sign * (from_rcs / tau_rcs_max) if tau_rcs_max > 0 else 0.0

        applied = sign * (from_diff + from_rcs)
        saturated = bool(want > tau_diff_max + tau_rcs_max + 1e-9)
        return float(applied), float(deflection), float(rcs_fraction), saturated

    # ------------------------------------------------------------------

    def report(self, vehicle: Vehicle, n_gimballing: int, throttle: float,
               inertia_xx: float) -> str:
        td = self.max_differential_roll(vehicle, n_gimballing, throttle)
        tr = self.max_rcs_roll()
        return (
            f"roll authority ({n_gimballing} gimballing @ {throttle:.0%})\n"
            f"  differential gimbal {td:10.3e} N m  "
            f"-> {np.degrees(td / inertia_xx):7.3f} deg/s^2\n"
            f"  RCS                 {tr:10.3e} N m  "
            f"-> {np.degrees(tr / inertia_xx):7.3f} deg/s^2\n"
            f"  ratio               {td / tr if tr > 0 else float('inf'):10.1f}x\n"
        )
