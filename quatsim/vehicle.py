"""
Vehicle mass properties, engine sequencing, and propellant budgeting.

BODY AXES (right-handed, origin at the instantaneous centre of mass)
    x : longitudinal, positive toward the NOSE
    y : lateral
    z : completes the triad

Roll is about x; pitch and yaw are about y and z. The transverse moments are
roughly 40x the axial one, which is the single most important number here --
it is why the booster is sluggish in pitch and nearly free in roll, and it
sets the scale of every control gain in Stage 3.

CONFIGURATION MODELLED: Super Heavy V3 / Block 3, Raptor 3.

TWO ENGINE COUNTS, AND THEY ARE NOT THE SAME NUMBER
----------------------------------------------------
    n_lit        engines running -> axial thrust and propellant flow
    n_gimballing engines with actuators -> CONTROL TORQUE

On Raptor 2 only the inner 13 could relight at all. On Raptor 3 all 33 relight,
and Flight 13 flew the high-thrust portion of boostback on all 33. But the
gimbal actuators are documented for the inner 13; whether that changed on V3 is
NOT something this file can confirm. So the two counts are tracked separately.
Passing 33 where 13 belongs overstates pitch authority by a factor of 2.5.

VARIABLE MASS
-------------
Propellant drains, so m and I are functions of time. The exact variable-mass
rigid-body equations carry extra terms for momentum leaving through the nozzle
and for CoM motion relative to the structure. This module uses the standard
approximation: evaluate I(t) each step, feed it to the FIXED-mass Euler
equations. Defensible because the exhaust leaves nearly axially near the
centreline. Still an approximation, stated rather than hidden.

SOURCING
--------
Geometry, propellant capacity and Raptor 3 thrust are public figures. Isp is an
assumption (350 s vacuum) and every derived mass number scales with it -- a 5%
Isp error moves them all by 5%. Burn durations are NOT public: the defaults
below come from livestream timing of Flight 13, which flew off-nominal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# flight phases
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """
    One segment of the return sequence.

    n_lit drives thrust and propellant flow; n_gimballing drives control
    torque. During the flip both matter and they differ.
    """
    name: str
    n_lit: int
    n_gimballing: int
    duration: float          # s
    throttle: float = 1.0


# Flight 13 as flown, from livestream timing.
#
# OFF-NOMINAL. The boostback ended early and the booster came down roughly
# 30 km short of its Gulf target, and it was aiming at water rather than the
# tower. Treat these durations as a FLOOR on a nominal RTLS profile.
#
# This profile is useful precisely BECAUSE it is off-nominal: the observed
# miss distance is a validation target the model was not fitted to.
#
# PROVENANCE OF EVERY NUMBER BELOW
#   boostback durations : measured from livestream (Flight 13, V3)
#   landing durations   : scaled from a V2 observation, see note there
#   throttles           : inferred from thrust-to-weight, not observed
#   coast duration      : placeholder, falls out of the Stage 4 trajectory
FLIGHT_13_ACTUAL: list[Phase] = [
    Phase("hot_stage",       3,  3,   6.0, 0.40),

    # Boostback, 33 -> 13 -> 3. Durations from livestream timing of Flight 13.
    # This whole sequence IS the boostback; it ends in shutdown.
    Phase("boostback_33",   33, 13,  11.0, 1.00),
    Phase("boostback_13",   13, 13,   3.0, 1.00),
    Phase("boostback_3",     3,  3,   3.0, 1.00),

    # NO ENTRY BURN. Super Heavy does not have one, and that is a design
    # feature rather than an omission: Starship is a large enough fraction of
    # the stack that the booster stages far slower than a Falcon 9 first stage,
    # so it never accumulates the reentry heating that forces Falcon 9 to spend
    # propellant slowing down before thick air.
    #
    # CONSEQUENCE FOR THE CONTROLLER: this phase has ZERO gimbal authority.
    # Every bit of attitude control from boostback shutdown to landing-burn
    # ignition is aerodynamic, on grid fins alone. It is the longest and
    # hardest control segment in the flight.
    Phase("coast_entry",     0,  0, 200.0, 0.00),

    # Landing burn, 13 -> 3. SCALED FROM V2 OBSERVATION, NOT MEASURED ON V3.
    # Observed on a V2 booster: 7 s on 13 engines, 15 s on 3. Raptor 3 makes
    # 1.217x Raptor 2 thrust, so at equal throttle V3 needs 0.82x the duration
    # for the same delta-v. The throttle band is inferred from T/W: at full
    # throttle this burn would be 8 g of deceleration, which is not a thing.
    #
    # CAVEAT: at these thrust levels V3 sits near Raptor minimum throttle. If
    # the real floor is above ~0.4 this schedule is infeasible and the burn
    # must ignite later or drop to fewer engines sooner. That tension may be
    # the reason the 13 -> 3 downselect exists at all.
    Phase("landing_13",     13, 13,   5.8, 0.45),
    Phase("landing_3",       3,  3,  12.3, 0.45),
]


@dataclass
class Vehicle:
    """
    Uniform-cylinder approximation of a Super Heavy V3 booster stage.

    SI throughout. Defaults are public estimates or, where noted, assumptions.
    Nothing downstream depends on their exact values -- substitute better
    numbers freely.
    """

    name: str = "Super Heavy V3"

    # --- geometry and mass ------------------------------------------------
    dry_mass: float = 275_000.0        # kg, structure + engines
    length: float = 72.3               # m, Block 3
    diameter: float = 9.0              # m
    prop_capacity: float = 3_650_000.0 # kg, Block 3 full load

    # Residual propellant at hot-stage separation. This must cover ALL return
    # burns. 400 t is not a lookup -- it is the smallest value consistent with
    # the observed Flight 13 burn durations at plausible throttle while leaving
    # anything for entry and landing. See burn_budget().
    prop_mass: float = 400_000.0       # kg, at hot-stage separation

    # --- propulsion -------------------------------------------------------
    # Raptor 3: 280 tf. NOT 2.30e6 -- that is Raptor 2 and it is ~20% low.
    thrust_per_engine: float = 2.746e6 # N
    isp_sl: float = 330.0              # s
    isp_vac: float = 350.0             # s, ASSUMPTION
    gimbal_limit: float = np.radians(15.0)

    n_engines_total: int = 33
    n_gimballing_max: int = 13         # inner ring; verify for V3

    # Distance from CoM to the gimbal plane. Moves as propellant drains;
    # held fixed here, revisited in Stage 4.
    gimbal_arm: float = 30.0           # m

    g0: float = 9.80665

    # A perfect cylinder has Iyy == Izz exactly, which is degenerate: the
    # gyroscopic term vanishes identically in pitch/yaw and you would never
    # notice if you had dropped it. Real vehicles are never symmetric.
    transverse_asymmetry: float = 0.02

    # ------------------------------------------------------------------
    # geometry and mass
    # ------------------------------------------------------------------

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.prop_mass

    def mass(self, prop_remaining: float) -> float:
        return self.dry_mass + max(prop_remaining, 0.0)

    # ------------------------------------------------------------------
    # propulsion
    # ------------------------------------------------------------------

    def isp(self, vacuum: bool = True) -> float:
        return self.isp_vac if vacuum else self.isp_sl

    def mass_flow(self, n_lit: int, throttle: float = 1.0,
                  vacuum: bool = True) -> float:
        """
        Total propellant flow, kg/s.

        Note mdot is roughly insensitive to altitude: thrust and Isp both rise
        in vacuum and largely cancel in F/(Isp*g0).
        """
        t = float(np.clip(throttle, 0.0, 1.0))
        return n_lit * self.thrust_per_engine * t / (self.isp(vacuum) * self.g0)

    def axial_thrust(self, n_lit: int, throttle: float = 1.0) -> float:
        """Total thrust magnitude, N. Uses n_lit -- ALL running engines."""
        return n_lit * self.thrust_per_engine * float(np.clip(throttle, 0.0, 1.0))

    # ------------------------------------------------------------------
    # inertia
    # ------------------------------------------------------------------

    def inertia(self, prop_remaining: float) -> np.ndarray:
        """
        Body-frame inertia tensor, 3x3, about the CoM.

        Uniform solid cylinder:
            Ixx = (1/2) m r^2                  (long axis)
            Iyy = Izz = (1/12) m (3r^2 + L^2)  (transverse)

        Diagonal because the body axes are principal axes by symmetry. Model
        off-axis tanks or a shifted CoM later and this picks up off-diagonal
        terms -- the propagator handles that already, nothing downstream
        assumes diagonality.

        Iyy falls by nearly half between separation and catch, so a controller
        tuned at one end is badly detuned at the other. Gain scheduling on this
        tensor is required, not optional.
        """
        m = self.mass(prop_remaining)
        r, L = self.radius, self.length

        ixx = 0.5 * m * r * r
        it = (m / 12.0) * (3.0 * r * r + L * L)

        iyy = it * (1.0 - 0.5 * self.transverse_asymmetry)
        izz = it * (1.0 + 0.5 * self.transverse_asymmetry)

        return np.diag([ixx, iyy, izz])

    def inertia_inv(self, prop_remaining: float) -> np.ndarray:
        return np.linalg.inv(self.inertia(prop_remaining))

    # ------------------------------------------------------------------
    # control torque
    # ------------------------------------------------------------------

    def gimbal_torque(
        self,
        n_gimballing: int,
        pitch_deflection: float,
        yaw_deflection: float,
        throttle: float = 1.0,
    ) -> np.ndarray:
        """
        Body-frame control torque from deflecting the gimballing engines.

        n_gimballing is the number of engines WITH ACTUATORS that are running --
        NOT the number lit. Engines without actuators contribute thrust and
        propellant flow and nothing to control torque.

        Engines sit aft of the CoM and push along -x undeflected. A small
        deflection produces a side force at the gimbal plane; force times
        moment arm is the control torque.

        Deflections are clamped to the mechanical limit. That saturation is
        real and Stage 3 depends on it being here -- a controller that silently
        commands 40 deg of gimbal is not a controller.
        """
        n = min(n_gimballing, self.n_gimballing_max)
        lim = self.gimbal_limit
        pitch = float(np.clip(pitch_deflection, -lim, lim))
        yaw = float(np.clip(yaw_deflection, -lim, lim))

        thrust = self.axial_thrust(n, throttle)

        f_z = thrust * np.sin(pitch)
        f_y = thrust * np.sin(yaw)

        r = np.array([-self.gimbal_arm, 0.0, 0.0])
        f = np.array([0.0, f_y, f_z])
        return np.cross(r, f)

    def max_angular_accel(self, n_gimballing: int, prop_remaining: float,
                          throttle: float = 1.0) -> float:
        """Peak pitch acceleration at full gimbal, rad/s^2."""
        tau = self.gimbal_torque(n_gimballing, self.gimbal_limit, 0.0, throttle)
        iyy = self.inertia(prop_remaining)[1, 1]
        return float(np.linalg.norm(tau) / iyy)

    # ------------------------------------------------------------------
    # propellant budget -- a conservation constraint, not decoration
    # ------------------------------------------------------------------

    def burn_budget(self, phases: list[Phase]) -> dict:
        """
        Check a burn schedule against the propellant aboard.

        If the schedule overruns the residual, the model is inconsistent and
        the sim will fly a vehicle that cannot exist. Because it is a hard
        constraint it also runs backwards: fix the residual and the observed
        durations, solve for the throttle that closes the budget.
        """
        used, rows, remaining = 0.0, [], self.prop_mass
        for p in phases:
            m = self.mass_flow(p.n_lit, p.throttle) * p.duration
            used += m
            remaining -= m
            rows.append({
                "name": p.name, "n_lit": p.n_lit, "duration": p.duration,
                "throttle": p.throttle, "prop_used": m, "prop_after": remaining,
            })
        return {
            "rows": rows,
            "used": used,
            "available": self.prop_mass,
            "margin": self.prop_mass - used,
            "closes": used <= self.prop_mass,
        }

    def required_throttle(self, phases: list[Phase], reserve: float = 0.0) -> float:
        """
        Uniform throttle that makes a schedule exactly fit the residual.

        A value above 1.0 means the schedule is infeasible at any throttle,
        which IS the answer: the durations or engine counts are wrong.
        """
        full = sum(self.mass_flow(p.n_lit, 1.0) * p.duration for p in phases)
        return (self.prop_mass - reserve) / full if full > 0 else 0.0

    def delta_v(self, phases: list[Phase], vacuum: bool = True) -> float:
        """Ideal delta-v of a schedule via the rocket equation, m/s."""
        used = sum(self.mass_flow(p.n_lit, p.throttle) * p.duration for p in phases)
        m0 = self.wet_mass
        m1 = m0 - used
        if m1 <= self.dry_mass:
            return float("inf")
        return float(self.isp(vacuum) * self.g0 * np.log(m0 / m1))

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def summary(self, prop_remaining: float | None = None) -> str:
        p = self.prop_mass if prop_remaining is None else prop_remaining
        ixx, iyy, izz = np.diag(self.inertia(p))
        return (
            f"{self.name}   propellant {p/1000:.1f} t\n"
            f"  mass                {self.mass(p)/1000:10.1f} t\n"
            f"  Ixx (roll)          {ixx:10.3e} kg m^2\n"
            f"  Iyy (pitch)         {iyy:10.3e} kg m^2\n"
            f"  Izz (yaw)           {izz:10.3e} kg m^2\n"
            f"  Iyy/Ixx             {iyy/ixx:10.1f}\n"
            f"  mdot/engine         {self.mass_flow(1):10.1f} kg/s\n"
            f"  pitch accel, 13 gim {np.degrees(self.max_angular_accel(13, p)):10.2f} deg/s^2\n"
            f"  pitch accel,  3 gim {np.degrees(self.max_angular_accel(3, p)):10.2f} deg/s^2\n"
        )

    def budget_report(self, phases: list[Phase]) -> str:
        b = self.burn_budget(phases)
        out = [f"{self.name} propellant budget   (residual {self.prop_mass/1000:.0f} t)",
               f"  {'phase':16s} {'lit':>4s} {'gim':>4s} {'dur':>6s} {'thr':>5s} "
               f"{'used':>8s} {'after':>8s}"]
        for p, r in zip(phases, b["rows"]):
            out.append(f"  {r['name']:16s} {r['n_lit']:4d} {p.n_gimballing:4d} "
                       f"{r['duration']:6.1f} {r['throttle']:5.2f} "
                       f"{r['prop_used']/1000:8.1f} {r['prop_after']/1000:8.1f}")
        out.append(f"  {'':16s} {'':4s} {'':4s} {'':6s} {'TOTAL':>5s} "
                   f"{b['used']/1000:8.1f} {b['margin']/1000:8.1f}")
        out.append(f"  closes: {b['closes']}     "
                   f"uniform throttle to fit: {self.required_throttle(phases):.2f}")
        out.append(f"  ideal delta-v: {self.delta_v(phases):.0f} m/s")
        return "\n".join(out)


SUPER_HEAVY = Vehicle()


def test_body(ixx: float = 1.0, iyy: float = 2.0, izz: float = 3.0) -> np.ndarray:
    """
    Generic asymmetric inertia tensor with three distinct moments.

    Needed because a cylinder is axisymmetric and therefore CANNOT show the
    intermediate-axis instability -- the best propagator validation case
    requires Ixx < Iyy < Izz all distinct.
    """
    return np.diag([ixx, iyy, izz])
