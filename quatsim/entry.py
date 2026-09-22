"""
Entry guidance: steering the unpowered descent with the grid fins.

WHY THIS EXISTS
---------------
Between boostback cutoff and landing-burn ignition the booster falls ~100 km
in ~230 s with its engines off. The boostback predictor aims that arc using
the NOMINAL drag and no wind, so every drag, density and wind dispersion
lands in the arrival point. In the Monte Carlo those errors reached several
hundred metres -- and a 13-engine landing burn, which cannot throttle below
~26 m/s^2 of deceleration and therefore lasts only ~7 s, can divert only
about +/-250 m. The missing capability is the one the real vehicle uses: fly
a few degrees off the relative wind and let body lift move the arrival point.

HOW
---
Above the atmosphere the reference is simply tail-first along the relative
wind, held on RCS, so the vehicle enters aligned. Then every ``period``
seconds, once dynamic pressure is useful:

  1. PREDICT where the vehicle will cross ``aim_altitude`` if it flew
     straight down the wind from here (point mass, guidance's own drag model,
     no lift). Cheap, and re-run often enough that its errors are corrected
     by the next prediction -- the same closed-loop idea as the boostback.
  2. COMMAND a lateral acceleration a = K * miss / t_go^2 perpendicular to
     the velocity.
  3. CONVERT to an angle of attack using the normal-force slope,
         alpha = a m / (q S cn_alpha),
     capped by what the fins can actually hold against the weathercocking
     moment (with margin) and an absolute limit.
  4. POINT the body: tail-first along the relative wind, tilted by alpha in
     the direction whose normal force produces the commanded acceleration.

The attitude reference is flown by the grid fins plus RCS; see
``FlightSequencer._coast_control``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import environment as ENV
from . import quaternion as Q
from .aero import AeroModel, GridFinModel
from .guidance import attitude_from_pointing
from .vehicle import Vehicle


@dataclass
class EntryConfig:
    aim: np.ndarray                       # (x, y) to arrive over
    aim_altitude: float = 1_200.0         # altitude the aim point refers to
    q_min: float = 1_500.0                # Pa, steering starts above this
    gain: float = 3.0                     # K in a = K miss / t_go^2
    alpha_max: float = np.radians(6.0)
    fin_margin: float = 0.8               # fraction of fin torque to plan on
    period: float = 0.5                   # s between predictions
    t_go_floor: float = 4.0


class EntryGuidance:
    def __init__(self, cfg: EntryConfig, vehicle: Vehicle, aero: AeroModel,
                 fins: GridFinModel | None, cm_alpha: float = 0.35,
                 ref_length: float = 72.3):
        self.cfg = cfg
        self.vehicle = vehicle
        self.aero = aero
        self.fins = fins or GridFinModel()
        self.cm_alpha = cm_alpha
        self.ref_length = ref_length
        self._t_last = -1e9
        self._a_cmd = np.zeros(3)
        self.log = []

    # ------------------------------------------------------------------

    def predict(self, r, v, prop, dt=0.25):
        """Straight-down-the-wind point-mass prediction to aim_altitude.
        Returns (crossing point, time to go)."""
        r = np.array(r, float)
        v = np.array(v, float)
        m = self.vehicle.mass(prop)
        t = 0.0
        while r[2] > self.cfg.aim_altitude and t < 400.0:
            a1 = ENV.gravity(r, spherical=True) + self.aero.drag_force(v, r[2]) / m
            vm = v + 0.5 * dt * a1
            rm = r + 0.5 * dt * v
            a2 = ENV.gravity(rm, spherical=True) + self.aero.drag_force(vm, rm[2]) / m
            r_new = r + dt * vm
            v = v + dt * a2
            if r_new[2] <= self.cfg.aim_altitude:
                f = (r[2] - self.cfg.aim_altitude) / max(r[2] - r_new[2], 1e-9)
                r = r + f * (r_new - r)
                t += f * dt
                break
            r = r_new
            t += dt
        return r, t

    def alpha_limit(self, q_dyn: float, mach: float) -> float:
        """Largest trim angle the fins can hold against weathercocking."""
        fin_max = self.fins.max_torque(mach, q_dyn) * self.cfg.fin_margin
        wc = self.cm_alpha * q_dyn * self.aero.reference_area * self.ref_length
        if wc <= 0.0:
            return self.cfg.alpha_max
        return float(min(np.arcsin(min(fin_max / wc, 1.0)), self.cfg.alpha_max))

    # ------------------------------------------------------------------

    def command(self, r, v, q, prop, t):
        """Attitude reference (quaternion) for this instant, or None when the
        air is too thin to steer."""
        c = self.cfg
        r = np.asarray(r, float)
        v = np.asarray(v, float)
        v_air = self.aero.air_velocity(v, r[2])
        speed = float(np.linalg.norm(v_air))
        if speed < 5.0:
            return None, {}
        q_dyn = 0.5 * ENV.density(r[2]) * speed * speed
        m = self.vehicle.mass(prop)
        v_hat = v_air / speed
        z_now = Q.rotate(q, np.array([0.0, 0.0, 1.0]))
        if q_dyn < c.q_min:
            # Too thin to steer, but not too thin to matter: hold tail-first
            # along the relative wind (RCS) so the vehicle ENTERS aligned.
            # Left alone it arrived at the sensible atmosphere off-trim and
            # swung +/-25 deg about it, and with body lift those swings threw
            # it ~1 km sideways before the fins had any authority.
            return (attitude_from_pointing(-v_hat, roll_reference=z_now),
                    {"alpha_deg": 0.0, "alpha_lim_deg": 0.0,
                     "q_dyn": q_dyn, "mach": speed / ENV.speed_of_sound(r[2])})

        if t - self._t_last >= c.period:
            self._t_last = t
            r_pred, t_go = self.predict(r, v, prop)
            miss = np.array([c.aim[0] - r_pred[0], c.aim[1] - r_pred[1], 0.0])
            a = c.gain * miss / max(t_go, c.t_go_floor) ** 2
            a = a - float(a @ v_hat) * v_hat        # perpendicular to the flow
            self._a_cmd = a
            self.log.append((t, float(r[2]), float(np.linalg.norm(miss)),
                             float(r_pred[0]), float(r_pred[1]), t_go))

        a = self._a_cmd
        a_mag = float(np.linalg.norm(a))
        mach = speed / ENV.speed_of_sound(r[2])
        alpha_lim = self.alpha_limit(q_dyn, mach)
        tail_first = -v_hat                     # body +x for zero AoA
        if a_mag < 1e-6:
            d = tail_first
            alpha = 0.0
        else:
            alpha = min(a_mag * m / (q_dyn * self.aero.reference_area
                                     * self.aero.cn_alpha), alpha_lim)
            u = a / a_mag
            # Nose tilted AWAY from the desired force direction produces a
            # normal force TOWARD it (crossflow opposes v_perp).
            d = np.cos(alpha) * tail_first - np.sin(alpha) * u
        q_ref = attitude_from_pointing(d, roll_reference=z_now)
        return q_ref, {"alpha_deg": float(np.degrees(alpha)),
                       "alpha_lim_deg": float(np.degrees(alpha_lim)),
                       "q_dyn": q_dyn, "mach": mach}
