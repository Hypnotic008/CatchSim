"""
Catch tower: structure, and arms that track the booster.

WHY THE TOWER IS PART OF THE GUIDANCE PROBLEM
---------------------------------------------
The chopsticks are not a fixed target. The booster transmits its predicted
trajectory and the arms translate to meet it, so a metre of lateral error at
the catch plane is absorbed by the tower rather than failing the catch. That
is why Flight 6 was waved off when the tower link dropped: without the tracking
the arms cannot compensate, and the vehicle alone has to hit a fixed point.

What the arms CANNOT do is match velocity. They can be somewhere else; they
cannot be moving at the booster's speed and take up the momentum. So:

    POSITION error   -> absorbed, within the arm travel envelope
    VELOCITY error   -> NOT absorbed, must be nulled by the vehicle
    ATTITUDE error   -> not absorbed either; the fins have to line up with
                        the arms or they hit structure instead of seating

Scoring a simulation against a FIXED catch point therefore models a harder
problem than the real one, and specifically the wrong harder problem: it spends
control authority on position accuracy the tower would have provided, at the
expense of the velocity accuracy only the vehicle can provide.

TRAVEL LIMITS ARE ESTIMATES
---------------------------
How far the arms can translate, and how fast, is not public. The values below
are plausible rather than known, and every result that depends on them should
be quoted with that caveat.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


# Arm heading (compass degrees, 0 = north, 90 = east: the direction from the
# tower to the catch point) of the Starbase towers, from satellite imagery.
# Sim frame: x = east (downrange, offshore), y = north, z = up.
#   Tower 1 (Pad A): arms open toward the Gulf, roughly east. This is the
#                    orientation the sim used before, and the easy one: the
#                    returning booster flies straight into the open mouth.
#   Tower 2 (Pad B): arms open to the SOUTH-southwest, with the tower north of
#                    the catch point. A booster coming home from the east
#                    approaches ACROSS the arms, not into them.
TOWER_FACING_DEG = {1: 90.0, 2: 190.0}


def compass_vector(deg: float) -> np.ndarray:
    """Unit (east, north) vector of a compass heading."""
    a = np.radians(deg)
    return np.array([np.sin(a), np.cos(a)])


@dataclass
class CatchTower:
    """
    Mechazilla-style catch tower with tracking arms.

    Geometry is roughly to scale; the compensation envelope is an estimate.
    All geometry is built in the TOWER frame (a = along the arms, from the
    tower toward the catch point; c = across the arms, a rotated 90 deg
    counter-clockwise) and rotated into the sim frame by `facing_deg`.
    """

    # --- orientation ----------------------------------------------------
    facing_deg: float = TOWER_FACING_DEG[2]   # arm heading, compass deg
    standoff: float = 30.0          # m, tower centre to catch point
    center: tuple = (0.0, 0.0)      # (x, y) of the nominal catch point

    # --- structure ----------------------------------------------------
    height: float = 146.0          # m, tower height
    width: float = 12.0            # m, square cross-section

    catch_altitude: float = 105.0  # m, nominal arm height
    arm_length: float = 30.0       # m, reach from the tower face (past the catch point)
    arm_gap: float = 12.0          # m, clear width between the closed arms
    arm_gap_open: float = 22.0     # m, between the arm tips, fully open
    booster_radius: float = 4.5    # m

    # --- tracking envelope (ESTIMATES) ---------------------------------
    # How far the arms can translate to meet the booster, in the TOWER frame.
    # ASYMMETRIC, and deliberately so: the arms swing and slide sideways more
    # easily (across, c) than the carriage can move toward or away from the
    # tower face (along, a).
    #
    # This is not a freebie. It absorbs position error only -- a booster 3 m
    # off centre is fine, a booster arriving at 3 m/s is not.
    crossrange_travel: float = 5.0   # m, +/- across the arms (c)
    downrange_travel: float = 3.0    # m, +/- toward/from the tower (a)
    vertical_travel: float = 3.0     # m, carriage travel on the tower

    @classmethod
    def numbered(cls, n: int, **kw) -> "CatchTower":
        """Tower 1 or Tower 2 at Starbase."""
        return cls(facing_deg=TOWER_FACING_DEG[int(n)], **kw)

    # --- frame ----------------------------------------------------------
    @property
    def facing(self) -> np.ndarray:
        """Unit vector (sim x, y, z) from the tower toward the catch point.
        The rudder (third) grid fin points this way at the catch."""
        f = compass_vector(self.facing_deg)
        return np.array([f[0], f[1], 0.0])

    @property
    def across(self) -> np.ndarray:
        f = self.facing
        return np.array([-f[1], f[0], 0.0])

    @property
    def pad(self) -> tuple:
        """Tower centre (x, y)."""
        f = self.facing
        return (self.center[0] - self.standoff * f[0],
                self.center[1] - self.standoff * f[1])

    def to_tower(self, d: np.ndarray) -> np.ndarray:
        """Sim-frame offset -> (along, across, up)."""
        d = np.asarray(d, float)
        return np.array([float(d[:2] @ self.facing[:2]),
                         float(d[:2] @ self.across[:2]), float(d[2])])

    def from_tower(self, a, c, z=0.0) -> np.ndarray:
        """Tower-frame offset (along, across, up) -> sim-frame offset."""
        return a * self.facing + c * self.across + np.array([0.0, 0.0, z])

    def _xyz(self, a, c, z):
        """Tower-frame point relative to the tower CENTRE -> sim point."""
        px, py = self.pad
        return np.array([px, py, 0.0]) + self.from_tower(a, c, z)

    # --- envelope -------------------------------------------------------
    def compensated_error(self, r: np.ndarray, target: np.ndarray):
        """
        Split the position error into what the tower absorbs and what remains.

        Returns (residual_lateral_m, residual_vertical_m, within_envelope).
        A booster 3 m off-centre with 5 m of arm travel has ZERO residual
        error: the arms simply meet it there.
        """
        d = self.to_tower(np.asarray(r, float) - np.asarray(target, float))
        # Per-axis, because the envelope is not circular.
        res_a = max(abs(d[0]) - self.downrange_travel, 0.0)
        res_c = max(abs(d[1]) - self.crossrange_travel, 0.0)
        res_lat = float(np.hypot(res_a, res_c))
        res_vert = max(abs(float(d[2])) - self.vertical_travel, 0.0)
        within = (res_lat <= 0.0 and res_vert <= 0.0)
        return res_lat, res_vert, within

    def deadband(self, pos_err: np.ndarray) -> np.ndarray:
        """
        Shrink a lateral (x, y) position error by the compensation envelope.

        This is what lets the VEHICLE stop correcting position the TOWER will
        absorb. Inside the envelope the returned error is zero, so the lateral
        controller spends its authority on velocity instead -- which is the
        one thing the arms cannot supply.
        """
        e = np.asarray(pos_err, dtype=float)
        d = self.to_tower(np.array([e[0], e[1], 0.0]))
        lim = np.array([self.downrange_travel, self.crossrange_travel])
        dt = np.sign(d[:2]) * np.maximum(np.abs(d[:2]) - lim, 0.0)
        return self.from_tower(dt[0], dt[1])[:2]

    def arm_offset(self, r: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Tower-frame (along, across, up) shift of the arms tracking the
        booster, clipped to the envelope."""
        d = self.to_tower(np.asarray(r, float) - np.asarray(target, float))
        return np.array([
            float(np.clip(d[0], -self.downrange_travel, self.downrange_travel)),
            float(np.clip(d[1], -self.crossrange_travel, self.crossrange_travel)),
            float(np.clip(d[2], -self.vertical_travel, self.vertical_travel))])

    def arm_position(self, r: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Where the arms actually sit, having tracked toward the booster."""
        o = self.arm_offset(r, target)
        return np.asarray(target, float) + self.from_tower(o[0], o[1], o[2])

    # --- approach corridor ----------------------------------------------
    def clearance(self, r: np.ndarray, body_below: float = 30.0,
                  body_above: float = 40.0, arm_z: float | None = None):
        """
        Horizontal clearance [m] between the booster hull and the open arms
        or tower at CoM position r (negative = the hull would hit them), or
        +inf while the hull is not at arm height.

        The arms are open to `arm_gap_open` while the booster comes in; the
        hull spans [z - body_below, z + body_above] about the CoM and the
        arms sit `arm_z` (default: hard-point height, 35 m above the catch
        CoM) above the ground. The booster may come in from the open side
        (along > the arm tips) freely; between the arms it must stay inside
        the mouth.
        """
        r = np.asarray(r, float)
        z_arm = self.catch_altitude + 35.0 if arm_z is None else arm_z
        d = self.to_tower(r - np.array([self.center[0], self.center[1], 0.0]))
        p = d[:2] + np.array([self.standoff, 0.0])    # relative to tower centre
        R = self.booster_radius
        w = 0.5 * self.width
        out = float("inf")
        if r[2] - body_below <= self.height:          # beside the tower
            q = np.abs(p) - w
            out = float(np.linalg.norm(np.maximum(q, 0.0))
                        + min(max(q[0], q[1]), 0.0)) - R
        if r[2] - body_below <= z_arm <= r[2] + body_above:   # at arm height
            for side in (+1.0, -1.0):
                h0 = np.array([w, side * w])
                h1 = np.array([w + self.arm_length,
                               side * 0.5 * self.arm_gap_open])
                e = h1 - h0
                u = float(np.clip((p - h0) @ e / (e @ e), 0.0, 1.0))
                out = min(out, float(np.linalg.norm(p - (h0 + u * e))) - R)
        return out

    # ------------------------------------------------------------------
    # geometry for plotting
    # ------------------------------------------------------------------

    def structure_segments(self):
        """
        Wireframe rectangular prism for the tower, rotated to its heading.

        Returns a list of 2x3 arrays. A box reads as a structure at a glance;
        the three-line prism it replaces looked like a tripod.
        """
        h, w = self.height, 0.5 * self.width
        corners = [(-w, -w), (w, -w), (w, w), (-w, w)]
        segs = []
        for i in range(4):
            a0, c0 = corners[i]
            a1, c1 = corners[(i + 1) % 4]
            P = lambda a, c, z: self._xyz(a, c, z)
            segs.append(np.array([P(a0, c0, 0.0), P(a0, c0, h)]))      # vertical
            segs.append(np.array([P(a0, c0, 0.0), P(a1, c1, 0.0)]))    # base
            segs.append(np.array([P(a0, c0, h), P(a1, c1, h)]))        # top
            for frac in (0.33, 0.66):
                z = h * frac
                segs.append(np.array([P(a0, c0, z), P(a1, c1, z)]))
        return segs

    def arm_segments(self, r: np.ndarray | None = None,
                     target: np.ndarray | None = None,
                     z: float | None = None):
        """
        Chopstick arms, drawn where they have TRACKED to.

        Hinged at the tower face and angled inward at the tips, so the pair
        forms the corridor the booster actually has to fly into. Passing the
        vehicle position shows the arms meeting it rather than sitting at the
        nominal point.
        """
        z0 = self.catch_altitude if z is None else z
        o = np.zeros(3)
        if r is not None and target is not None:
            o = self.arm_offset(r, target)
        segs = []
        hinge_a = 0.5 * self.width + o[0]
        zz = z0 + o[2]
        for side in (+1, -1):
            c = side * 0.5 * self.arm_gap + o[1]
            a1 = hinge_a + self.arm_length
            P = lambda a, cc: self._xyz(a, cc, zz)
            segs.append(np.array([P(hinge_a, c), P(a1, c - side * 3.0)]))   # arm
            segs.append(np.array([P(a1, c - side * 3.0),
                                  P(a1, c - side * 5.0)]))                  # tip
            segs.append(np.array([P(o[0], o[1]), P(hinge_a, c)]))           # hinge
        return segs


def current_tower(target=(0.0, 0.0, 105.0), n: int | None = None) -> CatchTower:
    """The tower selected for this run (CATCHSIM_TOWER, default Tower 2),
    centred on the catch target."""
    if n is None:
        n = int(os.environ.get("CATCHSIM_TOWER", "2"))
    t = np.asarray(target, float)
    return CatchTower.numbered(n, center=(float(t[0]), float(t[1])),
                               catch_altitude=float(t[2]))
