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

from dataclasses import dataclass

import numpy as np


@dataclass
class CatchTower:
    """
    Mechazilla-style catch tower with tracking arms.

    Geometry is roughly to scale; the compensation envelope is an estimate.
    """

    # --- structure ----------------------------------------------------
    height: float = 146.0          # m, tower height
    width: float = 12.0            # m, square cross-section
    pad: tuple = (-30.0, 0.0)       # visual tower base; arms center over the nominal catch point

    catch_altitude: float = 105.0  # m, nominal arm height
    arm_length: float = 15.0       # m, reach from the tower face
    arm_gap: float = 12.0          # m, clear width between the arms

    # --- tracking envelope (ESTIMATES) ---------------------------------
    # How far the arms can translate to meet the booster. Lateral travel is
    # the carriage running along the arms plus the arms closing; vertical is
    # the whole carriage riding the tower.
    # ASYMMETRIC, and deliberately so. The arms are shorter than V1's, so the
    # carriage has more room running ALONG the arms (crossrange, left/right of
    # the tower) than the rails have closing TOWARD the tower (downrange).
    #
    # This is not a freebie. It absorbs position error only -- a booster 3 m
    # left of centre is fine, a booster arriving at 3 m/s is not.
    crossrange_travel: float = 5.0   # m, +/- along the arms (body y)
    downrange_travel: float = 3.0    # m, +/- toward/from the tower (body x)
    vertical_travel: float = 3.0     # m, carriage travel on the tower

    def compensated_error(self, r: np.ndarray, target: np.ndarray):
        """
        Split the position error into what the tower absorbs and what remains.

        Returns (residual_lateral_m, residual_vertical_m, within_envelope).
        A booster 3 m off-centre with 5 m of arm travel has ZERO residual
        error: the arms simply meet it there.
        """
        r = np.asarray(r, dtype=float)
        t = np.asarray(target, dtype=float)

        d = r - t
        # Per-axis, because the envelope is not circular.
        res_x = max(abs(d[0]) - self.downrange_travel, 0.0)
        res_y = max(abs(d[1]) - self.crossrange_travel, 0.0)
        res_lat = float(np.hypot(res_x, res_y))
        res_vert = max(abs(float(d[2])) - self.vertical_travel, 0.0)
        within = (res_lat <= 0.0 and res_vert <= 0.0)
        return res_lat, res_vert, within

    def deadband(self, pos_err: np.ndarray) -> np.ndarray:
        """
        Shrink a lateral position error by the compensation envelope.

        This is what lets the VEHICLE stop correcting position the TOWER will
        absorb. Inside the envelope the returned error is zero, so the lateral
        controller spends its authority on velocity instead -- which is the
        one thing the arms cannot supply.
        """
        e = np.asarray(pos_err, dtype=float)
        lim = np.array([self.downrange_travel, self.crossrange_travel])
        return np.sign(e) * np.maximum(np.abs(e) - lim, 0.0)

    def arm_position(self, r: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Where the arms actually sit, having tracked toward the booster."""
        r = np.asarray(r, dtype=float)
        t = np.asarray(target, dtype=float)
        d = r - t
        lateral = np.array([
            float(np.clip(d[0], -self.downrange_travel, self.downrange_travel)),
            float(np.clip(d[1], -self.crossrange_travel, self.crossrange_travel)),
        ])
        dz = float(np.clip(d[2], -self.vertical_travel, self.vertical_travel))
        return t + np.array([lateral[0], lateral[1], dz])

    # ------------------------------------------------------------------
    # geometry for plotting
    # ------------------------------------------------------------------

    def structure_segments(self):
        """
        Wireframe rectangular prism for the tower.

        Returns a list of 2x3 arrays. A box reads as a structure at a glance;
        the three-line prism it replaces looked like a tripod.
        """
        px, py = self.pad
        h, w = self.height, 0.5 * self.width
        corners = [(px - w, py - w), (px + w, py - w),
                   (px + w, py + w), (px - w, py + w)]
        segs = []
        for i in range(4):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 4]
            segs.append(np.array([[x0, y0, 0.0], [x0, y0, h]]))      # vertical
            segs.append(np.array([[x0, y0, 0.0], [x1, y1, 0.0]]))    # base
            segs.append(np.array([[x0, y0, h], [x1, y1, h]]))        # top
            # a couple of intermediate bands so it reads as a lattice
            for frac in (0.33, 0.66):
                z = h * frac
                segs.append(np.array([[x0, y0, z], [x1, y1, z]]))
        return segs

    def arm_segments(self, r: np.ndarray | None = None,
                     target: np.ndarray | None = None):
        """
        Chopstick arms, drawn where they have TRACKED to.

        Hinged at the tower face and angled inward at the tips, so the pair
        forms the corridor the booster actually has to fly into. Passing the
        vehicle position shows the arms meeting it rather than sitting at the
        nominal point.
        """
        px, py = self.pad
        z = self.catch_altitude
        offset = np.zeros(3)
        if r is not None and target is not None:
            offset = self.arm_position(r, target) - np.asarray(target, float)

        segs = []
        hinge_x = px + 0.5 * self.width
        for side in (+1, -1):
            y = py + side * 0.5 * self.arm_gap + offset[1]
            x0 = hinge_x + offset[0]
            x1 = x0 + self.arm_length
            zz = z + offset[2]
            segs.append(np.array([[x0, y, zz], [x1, y, zz]]))          # arm
            segs.append(np.array([[x1, y, zz],
                                  [x1, y - side * 3.0, zz]]))          # tip
            # hinge stub back to the tower face
            segs.append(np.array([[x0, y, zz],
                                  [px + offset[0], py + offset[1], zz]]))
        return segs
