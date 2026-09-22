"""
Environment: gravity and atmosphere.

FRAME (local, flat-earth, adequate for the terminal phase)
    x : downrange / east, metres
    y : crossrange / north
    z : UP, metres. Ground is z = 0, the catch happens near z = 105.

Flat earth is fine here and not fine later. Over the last few hundred metres
the curvature and gravity-gradient errors are far below every other
uncertainty in the model. Over the ~100 km boostback arc they are not, so
Stage 4b should switch to inverse-square gravity -- the hook is already in
gravity() via the `spherical` flag, so that swap is one argument rather than a
rewrite.
"""

from __future__ import annotations

import numpy as np

G0 = 9.80665          # m/s^2, standard gravity
R_EARTH = 6_371_000.0  # m, mean radius
MU_EARTH = 3.986004418e14  # m^3/s^2


def gravity(position: np.ndarray, spherical: bool = False) -> np.ndarray:
    """
    Gravitational acceleration in the local frame, m/s^2.

    Flat mode returns a constant downward vector, which is what the terminal
    phase wants. Spherical mode treats z as altitude above the surface and
    applies inverse-square falloff; at 70 km that is already a 2% correction,
    which matters for a boostback arc and not at all for a hover.
    """
    if not spherical:
        return np.array([0.0, 0.0, -G0])
    r = R_EARTH + float(np.asarray(position, dtype=float)[2])
    return np.array([0.0, 0.0, -MU_EARTH / (r * r)])


def density(altitude: float) -> float:
    """
    Exponential atmosphere, kg/m^3.

    A two-parameter fit, not a standard atmosphere. Good to roughly 10% below
    30 km, worse above. That error is small next to the grid fin coefficient
    uncertainty, so refining this before the aero database would be polishing
    the wrong thing.
    """
    h = max(float(altitude), 0.0)
    return 1.225 * np.exp(-h / 8500.0)


def speed_of_sound(altitude: float) -> float:
    """
    Speed of sound, m/s, from a piecewise-linear temperature profile.

    Only the troposphere lapse and a constant stratosphere are modelled. Mach
    number feeds the aero lookup, so this needs to be about right rather than
    exact.
    """
    h = max(float(altitude), 0.0)
    T = 288.15 - 0.0065 * h if h < 11_000.0 else 216.65
    return float(np.sqrt(1.4 * 287.05 * T))


def dynamic_pressure(velocity: np.ndarray, altitude: float) -> float:
    """q = 0.5 rho V^2, Pa."""
    v = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
    return 0.5 * density(altitude) * v * v


def mach(velocity: np.ndarray, altitude: float) -> float:
    v = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
    return v / speed_of_sound(altitude)
