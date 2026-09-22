"""
Coupled 6-DOF dynamics.

STATE VECTOR (14 elements, flat float64)
    0:3    r      position, inertial frame, m
    3:6    v      velocity, inertial frame, m/s
    6:10   q      attitude quaternion, body -> inertial
    10:13  omega  body rates, rad/s, BODY axes
    13     prop   propellant remaining, kg

WHAT CHANGES FROM STAGE 2
-------------------------
Until now attitude and position were independent: rigidbody.py propagated
rotation with no notion of where the vehicle was. Here they couple, through
exactly one line --

    a_thrust = rotate(q, [T, 0, 0]) / m

Thrust is fixed along the body x-axis (the engines push toward the nose), so
WHERE THAT THRUST POINTS IN INERTIAL SPACE IS DETERMINED ENTIRELY BY THE
QUATERNION. An attitude error of theta sends the thrust theta off course and
becomes a position error that grows as the burn proceeds. That coupling is the
reason this project is about quaternions rather than about plotting a
trajectory.

Propellant is a real integrated state rather than a bookkeeping variable,
because mass and inertia both depend on it and both feed back into the
dynamics within a single step.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import environment as ENV
from . import quaternion as Q

STATE_SIZE = 14
R_SLICE = slice(0, 3)
V_SLICE = slice(3, 6)
Q_SLICE = slice(6, 10)
W_SLICE = slice(10, 13)
P_INDEX = 13


def make_state(r, v, q, omega, prop) -> np.ndarray:
    s = np.empty(STATE_SIZE)
    s[R_SLICE] = np.asarray(r, dtype=float)
    s[V_SLICE] = np.asarray(v, dtype=float)
    s[Q_SLICE] = Q.normalize(q)
    s[W_SLICE] = np.asarray(omega, dtype=float)
    s[P_INDEX] = float(prop)
    return s


def unpack(state: np.ndarray):
    """Returns (r, v, q, omega, prop). Slices are views; prop is a float."""
    return (state[R_SLICE], state[V_SLICE], state[Q_SLICE],
            state[W_SLICE], float(state[P_INDEX]))


class Forces:
    """
    What the actuators and the environment are doing at one instant.

    Separated from the derivative so a controller can be called at a slower
    rate than the integrator: compute Forces once per control tick, hold it
    across the RK4 substages. Recomputing the control law inside each substage
    would simulate a controller with infinite bandwidth, which is not the one
    you are designing.
    """

    def __init__(self, thrust: float = 0.0, torque=None,
                 mass_flow: float = 0.0, drag=None):
        self.thrust = float(thrust)
        self.torque = np.zeros(3) if torque is None else np.asarray(torque, float)
        self.mass_flow = float(mass_flow)
        self.drag = np.zeros(3) if drag is None else np.asarray(drag, float)


def derivative(
    t: float,
    state: np.ndarray,
    forces: Forces,
    mass_fn: Callable[[float], float],
    inertia_fn: Callable[[float], np.ndarray],
    spherical_gravity: bool = False,
) -> np.ndarray:
    """
    Full 14-state derivative.

    mass_fn and inertia_fn take propellant remaining and return the current
    mass and inertia tensor, so both track the draining tanks inside the step
    rather than being frozen at its start.
    """
    r, v, q, w, prop = unpack(state)

    m = mass_fn(prop)
    I = inertia_fn(prop)

    # --- translation --------------------------------------------------
    # Thrust acts along body +x; the quaternion is what puts it in inertial space.
    a_thrust = Q.rotate(q, np.array([forces.thrust, 0.0, 0.0])) / m
    a_grav = ENV.gravity(r, spherical=spherical_gravity)
    a_drag = forces.drag / m

    # --- rotation -----------------------------------------------------
    w_dot = np.linalg.solve(I, forces.torque - np.cross(w, I @ w))

    d = np.empty(STATE_SIZE)
    d[R_SLICE] = v
    d[V_SLICE] = a_thrust + a_grav + a_drag
    d[Q_SLICE] = Q.derivative(q, w)
    d[W_SLICE] = w_dot
    d[P_INDEX] = -forces.mass_flow
    return d


def rk4_step(t, state, dt, deriv, renormalize: bool = True):
    """
    Classical RK4 with the quaternion renormalized ONCE after the full step.

    Same rule as Stage 2 and for the same reason: the k values are derivative
    estimates, not states, and normalizing them corrupts the weighted average
    and silently costs you two orders of accuracy.

    Propellant is clamped at zero afterwards -- a negative propellant state
    would give a mass below dry and an inertia tensor that is not positive
    definite, which produces nonsense rather than an error.
    """
    k1 = deriv(t, state)
    k2 = deriv(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = deriv(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = deriv(t + dt, state + dt * k3)

    new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    norm_err = float(abs(np.linalg.norm(new[Q_SLICE]) - 1.0))
    if renormalize:
        new[Q_SLICE] = Q.normalize(new[Q_SLICE])
    new[P_INDEX] = max(new[P_INDEX], 0.0)
    return new, norm_err


def altitude(state: np.ndarray) -> float:
    return float(state[R_SLICE][2])


def speed(state: np.ndarray) -> float:
    return float(np.linalg.norm(state[V_SLICE]))


def tilt_from_vertical(state: np.ndarray) -> float:
    """
    Angle between the body x-axis (nose) and inertial up, radians.

    This is the catch criterion. Reporting it directly avoids any temptation
    to read a pitch angle out of an Euler conversion, which near vertical is
    exactly where that conversion is worst behaved.
    """
    nose = Q.rotate(state[Q_SLICE], np.array([1.0, 0.0, 0.0]))
    return float(np.arccos(np.clip(nose[2], -1.0, 1.0)))
