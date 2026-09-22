"""
Rigid-body attitude dynamics and the RK4 propagator.

STATE VECTOR (7 elements, flat float64)
    index 0:4  quaternion  [qw, qx, qy, qz]   body -> inertial
    index 4:7  body rates  [wx, wy, wz]       rad/s, IN BODY AXES

Flat rather than a class because that is what an integrator wants: the RK4
weighted sum is a plain vector operation, and the quaternion functions read
their operand straight out of a slice with no unpacking cost.

The two halves of the derivative come from different places:

    kinematics   qdot  = 0.5 * q (X) [0, omega]          (Stage 1)
    dynamics     wdot  = I^-1 (tau - omega x (I omega))  (here)

The gyroscopic term omega x (I omega) is what makes this interesting. Drop it
and you get three decoupled axes that never exchange angular momentum. Keep it
and you get nutation, precession, and the intermediate-axis instability.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import quaternion as Q

STATE_SIZE = 7
Q_SLICE = slice(0, 4)
W_SLICE = slice(4, 7)


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------

def make_state(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Pack an attitude and body rate into the 7-element state vector."""
    s = np.empty(STATE_SIZE)
    s[Q_SLICE] = Q.normalize(q)
    s[W_SLICE] = np.asarray(omega, dtype=float)
    return s


def unpack(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the state into (quaternion, body rate). Returns views, not copies."""
    return state[Q_SLICE], state[W_SLICE]


# ---------------------------------------------------------------------------
# the equations of motion
# ---------------------------------------------------------------------------

def omega_dot(
    omega: np.ndarray,
    torque: np.ndarray,
    inertia: np.ndarray,
    inertia_inv: np.ndarray,
) -> np.ndarray:
    """
    Euler's rotational equations solved for the angular acceleration:

        I wdot + omega x (I omega) = tau

    Everything is in BODY axes, which is the only frame in which the inertia
    tensor is constant. That is the whole reason attitude dynamics are posed in
    the body frame rather than the inertial one.

    inertia_inv is passed in rather than computed here because inverting a 3x3
    four times per RK4 step, thousands of steps per run, is pure waste.
    """
    return inertia_inv @ (torque - np.cross(omega, inertia @ omega))


def derivative(
    t: float,
    state: np.ndarray,
    torque_fn: Callable[[float, np.ndarray], np.ndarray],
    inertia: np.ndarray,
    inertia_inv: np.ndarray,
) -> np.ndarray:
    """
    Full state derivative. torque_fn(t, state) returns the body-frame torque,
    which is where the controller will plug in during Stage 3.
    """
    q, w = unpack(state)
    tau = torque_fn(t, state)

    d = np.empty(STATE_SIZE)
    d[Q_SLICE] = Q.derivative(q, w)
    d[W_SLICE] = omega_dot(w, tau, inertia, inertia_inv)
    return d


# ---------------------------------------------------------------------------
# integrator
# ---------------------------------------------------------------------------

def rk4_step(
    t: float,
    state: np.ndarray,
    dt: float,
    deriv: Callable[[float, np.ndarray], np.ndarray],
    renormalize: bool = True,
) -> tuple[np.ndarray, float]:
    """
    One classical RK4 step. Returns (new_state, norm_error_before_renormalizing).

    THE SUBTLE PART: renormalize the quaternion ONCE, after combining the four
    stages -- never inside them. The k values are derivative ESTIMATES, not
    states; normalizing them corrupts the weighted average and silently drops
    the method below fourth order. You would still get plausible output, just
    with an error that shrinks like dt^2 instead of dt^4, and nothing would
    announce the problem.

    The returned norm error is the diagnostic worth logging. Integrating qdot
    numerically always walks the quaternion slightly off the unit sphere; the
    magnitude of that walk per step tells you whether dt is small enough,
    independently of any reference solution.
    """
    k1 = deriv(t, state)
    k2 = deriv(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = deriv(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = deriv(t + dt, state + dt * k3)

    new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    norm_error = float(abs(np.linalg.norm(new[Q_SLICE]) - 1.0))
    if renormalize:
        new[Q_SLICE] = Q.normalize(new[Q_SLICE])
    return new, norm_error


def propagate(
    state0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
    inertia: np.ndarray,
    torque_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
    renormalize: bool = True,
) -> dict:
    """
    Integrate the attitude over t_span at fixed dt.

    Returns a dict of history arrays: t, state, q, omega, norm_error, plus the
    conserved-quantity diagnostics H_inertial and kinetic_energy. Those last
    two are not decoration -- for a torque-free body they must stay constant,
    and watching them drift is how you catch an integrator that is quietly
    wrong.

    Fixed step, not adaptive. Deliberate: fixed steps make the convergence
    study meaningful and keep the controller sample rate honest in Stage 3.
    """
    if torque_fn is None:
        def torque_fn(t, s):                       # noqa: ARG001
            return np.zeros(3)

    inertia = np.asarray(inertia, dtype=float)
    inertia_inv = np.linalg.inv(inertia)

    def deriv(t, s):
        return derivative(t, s, torque_fn, inertia, inertia_inv)

    t0, t1 = t_span
    n = int(np.ceil((t1 - t0) / dt))

    ts = np.empty(n + 1)
    states = np.empty((n + 1, STATE_SIZE))
    norm_err = np.zeros(n + 1)

    ts[0] = t0
    states[0] = make_state(state0[Q_SLICE], state0[W_SLICE])

    t, s = t0, states[0].copy()
    for i in range(1, n + 1):
        s, e = rk4_step(t, s, dt, deriv, renormalize=renormalize)
        t += dt
        ts[i] = t
        states[i] = s
        norm_err[i] = e

    qs = states[:, Q_SLICE]
    ws = states[:, W_SLICE]

    return {
        "t": ts,
        "state": states,
        "q": qs,
        "omega": ws,
        "norm_error": norm_err,
        "H_inertial": np.array([
            angular_momentum_inertial(q, w, inertia) for q, w in zip(qs, ws)
        ]),
        "kinetic_energy": np.array([
            kinetic_energy(w, inertia) for w in ws
        ]),
    }


# ---------------------------------------------------------------------------
# conserved quantities -- the validation instruments
# ---------------------------------------------------------------------------

def angular_momentum_body(omega: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """H = I omega, in body axes. NOT conserved even without torque."""
    return inertia @ np.asarray(omega, dtype=float)


def angular_momentum_inertial(
    q: np.ndarray, omega: np.ndarray, inertia: np.ndarray
) -> np.ndarray:
    """
    The same H rotated into inertial axes, where it IS conserved under zero
    torque -- all three components, not just the magnitude.

    This distinction catches a specific and common bug. In the body frame H
    wanders around as the body tumbles beneath it, which is correct physics and
    looks alarming if you expected a constant. Checking conservation in the
    wrong frame sends people hunting for integrator bugs that do not exist.
    """
    return Q.rotate(q, angular_momentum_body(omega, inertia))


def kinetic_energy(omega: np.ndarray, inertia: np.ndarray) -> float:
    """Rotational kinetic energy T = 0.5 omega . I omega."""
    w = np.asarray(omega, dtype=float)
    return 0.5 * float(w @ inertia @ w)


# ---------------------------------------------------------------------------
# torque sources for testing
# ---------------------------------------------------------------------------

def zero_torque(t: float, state: np.ndarray) -> np.ndarray:      # noqa: ARG001
    return np.zeros(3)


def constant_torque(tau: np.ndarray) -> Callable:
    """Fixed body-frame torque, for checking wdot = tau / I."""
    tau = np.asarray(tau, dtype=float)

    def fn(t, state):                                            # noqa: ARG001
        return tau
    return fn
