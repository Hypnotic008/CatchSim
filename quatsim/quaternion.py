"""
Quaternion algebra for rigid-body attitude.

CONVENTIONS (fixed for the entire project -- do not mix)
--------------------------------------------------------
Storage    : scalar-first, q = [w, x, y, z], a plain numpy array of shape (4,).
Algebra    : Hamilton product (ij = k), NOT the JPL/shuster convention.
Meaning    : q is the BODY-to-INERTIAL attitude. That is, given a vector
             expressed in body axes, v_b, the same vector in inertial axes is
                 v_i = q (X) [0, v_b] (X) q*
             and equivalently v_i = to_dcm(q) @ v_b.
Angular    : omega is the angular velocity of the body w.r.t. the inertial
velocity     frame, EXPRESSED IN BODY AXES (rad/s). This is what a rate gyro
             bolted to the vehicle measures, and it is what Euler's equations
             use, so it is the only representation stored in the state vector.
Handedness : all rotations are right-handed / active.

Every function takes and returns raw numpy arrays so that a quaternion can live
as a slice of a flat integrator state vector (state[6:10]) with no boxing cost.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "IDENTITY",
    "normalize",
    "conjugate",
    "inverse",
    "multiply",
    "rotate",
    "rotate_inverse",
    "to_dcm",
    "from_dcm",
    "from_axis_angle",
    "to_axis_angle",
    "from_rotvec",
    "to_rotvec",
    "from_euler_321",
    "to_euler_321",
    "canonical",
    "angle_between",
    "slerp",
    "derivative",
    "omega_matrix",
    "attitude_error",
    "integrate_body_rate",
]

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

# Below this quaternion-vector norm we are within ~1e-8 rad of no rotation and
# the axis is numerically meaningless; fall back to series expansions.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# basic algebra
# ---------------------------------------------------------------------------

def normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm. Raises if q is degenerate."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < _EPS:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate q* = [w, -x, -y, -z]. For unit q this is the inverse rotation."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def inverse(q: np.ndarray) -> np.ndarray:
    """True inverse q* / |q|^2. Use conjugate() directly when q is known unit."""
    q = np.asarray(q, dtype=float)
    n2 = float(q @ q)
    if n2 < _EPS:
        raise ValueError("cannot invert a zero quaternion")
    return conjugate(q) / n2


def multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Hamilton product q (X) p.

    Composition reads right-to-left like matrix products: if p is a rotation
    applied first and q second, the total is multiply(q, p), and
    to_dcm(multiply(q, p)) == to_dcm(q) @ to_dcm(p).
    """
    w1, x1, y1, z1 = np.asarray(q, dtype=float)
    w2, x2, y2, z2 = np.asarray(p, dtype=float)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# ---------------------------------------------------------------------------
# acting on vectors
# ---------------------------------------------------------------------------

def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a body-frame vector into the inertial frame: v_i = q (X) v_b (X) q*.

    Uses the factored form rather than two quaternion products -- same result,
    roughly half the flops, and it is the form you want inside an RK4 loop.
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    w, u = q[0], q[1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate an inertial-frame vector into body axes: v_b = q* (X) v_i (X) q."""
    return rotate(conjugate(q), v)


def to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Direction cosine matrix R such that v_inertial = R @ v_body.

    Prefer rotate() for a single vector; build the DCM when you need to
    transform many vectors at once or want to inspect the axes as columns
    (column 0 is the body x-axis expressed in inertial coordinates).
    """
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def from_dcm(R: np.ndarray) -> np.ndarray:
    """
    Recover a quaternion from a body-to-inertial DCM (Shepperd's method).

    The naive w = 0.5*sqrt(1+trace) formula loses precision and can take the
    square root of a small negative number near 180 deg rotations. Shepperd
    picks whichever of the four components is largest and divides by it, so the
    divisor is always >= 0.5 and the result is well conditioned at every angle.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {R.shape}")

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    candidates = (trace, R[0, 0], R[1, 1], R[2, 2])
    best = int(np.argmax(candidates))

    if best == 0:
        s = np.sqrt(1.0 + trace) * 2.0
        q = np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    elif best == 1:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ])
    elif best == 2:
        s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
        q = np.array([
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ])
    else:
        s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
        q = np.array([
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ])
    return normalize(q)


# ---------------------------------------------------------------------------
# axis-angle and rotation vector
# ---------------------------------------------------------------------------

def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion for a rotation of `angle` rad about `axis` (auto-normalized)."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < _EPS:
        raise ValueError("rotation axis must be nonzero")
    axis = axis / n
    h = 0.5 * angle
    return np.concatenate(([np.cos(h)], np.sin(h) * axis))


def to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Inverse of from_axis_angle, returning (unit_axis, angle) with angle in [0, pi].

    The quaternion is canonicalized first so the returned angle is the short
    way around; for a near-identity rotation the axis is arbitrary and x is
    returned by convention.
    """
    q = canonical(normalize(q))
    vec_norm = np.linalg.norm(q[1:])
    if vec_norm < _EPS:
        return np.array([1.0, 0.0, 0.0]), 0.0
    angle = 2.0 * np.arctan2(vec_norm, q[0])
    return q[1:] / vec_norm, float(angle)


def from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """
    Exponential map: a rotation vector (axis * angle, rad) to a quaternion.

    Series-expands for tiny angles so that small incremental updates -- the
    common case inside an integrator -- stay accurate instead of dividing by a
    vanishing norm.
    """
    r = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        # sin(t/2)/t -> 1/2 - t^2/48 ; cos(t/2) -> 1 - t^2/8
        return normalize(np.concatenate(
            ([1.0 - theta * theta / 8.0], r * (0.5 - theta * theta / 48.0))
        ))
    h = 0.5 * theta
    return np.concatenate(([np.cos(h)], (np.sin(h) / theta) * r))


def to_rotvec(q: np.ndarray) -> np.ndarray:
    """Logarithmic map: quaternion to rotation vector, magnitude in [0, pi]."""
    axis, angle = to_axis_angle(q)
    return axis * angle


# ---------------------------------------------------------------------------
# Euler angles -- provided mainly so the gimbal-lock comparison has something
# to compare against. Do not use these inside the propagator.
# ---------------------------------------------------------------------------

def from_euler_321(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Aerospace 3-2-1 sequence (yaw about z, then pitch about y, then roll about x),
    all in radians. Equivalent to R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def to_euler_321(q: np.ndarray, gimbal_tol: float = 1e-6) -> tuple[float, float, float]:
    """
    Extract (yaw, pitch, roll) in radians. Pitch is returned in [-pi/2, pi/2].

    THIS IS THE FUNCTION THAT BREAKS. As pitch approaches +/-90 deg the yaw and
    roll axes become parallel and only their sum or difference is observable --
    the individual values are not defined. Near the singularity roll is pinned
    to zero and the whole rotation is loaded into yaw, which is a legitimate
    answer but discontinuous, so a controller differentiating these angles sees
    an infinite rate. That is the failure mode the quaternion propagator avoids
    and the one worth plotting.
    """
    R = to_dcm(q)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)

    if abs(abs(sp) - 1.0) < gimbal_tol:
        # Gimbal lock: R[2,1] and R[2,2] are both ~0, atan2 is meaningless.
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


# ---------------------------------------------------------------------------
# double cover, distance, interpolation
# ---------------------------------------------------------------------------

def canonical(q: np.ndarray) -> np.ndarray:
    """
    Force w >= 0.

    q and -q are the same physical attitude (the double cover). Left alone, a
    controller can read a 10 deg error as a 350 deg error and slew the long way
    around. Canonicalizing the ERROR quaternion -- not the state -- is the fix.
    """
    q = np.asarray(q, dtype=float)
    return -q if q[0] < 0.0 else q.copy()


def angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Shortest geodesic angle in radians between two attitudes, in [0, pi].

    Computed via atan2 on the relative quaternion rather than arccos of the
    dot product. The arccos form is the one in most textbooks and it is fine
    for large angles, but it is ill-conditioned near zero: arccos has infinite
    slope at 1, so a dot product accurate to machine epsilon gives an angle
    accurate only to about 1e-8 rad, and below that it returns exactly zero.
    That silently breaks integrator convergence studies, where the whole point
    is to measure errors far smaller than 1e-8.
    """
    a = normalize(q1)
    b = normalize(q2)
    dq = multiply(conjugate(a), b)
    return float(2.0 * np.arctan2(np.linalg.norm(dq[1:]), abs(dq[0])))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float | np.ndarray):
    """
    Constant-rate interpolation along the shortest great-circle arc.

    This is how commanded attitude is generated for a slew: give the guidance
    layer a start and end attitude and a duration, and slerp gives a reference
    with constant angular rate. Scalar t returns one quaternion; array t
    returns an (N, 4) array.
    """
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:            # take the short way around
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))

    if dot > 0.9995:
        # Nearly parallel: sin(theta) underflows, so lerp and renormalize.
        out = q0 + t_arr[:, None] * (q1 - q0)
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
    else:
        theta = np.arccos(dot)
        s = np.sin(theta)
        w0 = np.sin((1.0 - t_arr) * theta) / s
        w1 = np.sin(t_arr * theta) / s
        out = w0[:, None] * q0 + w1[:, None] * q1

    return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out


# ---------------------------------------------------------------------------
# kinematics and control
# ---------------------------------------------------------------------------

def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """
    The 4x4 Omega such that qdot = 0.5 * Omega(omega_body) @ q.

    Same content as derivative(), exposed in matrix form because it is what you
    need for a linearized covariance propagation or a discrete state-transition
    matrix later on.
    """
    p, qy, r = np.asarray(omega, dtype=float)
    return np.array([
        [0.0, -p,  -qy, -r],
        [p,   0.0,  r,  -qy],
        [qy, -r,   0.0,  p],
        [r,   qy,  -p,  0.0],
    ])


def derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """
    Attitude kinematics: qdot = 0.5 * q (X) [0, omega_body].

    Note the ORDER. Because q is body-to-inertial and omega is expressed in
    body axes, the pure quaternion multiplies on the RIGHT. Swapping the
    operands silently gives you a simulation that rotates about inertial axes
    instead, which looks plausible until the vehicle is far from upright.
    """
    w = np.asarray(omega_body, dtype=float)
    return 0.5 * multiply(q, np.array([0.0, w[0], w[1], w[2]]))


def attitude_error(q_actual: np.ndarray, q_cmd: np.ndarray) -> np.ndarray:
    """
    Error quaternion dq = q_actual* (X) q_cmd, canonicalized to w >= 0.

    dq is the rotation that carries the vehicle from where it is to where it
    is commanded, EXPRESSED IN BODY AXES -- which is exactly the frame the
    gimbals and fins produce torque in. The vector part 2*dq[1:] is the
    small-angle rotation-vector error, so a PD law reads

        tau = Kp * (2.0 * attitude_error(q, q_cmd)[1:]) - Kd * omega_body

    with a POSITIVE proportional sign. (This is the reverse operand order from
    the q_cmd* (X) q_actual form; that one lives in the commanded frame and
    needs a negative sign. Either works, but mixing them is a sign-error bug
    that presents as a controller driving smoothly to the wrong attitude.)
    """
    return canonical(multiply(conjugate(normalize(q_actual)), normalize(q_cmd)))


def integrate_body_rate(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """
    Exact closed-form attitude update for a CONSTANT body rate over dt.

    q_new = q (X) exp(0.5 * omega * dt). Unlike an RK4 step on qdot this is
    exact for constant omega and returns a unit quaternion by construction, so
    it is the right tool for reference-trajectory generation and for checking
    how much error your numerical integrator is actually accruing.
    """
    return normalize(multiply(q, from_rotvec(np.asarray(omega_body, dtype=float) * dt)))
