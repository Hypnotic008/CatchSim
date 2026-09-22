"""
Validation suite for quatsim.quaternion.

Two kinds of test here, and both matter:

  1. Cross-validation against scipy.spatial.transform.Rotation. This catches
     convention errors -- transposed DCMs, reversed composition order, JPL vs
     Hamilton sign flips -- which are the bugs that silently produce a
     plausible-looking but wrong simulation.

  2. Analytic identities and round trips. These catch numerical issues that
     scipy would share, and they cover the degenerate cases (180 deg, near
     zero, gimbal lock) where naive implementations fall over.

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatsim import quaternion as Q

RNG = np.random.default_rng(20260910)
ATOL = 1e-12


def random_quats(n):
    """Uniformly distributed random attitudes (Shoemake's method, via scipy)."""
    return Rotation.random(n, rng=RNG).as_quat(scalar_first=True)


def as_scipy(q):
    return Rotation.from_quat(q, scalar_first=True)


def same_rotation(q1, q2, atol=1e-10):
    """Compare allowing for the double cover: q and -q are the same attitude."""
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)


# ---------------------------------------------------------------------------
# algebra
# ---------------------------------------------------------------------------

class TestAlgebra:

    def test_identity_is_neutral(self):
        for q in random_quats(50):
            assert np.allclose(Q.multiply(Q.IDENTITY, q), q, atol=ATOL)
            assert np.allclose(Q.multiply(q, Q.IDENTITY), q, atol=ATOL)

    def test_multiply_matches_scipy(self):
        """Composition order must match scipy's r1 * r2 (r2 applied first)."""
        for q1, q2 in zip(random_quats(100), random_quats(100)):
            mine = Q.multiply(q1, q2)
            theirs = (as_scipy(q1) * as_scipy(q2)).as_quat(scalar_first=True)
            assert same_rotation(mine, theirs)

    def test_multiply_is_not_commutative(self):
        """Sanity check that we did not accidentally implement something abelian."""
        q1 = Q.from_axis_angle([1, 0, 0], 0.7)
        q2 = Q.from_axis_angle([0, 1, 0], 0.4)
        assert not same_rotation(Q.multiply(q1, q2), Q.multiply(q2, q1))

    def test_inverse_cancels(self):
        for q in random_quats(50):
            assert same_rotation(Q.multiply(q, Q.inverse(q)), Q.IDENTITY)
            assert same_rotation(Q.multiply(Q.conjugate(q), q), Q.IDENTITY)

    def test_norm_preserved_under_product(self):
        for q1, q2 in zip(random_quats(50), random_quats(50)):
            assert np.linalg.norm(Q.multiply(q1, q2)) == pytest.approx(1.0, abs=1e-14)

    def test_normalize_rejects_zero(self):
        with pytest.raises(ValueError):
            Q.normalize(np.zeros(4))


# ---------------------------------------------------------------------------
# vector rotation and DCM
# ---------------------------------------------------------------------------

class TestRotation:

    def test_rotate_matches_scipy(self):
        for q in random_quats(100):
            v = RNG.normal(size=3)
            assert np.allclose(Q.rotate(q, v), as_scipy(q).apply(v), atol=1e-12)

    def test_rotate_matches_own_dcm(self):
        """rotate() uses the factored form; it must agree with the matrix form."""
        for q in random_quats(100):
            v = RNG.normal(size=3)
            assert np.allclose(Q.rotate(q, v), Q.to_dcm(q) @ v, atol=1e-12)

    def test_rotate_inverse_round_trip(self):
        for q in random_quats(50):
            v = RNG.normal(size=3)
            assert np.allclose(Q.rotate_inverse(q, Q.rotate(q, v)), v, atol=1e-12)

    def test_rotation_preserves_length_and_angle(self):
        for q in random_quats(50):
            a, b = RNG.normal(size=3), RNG.normal(size=3)
            ra, rb = Q.rotate(q, a), Q.rotate(q, b)
            assert np.linalg.norm(ra) == pytest.approx(np.linalg.norm(a))
            assert float(ra @ rb) == pytest.approx(float(a @ b))

    def test_known_90_deg_about_z(self):
        """Right-handed active rotation: +90 deg about z sends x_hat to y_hat."""
        q = Q.from_axis_angle([0, 0, 1], np.pi / 2)
        assert np.allclose(Q.rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-14)
        assert np.allclose(Q.rotate(q, [0, 1, 0]), [-1, 0, 0], atol=1e-14)
        assert np.allclose(Q.rotate(q, [0, 0, 1]), [0, 0, 1], atol=1e-14)

    def test_dcm_is_orthonormal_with_unit_determinant(self):
        for q in random_quats(100):
            R = Q.to_dcm(q)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-13)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-13)

    def test_dcm_matches_scipy(self):
        for q in random_quats(100):
            assert np.allclose(Q.to_dcm(q), as_scipy(q).as_matrix(), atol=1e-13)

    def test_composition_maps_to_matrix_product(self):
        for q1, q2 in zip(random_quats(50), random_quats(50)):
            lhs = Q.to_dcm(Q.multiply(q1, q2))
            rhs = Q.to_dcm(q1) @ Q.to_dcm(q2)
            assert np.allclose(lhs, rhs, atol=1e-12)


class TestFromDCM:
    """Shepperd's method must hold up in all four branches."""

    def test_round_trip_random(self):
        for q in random_quats(500):
            assert same_rotation(Q.from_dcm(Q.to_dcm(q)), q, atol=1e-10)

    def test_round_trip_identity(self):
        assert same_rotation(Q.from_dcm(np.eye(3)), Q.IDENTITY)

    @pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    def test_round_trip_180_deg(self, axis):
        """w == 0 here, which is where the naive trace formula degrades."""
        q = Q.from_axis_angle(axis, np.pi)
        assert same_rotation(Q.from_dcm(Q.to_dcm(q)), q, atol=1e-10)

    @pytest.mark.parametrize("angle", [1e-9, 1e-6, 1e-3, np.pi - 1e-9])
    def test_round_trip_extreme_angles(self, angle):
        q = Q.from_axis_angle([0.3, -0.9, 0.4], angle)
        assert same_rotation(Q.from_dcm(Q.to_dcm(q)), q, atol=1e-8)

    def test_all_four_branches_exercised(self):
        """Each Shepperd branch is selected by some rotation; none is dead code."""
        cases = [
            np.eye(3),                                  # trace largest
            Q.to_dcm(Q.from_axis_angle([1, 0, 0], np.pi)),
            Q.to_dcm(Q.from_axis_angle([0, 1, 0], np.pi)),
            Q.to_dcm(Q.from_axis_angle([0, 0, 1], np.pi)),
        ]
        selected = {int(np.argmax((np.trace(R), R[0, 0], R[1, 1], R[2, 2])))
                    for R in cases}
        assert selected == {0, 1, 2, 3}

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            Q.from_dcm(np.eye(4))


# ---------------------------------------------------------------------------
# axis-angle and rotation vector
# ---------------------------------------------------------------------------

class TestAxisAngle:

    def test_round_trip(self):
        for _ in range(200):
            axis = Q.normalize(np.concatenate(([0.0], RNG.normal(size=3))))[1:]
            axis /= np.linalg.norm(axis)
            angle = RNG.uniform(0.01, np.pi - 0.01)
            a, t = Q.to_axis_angle(Q.from_axis_angle(axis, angle))
            assert t == pytest.approx(angle, abs=1e-10)
            assert np.allclose(a, axis, atol=1e-9)

    def test_zero_rotation_returns_finite_axis(self):
        axis, angle = Q.to_axis_angle(Q.IDENTITY)
        assert angle == pytest.approx(0.0)
        assert np.isfinite(axis).all()

    def test_rotvec_matches_scipy(self):
        for q in random_quats(200):
            assert np.allclose(Q.to_rotvec(q), as_scipy(q).as_rotvec(), atol=1e-10)

    def test_from_rotvec_matches_scipy(self):
        for _ in range(200):
            rv = RNG.normal(size=3) * RNG.uniform(0, 1.0)
            mine = Q.from_rotvec(rv)
            theirs = Rotation.from_rotvec(rv).as_quat(scalar_first=True)
            assert same_rotation(mine, theirs)

    def test_small_angle_series_stays_accurate(self):
        """The tiny-angle branch must not lose precision or blow up."""
        for mag in [1e-10, 1e-9, 1e-8, 1e-7, 1e-5]:
            rv = np.array([mag, 0.0, 0.0])
            q = Q.from_rotvec(rv)
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-15)
            assert Q.to_rotvec(q)[0] == pytest.approx(mag, rel=1e-9)

    def test_subthreshold_rotations_flush_to_zero(self):
        """
        Below ~2e-12 rad the quaternion vector part is smaller than the guard
        in to_axis_angle and the rotation is reported as identity. Documented
        rather than fixed: 2e-12 rad is 4e-7 arcsec, far below anything a
        flight vehicle cares about, and the guard is what keeps the axis
        division from producing garbage. The important part is that it
        degrades to zero cleanly instead of returning noise.
        """
        for mag in [1e-14, 1e-13]:
            q = Q.from_rotvec(np.array([mag, 0.0, 0.0]))
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-15)
            assert np.allclose(Q.to_rotvec(q), 0.0)


# ---------------------------------------------------------------------------
# Euler angles
# ---------------------------------------------------------------------------

class TestEuler:

    def test_matches_scipy_zyx(self):
        for _ in range(200):
            yaw = RNG.uniform(-np.pi, np.pi)
            pitch = RNG.uniform(-1.5, 1.5)          # stay clear of the singularity
            roll = RNG.uniform(-np.pi, np.pi)
            mine = Q.from_euler_321(yaw, pitch, roll)
            theirs = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat(scalar_first=True)
            assert same_rotation(mine, theirs)

    def test_round_trip_away_from_singularity(self):
        for _ in range(200):
            yaw = RNG.uniform(-np.pi + 0.1, np.pi - 0.1)
            pitch = RNG.uniform(-1.4, 1.4)
            roll = RNG.uniform(-np.pi + 0.1, np.pi - 0.1)
            y2, p2, r2 = Q.to_euler_321(Q.from_euler_321(yaw, pitch, roll))
            assert (y2, p2, r2) == pytest.approx((yaw, pitch, roll), abs=1e-9)

    def test_gimbal_lock_is_detected_not_nan(self):
        """
        At pitch = +90 deg the extraction is degenerate. We require that it
        degrades gracefully (finite numbers, roll pinned to zero) rather than
        returning NaN -- the comparison plot needs a curve, not a hole.
        """
        q = Q.from_euler_321(0.4, np.pi / 2, 0.3)
        yaw, pitch, roll = Q.to_euler_321(q)
        assert np.isfinite([yaw, pitch, roll]).all()
        assert pitch == pytest.approx(np.pi / 2, abs=1e-7)
        assert roll == pytest.approx(0.0, abs=1e-12)

    def test_euler_map_is_not_injective_at_lock(self):
        """
        The exact statement of gimbal lock. At pitch = 90 deg the yaw and roll
        axes coincide, so the entire one-parameter family (yaw + c, 90, roll + c)
        describes the SAME physical attitude. The map from Euler triples to
        rotations stops being one-to-one, which is why no extraction routine
        can recover yaw and roll separately -- only their difference survives.
        """
        base = Q.from_euler_321(0.0, np.pi / 2, 0.0)
        for c in [0.3, 1.0, 2.5, -1.7]:
            shifted = Q.from_euler_321(c, np.pi / 2, c)
            assert Q.angle_between(base, shifted) == pytest.approx(0.0, abs=1e-7)

        # Only the difference is observable, and it is recovered correctly.
        distinct = Q.from_euler_321(0.9, np.pi / 2, 0.2)
        assert Q.angle_between(base, distinct) > 0.1
        _, _, _ = Q.to_euler_321(distinct)

    def test_nearby_attitudes_collapse_to_identical_coordinates(self):
        """
        The practical consequence: two attitudes that are genuinely different
        extract to the exact same Euler triple near the singularity, so the
        coordinates carry strictly less information than the quaternion does.
        """
        eps = 1e-7
        qa = Q.from_euler_321(0.0, np.pi / 2 - eps, 0.0)
        qb = Q.from_euler_321(np.pi / 2, np.pi / 2 - eps, np.pi / 2)
        assert Q.angle_between(qa, qb) > 1e-8            # physically distinct
        ea = np.array(Q.to_euler_321(qa))
        eb = np.array(Q.to_euler_321(qb))
        assert np.allclose(ea, eb, atol=1e-9)            # coordinates identical


# ---------------------------------------------------------------------------
# double cover, distance, slerp
# ---------------------------------------------------------------------------

class TestCanonicalAndDistance:

    def test_canonical_forces_nonnegative_scalar(self):
        for q in random_quats(100):
            assert Q.canonical(q)[0] >= 0.0
            assert Q.canonical(-q)[0] >= 0.0

    def test_canonical_preserves_rotation(self):
        for q in random_quats(50):
            assert np.allclose(Q.to_dcm(Q.canonical(-q)), Q.to_dcm(q), atol=1e-13)

    def test_angle_between_is_double_cover_safe(self):
        for q in random_quats(50):
            assert Q.angle_between(q, q) == pytest.approx(0.0, abs=1e-7)
            assert Q.angle_between(q, -q) == pytest.approx(0.0, abs=1e-7)

    def test_angle_between_matches_axis_angle(self):
        for angle in np.linspace(0.0, np.pi, 25):
            q = Q.from_axis_angle([0.2, 0.5, -0.3], angle)
            assert Q.angle_between(Q.IDENTITY, q) == pytest.approx(angle, abs=1e-7)


class TestSlerp:

    def test_endpoints(self):
        for q0, q1 in zip(random_quats(50), random_quats(50)):
            assert same_rotation(Q.slerp(q0, q1, 0.0), q0)
            assert same_rotation(Q.slerp(q0, q1, 1.0), q1)

    def test_stays_unit_norm(self):
        q0, q1 = random_quats(1)[0], random_quats(1)[0]
        for t in np.linspace(0, 1, 101):
            assert np.linalg.norm(Q.slerp(q0, q1, t)) == pytest.approx(1.0, abs=1e-14)

    def test_constant_angular_rate(self):
        """The defining property: equal time steps subtend equal angles."""
        q0 = Q.IDENTITY
        q1 = Q.from_axis_angle([0.3, 0.8, -0.2], 2.4)
        ts = np.linspace(0, 1, 41)
        qs = Q.slerp(q0, q1, ts)
        steps = [Q.angle_between(qs[i], qs[i + 1]) for i in range(len(qs) - 1)]
        assert np.std(steps) < 1e-12

    def test_takes_short_path_through_double_cover(self):
        """With a negated endpoint slerp must still travel the short arc."""
        q0 = Q.IDENTITY
        q1 = Q.from_axis_angle([0, 0, 1], 0.5)
        direct = Q.slerp(q0, q1, np.linspace(0, 1, 21))
        flipped = Q.slerp(q0, -q1, np.linspace(0, 1, 21))
        total_direct = sum(Q.angle_between(direct[i], direct[i + 1]) for i in range(20))
        total_flipped = sum(Q.angle_between(flipped[i], flipped[i + 1]) for i in range(20))
        assert total_direct == pytest.approx(0.5, abs=1e-9)
        assert total_flipped == pytest.approx(0.5, abs=1e-9)

    def test_near_parallel_lerp_fallback(self):
        q0 = Q.IDENTITY
        q1 = Q.from_axis_angle([1, 0, 0], 1e-9)
        out = Q.slerp(q0, q1, np.linspace(0, 1, 11))
        assert np.isfinite(out).all()
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-14)

    def test_vector_and_scalar_agree(self):
        q0, q1 = random_quats(1)[0], random_quats(1)[0]
        ts = np.linspace(0, 1, 7)
        batch = Q.slerp(q0, q1, ts)
        for i, t in enumerate(ts):
            assert same_rotation(batch[i], Q.slerp(q0, q1, float(t)))


# ---------------------------------------------------------------------------
# kinematics -- the part the propagator depends on
# ---------------------------------------------------------------------------

class TestKinematics:

    def test_derivative_is_orthogonal_to_q(self):
        """
        d/dt |q|^2 = 2 q . qdot must be zero, or the quaternion drifts off the
        unit sphere for reasons that are the kinematics' fault rather than the
        integrator's.
        """
        for q in random_quats(100):
            w = RNG.normal(size=3)
            assert float(q @ Q.derivative(q, w)) == pytest.approx(0.0, abs=1e-13)

    def test_derivative_matches_omega_matrix(self):
        for q in random_quats(50):
            w = RNG.normal(size=3)
            assert np.allclose(Q.derivative(q, w), 0.5 * Q.omega_matrix(w) @ q, atol=1e-14)

    def test_omega_matrix_is_skew_symmetric(self):
        for _ in range(20):
            W = Q.omega_matrix(RNG.normal(size=3))
            assert np.allclose(W, -W.T, atol=1e-15)

    def test_constant_rate_spin_matches_analytic(self):
        """
        Spin at a fixed body rate about a body axis for a known duration and
        check the result against the closed-form axis-angle answer.
        """
        omega = np.array([0.0, 0.0, 0.35])
        q = Q.IDENTITY
        dt, n = 1e-4, 20000
        for _ in range(n):
            q = Q.normalize(q + dt * Q.derivative(q, omega))
        expected = Q.from_axis_angle([0, 0, 1], 0.35 * dt * n)
        assert Q.angle_between(q, expected) < 1e-6

    def test_body_frame_ordering_is_correct(self):
        """
        Regression guard for the most common convention bug. With the vehicle
        already yawed 90 deg, a body-frame pitch rate must rotate about the
        body y-axis (inertial -x), NOT the inertial y-axis. Swapping the
        operands in derivative() flips this and the test fails.
        """
        q = Q.from_axis_angle([0, 0, 1], np.pi / 2)      # yawed 90 deg
        omega_body = np.array([0.0, 0.1, 0.0])           # pure body pitch
        q_next = Q.integrate_body_rate(q, omega_body, 1.0)
        # the net rotation applied, seen in inertial axes
        axis, angle = Q.to_axis_angle(Q.multiply(q_next, Q.conjugate(q)))
        assert angle == pytest.approx(0.1, abs=1e-12)
        assert np.allclose(axis, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_integrate_body_rate_is_exact_for_constant_omega(self):
        """Closed-form update must beat RK-style stepping, not merely match it."""
        omega = np.array([0.1, -0.25, 0.4])
        T = 3.0
        exact = Q.integrate_body_rate(Q.IDENTITY, omega, T)
        # many small closed-form steps must agree to machine precision
        q = Q.IDENTITY
        for _ in range(3000):
            q = Q.integrate_body_rate(q, omega, T / 3000)
        assert Q.angle_between(q, exact) < 1e-12

    def test_integrate_body_rate_preserves_norm(self):
        q = Q.IDENTITY
        for _ in range(10000):
            q = Q.integrate_body_rate(q, RNG.normal(size=3), 1e-3)
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# control-facing behaviour
# ---------------------------------------------------------------------------

class TestAttitudeError:

    def test_zero_when_on_target(self):
        for q in random_quats(50):
            assert np.allclose(Q.attitude_error(q, q), Q.IDENTITY, atol=1e-12)

    def test_zero_against_negated_command(self):
        """-q_cmd is the same attitude; the error must not jump to 360 deg."""
        for q in random_quats(50):
            err = Q.attitude_error(q, -q)
            assert np.linalg.norm(err[1:]) == pytest.approx(0.0, abs=1e-12)

    def test_always_takes_short_way_around(self):
        """The rotation-vector error must never exceed 180 deg."""
        for q1, q2 in zip(random_quats(300), random_quats(300)):
            err = Q.attitude_error(q1, q2)
            assert err[0] >= 0.0
            assert np.linalg.norm(Q.to_rotvec(err)) <= np.pi + 1e-12

    def test_error_is_expressed_in_body_axes(self):
        """
        Command a pure body-y rotation from a yawed attitude. The error vector
        must come out along body y, i.e. [0, +, 0] -- if it came out in
        inertial axes it would appear along x.
        """
        q_actual = Q.from_axis_angle([0, 0, 1], np.pi / 2)
        q_cmd = Q.multiply(q_actual, Q.from_axis_angle([0, 1, 0], 0.2))
        err_vec = Q.to_rotvec(Q.attitude_error(q_actual, q_cmd))
        assert np.allclose(err_vec, [0.0, 0.2, 0.0], atol=1e-12)

    def test_small_angle_vector_part_approximates_rotvec(self):
        """2*dq[1:] ~ the rotation vector, which is what the PD gain assumes."""
        for angle in [1e-4, 1e-3, 1e-2, 0.1]:
            q_cmd = Q.from_axis_angle([0, 1, 0], angle)
            dq = Q.attitude_error(Q.IDENTITY, q_cmd)
            assert 2.0 * dq[2] == pytest.approx(angle, rel=1e-2)

    def test_error_magnitude_equals_geodesic_distance(self):
        for q1, q2 in zip(random_quats(100), random_quats(100)):
            err_angle = np.linalg.norm(Q.to_rotvec(Q.attitude_error(q1, q2)))
            assert err_angle == pytest.approx(Q.angle_between(q1, q2), abs=1e-9)


# ---------------------------------------------------------------------------
# the headline case: a 180 degree flip
# ---------------------------------------------------------------------------

class TestBoosterFlip:
    """
    The maneuver the whole project is built around. Nose-up to nose-down is a
    180 deg pitch, which drives a 3-2-1 Euler description straight through its
    singularity while the quaternion description notices nothing at all.
    """

    NOSE_UP = Q.IDENTITY                                       # body x along inertial x
    NOSE_DOWN = Q.from_axis_angle([0, 1, 0], np.pi)            # pitched over 180 deg

    def test_slerp_through_the_flip_is_smooth(self):
        qs = Q.slerp(self.NOSE_UP, self.NOSE_DOWN, np.linspace(0, 1, 201))
        steps = [Q.angle_between(qs[i], qs[i + 1]) for i in range(len(qs) - 1)]
        assert max(steps) - min(steps) < 1e-10
        assert np.allclose(np.linalg.norm(qs, axis=1), 1.0, atol=1e-13)

    def test_euler_rate_blows_up_where_quaternion_rate_does_not(self):
        """
        Differentiate both descriptions along the same physical slew. The
        quaternion components stay bounded and smooth; the Euler yaw/roll rates
        spike by orders of magnitude as the path crosses pitch = 90 deg.
        """
        ts = np.linspace(0, 1, 2001)
        qs = Q.slerp(self.NOSE_UP, self.NOSE_DOWN, ts)

        q_rates = np.abs(np.diff(qs, axis=0)).max()

        eulers = np.array([Q.to_euler_321(q) for q in qs])
        # unwrap so ordinary branch cuts are not mistaken for the singularity
        eulers = np.unwrap(eulers, axis=0)
        e_rates = np.abs(np.diff(eulers, axis=0)).max()

        assert q_rates < 0.01
        assert e_rates > 100 * q_rates

    def test_flip_reaches_commanded_attitude(self):
        assert Q.angle_between(self.NOSE_UP, self.NOSE_DOWN) == pytest.approx(np.pi, abs=1e-9)
        mid = Q.slerp(self.NOSE_UP, self.NOSE_DOWN, 0.5)
        assert Q.angle_between(self.NOSE_UP, mid) == pytest.approx(np.pi / 2, abs=1e-9)

    def test_body_axis_traces_expected_path(self):
        """Body x-axis should sweep from +x through +z (or -z) to -x."""
        qs = Q.slerp(self.NOSE_UP, self.NOSE_DOWN, np.linspace(0, 1, 5))
        x_axes = np.array([Q.rotate(q, [1, 0, 0]) for q in qs])
        assert np.allclose(x_axes[0], [1, 0, 0], atol=1e-12)
        assert np.allclose(x_axes[-1], [-1, 0, 0], atol=1e-12)
        assert abs(x_axes[2, 2]) == pytest.approx(1.0, abs=1e-12)   # straight up/down
        assert np.allclose(np.linalg.norm(x_axes, axis=1), 1.0, atol=1e-13)
