"""
CFD aerodynamics for CatchSim (drop-in for quatsim/aero.py's guessed coefficients).

In CatchSim: python run_mission.py --aero-cfd aero_db.json (the aero_db.json written by
rocketcfd's cases/aero_database.py). In code:

    from quatsim.aero_cfd import CFDAeroModel, cfd_restoring_moment, CFDAeroDatabase
    db = CFDAeroDatabase("aero_db.json")
    aero = CFDAeroModel(db=db)                       # replaces AeroModel(...)
    M = cfd_restoring_moment(db, q, v, h, omega)    # replaces aero.restoring_moment(...)

What it replaces
  * AeroModel.drag_force: CD at zero incidence from CFD (grid fins included, so no
    fin_drag_factor). This is all a point-mass (3-DOF) caller can use.
  * AeroModel.normal_force: the rest of the full CFD force at the actual angle of attack
    and roll attitude, so drag_force + normal_force = the complete aerodynamic force.
  * restoring_moment: the full CFD moment (pitch, yaw and roll) about the centre of mass,
    plus CatchSim's pitch damping term (Cm_q is not computed by the CFD).

Axes (see rocketcfd/catchsim_export.py): body x from the engines toward the top, which is
CatchSim's nose axis. Roll attitude is the azimuth of the windward side in the
spacecraft-models dimensions.md convention (0 = rudder fin, 90 / 270 = catch fins,
180 = raceway). `roll_offset_deg` maps CatchSim's body y/z axes onto that convention: with
offset 0, CatchSim +y is rocketcfd +y (azimuth 270, a catch fin) and +z is the rudder fin.

Valid range: the Mach numbers and angles of attack in the table (clamped outside), engines-
first flight (angle of attack measured from engines-first, 0-90 deg). Inviscid tables give
pressure forces only; viscous tables include skin friction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

try:                                  # inside catchsim/quatsim
    from . import environment as ENV
    from . import quaternion as Q
    from .aero import AeroModel
except ImportError:                   # standalone use / tests
    ENV = Q = None
    AeroModel = object


def _interp_axis(grid, x):
    """Index and weight of x in a sorted 1-D grid (clamped)."""
    g = np.asarray(grid, float)
    if len(g) == 1:
        return 0, 0, 0.0
    x = float(np.clip(x, g[0], g[-1]))
    i = int(np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2))
    return i, i + 1, (x - g[i]) / (g[i + 1] - g[i])


class CFDAeroDatabase:
    """Trilinear lookup in (Mach, angle of attack from engines-first, roll)."""

    def __init__(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.d = d
        self.mach = np.asarray(d["mach"], float)
        self.aoa = np.asarray(d["aoa_deg"], float)
        self.roll = np.asarray(d["roll_deg"], float)
        self.tables = {k: np.asarray([[[np.nan if v is None else v for v in r] for r in a] for a in t], float)
                       for k, t in d["tables"].items()}
        self.S = d["reference"]["S_ref"]
        self.L = d["reference"]["L_ref"]
        self.x_ref = d["reference"]["x_ref"][0]
        self.sign = d["mirror_sign"]
        self.swap = d["fins"]["mirror_swap"]
        cq = d.get("cm_q_quasi_steady")
        self.cm_q_table = None if cq is None else np.asarray([[np.nan if v is None else v for v in r] for r in cq], float)

    def cm_q(self, mach, roll_deg=0.0, default=12.0):
        """Quasi-steady pitch damping (CatchSim normalisation, positive = damping) at this Mach,
        averaged over the computed roll angles; `default` where the table has none."""
        if self.cm_q_table is None:
            return default
        im = _interp_axis(self.mach, mach)
        col = (1 - im[2]) * self.cm_q_table[im[0]] + im[2] * self.cm_q_table[im[1]]
        col = col[np.isfinite(col)]
        return float(col.mean()) if col.size else default

    def coefficients(self, mach, aoa_deg, roll_deg):
        """Crossflow-frame coefficients at a flight condition (dict). roll in degrees, any value."""
        r = float(roll_deg) % 360.0
        mirrored = r > 180.0
        if mirrored:
            r = 360.0 - r
        # Beyond the last computed incidence (a landing burn leans the booster up to ~30 deg
        # off the relative wind; the tables stop at 20): crossflow terms keep growing like
        # sin(alpha), the axial force like cos(alpha), from the last computed column, instead of
        # freezing there.
        a_max = float(self.aoa[-1])
        beyond = float(aoa_deg) > a_max and 0.0 < a_max < 90.0
        k_cross = (np.sin(np.radians(min(float(aoa_deg), 90.0))) / np.sin(np.radians(a_max))
                   if beyond else 1.0)
        k_axial = (np.cos(np.radians(min(float(aoa_deg), 90.0))) / np.cos(np.radians(a_max))
                   if beyond else 1.0)
        im = _interp_axis(self.mach, mach)
        ia = _interp_axis(self.aoa, aoa_deg)
        ir = _interp_axis(self.roll, r)
        out = {}
        for k, T in self.tables.items():
            v = 0.0
            for (i, wi) in ((im[0], 1 - im[2]), (im[1], im[2])):
                for (j, wj) in ((ia[0], 1 - ia[2]), (ia[1], ia[2])):
                    for (l, wl) in ((ir[0], 1 - ir[2]), (ir[1], ir[2])):
                        w = wi * wj * wl
                        if w > 0.0:
                            v += w * T[i, j, l]
            out[k] = v
        if beyond:
            for k in out:
                if k in ("CN", "CS", "Cl", "C_cross", "C_restore"):
                    out[k] *= k_cross
                elif k == "CX":
                    out[k] *= k_axial
        if mirrored:
            for k in list(out):
                if k in self.sign:
                    out[k] *= self.sign[k]
            hinge = {k: v for k, v in out.items() if k.startswith("C_hinge[")}
            for k, v in hinge.items():
                name = k[len("C_hinge["):-1]
                out[f"C_hinge[{self.swap.get(name, name)}]"] = -v
        return out


def crossflow_frame(roll_deg):
    tw = np.radians(90.0 + roll_deg)
    cross = np.array([0.0, -np.cos(tw), -np.sin(tw)])
    return cross, np.cross([1.0, 0.0, 0.0], cross)


def body_loads(db: CFDAeroDatabase, wind_body, mach, q_dyn, x_cg=None, roll_offset_deg=0.0):
    """Aerodynamic force [N] and moment about the centre of mass [N m], body frame.

    wind_body: relative wind in body axes (air velocity relative to the vehicle; engines-first
    flight has wind_body ~ +x). x_cg: centre of mass measured from the aft end along +x
    (default: the table's reference point)."""
    w = np.asarray(wind_body, float)
    speed = float(np.linalg.norm(w))
    if speed < 1e-6 or q_dyn <= 0.0:
        return np.zeros(3), np.zeros(3)
    e = w / speed
    aoa = float(np.degrees(np.arccos(np.clip(e[0], -1.0, 1.0))))      # 0 = exactly engines-first
    perp = np.array([0.0, e[1], e[2]])
    if np.linalg.norm(perp) > 1e-9:
        cross = perp / np.linalg.norm(perp)
        # windward azimuth: the crossflow comes FROM the windward side, so windward = -cross
        ang = np.degrees(np.arctan2(-cross[2], -cross[1]))           # angle in the y-z plane
        roll = (ang - 90.0 + roll_offset_deg) % 360.0
    else:
        cross = crossflow_frame(-roll_offset_deg)[0]
        roll = 0.0
    c = db.coefficients(mach, aoa, roll)
    ex = np.array([1.0, 0.0, 0.0])
    side = np.cross(ex, cross)              # the frame comes from the actual wind, in the caller's axes
    F = q_dyn * db.S * (c["CX"] * ex + c["CN"] * cross + c["CS"] * side)
    M = q_dyn * db.S * db.L * (c["Cl"] * ex + c["C_cross"] * cross + c["C_restore"] * side)
    if x_cg is not None:                    # move the moment from x_ref to the centre of mass
        M = M + np.cross((db.x_ref - x_cg) * ex, F)
    return F, M


@dataclass
class CFDAeroModel(AeroModel):
    """AeroModel with forces from the CFD table (inherits wind and dispersion hooks)."""

    db: CFDAeroDatabase = None
    x_cg: float | None = None
    roll_offset_deg: float = 0.0
    fin_drag_factor: float = 1.0            # fins are in the CFD; keep 1.0

    def _q(self, speed, altitude):
        return 0.5 * ENV.density(altitude) * self.density_scale * speed * speed

    def drag_force(self, velocity, altitude, fins_deployed=True):
        """Drag at zero incidence (point-mass callers): -q S CD(M, 0) v / |v|."""
        v = self.air_velocity(velocity, altitude)
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return np.zeros(3)
        mach = ENV.mach(v, altitude)
        cd = self.db.coefficients(mach, 0.0, 0.0)["CD"] * self.cd_scale
        return -(self._q(speed, altitude) * cd * self.db.S) * (v / speed)

    def airframe_moment(self, q, v_air, altitude, omega=None):
        """The full CFD moment about the centre of mass plus pitch damping (replaces the
        guessed weathercocking model). v_air is already relative to the air."""
        return cfd_restoring_moment(self.db, q, v_air, altitude, omega=omega, x_cg=self.x_cg,
                                    roll_offset_deg=self.roll_offset_deg,
                                    density_scale=self.density_scale)

    def normal_force(self, q, velocity, altitude):
        """Full CFD force at the actual attitude minus drag_force, so the two sum to the total."""
        v = self.air_velocity(velocity, altitude)
        speed = float(np.linalg.norm(v))
        if speed < 1.0:
            return np.zeros(3)
        mach = ENV.mach(v, altitude)
        wind_body = Q.rotate_inverse(q, -v)
        F_body, _ = body_loads(self.db, wind_body, mach, self._q(speed, altitude),
                               roll_offset_deg=self.roll_offset_deg)
        F = Q.rotate(q, F_body) * self.cd_scale
        return F - self.drag_force(velocity, altitude)


def cfd_restoring_moment(db: CFDAeroDatabase, q, velocity, altitude, omega=None, cm_q=None,
                         x_cg=None, roll_offset_deg=0.0, density_scale=1.0):
    """Body-frame aerodynamic moment about the centre of mass from the CFD table, plus pitch
    damping in CatchSim's form; cm_q None = the table's quasi-steady estimate at this Mach
    (CatchSim's 12.0 if the table has none)."""
    v = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed < 1.0:
        return np.zeros(3)
    q_dyn = 0.5 * ENV.density(altitude) * density_scale * speed * speed
    mach = ENV.mach(v, altitude)
    _, M = body_loads(db, Q.rotate_inverse(q, -v), mach, q_dyn, x_cg=x_cg, roll_offset_deg=roll_offset_deg)
    if omega is not None:
        if cm_q is None:
            cm_q = db.cm_q(mach)
        w = np.asarray(omega, dtype=float)
        M = M - (cm_q * q_dyn * db.S * db.L * db.L / (2.0 * speed)) * np.array([0.0, w[1], w[2]])
    return M


class CFDFinControl:
    """Fin control derivatives (rocketcfd.catchsim_export.build_fin_control -> fin_control.json).

    control_loads(mach, aoa_deg, deflections_rad, q_dyn) -> (force N, moment N m about x_ref,
    hinge moments N m) in body axes, by linear superposition of each fin's derivatives:
    deflections are (rudder fin, catch fin 90, catch fin 270), positive about each fin's
    outward radial axis. Derivatives were computed at roll 0 / 180 and are used at any roll
    (the fins' own effect depends mainly on Mach and incidence).
    """

    def __init__(self, path, db: CFDAeroDatabase):
        with open(path) as fh:
            d = json.load(fh)
        self.db = db
        self.fins = []
        for name in ("rudder fin", "catch fin 90", "catch fin 270"):
            rows = d["fins"].get(name, [])
            mach = sorted({r["mach"] for r in rows})
            aoa = sorted({r["aoa"] for r in rows})
            dF = np.zeros((len(mach), len(aoa), 3))
            dM = np.zeros((len(mach), len(aoa), 3))
            dH = np.zeros((len(mach), len(aoa)))
            seen = np.zeros((len(mach), len(aoa)), bool)
            for r in rows:
                i, j = mach.index(r["mach"]), aoa.index(r["aoa"])
                if seen[i, j]:
                    continue
                dF[i, j], dM[i, j] = r["dF"], r["dM"]
                dH[i, j] = r["dC_h"] or 0.0
                seen[i, j] = True
            self.fins.append(dict(name=name, mach=mach, aoa=aoa, dF=dF, dM=dM, dH=dH))
        self.S_fin = 11.26
        self.c_fin = 0.5

    def _lookup(self, f, mach, aoa):
        if not f["mach"]:
            return np.zeros(3), np.zeros(3), 0.0
        im = _interp_axis(f["mach"], mach)
        ia = _interp_axis(f["aoa"], aoa)
        out = [0.0, 0.0, 0.0]
        for (i, wi) in ((im[0], 1 - im[2]), (im[1], im[2])):
            for (j, wj) in ((ia[0], 1 - ia[2]), (ia[1], ia[2])):
                w = wi * wj
                out[0] = out[0] + w * f["dF"][i, j]
                out[1] = out[1] + w * f["dM"][i, j]
                out[2] = out[2] + w * f["dH"][i, j]
        return out

    def control_loads(self, mach, aoa_deg, deflections_rad, q_dyn):
        F = np.zeros(3)
        M = np.zeros(3)
        H = np.zeros(3)
        for k, (f, dl) in enumerate(zip(self.fins, deflections_rad)):
            dF, dM, dH = self._lookup(f, mach, aoa_deg)
            F += q_dyn * self.db.S * np.asarray(dF) * dl
            M += q_dyn * self.db.S * self.db.L * np.asarray(dM) * dl
            H[k] = q_dyn * self.S_fin * self.c_fin * dH * dl
        return F, M, H
