# Running quatsim

## 1. Unzip and set up

```bash
unzip quatsim.zip
cd quatsim
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. NumPy 2.x is assumed -- the code uses `np.ptp(arr)`
rather than the `arr.ptp()` method that NumPy 2 removed.

## 2. Check it works

```bash
python -m pytest tests/ -q
```

Expect `64 passed`. These are the quaternion library tests, cross-validated
against SciPy. If they fail, nothing downstream is trustworthy.

## 3. Run the mission

```bash
jupyter notebook main.ipynb
```

Run the cells top to bottom. Cell 6 flies the converged solution and prints
the catch report (5 of 6 criteria pass). Cell 7 re-solves the burn duration
from scratch and takes about 3 minutes.

Or from a script, without Jupyter:

```python
import numpy as np
from quatsim.vehicle import Vehicle
from quatsim.aero import AeroModel, GridFinModel
from quatsim.control import AttitudeController, ControlGains
from quatsim.position import PositionController, PositionGains
from quatsim.phases import FlightSequencer, solve_ignition_altitude, catch_report
from quatsim.mission import (SeparationState, solve_boostback,
                             propagate_flip, build_mission)

V, aero, fins = Vehicle(prop_mass=500e3), AeroModel(), GridFinModel()
TARGET = np.array([0., 0., 105.])
sep = SeparationState(altitude=68e3, downrange=83e3, speed=1500.,
                      flight_path_angle=25., prop_remaining=500e3)

sf   = propagate_flip(V, aero, sep, 21.0, n_lit=5)
bb0  = solve_boostback(V, aero, sep, start_state=sf, throttle=0.87,
                       bracket=(1., 40.))
land = solve_ignition_altitude(V, aero, bb0['prop_after'], 105., 13, 0.45,
                               v_entry=bb0['arrival_speed'])

seq = FlightSequencer(V, aero,
                      AttitudeController(V, ControlGains(wn=1.5, zeta=0.8)),
                      PositionController(V, PositionGains(wn=0.35, zeta=0.95)),
                      fins=fins)

bb = dict(bb0); bb['duration'] = 11.8863 + 6.0
segs = build_mission(V, aero, sep, bb, land, TARGET, np.zeros(3),
                     flip_duration=21.0, flip_engines=5,
                     boostback_elevation=np.radians(-14.0))
for s in segs:
    if s.name.startswith('boostback'):
        s.throttle = 0.87
    if s.name == 'landing':
        s.design_frac = 0.35; s.k_v = 2.0; s.v_touch = 0.2
        s.k_lat = 0.6; s.max_tilt = np.radians(15.)
        s.taper_altitude = 600.0; s.catch_altitude = 105.0 - 7.7

out = seq.run(sep.state_vector(), segs, dt=0.02, log_every=20)
print(catch_report(out['state'][-1], TARGET))
```

Runtime: about 12 s for one trajectory at `dt=0.02`.

## 4. Figures and animation

```python
from quatsim import telemetry as T
from quatsim.realtime import animate_realtime

T.tracking_error(out,   path='tracking_error.png')
T.telemetry_panel(out,  target=TARGET, path='panel.png')
T.animate_mission(out,  TARGET, n_frames=150, fps=24, path='summary.gif')

# speed=1.0 is true real time: ~240 s of video, and SLOW to render
# (roughly 20-40 min). speed=4.0 keeps phase proportions honest in ~8 min.
animate_realtime(out, TARGET, path='mission.mp4', speed=1.0, fps=24)
```

MP4 output needs `ffmpeg` on your PATH. Without it, pass a `.gif` filename
instead -- much larger files, but no external dependency.

## 5. If you change anything

Any change to the vehicle, aero, or guidance invalidates the burn duration.
Re-solve it (notebook cell 7) rather than keeping 11.8863 -- that number is
specific to this exact configuration, and the trajectory is sensitive at
roughly 35 km of range per second of boostback.
