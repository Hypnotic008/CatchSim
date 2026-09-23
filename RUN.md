# Running quatsim

See `README.md` for the full picture. Short version:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Python 3.10+, NumPy 2.x
python -m pytest tests -q          # expect 73 passed

python run_mission.py                      # fly the nominal mission, print the catch report
python run_mission.py --figures            # + figs/mission_overview.png, figs/landing_detail.png
python run_mission.py --animate3d          # + figs/catch3d.mp4 and figs/catch3d_viewer.html (needs `cd viz3d && npm install`)
python run_mission.py --montecarlo 100     # + figs/monte_carlo.{png,csv,pkl}
python run_mission.py --no-steer           # fly without grid-fin entry steering, for comparison
```

MP4 export uses `ffmpeg` from PATH, or the `imageio-ffmpeg` wheel in
`requirements.txt`; without either it falls back to GIF.

Runtime: ~30 s per trajectory at `dt = 0.02` (the boostback predictor
re-solves in flight). The Monte Carlo runs cases in parallel (`--workers`).

From Python:

```python
import run_mission as RM
from quatsim.phases import catch_report
from quatsim import visuals as VIS

vehicle, aero, sep, bb0, land, seq = RM.build()
out = RM.fly(RM.T33_BURN, vehicle, aero, sep, bb0, land, seq)
rep = catch_report(out["state"][-1], RM.TARGET)
VIS.landing_detail(out, RM.TARGET, rep, path="figs/landing_detail.png")
```

Tuning lives in `quatsim/landing.py::LandingConfig` and
`quatsim/entry.py::EntryConfig`; the boostback aim point is
`run_mission.AIM_X_1200`.
