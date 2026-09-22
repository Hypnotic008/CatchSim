# quatsim — Super Heavy RTLS simulation

## Status: 6/7 on the single-phase baseline. Two-phase landing is phase-1-complete.

```
lateral_error_m         0.524  PASS   (tower absorbs position error)
fin_roll_deg            0.006  PASS
vertical_error_m        0.000  PASS
vertical_speed_ms       0.261  PASS
tilt_deg                0.412  PASS
body_rate_deg_s         0.580  PASS
horizontal_speed_ms     3.638  FAIL   (limit 0.5)
```

## PHASE 1 OF THE TWO-PHASE LANDING: DONE, AND KEEP IT

`solve_brake=True` replaces the fixed `design_frac * a_avail` deceleration with
continuous energy matching -- recompute, every step, the deceleration still
required to reach the handover FROM THE CURRENT STATE:

    a = (vz^2 - v_hand^2) / (2 * (h - h_hand))

Throttle becomes an OUTPUT of tracking that, not a commanded schedule.

| | before | after |
|---|---|---|
| handover velocity error | ~32 m/s | 1.53 m/s |
| 3-engine phase duration | 5.8 s | 35.6 s |
| ignition altitude | 2995 m | 1393 m (Flight 9 says 1-2 km) |
| 13-engine burn | 14 s | 4.8 s |
| throttle | commanded | 0.40-0.62, emergent |

Ignition is the capability test against the HANDOVER target, not against a
velocity profile -- the first attempt was circular (a_brake computed at
ignition, but ignition needed a_brake) and the burn never lit.

Best settings: `brake_margin=0.40`, `handover_altitude=180`,
`v_terminal_descent=8`.

## TWO-PHASE LANDING: VALIDATED CHECKPOINT

The phase-2 architecture is now converted into a passing end-to-end run. The
important fixes were structural, not a gain sweep:

1. **The landing segment no longer terminates at the handover velocity.** In
   two-phase mode, reaching the handover state starts the 3-engine approach;
   the segment ends only at catch altitude.
2. **The 3-engine vertical controller is a true precision controller.** It
   tracks the gentle descent reference around hover instead of carrying the
   13-engine braking acceleration into phase 2.
3. **Lateral stopping is solved from the current state.** For each horizontal
   axis, when moving toward the target faster than the handover allowance, the
   controller uses

       a_stop = (v^2 - v_handover^2) / (2*d)

   and makes throttle/attitude the actuator outputs. The x and y axes are
   independent, so the nominal planar y=vy=0 state remains invariant instead of
   rotating a tiny crossrange perturbation into a large 2-D radial command.
4. **Roll is separated from lateral pointing.** During the 13-engine brake the
   current roll reference is preserved. Across the terminal altitude window the
   reference transitions smoothly to the catch-fin alignment. This removed the
   ~500 m crossrange injection caused by changing the thrust-frame roll while
   simultaneously commanding lateral acceleration.
5. **The terminal tilt taper is separate from the long landing taper.** The
   main 600 m taper no longer consumes the lateral authority needed by the
   3-engine precision phase.
6. **The 40% main-burn throttle floor is not imposed on the final 3-engine
   approach.** The model uses a 34% effective terminal floor because three
   engines at 40% have T/W > 1 in this mass model and otherwise the vehicle
   bounces upward above the catch plane. This is an explicit modelling
   assumption, not a measured Raptor minimum-throttle value.

Validated runnable configuration (`run_mission.py`):

```
hand-over altitude       250 m
hand-over vertical speed   8 m/s downward
brake margin              0.40
terminal taper            145 m
terminal throttle floor   0.34
13 -> 3 engine transition at handover
```

One end-to-end run at `dt=0.02` gives:

```
lateral error             0.581 m   PASS
fin roll                  ~0.000 deg PASS
vertical error             0.011 m   PASS
horizontal speed           0.075 m/s PASS
vertical speed             0.427 m/s PASS
tilt                       0.198 deg PASS
body rate                  ~0.000 deg/s PASS
```

This is a 7/7 catch in the current model. The vehicle also finishes with about
52.7 t of propellant remaining. The landing-burn timing is still longer than
the Flight 7 timing constraint, so timing should be treated as the next
validation item rather than claiming the burn sequence is already matched.

The earlier crossrange anomaly was real, but it was not a mysterious lateral
force: the landing attitude arrived with an ambiguous 180-degree roll state,
and the old attitude/thrust-vector command coupled that roll correction into
the lateral maneuver. Logging body-z and body-x through the landing phase made
that visible.

## Tower model

The chopsticks track the booster, so POSITION error inside the arm envelope is
absorbed. VELOCITY is not -- the arms can be elsewhere, they cannot match the
vehicle's speed. Flight 6 was waved off when the tower link dropped, which is
the same statement.

Envelope (ESTIMATES): +/-5 m crossrange, +/-3 m downrange. Asymmetric because
the V2 arms are shorter than V1's, so travel along the arms exceeds travel
toward the tower. Scoring against a FIXED point models a harder problem than
the real one, and the wrong harder problem: it spends vehicle authority on
position the tower supplies, at the expense of velocity only the vehicle can.

Note the tower is used in SCORING only. Feeding it into the guidance as a
position deadband changes the trajectory and costs a criterion (6/7 -> 5/7).

## Validation -- four independent observations reproduced

| quantity | model | observed | source |
|---|---|---|---|
| landing burn ignition speed | 362.8 m/s | 361 m/s | Flight 9 livestream |
| apogee (at BB_ELEV = 7 deg) | 95.5 km | 95.7 km | Flight 13 livestream |
| 33-engine boostback | 11.01 s | ~11 s | Flight 13 timing |
| coast duration | 198 s | 204 s | published timeline |

Separation downrange (83 km) was fitted to ONE number -- the 51 km splashdown.
Everything above is independent prediction.

Engine sequence 5 / 33 / 13 / 3 / 0 / 13 matches flown hardware.

Note BB_ELEVATION trades these: 1 deg matches the ignition velocity, 7 deg
matches apogee. They cannot both be matched with the current separation state,
which suggests the separation state still needs work.

## Terminal state (single point)

| criterion | value | limit | |
|---|---|---|---|
| lateral error | 2.73 m | 1 m | FAIL |
| horizontal speed | 3.23 m/s | 0.5 m/s | FAIL |
| vertical error | ~0.04 m | 1 m | PASS |
| vertical speed | ~0.3 m/s | 1 m/s | PASS |
| tilt from vertical | 0.325 deg | 0.5 deg | PASS |
| body rate | 0.464 deg/s | 1 deg/s | PASS |
| fin roll alignment | ~0 deg | 10 deg | PASS |

## The open problem: the basin is still a spike

Across t33 = 9.1472 +/- 0.016 s:

```
 k_lat  ldf |   9.1312   9.1392   9.1472   9.1552   9.1632
  0.90 0.15 |   390.93   152.27     2.81   244.98   546.26
  1.40 0.20 |   336.75   119.77     1.90   226.59   518.48
```

A good point solution surrounded by hundreds of metres. Gains barely move it
(391 -> 337 across a 2.3x change), which is again the saturation signature.

The landing burn HAS the authority on paper: 22 s at 30 deg tilt and
a_vert ~ 30 m/s^2 gives ~15 m/s^2 lateral, enough to move kilometres. So the
braking profile is not commanding what is available. That is where to look
next -- instrument v_ref_mag against d_lat over a FAILING duration (9.1312)
and compare to the passing one, exactly as the 9.155 vs 9.134 comparison
found the last root cause.

## Rules learned the hard way

1. **Shooting solve must use the same dt as evaluation.** dt=0.03 vs dt=0.02
   moved lateral error from 3.5 m to 47 m.
2. **Every structural change invalidates T33_BURN and BB_ELEVATION.** Flip
   restructure moved apogee 94 -> 157 km with nothing else altered.
3. **Identical results across a gain sweep means the gain is not reaching the
   actuator.** Cost several sweeps before it was recognised.
4. **grep before patching.** Several edits silently failed an assertion and
   were swept as if applied.

## Estimated coefficients still open

`cm_alpha` 0.35, `cm_q` 3.0, `cn_alpha` 1.8, fin `cn_delta` 1.2, RCS 7 kN,
Isp 350 s. The aero ones set descent attitude dynamics and are what grid fin
CFD would replace. `fin_drag_factor` is now calibrated rather than guessed --
the others are not.
