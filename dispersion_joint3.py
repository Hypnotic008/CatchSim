import numpy as np
from run_mission import build, fly
from quatsim.phases import catch_report

cases=[]
for dx in (-5,5):
    for dvx in (-0.5,0.5):
        cases.append((dx,dvx))

vehicle,aero,sep,bb0,land,seq=build()
rows=[]
for dx,dvx in cases:
    # Fresh build because sequencer state is intentionally one-shot.
    vehicle,aero,sep,bb0,land,seq=build()
    seq.terminal_dispersion={'dr':[dx,0,0],'dv':[dvx,0,0]}
    out=fly(9.1472,vehicle,aero,sep,bb0,land,seq,dt=0.02)
    rep=catch_report(out['state'][-1], np.array([0.,0.,105.]))
    rows.append((dx,dvx,rep['caught'],rep['checks']['lateral_error_m'][0],rep['checks']['horizontal_speed_ms'][0],rep['checks']['vertical_error_m'][0]))

print('dx(m) dvx(m/s) caught lateral(m) hspeed(m/s) zerr(m)')
for r in rows: print(f'{r[0]:6.1f} {r[1]:9.2f} {str(r[2]):>6} {r[3]:10.3f} {r[4]:12.3f} {r[5]:8.3f}')
print('caught',sum(r[2] for r in rows),'/',len(rows))
