import numpy as np, multiprocessing as mp, sys
sys.path.insert(0,'.')
from run_mission import build, fly, TARGET
from quatsim.phases import catch_report

def one(c):
 dx,dvx=c; v,a,s,b,l,q=build(); q.terminal_dispersion={'dr':[dx,0,0],'dv':[dvx,0,0]}
 o=fly(9.1472,v,a,s,b,l,q,dt=.15); rep=catch_report(o['state'][-1],TARGET)
 return dx,dvx,rep['caught'],rep['checks']['lateral_error_m'][0],rep['checks']['horizontal_speed_ms'][0]
if __name__=='__main__':
 cases=[(x,v) for x in (-5,0,5) for v in (-.5,0,.5)]
 with mp.Pool(3) as p: rows=p.map(one,cases)
 print('dx dvx caught lateral hspeed')
 for r in sorted(rows): print(f'{r[0]:4.1f} {r[1]:5.1f} {str(r[2]):>6} {r[3]:7.3f} {r[4]:7.3f}')
 print('caught',sum(r[2] for r in rows),'/',len(rows))
