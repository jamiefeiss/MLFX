import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

f = h5py.File('shockwave_single.h5', 'r')

dset1 = f['1']
dset2 = f['2']

P = dset1['density'][...]
P_diff = dset1['density_diff'][...]
x = dset1['x'][...]
t = dset1['t'][...]

t_c = dset2['t_c'][...] # collision time
Omega = dset2['omega'][...] # rotational velocity

fig1, ax1 = plt.subplots()
c1 = ax1.contourf(x/math.pi, t/t_c, P, cmap='Reds')
ax1.set_xlabel('x/pi')
ax1.set_ylabel('t/t_c')
ax1.set_title('Density, Omega={}'.format(Omega))
fig1.colorbar(c1)
fig1.savefig('density_contours.png')

fig2, ax2 = plt.subplots()
c2 = ax2.contourf(x/math.pi, t/t_c, P_diff, cmap='bwr')
ax2.set_xlabel('x/pi')
ax2.set_ylabel('t/t_c')
ax2.set_title('Density derivative, Omega={}'.format(Omega))
fig2.colorbar(c2)
fig2.savefig('density_diff_contours.png')