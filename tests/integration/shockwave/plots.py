import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

f = h5py.File('shockwave.h5', 'r')

dset1 = f['1']

P = dset1['density'][...]
x = dset1['x'][...]
t = dset1['t'][...]

fig1, ax1 = plt.subplots()
c1 = ax1.contourf(x/math.pi, t, P, cmap='Reds')
ax1.set_xlabel('x/pi')
ax1.set_ylabel('t')
ax1.set_title('Density')
fig1.colorbar(c1)
fig1.savefig('density.png')

def polar_heat(values, thetas=None, radii=None, ax=None, fraction=0.1,
               **kwargs):

    values = np.atleast_2d(values)
    if thetas is None:
        thetas = np.linspace(0, 2*np.pi, values.shape[1]).reshape(1, -1)
    if radii is None:
        radii = np.linspace(0, 1, values.shape[0] + 1).reshape(-1, 1)
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'polar':True})

    mesh = ax.pcolormesh(thetas, radii, values, **kwargs)

    radrange = radii.ptp()
    ax.set_rlim(radrange * (1 - 1. / fraction), radrange)
    ax.set_axis_off()
    fig.savefig('ring.png')

polar_heat(P[-1])