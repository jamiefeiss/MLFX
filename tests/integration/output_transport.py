import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('bec_transport.h5', 'r')

dset1 = f['1']
dset2 = f['4']
dset3 = f['5']

d1 = dset1['density']
x1 = dset1['x']

d2 = dset2['density2']
x2 = dset2['x']

overlap = float(dset3['overlap1'][...])

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(x1[...], d1[...], label='final state')  # Plot some data on the axes.
ax.plot(x2[...], d2[...], label='final ground state')  # Plot some data on the axes.
ax.set_xlabel('x')
ax.set_ylabel('density')
ax.set_title('Density plots, overlap = ' + str(round(overlap, 4)))
ax.legend()
fig.savefig('fig.png')