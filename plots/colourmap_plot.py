import h5py
import matplotlib.pyplot as plt

f = h5py.File('bec_transport_generated.h5', 'r')

dset1 = f['1'] # psi

d1 = dset1['density']
x1 = dset1['x']
t1 = dset1['t']

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.imshow(d1[...], extent=[x1[0], x1[-1], t1[0], t1[-1]], origin='lower')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('State density over time, x0=10, T=10, k=1')
fig.savefig('colourmap.png')
