import h5py
import matplotlib.pyplot as plt

f1 = h5py.File('ml_transport_unopt.h5', 'r')
dset1 = f1['2']
d1 = dset1['density']
x1 = dset1['x']
t1 = dset1['t']
overlap1 = f1['1']['overlap'][...]
print(overlap1)

f2 = h5py.File('ml_transport.h5', 'r')
dset2 = f2['2']
d2 = dset2['density']
x2 = dset2['x']
t2 = dset2['t']
overlap2 = f2['1']['overlap'][...]
print(overlap2)

fig, axes = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle('BEC transport optimisation')
axes[0].imshow(d1[...], extent=[x1[0], x1[-1], t1[0], t1[-1]], origin='lower') # unoptimised
text1 = axes[0].text(15, 8, 'k=1.000', color='white', bbox=dict(facecolor='black', alpha=0.5))
axes[1].imshow(d2[...], extent=[x2[0], x2[-1], t2[0], t2[-1]], origin='lower') # optimised
text2 = axes[1].text(15, 8, 'k=1.596', color='white', bbox=dict(facecolor='black', alpha=0.5))
# for ax in axes:
#     ax.legend(handlelength=0)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('x')
plt.ylabel('t')
fig.savefig('colourmap.png')