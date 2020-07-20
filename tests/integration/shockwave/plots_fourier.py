import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

f = h5py.File('shockwave_single.h5', 'r')

dset2 = f['2']
dset3 = f['3']

t_c = dset2['t_c'][...] # collision time
k_2 = dset2['k_2'][...] # 2k
phi = dset2['phi'][...] # Sagnac phase-shift
Omega = dset2['omega'][...] # rotational velocity
# print('Sagnac phase-shift = {}'.format(phi))

k_density_re = dset3['k_density_re'][...]
k_density_im = dset3['k_density_im'][...]
kx = dset3['kx'][...]
t = dset3['t'][...]

phi_vec = np.zeros(t.shape)

for i in range(phi_vec.shape[0]):
    phi_vec[i] = phi

k_density = k_density_re + 1j * k_density_im

fig1, ax1 = plt.subplots()
c1 = ax1.contourf(kx, t/t_c, np.abs(k_density), cmap='Reds')
ax1.set_xlabel('kx')
ax1.set_ylabel('t/t_c')
ax1.set_title('Fourier density magnitude, Omega={}'.format(Omega))
fig1.colorbar(c1)
fig1.savefig('fourier_density_contours.png')

fig2, ax2 = plt.subplots()
c2 = ax2.contourf(kx, t/t_c, np.angle(k_density), cmap='bwr')
ax2.set_xlabel('kx')
ax2.set_ylabel('t/t_c')
ax2.set_title('Fourier density phase, Omega={}'.format(Omega))
fig2.colorbar(c2)
fig2.savefig('fourier_density_phase_contours.png')

k_i = 0

for i in range(len(kx)):
    if kx[i] == k_2:
        k_i = i

F_mag = np.abs(k_density[..., k_i])
F_arg = np.angle(k_density[..., k_i])

# print(max(F_arg))

fig3, ax3 = plt.subplots()
ax3.plot(t/t_c, F_mag)
ax3.set_xlabel('t/t_c')
ax3.set_ylabel('F_mag')
ax3.set_title('Fourier density magnitude at 2k={}, Omega={}'.format(int(k_2), Omega))
fig3.savefig('fourier_mag_2k.png')

fig4, ax4 = plt.subplots()
ax4.plot(t/t_c, F_arg, label='phase')
ax4.plot(t/t_c, phi_vec, label='Sagnac, n=1')
ax4.set_xlabel('t/t_c')
ax4.set_ylabel('F_arg')
ax4.set_title('Fourier density phase at 2k={}, Omega={}'.format(int(k_2), Omega))
ax4.legend()
fig4.savefig('fourier_arg_2k.png')
