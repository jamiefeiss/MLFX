import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

f = h5py.File('shockwave_single.h5', 'r')

dset1 = f['1']
dset2 = f['2']
dset3 = f['3']

P = dset1['density'][...]
P_diff = dset1['density_diff'][...]
x = dset1['x'][...]
t = dset1['t'][...]

t_c = dset2['t_c'][...] # collision time
phi = dset2['phi'][...] # Sagnac phase-shift

F_Q_1_re = dset3['F_Q_1_re'][...]
F_Q_1_im = dset3['F_Q_1_im'][...]

F_Q_1 = F_Q_1_re + 1j * F_Q_1_im

F_Q_2_re = dset3['F_Q_2_re'][...]
F_Q_2_im = dset3['F_Q_2_im'][...]

F_Q_2 = F_Q_2_re + 1j * F_Q_2_im

F_Q = 4 * (np.abs(F_Q_1) - np.abs(F_Q_2)**2)

F_C = dset3['F_C'][...]

print('phi = {}'.format(phi))

# contour plot of density over x & t
fig1, ax1 = plt.subplots()
ax1.contourf(x/math.pi, t/t_c, P)
ax1.set_xlabel('x/pi')
ax1.set_ylabel('t/t_c')
ax1.set_title('Density')
fig1.savefig('density_contours.png')

p_x = P[0, ...]
p_x_c = P[round(len(P[..., 0])/2), ...]
p_t = P[..., round(len(P[0, ...])/2)]

# plot over time at x=0
fig2, ax2 = plt.subplots()
ax2.plot(t/t_c, p_t)
ax2.set_xlabel('t/t_c')
ax2.set_ylabel('P(t, 0)')
ax2.set_title('Density vs. time (x=0)')
fig2.savefig('density_time.png')

# plot over space at t=0
fig3, ax3 = plt.subplots()
ax3.plot(x/math.pi, p_x)
ax3.set_xlabel('x/pi')
ax3.set_ylabel('P(0, x)')
ax3.set_title('Density vs. space (t=0)')
fig3.savefig('density_space.png')

# fourier transform of density
f_x = np.fft.fft(p_x) # t = 0
f_x = np.fft.fftshift(f_x)
f_x_c = np.fft.fft(p_x_c) # t = t_c
f_x_c = np.fft.fftshift(f_x_c)
k_x = np.fft.fftfreq(x.shape[0])
k_x = np.fft.fftshift(k_x)

# magnitude of fourier transform @ t=0 & t=t_c 
fig4, ax4 = plt.subplots()
ax4.plot(k_x, np.abs(f_x), label='t=0')
ax4.plot(k_x, np.abs(f_x_c), label='t=t_c')
ax4.set_xlabel('k')
ax4.set_ylabel('F')
ax4.legend()
ax4.set_title('Fourier transform of density vs. space (t=0)')
fig4.savefig('fourier_space.png')

# argument of fourier transform @ t=0 & t=t_c 
fig5, ax5 = plt.subplots()
ax5.plot(k_x, np.angle(f_x), label='t=0')
ax5.plot(k_x, np.angle(f_x_c), label='t=t_c')
ax5.set_xlabel('k')
ax5.set_ylabel('F')
ax5.legend()
ax5.set_title('Fourier transform of density vs. space (t=0)')
fig5.savefig('fourier_space_phase.png')

def fourier_transform(f, x):
    # performs fourier transform at each time step
    # f[t, x]
    F = np.zeros(f.shape, dtype=np.complex128)
    for i in range(f.shape[0]):
        F[i, ...] = (np.fft.fft(f[i, ...]))
    kx = np.fft.fftfreq(x.shape[0])
    F = np.fft.fftshift(F)
    kx = np.fft.fftshift(kx)
    return F, kx

def dominant_k(F, kx):
    # get highest magnitude at t=0
    F_0 = F[0, ...]
    max_F = 0
    max_k = 0
    for i in range(len(F_0)):
        if F_0[i] > max_F:
            max_F = np.abs(F_0[i])
            max_k = i
    return max_F, max_k, kx[max_k]

F, kx = fourier_transform(P, x)

dom_k = dominant_k(F, kx)[1]

F_t = F[..., dom_k]

fig6, ax6 = plt.subplots()
ax6.plot(t/t_c, np.abs(F_t))
ax6.set_xlabel('t/t_c')
ax6.set_ylabel('F_k')
ax6.set_title('Magnitude')
fig6.savefig('fourier_time_dom_k.png')

fig7, ax7 = plt.subplots()
ax7.plot(t/t_c, np.angle(F_t))
ax7.set_xlabel('t/t_c')
ax7.set_ylabel('F_k')
ax7.set_title('Phase')
fig7.savefig('fourier_time_phase.png')

fig8, ax8 = plt.subplots()
ax8.contourf(kx, t/t_c, np.abs(F))
ax8.set_xlabel('k')
ax8.set_ylabel('t/t_c')
ax8.set_title('Magnitude of Fourier transform')
fig8.savefig('fourier_mag_contours.png')

fig9, ax9 = plt.subplots()
ax9.contourf(kx, t/t_c, np.angle(F))
ax9.set_xlabel('k')
ax9.set_ylabel('t/t_c')
ax9.set_title('Phase of Fourier transform')
fig9.savefig('fourier_phase_contours.png')

fig10, ax10 = plt.subplots()
ax10.plot(t/t_c, F_Q, label='F_Q')
ax10.plot(t/t_c, F_C, label='F_C')
ax10.set_xlabel('t/t_c')
ax10.set_ylabel('F')
ax10.set_title('Fisher information')
fig10.savefig('fisher.png')