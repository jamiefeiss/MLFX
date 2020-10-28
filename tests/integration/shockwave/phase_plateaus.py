import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

# multiple h5 files?
f = h5py.File('shockwave.h5', 'r')

dset2 = f['2'] # fourier
k_density_re = dset2['k_density_re'][...]
k_density_im = dset2['k_density_im'][...]
kx = dset2['kx'][...]
tk = dset2['t'][...]
k_density = k_density_re + 1j * k_density_im

dset4 = f['4'] # constants
R = dset4['radius'][...]
Omega = dset4['omega'][...]
delta = dset4['d_omega'][...]
L = dset4['length'][...]
N = dset4['no_atoms'][...]
phi = dset4['phi'][...]
g = dset4['non_lin'][...]
A_psi = dset4['amplitude_psi'][...]
w_psi = dset4['width_psi'][...]
A = dset4['amplitude'][...]
w = dset4['width'][...]
T_imag = dset4['t_imag'][...]
T_evo = dset4['t_evo'][...]

def find_zero_gradient(f_mag, t, k_index):
    f_2k = np.abs(f_mag[..., k_index])
    max_indices = []
    f_grad = np.gradient(f_2k)
    for i in range(1, f_grad.shape[0] - 1):
        if (abs(f_grad[i - 1]) > abs(f_grad[i]) < abs(f_grad[i + 1])) and (f_grad[i - 1] > 0 > f_grad[i + 1]):
            max_indices.append(i)
    return max_indices

# samples
imag_samples = 0
evo_samples = 1000
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples
t_evo_start = 0
t_evo_end = t_evo_start + evo_samples - 1

k_dom = 4
k_half = int(kx.shape[0]/2)
k_index = k_half + k_dom
max_indices = find_zero_gradient(k_density, tk, k_index)

t_max = []
f_max = []
f_plats = []
for index in max_indices:
    t_max.append(tk[index])
    f_max.append(np.abs(k_density[index, k_index]))
    f_plats.append(np.angle(k_density[index, k_index]))

start_index = 0
t_plat = []
plat_vals = []
for i in range(start_index, len(t_max)):
    t_plat.append(t_max[i])
    if f_plats[i] < 0:
        plat_vals.append(f_plats[i] + math.pi)
    else:
        plat_vals.append(f_plats[i])

# plot fourier mag peaks matching plateaus
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Fourier transform (2k={}, '.format(str(round(kx[k_index], 4))) + r'$\Omega$' + '={})'.format(Omega))
ax1.plot(tk - T_imag, np.abs(k_density[..., k_index]))
ax1.plot(t_max - T_imag, f_max, 'r.')
ax1.set_ylabel(r'$|F_{2k}|$')
ax2.plot(tk - T_imag, np.angle(k_density[..., k_index]), label='phase')
ax2.plot(tk - T_imag, np.angle(k_density[..., k_index]) + math.pi, label='phase + ' + r'$\pi$')
ax2.plot(t_plat - T_imag, plat_vals, 'r.')
ax2.set_ylabel(r'$arg(F_{2k})$')
ax2.set_ylim([-math.pi, math.pi])
ax2.legend(loc='lower right')
ax2.set_yticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi])
ax2.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
plt.xlabel('t')
fig.savefig('fourier_combined.png')

def phase_adjust(k):
    phase = np.angle(k_density[..., k_half + k])
    grad = np.gradient(phase)
    t_phase = []
    if not k % 2 == 0: # shift odd wavenumbers
        phase = phase - math.pi/2
    adjusted = []
    for i in range(len(phase)):
        if abs(grad[i]) < 0.005: # gradient threshold
            if phase[i] < 0:
                phase[i] += math.pi
            if 0.1 < phase[i] < 2:
                adjusted.append(phase[i])
                t_phase.append(tk[i] - T_imag)
    return t_phase, adjusted

# plot plateaus for different k
fig, ax = plt.subplots()
ax.plot(phase_adjust(2)[0], phase_adjust(2)[1], label='2k=2')
ax.plot(phase_adjust(3)[0], phase_adjust(3)[1], label='2k=3')
ax.plot(phase_adjust(4)[0], phase_adjust(4)[1], label='2k=4')
ax.plot(phase_adjust(5)[0], phase_adjust(5)[1], label='2k=5')
ax.plot(phase_adjust(6)[0], phase_adjust(6)[1], label='2k=6')
ax.set_xlabel('t')
ax.set_ylabel(r'$arg(F_{2k})$')
ax.set_ylim([0, math.pi])
ax.set_title('Fourier density phase (' + r'$\Omega$' + '={})'.format(Omega))
ax.legend()
plt.yticks([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi], ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
fig.savefig('fourier_phase.png')

fig, ax = plt.subplots()
ax.plot(tk - T_imag, np.abs(k_density[..., k_index]))
ax.plot(t_max - T_imag, f_max, 'r.')
ax.set_ylim([0, 1])
fig.savefig('fourier_peaks.png')