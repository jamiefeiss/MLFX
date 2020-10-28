import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def principal_n(phase):
    arg = phase
    n = 0
    while arg <= -math.pi or math.pi < arg:
        arg = arg - math.pi
        n += 1
    return n

f = h5py.File('shockwave.h5', 'r')

dset1 = f['1']
dset2 = f['2']
dset3 = f['3']
dset4 = f['4']
dset5 = f['5']

# density
P = dset1['density'][...]
psi_re = dset1['psi_re']
psi_im = dset1['psi_im']
x = dset1['x'][...]
t = dset1['t'][...]

# print(P[0, 0])
# print(P[0, -1])

# fourier transform
k_density_re = dset2['k_density_re'][...]
k_density_im = dset2['k_density_im'][...]
kx = dset2['kx'][...]
tk = dset2['t'][...]

# constants
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

# integration (Fisher info)
tf = dset3['t'][...]
F_Q_1 = dset3['F_Q_1'][...]
F_Q_2_re = dset3['F_Q_2_re'][...]
F_Q_2_im = dset3['F_Q_2_im'][...]
F_C = dset3['F_C'][...]

F_Q_2 = F_Q_2_re + 1j * F_Q_2_im

F_Q = N * 4 * (F_Q_1 - np.abs(F_Q_2)**2)
# F_Q_func = 4 * t**2 * R**2 * k**2
F_S = (2 * math.pi * R**2)**2

# functions
psi_init = dset5['psi_init'][...]
laser = dset5['laser'][...]
x_3 = dset5['x'][...]

# number of samples per integration block
imag_samples = 0
evo_samples = 1000

# time indicies per integration block
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples

# t_evo_start = t_imag_end + 1
t_evo_start = 0
t_evo_end = t_evo_start + evo_samples - 1

# # density contour plot
# fig1, ax1 = plt.subplots()
# c1 = ax1.contourf(x/(L/2), t, P, cmap='Reds')
# ax1.set_xlabel(r'$\frac{x}{\pi}$')
# ax1.set_ylabel('t')
# ax1.set_title('Density, R={}'.format(R))
# fig1.colorbar(c1)
# fig1.savefig('density.png')

def plot(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([-1, 1])
    ax.set_title(title)
    fig.savefig(filename + '.png')

def density_time_slice(time_index, title, filename):
    fig, ax = plt.subplots()
    ax.plot(x/(L/2), P[time_index])
    # ax.plot(x/(L/2), psi_re[time_index])
    # ax.plot(x/(L/2), psi_im[time_index])
    # ax.legend(['real', 'imag'])
    ax.set_xlabel(r'$\frac{x}{\pi}$')
    ax.set_ylabel(r'$|\psi|^2$')
    ax.set_ylim([0, 1])
    ax.set_title('Density at {} (t={})'.format(title, str(round(t[time_index], 4))))
    fig.savefig(filename + '.png')

def polar_heat(time_index, title, filename):
    values = np.atleast_2d(P[time_index])
    thetas = np.linspace(-np.pi, np.pi, values.shape[1]).reshape(1, -1)
    radii = np.linspace(0, 1, values.shape[0] + 1).reshape(-1, 1)
    fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
    mesh = ax.pcolormesh(thetas, radii, values)

    radrange = radii.ptp()
    ring_width = 0.1
    ax.set_rlim(radrange * (1 - 1. / ring_width), radrange)
    ax.set_theta_offset(np.pi/2)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['0', r'$\frac{\pi R}{2}$', r'$\pi R$', r'$\frac{3\pi R}{2}$'])
    ax.set_yticklabels([])
    ax.set_title('Density at {} (t={})'.format(title, str(round(t[time_index], 4))))
    fig.savefig(filename + '.png')

def find_maximums(f_mag, t, k_index):
    f_2k = np.abs(f_mag[..., k_index])
    max_indices = []
    for i in range(t.shape[0] - 1):
        if f_2k[i-1] < f_2k[i] > f_2k[i+1]:
            max_indices.append(i)
    return max_indices

# def peak_segments(f_mag, t, k_index):
#     f_2k = np.abs(f_mag[..., k_index])
#     f_grad = np.gradient(f_2k)
#     segment_end_indices = []
#     for i in range(f_grad.shape[0] - 1):
#         if f_grad[i] < 0 and f_grad[i + 1] > 0:
#             # print('F={}, t={}'.format(f_grad[i], t[i]))
#             segment_end_indices.append(i)
#     f_seg = np.split(f_2k, segment_end_indices)
#     # t_seg = np.split(t, segment_end_indices)
#     # segments = []
#     # for i in range(len(f_seg)):
#     #     tup = t_seg[i], f_seg[i]
#     #     segments.append(tup)
#     f_max = []
#     for i in range(len(f_seg)):
#         max_index = np.argmax(f_seg[i])
#         if i > 0:
#             max_index += len(f_seg[i-1]) - 1
#         f_max.append(max_index)
#     print(f_max)
#     # return f_max
      

def find_zero_gradient(f_mag, t, k_index):
    # grad_threshold = 0.001
    f_2k = np.abs(f_mag[..., k_index])
    max_indices = []
    f_grad = np.gradient(f_2k)
    for i in range(1, f_grad.shape[0] - 1):
        if (abs(f_grad[i - 1]) > abs(f_grad[i]) < abs(f_grad[i + 1])) and (f_grad[i - 1] > 0 > f_grad[i + 1]):
            max_indices.append(i)
    # print(max_indices)
    return max_indices

# time slice plots
# density_time_slice(t_imag_start, 'initial', 'density_imag_start')
density_time_slice(t_imag_end, 'end of imaginary time', 'density_imag_end')
# density_time_slice(t_evo_end, 'end of free evolution', 'density_evo_end')

# # ring heatmaps
# polar_heat(t_imag_start, 'initial', 'ring_imag_start.png')
# polar_heat(t_imag_end, 'end of imaginary time', 'ring_imag_end.png')
# polar_heat(t_evo_end, 'end of free evolution', 'ring_evo_end.png')

# # plot functions
# plot(x_3/(L/2), psi_init, r'$\frac{x}{\pi}$', 'psi', 'initial wavefunction', 'psi')
# plot(x_3/(L/2), laser, r'$\frac{x}{\pi}$', 'V', 'Gaussian potential', 'gaussian')

# plot fourier transform
k_density = k_density_re + 1j * k_density_im

# density contours
fig0, ax0 = plt.subplots()
c0 = ax0.contourf(x, t, P, cmap='Reds')
# c0 = ax0.contourf(kx, tk, np.abs(k_density), cmap='Reds', levels=np.linspace(0, 0.8, 20), extend='max')
# ax0.set_xlim([-2, 2])
ax0.set_xlabel('x')
ax0.set_ylabel('t')
ax0.set_title('Density, Omega={}'.format(Omega))
fig0.colorbar(c0)
fig0.savefig('density_contours.png')

# fourier density magnitude contours
fig1, ax1 = plt.subplots()
# c1 = ax1.contourf(kx, tk, np.abs(k_density), cmap='Reds')
c1 = ax1.contourf(kx, tk, np.abs(k_density), cmap='Reds', levels=np.linspace(0, 1.8, 20), extend='max')
ax1.set_xlim([-2, 2])
ax1.set_xlabel('kx')
ax1.set_ylabel('t')
ax1.set_title('Fourier density magnitude, Omega={}'.format(Omega))
fig1.colorbar(c1)
fig1.savefig('fourier_density_contours.png')

# # fourier density phase contours
# fig2, ax2 = plt.subplots()
# c2 = ax2.contourf(kx, tk, np.angle(k_density), cmap='bwr')
# ax2.set_xlabel('kx')
# ax2.set_ylabel('t')
# ax2.set_xlim([-2, 2])
# ax2.set_title('Fourier density phase, Omega={}'.format(Omega))
# fig2.colorbar(c2)
# fig2.savefig('fourier_density_phase_contours.png')

# t_index = 600 # t=172

# # fourier density mag at time slice
# fig3, ax3 = plt.subplots()
# ax3.plot(kx, np.abs(k_density[t_index]))
# ax3.set_xlabel('kx')
# ax3.set_ylabel('|F|')
# # ax3.set_xlim([-2, 2])
# ax3.set_title('Fourier density at t={}'.format(str(round(tk[t_index], 4))))
# fig3.savefig('fourier_timeslice.png')

k_dom = 4
k_half = int(kx.shape[0]/2)
k_index = k_half + k_dom

# print(find_maximums(k_density, tk, k_index))

# print(find_zero_gradient(k_density, tk, k_index))

max_indices = find_zero_gradient(k_density, tk, k_index)

t_max = []
f_max = []
f_plats = []
for index in max_indices:
    t_max.append(tk[index])
    f_max.append(np.abs(k_density[index, k_index]))
    f_plats.append(np.angle(k_density[index, k_index]))

# fourier density mag at dominant k slice
fig4, ax4 = plt.subplots()
ax4.plot(tk, np.abs(k_density[..., k_index]))
ax4.plot(t_max, f_max, 'r.')
ax4.set_xlabel('t')
ax4.set_ylabel('|F|')
# ax4.set_xlim([-2, 2])
ax4.set_title('Fourier density at 2k={}'.format(str(round(kx[k_index], 4))))
fig4.savefig('fourier_kslice.png')

# fig10, ax10 = plt.subplots()
# ax10.plot(tk, np.gradient(np.abs(k_density[..., k_index])))
# ax10.set_xlabel('t')
# ax10.set_ylabel('dF')
# # ax10.set_xlim([-2, 2])
# ax10.set_title('Fourier density gradient at 2k={}'.format(str(round(kx[k_index], 4))))
# fig10.savefig('fourier_grad_kslice.png')

# fig11, ax11 = plt.subplots()
# for segment in peak_segments(k_density, tk, k_index):
#     ax11.plot(segment[0], segment[1])
# ax11.set_xlabel('t')
# ax11.set_ylabel('F')
# # ax11.set_xlim([-2, 2])
# ax11.set_title('Fourier density segments at 2k={}'.format(str(round(kx[k_index], 4))))
# fig11.savefig('fourier_segment_kslice.png')

# arg = np.angle(k_density[t_index, k_index])

# print('arg={}'.format(arg))
# print('sagnac={}'.format(phi))

# factor = (phi-arg)/math.pi

# print(factor)

# vector for plotting Sagnac phase shift
phi_vec = np.zeros(tk.shape)
# n = round(factor, 3)
# n = factor
# n = principal_n(phi)
# n = 0
for i in range(phi_vec.shape[0]):
    # phi_vec[i] = phi - math.pi * n
    phi_vec[i] = phi

# build arrays for linear regression
start_index = 0

t_plat = []
plat_vals = []
for i in range(start_index, len(t_max)):
    t_plat.append(t_max[i])
    if f_plats[i] < 0:
        plat_vals.append(f_plats[i] + math.pi)
    else:
        plat_vals.append(f_plats[i])

# find delta t
delta_t = []
for i in range(len(t_plat) - 1):
    dt = t_plat[i + 1] - t_plat[i]
    delta_t.append(dt)

round_factor = 9

# print('delta_t = {}'.format([round(value, round_factor) for value in delta_t]))
# print('delta_t average = {}'.format(round(sum(delta_t) / len(delta_t), round_factor)))

# linear regression
x = np.array(t_plat).reshape((-1, 1))
y = np.array(plat_vals)
model = LinearRegression().fit(x, y)
# print('y = {}x + {}, R^2={}'.format(round(model.coef_[0], round_factor), round(model.intercept_, round_factor), round(model.score(x, y), round_factor)))
y_pred = model.predict(x)

# fourier density phase at dominant k slice
fig5, ax5 = plt.subplots()
ax5.plot(tk, np.angle(k_density[..., k_index]), label='phase')
ax5.plot(tk, np.angle(k_density[..., k_index]) + math.pi, label='phase + pi')
# ax5.plot(t_max, f_plats, 'r.')
# ax5.plot(t_max, [plat + math.pi for plat in f_plats], 'k.')
ax5.plot(t_plat, plat_vals, 'r.')
ax5.plot(x, y_pred, 'k-')
# ax5.plot(tk, np.angle(k_density[..., k_index + 1]), label='phase 2')
# ax5.plot(tk, np.angle(k_density[..., k_index + 2]), label='phase 3')
# ax5.plot(tk, np.angle(k_density[..., k_index + 3]), label='phase 4')
# ax5.plot(tk, np.angle(k_density[..., k_index + 4]), label='phase 5')
# ax5.plot(tk, phi_vec, label='Sagnac')
ax5.set_xlabel('t')
ax5.set_ylabel('F_arg')
ax5.set_ylim([-math.pi, math.pi])
ax5.set_title('Fourier density phase at 2k={}, Omega={}'.format(str(round(kx[k_index], 4)), Omega))
ax5.legend()
fig5.savefig('fourier_phase_plateau.png')



# # fourier density phase at time slice
# fig6, ax6 = plt.subplots()
# ax6.plot(kx, np.angle(k_density[t_index, ...]))
# ax6.set_xlabel('kx')
# ax6.set_ylabel('F_arg')
# ax6.set_xlim([-1, 1])
# ax6.set_title('Fourier density phase at t={}, Omega={}'.format(str(round(tk[t_index], 4)), Omega))
# fig6.savefig('fourier_phase_timeslice.png')

# Fisher info
# classical
fig7, ax7 = plt.subplots()
ax7.plot(tf, F_C)
ax7.set_xlabel('t')
ax7.set_ylabel('F_C')
ax7.set_title('Classical Fisher information, Omega={}'.format(Omega))
fig7.savefig('F_C.png')

# quantum
fig8, ax8 = plt.subplots()
ax8.plot(tf, F_Q)
ax8.set_xlabel('t')
ax8.set_ylabel('F_Q')
ax8.set_title('Quantum Fisher information, Omega={}'.format(Omega))
fig8.savefig('F_Q.png')

# combined / Fs
fig9, ax9 = plt.subplots()
ax9.plot(tf, F_Q/F_S, label='F_Q')
ax9.plot(tf, F_C/F_S, label='F_C')
ax9.set_xlabel('t')
ax9.set_ylabel('F/F_S')
ax9.set_title('Fisher information, Omega={}'.format(Omega))
ax9.legend()
fig9.savefig('F_combined.png')

# print('t_1={}, p_1={}'.format(t_plat[0], plat_vals[0]))
# print('t_2={}, p_2={}'.format(t_plat[1], plat_vals[1]))

def grad(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)

print('R={}, Omega={}, 2k={}, t_imag={}, t_evo={}'.format(R, Omega, k_dom, T_imag, T_evo))
print('y = {}x + {}, R^2={}'.format(round(model.coef_[0], round_factor), round(model.intercept_, round_factor), round(model.score(x, y), round_factor)))
print('grad/k = {}'.format(grad(t_plat[0], plat_vals[0], t_plat[1], plat_vals[1])/(k_dom/R)))