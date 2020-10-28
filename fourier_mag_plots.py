import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

f_1 = h5py.File('shockwave_unopt.h5', 'r')
dset2_1 = f_1['2']
dset4_1 = f_1['4']

f_2 = h5py.File('shockwave.h5', 'r')
dset2_2 = f_2['2']
dset4_2 = f_2['4']

# constants
R = dset4_1['radius'][...]
Omega = dset4_1['omega'][...]
A_psi = dset4_1['amplitude_psi'][...]
w_psi = dset4_1['width_psi'][...]
A_1 = dset4_1['amplitude'][...]
w_1 = dset4_1['width'][...]
T_imag = dset4_1['t_imag'][...]
T_evo = dset4_1['t_evo'][...]

A_2 = dset4_2['amplitude'][...]
w_2 = dset4_2['width'][...]

# fourier transform
k_density_re_1 = dset2_1['k_density_re'][...]
k_density_im_1 = dset2_1['k_density_im'][...]
kx_1 = dset2_1['kx'][...]
tk_1 = dset2_1['t'][...] - T_imag
k_density_1 = k_density_re_1 + 1j * k_density_im_1

k_density_re_2 = dset2_2['k_density_re'][...]
k_density_im_2 = dset2_2['k_density_im'][...]
kx_2 = dset2_2['kx'][...]
tk_2 = dset2_2['t'][...] - T_imag
k_density_2 = k_density_re_2 + 1j * k_density_im_2

k_dom = 4
k_half = int(kx_1.shape[0]/2)
k_index = k_half + k_dom

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
fig.set_size_inches(6.3, 3.5)
plt.tick_params(labelsize=14)

ax.plot(tk_1, np.abs(k_density_1[..., k_index]), 'r--', label='Unoptimised')
ax.plot(tk_2, np.abs(k_density_2[..., k_index]), 'k-', label='Optimised')
ax.legend(fontsize=14)
ax.set_xlabel('t', fontsize=18)
ax.set_ylabel(r'$|F_{2k}|$', fontsize=18)
ax.set_ylim([0, 0.6])
ax.set_title('Fourier magnitude optimisation', fontsize=18)

fig.savefig('fourier_kslice_resize.png')

print('unopt, A={}, w={}'.format(A_1, w_1))
print('opt, A={}, w={}'.format(A_2, w_2))

# time = 159
# t_index = int(round((time/T_evo)*1000))
# fig, ax = plt.subplots()
# ax.plot(kx_1, np.abs(k_density_1[t_index, ...]), label='before')
# ax.plot(kx_2, np.abs(k_density_2[t_index, ...]), label='after')
# ax.legend()
# ax.set_xlim([-2, 2])
# ax.set_xlabel('k')
# ax.set_ylabel(r'$|F_{2k}|$')
# ax.set_title('Fourier magnitude optimisation (R={}, '.format(R) + r'$\Omega$' + '={}. t={})'.format(Omega, str(round(tk_1[t_index], 4))))
# fig.savefig('fourier_tslice.png')

# linear fit of plats
def find_zero_gradient(f_mag, t, k_index):
    f_2k = np.abs(f_mag[..., k_index])
    max_indices = []
    f_grad = np.gradient(f_2k)
    for i in range(1, f_grad.shape[0] - 1):
        if (abs(f_grad[i - 1]) > abs(f_grad[i]) < abs(f_grad[i + 1])) and (f_grad[i - 1] > 0 > f_grad[i + 1]):
            max_indices.append(i)
    return max_indices

max_indices_1 = find_zero_gradient(k_density_1, tk_1, k_index)
t_max_1 = []
f_max_1 = []
f_plats_1 = []
for index in max_indices_1:
    t_max_1.append(tk_1[index])
    f_max_1.append(np.abs(k_density_1[index, k_index]))
    f_plats_1.append(np.angle(k_density_1[index, k_index]))

max_indices_2 = find_zero_gradient(k_density_2, tk_2, k_index)
t_max_2 = []
f_max_2 = []
f_plats_2 = []
for index in max_indices_2:
    t_max_2.append(tk_2[index])
    f_max_2.append(np.abs(k_density_2[index, k_index]))
    f_plats_2.append(np.angle(k_density_2[index, k_index]))

start_index = 0
t_plat_1 = []
plat_vals_1 = []
for i in range(start_index, len(t_max_1)):
    t_plat_1.append(t_max_1[i])
    if f_plats_1[i] < 0:
        plat_vals_1.append(f_plats_1[i] + math.pi)
    else:
        plat_vals_1.append(f_plats_1[i])

t_plat_2 = []
plat_vals_2 = []
for i in range(start_index, len(t_max_2)):
    t_plat_2.append(t_max_2[i])
    if f_plats_2[i] < 0:
        plat_vals_2.append(f_plats_2[i] + math.pi)
    else:
        plat_vals_2.append(f_plats_2[i])

# manual linear regression
def lin_reg(x, y):
    n = np.size(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SS_xx = sum((x - x_mean)**2)
    SS_yy = sum((y - y_mean)**2)
    SS_xy = 0
    for i in range(n):
        SS_xy += (x[i] - x_mean) * (y[i] - y_mean)

    m = SS_xy / SS_xx # slope
    c = y_mean - m * x_mean # intercept

    def y_pred(x_i): # predicting using fit
        return m * x_i + c
    
    SS_E = 0
    for i in range(n):
        SS_E += (y[i] - y_pred(x[i]))**2

    # slope uncertainty
    m_uncert = math.sqrt(SS_E / ((n - 2) * (SS_xx)))
    return m, m_uncert

# linear regression
x_1 = np.array(t_plat_1).reshape((-1, 1))
y_1 = np.array(plat_vals_1)
model_1 = LinearRegression().fit(x_1, y_1)
y_pred_1 = model_1.predict(x_1)

x_2 = np.array(t_plat_2).reshape((-1, 1))
y_2 = np.array(plat_vals_2)

# remove last element from fit
x_2 = x_2[:-1]
y_2 = y_2[:-1]

model_2 = LinearRegression().fit(x_2, y_2)
y_pred_2 = model_2.predict(x_2)

fig, ax = plt.subplots()
ax.plot(t_plat_1, plat_vals_1, 'bo', label='peaks 1')
ax.plot(x_1, y_pred_1, 'b-', label='fit 1')
ax.plot(t_plat_2, plat_vals_2, 'ro', label='peaks 2')
ax.plot(x_2, y_pred_2, 'r-', label='fit 2')
ax.set_xlabel('t')
ax.set_ylabel('arg(' + r'$F_{2k}$' + ')')
ax.set_ylim([0, math.pi])
ax.set_title('Linear fit')
ax.legend(loc='lower right')
fig.savefig('phase_fit.png')

m_1, s_m_1 = lin_reg(x_1, y_1)
m_2, s_m_2 = lin_reg(x_2, y_2)

print('unopt m/k={}, {}'.format(m_1[0]/(k_dom/R), s_m_1/(k_dom/R)))
print('opt m/k={}, {}'.format(m_2[0]/(k_dom/R), s_m_2/(k_dom/R)))

peak_avg_1 = (f_max_1[0] + f_max_1[1])/2
peak_avg_2 = (f_max_2[0] + f_max_2[1])/2

print('unopt avg={}'.format(peak_avg_1))
print('opt avg={}'.format(peak_avg_2))