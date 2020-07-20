import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

f = h5py.File('shockwave_single.h5', 'r')

dset = f['4']
dset2 = f['2']

t_c = dset2['t_c'][...] # collision time
k = dset2['k'][...] # kick momentum
R = dset2['r'][...] # radius
Omega = dset2['omega'][...] # rotational velocity

t = dset['t'][...]
F_Q_1 = dset['F_Q_1'][...]
F_Q_2_re = dset['F_Q_2_re'][...]
F_Q_2_im = dset['F_Q_2_im'][...]
F_C = dset['F_C'][...]
# F_Q = dset['F_Q'][...]

F_Q_2 = F_Q_2_re + 1j * F_Q_2_im

F_Q = 4 * (F_Q_1 - np.abs(F_Q_2)**2)
# F_Q = 4 * (F_Q_1 - np.abs(np.conj(F_Q_2)*F_Q_2))

F_Q_func = 4 * t**2 * R**2 * k**2

F_S = (2 * math.pi * R**2)**2

fig1, ax1 = plt.subplots()
ax1.plot(t/t_c, F_C)
ax1.set_xlabel('t/t_c')
ax1.set_ylabel('F_C')
ax1.set_title('Classical Fisher information, Omega={}'.format(Omega))
fig1.savefig('F_C.png')

fig2, ax2 = plt.subplots()
ax2.plot(t/t_c, F_Q/F_S, label='Fq')
ax2.plot(t/t_c, F_Q_func/F_S, label='4t^2R^2k^2')
ax2.plot(t/t_c, F_C/F_S, label='Fc')
ax2.set_xlabel('t/t_c')
ax2.set_ylabel('F/F_S')
ax2.set_title('Quantum Fisher information, Omega={}'.format(Omega))
ax2.legend()
fig2.savefig('F_Q.png')