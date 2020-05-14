from mlfx import Project

p = Project()

config = {
    'name': 'bec_transport_generated',
    'author': 'Jamie Feiss',
    'description': 'Testing generating the BEC transport problem using the library',
    # 'arguments': [],
    'auto_vectorise': True,
    'benchmark': True,
    # 'bing': False,
    # 'cflags': False,
    # 'chunked_output': False,
    # 'diagnostics': True,
    # 'error_check': True,
    # 'halt_non_finite': True,
    'fftw': 'patient',
    # 'openmp': False,
    # 'precision': False,
    'validation': 'run-time',
    # 'driver': False,
    'prop_dim': 't',
    'trans_dim': [
        {
            'name': 'x',
            'lattice': '200',
            'domain': '(-5, 20)'
        }
    ]
}

p.add_global('const real T_i = 1e-1; // Imaginary time duration')
p.add_global('const real T = 10.0; // End time')
p.add_global('const real x_0 = 10.0; // Final position')

p.add_argument('k', 'real', '1.0')

p.config(config)

wavefunction = p.vec(type = 'complex', dimensions = 'x')
wavefunction.comment('Wavefunction')
init = 'psi = (1.0 / pow(M_PI, 0.25)) * exp(-pow(x, 2) / 2.0); // ground state of HO'
wavefunction.add_eq(init)

wavefunction_final = p.vec(type = 'complex', dimensions = 'x')
wavefunction_final.comment('Comparison wavefunction')
init = 'psi2 = (1.0 / pow(M_PI, 0.25)) * exp(-pow(x - x_0, 2) / 2.0); // ground state of HO'
wavefunction_final.add_eq(init)

timing_function = p.comp_vec()
timing_function.type = 'real'
timing_function.dimensions = ''
timing_function.comment('Timing function (sigmoid)')
# timing_function.add_eq('lambda = (t - T_i) / T;')
# timing_function.add_eq('lambda = 1 - (1 / (1 + (1/pow((1 / ((t - T_i) / T)) - 1, k))));')
timing_function.add_eq('lambda = 0; if (t <= T_i) {lambda = 0;} else if (t >= (T + T_i)) {lambda = 1;} else {lambda = 1.0 - (1.0 / (1.0 + (1.0/pow(((1.0 / ((t - T_i) / T)) - 1.0), k))));}')

potential = p.vec()
potential.type = 'real'
potential.dimensions = 'x'
potential.comment('Initial harmonic potential')
potential.add_eq('V = pow(x, 2) / 2.0;')

potential2 = p.vec()
potential2.type = 'real'
potential2.dimensions = 'x'
potential2.comment('Potential at final position')
potential2.add_eq('V2 = pow(x - x_0, 2) / 2.0;')

moving_potential = p.comp_vec()
moving_potential.type = 'real'
moving_potential.dimensions = 'x'
moving_potential.comment('Moving harmonic potential')
moving_potential.add_eq('Vt = pow(x - lambda * x_0, 2) / 2.0;')

seq = p.sequence()

imag_time = p.integrate('RK4', 'T_i', '10000', samples = '0 0 0 0 0')
op1 = p.operator(imag_time._head, 'ip', 'real', 'yes')
op1.add_eq('Ltt = -pow(kx, 2) / 2.0;')
imag_time.add_operator(op1)
imag_time.add_eq('dpsi_dt = Ltt[psi] - (V + mod2(psi)) * psi;')
imag_time.comment('imaginary time to find ground state')

gpe = p.integrate('ARK45', 'T', tolerance = '1e-8', samples = '1 50 50 0 0')
op2 = p.operator(gpe._head, 'ip', 'imaginary', 'yes')
op2.add_eq('Ltt = -i * pow(kx, 2) / 2.0;')
gpe.add_operator(op2)
gpe.add_eq('dpsi_dt = Ltt[psi] - i * (Vt + mod2(psi)) * psi;')
gpe.comment('gpe')

imag_time2 = p.integrate('RK4', 'T_i', '10000', samples = '0 0 0 1 1')
op1 = p.operator(imag_time2._head, 'ip', 'real', 'yes')
op1.add_eq('Ltt = -pow(kx, 2) / 2.0;')
imag_time2.add_operator(op1)
imag_time2.add_eq('dpsi2_dt = Ltt[psi2] - (V2 + mod2(psi2)) * psi2;')
imag_time2.comment('Ground state at final position')

o = p.output()

s1 = p.sampling_group('x', 'no')
s1.add_eq('psi_real = psi.Re();')
s1.add_eq('psi_imag = psi.Im();')
s1.add_eq('density = mod2(psi);')
s1.comment('state')

s2 = p.sampling_group('x', 'no')
s2.add_eq('p = Vt;')
s2.comment('potential')

s3 = p.sampling_group('x(1)', 'yes')
s3.add_eq('l = lambda;')
s3.comment('timing function')

s4 = p.sampling_group('x', 'no')
s4.add_eq('density2 = mod2(psi2);')
s4.comment('ground state at final position')

s5 = p.sampling_group('x(0)', 'no')
s5.add_eq('overlap1 = abs(psi)*abs(psi2);')
s5.add_eq('overlap2 = mod2(conj(psi2)*psi);')
# s5.add_eq('fidelity = 1 - mod2(conj(psi2)*psi); // fidelity')
s5.comment('overlap of final state')

p.generate('filename')

p.run('filename', config['name'], 'fig')