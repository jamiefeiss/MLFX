from mlfx import Project

p = Project()

config = {
    'xmds_settings': {
        'name': 'ml_transport_unopt',
        'author': 'Jamie Feiss',
        'description': 'Testing machine learning optimisation using the BEC transport problem',
        'auto_vectorise': True,
        'benchmark': True,
        'fftw': 'patient',
        'validation': 'run-time',
        'prop_dim': 't',
        'trans_dim': [
            {
                'name': 'x',
                'lattice': '200',
                'domain': '(-5, 20)'
            }
        ]
    },
    # 'ml_settings': {
    #     'train_learning_rate': 0.01,
    #     'train_learning_decay': False,
    #     'opt_learning_rate': 0.01,
    #     'opt_learning_decay': True,
    #     # 'refine_learning_rate': 0.01,
    #     'training_size': 50,
    #     'neurons': (16, 8),
    #     'train_epochs': 200,
    #     'opt_epochs': 200,
    #     # 'refine_epochs': 20,
    #     'early_stop': False,
    #     'early_stop_patience': 10,
    #     'early_stop_delta': 0,
    #     'validation_split': 0.2
    # }
}

# p.add_global('real', 'N', 10)
# p.add_global('real', 'g', 1.0)
p.add_global('real', 'T_i', 1e-1)
p.add_global('real', 'T', 10.0)
p.add_global('real', 'x_0', 10.0)

p.add_global('real', 'k', 1.0)
# p.parameter('real', 'k', default_value=1.5, min=1.0, max=2.0)

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

imag_time = p.integrate('RK4', 'T_i', '10000', samples = '0 0')
op1 = p.operator(imag_time._head, 'ip', 'real', 'yes')
op1.add_eq('Ltt = -pow(kx, 2) / 2.0;')
imag_time.add_operator(op1)
imag_time.add_eq('dpsi_dt = Ltt[psi] - (V + mod2(psi)) * psi;')
imag_time.comment('imaginary time to find ground state')

gpe = p.integrate('ARK45', 'T', tolerance = '1e-8', samples = '0 100')
op2 = p.operator(gpe._head, 'ip', 'imaginary', 'yes')
op2.add_eq('Ltt = -i * pow(kx, 2) / 2.0;')
gpe.add_operator(op2)
gpe.add_eq('dpsi_dt = Ltt[psi] - i * (Vt + mod2(psi)) * psi;')
gpe.comment('gpe')

imag_time2 = p.integrate('RK4', 'T_i', '10000', samples = '1 0')
op1 = p.operator(imag_time2._head, 'ip', 'real', 'yes')
op1.add_eq('Ltt = -pow(kx, 2) / 2.0;')
imag_time2.add_operator(op1)
imag_time2.add_eq('dpsi2_dt = Ltt[psi2] - (V2 + mod2(psi2)) * psi2;')
imag_time2.comment('Ground state at final position')

o = p.output()

s1 = p.sampling_group('x(0)', 'no')
s1.add_eq('overlap = abs(psi)*abs(psi2);')
s1.comment('overlap of final state')

s2 = p.sampling_group(basis='x', initial_sample='no')
s2.add_eq('density = mod2(psi);')
s2.comment('density')

p.cost_variable('overlap')

# user-defined cost function from xmds output variable
def cost(f):
    dataset = f['1']
    overlap = dataset['overlap'][...]
    f.close()
    return -overlap

p.cost_fn(cost)

p.generate('xmds_ml_transport')

# p.optimise()