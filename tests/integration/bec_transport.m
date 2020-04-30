if (exist('OCTAVE_VERSION', 'builtin')) % Octave
  load bec_transport.h5
  t_1 = eval('_1.t');
  x_1 = eval('_1.x');
  mean_psi_real_1 = eval('_1.mean_psi_real');
  mean_psi_real_1 = permute(mean_psi_real_1, ndims(mean_psi_real_1):-1:1);
  mean_psi_imag_1 = eval('_1.mean_psi_imag');
  mean_psi_imag_1 = permute(mean_psi_imag_1, ndims(mean_psi_imag_1):-1:1);
  mean_density_1 = eval('_1.mean_density');
  mean_density_1 = permute(mean_density_1, ndims(mean_density_1):-1:1);
  stderr_psi_real_1 = eval('_1.stderr_psi_real');
  stderr_psi_real_1 = permute(stderr_psi_real_1, ndims(stderr_psi_real_1):-1:1);
  stderr_psi_imag_1 = eval('_1.stderr_psi_imag');
  stderr_psi_imag_1 = permute(stderr_psi_imag_1, ndims(stderr_psi_imag_1):-1:1);
  stderr_density_1 = eval('_1.stderr_density');
  stderr_density_1 = permute(stderr_density_1, ndims(stderr_density_1):-1:1);
  clear _1;
else % MATLAB
  t_1 = hdf5read('bec_transport.h5', '/1/t');
  x_1 = hdf5read('bec_transport.h5', '/1/x');
  mean_psi_real_1 = hdf5read('bec_transport.h5', '/1/mean_psi_real');
  mean_psi_real_1 = permute(mean_psi_real_1, ndims(mean_psi_real_1):-1:1);
  mean_psi_imag_1 = hdf5read('bec_transport.h5', '/1/mean_psi_imag');
  mean_psi_imag_1 = permute(mean_psi_imag_1, ndims(mean_psi_imag_1):-1:1);
  mean_density_1 = hdf5read('bec_transport.h5', '/1/mean_density');
  mean_density_1 = permute(mean_density_1, ndims(mean_density_1):-1:1);
  stderr_psi_real_1 = hdf5read('bec_transport.h5', '/1/stderr_psi_real');
  stderr_psi_real_1 = permute(stderr_psi_real_1, ndims(stderr_psi_real_1):-1:1);
  stderr_psi_imag_1 = hdf5read('bec_transport.h5', '/1/stderr_psi_imag');
  stderr_psi_imag_1 = permute(stderr_psi_imag_1, ndims(stderr_psi_imag_1):-1:1);
  stderr_density_1 = hdf5read('bec_transport.h5', '/1/stderr_density');
  stderr_density_1 = permute(stderr_density_1, ndims(stderr_density_1):-1:1);
end
