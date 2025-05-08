# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:31:47 2022

@author: voumardt
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
from scipy.constants import c
from scipy.io import loadmat

import sys
sys.path.append(os.getcwd() + '../../../')

from src import  *
from src.pychi import light,materials,models,solvers

"""
User parameters
"""
### Simulation
t_pts = 2**17

### Light
pulse_duration = 80e-15
pulse_wavelength = 1.56e-6
pulse_energy = 1e-10

### Waveguide
wg_length = 0.0058
wg_chi_2 = 0
wg_chi_3 = 2.8e-21
wg_a_eff = 3e-13
wg_freq, wg_n_eff = np.load('n_eff_data_SiN.npy')
wg_atten=53

"""
Nonlinear propagation
"""
waveguide = materials.Waveguide(wg_freq, wg_n_eff, wg_chi_2, wg_chi_3,
                                wg_a_eff, wg_length, t_pts=t_pts,atten=wg_atten)
pulse = light.Sech(waveguide, pulse_duration, pulse_energy, pulse_wavelength)
model = models.SpmChi3(waveguide, pulse)
solver = solvers.Solver(model)
solver.solve()


"""
Plots
"""
pulse.plot_propagation()

# # Load experimental results
# exp_data = loadmat('exp_SiN.mat')
# exp_wl = exp_data['wavelength'][0]
# exp_int = exp_data['intensity'][0]
#
# # Compare experimental results and simulation
# plt.figure()
# plt.plot(pulse.wl, 10*np.log10(pulse.spectrum_wl[-1]/np.amax(pulse.spectrum_wl[-1]))-15)
# plt.plot(exp_wl, exp_int - np.amax(exp_int))
# plt.xlim(3.5e-7, 1.75e-6)
# plt.ylim(-75, 5)
# plt.title('Experimental vs simulation - SiN')
# plt.xlabel('Wavelength [m]')
# plt.ylabel('Intensity [dB]')
# plt.legend(('Simulation', 'Experiment'))
# plt.savefig('Experimental_vs_simulation_SiN.png')
waveguide.z = waveguide.length
beta = waveguide.k  # beta = neff * omega / c
omega = waveguide.omega
omega0 = waveguide.center_omega
from scipy.interpolate import interp1d

b_interp = interp1d(omega, beta, kind='cubic')
beta0 = b_interp(omega0)

# 一阶导数 β₁
d_beta = np.gradient(beta, omega)
d_beta_interp = interp1d(omega, d_beta, kind='cubic')
beta1 = d_beta_interp(omega0)
D_int = beta - beta0 - (omega - omega0) * beta1
import matplotlib.pyplot as plt

plt.figure()
plt.plot(omega/(2*np.pi*1e12), D_int)  # 横轴单位 THz
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Frequency [THz]")
plt.ylabel("Integrated dispersion $D_{int}$ [1/m]")
plt.title("Integrated Dispersion Diagram")
plt.grid(True)
plt.show()
