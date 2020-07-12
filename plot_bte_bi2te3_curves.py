# -*- coding: utf-8 -*-
"""
Plot Figure 7 of the paper.

The excel file "plot_bte_bi2te3_curves.xlsx" is needed.

Last updated on July 12 2020

@author: Byungki Ryu and Jaywan Chung
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import interpolate_barycentric_cheb

# matplotlib settings
font = {'size': 20}
matplotlib.rc('font', **font)


SEEBECK_SPLINE_ORDER = 2
ELEC_RESI_SPLINE_ORDER = 1
THRM_COND_SPLINE_ORDER = 1

NUM_CHEB_NODES = 7
XLS_FILENAME = "plot_bte_bi2te3_curves.xlsx"

tau0 = 2
n_el = 2e19
bmfp = 1e1/1e9
anisotropy = 2/3

Temp_i = 300
Temp_f = 550

df_bte_el = pd.read_excel(XLS_FILENAME, sheet_name='bte_electron_curves').dropna()
df_bte_ph = pd.read_excel(XLS_FILENAME, sheet_name='bte_phonon_curves').dropna()
df_bte_expt = pd.read_excel(XLS_FILENAME, sheet_name='id19').dropna()

logical_el = np.logical_and(df_bte_el.Temperature <= Temp_f, Temp_i <= df_bte_el.Temperature)
logical_ph = np.logical_and(df_bte_ph.Temperature <= Temp_f, Temp_i <= df_bte_ph.Temperature)

df_bte_el = df_bte_el[logical_el]
df_bte_ph = df_bte_ph[logical_ph]

df_bte_el_select = df_bte_el[df_bte_el.n_el == n_el]
df_bte_ph_select = df_bte_ph[df_bte_ph.bmfp == bmfp]

T_array = df_bte_el_select.Temperature.to_numpy()
tau = 300/T_array * tau0 * 1e-14

alpha_xx = df_bte_el_select.alpha_xx.to_numpy()
sigma_xx = df_bte_el_select.sigma_xx.to_numpy() * tau
kappa0_xx = df_bte_el_select.kappa0_xx.to_numpy() * tau
kappa_el_xx = kappa0_xx - alpha_xx**2 * sigma_xx * T_array
kappa_ph_xx = df_bte_ph_select.k_ph_xx
kappa_xx = kappa_el_xx + kappa_ph_xx
zt_xx = alpha_xx * alpha_xx * sigma_xx / kappa_xx * T_array

alpha_zz = df_bte_el_select.alpha_zz.to_numpy()
sigma_zz = df_bte_el_select.sigma_zz.to_numpy() * tau
kappa0_zz = df_bte_el_select.kappa0_zz.to_numpy() * tau
kappa_el_zz = kappa0_zz - alpha_zz**2 * sigma_zz * T_array
kappa_ph_zz = df_bte_ph_select.k_ph_zz
kappa_zz = kappa_el_zz + kappa_ph_zz
zt_zz = alpha_zz * alpha_zz * sigma_zz / kappa_zz * T_array

alpha = alpha_xx * anisotropy + alpha_zz * (1-anisotropy)
sigma = sigma_xx * anisotropy + sigma_zz * (1-anisotropy)
kappa = kappa_xx * anisotropy + kappa_zz * (1-anisotropy)

alpha = alpha*1

temp_expt = df_bte_expt.Temperature.to_numpy()
alpha_expt = df_bte_expt.alpha.to_numpy()
sigma_expt = df_bte_expt.sigma.to_numpy()
kappa_expt = df_bte_expt.kappa.to_numpy()

zt = alpha * alpha * sigma / kappa * T_array
zt_expt = df_bte_expt.ZT

# interpolation
seebeck_expt_interp_func, _, seebeck_expt_cheb_nodes = interpolate_barycentric_cheb(
    temp_expt, alpha_expt, NUM_CHEB_NODES, SEEBECK_SPLINE_ORDER)
elec_resi_expt_interp_func, _, elec_resi_expt_cheb_nodes = interpolate_barycentric_cheb(
    temp_expt, 1/sigma_expt, NUM_CHEB_NODES, ELEC_RESI_SPLINE_ORDER)
thrm_cond_expt_interp_func, _, thrm_cond_expt_cheb_nodes = interpolate_barycentric_cheb(
    temp_expt, kappa_expt, NUM_CHEB_NODES, THRM_COND_SPLINE_ORDER)
seebeck_bte_interp_func, _, seebeck_bte_cheb_nodes = interpolate_barycentric_cheb(
    T_array, alpha, NUM_CHEB_NODES, SEEBECK_SPLINE_ORDER)
elec_resi_bte_interp_func, _, elec_resi_bte_cheb_nodes = interpolate_barycentric_cheb(
    T_array, 1/sigma, NUM_CHEB_NODES, ELEC_RESI_SPLINE_ORDER)
thrm_cond_bte_interp_func, _, thrm_cond_bte_cheb_nodes = interpolate_barycentric_cheb(
    T_array, kappa, NUM_CHEB_NODES, THRM_COND_SPLINE_ORDER)

# draw the figure
LABEL_CHEB_NODES = '{} nodes'.format(NUM_CHEB_NODES)

plt.figure(figsize=(21, 7))
T_mesh_expt = np.linspace(temp_expt[0], temp_expt[-1], num=100)
T_mesh_bte = np.linspace(T_array[0], T_array[-1], num=100)
plt.subplot(131)
plt.plot(T_array, alpha * 1e6, color='b', marker='.', markersize=10, linestyle='None', label='Simulation')
plt.plot(T_mesh_bte, seebeck_bte_interp_func(T_mesh_bte) * 1e6, linestyle='-', color='b',
         label=LABEL_CHEB_NODES)
plt.plot(temp_expt, alpha_expt * 1e6, color='r', marker='.', markersize=10, linestyle='None', label='Experiment')
plt.plot(T_mesh_expt, seebeck_expt_interp_func(T_mesh_expt) * 1e6, linestyle='-', color='r',
         label=LABEL_CHEB_NODES)
plt.ylim(0, 300)
plt.ylabel(r"Seebeck coefficient [$\mu$V/K]")
plt.xlabel("Temperature [K]")
plt.legend(loc='lower right')

plt.subplot(132)
plt.plot(T_array, 1/sigma * 1e2, color='b', marker='.', markersize=10, linestyle='None', label='Simulation')
plt.plot(T_mesh_bte, elec_resi_bte_interp_func(T_mesh_bte) * 1e2, linestyle='-', color='b',
         label=LABEL_CHEB_NODES)
plt.plot(temp_expt, 1/sigma_expt * 1e2, color='r', marker='.', markersize=10, linestyle='None', label='Experiment')
plt.plot(T_mesh_expt, elec_resi_expt_interp_func(T_mesh_expt) * 1e2, linestyle='-', color='r',
         label=LABEL_CHEB_NODES)
plt.ylim(0, 0.0030)
plt.ylabel(r"Electrical resistivity [$\Omega\cdot$cm]")
plt.xlabel("Temperature [K]")
plt.legend(loc='upper left')

plt.subplot(133)
plt.plot(T_array, kappa, color='b', marker='.', markersize=10, linestyle='None', label='Simulation')
plt.plot(T_mesh_bte, thrm_cond_bte_interp_func(T_mesh_bte), linestyle='-', color='b',
         label=LABEL_CHEB_NODES)
plt.plot(temp_expt, kappa_expt, color='r', marker='.', markersize=10, linestyle='None', label='Experiment')
plt.plot(T_mesh_expt, thrm_cond_expt_interp_func(T_mesh_expt), linestyle='-', color='r',
         label=LABEL_CHEB_NODES)
plt.ylim(0, 3.0)
plt.ylabel("Thermal conductivity [W/m/K]")
plt.xlabel("Temperature [K]")
plt.legend(loc='upper left')

plt.tight_layout()

plt.savefig("Fig7_BTE_Bi2Te3_{}_nodes.eps".format(NUM_CHEB_NODES), dpi=300)