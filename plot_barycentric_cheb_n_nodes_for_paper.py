# -*- coding: utf-8 -*-
"""
Plot the bottom of Figure 1 of the paper.

Last updated on Apr 27 2020

@author: Jaywan Chung
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from libs.pykeri.TEProp import TEProp
from utils import interpolate_barycentric_cheb

font = {'size': 20}
matplotlib.rc('font', **font)

INFO_FILENAME = "data_info.csv"
DB_FILENAME = "tep_20180409.db"

SEEBECK_SPLINE_ORDER = 2
ELEC_RESI_SPLINE_ORDER = 1
THRM_COND_SPLINE_ORDER = 1

NUM_CHEB_NODES = 13

print("{} Chebyshev nodes.".format(NUM_CHEB_NODES))

for id_num in [192]:  # for Figure 2
    print("processing id={}...".format(id_num))
    mat = TEProp(db_filename=DB_FILENAME, id_num=id_num)
    Tc = mat.min_raw_T
    Th = mat.max_raw_T

    Seebeck_xy_data = np.asarray(mat.Seebeck.raw_data())
    elec_resi_xy_data = np.asarray(mat.elec_resi.raw_data())
    thrm_cond_xy_data = np.asarray(mat.thrm_cond.raw_data())

    # make unique, sorted on temperature
    _, Seebeck_unique_indices = np.unique(Seebeck_xy_data[:, 0], return_index=True)
    Seebeck_xy_data = Seebeck_xy_data[Seebeck_unique_indices, :]
    _, elec_resi_unique_indices = np.unique(elec_resi_xy_data[:, 0], return_index=True)
    elec_resi_xy_data = elec_resi_xy_data[elec_resi_unique_indices, :]
    _, thrm_cond_unique_indices = np.unique(thrm_cond_xy_data[:, 0], return_index=True)
    thrm_cond_xy_data = thrm_cond_xy_data[thrm_cond_unique_indices, :]

    num_Seebeck_samples = len(Seebeck_xy_data)
    num_elec_resi_samples = len(elec_resi_xy_data)
    num_thrm_cond_samples = len(thrm_cond_xy_data)

    Seebeck_interp_func, Seebeck_exact_func, Seebeck_cheb_nodes = interpolate_barycentric_cheb(
        Seebeck_xy_data[:, 0], Seebeck_xy_data[:, 1], NUM_CHEB_NODES, spline_order=SEEBECK_SPLINE_ORDER
    )
    elec_resi_interp_func, elec_resi_exact_func, elec_resi_cheb_nodes = interpolate_barycentric_cheb(
        elec_resi_xy_data[:, 0], elec_resi_xy_data[:, 1], NUM_CHEB_NODES, spline_order=ELEC_RESI_SPLINE_ORDER
    )
    thrm_cond_interp_func, thrm_cond_exact_func, thrm_cond_cheb_nodes = interpolate_barycentric_cheb(
        thrm_cond_xy_data[:, 0], thrm_cond_xy_data[:, 1], NUM_CHEB_NODES, spline_order=THRM_COND_SPLINE_ORDER
    )

    # compute the max zT
    Seebeck_T = np.linspace(Seebeck_xy_data[:, 0][0], Seebeck_xy_data[:, 0][-1], num=1000, dtype=np.float64)
    elec_resi_T = np.linspace(elec_resi_xy_data[:, 0][0], elec_resi_xy_data[:, 0][-1], num=1000, dtype=np.float64)
    thrm_cond_T = np.linspace(thrm_cond_xy_data[:, 0][0], thrm_cond_xy_data[:, 0][-1], num=1000, dtype=np.float64)
    T_grid = np.linspace(Tc, Th, num=1000, dtype=np.float64)
    Seebeck_interp_curve = Seebeck_interp_func(Seebeck_T)
    Seebeck_exact_curve = Seebeck_exact_func(Seebeck_T)
    elec_resi_interp_curve = elec_resi_interp_func(elec_resi_T)
    elec_resi_exact_curve = elec_resi_exact_func(elec_resi_T)
    thrm_cond_interp_curve = thrm_cond_interp_func(thrm_cond_T)
    thrm_cond_exact_curve = thrm_cond_exact_func(thrm_cond_T)
    zT_interp_curve = Seebeck_interp_func(T_grid) ** 2 / elec_resi_interp_func(T_grid) * T_grid / thrm_cond_interp_func(
        T_grid)
    zT_exact_curve = Seebeck_exact_func(T_grid) ** 2 / elec_resi_exact_func(T_grid) * T_grid / thrm_cond_exact_func(
        T_grid)

    # draw the figures
    xticks = np.linspace(300, 900, 7, dtype=int)
    # Seebeck curve
    plt.figure(figsize=(21, 7))
    ax1 = plt.subplot(131)
    plt.plot(Seebeck_T, Seebeck_interp_curve * 1e6, linestyle='-', color='r',
             label='Chebyshev\n({} nodes)'.format(NUM_CHEB_NODES))
    plt.plot(Seebeck_T, Seebeck_exact_curve * 1e6, linestyle='--', color='k', label='exact')
    plt.plot(Seebeck_cheb_nodes, Seebeck_interp_func(Seebeck_cheb_nodes) * 1e6, marker='.', markersize=20,
             linestyle='None', color='r')
    plt.plot(Seebeck_xy_data[:, 0], Seebeck_exact_func(Seebeck_xy_data[:, 0]) * 1e6, marker='.', markersize=10,
             linestyle='None', color='k')
    plt.legend()
    plt.ylabel(r"Seebeck coefficient [$\mu$V/K]")
    plt.xlabel("Temperature [K]")
    plt.xticks(xticks)

    # elec_resi curve
    plt.subplot(132, sharex=ax1)
    plt.plot(elec_resi_T, elec_resi_interp_curve * 1e2, linestyle='-', color='r',
             label='Chebyshev ({} nodes)'.format(NUM_CHEB_NODES))
    plt.plot(elec_resi_T, elec_resi_exact_curve * 1e2, linestyle='--', color='k', label='exact')
    plt.plot(elec_resi_cheb_nodes, elec_resi_interp_func(elec_resi_cheb_nodes) * 1e2, marker='.', markersize=20,
             linestyle='None', color='r')
    plt.plot(elec_resi_xy_data[:, 0], elec_resi_exact_func(elec_resi_xy_data[:, 0]) * 1e2, marker='.', markersize=10,
             linestyle='None', color='k')
    plt.legend()
    plt.ylabel(r"Electrical resistivity [$\Omega\cdot$cm]")
    plt.xlabel("Temperature [K]")

    # thrm_cond curve
    plt.subplot(133, sharex=ax1)
    plt.plot(thrm_cond_T, thrm_cond_interp_curve, linestyle='-', color='r',
             label='Chebyshev ({} nodes)'.format(NUM_CHEB_NODES))
    plt.plot(thrm_cond_T, thrm_cond_exact_curve, linestyle='--', color='k', label='exact')
    plt.plot(thrm_cond_cheb_nodes, thrm_cond_interp_func(thrm_cond_cheb_nodes), marker='.', markersize=20,
             linestyle='None', color='r')
    plt.plot(thrm_cond_xy_data[:, 0], thrm_cond_exact_func(thrm_cond_xy_data[:, 0]), marker='.', markersize=10,
             linestyle='None', color='k')
    plt.legend()
    plt.ylabel("Thermal conductivity [W/m/K]")
    plt.xlabel("Temperature [K]")

    plt.tight_layout()
plt.show()
