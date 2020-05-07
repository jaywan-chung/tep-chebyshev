# -*- coding: utf-8 -*-
"""
Plot Figure 8 of the paper.

Last updated on Apr 29 2020

@author: Jaywan Chung
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from libs.pykeri.TEProp import TEProp
from utils import interpolate_barycentric_cheb
from utils import BarycentricLagrangeChebyshevNodes

font = {'size': 20}
matplotlib.rc('font', **font)

INFO_FILENAME = "data_info.csv"
DB_FILENAME = "tep_20180409.db"
SEEBECK_SPLINE_ORDER = 2
ELEC_RESI_SPLINE_ORDER = 1
THRM_COND_SPLINE_ORDER = 1

NUM_CHEB_NODES = 13
NUM_CHEB_NODES_LEFT = 7
NUM_CHEB_NODES_RIGHT = 7

print("{}+{} Chebyshev nodes.".format(NUM_CHEB_NODES_LEFT, NUM_CHEB_NODES_RIGHT))

DISCONTINUITY_RANGE = [446.329, 472.455]
DISCONTINUITY_POINT = DISCONTINUITY_RANGE[1]


def interpolate_barycentric_cheb_with_exact_func(xl, xr, num_cheb_nodes, exact_func):
    cheb_nodes = (xl - xr) / 2 * np.cos(
        np.pi * np.linspace(0, num_cheb_nodes - 1, num_cheb_nodes) / (num_cheb_nodes - 1)
    ) + (xl + xr) / 2
    cheb_interp_func = BarycentricLagrangeChebyshevNodes(cheb_nodes, exact_func(cheb_nodes))
    return cheb_interp_func, cheb_nodes


for id_num in [162]:  # for Figure 11
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

    Seebeck_x_left = Seebeck_xy_data[0, 0]
    Seebeck_x_right = Seebeck_xy_data[-1, 0]
    elec_resi_x_left = elec_resi_xy_data[0, 0]
    elec_resi_x_right = elec_resi_xy_data[-1, 0]
    thrm_cond_x_left = thrm_cond_xy_data[0, 0]
    thrm_cond_x_right = thrm_cond_xy_data[-1, 0]

    _, Seebeck_exact_func, _ = interpolate_barycentric_cheb(
        Seebeck_xy_data[:, 0], Seebeck_xy_data[:, 1], NUM_CHEB_NODES, spline_order=SEEBECK_SPLINE_ORDER
    )
    _, elec_resi_exact_func, _ = interpolate_barycentric_cheb(
        elec_resi_xy_data[:, 0], elec_resi_xy_data[:, 1], NUM_CHEB_NODES, spline_order=ELEC_RESI_SPLINE_ORDER
    )
    _, thrm_cond_exact_func, _ = interpolate_barycentric_cheb(
        thrm_cond_xy_data[:, 0], thrm_cond_xy_data[:, 1], NUM_CHEB_NODES, spline_order=THRM_COND_SPLINE_ORDER
    )

    Seebeck_interp_func_left, Seebeck_cheb_nodes_left = interpolate_barycentric_cheb_with_exact_func(
        Seebeck_x_left, DISCONTINUITY_POINT, NUM_CHEB_NODES_LEFT, Seebeck_exact_func
    )
    Seebeck_interp_func_right, Seebeck_cheb_nodes_right = interpolate_barycentric_cheb_with_exact_func(
        DISCONTINUITY_POINT, Seebeck_x_right, NUM_CHEB_NODES_RIGHT, Seebeck_exact_func
    )
    elec_resi_interp_func_left, elec_resi_cheb_nodes_left = interpolate_barycentric_cheb_with_exact_func(
        elec_resi_x_left, DISCONTINUITY_POINT, NUM_CHEB_NODES_LEFT, elec_resi_exact_func
    )
    elec_resi_interp_func_right, elec_resi_cheb_nodes_right = interpolate_barycentric_cheb_with_exact_func(
        DISCONTINUITY_POINT, elec_resi_x_right, NUM_CHEB_NODES_RIGHT, elec_resi_exact_func
    )
    thrm_cond_interp_func_left, thrm_cond_cheb_nodes_left = interpolate_barycentric_cheb_with_exact_func(
        thrm_cond_x_left, DISCONTINUITY_POINT, NUM_CHEB_NODES_LEFT, thrm_cond_exact_func
    )
    thrm_cond_interp_func_right, thrm_cond_cheb_nodes_right = interpolate_barycentric_cheb_with_exact_func(
        DISCONTINUITY_POINT, thrm_cond_x_right, NUM_CHEB_NODES_RIGHT, thrm_cond_exact_func
    )

    # compute the curves
    Seebeck_T = np.linspace(Seebeck_x_left, Seebeck_x_right, num=1000, dtype=np.float64)
    Seebeck_T_left = np.linspace(Seebeck_x_left, DISCONTINUITY_POINT, num=1000, dtype=np.float64)
    Seebeck_T_right = np.linspace(DISCONTINUITY_POINT, Seebeck_x_right, num=1000, dtype=np.float64)
    elec_resi_T = np.linspace(elec_resi_x_left, elec_resi_x_right, num=1000, dtype=np.float64)
    elec_resi_T_left = np.linspace(elec_resi_x_left, DISCONTINUITY_POINT, num=1000, dtype=np.float64)
    elec_resi_T_right = np.linspace(DISCONTINUITY_POINT, elec_resi_x_right, num=1000, dtype=np.float64)
    thrm_cond_T = np.linspace(thrm_cond_x_left, thrm_cond_x_right, num=1000, dtype=np.float64)
    thrm_cond_T_left = np.linspace(thrm_cond_x_left, DISCONTINUITY_POINT, num=1000, dtype=np.float64)
    thrm_cond_T_right = np.linspace(DISCONTINUITY_POINT, thrm_cond_x_right, num=1000, dtype=np.float64)
    T_grid = np.linspace(Tc, Th, num=1000, dtype=np.float64)
    Seebeck_exact_curve = Seebeck_exact_func(Seebeck_T)
    Seebeck_interp_curve_left = Seebeck_interp_func_left(Seebeck_T_left)
    Seebeck_interp_curve_right = Seebeck_interp_func_right(Seebeck_T_right)
    elec_resi_exact_curve = elec_resi_exact_func(elec_resi_T)
    elec_resi_interp_curve_left = elec_resi_interp_func_left(elec_resi_T_left)
    elec_resi_interp_curve_right = elec_resi_interp_func_right(elec_resi_T_right)
    thrm_cond_exact_curve = thrm_cond_exact_func(thrm_cond_T)
    thrm_cond_interp_curve_left = thrm_cond_interp_func_left(thrm_cond_T_left)
    thrm_cond_interp_curve_right = thrm_cond_interp_func_right(thrm_cond_T_right)

    # draw the figures
    xticks = np.linspace(300, 900, 7, dtype=int)
    # Seebeck curve
    plt.figure(figsize=(21, 7))
    ax1 = plt.subplot(131)
    plt.plot(Seebeck_T_left, Seebeck_interp_curve_left * 1e6, linestyle='-', color='r',
             label='Chebyshev\n({}+{} nodes)'.format(NUM_CHEB_NODES_LEFT, NUM_CHEB_NODES_RIGHT))
    plt.plot(Seebeck_T_right, Seebeck_interp_curve_right * 1e6, linestyle='-', color='r')
    plt.plot(Seebeck_T, Seebeck_exact_curve * 1e6, linestyle='--', color='k', label='exact')
    plt.plot(Seebeck_cheb_nodes_left, Seebeck_interp_func_left(Seebeck_cheb_nodes_left) * 1e6, marker='.',
             markersize=20,
             linestyle='None', color='r')
    plt.plot(Seebeck_cheb_nodes_right, Seebeck_interp_func_right(Seebeck_cheb_nodes_right) * 1e6, marker='.',
             markersize=20,
             linestyle='None', color='r')
    plt.plot(Seebeck_xy_data[:, 0], Seebeck_exact_func(Seebeck_xy_data[:, 0]) * 1e6, marker='.', markersize=10,
             linestyle='None', color='k')
    plt.axvline(x=DISCONTINUITY_POINT, color='k', linestyle=':')
    plt.legend()
    plt.ylabel(r"Seebeck coefficient [$\mu$V/K]")
    plt.xlabel("Temperature [K]")
    plt.xticks(xticks)

    # elec_resi curve
    plt.subplot(132, sharex=ax1)
    plt.plot(elec_resi_T_left, elec_resi_interp_curve_left * 1e2, linestyle='-', color='r',
             label='Chebyshev\n({}+{} nodes)'.format(NUM_CHEB_NODES_LEFT, NUM_CHEB_NODES_RIGHT))
    plt.plot(elec_resi_T_right, elec_resi_interp_curve_right * 1e2, linestyle='-', color='r')
    plt.plot(elec_resi_T, elec_resi_exact_curve * 1e2, linestyle='--', color='k', label='exact')
    plt.plot(elec_resi_cheb_nodes_left, elec_resi_interp_func_left(elec_resi_cheb_nodes_left) * 1e2, marker='.',
             markersize=20,
             linestyle='None', color='r')
    plt.plot(elec_resi_cheb_nodes_right, elec_resi_interp_func_right(elec_resi_cheb_nodes_right) * 1e2, marker='.',
             markersize=20,
             linestyle='None', color='r')
    plt.plot(elec_resi_xy_data[:, 0], elec_resi_exact_func(elec_resi_xy_data[:, 0]) * 1e2, marker='.', markersize=10,
             linestyle='None', color='k')
    plt.axvline(x=DISCONTINUITY_POINT, color='k', linestyle=':')
    plt.legend()
    plt.ylabel(r"Electrical resistivity [$\Omega\cdot$cm]")
    plt.xlabel("Temperature [K]")

    # thrm_cond curve
    plt.subplot(133, sharex=ax1)
    plt.plot(thrm_cond_T_left, thrm_cond_interp_curve_left, linestyle='-', color='r',
             label='Chebyshev\n({}+{} nodes)'.format(NUM_CHEB_NODES_LEFT, NUM_CHEB_NODES_RIGHT))
    plt.plot(thrm_cond_T_right, thrm_cond_interp_curve_right, linestyle='-', color='r', )
    plt.plot(thrm_cond_T, thrm_cond_exact_curve, linestyle='--', color='k', label='exact')
    plt.plot(thrm_cond_cheb_nodes_left, thrm_cond_interp_func_left(thrm_cond_cheb_nodes_left), marker='.',
             markersize=20,
             linestyle='None', color='r')
    plt.plot(thrm_cond_cheb_nodes_right, thrm_cond_interp_func_right(thrm_cond_cheb_nodes_right), marker='.',
             markersize=20,
             linestyle='None', color='r')
    plt.plot(thrm_cond_xy_data[:, 0], thrm_cond_exact_func(thrm_cond_xy_data[:, 0]), marker='.', markersize=10,
             linestyle='None', color='k')
    plt.axvline(x=DISCONTINUITY_POINT, color='k', linestyle=':')
    plt.legend()
    plt.ylabel("Thermal conductivity [W/m/K]")
    plt.xlabel("Temperature [K]")

    plt.tight_layout()
plt.show()
