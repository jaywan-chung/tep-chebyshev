# -*- coding: utf-8 -*-
"""
Compute the performance of thermoelectric modules using the exact TEPs, and save it to a csv file.

The exact TEPs are assumed to be spline curves: 2nd-order spline for Seebeck coefficient,
piecewise linear curves for electrical resistivity and thermal conductivity.

Last updated on Apr 21 2020

@author: Jaywan Chung
"""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from libs.pykeri.TEProp import TEProp
from utils import ThermoelctricEquationSolver

INFO_FILENAME = "data_info.csv"
DB_FILENAME = "tep_20180409.db"
RESULT_DIR = os.path.join("results", "performance")

SEEBECK_SPLINE_ORDER = 2
ELEC_RESI_SPLINE_ORDER = 1
THRM_COND_SPLINE_ORDER = 1

# spec of a thermoelectric module
L = 1e-3  # [m]
A = 1e-6  # [m^2]

RESULT_COLUMNS = [
    "max_zT", "T_for_(max_zT)",
    "max_power_density", "I_for_(max_power_density)",
    "max_efficiency", "I_for_(max_efficiency)",
    "num_Seebeck_samples", "num_elec_resi_samples", "num_thrm_cond_samples",
]

RESULT_FILENAME = os.path.join(RESULT_DIR, "performance_exact.csv")

print("Performance for exact curves (splines); File = {}".format(RESULT_FILENAME))

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
result_df = pd.DataFrame(columns=RESULT_COLUMNS)
result_df.index.name = 'id'

for id_num in info_df.index:
    print("processing id={}...".format(id_num))
    mat = TEProp(db_filename=DB_FILENAME, id_num=id_num)
    Tc = mat.min_raw_T
    Th = mat.max_raw_T

    Seebeck_xy_data = np.asarray(mat.Seebeck.raw_data())
    elec_resi_xy_data = np.asarray(mat.elec_resi.raw_data())
    thrm_cond_xy_data = np.asarray(mat.thrm_cond.raw_data())

    # make unique on temperature
    _, Seebeck_unique_indices = np.unique(Seebeck_xy_data[:, 0], return_index=True)
    Seebeck_xy_data = Seebeck_xy_data[Seebeck_unique_indices, :]
    _, elec_resi_unique_indices = np.unique(elec_resi_xy_data[:, 0], return_index=True)
    elec_resi_xy_data = elec_resi_xy_data[elec_resi_unique_indices, :]
    _, thrm_cond_unique_indices = np.unique(thrm_cond_xy_data[:, 0], return_index=True)
    thrm_cond_xy_data = thrm_cond_xy_data[thrm_cond_unique_indices, :]

    num_Seebeck_samples = len(Seebeck_xy_data)
    num_elec_resi_samples = len(elec_resi_xy_data)
    num_thrm_cond_samples = len(thrm_cond_xy_data)

    Seebeck_exact_func = UnivariateSpline(
        Seebeck_xy_data[:, 0], Seebeck_xy_data[:, 1], k=SEEBECK_SPLINE_ORDER, s=0
    )
    elec_resi_exact_func = UnivariateSpline(
        elec_resi_xy_data[:, 0], elec_resi_xy_data[:, 1], k=ELEC_RESI_SPLINE_ORDER, s=0
    )
    thrm_cond_exact_func = UnivariateSpline(
        thrm_cond_xy_data[:, 0], thrm_cond_xy_data[:, 1], k=THRM_COND_SPLINE_ORDER, s=0
    )

    # compute the max zT
    T_grid = np.linspace(Tc, Th, num=10000)
    Seebeck_curve = Seebeck_exact_func(T_grid)
    elec_resi_curve = elec_resi_exact_func(T_grid)
    thrm_cond_curve = thrm_cond_exact_func(T_grid)
    zT_curve = Seebeck_curve ** 2 / elec_resi_curve / thrm_cond_curve * T_grid
    index_for_max_zT = np.argmax(zT_curve)
    max_zT = zT_curve[int(index_for_max_zT)]
    T_for_max_zT = T_grid[index_for_max_zT]

    # compute the maximum power density and efficiency
    solver = ThermoelctricEquationSolver(L, A)
    solver.set_bc(Tc, Th)
    solver.set_te_mat_func(thrm_cond_exact_func,
                           elec_resi_exact_func,
                           Seebeck_exact_func, Seebeck_exact_func.derivative())

    res = solver.compute_max_power_density()
    if not res.success:
        print("computation of max. power density failed for id={}".format(id_num))
        continue
    I_for_max_power_density = res.x
    max_power_density = -res.fun
    num_eval_for_max_power_density = res.nfev

    res = solver.compute_max_efficiency()
    if not res.success:
        print("computation of max. efficiency failed for id={}".format(id_num))
        continue
    I_for_max_efficiency = res.x
    max_efficiency = -res.fun
    num_eval_for_max_efficiency = res.nfev

    # test the validity
    Carnot_efficiency = (Th - Tc) / Th * 100
    assert (max_efficiency < Carnot_efficiency)

    print("\tmax. power density= {} [W/cm^2] at I= {} [A] (nfev={})".format(max_power_density, I_for_max_power_density,
                                                                            num_eval_for_max_power_density))
    print("\tmax. efficiency= {} [%] at I= {} [A] (nfev={})".format(max_efficiency, I_for_max_efficiency,
                                                                    num_eval_for_max_efficiency))

    # save the result
    result_df.loc[id_num] = [max_zT, T_for_max_zT,
                             max_power_density, I_for_max_power_density,
                             max_efficiency, I_for_max_efficiency,
                             num_Seebeck_samples, num_elec_resi_samples, num_thrm_cond_samples,
                             ]

result_df = result_df.astype({"num_Seebeck_samples": "int32",
                              "num_elec_resi_samples": "int32",
                              "num_thrm_cond_samples": "int32"})
result_df.to_csv(RESULT_FILENAME)
