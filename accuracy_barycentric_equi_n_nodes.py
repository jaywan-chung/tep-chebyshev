# -*- coding: utf-8 -*-
"""
Create a csv file showing the interpolation accuracy of equidistant nodes.
Type "python accuracy_barycentric_equi_n_nodes.py [NUM_NODES]" in the command line to execute it.
Default NUM_NODES is 11.

Last updated on Apr 21 2020

@author: Jaywan Chung
"""

import os
import sys

import numpy as np
import pandas as pd

from libs.pykeri.TEProp import TEProp
from utils import interpolate_barycentric_equi

INFO_FILENAME = "data_info.csv"
DB_FILENAME = "tep_20180409.db"
RESULT_DIR = os.path.join("results", "accuracy")

SEEBECK_SPLINE_ORDER = 2
ELEC_RESI_SPLINE_ORDER = 1
THRM_COND_SPLINE_ORDER = 1

RESULT_COLUMNS = [
    "Seebeck_RMAE", "dSeebeck_dT_RMAE", "elec_resi_RMAE", "thrm_cond_RMAE",
    "Seebeck_RL2E", "dSeebeck_dT_RL2E", "elec_resi_RL2E", "thrm_cond_RL2E",
    "Seebeck_RL1E", "dSeebeck_dT_RL1E", "elec_resi_RL1E", "thrm_cond_RL1E",
]

DEFAULT_NUM_EQUI_NODES = 11


def main(num_equi_nodes):
    result_filename_wo_path = "accuracy_barycentric_equi_{}_nodes.csv".format(num_equi_nodes)
    result_filename = os.path.join(RESULT_DIR, result_filename_wo_path)

    print("{} NODES; FILE = {}".format(num_equi_nodes, result_filename))

    info_df = pd.read_csv(INFO_FILENAME, index_col=0)
    result_df = pd.DataFrame(columns=RESULT_COLUMNS)
    result_df.index.name = 'id'

    for id_num in info_df.index:
        print("processing id={}...".format(id_num))
        mat = TEProp(db_filename=DB_FILENAME, id_num=id_num)

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

        Seebeck_interp_func, Seebeck_exact_func, Seebeck_cheb_nodes = interpolate_barycentric_equi(
            Seebeck_xy_data[:, 0], Seebeck_xy_data[:, 1], num_equi_nodes, spline_order=SEEBECK_SPLINE_ORDER
        )
        elec_resi_interp_func, elec_resi_exact_func, elec_resi_cheb_nodes = interpolate_barycentric_equi(
            elec_resi_xy_data[:, 0], elec_resi_xy_data[:, 1], num_equi_nodes, spline_order=ELEC_RESI_SPLINE_ORDER
        )
        thrm_cond_interp_func, thrm_cond_exact_func, thrm_cond_cheb_nodes = interpolate_barycentric_equi(
            thrm_cond_xy_data[:, 0], thrm_cond_xy_data[:, 1], num_equi_nodes, spline_order=THRM_COND_SPLINE_ORDER
        )

        # compute the curves
        Seebeck_T = np.linspace(Seebeck_xy_data[:, 0][0], Seebeck_xy_data[:, 0][-1], num=10000)
        elec_resi_T = np.linspace(elec_resi_xy_data[:, 0][0], elec_resi_xy_data[:, 0][-1], num=10000)
        thrm_cond_T = np.linspace(thrm_cond_xy_data[:, 0][0], thrm_cond_xy_data[:, 0][-1], num=10000)
        Seebeck_interp_curve = Seebeck_interp_func(Seebeck_T)
        Seebeck_exact_curve = Seebeck_exact_func(Seebeck_T)
        dSeebeck_dT_interp_curve = Seebeck_interp_func.derivative()(Seebeck_T)
        dSeebeck_dT_exact_curve = Seebeck_exact_func.derivative()(Seebeck_T)
        elec_resi_interp_curve = elec_resi_interp_func(elec_resi_T)
        elec_resi_exact_curve = elec_resi_exact_func(elec_resi_T)
        thrm_cond_interp_curve = thrm_cond_interp_func(thrm_cond_T)
        thrm_cond_exact_curve = thrm_cond_exact_func(thrm_cond_T)

        # RMAE (Relative Maximum Absolute Error), divided by maximum exact value
        Seebeck_RMAE = np.max(np.abs(Seebeck_interp_curve - Seebeck_exact_curve)) / np.max(np.abs(Seebeck_exact_curve))
        dSeebeck_dT_RMAE = np.max(
            np.abs(dSeebeck_dT_interp_curve - dSeebeck_dT_exact_curve)
        ) / np.max(np.abs(dSeebeck_dT_exact_curve))
        elec_resi_RMAE = np.max(
            np.abs(elec_resi_interp_curve - elec_resi_exact_curve)
        ) / np.max(np.abs(elec_resi_exact_curve))
        thrm_cond_RMAE = np.max(
            np.abs(thrm_cond_interp_curve - thrm_cond_exact_curve)
        ) / np.max(np.abs(thrm_cond_exact_curve))

        # RL2E (Relative L^2-Error)
        Seebeck_RL2E = np.sqrt(
            np.trapz((Seebeck_interp_curve - Seebeck_exact_curve) ** 2, Seebeck_T)
            / np.trapz(Seebeck_exact_curve ** 2, Seebeck_T)
        )
        dSeebeck_dT_RL2E = np.sqrt(
            np.trapz((dSeebeck_dT_interp_curve - dSeebeck_dT_exact_curve) ** 2, Seebeck_T)
            / np.trapz(dSeebeck_dT_exact_curve ** 2, Seebeck_T)
        )
        elec_resi_RL2E = np.sqrt(
            np.trapz((elec_resi_interp_curve - elec_resi_exact_curve) ** 2, elec_resi_T)
            / np.trapz(elec_resi_exact_curve ** 2, elec_resi_T)
        )
        thrm_cond_RL2E = np.sqrt(
            np.trapz((thrm_cond_interp_curve - thrm_cond_exact_curve) ** 2, thrm_cond_T)
            / np.trapz(thrm_cond_exact_curve ** 2, thrm_cond_T)
        )

        # RL1E (Relative L^1-Error)
        Seebeck_RL1E = np.trapz(
            np.abs(Seebeck_interp_curve - Seebeck_exact_curve), Seebeck_T
        ) / np.trapz(np.abs(Seebeck_exact_curve), Seebeck_T)
        dSeebeck_dT_RL1E = np.trapz(
            np.abs(dSeebeck_dT_interp_curve - dSeebeck_dT_exact_curve), Seebeck_T
        ) / np.trapz(np.abs(dSeebeck_dT_exact_curve), Seebeck_T)
        elec_resi_RL1E = np.trapz(
            np.abs(elec_resi_interp_curve - elec_resi_exact_curve), elec_resi_T
        ) / np.trapz(np.abs(elec_resi_exact_curve), elec_resi_T)
        thrm_cond_RL1E = np.trapz(
            np.abs(thrm_cond_interp_curve - thrm_cond_exact_curve), thrm_cond_T
        ) / np.trapz(np.abs(thrm_cond_exact_curve), thrm_cond_T)

        # save the result
        result_df.loc[id_num] = [Seebeck_RMAE, dSeebeck_dT_RMAE, elec_resi_RMAE, thrm_cond_RMAE,
                                 Seebeck_RL2E, dSeebeck_dT_RL2E, elec_resi_RL2E, thrm_cond_RL2E,
                                 Seebeck_RL1E, dSeebeck_dT_RL1E, elec_resi_RL1E, thrm_cond_RL1E]

    result_df.to_csv(result_filename)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main(DEFAULT_NUM_EQUI_NODES)
