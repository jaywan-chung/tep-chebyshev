# -*- coding: utf-8 -*-
"""
Plot the bottom of Figure 2 and the bottom of Figure 3 of the paper.

Before executing the script, the interpolation accuracy for each node should be computed
by using "accuracy_barycentric_cheb_n_nodes.py".

Last updated on Apr 29 2020

@author: Jaywan Chung
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import t

# matplotlib settings
font = {'size': 20}
matplotlib.rc('font', **font)
MARKERSIZE = 8
CAPSIZE = 5
LINEWIDTH = 3

INFO_FILENAME = "data_info.csv"
RESULT_DIR = os.path.join("results", "accuracy")

START_NUM_NODES = 3
END_NUM_NODES = 16

chebl_csv_filename_dict = {}
for i in range(START_NUM_NODES, END_NUM_NODES+1):
    chebl_csv_filename_dict[i] = os.path.join(RESULT_DIR, "accuracy_barycentric_cheb_{}_nodes.csv".format(i))

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
cheb_df_dict = {}
order_list = sorted(chebl_csv_filename_dict.keys())

for order in order_list:
    cheb_csv_filename = chebl_csv_filename_dict[order]
    cheb_df_dict[order] = pd.read_csv(cheb_csv_filename, index_col=0)


def get_confidence_interval(data_series):
    # 95% confidence interval, using the student t-distribution
    confidence_interval = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)
    return confidence_interval


def plot_accuracy(error_id, error_ylabel, yticks_):
    cheb_Seebeck_mean_list = []
    cheb_dSeebeck_dT_mean_list = []
    cheb_elec_resi_mean_list = []
    cheb_thrm_cond_mean_list = []

    # for confidence interval
    cheb_Seebeck_conf_int_list = []
    cheb_dSeebeck_dT_conf_int_list = []
    cheb_elec_resi_conf_int_list = []
    cheb_thrm_cond_conf_int_list = []

    for _order in order_list:
        # mean
        cheb_Seebeck_mean = cheb_df_dict[_order]["Seebeck_{}".format(error_id)].mean() * 100
        cheb_Seebeck_mean_list.append(cheb_Seebeck_mean)
        cheb_dSeebeck_dT_mean = cheb_df_dict[_order]["dSeebeck_dT_{}".format(error_id)].mean() * 100
        cheb_dSeebeck_dT_mean_list.append(cheb_dSeebeck_dT_mean)
        cheb_elec_resi_mean = cheb_df_dict[_order]["elec_resi_{}".format(error_id)].mean() * 100
        cheb_elec_resi_mean_list.append(cheb_elec_resi_mean)
        cheb_thrm_cond_mean = cheb_df_dict[_order]["thrm_cond_{}".format(error_id)].mean() * 100
        cheb_thrm_cond_mean_list.append(cheb_thrm_cond_mean)

        # confidence interval
        cheb_Seebeck_conf_int = get_confidence_interval(cheb_df_dict[_order]["Seebeck_{}".format(error_id)] * 100)
        cheb_Seebeck_conf_int_list.append(cheb_Seebeck_conf_int)
        cheb_dSeebeck_dT_conf_int = get_confidence_interval(cheb_df_dict[_order]["dSeebeck_dT_{}".format(error_id)] * 100)
        cheb_dSeebeck_dT_conf_int_list.append(cheb_dSeebeck_dT_conf_int)
        cheb_elec_resi_conf_int = get_confidence_interval(cheb_df_dict[_order]["elec_resi_{}".format(error_id)] * 100)
        cheb_elec_resi_conf_int_list.append(cheb_elec_resi_conf_int)
        cheb_thrm_cond_conf_int = get_confidence_interval(cheb_df_dict[_order]["thrm_cond_{}".format(error_id)] * 100)
        cheb_thrm_cond_conf_int_list.append(cheb_thrm_cond_conf_int)

    # Seebeck error curve / averaged
    plt.figure(figsize=(21, 7))
    ax1 = plt.subplot(131)
    plt.grid(True)
    plt.plot(order_list, cheb_Seebeck_mean_list, color='r', label="Chebyshev nodes", linewidth=LINEWIDTH)
    plt.plot(order_list, cheb_Seebeck_mean_list, color='r', marker='.', markersize=MARKERSIZE, linestyle='None')
    plt.errorbar(
        order_list, cheb_Seebeck_mean_list, yerr=cheb_Seebeck_conf_int_list,
        color='r', capsize=CAPSIZE, linewidth=LINEWIDTH, capthick=LINEWIDTH
    )
    plt.xticks(order_list)
    if yticks_ is not None:
        plt.yticks(yticks_)
    plt.axhline(y=0, color='k', linestyle=':')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean {} [%]".format(error_ylabel))
    plt.title("Seebeck coefficient")
    plt.legend()

    # Electrical resistivity error curve / averaged
    plt.subplot(132, sharey=ax1, sharex=ax1)
    plt.grid(True)
    plt.plot(order_list, cheb_elec_resi_mean_list, color='r', label="Chebyshev nodes", linewidth=LINEWIDTH)
    plt.plot(order_list, cheb_elec_resi_mean_list, color='r', marker='.', markersize=MARKERSIZE, linestyle='None')
    plt.errorbar(
        order_list, cheb_elec_resi_mean_list, yerr=cheb_elec_resi_conf_int_list,
        color='r', capsize=CAPSIZE, linewidth=LINEWIDTH, capthick=LINEWIDTH
    )
    plt.axhline(y=0, color='k', linestyle=':')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean {} [%]".format(error_ylabel))
    plt.title("Electrical resistivity")
    plt.legend()

    # Thermal conductivity error curve / averaged
    plt.subplot(133, sharey=ax1, sharex=ax1)
    plt.grid(True)
    plt.plot(order_list, cheb_thrm_cond_mean_list, color='r', label="Chebyshev nodes", linewidth=LINEWIDTH)
    plt.plot(order_list, cheb_thrm_cond_mean_list, color='r', marker='.', markersize=MARKERSIZE, linestyle='None')
    plt.errorbar(
        order_list, cheb_thrm_cond_mean_list, yerr=cheb_thrm_cond_conf_int_list,
        color='r', capsize=CAPSIZE, linewidth=LINEWIDTH, capthick=LINEWIDTH
    )
    plt.axhline(y=0, color='k', linestyle=':')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean {} [%]".format(error_ylabel))
    plt.title("Thermal conductivity")
    plt.legend()
    plt.tight_layout()


yticks = np.linspace(0, 7, 8, dtype=int)
plot_accuracy("RL1E", r"Relative $L^1$-Norm Error", yticks)
yticks = np.linspace(0, 9, 10, dtype=int)
plot_accuracy("RMAE", r"Relative $L^\infty$-Norm Error", yticks)
plt.show()
