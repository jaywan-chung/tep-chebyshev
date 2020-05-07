# -*- coding: utf-8 -*-
"""
Plot the top of Figure 2 and the top of Figure 3 of the paper.

Before executing the script, the interpolation accuracy for each node should be computed
by using "accuracy_barycentric_cheb_n_nodes.py" and "accuracy_barycentric_equi_n_nodes.py".

Last updated on Apr 29 2020

@author: Jaywan Chung
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# matplotlib settings
font = {'size': 20}
matplotlib.rc('font', **font)
MARKERSIZE = 20
CAPSIZE = 5
LINEWIDTH = 3

INFO_FILENAME = "data_info.csv"
RESULT_DIR = os.path.join("results", "accuracy")

START_NUM_NODES = 3
END_NUM_NODES = 16

cheb_csv_filename_dict = {}
equi_csv_filename_dict = {}
for i in range(START_NUM_NODES, END_NUM_NODES+1):
    cheb_csv_filename_dict[i] = os.path.join(RESULT_DIR, "accuracy_barycentric_cheb_{}_nodes.csv".format(i))
    equi_csv_filename_dict[i] = os.path.join(RESULT_DIR, "accuracy_barycentric_equi_{}_nodes.csv".format(i))

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
cheb_df_dict = {}
equi_df_dict = {}
order_list = sorted(cheb_csv_filename_dict.keys())

for order in order_list:
    cheb_csv_filename = cheb_csv_filename_dict[order]
    equi_csv_filename = equi_csv_filename_dict[order]
    cheb_df_dict[order] = pd.read_csv(cheb_csv_filename, index_col=0)
    equi_df_dict[order] = pd.read_csv(equi_csv_filename, index_col=0)


def plot_accuracy(error_id, error_ylabel):
    cheb_Seebeck_mean_list = []
    cheb_dSeebeck_dT_mean_list = []
    cheb_elec_resi_mean_list = []
    cheb_thrm_cond_mean_list = []

    equi_Seebeck_mean_list = []
    equi_dSeebeck_dT_mean_list = []
    equi_elec_resi_mean_list = []
    equi_thrm_cond_mean_list = []

    cheb_Seebeck_std_list = []
    cheb_dSeebeck_dT_std_list = []
    cheb_elec_resi_std_list = []
    cheb_thrm_cond_std_list = []

    equi_Seebeck_std_list = []
    equi_dSeebeck_dT_std_list = []
    equi_elec_resi_std_list = []
    equi_thrm_cond_std_list = []

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

        equi_Seebeck_mean = equi_df_dict[_order]["Seebeck_{}".format(error_id)].mean() * 100
        equi_Seebeck_mean_list.append(equi_Seebeck_mean)
        equi_dSeebeck_dT_mean = equi_df_dict[_order]["dSeebeck_dT_{}".format(error_id)].mean() * 100
        equi_dSeebeck_dT_mean_list.append(equi_dSeebeck_dT_mean)
        equi_elec_resi_mean = equi_df_dict[_order]["elec_resi_{}".format(error_id)].mean() * 100
        equi_elec_resi_mean_list.append(equi_elec_resi_mean)
        equi_thrm_cond_mean = equi_df_dict[_order]["thrm_cond_{}".format(error_id)].mean() * 100
        equi_thrm_cond_mean_list.append(equi_thrm_cond_mean)

        # confidence interval
        cheb_Seebeck_std = cheb_df_dict[_order]["Seebeck_{}".format(error_id)].std() * 100
        cheb_Seebeck_std_list.append(cheb_Seebeck_std)
        cheb_dSeebeck_dT_std = cheb_df_dict[_order]["dSeebeck_dT_{}".format(error_id)].std() * 100
        cheb_dSeebeck_dT_std_list.append(cheb_dSeebeck_dT_std)
        cheb_elec_resi_std = cheb_df_dict[_order]["elec_resi_{}".format(error_id)].std() * 100
        cheb_elec_resi_std_list.append(cheb_elec_resi_std)
        cheb_thrm_cond_std = cheb_df_dict[_order]["thrm_cond_{}".format(error_id)].std() * 100
        cheb_thrm_cond_std_list.append(cheb_thrm_cond_std)

        equi_Seebeck_std = equi_df_dict[_order]["Seebeck_{}".format(error_id)].std() * 100
        equi_Seebeck_std_list.append(equi_Seebeck_std)
        equi_dSeebeck_dT_std = equi_df_dict[_order]["dSeebeck_dT_{}".format(error_id)].std() * 100
        equi_dSeebeck_dT_std_list.append(equi_dSeebeck_dT_std)
        equi_elec_resi_std = equi_df_dict[_order]["elec_resi_{}".format(error_id)].std() * 100
        equi_elec_resi_std_list.append(equi_elec_resi_std)
        equi_thrm_cond_std = equi_df_dict[_order]["thrm_cond_{}".format(error_id)].std() * 100
        equi_thrm_cond_std_list.append(equi_thrm_cond_std)

    # Seebeck error curve / averaged
    plt.figure(figsize=(21, 7))
    ax1 = plt.subplot(131)
    plt.grid(True)
    plt.plot(order_list, cheb_Seebeck_mean_list, color='r', label="Chebyshev nodes", linewidth=LINEWIDTH)
    plt.plot(order_list, cheb_Seebeck_mean_list, color='r', marker='.', markersize=MARKERSIZE, linestyle='None')
    plt.plot(
        order_list, equi_Seebeck_mean_list, color='g', label="equidistant nodes", linewidth=LINEWIDTH, linestyle=':'
    )
    plt.plot(order_list, equi_Seebeck_mean_list, color='g', marker='.', markersize=MARKERSIZE, linestyle='None')
    plt.xticks(order_list)
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
    plt.plot(
        order_list, equi_elec_resi_mean_list, color='g', label="equidistant nodes", linewidth=LINEWIDTH, linestyle=':'
    )
    plt.plot(order_list, equi_elec_resi_mean_list, color='g', marker='.', markersize=MARKERSIZE, linestyle='None')
    plt.xticks(order_list)
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
    plt.plot(
        order_list, equi_thrm_cond_mean_list, color='g', label="equidistant nodes", linewidth=LINEWIDTH, linestyle=':'
    )
    plt.plot(order_list, equi_thrm_cond_mean_list, color='g', marker='.', markersize=MARKERSIZE, linestyle='None')
    plt.xticks(order_list)
    plt.axhline(y=0, color='k', linestyle=':')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean {} [%]".format(error_ylabel))
    plt.title("Thermal conductivity")
    plt.legend()
    plt.tight_layout()


plot_accuracy("RL1E", r"Relative $L^1$-Norm Error")
plot_accuracy("RMAE", r"Relative $L^\infty$-Norm Error")
plt.show()

# find most deviant example
ex_order = 13
ex_cheb = cheb_df_dict[ex_order]["Seebeck_RL1E"] \
          + cheb_df_dict[ex_order]["elec_resi_RL1E"] \
          + cheb_df_dict[ex_order]["thrm_cond_RL1E"]
ex_equi = equi_df_dict[ex_order]["Seebeck_RL1E"] \
          + equi_df_dict[ex_order]["elec_resi_RL1E"] \
          + equi_df_dict[ex_order]["thrm_cond_RL1E"]
ex_diff = ex_equi - ex_cheb
print(ex_diff.sort_values(ascending=False))
