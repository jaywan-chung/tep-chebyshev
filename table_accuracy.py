# -*- coding: utf-8 -*-
"""
Make Tables 2, 3, 4, 5 in Supplementary Information of the paper.

Before executing the script, the interpolation accuracy for each node should be computed
by using "accuracy_barycentric_cheb_n_nodes.py" and "accuracy_barycentric_equi_n_nodes.py".

Last updated on Apr 24 2020

@author: Jaywan Chung
"""

import os

import pandas as pd
from scipy.stats import sem, t

INFO_FILENAME = "data_info.csv"
ACCURACY_RESULT_DIR = os.path.join("results", "accuracy")
SAVE_DIR = os.path.join("results", "table")

START_NUM_NODES = 3
END_NUM_NODES = 16

INDEX_COL = '# of nodes'
SEEBECK_MEAN_COL = 'Seebeck: sample mean [%]'
SEEBECK_CONF_INT_COL = 'Seebeck: half-length of 95% confidence interval for population mean [%]'
SEEBECK_STD_COL = 'Seebeck: sample standard deviation [%]'
ELEC_RESI_MEAN_COL = 'elec_resi: sample mean [%]'
ELEC_RESI_CONF_INT_COL = 'elec_resi: half-length of 95% confidence interval for population mean [%]'
ELEC_RESI_STD_COL = 'elec_resi: sample standard deviation [%]'
THRM_COND_MEAN_COL = 'thrm_cond: sample mean [%]'
THRM_COND_CONF_INT_COL = 'thrm_cond: half-length of 95% confidence interval for population mean [%]'
THRM_COND_STD_COL = 'thrm_cond: sample standard deviation [%]'

COLUMNS = [INDEX_COL,
           SEEBECK_MEAN_COL,
           SEEBECK_CONF_INT_COL,
           SEEBECK_STD_COL,
           ELEC_RESI_MEAN_COL,
           ELEC_RESI_CONF_INT_COL,
           ELEC_RESI_STD_COL,
           THRM_COND_MEAN_COL,
           THRM_COND_CONF_INT_COL,
           THRM_COND_STD_COL,
           ]

cheb_csv_filename_dict = {}
equi_csv_filename_dict = {}
for i in range(START_NUM_NODES, END_NUM_NODES+1):
    cheb_csv_filename_dict[i] = os.path.join(ACCURACY_RESULT_DIR, "accuracy_barycentric_cheb_{}_nodes.csv".format(i))
    equi_csv_filename_dict[i] = os.path.join(ACCURACY_RESULT_DIR, "accuracy_barycentric_equi_{}_nodes.csv".format(i))


info_df = pd.read_csv(INFO_FILENAME, index_col=0)
cheb_accuracy_df_dict = {}
equi_accuracy_df_dict = {}
order_list = sorted(cheb_csv_filename_dict.keys())

for order in order_list:
    cheb_csv_filename = cheb_csv_filename_dict[order]
    equi_csv_filename = equi_csv_filename_dict[order]
    cheb_accuracy_df_dict[order] = pd.read_csv(cheb_csv_filename, index_col=0)
    equi_accuracy_df_dict[order] = pd.read_csv(equi_csv_filename, index_col=0)


def get_confidence_interval(data_series):
    # 95% confidence interval, using the student t-distribution
    confidence_interval = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)
    return confidence_interval


def tabulate_accuracy(accuracy_df_dict, error_id):
    result_df = pd.DataFrame(columns=COLUMNS).set_index(INDEX_COL)

    for order_ in order_list:
        # sample mean
        Seebeck_mean = accuracy_df_dict[order_]["Seebeck_{}".format(error_id)].mean() * 100
        result_df.at[order_, SEEBECK_MEAN_COL] = Seebeck_mean
        elec_resi_mean = accuracy_df_dict[order_]["elec_resi_{}".format(error_id)].mean() * 100
        result_df.at[order_, ELEC_RESI_MEAN_COL] = elec_resi_mean
        thrm_cond_mean = accuracy_df_dict[order_]["thrm_cond_{}".format(error_id)].mean() * 100
        result_df.at[order_, THRM_COND_MEAN_COL] = thrm_cond_mean

        # confidence interval
        Seebeck_conf_int = get_confidence_interval(accuracy_df_dict[order_]["Seebeck_{}".format(error_id)] * 100)
        result_df.at[order_, SEEBECK_CONF_INT_COL] = Seebeck_conf_int
        elec_resi_conf_int = get_confidence_interval(accuracy_df_dict[order_]["elec_resi_{}".format(error_id)] * 100)
        result_df.at[order_, ELEC_RESI_CONF_INT_COL] = elec_resi_conf_int
        thrm_cond_conf_int = get_confidence_interval(accuracy_df_dict[order_]["thrm_cond_{}".format(error_id)] * 100)
        result_df.at[order_, THRM_COND_CONF_INT_COL] = thrm_cond_conf_int

        # sample standard deviation
        Seebeck_std = accuracy_df_dict[order_]["Seebeck_{}".format(error_id)].std() * 100
        result_df.at[order_, SEEBECK_STD_COL] = Seebeck_std
        elec_resi_std = accuracy_df_dict[order_]["elec_resi_{}".format(error_id)].std() * 100
        result_df.at[order_, ELEC_RESI_STD_COL] = elec_resi_std
        thrm_cond_std = accuracy_df_dict[order_]["thrm_cond_{}".format(error_id)].std() * 100
        result_df.at[order_, THRM_COND_STD_COL] = thrm_cond_std

    return result_df


cheb_RL1E_df = tabulate_accuracy(cheb_accuracy_df_dict, "RL1E")
equi_RL1E_df = tabulate_accuracy(equi_accuracy_df_dict, "RL1E")
cheb_RMAE_df = tabulate_accuracy(cheb_accuracy_df_dict, "RMAE")
equi_RMAE_df = tabulate_accuracy(equi_accuracy_df_dict, "RMAE")

cheb_RL1E_df.to_csv(os.path.join(SAVE_DIR, "table_accuracy_cheb_relative_L^1_norm_error.csv"))
equi_RL1E_df.to_csv(os.path.join(SAVE_DIR, "table_accuracy_equi_relative_L^1_norm_error.csv"))
cheb_RMAE_df.to_csv(os.path.join(SAVE_DIR, "table_accuracy_cheb_relative_L^infty_norm_error.csv"))
equi_RMAE_df.to_csv(os.path.join(SAVE_DIR, "table_accuracy_equi_relative_L^infty_norm_error.csv"))
