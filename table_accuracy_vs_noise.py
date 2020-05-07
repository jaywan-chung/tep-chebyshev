# -*- coding: utf-8 -*-
"""
Make Tables 8, 9, 10 in Supplementary Information of the paper.

Before executing the script, the interpolation accuracy for each node under noise should be computed
by using "accuracy_barycentric_cheb_n_nodes_with_noise.py".

Last updated on Apr 24 2020

@author: Jaywan Chung
"""

from collections import defaultdict
import os

import pandas as pd
from scipy.stats import sem, t

LIST_NUM_NODES = [8, 11, 14]
LIST_NOISE_PERCENT = list(range(0, 10 + 1))

INFO_FILENAME = "data_info.csv"
ACCURACY_RESULT_DIR = os.path.join("results", "accuracy_noise")
SAVE_DIR = os.path.join("results", "table")

INDEX_COL = 'Noise [%]'
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

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
cheb_df_dict = defaultdict(dict)

for num_nodes in LIST_NUM_NODES:
    for noise_percent in LIST_NOISE_PERCENT:
        result_subdir = "noise_{}%".format(noise_percent)
        result_filename_wo_path = "accuracy_cheb_{}_nodes_with_{}%_noise.csv".format(num_nodes, noise_percent)
        cheb_csv_filename = os.path.join(ACCURACY_RESULT_DIR, result_subdir, result_filename_wo_path)
        cheb_df_dict[num_nodes][noise_percent] = pd.read_csv(cheb_csv_filename, index_col=0)


def get_confidence_interval(data_series):
    # 95% confidence interval, using the student t-distribution
    confidence_interval = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)
    return confidence_interval


def tabulate_accuracy(num_nodes_):
    result_df_ = pd.DataFrame(columns=COLUMNS).set_index(INDEX_COL)

    for noise_percent_ in LIST_NOISE_PERCENT:
        # DataFrame containing all the errors
        cheb_Seebeck_error_df = cheb_df_dict[num_nodes_][noise_percent_]["Seebeck_RMAE"] * 100
        cheb_elec_resi_error_df = cheb_df_dict[num_nodes_][noise_percent_]["elec_resi_RMAE"] * 100
        cheb_thrm_cond_error_df = cheb_df_dict[num_nodes_][noise_percent_]["thrm_cond_RMAE"] * 100

        # sample mean
        result_df_.at[noise_percent_, SEEBECK_MEAN_COL] = cheb_Seebeck_error_df.mean()
        result_df_.at[noise_percent_, ELEC_RESI_MEAN_COL] = cheb_elec_resi_error_df.mean()
        result_df_.at[noise_percent_, THRM_COND_MEAN_COL] = cheb_thrm_cond_error_df.mean()

        # confidence interval
        result_df_.at[noise_percent_, SEEBECK_CONF_INT_COL] = get_confidence_interval(cheb_Seebeck_error_df)
        result_df_.at[noise_percent_, ELEC_RESI_CONF_INT_COL] = get_confidence_interval(cheb_elec_resi_error_df)
        result_df_.at[noise_percent_, THRM_COND_CONF_INT_COL] = get_confidence_interval(cheb_thrm_cond_error_df)

        # sample standard deviation
        result_df_.at[noise_percent_, SEEBECK_STD_COL] = cheb_Seebeck_error_df.std()
        result_df_.at[noise_percent_, ELEC_RESI_STD_COL] = cheb_elec_resi_error_df.std()
        result_df_.at[noise_percent_, THRM_COND_STD_COL] = cheb_thrm_cond_error_df.std()

    return result_df_


for num_nodes in LIST_NUM_NODES:
    result_df = tabulate_accuracy(num_nodes)
    result_df.to_csv(os.path.join(SAVE_DIR, "table_accuracy_cheb_{}_nodes_with_noise.csv".format(num_nodes)))
