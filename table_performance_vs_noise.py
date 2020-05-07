# -*- coding: utf-8 -*-
"""
Make Tables 11, 12, 13, 14, 15, 16 in Supplementary Information of the paper.

Before executing the script, the performance accuracy for each node under noise should be computed
by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes_with_noise.py".

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
EXACT_CSV_FILENAME = os.path.join("results", "performance", "performance_exact.csv")
PERFORMANCE_RESULT_DIR = os.path.join("results", "performance_noise")
SAVE_DIR = os.path.join("results", "table")

INDEX_COL = 'Noise [%]'
MAX_ZT_MEAN_COL = 'max_zT: sample mean [%]'
MAX_ZT_CONF_INT_COL = 'max_zT: half-length of 95% confidence interval for population mean [%]'
MAX_ZT_STD_COL = 'max_zT: sample standard deviation [%]'
T_FOR_MAX_ZT_MEAN_COL = 'T_for_(max_zT): sample mean [%]'
T_FOR_MAX_ZT_CONF_INT_COL = 'T_for_(max_zT): half-length of 95% confidence interval for population mean [%]'
T_FOR_MAX_ZT_STD_COL = 'T_for_(max_zT): sample standard deviation [%]'
MAX_POWER_DENSITY_MEAN_COL = 'max_power_density: sample mean [%]'
MAX_POWER_DENSITY_CONF_INT_COL = 'max_power_density: half-length of 95% confidence interval for population mean [%]'
MAX_POWER_DENSITY_STD_COL = 'max_power_density: sample standard deviation [%]'
I_FOR_MAX_POWER_DENSITY_MEAN_COL = 'I_for_(max_power_density): sample mean [%]'
I_FOR_MAX_POWER_DENSITY_CONF_INT_COL = 'I_for_(max_power_density): ' \
                                       'half-length of 95% confidence interval for population mean [%]'
I_FOR_MAX_POWER_DENSITY_STD_COL = 'I_for_(max_power_density): sample standard deviation [%]'
MAX_EFFICIENCY_MEAN_COL = 'max_efficiency: sample mean [%]'
MAX_EFFICIENCY_CONF_INT_COL = 'max_efficiency: half-length of 95% confidence interval for population mean [%]'
MAX_EFFICIENCY_STD_COL = 'max_efficiency: sample standard deviation [%]'
I_FOR_MAX_EFFICIENCY_MEAN_COL = 'I_for_(max_efficiency): sample mean [%]'
I_FOR_MAX_EFFICIENCY_CONF_INT_COL = 'I_for_(max_efficiency): ' \
                                    'half-length of 95% confidence interval for population mean [%]'
I_FOR_MAX_EFFICIENCY_STD_COL = 'I_for_(max_efficiency): sample standard deviation [%]'

COLUMNS = [INDEX_COL,
           MAX_ZT_MEAN_COL,
           MAX_ZT_CONF_INT_COL,
           MAX_ZT_STD_COL,
           T_FOR_MAX_ZT_MEAN_COL,
           T_FOR_MAX_ZT_CONF_INT_COL,
           T_FOR_MAX_ZT_STD_COL,
           MAX_POWER_DENSITY_MEAN_COL,
           MAX_POWER_DENSITY_CONF_INT_COL,
           MAX_POWER_DENSITY_STD_COL,
           I_FOR_MAX_POWER_DENSITY_MEAN_COL,
           I_FOR_MAX_POWER_DENSITY_CONF_INT_COL,
           I_FOR_MAX_POWER_DENSITY_STD_COL,
           MAX_EFFICIENCY_MEAN_COL,
           MAX_EFFICIENCY_CONF_INT_COL,
           MAX_EFFICIENCY_STD_COL,
           I_FOR_MAX_EFFICIENCY_MEAN_COL,
           I_FOR_MAX_EFFICIENCY_CONF_INT_COL,
           I_FOR_MAX_EFFICIENCY_STD_COL,
           ]

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
exact_df = pd.read_csv(EXACT_CSV_FILENAME, index_col=0)
cheb_df_dict = defaultdict(dict)
result_dict = defaultdict(dict)

for num_nodes in LIST_NUM_NODES:
    for noise_percent in LIST_NOISE_PERCENT:
        result_subdir = "noise_{}%".format(noise_percent)
        result_filename_wo_path = "performance_cheb_{}_nodes_with_{}%_noise.csv".format(num_nodes, noise_percent)
        cheb_csv_filename = os.path.join(PERFORMANCE_RESULT_DIR, result_subdir, result_filename_wo_path)
        cheb_df_dict[num_nodes][noise_percent] = pd.read_csv(cheb_csv_filename, index_col=0)


def get_confidence_interval(data_series):
    # 95% confidence interval, using the student t-distribution
    confidence_interval = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)
    return confidence_interval


def get_relative_error_df(exact_df_, approx_df):
    return (exact_df_ - approx_df).abs() / (exact_df_.abs())


def tabulate_performance(num_nodes_):
    result_df_ = pd.DataFrame(columns=COLUMNS).set_index(INDEX_COL)

    for noise_percent_ in LIST_NOISE_PERCENT:
        # DataFrame containing all the errors
        max_zT_error_df = get_relative_error_df(
            exact_df["max_zT"], cheb_df_dict[num_nodes_][noise_percent_]["max_zT"]
        ) * 100
        T_for_max_zT_error_df = get_relative_error_df(
            exact_df["T_for_(max_zT)"], cheb_df_dict[num_nodes_][noise_percent_]["T_for_(max_zT)"]
        ) * 100
        max_power_density_error_df = get_relative_error_df(
            exact_df["max_power_density"], cheb_df_dict[num_nodes_][noise_percent_]["max_power_density"]
        ) * 100
        I_for_max_power_density_error_df = get_relative_error_df(
            exact_df["I_for_(max_power_density)"], cheb_df_dict[num_nodes_][noise_percent_]["I_for_(max_power_density)"]
        ) * 100
        max_efficiency_error_df = get_relative_error_df(
            exact_df["max_efficiency"], cheb_df_dict[num_nodes_][noise_percent_]["max_efficiency"]
        ) * 100
        I_for_max_efficiency_error_df = get_relative_error_df(
            exact_df["I_for_(max_efficiency)"], cheb_df_dict[num_nodes_][noise_percent_]["I_for_(max_efficiency)"]
        ) * 100

        # sample mean
        result_df_.at[noise_percent_, MAX_ZT_MEAN_COL] = max_zT_error_df.mean()
        result_df_.at[noise_percent_, T_FOR_MAX_ZT_MEAN_COL] = T_for_max_zT_error_df.mean()
        result_df_.at[noise_percent_, MAX_POWER_DENSITY_MEAN_COL] = max_power_density_error_df.mean()
        result_df_.at[noise_percent_, I_FOR_MAX_POWER_DENSITY_MEAN_COL] = I_for_max_power_density_error_df.mean()
        result_df_.at[noise_percent_, MAX_EFFICIENCY_MEAN_COL] = max_efficiency_error_df.mean()
        result_df_.at[noise_percent_, I_FOR_MAX_EFFICIENCY_MEAN_COL] = I_for_max_efficiency_error_df.mean()

        # confidence interval
        result_df_.at[noise_percent_, MAX_ZT_CONF_INT_COL] = get_confidence_interval(max_zT_error_df)
        result_df_.at[noise_percent_, T_FOR_MAX_ZT_CONF_INT_COL] = get_confidence_interval(T_for_max_zT_error_df)
        result_df_.at[noise_percent_, MAX_POWER_DENSITY_CONF_INT_COL] = get_confidence_interval(max_power_density_error_df)
        result_df_.at[noise_percent_, I_FOR_MAX_POWER_DENSITY_CONF_INT_COL] = get_confidence_interval(I_for_max_power_density_error_df)
        result_df_.at[noise_percent_, MAX_EFFICIENCY_CONF_INT_COL] = get_confidence_interval(max_efficiency_error_df)
        result_df_.at[noise_percent_, I_FOR_MAX_EFFICIENCY_CONF_INT_COL] = get_confidence_interval(I_for_max_efficiency_error_df)

        # sample standard deviation
        result_df_.at[noise_percent_, MAX_ZT_STD_COL] = max_zT_error_df.std()
        result_df_.at[noise_percent_, T_FOR_MAX_ZT_STD_COL] = T_for_max_zT_error_df.std()
        result_df_.at[noise_percent_, MAX_POWER_DENSITY_STD_COL] = max_power_density_error_df.std()
        result_df_.at[noise_percent_, I_FOR_MAX_POWER_DENSITY_STD_COL] = I_for_max_power_density_error_df.std()
        result_df_.at[noise_percent_, MAX_EFFICIENCY_STD_COL] = max_efficiency_error_df.std()
        result_df_.at[noise_percent_, I_FOR_MAX_EFFICIENCY_STD_COL] = I_for_max_efficiency_error_df.std()

    return result_df_


for num_nodes in LIST_NUM_NODES:
    result_df = tabulate_performance(num_nodes)
    result_df.to_csv(os.path.join(SAVE_DIR, "table_performance_cheb_{}_nodes_with_noise.csv".format(num_nodes)))
