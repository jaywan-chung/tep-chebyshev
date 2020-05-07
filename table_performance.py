# -*- coding: utf-8 -*-
"""
Make Tables 6, 7 in Supplementary Information of the paper.

Before executing the script, the performance accuracy for each node should be computed
by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes.py".

Last updated on Apr 24 2020

@author: Jaywan Chung
"""

import os

import pandas as pd
from scipy.stats import sem, t

INFO_FILENAME = "data_info.csv"
PERFORMANCE_RESULT_DIR = os.path.join("results", "performance")
EXACT_CSV_FILENAME = os.path.join(PERFORMANCE_RESULT_DIR, "performance_exact.csv")
SAVE_DIR = os.path.join("results", "table")

START_NUM_NODES = 8
END_NUM_NODES = 16

INDEX_COL = '# of nodes'
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

cheb_csv_filename_dict = {}
result_df = pd.DataFrame(columns=COLUMNS).set_index(INDEX_COL)

for i in range(START_NUM_NODES, END_NUM_NODES+1):
    cheb_csv_filename_dict[i] = os.path.join(
        PERFORMANCE_RESULT_DIR, "performance_barycentric_cheb_{}_nodes.csv".format(i)
    )

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
exact_df = pd.read_csv(EXACT_CSV_FILENAME, index_col=0)
cheb_df_dict = {}
order_list = sorted(cheb_csv_filename_dict.keys())

for order in order_list:
    cheb_csv_filename = cheb_csv_filename_dict[order]
    cheb_df_dict[order] = pd.read_csv(cheb_csv_filename, index_col=0)


def get_confidence_interval(data_series):
    # 95% confidence interval, using the student t-distribution
    confidence_interval = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)
    return confidence_interval


def get_relative_error_df(exact_df_, approx_df):
    return (exact_df_ - approx_df).abs() / (exact_df_.abs())


for order in order_list:
    # DataFrame containing all the errors
    max_zT_error_df = get_relative_error_df(exact_df["max_zT"], cheb_df_dict[order]["max_zT"]) * 100
    T_for_max_zT_error_df = get_relative_error_df(
        exact_df["T_for_(max_zT)"], cheb_df_dict[order]["T_for_(max_zT)"]
    ) * 100
    max_power_density_error_df = get_relative_error_df(
        exact_df["max_power_density"], cheb_df_dict[order]["max_power_density"]
    ) * 100
    I_for_max_power_density_error_df = get_relative_error_df(
        exact_df["I_for_(max_power_density)"], cheb_df_dict[order]["I_for_(max_power_density)"]
    ) * 100
    max_efficiency_error_df = get_relative_error_df(
        exact_df["max_efficiency"], cheb_df_dict[order]["max_efficiency"]
    ) * 100
    I_for_max_efficiency_error_df = get_relative_error_df(
        exact_df["I_for_(max_efficiency)"], cheb_df_dict[order]["I_for_(max_efficiency)"]
    ) * 100

    # sample mean
    result_df.at[order, MAX_ZT_MEAN_COL] = max_zT_error_df.mean()
    result_df.at[order, T_FOR_MAX_ZT_MEAN_COL] = T_for_max_zT_error_df.mean()
    result_df.at[order, MAX_POWER_DENSITY_MEAN_COL] = max_power_density_error_df.mean()
    result_df.at[order, I_FOR_MAX_POWER_DENSITY_MEAN_COL] = I_for_max_power_density_error_df.mean()
    result_df.at[order, MAX_EFFICIENCY_MEAN_COL] = max_efficiency_error_df.mean()
    result_df.at[order, I_FOR_MAX_EFFICIENCY_MEAN_COL] = I_for_max_efficiency_error_df.mean()

    # confidence interval
    result_df.at[order, MAX_ZT_CONF_INT_COL] = get_confidence_interval(max_zT_error_df)
    result_df.at[order, T_FOR_MAX_ZT_CONF_INT_COL] = get_confidence_interval(T_for_max_zT_error_df)
    result_df.at[order, MAX_POWER_DENSITY_CONF_INT_COL] = get_confidence_interval(max_power_density_error_df)
    result_df.at[order, I_FOR_MAX_POWER_DENSITY_CONF_INT_COL] = get_confidence_interval(I_for_max_power_density_error_df)
    result_df.at[order, MAX_EFFICIENCY_CONF_INT_COL] = get_confidence_interval(max_efficiency_error_df)
    result_df.at[order, I_FOR_MAX_EFFICIENCY_CONF_INT_COL] = get_confidence_interval(I_for_max_efficiency_error_df)

    # sample standard deviation
    result_df.at[order, MAX_ZT_STD_COL] = max_zT_error_df.std()
    result_df.at[order, T_FOR_MAX_ZT_STD_COL] = T_for_max_zT_error_df.std()
    result_df.at[order, MAX_POWER_DENSITY_STD_COL] = max_power_density_error_df.std()
    result_df.at[order, I_FOR_MAX_POWER_DENSITY_STD_COL] = I_for_max_power_density_error_df.std()
    result_df.at[order, MAX_EFFICIENCY_STD_COL] = max_efficiency_error_df.std()
    result_df.at[order, I_FOR_MAX_EFFICIENCY_STD_COL] = I_for_max_efficiency_error_df.std()

result_df.to_csv(os.path.join(SAVE_DIR, "table_performance.csv"))