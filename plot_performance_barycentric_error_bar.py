# -*- coding: utf-8 -*-
"""
Plot Figure 4 of the paper.

Before executing the script, the performance accuracy for each node should be computed
by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes.py".

Last updated on Apr 29 2020

@author: Jaywan Chung
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
from scipy.stats import t

# matplotlib settings
font = {'size': 20}
matplotlib.rc('font', **font)
MARKERSIZE = 15
CAPSIZE = 5
ITEM_MAX_COLOR = (255/255, 140/255, 0/255)  # dark orange
ITEM_COR_COLOR = (65/255, 105/255, 225/255)  # royal blue

START_NUM_NODES = 8
END_NUM_NODES = 16

INFO_FILENAME = "data_info.csv"
RESULT_DIR = os.path.join("results", "performance")
EXACT_CSV_FILENAME = os.path.join(RESULT_DIR, "performance_exact.csv")

cheb_csv_filename_dict = {}
result_dict = {}

for i in range(START_NUM_NODES, END_NUM_NODES+1):
    cheb_csv_filename_dict[i] = os.path.join(RESULT_DIR, "performance_barycentric_cheb_{}_nodes.csv".format(i))

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


class ErrorContainer:
    def __init__(self, order_):
        self.order = order_

        self.max_zT_error_df = get_relative_error_df(exact_df["max_zT"], cheb_df_dict[order_]["max_zT"]) * 100
        self.max_zT_mean = self.max_zT_error_df.mean()
        self.max_zT_conf_int = get_confidence_interval(self.max_zT_error_df)

        self.T_for_max_zT_error_df = get_relative_error_df(exact_df["T_for_(max_zT)"],
                                                           cheb_df_dict[order_]["T_for_(max_zT)"]) * 100
        self.T_for_max_zT_mean = self.T_for_max_zT_error_df.mean()
        self.T_for_max_zT_conf_int = get_confidence_interval(self.T_for_max_zT_error_df)

        self.max_power_density_error_df = get_relative_error_df(exact_df["max_power_density"],
                                                                cheb_df_dict[order_]["max_power_density"]) * 100
        self.max_power_density_mean = self.max_power_density_error_df.mean()
        self.max_power_density_conf_int = get_confidence_interval(self.max_power_density_error_df)

        self.I_for_max_power_density_error_df = get_relative_error_df(
            exact_df["I_for_(max_power_density)"], cheb_df_dict[order_]["I_for_(max_power_density)"]) * 100
        self.I_for_max_power_density_mean = self.I_for_max_power_density_error_df.mean()
        self.I_for_max_power_density_conf_int = get_confidence_interval(self.I_for_max_power_density_error_df)

        self.max_efficiency_error_df = get_relative_error_df(exact_df["max_efficiency"],
                                                             cheb_df_dict[order_]["max_efficiency"]) * 100
        self.max_efficiency_mean = self.max_efficiency_error_df.mean()
        self.max_efficiency_conf_int = get_confidence_interval(self.max_efficiency_error_df)

        self.I_for_max_efficiency_error_df = get_relative_error_df(exact_df["I_for_(max_efficiency)"],
                                                                   cheb_df_dict[order_]["I_for_(max_efficiency)"]) * 100
        self.I_for_max_efficiency_mean = self.I_for_max_efficiency_error_df.mean()
        self.I_for_max_efficiency_conf_int = get_confidence_interval(self.I_for_max_efficiency_error_df)


max_zT_mean_list = []
max_zT_conf_int_list = []
T_for_max_zT_mean_list = []
T_for_max_zT_conf_int_list = []
max_power_density_mean_list = []
max_power_density_conf_int_list = []
I_for_max_power_density_mean_list = []
I_for_max_power_density_conf_int_list = []
max_efficiency_mean_list = []
max_efficiency_conf_int_list = []
I_for_max_efficiency_mean_list = []
I_for_max_efficiency_conf_int_list = []

for order in order_list:
    res = ErrorContainer(order)
    result_dict[order] = res

    max_zT_mean_list.append(res.max_zT_mean)
    max_zT_conf_int_list.append(res.max_zT_conf_int)

    T_for_max_zT_mean_list.append(res.T_for_max_zT_mean)
    T_for_max_zT_conf_int_list.append(res.T_for_max_zT_conf_int)

    max_power_density_mean_list.append(res.max_power_density_mean)
    max_power_density_conf_int_list.append(res.max_power_density_conf_int)

    I_for_max_power_density_mean_list.append(res.I_for_max_power_density_mean)
    I_for_max_power_density_conf_int_list.append(res.I_for_max_power_density_conf_int)

    max_efficiency_mean_list.append(res.max_efficiency_mean)
    max_efficiency_conf_int_list.append(res.max_efficiency_conf_int)

    I_for_max_efficiency_mean_list.append(res.I_for_max_efficiency_mean)
    I_for_max_efficiency_conf_int_list.append(res.I_for_max_efficiency_conf_int)

plt.figure(figsize=(21, 7))

ax1 = plt.subplot(131)
plt.grid(True)
plt.plot(order_list, max_power_density_mean_list, color=ITEM_MAX_COLOR, label="max. power density", linewidth=5)
plt.plot(
    order_list, max_power_density_mean_list,
    color=ITEM_MAX_COLOR, marker='.', markersize=MARKERSIZE, linestyle='None'
)
plt.errorbar(
    order_list, max_power_density_mean_list, yerr=max_power_density_conf_int_list,
    color=ITEM_MAX_COLOR, capsize=CAPSIZE, linestyle='None', linewidth=3, capthick=3
)

plt.plot(
    order_list, I_for_max_power_density_mean_list,
    color=ITEM_COR_COLOR, label=r"corresponding $I$", linestyle='--'
)
plt.plot(
    order_list, I_for_max_power_density_mean_list,
    color=ITEM_COR_COLOR, marker='.', markersize=MARKERSIZE, linestyle='None'
)
plt.errorbar(
    order_list, I_for_max_power_density_mean_list, yerr=I_for_max_power_density_conf_int_list,
    color=ITEM_COR_COLOR, capsize=CAPSIZE, linestyle='None'
)
plt.xticks(order_list)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel("Number of Nodes")
plt.ylabel("Mean Relative Error [%]")
plt.legend()

plt.subplot(132, sharex=ax1, sharey=ax1)
plt.grid(True)
plt.plot(order_list, max_efficiency_mean_list, color=ITEM_MAX_COLOR, label=r"max. efficiency", linewidth=5)
plt.plot(
    order_list, max_efficiency_mean_list,
    color=ITEM_MAX_COLOR, marker='.', markersize=MARKERSIZE, linestyle='None'
)
plt.errorbar(
    order_list, max_efficiency_mean_list, yerr=max_efficiency_conf_int_list,
    color=ITEM_MAX_COLOR, capsize=CAPSIZE, linestyle='None', linewidth=3, capthick=3
)

plt.plot(order_list, I_for_max_efficiency_mean_list, color=ITEM_COR_COLOR, label=r"corresponding $I$", linestyle='--')
plt.plot(
    order_list, I_for_max_efficiency_mean_list,
    color=ITEM_COR_COLOR, marker='.', markersize=MARKERSIZE, linestyle='None'
)
plt.errorbar(
    order_list, I_for_max_efficiency_mean_list, yerr=I_for_max_efficiency_conf_int_list,
    color=ITEM_COR_COLOR, capsize=CAPSIZE, linestyle='None'
)
plt.xticks(order_list)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel("Number of Nodes")
plt.ylabel("Mean Relative Error [%]")
plt.legend()

ax1 = plt.subplot(133, sharex=ax1)
plt.grid(True)
plt.plot(order_list, max_zT_mean_list, color=ITEM_MAX_COLOR, label=r"max. $zT$", linewidth=5)
plt.plot(order_list, max_zT_mean_list, color=ITEM_MAX_COLOR, marker='.', markersize=MARKERSIZE, linestyle='None')
plt.errorbar(
    order_list, max_zT_mean_list, yerr=max_zT_conf_int_list,
    color=ITEM_MAX_COLOR, capsize=CAPSIZE, linestyle='None', linewidth=3, capthick=3
)
plt.plot(order_list, T_for_max_zT_mean_list, color=ITEM_COR_COLOR, label=r"corresponding $T$", linestyle='--')
plt.plot(order_list, T_for_max_zT_mean_list, color=ITEM_COR_COLOR, marker='.', markersize=MARKERSIZE, linestyle='None')
plt.errorbar(
    order_list, T_for_max_zT_mean_list, yerr=T_for_max_zT_conf_int_list,
    color=ITEM_COR_COLOR, capsize=CAPSIZE, linestyle='None'
)
plt.xticks(order_list)
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel("Number of Nodes")
plt.ylabel("Mean Relative Error [%]")
plt.legend()

plt.tight_layout()
plt.show()
