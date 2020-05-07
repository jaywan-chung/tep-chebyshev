# -*- coding: utf-8 -*-
"""
Plot Figure 6 of the paper.

Before executing the script, the performance accuracy for each node under noise should be computed
by using "performance_exact.py" and "performance_barycentric_cheb_n_nodes_with_noise.py".

Last updated on Apr 29 2020

@author: Jaywan Chung
"""

from collections import defaultdict
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem, t

# matplotlib settings
font = {'size': 20}
matplotlib.rc('font', **font)
MARKERSIZE = 15
CAPSIZE = 5
LIST_COLOR_MAX_ITEM = [(255 / 255, 153 / 255, 153 / 255),
                       (153 / 255, 51 / 255, 255 / 255),
                       (51 / 255, 204 / 255, 51 / 255),
                       ]  # light pink, purple, green

LIST_NUM_NODES = [8, 11, 14]
LIST_NOISE_PERCENT = list(range(0, 10 + 1))

INFO_FILENAME = "data_info.csv"
EXACT_CSV_FILENAME = os.path.join("results", "performance", "performance_exact.csv")
RESULT_DIR = os.path.join("results", "performance_noise")

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
exact_df = pd.read_csv(EXACT_CSV_FILENAME, index_col=0)
cheb_df_dict = defaultdict(dict)
result_dict = defaultdict(dict)

for num_nodes in LIST_NUM_NODES:
    for noise_percent in LIST_NOISE_PERCENT:
        result_subdir = "noise_{}%".format(noise_percent)
        result_filename_wo_path = "performance_cheb_{}_nodes_with_{}%_noise.csv".format(num_nodes, noise_percent)
        cheb_csv_filename = os.path.join(RESULT_DIR, result_subdir, result_filename_wo_path)
        cheb_df_dict[num_nodes][noise_percent] = pd.read_csv(cheb_csv_filename, index_col=0)


def get_confidence_interval(data_series):
    # 95% confidence interval, using the student t-distribution
    confidence_interval = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)
    return confidence_interval


def get_relative_error_df(exact_df_, approx_df):
    return (exact_df_ - approx_df).abs() / (exact_df_.abs())


class ErrorContainer:
    def __init__(self, num_nodes_, noise_percent_):
        self.num_nodes = num_nodes_
        self.noise_percent = noise_percent_

        self.max_zT_error_df = get_relative_error_df(
            exact_df["max_zT"], cheb_df_dict[num_nodes_][noise_percent_]["max_zT"]
        ) * 100
        self.max_zT_mean = self.max_zT_error_df.mean()
        self.max_zT_conf_int = get_confidence_interval(self.max_zT_error_df)

        self.T_for_max_zT_error_df = get_relative_error_df(
            exact_df["T_for_(max_zT)"], cheb_df_dict[num_nodes_][noise_percent_]["T_for_(max_zT)"]
        ) * 100
        self.T_for_max_zT_mean = self.T_for_max_zT_error_df.mean()
        self.T_for_max_zT_conf_int = get_confidence_interval(self.T_for_max_zT_error_df)

        self.max_power_density_error_df = get_relative_error_df(
            exact_df["max_power_density"], cheb_df_dict[num_nodes_][noise_percent_]["max_power_density"]
        ) * 100
        self.max_power_density_mean = self.max_power_density_error_df.mean()
        self.max_power_density_conf_int = get_confidence_interval(self.max_power_density_error_df)

        self.I_for_max_power_density_error_df = get_relative_error_df(
            exact_df["I_for_(max_power_density)"],
            cheb_df_dict[num_nodes_][noise_percent_]["I_for_(max_power_density)"]
        ) * 100
        self.I_for_max_power_density_mean = self.I_for_max_power_density_error_df.mean()
        self.I_for_max_power_density_conf_int = get_confidence_interval(self.I_for_max_power_density_error_df)

        self.max_efficiency_error_df = get_relative_error_df(
            exact_df["max_efficiency"], cheb_df_dict[num_nodes_][noise_percent_]["max_efficiency"]
        ) * 100
        self.max_efficiency_mean = self.max_efficiency_error_df.mean()
        self.max_efficiency_conf_int = get_confidence_interval(self.max_efficiency_error_df)

        self.I_for_max_efficiency_error_df = get_relative_error_df(
            exact_df["I_for_(max_efficiency)"], cheb_df_dict[num_nodes_][noise_percent_]["I_for_(max_efficiency)"]
        ) * 100
        self.I_for_max_efficiency_mean = self.I_for_max_efficiency_error_df.mean()
        self.I_for_max_efficiency_conf_int = get_confidence_interval(self.I_for_max_efficiency_error_df)


max_zT_mean_dict = defaultdict(list)
max_zT_conf_int_dict = defaultdict(list)
max_power_density_mean_dict = defaultdict(list)
max_power_density_conf_int_dict = defaultdict(list)
max_efficiency_mean_dict = defaultdict(list)
max_efficiency_conf_int_dict = defaultdict(list)

T_for_max_zT_mean_dict = defaultdict(list)
T_for_max_zT_conf_int_dict = defaultdict(list)
I_for_max_power_density_mean_dict = defaultdict(list)
I_for_max_power_density_conf_int_dict = defaultdict(list)
I_for_max_efficiency_mean_dict = defaultdict(list)
I_for_max_efficiency_conf_int_dict = defaultdict(list)

for num_nodes in LIST_NUM_NODES:
    for noise_percent in LIST_NOISE_PERCENT:
        res = ErrorContainer(num_nodes, noise_percent)
        result_dict[num_nodes][noise_percent] = res

        max_zT_mean_dict[num_nodes].append(res.max_zT_mean)
        max_zT_conf_int_dict[num_nodes].append(res.max_zT_conf_int)

        max_power_density_mean_dict[num_nodes].append(res.max_power_density_mean)
        max_power_density_conf_int_dict[num_nodes].append(res.max_power_density_conf_int)

        max_efficiency_mean_dict[num_nodes].append(res.max_efficiency_mean)
        max_efficiency_conf_int_dict[num_nodes].append(res.max_efficiency_conf_int)

        T_for_max_zT_mean_dict[num_nodes].append(res.T_for_max_zT_mean)
        T_for_max_zT_conf_int_dict[num_nodes].append(res.T_for_max_zT_conf_int)

        I_for_max_power_density_mean_dict[num_nodes].append(res.I_for_max_power_density_mean)
        I_for_max_power_density_conf_int_dict[num_nodes].append(res.I_for_max_power_density_conf_int)

        I_for_max_efficiency_mean_dict[num_nodes].append(res.I_for_max_efficiency_mean)
        I_for_max_efficiency_conf_int_dict[num_nodes].append(res.I_for_max_efficiency_conf_int)

plt.figure(figsize=(21, 14))
ax1 = plt.subplot(231)
ax1.grid(True)
ax1.set_title("max. power density")
ax2 = plt.subplot(232, sharex=ax1, sharey=ax1)
ax2.grid(True)
ax2.set_title("max. efficiency")
ax3 = plt.subplot(233, sharex=ax1)
ax3.grid(True)
ax3.set_title(r"max. $zT$")

ax4 = plt.subplot(234)
ax4.grid(True)
ax4.set_title("$I$ for max. power density")
ax5 = plt.subplot(235, sharex=ax4, sharey=ax4)
ax5.grid(True)
ax5.set_title("$I$ for max. efficiency")
ax6 = plt.subplot(236, sharex=ax4)
ax6.grid(True)
ax6.set_title(r"$T$ for max. $zT$")

for i, num_nodes in enumerate(LIST_NUM_NODES):
    ax1.plot(LIST_NOISE_PERCENT, max_power_density_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=3)
    ax1.plot(LIST_NOISE_PERCENT, max_power_density_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax1.errorbar(LIST_NOISE_PERCENT, max_power_density_mean_dict[num_nodes],
                 yerr=max_power_density_conf_int_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE,
                 linestyle='None', linewidth=3, capthick=3)
    ax1.set_xticks(LIST_NOISE_PERCENT)
    ax1.axhline(y=0, color='k', linestyle=':')
    ax1.set_xlabel("Noise [%]")
    ax1.set_ylabel("Mean Relative Error [%]")

    ax2.plot(LIST_NOISE_PERCENT, max_efficiency_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=3)
    ax2.plot(LIST_NOISE_PERCENT, max_efficiency_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax2.errorbar(LIST_NOISE_PERCENT, max_efficiency_mean_dict[num_nodes], yerr=max_efficiency_conf_int_dict[num_nodes],
                 color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE, linestyle='None', linewidth=3, capthick=3)
    ax2.set_xticks(LIST_NOISE_PERCENT)
    ax2.axhline(y=0, color='k', linestyle=':')
    ax2.set_xlabel("Noise [%]")
    ax2.set_ylabel("Mean Relative Error [%]")

    ax3.plot(LIST_NOISE_PERCENT, max_zT_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=3)
    ax3.plot(LIST_NOISE_PERCENT, max_zT_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax3.errorbar(LIST_NOISE_PERCENT, max_zT_mean_dict[num_nodes], yerr=max_zT_conf_int_dict[num_nodes],
                 color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE, linestyle='None', linewidth=3, capthick=3)
    ax3.set_xticks(LIST_NOISE_PERCENT)
    ax3.axhline(y=0, color='k', linestyle=':')
    ax3.set_xlabel("Noise [%]")
    ax3.set_ylabel("Mean Relative Error [%]")

    ax4.plot(LIST_NOISE_PERCENT, I_for_max_power_density_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=3)
    ax4.plot(LIST_NOISE_PERCENT, I_for_max_power_density_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax4.errorbar(LIST_NOISE_PERCENT, I_for_max_power_density_mean_dict[num_nodes],
                 yerr=I_for_max_power_density_conf_int_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE,
                 linestyle='None', linewidth=3, capthick=3)
    ax4.set_xticks(LIST_NOISE_PERCENT)
    ax4.axhline(y=0, color='k', linestyle=':')
    ax4.set_xlabel("Noise [%]")
    ax4.set_ylabel("Mean Relative Error [%]")

    ax5.plot(LIST_NOISE_PERCENT, I_for_max_efficiency_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=3)
    ax5.plot(LIST_NOISE_PERCENT, I_for_max_efficiency_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax5.errorbar(LIST_NOISE_PERCENT, I_for_max_efficiency_mean_dict[num_nodes],
                 yerr=I_for_max_efficiency_conf_int_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE,
                 linestyle='None', linewidth=3, capthick=3)
    ax5.set_xticks(LIST_NOISE_PERCENT)
    ax5.axhline(y=0, color='k', linestyle=':')
    ax5.set_xlabel("Noise [%]")
    ax5.set_ylabel("Mean Relative Error [%]")

    ax6.plot(LIST_NOISE_PERCENT, T_for_max_zT_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=3)
    ax6.plot(LIST_NOISE_PERCENT, T_for_max_zT_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax6.errorbar(LIST_NOISE_PERCENT, T_for_max_zT_mean_dict[num_nodes], yerr=T_for_max_zT_conf_int_dict[num_nodes],
                 color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE, linestyle='None', linewidth=3, capthick=3)
    ax6.set_xticks(LIST_NOISE_PERCENT)
    ax6.axhline(y=0, color='k', linestyle=':')
    ax6.set_xlabel("Noise [%]")
    ax6.set_ylabel("Mean Relative Error [%]")

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()
plt.tight_layout()
plt.show()
