# -*- coding: utf-8 -*-
"""
Plot Figure 5 of the paper.

Before executing the script, the interpolation accuracy for each node under noise should be computed
by using "accuracy_barycentric_cheb_n_nodes_with_noise.py".

Last updated on Apr 29 2020

@author: Jaywan Chung
"""

from collections import defaultdict
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
LINEWIDTH = 3
LIST_COLOR_MAX_ITEM = [(255 / 255, 153 / 255, 153 / 255),
                       (153 / 255, 51 / 255, 255 / 255),
                       (51 / 255, 204 / 255, 51 / 255),
                       ]  # light pink, purple, green

LIST_NUM_NODES = [8, 11, 14]
LIST_NOISE_PERCENT = list(range(0, 10 + 1))

INFO_FILENAME = "data_info.csv"
RESULT_DIR = os.path.join("results", "accuracy_noise")

info_df = pd.read_csv(INFO_FILENAME, index_col=0)
cheb_df_dict = defaultdict(dict)
result_dict = defaultdict(dict)

for num_nodes in LIST_NUM_NODES:
    for noise_percent in LIST_NOISE_PERCENT:
        result_subdir = "noise_{}%".format(noise_percent)
        result_filename_wo_path = "accuracy_cheb_{}_nodes_with_{}%_noise.csv".format(num_nodes, noise_percent)
        cheb_csv_filename = os.path.join(RESULT_DIR, result_subdir, result_filename_wo_path)
        cheb_df_dict[num_nodes][noise_percent] = pd.read_csv(cheb_csv_filename, index_col=0)


def get_confidence_interval(data_series):
    # 95% confidence interval, using the student t-distribution
    confidence_interval = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)
    return confidence_interval


class ErrorContainer:
    def __init__(self, num_nodes_, noise_percent_):
        self.num_nodes = num_nodes_
        self.noise_percent = noise_percent_

        self.cheb_Seebeck_error_df = cheb_df_dict[num_nodes_][noise_percent_]["Seebeck_RMAE"] * 100
        self.cheb_Seebeck_mean = self.cheb_Seebeck_error_df.mean()
        self.cheb_Seebeck_conf_int = get_confidence_interval(self.cheb_Seebeck_error_df)

        self.cheb_elec_resi_error_df = cheb_df_dict[num_nodes_][noise_percent_]["elec_resi_RMAE"] * 100
        self.cheb_elec_resi_mean = self.cheb_elec_resi_error_df.mean()
        self.cheb_elec_resi_conf_int = get_confidence_interval(self.cheb_elec_resi_error_df)

        self.cheb_thrm_cond_error_df = cheb_df_dict[num_nodes_][noise_percent_]["thrm_cond_RMAE"] * 100
        self.cheb_thrm_cond_mean = self.cheb_thrm_cond_error_df.mean()
        self.cheb_thrm_cond_conf_int = get_confidence_interval(self.cheb_thrm_cond_error_df)


cheb_Seebeck_mean_dict = defaultdict(list)
cheb_Seebeck_conf_int_dict = defaultdict(list)
cheb_elec_resi_mean_dict = defaultdict(list)
cheb_elec_resi_conf_int_dict = defaultdict(list)
cheb_thrm_cond_mean_dict = defaultdict(list)
cheb_thrm_cond_conf_int_dict = defaultdict(list)

for num_nodes in LIST_NUM_NODES:
    for noise_percent in LIST_NOISE_PERCENT:
        res = ErrorContainer(num_nodes, noise_percent)
        result_dict[num_nodes][noise_percent] = res

        cheb_Seebeck_mean_dict[num_nodes].append(res.cheb_Seebeck_mean)
        cheb_Seebeck_conf_int_dict[num_nodes].append(res.cheb_Seebeck_conf_int)

        cheb_elec_resi_mean_dict[num_nodes].append(res.cheb_elec_resi_mean)
        cheb_elec_resi_conf_int_dict[num_nodes].append(res.cheb_elec_resi_conf_int)

        cheb_thrm_cond_mean_dict[num_nodes].append(res.cheb_thrm_cond_mean)
        cheb_thrm_cond_conf_int_dict[num_nodes].append(res.cheb_thrm_cond_conf_int)

plt.figure(figsize=(21, 7))
ax1 = plt.subplot(131)
ax1.grid(True)
ax1.set_title(r"Seebeck coefficient")
ax2 = plt.subplot(132, sharex=ax1)
ax2.grid(True)
ax2.set_title(r"Electrical resistivity")
ax3 = plt.subplot(133, sharex=ax1, sharey=ax2)
ax3.grid(True)
ax3.set_title(r"Thermal conductivity")

for i, num_nodes in enumerate(LIST_NUM_NODES):
    ax1.plot(LIST_NOISE_PERCENT, cheb_Seebeck_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=LINEWIDTH)
    ax1.plot(LIST_NOISE_PERCENT, cheb_Seebeck_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax1.errorbar(LIST_NOISE_PERCENT, cheb_Seebeck_mean_dict[num_nodes], yerr=cheb_Seebeck_conf_int_dict[num_nodes],
                 color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE, linestyle='None', linewidth=LINEWIDTH,
                 capthick=LINEWIDTH)
    ax1.set_xticks(LIST_NOISE_PERCENT)
    ax1.axhline(y=0, color='k', linestyle=':')
    ax1.set_xlabel("Noise [%]")
    ax1.set_ylabel(r"Mean Relative $L^\infty$-Norm Error [%]")

    ax2.plot(LIST_NOISE_PERCENT, cheb_elec_resi_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=LINEWIDTH)
    ax2.plot(LIST_NOISE_PERCENT, cheb_elec_resi_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax2.errorbar(LIST_NOISE_PERCENT, cheb_elec_resi_mean_dict[num_nodes], yerr=cheb_elec_resi_conf_int_dict[num_nodes],
                 color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE, linestyle='None', linewidth=LINEWIDTH,
                 capthick=LINEWIDTH)
    ax2.set_xticks(LIST_NOISE_PERCENT)
    ax2.axhline(y=0, color='k', linestyle=':')
    ax2.set_xlabel("Noise [%]")
    ax2.set_ylabel(r"Mean Relative $L^\infty$-Norm Error [%]")

    ax3.plot(LIST_NOISE_PERCENT, cheb_thrm_cond_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i],
             label="{} nodes".format(num_nodes), linewidth=LINEWIDTH)
    ax3.plot(LIST_NOISE_PERCENT, cheb_thrm_cond_mean_dict[num_nodes], color=LIST_COLOR_MAX_ITEM[i], marker='.',
             markersize=MARKERSIZE, linestyle='None')
    ax3.errorbar(LIST_NOISE_PERCENT, cheb_thrm_cond_mean_dict[num_nodes], yerr=cheb_thrm_cond_conf_int_dict[num_nodes],
                 color=LIST_COLOR_MAX_ITEM[i], capsize=CAPSIZE, linestyle='None', linewidth=LINEWIDTH,
                 capthick=LINEWIDTH)
    ax3.set_xticks(LIST_NOISE_PERCENT)
    ax3.axhline(y=0, color='k', linestyle=':')
    ax3.set_xlabel("Noise [%]")
    ax3.set_ylabel(r"Mean Relative $L^\infty$-Norm Error [%]")

ax1.legend()
ax2.legend()
ax3.legend()
plt.tight_layout()
plt.show()
