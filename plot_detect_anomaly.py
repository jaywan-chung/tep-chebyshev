# -*- coding: utf-8 -*-
"""
Plot Figure 8(a)-(c) of the paper.

Detect anomalous temperature points by inspecting large interpolation errors using a small number of Chebyshev nodes.

Last updated on July 09 2020

@author: Jaywan Chung
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from libs.pykeri.TEProp import TEProp
from utils import interpolate_barycentric_cheb
from utils import BarycentricLagrangeChebyshevNodes


font = {'size': 20}
matplotlib.rc('font', **font)

INFO_FILENAME = "data_info.csv"
DB_FILENAME = "tep_20180409.db"

info_df = pd.read_csv(INFO_FILENAME, index_col=0)

IS_DRAW_GRAPH = True
IS_SAVE_FIG = True
IS_DRAW_TITLE = False

NUM_CHEB_NODES = 7
NUM_MIN_CHEB_NODES = 3
NUM_MAX_DIVISION = 3

xticks = np.linspace(300, 900, 7, dtype=int)

# LIST_ID_NUM = info_df.index

#### Figure 8(a)
LIST_ID_NUM = [156]  # Figure 8(a)
NUM_MAX_DIVISION = 2  # id=156, Ag2Te, phase transition
xticks = np.linspace(300, 600, 7)

#### Figure 8(b)
# LIST_ID_NUM = [294]  # Figure 8(b)
# NUM_MAX_DIVISION = 3  # id=294, Cu2Se, continuous phase transition
# xticks = np.linspace(300, 450, 7)

#### Figure 8(c)
# LIST_ID_NUM = [27]  # Figure 8(c)
# NUM_MAX_DIVISION = 3  # id=27, SnSe, electron activation
# xticks = np.linspace(300, 900, 7, dtype=int)

NUM_TOTAL_DIVIDED_CHEB_NODES = 7 * (NUM_MAX_DIVISION + 1)

TEMP_RESOLUTION_RATIO = 5 / 100  # [K]
GAUSSIAN_SIGMA_RATIO = 1 / 100

FOLDER_NAME = "{}_to_sub{}_div{}".format(
    NUM_CHEB_NODES, NUM_TOTAL_DIVIDED_CHEB_NODES, NUM_MAX_DIVISION)

RESULT_DIR = ""

SEEBECK_SPLINE_ORDER = 1
ELEC_RESI_SPLINE_ORDER = 1
THRM_COND_SPLINE_ORDER = 1

# plot settings
EXACT_LINE_COLOR = 'silver'
EXACT_LINE_LINEWIDTH = 7
EXACT_LINE_ALPHA = 1.0
EXACT_LINE_LINESTYLE = '-'

ONE_POLY_LINE_COLOR = 'darkblue'
ONE_POLY_LINE_LINEWIDTH = 4
ONE_POLY_LINE_ALPHA = 1.0
ONE_POLY_LINE_LINESTYLE = '--'

VERTICAL_FROM_ERR_LINE_COLOR = 'k'
VERTICAL_FROM_ERR_LINE_LINEWIDTH = 2
VERTICAL_FROM_ERR_LINE_ALPHA = 1.0
VERTICAL_FROM_ERR_LINE_LINESTYLE = ':'

VERTICAL_FROM_DERIV_ERR_LINE_COLOR = 'k'
VERTICAL_FROM_DERIV_ERR_LINE_LINEWIDTH = 3
VERTICAL_FROM_DERIV_ERR_LINE_ALPHA = 0.5
VERTICAL_FROM_DERIV_ERR_LINE_LINESTYLE = '--'

print("{} nodes => divided {} nodes.".format(NUM_CHEB_NODES, NUM_TOTAL_DIVIDED_CHEB_NODES))
print("max. num of division={}.".format(NUM_MAX_DIVISION))


def find_rel_error(x_array, interp_func, exact_func):
    abs_err = np.abs(interp_func(x_array) - exact_func(x_array))
    rel_err = abs_err / np.max(np.abs(exact_func(x_array)))
    return rel_err


def find_rel_error_peaks(x_array, interp_func, exact_func):
    rel_err = find_rel_error(x_array, interp_func, exact_func)
    peak_indices, _ = find_peaks(rel_err)
    peak_rel_errs = rel_err[peak_indices]
    return peak_indices, peak_rel_errs


def interpolate_barycentric_cheb_with_exact_func(xl, xr, num_cheb_nodes, exact_func):
    cheb_nodes = (xl - xr) / 2 * np.cos(
        np.pi * np.linspace(0, num_cheb_nodes - 1, num_cheb_nodes) / (num_cheb_nodes - 1)
    ) + (xl + xr) / 2
    cheb_interp_func = BarycentricLagrangeChebyshevNodes(cheb_nodes, exact_func(cheb_nodes))
    return cheb_interp_func, cheb_nodes


def round_up_to_odd(value):
    return np.ceil(value) // 2 * 2 + 1


def divide_teps(
        seebeck_exact_func, elec_resi_exact_func, thrm_cond_exact_func,
        seebeck_interval, elec_resi_interval, thrm_cond_interval,
        division_point, num_cheb_nodes_left, num_cheb_node_right):
    # divide into two regions
    seebeck_interp_func_left, seebeck_cheb_nodes_left = interpolate_barycentric_cheb_with_exact_func(
        seebeck_interval[0], division_point, num_cheb_nodes_left, seebeck_exact_func
    )
    seebeck_interp_func_right, seebeck_cheb_nodes_right = interpolate_barycentric_cheb_with_exact_func(
        division_point, seebeck_interval[1], num_cheb_node_right, seebeck_exact_func
    )
    elec_resi_interp_func_left, elec_resi_cheb_nodes_left = interpolate_barycentric_cheb_with_exact_func(
        elec_resi_interval[0], division_point, num_cheb_nodes_left, elec_resi_exact_func
    )
    elec_resi_interp_func_right, elec_resi_cheb_nodes_right = interpolate_barycentric_cheb_with_exact_func(
        division_point, elec_resi_interval[1], num_cheb_node_right, elec_resi_exact_func
    )
    thrm_cond_interp_func_left, thrm_cond_cheb_nodes_left = interpolate_barycentric_cheb_with_exact_func(
        thrm_cond_interval[0], division_point, num_cheb_nodes_left, thrm_cond_exact_func
    )
    thrm_cond_interp_func_right, thrm_cond_cheb_nodes_right = interpolate_barycentric_cheb_with_exact_func(
        division_point, thrm_cond_interval[1], num_cheb_node_right, thrm_cond_exact_func
    )
    seebeck_interval_left = (seebeck_interval[0], division_point)
    seebeck_interval_right = (division_point, seebeck_interval[1])
    elec_resi_interval_left = (elec_resi_interval[0], division_point)
    elec_resi_interval_right = (division_point, elec_resi_interval[1])
    thrm_cond_interval_left = (thrm_cond_interval[0], division_point)
    thrm_cond_interval_right = (division_point, thrm_cond_interval[1])

    return seebeck_interp_func_left, seebeck_interp_func_right, \
           elec_resi_interp_func_left, elec_resi_interp_func_right, \
           thrm_cond_interp_func_left, thrm_cond_interp_func_right, \
           seebeck_interval_left, seebeck_interval_right, \
           elec_resi_interval_left, elec_resi_interval_right, \
           thrm_cond_interval_left, thrm_cond_interval_right, \
           seebeck_cheb_nodes_left, seebeck_cheb_nodes_right, \
           elec_resi_cheb_nodes_left, elec_resi_cheb_nodes_right, \
           thrm_cond_cheb_nodes_left, thrm_cond_cheb_nodes_right


for id_num in LIST_ID_NUM:
    TITLE = "id={}, mat={}, {}_to_divided {}, div={}".format(
        id_num, info_df.base_mat[id_num],
        NUM_CHEB_NODES, NUM_TOTAL_DIVIDED_CHEB_NODES, NUM_MAX_DIVISION)

    print("processing id={}...".format(id_num))
    mat = TEProp(db_filename=DB_FILENAME, id_num=id_num)
    Tc = mat.min_raw_T
    Th = mat.max_raw_T

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

    Seebeck_x_left = Seebeck_xy_data[0, 0]
    Seebeck_x_right = Seebeck_xy_data[-1, 0]
    elec_resi_x_left = elec_resi_xy_data[0, 0]
    elec_resi_x_right = elec_resi_xy_data[-1, 0]
    thrm_cond_x_left = thrm_cond_xy_data[0, 0]
    thrm_cond_x_right = thrm_cond_xy_data[-1, 0]

    Seebeck_interp_func, Seebeck_exact_func, Seebeck_cheb_nodes = interpolate_barycentric_cheb(
        Seebeck_xy_data[:, 0], Seebeck_xy_data[:, 1], NUM_CHEB_NODES, spline_order=SEEBECK_SPLINE_ORDER
    )
    elec_resi_interp_func, elec_resi_exact_func, elec_resi_cheb_nodes = interpolate_barycentric_cheb(
        elec_resi_xy_data[:, 0], elec_resi_xy_data[:, 1], NUM_CHEB_NODES, spline_order=ELEC_RESI_SPLINE_ORDER
    )
    thrm_cond_interp_func, thrm_cond_exact_func, thrm_cond_cheb_nodes = interpolate_barycentric_cheb(
        thrm_cond_xy_data[:, 0], thrm_cond_xy_data[:, 1], NUM_CHEB_NODES, spline_order=THRM_COND_SPLINE_ORDER
    )
    Seebeck_interval = (Seebeck_x_left, Seebeck_x_right)
    elec_resi_interval = (elec_resi_x_left, elec_resi_x_right)
    thrm_cond_interval = (thrm_cond_x_left, thrm_cond_x_right)

    # compute the division points
    T_array = np.unique(np.concatenate((Seebeck_xy_data[:, 0], elec_resi_xy_data[:, 0], thrm_cond_xy_data[:, 0])))
    rel_err_array = np.zeros(len(T_array))
    # compute the division points using the derivative error
    Seebeck_deriv_peak_rel_errs = find_rel_error(
        T_array, Seebeck_interp_func.derivative(), Seebeck_exact_func.derivative())
    elec_resi_deriv_peak_rel_errs = find_rel_error(
        T_array, elec_resi_interp_func.derivative(), elec_resi_exact_func.derivative())
    thrm_cond_deriv_peak_rel_errs = find_rel_error(
        T_array, thrm_cond_interp_func.derivative(), thrm_cond_exact_func.derivative())
    deriv_peak_rel_errs = Seebeck_deriv_peak_rel_errs + elec_resi_deriv_peak_rel_errs + thrm_cond_deriv_peak_rel_errs
    deriv_peak_indices, _ = find_peaks(gaussian_filter1d(deriv_peak_rel_errs, sigma=GAUSSIAN_SIGMA_RATIO * (Th - Tc)))
    rel_err_array[deriv_peak_indices] = deriv_peak_rel_errs[deriv_peak_indices]
    list_deriv_peak_pos = T_array[deriv_peak_indices]

    # compute the division points using the relative error
    Seebeck_peak_indices, Seebeck_peak_rel_errs = find_rel_error_peaks(
        T_array, Seebeck_interp_func, Seebeck_exact_func)
    elec_resi_peak_indices, elec_resi_peak_rel_errs = find_rel_error_peaks(
        T_array, elec_resi_interp_func, elec_resi_exact_func)
    thrm_cond_peak_indices, thrm_cond_peak_rel_errs = find_rel_error_peaks(
        T_array, thrm_cond_interp_func, thrm_cond_exact_func)
    rel_err_array[Seebeck_peak_indices] = np.maximum(Seebeck_peak_rel_errs, rel_err_array[Seebeck_peak_indices])
    rel_err_array[elec_resi_peak_indices] = np.maximum(elec_resi_peak_rel_errs, rel_err_array[elec_resi_peak_indices])
    rel_err_array[thrm_cond_peak_indices] = np.maximum(thrm_cond_peak_rel_errs, rel_err_array[thrm_cond_peak_indices])
    # take out if two points are too close
    len_rel_err_array = len(rel_err_array)
    for idx1, rel_err_next in enumerate(rel_err_array[1:], start=1):
        T1 = T_array[idx1]
        rel_err1 = rel_err_array[idx1]
        if rel_err1 > 0:
            for idx2 in range(idx1 + 1, len_rel_err_array):
                T2 = T_array[idx2]
                rel_err2 = rel_err_array[idx2]
                if rel_err2 > 0 and (np.abs(T1 - T2) < TEMP_RESOLUTION_RATIO * (Th - Tc)):
                    if rel_err1 > rel_err2:
                        rel_err_array[idx2] = 0.0  # take out the value
                    else:
                        rel_err_array[idx1] = 0.0  # take out the value
                        break

    # take out zero-error indices
    list_peak_indices = np.array(np.argsort(rel_err_array))[::-1]
    list_peak_indices = list_peak_indices[rel_err_array[list_peak_indices] > 0][:NUM_MAX_DIVISION]
    list_peak_indices = sorted(list_peak_indices)

    list_rel_err_peak_pos = np.unique(
        np.concatenate(
            (T_array[Seebeck_peak_indices], T_array[elec_resi_peak_indices], T_array[thrm_cond_peak_indices]))
    )

    # compute curves with one region
    list_Seebeck_interp_funcs = [Seebeck_interp_func]
    list_Seebeck_intervals = [Seebeck_interval]
    list_Seebeck_cheb_nodes = [Seebeck_cheb_nodes]
    list_elec_resi_interp_funcs = [elec_resi_interp_func]
    list_elec_resi_intervals = [elec_resi_interval]
    list_elec_resi_cheb_nodes = [elec_resi_cheb_nodes]
    list_thrm_cond_interp_funcs = [thrm_cond_interp_func]
    list_thrm_cond_intervals = [thrm_cond_interval]
    list_thrm_cond_cheb_nodes = [thrm_cond_cheb_nodes]

    length_interval = Seebeck_x_right - Seebeck_x_left
    num_remaining_cheb_nodes = NUM_TOTAL_DIVIDED_CHEB_NODES
    for idx, peak_index in enumerate(list_peak_indices):
        selected_division_point = T_array[peak_index]

        # compute number of nodes
        length_interval_left = selected_division_point - list_Seebeck_intervals[idx][0]
        num_cheb_nodes_left = round_up_to_odd(length_interval_left / length_interval * NUM_TOTAL_DIVIDED_CHEB_NODES)
        num_cheb_nodes_left = np.max((num_cheb_nodes_left, NUM_MIN_CHEB_NODES))
        num_cheb_nodes_right = num_remaining_cheb_nodes - num_cheb_nodes_left + 1
        num_cheb_nodes_right = np.max((num_cheb_nodes_right, NUM_MIN_CHEB_NODES))
        num_remaining_cheb_nodes -= (num_cheb_nodes_left - 1)

        # divide into two regions
        Seebeck_interp_func_left, Seebeck_interp_func_right, \
        elec_resi_interp_func_left, elec_resi_interp_func_right, \
        thrm_cond_interp_func_left, thrm_cond_interp_func_right, \
        Seebeck_interval_left, Seebeck_interval_right, \
        elec_resi_interval_left, elec_resi_interval_right, \
        thrm_cond_interval_left, thrm_cond_interval_right, \
        Seebeck_cheb_nodes_left, Seebeck_cheb_nodes_right, \
        elec_resi_cheb_nodes_left, elec_resi_cheb_nodes_right, \
        thrm_cond_cheb_nodes_left, thrm_cond_cheb_nodes_right = divide_teps(
            Seebeck_exact_func, elec_resi_exact_func, thrm_cond_exact_func,
            list_Seebeck_intervals[idx], list_elec_resi_intervals[idx], list_thrm_cond_intervals[idx],
            selected_division_point, num_cheb_nodes_left, num_cheb_nodes_right)

        list_Seebeck_interp_funcs[idx] = Seebeck_interp_func_left
        list_Seebeck_interp_funcs.insert(idx + 1, Seebeck_interp_func_right)
        list_Seebeck_intervals[idx] = Seebeck_interval_left
        list_Seebeck_intervals.insert(idx + 1, Seebeck_interval_right)
        list_Seebeck_cheb_nodes[idx] = Seebeck_cheb_nodes_left
        list_Seebeck_cheb_nodes.insert(idx + 1, Seebeck_cheb_nodes_right)

        list_elec_resi_interp_funcs[idx] = elec_resi_interp_func_left
        list_elec_resi_interp_funcs.insert(idx + 1, elec_resi_interp_func_right)
        list_elec_resi_intervals[idx] = elec_resi_interval_left
        list_elec_resi_intervals.insert(idx + 1, elec_resi_interval_right)
        list_elec_resi_cheb_nodes[idx] = elec_resi_cheb_nodes_left
        list_elec_resi_cheb_nodes.insert(idx + 1, elec_resi_cheb_nodes_right)

        list_thrm_cond_interp_funcs[idx] = thrm_cond_interp_func_left
        list_thrm_cond_interp_funcs.insert(idx + 1, thrm_cond_interp_func_right)
        list_thrm_cond_intervals[idx] = thrm_cond_interval_left
        list_thrm_cond_intervals.insert(idx + 1, thrm_cond_interval_right)
        list_thrm_cond_cheb_nodes[idx] = thrm_cond_cheb_nodes_left
        list_thrm_cond_cheb_nodes.insert(idx + 1, thrm_cond_cheb_nodes_right)

    num_regions = len(list_Seebeck_intervals)
    num_total_nodes = sum((len(cheb_node) for cheb_node in list_Seebeck_cheb_nodes)) - (num_regions - 1)
    # draw the figures
    if IS_DRAW_GRAPH:
        one_curve_interp_label = '1 polynomial\n({} nodes)'.format(NUM_CHEB_NODES)
        multipe_curve_interp_label = '{} polynomials\n({} nodes)'.format(num_regions, num_total_nodes)

        plt.figure(figsize=(21, 7))

        ax1 = plt.subplot(131)
        Seebeck_T = np.linspace(Seebeck_x_left, Seebeck_x_right, num=1000, dtype=np.float64)
        Seebeck_exact_curve = Seebeck_exact_func(Seebeck_T)
        Seebeck_interp_curve = Seebeck_interp_func(Seebeck_T)
        plt.plot(Seebeck_T, Seebeck_exact_curve * 1e6, label='exact', color=EXACT_LINE_COLOR,
                 linestyle=EXACT_LINE_LINESTYLE, linewidth=EXACT_LINE_LINEWIDTH, alpha=EXACT_LINE_ALPHA)
        plt.plot(Seebeck_T, Seebeck_interp_curve * 1e6, label=one_curve_interp_label, color=ONE_POLY_LINE_COLOR,
                 linestyle=ONE_POLY_LINE_LINESTYLE, linewidth=ONE_POLY_LINE_LINEWIDTH, alpha=ONE_POLY_LINE_ALPHA)
        plt.plot([None], [None], linestyle='-', color='r',
                 label=multipe_curve_interp_label)
        for idx in range(num_regions):
            Seebeck_T_region = np.linspace(list_Seebeck_intervals[idx][0], list_Seebeck_intervals[idx][1],
                                           num=1000, dtype=np.float64)
            Seebeck_interp_func_region = list_Seebeck_interp_funcs[idx]
            Seebeck_cheb_nodes_region = list_Seebeck_cheb_nodes[idx]
            plt.plot(Seebeck_T_region, Seebeck_interp_func_region(Seebeck_T_region) * 1e6,
                     linestyle='-', color='r')
            plt.plot(Seebeck_cheb_nodes_region, Seebeck_interp_func_region(Seebeck_cheb_nodes_region) * 1e6, marker='.',
                     markersize=10, linestyle='None', color='r')
            if idx + 1 < num_regions:
                x_pos = T_array[list_peak_indices][idx]
                if (x_pos in list_deriv_peak_pos) and (x_pos not in list_rel_err_peak_pos):
                    plt.axvline(x=x_pos, color=VERTICAL_FROM_DERIV_ERR_LINE_COLOR,
                                linestyle=VERTICAL_FROM_DERIV_ERR_LINE_LINESTYLE,
                                linewidth=VERTICAL_FROM_DERIV_ERR_LINE_LINEWIDTH,
                                alpha=VERTICAL_FROM_DERIV_ERR_LINE_ALPHA)
                else:
                    plt.axvline(x=x_pos, color=VERTICAL_FROM_ERR_LINE_COLOR,
                                linestyle=VERTICAL_FROM_ERR_LINE_LINESTYLE,
                                linewidth=VERTICAL_FROM_ERR_LINE_LINEWIDTH,
                                alpha=VERTICAL_FROM_ERR_LINE_ALPHA)
        plt.legend()
        plt.ylabel(r"Seebeck coefficient [$\mu$V/K]")
        plt.xlabel("Temperature [K]")
        plt.xticks(xticks)

        # elec_resi curve
        plt.subplot(132, sharex=ax1)
        elec_resi_T = np.linspace(elec_resi_x_left, elec_resi_x_right, num=1000, dtype=np.float64)
        elec_resi_exact_curve = elec_resi_exact_func(elec_resi_T)
        elec_resi_interp_curve = elec_resi_interp_func(elec_resi_T)
        plt.plot(elec_resi_T, elec_resi_exact_curve * 1e2, label='exact', color=EXACT_LINE_COLOR,
                 linestyle=EXACT_LINE_LINESTYLE, linewidth=EXACT_LINE_LINEWIDTH, alpha=EXACT_LINE_ALPHA)
        plt.plot(elec_resi_T, elec_resi_interp_curve * 1e2, label=one_curve_interp_label, color=ONE_POLY_LINE_COLOR,
                 linestyle=ONE_POLY_LINE_LINESTYLE, linewidth=ONE_POLY_LINE_LINEWIDTH, alpha=ONE_POLY_LINE_ALPHA)
        plt.plot([None], [None], linestyle='-', color='r',
                 label=multipe_curve_interp_label)
        for idx in range(num_regions):
            elec_resi_T_region = np.linspace(list_elec_resi_intervals[idx][0], list_elec_resi_intervals[idx][1],
                                             num=1000, dtype=np.float64)
            elec_resi_interp_func_region = list_elec_resi_interp_funcs[idx]
            elec_resi_cheb_nodes_region = list_elec_resi_cheb_nodes[idx]
            plt.plot(elec_resi_T_region, elec_resi_interp_func_region(elec_resi_T_region) * 1e2,
                     linestyle='-', color='r')
            plt.plot(elec_resi_cheb_nodes_region, elec_resi_interp_func_region(elec_resi_cheb_nodes_region) * 1e2,
                     marker='.', markersize=10, linestyle='None', color='r')
            if idx + 1 < num_regions:
                x_pos = T_array[list_peak_indices][idx]
                if (x_pos in list_deriv_peak_pos) and (x_pos not in list_rel_err_peak_pos):
                    plt.axvline(x=x_pos, color=VERTICAL_FROM_DERIV_ERR_LINE_COLOR,
                                linestyle=VERTICAL_FROM_DERIV_ERR_LINE_LINESTYLE,
                                linewidth=VERTICAL_FROM_DERIV_ERR_LINE_LINEWIDTH,
                                alpha=VERTICAL_FROM_DERIV_ERR_LINE_ALPHA)
                else:
                    plt.axvline(x=x_pos, color=VERTICAL_FROM_ERR_LINE_COLOR,
                                linestyle=VERTICAL_FROM_ERR_LINE_LINESTYLE,
                                linewidth=VERTICAL_FROM_ERR_LINE_LINEWIDTH,
                                alpha=VERTICAL_FROM_ERR_LINE_ALPHA)
        plt.legend()
        plt.ylabel(r"Electrical resistivity [$\Omega\cdot$cm]")
        plt.xlabel("Temperature [K]")

        # thrm_cond curve
        plt.subplot(133, sharex=ax1)
        thrm_cond_T = np.linspace(thrm_cond_x_left, thrm_cond_x_right, num=1000, dtype=np.float64)
        thrm_cond_exact_curve = thrm_cond_exact_func(thrm_cond_T)
        thrm_cond_interp_curve = thrm_cond_interp_func(thrm_cond_T)
        plt.plot(thrm_cond_T, thrm_cond_exact_curve, label='exact', color=EXACT_LINE_COLOR,
                 linestyle=EXACT_LINE_LINESTYLE, linewidth=EXACT_LINE_LINEWIDTH, alpha=EXACT_LINE_ALPHA)
        plt.plot(thrm_cond_T, thrm_cond_interp_curve, label=one_curve_interp_label, color=ONE_POLY_LINE_COLOR,
                 linestyle=ONE_POLY_LINE_LINESTYLE, linewidth=ONE_POLY_LINE_LINEWIDTH, alpha=ONE_POLY_LINE_ALPHA)
        plt.plot([None], [None], linestyle='-', color='r',
                 label=multipe_curve_interp_label)
        for idx in range(num_regions):
            thrm_cond_T_region = np.linspace(list_thrm_cond_intervals[idx][0], list_thrm_cond_intervals[idx][1],
                                             num=1000, dtype=np.float64)
            thrm_cond_interp_func_region = list_thrm_cond_interp_funcs[idx]
            thrm_cond_cheb_nodes_region = list_thrm_cond_cheb_nodes[idx]
            plt.plot(thrm_cond_T_region, thrm_cond_interp_func_region(thrm_cond_T_region),
                     linestyle='-', color='r')
            plt.plot(thrm_cond_cheb_nodes_region, thrm_cond_interp_func_region(thrm_cond_cheb_nodes_region),
                     marker='.', markersize=10, linestyle='None', color='r')
            if idx + 1 < num_regions:
                x_pos = T_array[list_peak_indices][idx]
                if (x_pos in list_deriv_peak_pos) and (x_pos not in list_rel_err_peak_pos):
                    plt.axvline(x=x_pos, color=VERTICAL_FROM_DERIV_ERR_LINE_COLOR,
                                linestyle=VERTICAL_FROM_DERIV_ERR_LINE_LINESTYLE,
                                linewidth=VERTICAL_FROM_DERIV_ERR_LINE_LINEWIDTH,
                                alpha=VERTICAL_FROM_DERIV_ERR_LINE_ALPHA)
                else:
                    plt.axvline(x=x_pos, color=VERTICAL_FROM_ERR_LINE_COLOR,
                                linestyle=VERTICAL_FROM_ERR_LINE_LINESTYLE,
                                linewidth=VERTICAL_FROM_ERR_LINE_LINEWIDTH,
                                alpha=VERTICAL_FROM_ERR_LINE_ALPHA)
        plt.legend()
        plt.ylabel("Thermal conductivity [W/m/K]")
        plt.xlabel("Temperature [K]")

        if IS_DRAW_TITLE:
            plt.suptitle(TITLE, y=1.0)

        plt.tight_layout()

        if IS_SAVE_FIG:
            plt.savefig(os.path.join(RESULT_DIR, "{}.png".format(TITLE)))
            plt.savefig(os.path.join(RESULT_DIR, "{}.pdf".format(TITLE)))
            plt.savefig(os.path.join(RESULT_DIR, "{}.eps".format(TITLE)), dpi=300)
            plt.close()
