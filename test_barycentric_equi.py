# -*- coding: utf-8 -*-
"""
Test the barycentric polynomial interpolation at equidistant nodes in utils.py.

Last updated on Apr 21 2020

@author: Jaywan Chung
"""

import numpy as np

from libs.pykeri.TEProp import TEProp
from utils import interpolate_barycentric_equi
from utils import interpolate_equi

DB_FILENAME = "tep_20180409.db"
SEEBECK_SPLINE_ORDER = 2
ELEC_RESI_SPLINE_ORDER = 1
THRM_RESI_SPLINE_ORDER = 1

NUM_CHEB_NODES = 5

print("Test Barycentric formula of Lagrange polynomial using {} equidistant nodes.".format(NUM_CHEB_NODES))

for id_num in [68]:
    print("processing id={}...".format(id_num))
    mat = TEProp(db_filename=DB_FILENAME, id_num=id_num)
    Tc = mat.min_raw_T
    Th = mat.max_raw_T

    Seebeck_xy_data = np.asarray(mat.Seebeck.raw_data())
    elec_resi_xy_data = np.asarray(mat.elec_resi.raw_data())
    thrm_cond_xy_data = np.asarray(mat.thrm_cond.raw_data())
    thrm_resi_xy_data = thrm_cond_xy_data.copy()
    thrm_resi_xy_data[:, 1] = 1 / thrm_cond_xy_data[:, 1]

    # make unique, sorted on temperature
    _, Seebeck_unique_indices = np.unique(Seebeck_xy_data[:, 0], return_index=True)
    Seebeck_xy_data = Seebeck_xy_data[Seebeck_unique_indices, :]
    _, elec_resi_unique_indices = np.unique(elec_resi_xy_data[:, 0], return_index=True)
    elec_resi_xy_data = elec_resi_xy_data[elec_resi_unique_indices, :]
    _, thrm_resi_unique_indices = np.unique(thrm_resi_xy_data[:, 0], return_index=True)
    thrm_resi_xy_data = thrm_resi_xy_data[thrm_resi_unique_indices, :]

    num_Seebeck_samples = len(Seebeck_xy_data)
    num_elec_resi_samples = len(elec_resi_xy_data)
    num_thrm_resi_samples = len(thrm_resi_xy_data)

    Seebeck_barycentric_func, Seebeck_exact_func, Seebeck_cheb_nodes = interpolate_barycentric_equi(
        Seebeck_xy_data[:, 0], Seebeck_xy_data[:, 1], NUM_CHEB_NODES, spline_order=SEEBECK_SPLINE_ORDER
    )
    elec_resi_barycentric_func, elec_resi_exact_func, elec_resi_cheb_nodes = interpolate_barycentric_equi(
        elec_resi_xy_data[:, 0], elec_resi_xy_data[:, 1], NUM_CHEB_NODES, spline_order=ELEC_RESI_SPLINE_ORDER
    )
    thrm_resi_barycentric_func, thrm_resi_exact_func, thrm_resi_cheb_nodes = interpolate_barycentric_equi(
        thrm_resi_xy_data[:, 0], thrm_resi_xy_data[:, 1], NUM_CHEB_NODES, spline_order=THRM_RESI_SPLINE_ORDER
    )

    Seebeck_lagrange_func, _, _ = interpolate_equi(
        Seebeck_xy_data[:, 0], Seebeck_xy_data[:, 1], NUM_CHEB_NODES, spline_order=SEEBECK_SPLINE_ORDER
    )
    elec_resi_lagrange_func, _, _ = interpolate_equi(
        elec_resi_xy_data[:, 0], elec_resi_xy_data[:, 1], NUM_CHEB_NODES, spline_order=ELEC_RESI_SPLINE_ORDER
    )
    thrm_resi_lagrange_func, _, _ = interpolate_equi(
        thrm_resi_xy_data[:, 0], thrm_resi_xy_data[:, 1], NUM_CHEB_NODES, spline_order=THRM_RESI_SPLINE_ORDER
    )

    # compute the interpolation curves
    Seebeck_T = np.linspace(Seebeck_xy_data[:, 0][0], Seebeck_xy_data[:, 0][-1], num=1000, dtype=np.float64)
    elec_resi_T = np.linspace(elec_resi_xy_data[:, 0][0], elec_resi_xy_data[:, 0][-1], num=1000, dtype=np.float64)
    thrm_resi_T = np.linspace(thrm_resi_xy_data[:, 0][0], thrm_resi_xy_data[:, 0][-1], num=1000, dtype=np.float64)
    T_grid = np.linspace(Tc, Th, num=1000, dtype=np.float64)
    Seebeck_barycentric_curve = Seebeck_barycentric_func(Seebeck_T)
    Seebeck_lagrange_curve = Seebeck_lagrange_func(Seebeck_T)
    elec_resi_barycentric_curve = elec_resi_barycentric_func(elec_resi_T)
    elec_resi_lagrange_curve = elec_resi_lagrange_func(elec_resi_T)
    thrm_resi_barycentric_curve = thrm_resi_barycentric_func(thrm_resi_T)
    thrm_resi_lagrange_curve = thrm_resi_lagrange_func(thrm_resi_T)

    dSeebeck_dT_barycentric_curve = Seebeck_barycentric_func.derivative()(Seebeck_T)
    dSeebeck_dT_lagrange_curve = Seebeck_lagrange_func.deriv()(Seebeck_T)

    assert (np.allclose(Seebeck_barycentric_curve, Seebeck_lagrange_curve))
    print("Seebeck coefficient ok.")
    assert (np.allclose(elec_resi_barycentric_curve, elec_resi_lagrange_curve))
    print("Electrical resistivity ok.")
    assert (np.allclose(thrm_resi_barycentric_curve, thrm_resi_lagrange_curve))
    print("Thermal resistivity ok.")
    assert (np.allclose(dSeebeck_dT_barycentric_curve, dSeebeck_dT_lagrange_curve))
    print("dSeebeck/dT ok.")
