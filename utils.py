# -*- coding: utf-8 -*-
"""
Barycentric polynomial interpolator, and one-dimensional thermoelectric equation solver.

Created on Jan 15 2020

@author: Jaywan Chung
"""

from functools import partial

import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_bvp
from scipy.interpolate import lagrange
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

from libs.pykeri.TEProp import TEProp


def interpolate_barycentric_cheb(x, y, num_cl_nodes, spline_order):
    xl = x[0]
    xr = x[-1]
    cl_nodes = (xl - xr) / 2 * np.cos(np.pi * np.linspace(0, num_cl_nodes - 1, num_cl_nodes) / (num_cl_nodes - 1)) + (
                xl + xr) / 2

    exact_func = UnivariateSpline(x, y, k=spline_order, s=0)

    cl_interp_func = BarycentricLagrangeChebyshevNodes(cl_nodes, exact_func(cl_nodes))

    return cl_interp_func, exact_func, cl_nodes


def interpolate_barycentric_equi(x, y, num_equi_nodes, spline_order):
    xl = x[0]
    xr = x[-1]
    equi_nodes = np.linspace(xl, xr, num=num_equi_nodes)

    exact_func = UnivariateSpline(x, y, k=spline_order, s=0)

    equi_interp_func = BarycentricLagrangeEquidistantNodes(equi_nodes, exact_func(equi_nodes))

    return equi_interp_func, exact_func, equi_nodes


def interpolate_cheb(x, y, num_cl_nodes, spline_order):
    xl = x[0]
    xr = x[-1]
    cl_nodes = (xl - xr) / 2 * np.cos(np.pi * np.linspace(0, num_cl_nodes - 1, num_cl_nodes) / (num_cl_nodes - 1)) + (
                xl + xr) / 2

    exact_func = UnivariateSpline(x, y, k=spline_order, s=0)

    cl_interp_func = lagrange(cl_nodes, exact_func(cl_nodes))

    return cl_interp_func, exact_func, cl_nodes


def interpolate_equi(x, y, num_equi_nodes, spline_order):
    xl = x[0]
    xr = x[-1]
    equi_nodes = np.linspace(xl, xr, num=num_equi_nodes)

    exact_func = UnivariateSpline(x, y, k=spline_order, s=0)
    equi_interp_func = lagrange(equi_nodes, exact_func(equi_nodes))

    return equi_interp_func, exact_func, equi_nodes


def get_material_list(db_filename, first_id, last_id, interp_opt=None):
    """Return a list of TEProp classes.

    :param db_filename: database filename containing the thermoelectric material properties
    :param first_id: smallest id of a TEP.
    :param last_id: largest id of a TEP.
    :param interp_opt: interpolation option to be used in TEProp.
    :return: (list of materials, list of material ids).
    """

    material_list = [None] * (last_id + 1)
    material_id_list = []
    for id_num in range(first_id, last_id + 1):
        try:
            material = TEProp(db_filename=db_filename, id_num=id_num)
        except ValueError:
            pass
        else:
            if interp_opt is not None:
                material.set_interp_opt(interp_opt)
            material_list[id_num] = material
            material_id_list.append(id_num)

    return material_list, material_id_list


class BarycentricLagrangeChebyshevNodes:
    """Compute a barycentric polynomial interpolation at Chebyshev nodes"""

    def __init__(self, raw_x, raw_y):
        """
        :param raw_x: it is assumed that x is a Chebyshev nodes.
        :param raw_y: interpolant values
        """
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.num_nodes = len(raw_x)

        self.w = np.power(-1.0, np.arange(self.num_nodes))
        self.w[0] *= 0.5
        self.w[-1] *= 0.5

    def __call__(self, x):
        return self.eval(x)

    def eval(self, x):
        x_array = np.asarray(x, dtype=np.float64).reshape(-1)
        len_x = x_array.size
        numer = np.zeros((len_x,), dtype=np.float64)
        denom = np.zeros((len_x,), dtype=np.float64)
        exact = np.zeros((len_x,), dtype=np.float64)
        idx_to_avoid = np.repeat(False, len_x)

        for raw_x_elem, raw_y_elem, w_elem in zip(self.raw_x, self.raw_y, self.w):
            diff = x_array - raw_x_elem
            is_diff_zero = np.isclose(diff, 0.0)
            exact[is_diff_zero] = raw_y_elem
            idx_to_avoid[is_diff_zero] = True
            diff[is_diff_zero] = 1.0
            numer += w_elem / diff * raw_y_elem
            denom += w_elem / diff

        idx = np.logical_not(idx_to_avoid)
        y = exact
        y[idx] = numer[idx] / denom[idx]

        # if the original value is a scalar, return a scalar
        if np.asarray(x).shape == ():
            return np.float64(y)
        return y

    def derivative(self):
        raw_dydx = np.zeros((self.num_nodes,), dtype=np.float64)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if not (j == i):
                    raw_dydx[i] += (self.w[j] / self.w[i]) * (self.raw_y[j] - self.raw_y[i]) \
                                   / (self.raw_x[i] - self.raw_x[j])

        return BarycentricLagrangeChebyshevNodes(self.raw_x, raw_dydx)


class BarycentricLagrangeEquidistantNodes:
    """Compute a barycentric polynomial interpolation at equidistant nodes"""

    def __init__(self, raw_x, raw_y):
        """
        :param raw_x: it is assumed that x is a Chebyshev nodes.
        :param raw_y: interpolant values
        """
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.num_nodes = len(raw_x)

        w = np.ones(self.num_nodes, dtype=np.float64)
        for idx in range(self.num_nodes):
            if idx > 0:
                w[idx] = w[idx - 1] * (self.num_nodes - idx) / idx
        w *= np.power(-1.0, np.arange(self.num_nodes))
        self.w = w

    def __call__(self, x):
        return self.eval(x)

    def eval(self, x):
        x_array = np.asarray(x, dtype=np.float64).reshape(-1)
        len_x = x_array.size
        numer = np.zeros((len_x,), dtype=np.float64)
        denom = np.zeros((len_x,), dtype=np.float64)
        exact = np.zeros((len_x,), dtype=np.float64)
        idx_to_avoid = np.repeat(False, len_x)

        for raw_x_elem, raw_y_elem, w_elem in zip(self.raw_x, self.raw_y, self.w):
            diff = x_array - raw_x_elem
            is_diff_zero = np.isclose(diff, 0.0)
            exact[is_diff_zero] = raw_y_elem
            idx_to_avoid[is_diff_zero] = True
            diff[is_diff_zero] = 1.0
            numer += w_elem / diff * raw_y_elem
            denom += w_elem / diff

        idx = np.logical_not(idx_to_avoid)
        y = exact
        y[idx] = numer[idx] / denom[idx]

        # if the original value is a scalar, return a scalar
        if np.asarray(x).shape == ():
            return np.float64(y)
        return y

    def derivative(self):
        raw_dydx = np.zeros((self.num_nodes,), dtype=np.float64)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if not (j == i):
                    raw_dydx[i] += (self.w[j] / self.w[i]) * (self.raw_y[j] - self.raw_y[i]) \
                                   / (self.raw_x[i] - self.raw_x[j])

        return BarycentricLagrangeEquidistantNodes(self.raw_x, raw_dydx)


class ThermoelctricEquationSolver:
    def __init__(self, L, A):
        self.L = L
        self.A = A

        self.Tc = None
        self.Th = None

        self.thrm_cond_func = None
        self.dthrm_resi_dT_func = None
        self.elec_resi_func = None
        self.Seebeck_func = None
        self.dSeebeck_dT_func = None

        self.efficiency = None
        self.power = None
        self.power_density = None
        self.gamma = None
        self.res = None

    def set_bc(self, Tc, Th):
        self.Tc = Tc
        self.Th = Th

    def set_te_mat_func(self, thrm_cond_func, elec_resi_func, Seebeck_func, dSeebeck_dT_func):
        self.thrm_cond_func = thrm_cond_func
        self.elec_resi_func = elec_resi_func
        self.Seebeck_func = Seebeck_func
        self.dSeebeck_dT_func = dSeebeck_dT_func

    def te_eqn(self, x, y, J):
        # y = [T; (kappa(T)T')]
        T = y[0]
        thrm_cond = self.thrm_cond_func(T)
        dTdx = y[1] / thrm_cond
        rhs = -self.elec_resi_func(T) * (J ** 2) + self.dSeebeck_dT_func(T) * T * dTdx * J

        return np.vstack((dTdx, rhs))

    def te_bc(self, ya, yb):
        return np.array([ya[0] - self.Th, yb[0] - self.Tc])

    def solve_te_eqn(self, I):
        J = I / self.A
        te_func = partial(self.te_eqn, J=J)

        # initial mesh
        x = np.linspace(0, self.L, 5)
        # initial guess: linear function
        initial_y0 = (self.Tc - self.Th) / self.L * x + self.Th
        initial_y1 = self.thrm_cond_func(initial_y0) * ((self.Tc - self.Th) / self.L + x * 0.0)
        y_guess = np.vstack((initial_y0, initial_y1))

        # solve the bvp
        res = solve_bvp(te_func, self.te_bc, x, y_guess, tol=1e-3, max_nodes=1e5)
        self.res = res

        T_for_V_gen = np.linspace(self.Tc, self.Th, 10000)
        V_gen = integrate.simps(self.Seebeck_func(T_for_V_gen), T_for_V_gen)
        # V_gen = integrate.quad(self.Seebeck_func, self.Tc, self.Th, limit=1000)[0]
        kappa_dTdx_at_0 = res.sol(0)[1]
        x_for_R = np.linspace(0, self.L, 10000)
        R = integrate.simps(self.elec_resi_func(res.sol(x_for_R)[0]), x_for_R) / self.A
        if np.abs(I) > 0:
            self.gamma = V_gen / (I * R) - 1
        else:
            self.gamma = None

        self.power = I * (V_gen - I * R)
        if self.power >= 0:
            self.efficiency = self.power / (-self.A * kappa_dTdx_at_0 + I * self.Seebeck_func(self.Th) * self.Th)
        else:
            self.efficiency = -np.infty  # efficiency is meaningless in a thermoelectric cooler

        self.power_density = self.power / self.A

        return res

    def compute_max_power_density(self):

        def min_fun(I):
            solver_res = self.solve_te_eqn(I)
            if not solver_res.success:
                return np.infty
            return -self.power_density / 1e4

        res = minimize_scalar(min_fun, method='brent')  # use the Brent-Dekker method for optimization

        return res

    def compute_max_efficiency(self):

        def min_fun(I):
            solver_res = self.solve_te_eqn(I)
            if not solver_res.success:
                return np.infty
            return -self.efficiency * 100

        res = minimize_scalar(min_fun, method='brent')  # use the Brent-Dekker method for optimization

        return res
