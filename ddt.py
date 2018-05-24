#!python
"""
Defregation to degredation transition model for PETN powders
Ref. Saenz and Stewart J. App. Phys. (2008)
"""
import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
from params import *

"""Initial conditions"""
phi_0 = 0.65
phi_0 = 0.75
p_0 = 1.0e-9  # GPa (equivalent to 1 atmosphere)
rho_0 = 1.6  # Initial density
v_0 = (rho_0)**(-1)  # Assume experimental condition

# Parameters correlated to an initial porosity
k = k_function(phi_0)
mu = mu_function(phi_0)
upsilon = upsilon_function(phi_0)

e_0_guess = 7.07  # kJ / cm3 Initial guess for starting internal energy
e_0 = e_0_guess
#TODO: Add in routine to compute e_0 from v_0 and p_0
"""Model"""
## Algebraic functions
# Compaction rate
def r_phi(p, phi):
    """
    Compaction rate (equation 20)
    :param p: Pressure
    :param phi: Porosity
    :return: r_phi  # compaction rate in us-1
    """
    return k_phi * (p - p_0 - P_h * (1 - np.sqrt((phi_0*(1 - phi))
                                                 / (phi*(1 - phi_0))
                                                 )
                                     )
                    )

# Reaction rate
def r_lambda(p, lambd):
    """
    Reaction rate (Equation 22)
    :param p: Pressure
    :param lambd: Reaction progression \in [0, 1]
    :return:
    """
    return np.heaviside(p - p_ign, 1) * k * (p/p_cj)**mu * (1 - lambd)**upsilon




"""Solver"""
def kdv_exact(x, c):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""
    u = 0.5 * c * np.cosh(0.5 * np.sqrt(c) * x) ** (-2)
    return u


def kdv(u, t, L):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    dudt = np.empty_like(u)
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.
    dudt = -6 * u * ux - uxxx

    return [dudt, 1]


def kdv_solution(u0, t, L):
    """Use odeint to solve the KdV equation on a periodic domain.

    `u0` is initial condition, `t` is the array of time values at which
    the solution is to be computed, and `L` is the length of the periodic
    domain."""

    sol = odeint(kdv, [u0, 0], t, args=(L,), mxstep=5000)
    return sol


if __name__ == "__main__":
    # Set the size of the domain, and create the discretized grid.
    L = 50.0
    N = 64
    dx = L / (N - 1.0)
    x = np.linspace(0, (1 - 1.0 / N) * L, N)

    # Set the initial conditions.
    # Not exact for two solitons on a periodic domain, but close enough...
    u0 = kdv_exact(x - 0.33 * L, 0.75) + kdv_exact(x - 0.65 * L, 0.4)

    # Set the time sample grid.
    T = 200
    t = np.linspace(0, T, 501)

    print
    "Computing the solution."
    sol = kdv_solution(u0, t, L)

    print
    "Plotting."

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.imshow(sol[::-1, :], extent=[0, L, 0, T])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.axis('normal')
    plt.title('Korteweg-de Vries on a Periodic Domain')
    plt.show()