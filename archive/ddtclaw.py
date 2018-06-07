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
u_0 = 0  # Assume initial velocity is zero

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



## PDE equations
# Solvers
"""Solver"""
from clawpack import pyclaw
from clawpack import riemann

rs = riemann.euler_1D_py.euler_hllc_1D
solver = pyclaw.SharpClawSolver1D(rs)
solver.kernel_language = 'Python'

# Boundary conditions
solver.bc_lower[0] = pyclaw.BC.extrap  #.wall  # Wall in left
solver.bc_upper[0] = pyclaw.BC.extrap  # non-wall (zero-order extrapolation) condition at the right boundary

# Domain
mx = 800
x = pyclaw.Dimension(0.0, 0.1, mx, name='x')
print(x)
domain = pyclaw.Domain([x])

# Solution container
solution = pyclaw.Solution(solver.num_eqn, domain)

# Initial conditions
state = solution.state
xc = state.grid.p_centers[0]      # Array containing the cell center coordinates
state.q[0, :] = rho_0
state.q[1,:] = 0.                       # Velocity: zero
print(state.q[0, :])
#state.q[1,:] = 0.

#state = pyclaw.State(domain, num_eqn)

if __name__ == "__main__":
    # Set the size of the domain, and create the discretized grid.


    if 0:
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