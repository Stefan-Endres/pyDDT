#!python
"""
Defregation to degredation transition model for PETN powders
Ref. Saenz and Stewart J. App. Phys. (2008)
"""
import numpy as np
from scipy import optimize
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff
from params import *
from wreos import *
import matplotlib.pyplot as plt


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

def p_v(energy, lambd, phi, guess):
    """
    Compute the pressure and volume, given energy, lambd and phi
    :param energy: energy
    :param lambd: reaction progress
    :param phi: porosity
    :param guess: Initial guess for p and v
    :return: p, v  # pressure and volume
    """
    # ###test
    p, v = guess
    print(f'energy - e(p, v, lambd, phi) = {energy - e(p, v, lambd, phi)}')
    ####
    def system(x):
        p, v = x
        return [abs(energy - e(p, v, lambd, phi)), 0]

    x_0 = guess
    #print(f'x_0 = {x_0}')
    #sol = optimize.root(system, x0, jac=jac, method='hybr')
    sol = optimize.root(system, x_0)
    p, v = sol.x
    print(f'energy - e(p, v, lambd, phi) = {energy - e(p, v, lambd, phi)}')
    return p, v

def p_from_e(energy, v, lambd, phi, guess):
    """
    Compute the pressure, given energy, volume, lambd and phi
    :param energy: energy
    :param v: specific volume
    :param lambd: reaction progress
    :param phi: porosity
    :param guess: Initial guess for p and v
    :return: p, v  # pressure and volume
    """
    # ###test
    print(f'energy - e(p, v, lambd, phi) = {energy - e(guess, v, lambd, phi)}')
    ####
    def system(p):
        return energy - e(p, v, lambd, phi)

    x_0 = guess
    #print(f'x_0 = {x_0}')
    #sol = optimize.root(system, x0, jac=jac, method='hybr')
    sol = optimize.root(system, x_0)
    p = sol.x[0]
    print(f'energy - e(p, v, lambd, phi) = {energy - e(p, v, lambd, phi)}')
    return p

"""Initiation"""
# Discretization parameters
# Space
N = 100  # Number of discrete spatial elements
x = np.linspace(0, L, N + 1)  # A vector of discrete points in space
dx = x[1] - x[0]

# Time
#dt = 1/(2 * alpha) * dx**2
#Nt = int(time/dt)

# Initialize U container
U_0 = np.zeros([5, x.size])

# Density
U_0[0] = rho_0  # Initial density 'rho' in tube

# Momentum
U_0[1] = rho_0 * 0.0  # Initial momentum 'rho * v'
U_0[1, 0] = rho_0 * u_0_1  # Initial momemtum 'rho * u' on the boundary

# Energy
e_0 = e(p_0, v_0, lambd_0, phi_0)  # Compute e_0
U_0[2] = rho_0 * (e_0 + u_0**2/2.0)  # Initial energy 'rho * (e + u**2/2.0)'
U_0[2, 0] = rho_0 * (e_0 + u_0_1**2/2.0)  # Initial energy 'rho * (e + u**2/2.0)
print(f'e_0 = {e_0}')
print(f'U_0[2] = {U_0[2]}')
# Porosity
U_0[3] = rho_0 * phi_0  # Initial porosity 'rho * phi'

# Reaction progress
U_0[4] = rho_0 * lambd_0  # Initial reaction progress 'rho_0 * lambd_0'

#print(U_0)
# Compute initial reactions (should be 0.0)
r_phi_0 = r_phi(p_0, phi_0)
r_lambda_0 = r_lambda(p_0, lambd_0)
print("----"*6)
print("Initial reaction rates:")
print("----"*6)
print(f'r_phi_0({p_0}, {phi_0}) = {r_phi_0}')
print(f'r_lambda_0({p_0}, {lambd_0}) = {r_lambda_0}')
print("----"*6)
#e(p, v, lambd, phi)

#RHO_0_fft = psdiff(RHO_0, period=L)


if 0:  # TODO: Pressure and volume calculations not working
    print('='*100)
    print(f'p_0 = {p_0}')
    print(f'v_0 = {v_0}')
    print(f'e_0 = {e_0}')
    p, v = p_v(e_0, lambd_0, phi_0, [p_0 - 3e-10, v_0 - 1e-3])
    print(f'Computed p, v at e_0 = {e_0} : {p, v}')
    print('='*30)
    p = p_from_e(e_0, v_0, lambd_0, phi_0, p_0)
    print(f'Computed p at e_0, v_0 = {e_0, v_0} : {p*5}')
    print('='*100)

def dUdt(U, t):
    """
    Compute dUdt, given U.
    Solve the algebraic system of equations to find the variables.
    (Given U, compute F and S
    :param U: Vectors of u
    :return: F, S
    """
    ## Compute F
    F = np.zeros([5, x.size])
    F[0] = U[1]  # 'rho * u'
    # Compute the velocity at every element
    u = U[1]/U[0]  # u = rho * u / rho
    #print(u)
    # Compute the energy (temperature) at every element  # Validated
    E = U[2]/U[0] - u**2/2.0  # 'e = rho * ((e + u**2/2.0))/rho - u**2/2.0'

    #print(f'E = {E}')
    # e(p, v, lambd, phi)
    #F[1] = U[0] * u**2 + p

    # Compute the porosity at every element
    Phi = U[3]/U[0]  # phi = (rho * phi) / rho
    #print(f'Phi = {Phi}')

    # Compute the reaction progress at every element
    Lambd = U[4]/U[0]  # lambd = (rho * lambd) / rho
    #print(f'Lambd = {Lambd}')

    F[3] = F[0] * Phi  # 'rho * u * phi'  # also U[1] * U[3]/U[0]
    F[4] = F[0] * Lambd  # 'rho * u * Lambd'  # also U[1] * U[4]/U[0]

    # Compute the pressure
    # Compute the specific volume from the density
    V = U[0]**(-1)  # specific volume
    #TODO: We need a working routine to find p from e
    P = np.zeros(x.size)
    P[:] = p_0
    #print(f'P = {P}')

    F[1] = U[0] * u**2 + P  # rho * u^2 + p
    #print(f'F[1] = {F[1]}')

    F[2] = F[0] * (E + u**2/2.0 + P//U[0])  # 'rho * u * (e + u^2/2 + p/rho)

    ## Compute S
    S = np.zeros([5, x.size])

    R_phi = r_phi(P, Phi)
    #print(f'R_phi = {R_phi}')
    #print(f'r_phi(P[0], Phi[0]) = {r_phi(P[0], Phi[0])}')
    #print(f'r_phi({P[0]}, {Phi[0]}) = {r_phi(P[0], Phi[0])}')
    R_lambd = r_lambda(P, Lambd)
    #print(f'R_lambd = {R_lambd}')

    S[3] = U[0] * R_phi  # rho * r_phi
    S[4] = U[0] * R_lambd # rho * r_lam
    #print(S)

    # Gradient of F
    dFdx0 = psdiff(F[0], period=L)
    dFdx1 = psdiff(F[1], period=L)
    dFdx2 = psdiff(F[2], period=L)
    dFdx3 = psdiff(F[3], period=L)
    dFdx4 = psdiff(F[4], period=L)

    #print(dFdx4)
    dFdx = np.array([dFdx0,
                     dFdx1,
                     dFdx2,
                     dFdx3,
                     dFdx4,
                     ])
    #print(dFdx)
    #print(S - dFdx)
    return S - dFdx

dUdt(U_0, 0)

# Set the time sample grid.
t = np.linspace(t0, tf, 501)
dt = t[1]

print(U_0[0,:].size)
print(U_0[0,:])
def rk3(U_0, t):
    # TVD RK 3 solver
    U = U_0
    t_c = t0  # current time tracker
    U_0_sol = np.zeros([t.size, U_0[0, :].size])
    ind = 0  # solution index traxker
    while t_c <= tf:
        U_1 = U + dt * dUdt(U, t_c)
        U_2 = 3/4.0 * U + 1/4.0 * U_1 + 1/4.0 * dt * dUdt(U_1, t_c)
        U = 1/3.0 * U + 2/3.0 * U_2 + 2/3.0 * dt * dUdt(U_2, t_c)

        U_0_sol[ind] = U[0]
        #sol[ind] = U  # Store results

        # Update time and index
        t_c += dt
        ind += 1

    U_0_sol[-1] = U_0_sol[-2]  # Set final row of zeros equal to 2nd to last
    sol = U_0_sol
    return sol

#print(rk3(U_0, t))
#print(sol[:, 0])

def plot_u_t(x, t, U, title=r'Density $\rho$ (g/mm$^3$)'):
    """
    Plots a variable U over time
    :param U:
    :param t:
    :return:
    """
    plt.pcolor(x, t, U, cmap='RdBu')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x (mm)')
    plt.ylabel(r'time ($\mu$s)')
    plt.show()

if __name__ == "__main__":
    sol = rk3(U_0, t)
    U = sol
    #for i in range(U[:, 0].size):
    #    U[i] += 0.1*i

    plot_u_t(x, t, U)
    #CS = plot.contour(x, t, U)
    #cbar = plot.colorbar(CS)

    #plot.contour(t, t, t)
    #plot.show()