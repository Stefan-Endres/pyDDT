#!python
"""
Defregation to degredation transition model for PETN powders
Ref. Saenz and Stewart J. App. Phys. (2008)

WENO hints

https://scicomp.stackexchange.com/questions/20054/implementation-of-1d-advection-in-python-using-weno-and-eno-schemes

"""
import numpy as np
from scipy import optimize
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff
from params import *
from wreos import *
import matplotlib.pyplot as plt
from integrator import WRKR
from progress.bar import IncrementalBar

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
    return 0.0
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
    return 0.0
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

def p_from_e_old(energy, v, lambd, phi, guess):
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
def init_cond(printout=True, x=None):
    # Discretization parameters
    # Space
    if x is None:
        x = np.linspace(0, L, N)  # A vector of discrete points in space

    # Initialize U container
    U_0 = np.zeros([5, x.size + 2 * gc])  # weno

    # Density
    U_0[0] = rho_0  # Initial density 'rho' in tube

    # Momentum
    U_0[1] = rho_0 * u_0  # Initial momentum 'rho * v'
    U_0[1, 0:3] = rho_0 * u_0_1  # Initial momemtum 'rho * u' on the boundary

    # Energy
#TODO: Do we need this?    e_0 = e(p_0, v_0, lambd_0, phi_0)  # Compute e_0
    U_0[2] = rho_0 * (e_0 + u_0**2/2.0)  # Initial energy 'rho * (e + u**2/2.0)'
    U_0[2, 0] = rho_0 * (e_0 + u_0_1**2/2.0)  # Initial energy 'rho * (e + u**2/2.0)
    if printout:
        print(f'e_0 = {e_0}')
        print(f'U_0[2] = {U_0[2]}')
    # Porosity
    U_0[3] = rho_0 * phi_0  # Initial porosity 'rho * phi'

    # Reaction progress
    U_0[4] = rho_0 * lambd_0  # Initial reaction progress 'rho_0 * lambd_0'

    # Compute initial reactions (should be 0.0)
    r_phi_0 = r_phi(p_0, phi_0)
    r_lambda_0 = r_lambda(p_0, lambd_0)
    if printout:
        print("----"*6)
        print("Initial reaction rates:")
        print("----"*6)
        print(f'r_phi_0({p_0}, {phi_0}) = {r_phi_0}')
        print(f'r_lambda_0({p_0}, {lambd_0}) = {r_lambda_0}')
        print("----"*6)

    return U_0

"""Equations"""
def f(U):
    """
    Compute f, given U
    :param U:
    :return:
    """
    # self.c * U
    F = np.zeros_like(U)
    F[0] = U[1]  # 'rho * u'
    # Compute the velocity at every element
    u = U[1] / U[0]  # u = rho * u / rho
    # print(u)
    # Compute the energy (temperature) at every element  # Validated
    E = U[2] / U[
        0] - u ** 2 / 2.0  # 'e = rho * ((e + u**2/2.0))/rho - u**2/2.0'

    # print(f'E = {E}')
    # e(p, v, lambd, phi)
    # F[1] = U[0] * u**2 + p

    # Compute the porosity at every element
    PHI = U[3] / U[0]  # phi = (rho * phi) / rho
    # print(f'Phi = {Phi}')

    # Compute the reaction progress at every element
    LAMBD = U[4] / U[0]  # lambd = (rho * lambd) / rho
    # print(f'Lambd = {Lambd}')

    F[3] = F[0] * PHI  # 'rho * u * phi'  # also U[1] * U[3]/U[0]
    F[4] = F[0] * LAMBD  # 'rho * u * Lambd'  # also U[1] * U[4]/U[0]

    # Compute the pressure
    # Compute the specific volume from the density
    V = U[0] ** (-1)  # specific volume
    # TODO: Try to vectorize
    ## P = p_from_e(E, V, LAMBD, PHI)
    P = np.zeros(np.shape(U)[1])
    for ind in range(np.shape(U)[1]):  # TODO: Extremely slow
        #P[ind] = p_from_e(E[ind], V[ind], LAMBD[ind], PHI[ind])
        P[ind] = E[ind]/V[ind]
    # print(f'P = {P}')

    F[1] = U[0] * u ** 2 + P  # rho * u^2 + p
    # print(f'F[1] = {F[1]}')

    F[2] = F[0] * (E + u ** 2 / 2.0 + P / U[0])

    pp = (u, E, PHI, LAMBD, V, P)
    return F, pp

def s(U, F, pp):
    """
    Compute s, given U
    :param U:
    :return:
    """

    S = np.zeros_like(U)

    (u, E, PHI, LAMBD, V, P) = pp

    ##### TODO: REUSED CODE FROM f(U)
    # self.c * U
    F = np.zeros_like(U)
    F[0] = U[1]  # 'rho * u'
    # Compute the velocity at every element
    u = U[1] / U[0]  # u = rho * u / rho
    # print(u)
    # Compute the energy (temperature) at every element  # Validated
    E = U[2] / U[
        0] - u ** 2 / 2.0  # 'e = rho * ((e + u**2/2.0))/rho - u**2/2.0'

    # print(f'E = {E}')
    # e(p, v, lambd, phi)
    # F[1] = U[0] * u**2 + p

    # Compute the porosity at every element
    PHI = U[3] / U[0]  # phi = (rho * phi) / rho
    # print(f'Phi = {Phi}')

    # Compute the reaction progress at every element
    LAMBD = U[4] / U[0]  # lambd = (rho * lambd) / rho
    # print(f'Lambd = {Lambd}')

    F[3] = F[0] * PHI  # 'rho * u * phi'  # also U[1] * U[3]/U[0]
    F[4] = F[0] * LAMBD  # 'rho * u * Lambd'  # also U[1] * U[4]/U[0]

    # Compute the pressure
    # Compute the specific volume from the density
    V = U[0] ** (-1)  # specific volume
    # TODO: Try to vectorize
    ## P = p_from_e(E, V, LAMBD, PHI)

    P = np.zeros(np.shape(U)[1])
    for ind in range(np.shape(U)[1]):  # TODO: Extremely slow
        #P[ind] = p_from_e(E[ind], V[ind], LAMBD[ind], PHI[ind])
        # P[ind] = p_from_e(E[ind], V[ind], LAMBD[ind], PHI[ind])
        P[ind] = E[ind] / V[ind]
    # print(f'P = {P}')


    #### REACTION COMPUTATION

    R_phi = r_phi(P, PHI)
    # print(f'R_phi = {R_phi}')
    # print(f'r_phi(P[0], Phi[0]) = {r_phi(P[0], Phi[0])}')
    # print(f'r_phi({P[0]}, {Phi[0]}) = {r_phi(P[0], Phi[0])}')
    R_lambd = r_lambda(P, LAMBD)

    # print(f'R_lambd = {R_lambd}')

    S[3] = U[0] * R_phi  # rho * r_phi
    S[4] = U[0] * R_lambd  # rho * r_lam

    return S, C

def plot_all_results(Usol, x, t):
    """
    Plot pressure, density, velocity and lambda
    :param U: Solution tensor
    :return:
    """
    Res = np.zeros_like(Usol)
    # Res[0] = P
    # Res[1] = Rho  # Density
    # Res[2] = velocity  #
    # Res[3] = Phi
    # Res[4] = lambda
    for tind, time in enumerate(t):
        U = Usol[tind, :, :]
        Res[tind, 1, :] = U[0]

        F = np.zeros_like(U)

        F[0] = U[1]  # 'rho * u'
        # Compute the velocity at every element
        u = U[1] / U[0]  # u = rho * u / rho

        Res[tind, 2, :] = u
        # print(u)
        # Compute the energy (temperature) at every element  # Validated
        E = U[2] / U[
            0] - u ** 2 / 2.0  # 'e = rho * ((e + u**2/2.0))/rho - u**2/2.0'
        # Compute the porosity at every element
        PHI = U[3] / U[0]  # phi = (rho * phi) / rho

        Res[tind, 3, :] = PHI
        # Compute the reaction progress at every element
        LAMBD = U[4] / U[0]  # lambd = (rho * lambd) / rho
        Res[tind, 4, :] = LAMBD

        F[3] = F[0] * PHI  # 'rho * u * phi'  # also U[1] * U[3]/U[0]
        F[4] = F[0] * LAMBD  # 'rho * u * Lambd'  # also U[1] * U[4]/U[0]

        # Compute the pressure
        # Compute the specific volume from the density
        V = U[0] ** (-1)  # specific volume
        P = np.zeros(np.shape(U)[1])
        for ind in range(np.shape(U)[1]):  # TODO: Extremely slow
            P[ind] = p_from_e(E[ind], V[ind], LAMBD[ind], PHI[ind])


        Res[tind, 0, :] = P
        F[1] = U[0] * u ** 2 + P  # rho * u^2 + p
        F[2] = F[0] * (E + u ** 2 / 2.0 + P / U[0])

    plot_u_t(x, t, Res[:, 0, :],
                  title=r'Pressure $P$ (GPa)', fign=0)

    plot_u_t(x, t, Res[:, 1, :],
                  title=r'Density $\rho$ (g/mm$^3$)', fign=1)

    plot_u_t(x, t, Res[:, 2, :] * 10,
                  title=r'Velocity $u$ (mm . $\mu$ s$^{-1}$)', fign=2)

    plot_u_t(x, t, Res[:, 3, :],
                  title=r'$\phi$ ', fign=3)

    plot_u_t(x, t, Res[:, 4, :],
                  title=r'$\lambda$ ', fign=4)

    return F

def plot_u_t(x, t, U, title=r'Density $\rho$ (g/mm$^3$)', fign=1):
    """
    Plots a variable U over time
    :param U:
    :param t:
    :return:
    """
    x = x * 10.0  # cm --> mm
    plt.figure(fign)
    plt.plot()
    plt.pcolor(x, t, U, cmap='RdBu')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x (mm)')
    plt.ylabel(r'time ($\mu$s)')
    #plt.show()

if __name__ == "__main__":
    solver = WRKR(f, s, N=N, x0=0.0, xf=L, t0=0.0,
                  tf=tf, #dt=0.5*tf,
                  dim=5)

    U_0 = init_cond(printout=True, x=solver.x)
    print(f'solver.x.shape = {solver.x.shape}')
    print(f'U_0.shape = {U_0.shape}')
    # Solve
    Urk3, solrk3 = solver.rk3(U_0)  # self.U_0_sol
    #Ue, sole = solver.euler(U_0)  # self.U_0_sol
    U = solrk3
    print(f'Finished simulation, plotting results...')

    plot_all_results(U, solver.x, solver.t)
    plt.show()
    if 0:
        plot_u_t(solver.x, solver.t, U[:, 0, :],
                 title=r'Density $\rho$ (g/mm$^3$)', fign=1)
        plot_u_t(solver.x, solver.t, U[:, 0, :],
                 title=r'Density $\rho$ (g/mm$^3$)', fign=2)
        #plot_u_t(solver.x, solver.t, U[:, 1, :], title=r'Velocity $u$ (mm/s$^-1$)')
        #plt.plot(solver.x, U[0, 0, :], '-', label='Initial 1')
        #plt.plot(solver.x, U[0, 0, :].T, '-', label='Initial 1')
        plt.show()
