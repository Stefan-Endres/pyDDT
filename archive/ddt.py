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
    #return 0.0
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
def init_cond(printout=True, method='s'):
    # Discretization parameters
    # Space
    #N = 60  # Number of discrete spatial elements
    x = np.linspace(0, L, N + 1)  # A vector of discrete points in space
    dx = x[1] - x[0]

    # Time
    #dt = 1/(2 * alpha) * dx**2
    #Nt = int(time/dt)

    # Initialize U container
    if method is 's':
        U_0 = np.zeros([5, x.size])  # pseudo spectral method
    elif method is 'w':
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

    #print(U_0)
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
    #e(p, v, lambd, phi)

    #RHO_0_fft = psdiff(RHO_0, period=L)

    # Set the time sample grid.
    t = np.linspace(t0, tf, 501)
    t = np.linspace(t0, tf, 500)
    dt = t[1]

    #WENO
    # adding ghost cells
    #gcr = x[-1] + np.linspace(1, gc, gc) * dx
    #gcl = x[0] + np.linspace(-gc, -1, gc) * dx
    #xc = np.append(x, gcr)
    #xc = np.append(gcl, xc)
    #uc = np.append(u, u[-gc:])
    #uc = np.append(u[0:gc], uc)

    return U_0, x, t, dt, dx

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
    #F = np.zeros([5, x.size])
    F = np.zeros_like(U)
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
    PHI = U[3]/U[0]  # phi = (rho * phi) / rho
    #print(f'Phi = {Phi}')

    # Compute the reaction progress at every element
    LAMBD = U[4]/U[0]  # lambd = (rho * lambd) / rho
    #print(f'Lambd = {Lambd}')

    F[3] = F[0] * PHI  # 'rho * u * phi'  # also U[1] * U[3]/U[0]
    F[4] = F[0] * LAMBD  # 'rho * u * Lambd'  # also U[1] * U[4]/U[0]

    # Compute the pressure
    # Compute the specific volume from the density
    V = U[0]**(-1)  # specific volume
    #TODO: Try to vectorize
    ## P = p_from_e(E, V, LAMBD, PHI)
    P = np.zeros(np.shape(U)[1])
    for ind in range(np.shape(U)[1]):  # TODO: Extremely slow
        P[ind] = p_from_e(E[ind], V[ind], LAMBD[ind], PHI[ind])
    #print(f'P = {P}')

    F[1] = U[0] * u**2 + P  # rho * u^2 + p
    #print(f'F[1] = {F[1]}')

    F[2] = F[0] * (E + u**2/2.0 + P/U[0])
    # 'rho * u * (e + u^2/2 + p/rho)

    ## Compute S
    #S = np.zeros([5, x.size])
    S = np.zeros_like(U)

    R_phi = r_phi(P, PHI)
    #print(f'R_phi = {R_phi}')
    #print(f'r_phi(P[0], Phi[0]) = {r_phi(P[0], Phi[0])}')
    #print(f'r_phi({P[0]}, {Phi[0]}) = {r_phi(P[0], Phi[0])}')
    R_lambd = r_lambda(P, LAMBD)

    #print(f'R_lambd = {R_lambd}')

    S[3] = U[0] * R_phi  # rho * r_phi
    S[4] = U[0] * R_lambd  # rho * r_lam
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

    d2Fdx0 = psdiff(F[0], order=2, period=L)
    d2Fdx1 = psdiff(F[1], order=2, period=L)
    d2Fdx2 = psdiff(F[2], order=2, period=L)
    d2Fdx3 = psdiff(F[3], order=2, period=L)
    d2Fdx4 = psdiff(F[4], order=2, period=L)

    dF2dx = np.array([d2Fdx0,
                      d2Fdx1,
                      d2Fdx2,
                      d2Fdx3,
                      d2Fdx4,
                      ])
    #print(dFdx)
    #print(S - dFdx)
    return S - dFdx - 0.1e-3 * dF2dx


def dUwdt(U, t, dx):
    """
    Compute dUdt, given U.
    Solve the algebraic system of equations to find the variables.
    (Given U, compute F and S
    :param U: Vectors of u
    :return: F, S
    """
    if 1:
        ## Compute F
        # F = np.zeros([5, x.size])
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
            P[ind] = p_from_e(E[ind], V[ind], LAMBD[ind], PHI[ind])

        # print(f'P = {P}')

        F[1] = U[0] * u ** 2 + P  # rho * u^2 + p
        # print(f'F[1] = {F[1]}')

        F[2] = F[0] * (E + u ** 2 / 2.0 + P / U[0])
        # 'rho * u * (e + u^2/2 + p/rho)

        ## Compute S
        # S = np.zeros([5, x.size])
        S = np.zeros_like(U)

        R_phi = r_phi(P, PHI)
        # print(f'R_phi = {R_phi}')
        # print(f'r_phi(P[0], Phi[0]) = {r_phi(P[0], Phi[0])}')
        # print(f'r_phi({P[0]}, {Phi[0]}) = {r_phi(P[0], Phi[0])}')
        R_lambd = r_lambda(P, LAMBD)
        # print(f'R_lambd = {R_lambd}')

        S[3] = U[0] * R_phi  # rho * r_phi
        S[4] = U[0] * R_lambd  # rho * r_lam
        # print(S)

    # Gradient of F
    # Compute fluxes
    # Note fluxes do not use ghost cells
    # NOTE: We assume that the f = f_plus + f_neg decomposition is f_plus due
    #       to the shockwave only travelling in one direction

    # Fluxes from 0 + 1/2 to j - 1/2
    #TODO: We can extend these schemes to optionally use more ghost cells.

    # Compute U^()_{i+1/2} to be used in weno
    # NOTE: These arrrays are only defined for f_{0 + 1/2} to f_{j + 1/2}
    F_1 = (3 / 8.0) * F[:, :-4] - (5 / 4.0) * F[:, 1:-3] + (15 / 8.0) * F[:, 2:-2]
    if 0:
        print(f'U1')
        print(f'F[0]= {F[0]}')
        print(f'F[0, :-2] = {F[0, :-2]}')
        print(f'F[0, 1:-1] = {F[0, 1:-1]}')
        print(f'F[0, 0, 2:] = {F[0, 2:]}')
    #print(f'U_1 = {U_1}')
    F_2 = (-1 / 8.0) * F[:, 1:-3] + (3 / 4.0) * F[:, 2:-2] + (3 / 8.0) * F[:, 3:-1]
    if 0:
        print(f'U2')
        print(f'F[0]= {F[0]}')
        print(f'F[0, 1:-2] = {F[0, 1:-2]}')
        print(f'F[0, 2:-1] = {F[0, 2:-1]}')
        print(f'F[0, 3:]= {F[0, 3:]}')

        print(f'np.shape(U_1) = {np.shape(U_1)}')
        print(f'np.shape(U_2) = {np.shape(U_2)}')
        print(f'np.shape(F) = {np.shape(F)}')
    F_3 = (3 / 8.0 ) * F[:, 2:-2] + (3/4.0) * F[:, 3:-1] - (1/8.0) * F[:, 4:]


    # ENO scheme
    gamma_1 = 1/16.0
    gamma_2 = 5/8.0
    gamma_3 = 5/16.0

    # u_{i + 1/2}
    # u_i12 = gamma_1 * F_1 + gamma_2 * F_2 + gamma_3 * F_3
    # The above is the ENO solution

    # The WENO scheme:
    # http://www.scholarpedia.org/article/WENO_methods
    fi_2 = F[:, :-4]  # f_{i - 2}   (this is u_{i - 2} in the article)
    fi_1 = F[:, 1:-3]  # f_{i - 1}
    f_i = F[:, 2:-2]  # f_{i}
    beta_1 = (1/3.0) * (4 * fi_2**2 - 19.0 * fi_2 * fi_1
                        + 25 * fi_1**2 + 11 * fi_2 * f_i
                        - 31 * fi_1 * f_i + 10 * f_i**2)

    fi_1 = F[:, 1:-3]  # f_{i - 1}   (this is u_{i - 1} in the article)
    fi = F[:, 2:-2]  # f_{i - 1}
    fi_p1 = F[:, 3:-1]
    beta_2 = (1/3.0) * (4 * fi_1**2 - 13 * fi_1 * fi
                        + 13 * fi**2 + 5 * fi_1 * fi_p1
                        - 13 * fi * fi_p1 + 4 * fi_p1**2)

    fi = F[:, 2:-2]  # f_i  (this is u_i in the article)
    fi_p1 = F[:, 3:-1]  # f_{i + 1}
    fi_p2 = F[:, 4:]  # f_{i + 2}
    beta_3 = (1/3.0) * (10 * fi**2 - 31 * fi * fi_p1
                        + 25 * fi_p1**2 + 11 * fi * fi_p2
                        - 19 * fi_p1 * fi_p2 + 4 * fi_p2**2)

    w_tilde_1 = gamma_1 / (1e-6 + beta_1)**2
    w_tilde_2 = gamma_2 / (1e-6 + beta_2)**2
    w_tilde_3 = gamma_3 / (1e-6 + beta_3)**2
    w_tilde_sum = w_tilde_1 + w_tilde_2 + w_tilde_3

    w_1 = w_tilde_1 / w_tilde_sum
    w_2 = w_tilde_2 / w_tilde_sum
    w_3 = w_tilde_3 / w_tilde_sum

    u_i12 = w_1 * F_1 + w_2 * F_2 + w_3 * F_3  # u_{i + 1/2}

    # Compute the flux f_{j+1/2} - f_{j-1/2}
    f = u_i12
    #NOTE: We define flux at j = 0 to be = f_{0+1/2} - 0
    #      therefore we assume f_{0-1/2} = 0
    #flux = np.zeros([5, np.shape(F)[1] - 4])  # remove 4 ghost cells
    flux = np.zeros_like(f)
    if 0:
        print(f'np.shape(f) = {np.shape(f)}')
        print(f'np.shape(flux) = {np.shape(flux)}')
        print(f'np.shape(flux[:, 1:]) = {np.shape(flux[:, 1:])}')
        print(f'np.shape(f[:, 1:] - f[:, :1]) = {np.shape(f[:, 1:] - f[:, :1])}')

    if 0:
        flux[:, 1:] = f[:, 1:] - f[:, :-1]
        flux[:, 0] = f[:, 0]  #TODO: Check
    else:
        #c = 1
        #c = F[0] / u
       # c = f[0, 1:] / u[3:-2]
        c = 1
        flux[:, 1:] = 0.5 * (c + np.fabs(c)) * f[:, 1:] + 0.5 * (
                c - np.fabs(c)) * f[:, :-1]


    if 1:
        #print(f'f = {f}')
        #print(f'f[:, 1:] - f[:, :-1] = {f[:, 1:] - f[:, :-1]}')
        print(f'flux = {flux}')

    # Return dFdt = - 1/dx (f_{j+1/2} - f_{j-1/2}) + S
    dFdx = np.zeros_like(F)

    if 0:
        for Fx_ind, Fx in enumerate(flux):
            #print(f'Fx_ind = {Fx_ind}')
            #print(f'Fx = {Fx}')
            for x_ind, x in enumerate(Fx):
                #print(f'x_ind = {x_ind}')
                #print(f'x_ind = {x_ind}')
                #print(f'x = {x}')
                #print(f'Fx[x_ind] = {Fx[x_ind]}')
                #print(f'Fx[{Fx_ind}][{x_ind}] = {Fx[Fx_ind][x_ind]}')
                if x < 0:
                    flux[Fx_ind][x_ind] = 0
                    x = 0

    print(f'flux = {flux}')

    dFdx[:, 2:-2] = - 1/dx * (flux)
    #sol = S - F
    sol = S + F
    sol = S + dFdx

    #print(f'break solver by printing unknown {unknown}')
    return sol


def rk3(U_0, t, dt):
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


def rk3w(U_0, t, dt, dx):
    # TVD RK 3 solver
    U = U_0
    #t_c = t0  # current time tracker
    U_0_sol = np.zeros([t.size, 5, U_0[0, :].size])
    U_0_sol[0] = U_0#[0]
    ind = 0  # solution index tracker
    #while t_c <= tf:
    for t_c in t[1:]:
        # Compute new U_t+1
        #U_1 = U + dt * dUdt(U, t_c)
        U_1 = U + dt * dUwdt(U, t_c, dx)
        #U_2 = 3/4.0 * U + 1/4.0 * U_1 + 1/4.0 * dt * dUdt(U_1, t_c)
        U_2 = 3/4.0 * U + 1/4.0 * U_1 + 1/4.0 * dt * dUwdt(U_1, t_c, dx)
        #U = 1/3.0 * U + 2/3.0 * U_2 + 2/3.0 * dt * dUdt(U_2, t_c)
        U = 1/3.0 * U + 2/3.0 * U_2 + 2/3.0 * dt * dUwdt(U_2, t_c, dx)

        ind += 1
        #U_0_sol[ind] = U#[0]  # density solution
        U_0_sol[ind] = U


    #U_0_sol[-1] = U_0_sol[-2]  # Set final row of zeros equal to 2nd to last
    #sol = U_0_sol[:, gc:-(1+gc)]  # All rows, but exclude column of ghost cells
    sol = U_0_sol[:, :, gc:-(1+gc)]  # All rows, but exclude column of ghost cells
    #sol = U_0_sol
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

    method = 'w'
    # 's' for Pseudo Spectral Method
    # 'w' for Weno schemes

    U_0, x, t, dt, dx = init_cond(printout=True, method=method)

    print(f'U_0 = {U_0}')

    # Test
    dUdt(U_0, 0)
    print(U_0[0,:].size)
    print(U_0[0,:])

    # Pseudo Spectral Method
    if method is 's':
        sol = rk3(U_0, t, dt)
        U = sol
        plot_u_t(x, t, U)

    # WENO schemes:
    elif method is 'w':
        sol = rk3w(U_0, t, dt, dx)
        U = sol
        #for i in range(U[:, 0].size):
        #    U[i] += 0.1*i
        print(U)
        print(U[0])
        plot_u_t(x, t, U[:, 0, :], title=r'Density $\rho$ (g/mm$^3$)')
        plot_u_t(x, t, U[:, 1, :], title=r'Velocity $u$ (mm/s$^-1$)')
        #CS = plot.contour(x, t, U)
        #cbar = plot.colorbar(CS)

        #plot.contour(t, t, t)
        #plot.show()