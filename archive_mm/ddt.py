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
def r_phi(p, phi):  # units validated
    """
    Compaction rate (equation 20)
    :param p: Pressure (GPa)
    :param phi: Porosity
    :return: r_phi  # compaction rate in us-1
    """
    #return 0.0
    rate_phi = k_phi * (p - p_0 - P_h * (1 - np.sqrt((phi_0*(1 - phi))
                                                     / (phi*(1 - phi_0))
                                                     )
                                         )
                        )

    rate_phi = np.nan_to_num(rate_phi)

    return rate_phi

# Reaction rate
def r_lambda(p, lambd):  # units validated
    """
    Reaction rate (Equation 22)
    :param p: Pressure (GPa)
    :param lambd: Reaction progression \in [0, 1]
    :return: r_lambda  # Reaction rate in in us-1
    """
    #print(f'np.heaviside(p - p_ign, 1) = {np.heaviside(p - p_ign, 1)}')
    #return 0.0
    rate_lambda = np.heaviside(p - p_ign, 1) * k_r * (p/p_cj)**mu * (1 - lambd)**upsilon

    if 0:
        print(f'rate_lambda = {rate_lambda}')
        print(f'np.heaviside(p - p_ign, 1)= {np.heaviside(p - p_ign, 1)}')
        print(f'k = {k}')
        print(f'p = {p}')
        print(f'p_cj = {p_cj}')
        print(f'mu = {mu}')
        print(f'(p/p_cj)**mu = {(p/p_cj)**mu}')
        print(f'(1 - lambd)**upsilon= {(1 - lambd)**upsilon}')
    rate_lambda = np.nan_to_num(rate_lambda)  #TODO: SHOULDN'T NEED THIS!!!
    return rate_lambda


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
#def init_cond(e_0, printout=True, x=None):
def init_cond(x=None, gc=2, printout=True):
    # Discretization parameters
    # Space, if not specified
    if x is None:
        x = np.linspace(0, L, N)  # A vector of discrete points in space

    # Unit conversions
    #E_0 = 0.01 * e_0  # kJ g-1 --> 0.01 cm2 us-2
    E_0 = e(p_0, v_0, lambd_0, phi_0)  #

    # Initialize U container
    U_0 = np.zeros([5, x.size + 2 * gc])  # weno


    # Density
    U_0[0] = rho_0  # g/mm3 # Initial density 'rho' in tube

    # Momentum
    U_0[1] = rho_0 * u_0  # g/mm3 * mm/us Initial momentum 'rho * v'
    U_0[1, 0:(gc + 1)] = rho_0 * u_0_1  # Initial momemtum 'rho * u' on the boundary
    #U_0[1, 0:(gc + 3)] = rho_0 * u_0_1  # Initial momemtum 'rho * u' on the boundary
    # Energy
    #TODO: Do we need this?    e_0 = e(p_0, v_0, lambd_0, phi_0)  # Compute e_0
    U_0[2] = rho_0 * (E_0 + (u_0**2)/2.0)  # Initial energy 'rho * (e + u**2/2.0)'
    #U_0[2, 0] = rho_0 * (E_0 + (u_0_1**2)/2.0)  # Initial energy 'rho * (e + u**2/2.0)
    U_0[2, 0:(gc + 1)] = rho_0 * (E_0 + (u_0_1**2)/2.0)   # Initial energy 'rho * (e + u**2/2.0)
    #U_0[2, 0:(gc + 3)] = rho_0 * (E_0 + (u_0_1**2)/2.0)   # Initial energy 'rho * (e + u**2/2.0)
    if printout:
        #print(f'e_0 = {e_0} cm2 us-2')
        #print(f'e_0 = {e_0 *1e2} kJ g-1')
        print(f'e(p_0={p_0}, v_0={v_0}, lambd=_0{lambd_0}, phi=_0{phi_0}) '
              f'= {e(p_0, v_0, lambd_0, phi_0)} kJ g-1')
        print(f'e(p_0*1e-2={p_0*1e-2}, v_0={v_0}, lambd_0={lambd_0}, phi_0={phi_0}) '
              f'= {e(p_0*1e-2, v_0, lambd_0, phi_0)} kJ g-1')
       # e(p, v, lambd, phi):
       #  phi_0
        #print(f'U_0[2] = {U_0[2]}')
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
    F[0] = U[1]  # 'rho * u' (g mm-2 us-1)
    # Compute the velocity at every element
    u = U[1] / U[0]  # u (mm s-1) = rho * u / rho
    if 0:
        print(f'-')
        print(f'-')
        print(f'u = {u}')
        print(f'u.shape = {u.shape}')
        print(f'U.shape = {U.shape}')
    # Compute the energy (temperature) at every element  # Validated
    # Units of mm2 us-2
    E = U[2] / U[0] - (u ** 2) / 2.0
    # 'e = rho * ((e + u**2/2.0))/rho - u**2/2.0'

    # print(f'E = {E}')
    # Compute the porosity at every element
    PHI = U[3] / U[0]  # phi = (rho * phi) / rho
    # print(f'Phi = {Phi}')
    if (PHI > 1).any():
        #print(f'WARNING: np.any(PHI) > 1')
        PHI = np.minimum(PHI, np.ones_like(PHI))

    # Compute the reaction progress at every element
    LAMBD = U[4] / U[0]  # lambd = (rho * lambd) / rho
    # print(f'Lambd = {Lambd}')
    if (LAMBD > 1).any():
        #print(f'WARNING: np.any(LAMBD) > 1')
        # TODO: This should never be above 1.0, but it is, check with S curves
        #       for now we force lambda to be max 1.0
        LAMBD = np.minimum(LAMBD, np.ones_like(LAMBD))

    F[3] = F[0] * PHI  # 'rho * u * phi'  # also U[1] * U[3]/U[0]
    F[4] = F[0] * LAMBD  # 'rho * u * Lambd'  # also U[1] * U[4]/U[0]

    # Compute the pressure
    # Compute the specific volume from the density
    V = U[0] ** (-1)  # specific volume (mm3 g-1)
    # TODO: Try to vectorize
    ## P = p_from_e(E, V, LAMBD, PHI)
    P = np.zeros(np.shape(U)[1])
    for ind in range(np.shape(U)[1]):  # TODO: Extremely slow
        P[ind] = p_from_e(E[ind], V[ind], LAMBD[ind], PHI[ind])
        #P[ind] = p_from_e_no_reaction(E[ind], V[ind], PHI[ind])
    # print(f'P = {P}')

    #TODO: SHOULD NOT BE NEEDED:
    P = np.maximum(P, np.zeros_like(P))

    F[1] = U[0] * (u ** 2) + P  # rho * u^2 + p  (g mm-2 us-2)
    # print(f'F[1] = {F[1]}')

    F[2] = F[0] * (E + u ** 2 / 2.0 + P / U[0])  # (g mm-1 us-3)

    pp = (u, E, PHI, LAMBD, V, P)
    if 0:
        print('-')
        print('-')
        print(f'F[1] = {F[1]}')
        print(f'U[0] = {U[0]}')
        print(f'U[1] = {U[1]}')
        print(f'P = {P}')
        print(f'u= {u }')

    if 0:
        print('='*30)
        print('='*30)
        print(f'F = {F}')
        print('='*30)
        print('='*30)

    return F, pp

def s(U, F, pp):
    """
    Compute s, given U
    :param U:
    :return:
    """

    S = np.zeros_like(U)

    (u, E, PHI, LAMBD, V, P) = pp

    #P = 1e2 * P  # Convert pressure from g cm-1 us-2 back to GPa
    P = P  # Convert pressure from g cm-1 us-2 back to GPa

    #### REACTION COMPUTATION
    R_phi = r_phi(P, PHI)
    # print(f'R_phi = {R_phi}')
    # print(f'r_phi(P[0], Phi[0]) = {r_phi(P[0], Phi[0])}')
    # print(f'r_phi({P[0]}, {Phi[0]}) = {r_phi(P[0], Phi[0])}')
    R_lambd = r_lambda(P, LAMBD)
    # print(f'R_lambd = {R_lambd}')
    S[3] = U[0] * R_phi  # rho * r_phi
    S[4] = U[0] * R_lambd  # rho * r_lam

    #TODO:
    C = np.ones_like(U)
    #C[:, :] = u
    for ind, cC in enumerate(C):
        pass#C[ind, :] = u

    if 0:
        print('=-'*30)
        print('=-'*30)
        print(f'S = {S}')
        print('=-'*30)
        print('=-'*30)

    #print(f'S = {S}')
    return S, C

# Boundary condition
def BC(U, x, t=None):
    # BCs?
   # return U
    if 1:
        #print(f'U before = {U}')
        for ind in range(5):
            U[ind, 0:gc] = U[ind, gc]#.T
            #U[ind, 0:gc] = 0.0#.T
            #U[ind, 0:gc] = U[ind, (gc + 1)]#.T
            #U[ind, 0:gc] = 0.0
            #U[ind, -gc:] = U[ind, -(gc+1)]#.T
            #U[ind, -gc:] = 0.0
        #print(f'U after BC = {U}')

        #U[1, 0:(gc)] = 0.0  # Set boundary velocity to 0
    # Periodic boundary conditions
    if 0:
        for ind in range(3):
            #print(f'U[ind, {(N-(gc + 1))}:{N}] = {U[ind, (N-(gc + 1)):N]}')
            U[ind, 0:(gc + 1)] = U[ind, (N-(gc + 1)):N]
            #print(f'U[ind, {gc}:{(gc + gc + 1)}] = {U[ind, -gc:(gc + gc + 1)]}')
            U[ind, -(gc + 1):] = U[ind, gc:(gc + gc + 1)]  # .T

    return U

def BC_flux(dFdx, xc, gc, t=None):
    # Use "outflow" boundary conditions proposed in:
    # http://physics.princeton.edu/~fpretori/Burgers/Boundary.html
    # print(f'dFdx = {dFdx}')
    # print(f'self.xc = {self.xc}')

    if 0:
        #dFdx[:,gc] = -dFdx[:, gc + 1]
        for i_gc in range(gc):
            dFdx[:, i_gc] = dFdx[:, gc + 1]
        #    dFdx[:, -(i_gc + 1)] = dFdx[:, -(gc + 1)]

    # use second order BC fit
    if 0:
        # print(f'self.xc[self.gc + 1:self.gc + 3] = {self.xc[self.gc + 1:self.gc + 4]}')
        # print(f'dFdx[:, self.gc + 1:self.gc + 3] = {dFdx[:, self.gc + 1:self.gc + 4]}')

        cells = 2  # cells to extrapolate
        for ind in range(3):
            #print('-')
            #print(f'c[gc:gc + 2] = {xc[gc:gc + cells]}')
            #print(f'dFdx[ind, gc:gc + 2] = {dFdx[ind, gc:gc + cells]}')

            #print('-')
            z = np.polyfit(xc[gc:gc + cells],
                           dFdx[ind, gc:gc + cells],
                           1)
            p = np.poly1d(z)

            dFdx[ind, 0:gc] = p(xc[0:gc])
            #print(f'dFdx[ind, 0:gc] = p(xc[0:gc]) = {p(xc[0:gc])}')

            z = np.polyfit(xc[-(gc + cells):],
                           dFdx[ind, -(gc + cells):],
                           1)
            p = np.poly1d(z)
            dFdx[ind, -gc:] = p(xc[-gc:])

            # z = np.polyfit(self.xc[self.gc+:self.gc + 4],
        #                dFdx[ind, self.gc + 1:self.gc + 4], 2)
        # p = np.poly1d(z)

    # set i=0 to i=1 (no flux at i=0, see above)
    return dFdx


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
    prog_bar = IncrementalBar('Finished simulation. '
                              'Computing plot variables...', max=len(t))

    for tind, time in enumerate(t):

        U = Usol[tind, :, :]

        # Compute physical values from U state vectors
        F, pp = f(U)
        S, C = s(U, F, pp)
        (u, E, PHI, LAMBD, V, P) = pp

        Res[tind, 0, :] = P
        Res[tind, 1, :] = U[0]
        Res[tind, 2, :] = u
        Res[tind, 3, :] = PHI
        Res[tind, 4, :] = LAMBD
        prog_bar.next()


    Pressure_plot = np.minimum(Res[:, 0, :]*1e3, np.ones_like(Res[:, 0, :])*50)
    plot_u_t(x, t, Pressure_plot,
                  #title=r'Pressure $P$ (GPa) $\times 10^{-3}$', fign=0)
                  title=r'Pressure $P$ (GPa)', fign=0)

    Density_plot = np.minimum(Res[:, 1, :]*1e3, np.ones_like(Res[:, 0, :])*5)
    plot_u_t(x, t, Density_plot,
                  #title=r'Density $\rho$ (g/mm$^3$)', fign=1)
                  title=r'Density $\rho$ (g/cm$^3$)', fign=1)

    Velocity_plot = np.minimum(Res[:, 2, :], np.ones_like(Res[:, 0, :])*5)
    Velocity_plot = np.maximum(Velocity_plot, np.ones_like(Res[:, 0, :])*-5)
    plot_u_t(x, t,  Velocity_plot,
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
    x = x  # mm
    #xscale = 8.4/2.0 * (0.89)
    #yscale = xscale*5.5/8.4
    yscale = 8.4/2.0 * (0.89)*5.5/8.4
    xscale = yscale * 8.4 / 5.5 * 1.18
    plt.figure(fign, figsize=(xscale, yscale))
    plt.plot()
    #plt.pcolor(x, t, U, cmap='RdBu')
    #plt.pcolor(x, t, U, cmap='inferno')
    plt.pcolor(x, t, U, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x (mm)')
    plt.ylabel(r'time ($\mu$s)')
    #plt.axis('scaled')
    #plt.set_size_inches(18.5, 10.5)
    #plt.show()

if __name__ == "__main__":
    solver = WRKR(f, s, N=N, x0=0.0, xf=L, t0=0.0,
                  #tf=tf, #dt=0.5*tf,
                  tf=tf, #dt=0.5*tf,
                  flux_bc=BC_flux,
                  bc=BC,
                  dim=5,
                  #k=k
                  k=50
                  )

    gc = solver.gc
    # Compute e_0
    #e_0 = e_0_guess
    #e_0 = e(p_0, v_0, lambd_0, phi_0)

    #U_0 = init_cond(e_0, printout=True, x=solver.x)
    U_0 = init_cond(x=solver.x, gc=solver.gc)
    # Solve
    Urk3, solrk3 = solver.rk3(U_0,)  # self.U_0_sol
    #Ue, sole = solver.euler(U_0)  # self.U_0_sol
    U = solrk3  # Solution without ghost cells
    print(f'Finished simulation, plotting results...')
    # All rows, but exclude column of ghost cells:
    if 1:
        U = U[:, :,gc:-(gc)]
        plot_all_results(U, solver.x, solver.t)
    else:  # plot with ghost cells
        plot_all_results(U, solver.xc, solver.t)

    plt.show()
