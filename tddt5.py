"""
Peng 2019 Sod problem
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from schemes import ENOweights, nddp, ENO, WENO
from integrator import WRKR

# parameters
gamma = 1.4
gc = 2
def IC(x):

   # U_0 = np.zeros([5, x.size + 2 * gc])  # weno
    # U = [rho, rho * u, E]
    # F = [rho*u, rho * u**2 + p, u * (E + p)]
    # E = p/(gamma - 1) + 0.5 * rho * u**2
    U_0 = np.zeros([3, x.size])
    # rho * u
    # rho * u
    for i, xi in enumerate(x):
        if xi <= 0:
            rho = 1.0
            u = 0.0
            p = 1.0
            E = p / (gamma - 1) + 0.5 * rho * u ** 2
            U_0[0, i] = rho
            U_0[1, i] = rho * u
            U_0[2, i] = E
        if xi > 0.0:
            rho = 0.125
            u = 0.0
            p = 0.1
            E = p / (gamma - 1) + 0.5 * rho * u ** 2
            U_0[0, i] = rho
            U_0[1, i] = rho * u
            U_0[2, i] = E

    return U_0

def f(U):
    """
    Compute f, given U
    :param U:
    :return:
    """
    F = np.zeros_like(U)
    # self.c * U
    # U = [rho, rho * u, E]
    # F = [rho*u, rho * u**2 + p, u * (E + p)]
    # E = p/(gamma - 1) + 0.5 * rho * u**2
    rho = U[0]
    u = U[1]/U[0]  # rho * u / rho
    E = U[2]
    p = (E - 0.5 * rho * u**2) * (gamma - 1)
  #  print(f'p = {p}')
   # print(f'rho = {rho}')
    F[0] = rho*u
    F[1] = rho * u**2 + p
    F[2] = u * (E + p)
    pp = (rho, u, E, p)
    return F, pp


def s(U, F, pp):
    """
    Compute s, given U
    :param U:
    :return:
    """
    (rho, u, E, p) = pp
    C = np.ones_like(U)
    #C = C + np.sqrt(gamma*p/rho)
    S = np.zeros_like(U)

    return S, C



if __name__ == "__main__":
    N = 81
    N = 200
    solver = WRKR(f, s, N=N, x0=-2.0, xf=2.0, t0=0.0,
                  #tf=0.014,
                  tf=0.14,
                  dt=0.002,
                  #dt=.02,
                  dim=3
                  )
    # IC's
    U_0 = IC(solver.xc)
    #print(f'U_0 = {U_0}')
    #print(f'U_0[0] = {U_0[0]}')
    #print(f'U_0[1] = {U_0[1]}')
    #print(f'U_0[2] = {U_0[2]}')

    Urk3, solrk3 = solver.rk3(U_0)  # self.U_0_sol
    # Ue, sole = solver.euler(U_0)  # self.U_0_sol
    # print(f'sol = {sol}')
    # fsol = self.U_0_sol
    # print(f'Ue.shape = {Ue.shape}')
    # print(f'sole.shape = {sole.shape}')
    #print(f'solver.xc.shape = {solver.xc.shape}')
    print(f'solver.xc.shape = {solver.x.shape}')
    if 1:
        plt.figure(1)
        plt.plot(1)
        plt.plot(solver.xc, U_0[0, :].T, '-', label='Initial 1')
        #plt.plot(solver.xc, U_0[1, :].T, 'x', label='Initial 2')
        #plt.plot(solver.xc, U_0[2, :].T, 'o', label='Initial 3')
        plt.legend()
    # plt.show()

    if 1:
        #print(
        #    f'solver.gc:-(solver.gc)].T = {Urk3[0, solver.gc:-(solver.gc)].T}')
        plt.figure(2)
        plt.plot(2)
        print(f' findal soluation = {Urk3[0, solver.gc:-(solver.gc)].T}')
        plt.plot(solver.x, Urk3[0, solver.gc:-(solver.gc)].T, '-',
                 label='RK3 1')
        #plt.plot(solver.x, Urk3[1, solver.gc:-(solver.gc)].T, 'x',
        #         label='RK3 2')
        #plt.plot(solver.x, Urk3[2, solver.gc:-(solver.gc)].T, 'o',
        #         label='RK3 3')

        plt.legend()
        plt.show()