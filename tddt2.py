"""
Peng 2019

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

    # rho * u
    # rho * u
    p = 1
    rho = 1 + 0.2 * np.sin(np.pi * x)
    u = 1
    E = p / (gamma - 1) + 0.5 * rho * u ** 2

    return np.array([1 + 0.2 * np.sin(np.pi * x),
                    (1 + 0.2 * np.sin(np.pi * x) )*1,
                     E])

def BC(U, x, t=None):
    # BCs?
   # return U
    if 0:
        #print(f'U before = {U}')
        for ind in range(3):  # dim=3
            U[ind, 0:gc] = U[ind, gc]#.T
            #U[ind, 0:gc] = 0.0
            U[ind, -gc:] = U[ind, -(gc+1)]#.T
            #U[ind, -gc:] = 0.0
        #print(f'U after BC = {U}')

    if 1:
        p = 1
        rho = 1 + 0.2 * np.sin(np.pi * (x - t))
        u = 1
        E = p / (gamma - 1) + 0.5 * rho * u ** 2

        exact = np.array([rho,
                         rho*u,
                         E])

        #print(f'exact = {exact}')
        #print(f'exact.shape = {exact.shape}')
        for ind in range(3):  # dim=3
            #print(f'exact[{ind}] = {exact[ind]}')
            #print(f'exact[ind, 0:gc] = {exact[ind, 0:gc]}')
            #print(f'exact[ind, -gc:] = {exact[ind, -gc:]}')

            U[ind, 0:gc] = exact[ind, 0:gc]#.T
            #U[ind, 0:gc] = 0.0
            U[ind, -gc:] = exact[ind, -gc]#.T
            #U[ind, -gc:] = 0.0

        return U

    return U

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
    C = C + np.sqrt(gamma*p/rho)
    S = np.zeros_like(U)

    return S, C

def exact(x, t):
    U1 = np.ones_like(x)
    U2 = np.ones_like(x)

    # U = [rho, rho * u, E]
    # F = [rho*u, rho * u**2 + p, u * (E + p)]
    # E = p/(gamma - 1) + 0.5 * rho * u**2

    # rho * u
    # rho * u
    p = 1
    rho = 1 + 0.2 * np.sin(np.pi * (x - t))
    u = 1
    E = p / (gamma - 1) + 0.5 * rho * u ** 2

    return np.array([[rho],
                     [rho*u],
                     [E]])

if __name__ == "__main__":
    N = 20
    N = 50
    #N = 200
    #N = 81
   # N = 400
  #  N = 400
    tf = 0.05
    tf = 0.05999

    tf = 2
    tf = 0.2
    tf = 0.5
    #tf = 0.45
    tf = 0.2
    tf = 0.3
    tf = 0.35
    tf = 1.0
    tf = 1.2
    tf = 0.2
    #tf = 0.
    #tf = 1.0
   # tf = 0.4
    #tf = 2.0
    #tf = 0.6
   # tf = 0.06
    solver = WRKR(f, s, #bc=BC,
                  N=N, x0=0.0, xf=2.0, t0=0.0,
                  tf=tf, dim=3, #dt= 0.001
                  )
    # IC's
    #U_0 = np.zeros(N)
    #U_0 = np.zeros([2, N])
    #U_0[:, int(.5 / solver.dx): int(1 / solver.dx + 1)] = 2
    #sol = solver.rk3(U_0)  # self.U_0_sol
    #U_0 = np.atleast_2d(U_0)
    # Add ghost cells
    #U_0 = np.hstack((U_0, U_0[:, -solver.gc:]))
    #U_0 = np.hstack((U_0[:, -solver.gc:], U_0))
    # Solve
    U_0 = IC(solver.xc)
    print(f'U_0 = {U_0}')
    print(f'U_0[0] = {U_0[0]}')
    print(f'U_0[1] = {U_0[1]}')
    print(f'U_0[2] = {U_0[2]}')

    #  Urk3, solrk3 = solver.rk3(U_0)  # self.U_0_sol
    Urk3, solrk3 = solver.euler(U_0)  # self.U_0_sol
    #Ue, sole = solver.euler(U_0)  # self.U_0_sol
    #print(f'sol = {sol}')
    #fsol = self.U_0_sol
    #print(f'Ue.shape = {Ue.shape}')
    #print(f'sole.shape = {sole.shape}')
    print(f'solver.xc.shape = {solver.xc.shape}')
    print(f'solver.xc.shape = {solver.x.shape}')
    if 1:
        plt.figure(1)
        plt.plot(1)
        plt.plot(solver.xc, U_0[0, :].T, '.', label='Initial 1')
        plt.plot(solver.xc, U_0[1, :].T, 'x', label='Initial 2')
        plt.plot(solver.xc, U_0[2, :].T, 'o', label='Initial 3')
        plt.plot(solver.xc, exact(solver.xc, 0)[0].T, label='Exact 1, t = 0')
        plt.plot(solver.xc, exact(solver.xc, 0)[1].T, label='Exact 2, t = 0')
        plt.plot(solver.xc, exact(solver.xc, 0)[2].T, label='Exact 3, t = 0')

        plt.legend()
       # plt.show()

    if 1:
        print(f'solver.gc:-(solver.gc)].T = {Urk3[0, solver.gc:-(solver.gc)].T}')
        plt.figure(2)
        plt.plot(2)
        #plt.plot(solver.x, Urk3[0, solver.gc:-(solver.gc)].T, '^', label='RK3 1')
        plt.plot(solver.xc, Urk3[0, :].T, '^', label='RK3 1')
        #plt.plot(solver.x, Urk3[1, solver.gc:-(solver.gc)].T, 'x', label='RK3 2')
        plt.plot(solver.xc, Urk3[1, :].T, 'x', label='RK3 2')
        #plt.plot(solver.x, Urk3[2, solver.gc:-(solver.gc)].T, 'o', label='RK3 3')
        plt.plot(solver.xc, Urk3[2, :].T, 'o', label='RK3 3')
        plt.plot(solver.x, exact(solver.x, tf)[0].T, label=f'Exact 1, t = {tf}')
        plt.plot(solver.x, exact(solver.x, tf)[1].T, label=f'Exact 2, t = {tf}')
        plt.plot(solver.x, exact(solver.x, tf)[2].T, label=f'Exact 3, t = {tf}')

        plt.legend()
        plt.show()