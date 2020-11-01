"""
https://scicomp.stackexchange.com/questions/20054/implementation-of-1d-advection-in-python-using-weno-and-eno-schemes

"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from schemes import ENOweights, nddp, ENO, WENO
from integrator import WRKR

def f(U):
    """
    Compute f, given U
    :param U:
    :return:
    """
    # self.c * U
    F = U
    pp = ()
    return F, pp


def s(U, F, pp):
    """
    Compute s, given U
    :param U:
    :return:
    """
    C = np.ones_like(U)
    S = np.zeros_like(U)
    return S, C


if __name__ == "__main__":
    N = 81
    solver = WRKR(f, s, N=N, x0=0.0, xf=2.0, t0=0.0,
                  tf=0.5, dt=.02
                  )
    # IC's
    U_0 = np.zeros(N)
    # U_0 = np.zeros([2, N])
    U_0[int(.5 / solver.dx): int(1 / solver.dx + 1)] = 2
    #sol = solver.rk3(U_0)  # self.U_0_sol
    U_0 = np.atleast_2d(U_0)
    # Add ghost cells
    U_0 = np.hstack((U_0, U_0[:, -solver.gc:]))
    U_0 = np.hstack((U_0[:, -solver.gc:], U_0))

    # Solve
    Urk3, solrk3 = solver.rk3(U_0)  # self.U_0_sol
    Ue, sole = solver.euler(U_0)  # self.U_0_sol
    #print(f'sol = {sol}')
    #fsol = self.U_0_sol
    print(f'solver.x = {solver.x.shape}')
    print(f'solver.xc = {solver.xc.shape}')
    print(f'Urk3[:, solver.gc:-(solver.gc)]. {Urk3[:, solver.gc:-(solver.gc)].shape}')
    if 1:
        #U_0 = np.atleast_2d(U_0)[:, solver.gc: -(1 + solver.gc)]
        #plt.plot(solver.x, U_0, '--', label='Initial')
        plt.plot(solver.xc, U_0[:].T, '--', label='Initial')
        # plt.plot(x, uc, '--')
        plt.plot(solver.x, Ue[:, solver.gc:-(solver.gc)].T, '--', label='Euler')
        #plt.plot(solver.xc, sol, '-.', label='RK3')
        #plt.plot(solver.x, sol[-1, :].T, '-.', label='RK3')
        plt.plot(solver.x, Urk3[:, solver.gc:-(solver.gc)].T, '-.', label='RK3')
        plt.legend()
        plt.show()