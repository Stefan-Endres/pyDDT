"""
Rathan2016 example 3
"""
import numpy as np
import matplotlib.pyplot as plt
from schemes import ENOweights, nddp, ENO, WENO
from integrator import WRKR


def ic(x):
    u_i = np.zeros_like(x)
    for i, xi in enumerate(x):
        if -1 <= xi < 0:
            u_i[i] = -np.sin(np.pi * xi) - 0.5 * xi**3
        if 0 <= xi < 1:
            u_i[i] = -np.sin(np.pi * xi) - 0.5 * xi**3 + 1
    return u_i


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
    # Domain
    N = 120  # Number of discrete spatial elements
    solver = WRKR(f, s, N=120, x0=-0.4, xf=0.4, t0=0.0,
                  #tf=2.0,
                  tf=0.005,
                  )

    # Initial conditions
    U_0 = ic(solver.x)
    U_0 = np.append(U_0, U_0[-solver.gc:])
    U_0 = np.append(U_0[0:solver.gc], U_0)
    U_0 = np.atleast_2d(U_0)

    # Solve
    Urk3, solrk3 = solver.rk3(U_0)  # self.U_0_sol
    Ue, sole = solver.euler(U_0)  # self.U_0_sol
    if 1:
        #U_0 = np.atleast_2d(U_0)[:, solver.gc: -(1 + solver.gc)]
        #plt.plot(solver.x, U_0, '--', label='Initial')
        plt.plot(solver.xc, U_0.T, '--', label='Initial')
        # plt.plot(x, uc, '--')
        plt.plot(solver.x, Ue[:, solver.gc:-(solver.gc)].T, '--', label='Euler')
        #plt.plot(solver.xc, sol, '-.', label='RK3')
        #plt.plot(solver.x, sol[-1, :].T, '-.', label='RK3')
        plt.plot(solver.x, Urk3[:, solver.gc:-(solver.gc)].T, '-.', label='RK3')
        plt.legend()
        plt.show()
    else:
        plt.plot(x, u_ic, '--', label='u_ic')
        plt.plot(x, U, '--', label='U')
        plt.show()


