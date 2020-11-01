"""
https://scicomp.stackexchange.com/questions/20054/implementation-of-1d-advection-in-python-using-weno-and-eno-schemes

"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from schemes import ENOweights, nddp, ENO, WENO



if __name__ == "__main__":
    # Domain
    N = 81  # Number of discrete spatial elements
    x = np.linspace(-0.4, 0.4, N)
    dx = (x[1] - x[0])
    print(f'dx = {dx}')

    dx = 2. / (N - 1)
    print(f'dx = {dx}')

    # Time domain
    # Simulation
    t0 = 0.0  # us (microseconds) # Simulation start time
    #dt = (dx) ** (5 / 4.0)
    dt = .02
    tf = t0 + dt
    tf = 0.5
    t = np.arange(t0, tf, dt)

    # Initial conditions
    u_ic = np.zeros(N)  # numpy function ones()
    u_ic[int(.5 / dx): int(1 / dx + 1)] = 2

    # WENO parameter
    k = 3  # number of weights Order= 2*k-1

    # WENO Schemes:
    gc = k - 1  # number of ghost cells
    gcr = x[-1] + np.linspace(1, gc, gc) * dx
    gcl = x[0] + np.linspace(-gc, -1, gc) * dx
    xc = np.append(x, gcr)
    xc = np.append(gcl, xc)
    uc = np.append(u_ic, u_ic[-gc:])
    uc = np.append(u_ic[0:gc], uc)
    # gs = np.zeros((N + 2 * gc, nt))
    flux = np.zeros(N + 2 * gc)

    if 1:
        for n in range(1, nt):
            un = uc.copy()
            # for i in range(1,nx):
            for i in range(2, nx):
                xloc = xc[i - (k - 1):i + k]
                floc = c * uc[i - (k - 1):i + k]
                # f_left,f_right = ENO(xloc,floc,k)
                f_left, f_right = WENO(xloc, floc, k)
                # uc[i] = un[i]-dt/dx*(f_right-f_left)
                flux[i] = 0.5 * (c + fabs(c)) * f_left + 0.5 * (
                            c - fabs(c)) * f_right

            for i in range(gc, nx - gc):
                if c > 0:
                    # uc[i] = un[i]-dt/dx*(flux[i]-flux[i-1])
                    uc[i] = uc[i] - dt / dx * (flux[i] - flux[i - 1])
                    U = uc
                    U_1 = U + dt / dx * (flux[i] - flux[i - 1])
                    U_2 = 3 / 4.0 * U + 1 / 4.0 * U_1 + 1 / 4.0 * dt / dx * (flux[i] - flux[i - 1])
                    U = 1 / 3.0 * U + 2 / 3.0 * U_2 + 2 / 3.0 * dt  / dx * (flux[i] - flux[i - 1])
                else:
                    # uc[i] = un[i]-dt/dx*(flux[i+1]-flux[i])
                    uc[i] = uc[i] - dt / dx * (flux[i + 1] - flux[i])

        if 0:  # old
            for n in range(1, nt):
                un = u.copy()
                for i in range(1, nx):
                    u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])

        plt.plot(x, u_ic, '--', label='Initial')
        # plt.plot(x, uc, '--')
        plt.plot(xc, uc, '--', label='Euler')
        plt.plot(xc, U, '-.', label='RK3')
        plt.legend()
        plt.show()