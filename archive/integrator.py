import numpy as np
from schemes import ENOweights, nddp, ENO, WENO
from progress.bar import IncrementalBar


class WRKR():
    def __init__(self, f, s, N=81, x0=0.0, xf=2.0, t0=0.0, tf=0.5, dt=None, dim=1,
                 k=3):
        self.f = f  # Vector function f
        self.s = s  # Vector function s
        # k number of weights Order= 2*k-1
        # Domain
        self.dim = dim
        self.N = N  # Number of discrete spatial elements
        self.x = np.linspace(x0, xf, N)
        self.dx = (self.x[1] - self.x[0])
        # Time domain
        # Simulation
        if dt == None:
                self.dt = (self.dx) ** (5 / 4.0) #* 0.1
                #sself.dt = 0.05 * (self.dx) ** (5 / 3.0) #* 0.1
        else:
            self.dt = dt

        self.t = np.arange(t0, tf, self.dt)

        # Initial conditions
        self.U_ic = np.zeros(N)  # numpy function ones()
        #self.U_ic[int(.5 / self.dx): int(1 / self.dx + 1)] = 2

        # WENO parameter
        self.k = k
        # WENO Schemes:
        self.gc = k - 1  # number of ghost cells
        gc = self.gc

        #TODO: Expand this to multiple dimensions
        gcr = self.x[-1] + np.linspace(1, gc, gc) * self.dx
        gcl = self.x[0] + np.linspace(-gc, -1, gc) * self.dx
        self.xc = np.append(self.x, gcr)
        self.xc = np.append(gcl, self.xc)
        self.uc = np.append(self.U_ic, self.U_ic[-gc:])
        self.uc = np.append(self.U_ic[0:gc], self.uc)
        # gs = np.zeros((N + 2 * gc, nt))
        self.flux = np.zeros(self.N + 2 * gc)
        self.c = 1  #TODO: TEMP

        print(f'self.xc = {self.xc}')
        print(f'self.xc.shape = {self.xc.shape}')

    def dFdt(self, F, C):
        # Solve U
        # Iterate over rows:
        dFdt = np.zeros_like(F)
        F_left = np.zeros_like(F)
        F_right = np.zeros_like(F)
        for ind, f in enumerate(F):
            alpha = max(np.abs(F[ind]) + C[ind])
            #for i in range(self.gc, self.N - 1 + self.gc + 1):
            for i in range(self.gc, self.N - 1 + self.gc + 1):
            #    print('-')
            #    print('-')
            #    print(f'i = {i}')
                #print(f'i - (self.k - 1) = {i - (self.k - 1)}')
                #print(f'i + self.k = {i + self.k}')
                #print(f'elf.xc[i - (self.k - 1):i + self.k] = {self.xc[i - (self.k - 1):i + self.k]}')
                # x stencil?
                xloc = self.xc[i - (self.k - 1):i + self.k]
                # f(u) = c * U
                floc = f[i - (self.k - 1):i + self.k]
                # Find fluxes f_left and f_right using a scheme
                # f_left,f_right = ENO(xloc,floc,k)
                f_left, f_right = WENO(xloc, floc, self.k)
             #   print(f'f_left = {f_left}')
             #   print(f'f_right  = {f_right }')
                F_left[ind, i] = f_left
                F_right[ind, i] = f_right

                # Compute flux, from c (TODO: Compute c)
                # In DDT c is basically the velocity? Since the variables
                # are multiplied as f(U) = c * U where c is the velocity in the
                # paper
                if 0:
                    self.c = C[ind, i]
                    self.flux[i] = 0.5*(self.c + np.fabs(self.c)) * f_left \
                                 + 0.5*(self.c - np.fabs(self.c)) * f_right
                elif 1:
                    #self.flux[i] = f_left - f_right
                    self.flux[i] = f_right - f_left
                    # self.flux[i] = 0.5*f_right + 0.5*f_left
                    #self.flux[i] = f_right + f_left
                    #self.flux[i] = f_right
                    #self.flux[i] = f_left
                    #print(f'self.flux[i] = {self.flux[i]}')
                    #f_plus = 0.5*(F[ind, i] + alpha * U[ind, i])
                    #f_neg = 0.5*(F[ind, i] - alpha * U[ind, i])
                    #if f_left - f_right >= 0:
                   #     self.flux[i] = f_left - f_right
                    #else:
                    #    self.flux[i] = f_right - f_left
                elif 0:
                    ap = max(C[ind, i], 0)
                    an = min(C[ind, i], 0)
                    self.flux[i] = ap * f_right + an * f_left
                elif 0:
                    self.flux[i] = f_right + f_left

            for i in range(self.gc, self.N - 1 + self.gc + 1):
                if 0:
                    dFdt[ind, i] = (self.flux[i] - self.flux[i - 1]) / self.dx
                    if i == (self.gc):
                        dFdt[ind, i] = (self.flux[i+1] - self.flux[i]) / self.dx
                elif 1:
                    dFdt[ind, i] = (self.flux[i]) / self.dx
                elif 0:
                    if (self.flux[i] - self.flux[i - 1]) > 0:
                        dFdt[ind, i] = (self.flux[i] - self.flux[i - 1]) / self.dx
                    else:
                        dFdt[ind, i] = (self.flux[i + 1] - self.flux[i]) / self.dx
        if 0:
            print('=' * 30)
            print('=' * 30)
            print(f'dFdt = {dFdt}')
            print('=' * 30)
            print('=' * 30)

        return dFdt

    def rk3(self, U_0):
        # TVD RK 3 solver
        U = np.atleast_2d(U_0.copy())
        #t_c = t0  # current time tracker.
        self.U_0_sol = U_0
        self.U_0_sol = np.zeros([self.t.size, self.dim, U[0, :].size])
        self.U_0_sol[0] = U_0  #np.atleast_2d(U_0)#[0]
        ind = 0  # solution index tracker

        prog_bar = IncrementalBar('Simulation progress', max=len(self.t))

        for t_c in self.t[1:]:
            # if c > 0:
            #   uc[i] = uc[i] - dt / dx * (flux[i] - flux[i - 1])
            # Compute new U_t+1
            Uc = U.copy()

            # Compute U_1
            F, pp = self.f(U)
            S, C = self.s(U, F, pp)
            dFdt = self.dFdt(F, C)
            dLdt = S - dFdt
            #print(f'dLdt = {dLdt}')
            U_1 = Uc + self.dt * dLdt
            # Compute U_2
            F, pp = self.f(U_1)
            S, C = self.s(U_1, F, pp)
            dFdt = self.dFdt(F, C)
            dLdt = S - dFdt
            U_2 = 3/4.0 * Uc + 1/4.0 * U_1 + 1/4.0 * self.dt * dLdt
            # Compute U
            F, pp = self.f(U_2)
            S, C = self.s(U_2, F, pp)
            dFdt = self.dFdt(F, C)
            dLdt = S - dFdt

            U = 1/3.0 * Uc + 2/3.0 * U_2 + 2/3.0 * self.dt * dLdt

            ind += 1
            self.U_0_sol[ind] = U
            #else:
            #   uc[i] = uc[i] - dt / dx * (flux[i + 1] - flux[i])
            prog_bar.next()
        prog_bar.finish()
        sol = self.U_0_sol[:, :, self.gc:-(self.gc)]  # All rows, but exclude column of ghost cells
        return U, sol


    def euler(self, U_0):
        # TVD RK 3 solver
        U = U_0.copy()
        #t_c = t0  # current time tracker.
        self.U_0_sol = np.atleast_2d(U_0)
        self.U_0_sol = np.zeros([self.t.size, self.dim, U[0, :].size])
        self.U_0_sol[0] = U_0#np.atleast_2d(U_0)#[0]
        ind = 0  # solution index tracker
        prog_bar = IncrementalBar('Simulation progress', max=len(self.t))
        for t_c in self.t[1:]:
            # if c > 0:
            #   uc[i] = uc[i] - dt / dx * (flux[i] - flux[i - 1])
            # Compute new U_t+1
            F, pp = self.f(U)
            S, C = self.s(U, F, pp)
            dFdt = self.dFdt(F, C)
            dLdt = S - dFdt
            #U = U + self.dt * self.dUdt(U)
            U = U + self.dt * dLdt
            ind += 1
            self.U_0_sol[ind] = U
            #else:
            #   uc[i] = uc[i] - dt / dx * (flux[i + 1] - flux[i])
            prog_bar.next()
        prog_bar.finish()
        sol = self.U_0_sol[:, :, self.gc:-(self.gc)]  # All rows, but exclude column of ghost cells
        return U, sol



if __name__ == "__main__":
    solver = WRKR()
    sol = solver.rk3()
