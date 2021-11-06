import numpy as np
from schemes import ENOweights, nddp, ENO, WENO
from progress.bar import IncrementalBar


class WRKR():
    def __init__(self, f, s, bc=None, flux_bc=None, c_func=None,
                 N=81, x0=0.0, xf=2.0, t0=0.0, tf=0.5,
                 dt=None, dim=1,
                 k=3, **param_dict):
        self.f = f  # Vector function f
        self.s = s  # Vector function s
        self.bc = bc  # Dirichlet boundary condition u function for ghost cells
        self.flux_bc = flux_bc  # Neumann boundary condition function for ghost cells
        self.c_func = c_func
        self.param_dict = param_dict
        # k number of weights Order= 2*k-1
        # Domain
        self.dim = dim
        self.N = N  # Number of discrete spatial elements
        self.x = np.linspace(x0, xf, N)
        self.dx = (self.x[1] - self.x[0])
        # Time domain
        # Simulation
        if dt == None:
                #self.dt =  (self.dx) ** (5 / 4.0) #* 0.1
                #     self.dt = 0.5* (self.dx) ** (5 / 4.0) #* 0.1
                self.dt = 0.5*0.5* (self.dx) ** (5 / 4.0) #* 0.1
                #self.dt = 0.1* 0.5*0.5* (self.dx) ** (5 / 4.0) #* 0.1
              #         self.dt = 0.5* 0.1* 0.5*0.5* (self.dx) ** (5 / 4.0) #* 0.1
             #   self.dt = 0.5*0.5* 0.1* 0.5*0.5* (self.dx) ** (5 / 4.0) #* 0.1
                #self.dt = 0.05 * (self.dx) ** (5 / 3.0) #* 0.1
        else:
            self.dt = dt

        print(f'self.N = {self.N}')
        print(f'self.dt = {self.dt}')
        self.t = np.arange(t0, tf, self.dt)

        # Initial conditions
        self.U_ic = np.zeros(N)  # numpy function ones()
        #self.U_ic[int(.5 / self.dx): int(1 / self.dx + 1)] = 2

        # WENO parameter
        self.k = k
        # WENO Schemes:
        self.gc = k - 1  # number of ghost cells
        #self.gc = 50  # number of ghost cells
        gc = self.gc
        #print(f'gc = {gc}')
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

        #print(f'self.xc = {self.xc}')
        #print(f'self.xc.shape = {self.xc.shape}')



    def dFdx(self, F, C, WENO=True, t=None):
        """
        Computes the numerical flux values 1/dx()

        The WENO M/Z method from Peng et. al. (2019)
        :param F:
        :param C:
        :param WENO: If WENO is False, compute ENO scheme
        :return:
        """
        # Solver: parameters
        ep = 1e-6  # parameter to avoid division by zero
        q = 1#2  # q power parameter?
        #q = 3  # q power parameter?

        dFdx = np.zeros_like(F)
        if 0:
            print(f'F = {F}')
            print(f'F[:, :-4] = {F[:, :-4]}')
            print(f'F[:, 1:-3] = {F[:, 1:-3]}')
            print(f'F[:, 2:-2] = {F[:, 2:-2]}')
            print(f'F[:, 3:-1] = {F[:, 3:-1]}')
            print(f'F[:, 4:] = {F[:, 4:]}')

        fi_n2 = F[:, :-4]  # f_{i - 2}   (this is u_{i - 2} in the article)
        fi_n1 = F[:, 1:-3]  # f_{i - 1}
        f_i = F[:, 2:-2]  # f_{i}
        fi_p1 = F[:, 3:-1]  # f_{i + 1}
        fi_p2 = F[:, 4:]  # f_{i + 2}
        # f_{0, i + 1/2}
        f_0_i_p1_2 = (1/3.0)*fi_n2 - (7/6.0)*fi_n1 + (11/6.0)*f_i
        # f_{1, i + 1/2}
        f_1_i_p1_2 = -(1/6.0)*fi_n1 + (5/6.0)*f_i + (1/3.0)*fi_p1
        # f_{2, i + 1/2}
        f_2_i_p1_2 = (1/3.0)*f_i + (5/6.0)*fi_p1 - (1/6.0)*fi_p2

        ########################################################################
        ## Linear ENO:
        # Linear weights)
        c_0 = 0.1
        c_1 = 0.6
        c_2 = 0.3
        if not WENO:
            # f_{i + 1/2}  (Linear ENO)
            f_i_p1_2 = c_0 * f_0_i_p1_2 + c_1 * f_1_i_p1_2 + c_2 * f_2_i_p1_2
        ########################################################################
        else:
            # Smoothness indicators:
            beta_0 = (13/12.0)*(fi_n2 - 2*fi_n1 + f_i)**2 \
                     + (1/4.0)*(fi_n2 - 4*fi_n1 + 3*f_i)**2
            beta_1 = (13/12.0)*(fi_n1 - 2*f_i + fi_p1)**2 \
                     + (1/4.0)*(fi_n1 - fi_p1)**2
            beta_2 = (13/12.0)*(f_i - 2*fi_p1 + fi_p2)**2 \
                     + (1/4.0)*(3*f_i - 4*fi_p1 + fi_p2)**2

            # 5th order WENO Z scheme
            if 1:
                # Borges et al. parameter:
                tau_5 = np.abs(beta_0 - beta_2)

                alpha_0 = c_0*(1 + (tau_5/(beta_0 + ep))**q)
                alpha_1 = c_1*(1 + (tau_5/(beta_1 + ep))**q)
                alpha_2 = c_2*(1 + (tau_5/(beta_2 + ep))**q)
            # 5th order WENO-M:
            elif 0:
                alpha_0 = c_0/(beta_0 + ep)**q
                alpha_1 = c_1/(beta_1 + ep)**q
                alpha_2 = c_2/(beta_2 + ep)**q

            sigma_alpha = alpha_0 + alpha_1 + alpha_2
            omega_0 = alpha_0/sigma_alpha
            omega_1 = alpha_1/sigma_alpha
            omega_2 = alpha_2/sigma_alpha
            f_i_p1_2 = omega_0*f_0_i_p1_2 + omega_1*f_1_i_p1_2 + omega_2*f_2_i_p1_2
        ########################################################################
        # Numerical flux = f_i = f_{i + 1/2} - f_{i - 1/2}
        #print(f'f_i_p1_2.shape = {f_i_p1_2.shape}')
        self.flux = np.zeros_like(F[:, 2:-2])
        self.flux[:, 1:] = f_i_p1_2[:, 1:] - f_i_p1_2[:, 0:-1]

        # TODO: What do we do with the cell value at i = 0??? It is not possible
        #       to calculate a F_{i-1/2} value, Equation 12 works for f_{i+1/2}
        #       but we need f_{i-2}, f_{i-1}, f_{i}, F_{i+1}, F_{i+2}.
        #       so at i = -1: f_{i+1/2} = f_{-1/2} we step a f_{i-2} = f_{-3}
        #       value. For now we set it to zero (and modify in BC):
        #self.flux[:, 0] = np.zeros_like(F[:, 0])
        # Set all rows and all columns (excluding ghost cells) to flux
        dFdx[:, 2:-2] = self.flux
        #dFdx[:, self.gc] = dFdx[:, self.gc + 1]

        # Boundary conditions, if any:
        if self.flux_bc is not None:
            dFdx = self.flux_bc(dFdx, self.xc, self.gc, t=t)
            #self.flux[:, 0] = -(f_i_p1_2[:, 1] - f_i_p1_2[:, 0])

        #print(f'dFdx = {dFdx}')
        return dFdx/self.dx

    def Alpha(self, U, F, C=None):
        """
        Compute Alpha values for Lax-Friedrich splitting
        :param U:
        :param F:
        :return:
        """
        if C is not None:
            Alpha = np.zeros(self.dim)
            Alpha[:] = C
            return Alpha

        Alpha = np.zeros(self.dim)

        if 0:
            for ind, FI in enumerate(F):
                # print(f'U[ind] = {U[ind]}')
                # print(f'U[ind, :-1] = {U[ind, :-1]}')
                # print(f'U[ind, 1:] = {U[ind, 1:]}')
                # TODO: Should exclude ghost cells?
                dF = F[ind, 1:] - F[ind, :-1]
                dx = self.dx

                dFdX = dF / self.dx
                #  print(f'dFdU = {dFdU}')
                dFdX = np.nan_to_num(dFdX)
                Alpha[ind] = np.max(np.abs(dFdX))

                # Alpha[ind] = np.max(np.abs((dF + ep) / (dU + ep)))  # + 1.4
            # Alpha[ind] = np.max(np.abs(U[ind]))  # + 1.4

            # for u, f in zip(U[ind], F[ind]):
            #    dudf =
            #    print(u)
            #    print(f)
            #print(f'Alpha = {Alpha}')
            return Alpha

        ep = 1e-1  # 1.0#0.0#1e-6
        #ep = 5  # 1.0#0.0#1e-6

        for ind, FI in enumerate(F):
            if 0:
                #print(f'U[ind] = {U[ind]}')
                #print(f'U[ind, :-1] = {U[ind, :-1]}')
                #print(f'U[ind, 1:] = {U[ind, 1:]}')
                #TODO: Should exclude ghost cells?
                dF = F[ind, 1:] - F[ind, :-1]
                dU = U[ind, 1:] - U[ind, :-1]

                #print(f'dU = {dU}')
                #print(f'dF = {dF}')
                #print(f'dU/dF = {dU/dF}')

                #Alpha[ind] = np.max(np.abs((dU + ep) / (dF + ep)))  # + 1.4
             #   print(f'dU = {dU}')
             #   print(f'dF = {dF}')
                #dUdF = dU/dF
                #dUdF = dU/dF
                #dUdF = np.nan_to_num(dUdF)
                #Alpha[ind] = np.max(np.abs(dUdF))
                #dFdU = dF/dU
              #  dFdU = (dF + ep)/(dU + ep)
                #dFdU = (dF)/(dU)
                dFdU = (dF)/(self.dx)
                #print(f'dFdU = {dFdU}')
                dFdU = np.nan_to_num(dFdU)
               # print(f'dFdU = {dFdU}')
                Alpha[ind] = np.max(np.abs(dFdU))

            # Value from Peng (2019)
            if 1:
                gamma = 1.4
                rho = U[0]
                #print(f'rho  {rho}')
                u = U[1] / U[0]  # rho * u / rho
                # u = np.divide(U[1], U[0])  # rho * u / rho
                E = U[2]
                #print(f'u = {u}')
                p = (E - 0.5 * rho * (u ** 2)) * (gamma - 1)
                #print(f'E = {E}')
                #print(f'p = {p}')
                c = np.sqrt(gamma*p/rho)  # Speed of sound c
                Alpha[ind] = np.max(np.abs(U) + c)
                Alpha[ind] = np.nan_to_num(Alpha[ind])

            if 0:
                dFdU = np.gradient(dF, dU#, dtype=float
                                   )
                print(f'dFdU = {dFdU}')
                Alpha[ind] = np.max(np.abs(dFdU))

            #Alpha[ind] = np.max(np.abs((dF + ep) / (dU + ep)))  # + 1.4
           # Alpha[ind] = np.max(np.abs(U[ind]))  # + 1.4

            #for u, f in zip(U[ind], F[ind]):
            #    dudf =
            #    print(u)
            #    print(f)
        #print(f'Alpha = {Alpha}')
        return Alpha

    def dLdt(self, U, t=None):

        F, pp = self.f(U)
        S, C = self.s(U, F, pp)

        # Lax-Friedrich's splitting
        A = self.Alpha(U, F)
        Fp = np.zeros_like(F)
        Fn = np.zeros_like(F)
        for ind, FI in enumerate(F):
            #print(f'A[ind] = {A[ind]}')
            Fp[ind] = 0.5 * (F[ind] + A[ind] * U[ind])
            Fn[ind] = 0.5 * (F[ind] - A[ind] * U[ind])

        if 0:
            print(f'F = {F}')
            print(f'A = {A}')
            print(f'Fp = {Fp}')
            print(f'Fn = {Fn}')

        dFpdx = self.dFdx(Fp, C)
        Fn_flipped = np.flip(Fn, axis=1)
        #dFndx = self.dFdx(-Fn_flipped, C)
        dFndx = self.dFdx(Fn_flipped, C)
        dFndx = -np.flip(dFndx, axis=1)
        #dFndx = np.flip(dFndx, axis=1)
        #print(f'dFndt = {dFndt}')
        #print(f'dFpdt = {dFpdt}')
        if 0:
            dFndx = self.dFdx(Fn, C)
        dFdx = (dFpdx + dFndx)
        dFdx = dFdx
        #TEST DELETE THIS:
        if 0:
            dFdx[:, self.gc] = 0.0
            dFdx[:, -(self.gc+1)] = 0.0
        #print(f'dFdx = {dFdx}')
        return S - dFdx

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
            self.t_c = t_c
            # Compute boundary conditions:
            if self.bc is None:
                pass
            else:
                U = self.bc(U, self.xc, t_c)  # Apply boundary conditions
            #
            Uc = U.copy()
            # Compute U_1
            dLdt = self.dLdt(U, t_c)  #dLdt = S - dFdt
            U_1 = Uc + self.dt * dLdt
            # Compute U_2
            dLdt = self.dLdt(U_1, t_c)
            U_2 = 3/4.0 * Uc + 1/4.0 * U_1 + 1/4.0 * self.dt * dLdt
            # Compute U
            dLdt = self.dLdt(U_2, t_c)
            U = 1/3.0 * Uc + 2/3.0 * U_2 + 2/3.0 * self.dt * dLdt

            ind += 1
            self.U_0_sol[ind] = U
            prog_bar.next()

            if 0:
                from matplotlib import pyplot as plt
                plt.figure(2)
                plt.plot(2)
                plt.plot(self.xc, U[0, :].T, 'x-',
                         label='RK3 1 (rho)')
                plt.legend()
                plt.show()

        prog_bar.finish()
        #sol = self.U_0_sol[:, :, self.gc:-(self.gc)]  # All rows, but exclude column of ghost cells
        sol = self.U_0_sol  # All rows, but exclude column of ghost cells

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
            self.t_c = t_c

            # Compute boundary conditions:
            if self.bc is None:
                pass
            else:
                U = self.bc(U, self.xc, t_c)  # Apply boundary conditions

            dLdt = self.dLdt(U, t_c)
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
