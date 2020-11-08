import numpy as np
from scipy import optimize
from params import *

from shgo import shgo, SHGO

def obj_func(y):
    e_0 = y

    """WR-EOS"""
    def e(p, v, lambd, phi, e_0):
        """
        A combination of Equations 16 and 17
        :param p: pressure
        :param v: specific volume (cm3 g-1)
        :param lambd: Reaction progression \in [0, 1]
        :param phi: Porosity
        :return: e (kJ g-1  ?)
        """
        # First compute the volumes (Equation 16)
        v_p, v_ps, v_r, Phi = volumes(v, lambd, phi)

        # Using the computed volumes, compute the internal energy at v, p
        return (1 - lambd) * e_r(p / phi, v_r, e_0) + lambd * e_p(p, v_p)


    def volumes(v, lambd, phi):
        def denom(lambd, Phi):
            """
            Denominator of closure relationships in equation 16
            :param lambd:
            :param Phi: closure constant/function. Varies between 0.8 to 1.0
                        If lambd = 0 then Phi = 1, if lambd > 0 then Phi ~ 0.95
            :return:
            """
            return lambd + (1 - lambd) * Phi

        v_ps_g = v  # Initial guess value for v_ps
        v_r_g = v * phi  # Initial guess value for v_r
        Phi_g = 0.95  # Initial guess value for Phi
        v_p_g = Phi_g * v_ps_g  # Initial guess value for v_p
        #TODO: Update routine to use previous values in the element as guess
        x0 = [v_p_g, v_ps_g, v_r_g, Phi_g]
        x0 = np.array(x0) - 1e-2
        def system(x):
            v_p, v_ps, v_r, Phi = x
            return [v_p - (v / denom(lambd, Phi)),
                    v_ps - (Phi * v / denom(lambd, Phi)),
                    v_r - (phi * Phi * v / denom(lambd, Phi)),
                    Phi - v_ps/v_p
                    ]

        def jac(x):  # Jacobian computed with SymPy
            v_p, v_ps, v_r, Phi = x
            return np.array([[1, 0, 0, -v * (lambd - 1) / (Phi * (-lambd + 1) + lambd) ** 2],
                             [0, 1, 0, -Phi * v * (lambd - 1) / ( Phi * (-lambd + 1) + lambd) ** 2 - v / (Phi * (-lambd + 1) + lambd)],
                             [0, 0, 1, -Phi * phi * v * (lambd - 1) / (Phi * (-lambd + 1) + lambd) ** 2 - phi * v / (Phi * (-lambd + 1) + lambd)],
                             [v_ps / v_p ** 2, -1 / v_p, 0, 1]])

        sol = optimize.root(system, x0, jac=jac, method='hybr')
        return sol.x


    def p_from_e(energy, v, lambd, phi):
        """
        Compute the pressure, given energy, volume, lambd and phi
        :param energy: energy
        :param v: specific volume
        :param lambd: reaction progress
        :param phi: porosity
        :param guess: Initial guess for p and v
        :return: p, v  # pressure and volume
        """
        e = energy
        v_p, v_ps, v_r, Phi = volumes(v, lambd, phi)

        #TODO: Deleate reduntant computation
        P = phi*(gamma_p(v_p)*lambd*p_s_p(v_p)*v_r
                 - gamma_r(v_r)*lambd*p_s_r(v_r)*v_p
                 + gamma_r(v_r)*p_s_r(v_r)*v_p
                 + e*v_p*v_r
                 - e_s_p(v_p)*lambd*v_p*v_r
                 + e_s_r(v_r)*lambd*v_p*v_r
                 - e_s_r(v_r)*v_p*v_r
                 )/(
                gamma_p(v_p)*lambd*phi*v_r
                - gamma_r(v_r)*lambd*v_p
                + gamma_r(v_r)*v_p)

        P = phi*(gamma_p(v_p)*gamma_r(v_r)*e
                 - gamma_p(v_p)*gamma_r(v_r)*e_s_p(v_p)*lambd
                 + gamma_p(v_p)*gamma_r(v_r)*e_s_r(v_r)*lambd
                 - gamma_p(v_p)*gamma_r(v_r)*e_s_r(v_r)
                 - gamma_p(v_p)*lambd*p_s_r(v_r)*v_r
                 + gamma_p(v_p)*p_s_r(v_r)*v_r
                 + gamma_r(v_r)*lambd*p_s_p(v_p)*v_p)/(-gamma_p(v_p)*lambd*v_r
                                                       + gamma_p(v_p)*v_r
                                                       + gamma_r(v_r)*lambd*phi*v_p)
        return P

    def e_i(p, v, lambd, phi):
        """
        Compute the initial energy, given p_0, v_0, lambd_0 and phi_0
        :param p:
        :param v:
        :param lambd:
        :param phi:
        :return:
        """


    """Detonation products"""
    # EOS
    def e_p(p, v):
        """
        Equation A1 product energy
        :param p: pressure
        :param v: specific volume
        :return: energy e_p
        """
        return e_s_p(v) + (v/gamma_p(v))*(p - p_s_p(v))


    def p_p(e, v):
        """
        Equation A2 product pressure
        :param e: energy
        :param v: specific volume
        :return: pressure p_p
        """
        return p_s_p(v) + (gamma_p(v)/v)*(e - e_s_p(v))

    # Functionals
    def p_s_p(v):
        """
        Equation A3
        :param v: specific volume
        :return:
        """
        return p_c * ((0.5*(v/v_c)**n + 0.5*(v/v_c)**(-n))**(a/n)
                      / ((v/v_c)**(k_wr + a))) * ((k_wr - 1 + F_wr(v))/
                                                 (k_wr - 1 + a))


    def F_wr(v):
        """
        Equation A4
        :param v: specific volume
        :return:
        """
        return (2 * a * (v/v_c)**(-n))/((v/v_c)**n + (v/v_c)**(-n))


    def gamma_p(v):
        """
        Equation A5
        :param v: specific volume
        :return:
        """
        return k_wr - 1 + (1 - b) * F_wr(v)


    def e_s_p(v):
        """
        Equation A6
        :param v: specific volume
        :return: e_s_p
        """
        return e_c * ((0.5*(v/v_c)**n + 0.5*(v/v_c)**(-n))**(a/n)
                      / ((v/v_c)**(k_wr - 1 + a)))

    # Equation A7
    e_c = (p_c * v_c) / (k_wr - 1 + a)


    """Detonation reactants"""
    # EOS
    def e_r(p, v, e_0):
        """
        Equation A8 reactant energy
        :param p: pressure
        :param v: specific volume
        :return: energy e_p
        """
        return e_s_r(v, e_0) + (v/gamma_r(v))*(p - p_s_r(v))


    def p_r(e, v, e_0):
        """
        Equation A9 reactant pressure
        :param e: energy
        :param v: specific volume
        :return: pressure p_p
        """
        return p_s_r(v) + (gamma_r(v)/v)*(e - e_s_r(v, e_0))


    def p_s_r(v):
        """
        Equation A10
        :param v: specific volume
        """
        sigma_j = 0
        for j in range(1, 4):
            sigma_j += ((4 * B * y(v))**j / np.math.factorial(j))
        return p_hat * (sigma_j
                        + C * ((4*B*y(v))**4) / 24.0 # np.math.factorial(4) = 24
                        + (y(v)**2) / (1 - y(v))**4
                        )


    def y(v):
        return 1 - v/v_0

    p_hat = rho_0 * A**2 / (4 * B)


    def p_s_r_y(y):
        """
        Equation for integral
        """
        sigma_j = 0
        for j in range(1, 4):
            sigma_j += ((4 * B * y)**j / np.math.factorial(j))

        return p_hat * (sigma_j
                        + C * ((4*B*y)**4) / 24.0  # np.math.factorial(4) = 24
                        + (y**2) / (1 - y)**4
                        )


    def e_s_r(v, e_0):
        """
        Equation A11
        :param v: specific volume
        :return:
        """
        from scipy import integrate
        if v >= v_0:
            print(f"WARNING: v[={v}] >= v_0[={v_0}]")
        y_lim = y(v)
        INT = integrate.quad(p_s_r_y, 0, y_lim)
        if INT[1] >= 1e-1:
            print(f"WARNING: Integral error in e_s_r is high {INT}")
        #return v_0 * INT[0] + 7.07  #+ e_0  #TODO: Check
        print(f'e_0 in e_s_r = {e_0}')
        return v_0 * INT[0] + e_0  #TODO: Check


    def gamma_r(v):
        """
        Equation A12
        :param v: specific volume
        :return:
        """
        return gamma_0_r + Z * y(v)

    #lambd_0 = 0.1
    print(
        f'e(p_0*1e-2={p_0}, v_0={v_0}, lambd_0={lambd_0}, phi_0={phi_0}) '
        f'= {e(p_0, v_0, lambd_0, phi_0, e_0)} kJ g-1')
    print(f'obj = {e_0 - e(p_0, v_0, lambd_0, phi_0, e_0)} kJ g-1')
    return (e_0 - e(p_0, v_0, lambd_0, phi_0, e_0))**2

from matplotlib import pyplot as plt

res = shgo(obj_func, bounds=[(0, 1e10)], n=1000, sampling_method='sobol')
SHc = SHGO(obj_func, bounds=[(0, 1e1)])
SHc.HC.plot_complex()
print(res)

espan = np.linspace(0, 1000)
for et in espan:
    pass
    #print(f'et= {et}')
    #print(f'obj_func(et) = {obj_func(et)}')

plt.show()
