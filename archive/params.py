"""Parameters"""
import numpy as np


"""Initial conditions and simulation parameters"""
# Simulation
t0 = 0.0  # us (microseconds) # Simulation start time
#tf = 0.025  # us (microseconds) # Simulation finish time
tf = 0.1  # us (microseconds) # Simulation finish time
tf = 0.002  # us (microseconds) # Simulation finish time
tf = 0.0015  # us (microseconds) # Simulation finish time
tf = 0.0015  # us (microseconds) # Simulation finish time
#tf = 0.0000025  # us (microseconds) # Simulation finish time
L = 0.1  # 0.10 mm # Length of the tube
L = 0.001  # 0.10 mm # Length of the tube
L = 0.002  # 0.10 mm # Length of the tube
#L = 0.00001  # 0.10 mm # Length of the tube

# WENO (DO NOT CHANGE UNTIL INITS ARE INVARIANT)
N = 120  # Number of discrete spatial elements
#N = 1000  # Number of discrete spatial elements
N = 50  # Number of discrete spatial elements
N = 40  # Number of discrete spatial elements
N = 80  # Number of discrete spatial elements
N = 100  # Number of discrete spatial elements
#N = 200  # Number of discrete spatial elements
k = 3  # number of weights Order= 2*k-1
gc = k - 1  # number of ghost cells

# parameters
#phi_0 = 0.75  # 0.65
phi_0 = 0.8125
#u_0 = -1.0  # mm/us (0.1 cm/us  = 1.0 mm/us) Initial velocity in the x-direction
u_0_1 = 0.0  # mm # Pellet velocity relative to bulk

# 375 m/s
#u_0 = 0.434  # mm/us
u_0 = -0.475  # mm/us

#p_0 = 1.0e-9  # GPa (equivalent to 1 atmosphere)
p_0 = 1.01325e-07  # mm-1 us-2  (equivalent to 1 atmosphere)
#rho_0 = 1.32*1e-3  #  g/mm^3 # rho_0 = TMD * Phi_0 = 0.75*1.76 Initial density g/cm3
rho_0 = 1.43*1e-3  #  g/mm^3 # rho_0 = TMD * Phi_0 = 0.75*1.76 Initial density g/cm3
v_0 = (rho_0)**(-1)  # mm3/g  Assume experimental condition
#alculated the initial density to be 1430kg/m3
#That's a 80.3 %of the tmd
#v_0 = (1.76*1e-3)**(-1)  # mm3/g  Assume experimental condition
lambd_0 = 0  # Initial reaction progress
             # Ratio of mass products / total mass of a volume element

"""Calibrated parameters for the reactant WR-EOS"""
A = 2.3  # mm/us  # us = microsecond
B = 2.50  # -
C = 0.70  # -
Z = -0.8066  # -
gamma_0_r = 1.22  # -
#q = 5.71  # kJ/g
TMD = 1.76*1e-3  # g/mm^3
#C_v = 992.0  # J / kg K
# Calibrated parameters for the product WR-EOS
a = 0.7579
k_wr = 1.30
# v_c = 1.2171  # cm^3/g
v_c = 1.2171 * 1e3  # mm^3/g
# p_c = 1.5899  # Gpa
p_c = 1.5899 * 1e-3  # g mm-1 us-2
n = 0.957
b = 0.80
#C_v = 650  # J /(kg K)
#  Calibrated parameters for the compaction rate equation
#P_h = 0.07  # GPa
P_h = 0.07*1e-3  # g mm-1 us-2
#k_phi = 31.5  # GPa-1 us-1
k_phi = 31.5*1e3  # g mm-1 us-2 us-1

#p_ign = 0.01  # GPa  xu and stewart 1997
#p_ign = 0.01 * 1e-3  # g mm-1 us-2

p_ign = p_c  # # g mm-1 us-2 TODO: It is assumed that Table II of the paper implies this
#p_ign = 0.1  # GPa TODO: This if from the paper Xu Stewart '97
#p_cj = 1  # GPa TODO: Find0
#p_cj = 15.928*1e-3  # g mm-1 us-2
p_cj = 19.05*1e-3  # g mm-1 us-2

#p_cj = 20.5*1e-3  # g mm-1 us-2  at a detonation vel ~7.24 mm us−1
#p_cj = 1.1*15.928*1e-3  # g mm-1 us-2
"""
p_cj was interpolated from data for PETN (TABLE 20.1 in unknown textbook)
                        # Detonations, general observations
for rho_0 =  1.32  g / cm3

(1.32 - 1.23) * (17.3 - 13.87) / (1.38 - 1.23) + 13.87 = 15.928000000000003

Detonation velocity:
(1.32 - 1.23) * (6.91 - 6.368) / (1.38 - 1.23) + 6.368
6.693200000000001 km/s (= mm/us)

for TMD = 1.76  g / cm3
p_cj ~ 33.7 GPa

detonation velocity ~8.27 km/s (= mm/us)

for rho_0 = 1.43   g / cm3

(1.43 - 1.38) * (20.17 - 17.3 ) / (1.45 - 1.38) + 17.3 =  19.05 GPa

(1.43 - 1.38) * (7.18 - 6.91) / (1.45 - 1.38) + 6.91 =  7.10285714285  (= mm/us)


"""

"""
The detonation wave speed is
7.24 mm us−1 through the compacted material and slows
down to 5.6 mm us−s
"""
# Initialize the equation of state
if 0:
    p_hat = rho_0 * A**2 / (4 * B)

## e_0 value
e_0_guess = 7.07  # kJ / cm3 Initial guess for starting internal energy
e_0_guess = 3.731  # kJ / g  (Wescott(?))
e_0_guess = 0.027   # kJ / g  (Wescott(?))

e_0_guess = 1.89   # kJ / g  (Wescott(?))
e_0_guess = 5.71   # kJ / g  q?

e_0_guess = 0.2957648   # kJ / g  From Cv
e_0_guess = 5.71  # kJ / g  From Cv
e_0_guess = 0.2957648  # kJ / g  From Cv
e_0_guess = 3.301014946420688  # kJ / g assuming e(_0) = 0 and solving e_0
#e_0_guess = 330.10895708130874

# rho_0 = 1.43*1e-3
e_0_guess = 1.892476153028025  # kJ / g

e_0 = e_0_guess  # NOTE: This value is computed in ddt.py initialization
                 #  e(p_0, v_0, lambd_0, phi_0) =  3.983295207817231

# Parameters correlated to an initial porosity
def k_function(phi_0):  # Validated correct against paper values (Table IV)
    """
    Reaction rate parameter k(phi_o) units of microsecond-1
    :param phi_0: Initial compaction v_r/v
    :return: k  # us-1
    """
    return 4.31 * np.exp(76.47 * phi_0**2 - 142.22 * phi_0 + 71.38)

def mu_function(phi_0):
    """
    Reaction order mu correlation (empiracally related to system microstructure)
    :param phi_0:  Initial compaction v_r/v
    :return: mu
    """
    return 25.627 * phi_0**2 - 42.504 * phi_0 + 20.676


def upsilon_function(phi_0):
    """
    Reaction order up correlation (empiracally related to system microstructure)
    :param phi_0:  Initial compaction v_r/v
    :return: mu
    """
    return 0.4  # constant fit

# Parameters correlated to an initial porosity
k_r = k_function(phi_0)
mu = mu_function(phi_0)
upsilon = upsilon_function(phi_0)


