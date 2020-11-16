"""Parameters"""
import numpy as np


"""Initial conditions and simulation parameters"""
# Simulation
t0 = 0.0  # us (microseconds) # Simulation start time
#tf = 0.0025  # us (microseconds) # Simulation finish time
tf = 0.025  # us (microseconds) # Simulation finish time
tf = 0.001  # us (microseconds) # Simulation finish time
tf = 0.025  # us (microseconds) # Simulation finish time
tf = 0.0005  # us (microseconds) # Simulation finish time
#tf = 0.0004  # us (microseconds) # Simulation finish time
#tf = 0.005#25  # us (microseconds) # Simulation finish time
L = 0.01  # cm (0.10 mm = 0.01 cm) # Length of the tube
#L = 0.1  # 0.10 mm = 0.01 cm # Length of the tube

# WENO (DO NOT CHANGE UNTIL INITS ARE INVARIANT)
N = 120  # Number of discrete spatial elements
#N = 1000  # Number of discrete spatial elements
N = 100  # Number of discrete spatial elements
N = 80  # Number of discrete spatial elements
#N = 10  # Number of discrete spatial elements
k = 3  # number of weights Order= 2*k-1
gc = k - 1  # number of ghost cells

# parameters
phi_0 = 0.75  # 0.65
u_0 = -0.1  # cm/us (0.1 cm/us  = 1.0 mm/us) Initial velocity in the x-direction
u_0_1 = 0.0  # cm # Pellet velocity relative to bulk
# 375 m/s

p_0 = 1.0e-9  # GPa (equivalent to 1 atmosphere)
#p_0 = 10e-4#e-9  # GPa (equivalent to 1 atmosphere)
#rho_0 = 1.6  # Initial density g/cm3
rho_0 = 1.32  # g/cm3 # rho_0 = TMD * Phi_0 = 0.75*1.76 Initial density g/cm3
v_0 = (rho_0)**(-1)  # cm3/g  Assume experimental condition
lambd_0 = 0  # Initial reaction progress
             # Ratio of mass products / total mass of a volume element



"""Calibrated parameters for the reactant WR-EOS"""
#A = 2.0  # mm/us  # us = microsecond  NOTE: WAS WRONG
A = 2.3  # mm/us  # us = microsecond
B = 2.50
C = 0.70
Z = -0.8066
gamma_0_r = 1.22
q = 5.71  # kJ/g
TMD = 1.76  # g/cm^3
C_v = 992.0
# Calibrated parameters for the product WR-EOS
a = 0.7579
k_wr = 1.30
v_c = 1.2171  # cm^3/g
p_c = 1.5899  # Gpa
#p_c = 6.0  # Gpa
n = 0.9570
b = 0.80
C_v = 650  # J /(kg K)
#  Calibrated parameters for the compaction rate equation
P_h = 0.07  # GPa
k_phi = 31.5  # GPa-1 us-1
# A lost parameter
p_c = 1.5899  # GPa
p_ign = p_c  # GPa TODO: It is assumed that Table II of the paper implies this
#p_ign = 0.1  # GPa TODO: This if from the paper Xu Stewart '97
p_cj = 1  # GPa TODO: Find

# Initialize the equation of state
p_hat = rho_0 * A**2 / (4 * B)

## e_0 value
e_0_guess = 7.07  # kJ / cm3 Initial guess for starting internal energy
e_0_guess = 3.731  # kJ / g  (Wescott(?))
#e_0_guess = 0.2957648  # kJ / g  (Wescott(?))
#e_0_guess = 7.31824233  # kJ / cm3 Initial guess for starting internal energy
#e_0_guess = 4.573901456496666
  # kJ / cm3 Initial guess for starting internal energy
#e_0_guess = 3.98329  # kJ / cm3 Initial guess for starting internal energy
e_0 = e_0_guess  # NOTE: This value is computed in ddt.py initialization
                 #  e(p_0, v_0, lambd_0, phi_0) =  3.983295207817231

#e_0 = e_0 #* 0.01
# Compute e_0
#e_0_guess = 1.671718701717153
#e_0 = e_0_guess
from wreos import e
#e_0 = e(p_0, v_0, lambd_0, phi_0)
#print(f'Computed e_0 from params.py = {e_0}')

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
k = k_function(phi_0)
mu = mu_function(phi_0)
upsilon = upsilon_function(phi_0)
