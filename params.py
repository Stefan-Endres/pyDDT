"""Parameters"""
import numpy as np
# Calibrated parameters for the reactant WR-EOS
A = 2.0  # mm/us  # us = microsecond
B = 2.50
C = 0.70
Z = -0.8066
gammma_o_r = 1.22
q = 5.71  # kJ/g
TMD = 1.76  # g/cm^3
C_v = 992.0
# Calibrated parameters for the product WR-EOS
a = 0.7579
k_wr = 1.30
v_c = 1.2171  # cm^3/g
p_c = 1.5899  #Gpa
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