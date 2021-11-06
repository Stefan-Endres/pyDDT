import json
import codecs
import numpy as np
import scipy as sp
import scipy.ndimage
from progress.bar import IncrementalBar
from ddt import f, s
import matplotlib.pyplot as plt

#str = './data/.N_20_dt_0.00035440350429363906_1635105081.4507616.json'
#str = './data/N_20_dt_0.00035440350429363906_1635108730.9763746.json'
str = './data/'
#str += 'N_100_dt_3.3859731505875405e-07_1635112979.3057904.json'
#str += 'N_20_dt_2.6655418288946322e-06_1635110487.8039382.json'
#str += 'N_100_dt_2.2509508892419513e-06_1635110205.5979662.json'
str += 'N_100_dt_3.3859731505875405e-07_1635112979.3057904.json'
#str += 'N_40_dt_4.5615164973574046e-07_1635113012.329041.json'
#str += 'N_80_dt_2.984531375728416e-06_1635110030.084805.json'


def plot_all_results(Usol, x, t):
    """
    Plot pressure, density, velocity and lambda
    :param U: Solution tensor
    :return:
    """
    Res = np.zeros_like(Usol)
    # Res[0] = P
    # Res[1] = Rho  # Density
    # Res[2] = velocity  #
    # Res[3] = Phi
    # Res[4] = lambda
    prog_bar = IncrementalBar('Finished simulation. '
                              'Computing plot variables...', max=len(t))

    for tind, time in enumerate(t):

        U = Usol[tind, :, :]

        # Compute physical values from U state vectors
        F, pp = f(U)
        S, C = s(U, F, pp)
        (u, E, PHI, LAMBD, V, P) = pp

        Res[tind, 0, :] = P
        Res[tind, 1, :] = U[0]
        Res[tind, 2, :] = u
        Res[tind, 3, :] = PHI
        Res[tind, 4, :] = LAMBD
        prog_bar.next()


    Pressure_plot = np.minimum(Res[:, 0, :]*1e3, np.ones_like(Res[:, 0, :])*300)
    plot_u_t(x, t, Pressure_plot,
                  #title=r'Pressure $P$ (GPa) $\times 10^{-3}$', fign=0)
                  title=r'Pressure $P$ (GPa)', fign=0)

    Density_plot = np.minimum(Res[:, 1, :]*1e3, np.ones_like(Res[:, 0, :])*20)
    plot_u_t(x, t, Density_plot,
                  #title=r'Density $\rho$ (g/mm$^3$)', fign=1)
                  title=r'Density $\rho$ (g/cm$^3$)', fign=1)

    Velocity_plot = np.minimum(Res[:, 2, :], np.ones_like(Res[:, 0, :])*20)
    Velocity_plot = np.maximum(Velocity_plot, np.ones_like(Res[:, 0, :])*-20)
    plot_u_t(x, t,  Velocity_plot,
                  title=r'Velocity $u$ (mm . $\mu$ s$^{-1}$)', fign=2)

    plot_u_t(x, t, Res[:, 3, :],
                  title=r'$\phi$ ', fign=3)

    plot_u_t(x, t, Res[:, 4, :],
                  title=r'$\lambda$ ', fign=4)

    return F

def plot_u_t(x, t, U, title=r'Density $\rho$ (g/mm$^3$)', fign=1):
    """
    Plots a variable U over time
    :param U:
    :param t:
    :return:
    """
    x = x  # mm
    #xscale = 8.4/2.0 * (0.89)
    #yscale = xscale*5.5/8.4
    if 0:
        yscale = 8.4/2.0 * (0.89)*5.5/8.4
        xscale = yscale * 8.4 / 5.5 * 1.18
        plt.figure(fign, figsize=(xscale, yscale))
    plt.figure()
    plt.plot()
    #plt.pcolor(x, t, U, cmap='RdBu')
    #plt.pcolor(x, t, U, cmap='inferno')
    plt.pcolor(x, t, U, cmap='hot')
    plt.colorbar()

    font_props = {'family': 'normal',
                  'weight': 'bold',
                  'size': 11}
    plt.title(title,fontdict=font_props)
    plt.xlabel('x (mm)',fontdict=font_props)
    plt.ylabel(r'time ($\mu$s)',fontdict=font_props)
    plt.minorticks_on()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #plt.axis('scaled')
    #plt.set_size_inches(18.5, 10.5)
    #plt.show()


obj_text = codecs.open(str, 'r', encoding='utf-8').read()
data = json.loads(obj_text)
U = data['U']
U = np.array(U)
solver_x = np.array(data['solver_x'])
solver_t = np.array(data['solver_t'])
gc = data['gc']
U = U[:, :, gc:-(gc)]

# Trim data set
if 1:
    trim = 7638 #  0.025
    trim = 7638 #  0.001596
    print(f'solver_t = {solver_t}')
    print(f'len(solver_t) = {len(solver_t)}')
    print(f'int(0.25*(len(solver_t))) = {int(0.25*(len(solver_t)))}')
    print(f'solver_t[int(0.25*(len(solver_t)))] = {solver_t[int(0.25*(len(solver_t)))]}')
    solver_t = solver_t[:7383]
    solver_x = solver_x[:7383]
    U = U[:7383, :, :]


# Smoothing
if 1:
    sigma_y = 0.5
    sigma_x = 1
    sigma = [sigma_y, sigma_x]
    U_smooth = np.zeros_like(U)
    for i, u in enumerate(U):
        U_smooth[i] = sp.ndimage.filters.gaussian_filter(u, sigma,
                         #mode=‘reflect’
                         mode='nearest'
                                                         )
    U = U_smooth
# plot
if 1:
    plot_all_results(U, solver_x, solver_t)
    plt.show()
#data = json.loads(str)
#print(data['U'])