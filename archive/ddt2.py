"""
https://scicomp.stackexchange.com/questions/20054/implementation-of-1d-advection-in-python-using-weno-and-eno-schemes

"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from schemes import *

# Data for plotting
nx = 81
dx = 2./(nx-1)
x = linspace(0,2,nx)
nt = 25
dt = .02
c = 1.      #assume wavespeed of c = 1
u = zeros(nx)      #numpy function ones()
u[int(.5/dx) : int(1/dx+1)]=2  #setting u = 2 between 0.5 and 1 as per our I.C.s
k = 3 # number of weights Order= 2*k-1
gc = k-1 #number of ghost cells
#adding ghost cells
gcr=x[-1]+linspace(1,gc,gc)*dx
gcl=x[0]+linspace(-gc,-1,gc)*dx
xc = append(x,gcr)
xc = append(gcl,xc)
uc = append(u,u[-gc:])
uc = append(u[0:gc],uc)

gs = zeros((nx+2*gc,nt))
flux = zeros(nx+2*gc)


plt.plot(x, u, '--', label='Initial')


for n in range(1,nt):
    un = uc.copy()
    #for i in range(1,nx):
    for i in range(2,nx):
        xloc = xc[i-(k-1):i+k]
        floc = c*uc[i-(k-1):i+k]
        #f_left,f_right = ENO(xloc,floc,k)
        f_left,f_right = WENO(xloc,floc,k)
        #uc[i] = un[i]-dt/dx*(f_right-f_left)
        flux[i] = 0.5 * (c + fabs(c)) * f_left + 0.5 * (c - fabs(c)) * f_right

    for i in range(gc,nx-gc):
        if c>0:
            #uc[i] = un[i]-dt/dx*(flux[i]-flux[i-1])
            uc[i] = uc[i]-dt/dx*(flux[i]-flux[i-1])
        else:
            #uc[i] = un[i]-dt/dx*(flux[i+1]-flux[i])
            uc[i] = uc[i]-dt/dx*(flux[i+1]-flux[i])

        U_1 = U + dt * dUdt(U, t_c)
        U_2 = 3/4.0 * U + 1/4.0 * U_1 + 1/4.0 * dt * dUdt(U_1, t_c)
        U = 1/3.0 * U + 2/3.0 * U_2 + 2/3.0 * dt * dUdt(U_2, t_c)

plt.plot(x, u, '--', label='Step')
#plt.plot(x, uc, '--')
plt.plot(xc, uc, '--')
plt.show()
