from scipy.integrate import solve_ivp
import numpy as np
import time
def dydt(t, y):
    #dy[0] = 2 * y[1] - y[0]
    #dy[1] = y[0] - y[1] ** 2

    #dydt = [2 * y[1] - y[0], y[0] - y[1] ** 2]
    return [0.1 * y[1] - y[0], y[0] - y[1] ** 2]

    return dy

#t = np.linspace(0, 30, 100)
t0, tf = 0, 30
x_0 = [1, 0]
start = time.clock()
sol = solve_ivp(dydt, [t0, tf], x_0, method='RK45')
print(time.clock() - start)  # 0.0013299999999999979  #
import matplotlib.pyplot as plt
plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[1])
plt.show()
