import numpy as np
import matplotlib.pyplot as plt
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from cythoncode import rkf, F
import time
start = time.clock()
#your code here


x0 = np.array([1, 0], dtype=np.float64)
f = F(a=0.1)

t = np.linspace(0, 30, 100)


start = time.clock()
y = rkf(f, t, x0)
print(time.clock() - start)  # 3.6000000000036e-05

plt.plot(t, y)
plt.show()


#from scipy.integrate import solve_ivp
#def exponential_decay(t, y): return -0.5 * y
#sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], method='RK45')