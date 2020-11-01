import numpy as np
from archive import pyweno

x = np.linspace(0.0, 2*np.pi, 21)
f = (np.cos(x[1:]) - np.cos(x[:-1])) / (x[1] - x[0])
q = pyweno.weno.reconstruct(f, 5, 'left')


