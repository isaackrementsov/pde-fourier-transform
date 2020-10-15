# Fourier Transform Wave Equation Solution

import numpy as np

from export_utils import gen_animation
from math_utils import aaf, applyeach, transform

# Range of values to perform Fourier transform over
L = 400
dx = 0.01
inp_range = np.arange(-L, L, dx)

# Wave speed
v = 1
# Initial wave height distribution
f = lambda x: np.exp(-x**2)
# Initial wave velocity distribution
g = lambda x: 2*x*np.exp(-x**2)

# Fourier transformed initial conditions
f_hat = transform(f, inp_range)
g_hat = transform(g, inp_range)

# Solution in Fourier Domain
def gen_u_hat(t):
    u_hat = lambda w: aaf(f_hat, w, inp_range)*np.cos(v*w*t) + aaf(g_hat, w, inp_range)*np.sin(v*w*t)/(v*w)

    return u_hat

# Inverse transformed solution
def u(t):
    return transform(gen_u_hat(t), inp_range, mode='inverse')

# Animate the solution from x=-50 to 50 and t=0 to 10
gen_animation(u, inp_range, (-50, 50), (0, 10), 1)
