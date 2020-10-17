# Fourier Transform Wave Equation Solution, based on Steve Brunton's lecture (https://www.youtube.com/watch?v=hDeARtZdq-U)

import time
import multiprocessing as mp
import numpy as np
import numpy.fft as nft
from scipy.integrate import odeint

from export_utils import gen_animation, graph
from math_utils import aaf, applyeach, separate, join, join_matrix

# Wave velocity
v = 3

# Domain
L = 20
dx = 0.1
x_range = np.arange(-L/2, L/2, dx)

dt = 0.02
t = np.arange(0, 40, dt)

# Number of discrete data points
n = int(L/dx)

# Wavenumber vector
omega = 2*np.pi*nft.fftfreq(n, d=dx)

# Initial wave distribution
f = lambda x: 2*x*np.exp(-x**2)
f = applyeach(f, x_range)
# Intial velocity distribution
g = lambda x: 0
g = applyeach(g, x_range)

# Initial condition Fourier coefficients
f_hat = nft.fft(f)
g_hat = nft.fft(g)

# Represent the transformed wave equation as a system of 1st order DEs
def wave_eq(state, t, omega, v, m):
    # Separate system state
    u_hat_ri = state[:2*m]
    q_ri = state[2*m:]
    # Join back into a complex number
    u_hat = join(u_hat_ri, m)
    q = join(q_ri, m)

    # q is a variable to avoid second derivatives in the equations
    u_hat_t = q
    q_t = -v**2*np.power(omega, 2)*u_hat

    # Separate into real/imaginary again
    u_hat_t_ri = separate(u_hat_t)
    q_t_ri = separate(q_t)

    return np.concatenate((u_hat_t_ri, q_t_ri))

# Self-contained ode solution
def solve(f_hat_i, g_hat_i, omega_i):
    # Separate real and imaginary parts
    f_hat_ri = separate(f_hat_i)
    g_hat_ri = separate(g_hat_i)

    # Length of data piece
    m = len(omega_i)

    # Solve u_hat ODE
    u_hat_ri = odeint(wave_eq, np.concatenate((f_hat_ri, g_hat_ri)), t, args=(omega_i, v, m))[:,:2*m]

    return join_matrix(u_hat_ri, m)

# Get maximum usable number of threads
threads = 12
while n/threads < 1:
    threads -= 1

pool = mp.Pool(threads)
# Break down inputs
f_hats = np.array_split(f_hat, threads)
g_hats = np.array_split(g_hat, threads)
omegas = np.array_split(omega, threads)
# Put input sets into their own arrays
inputs = zip(f_hats, g_hats, omegas)

# Time solution generation
t1 = time.time()

# Solve ODE on multiple threads
u_hats = pool.starmap(solve, inputs)
# Join solutions together
u_hat = np.hstack(u_hats)

# Inverse transform
def inverse(iter_range):
    # Initialize empty solution array
    u_i = np.zeros((len(iter_range), n))

    for k in range(len(iter_range)):
        # Construct solution from new Fourier Coefficients
        i = k + iter_range[0]
        u_i[k,:] = nft.ifft(u_hat[i,:])

    return u_i

# Get maximum usable number of threads for time range
threads = 12
nt = len(t)
while nt/threads < 1:
    threads -= 1

tpool = mp.Pool(threads)
# Run inverse transform on sections of t range
ranges = np.array_split(range(len(t)), threads)
us = tpool.map(inverse, ranges)
# Stack u pieces back together
u = np.vstack(us)

# Only the real part is needed
u = u.real

t2 = time.time()
# Display final solution time
print('Took', str(np.round(t2 - t1, 3)), 'seconds')

# Save mp4 file representing solution
gen_animation(u, x_range, t, (-2,2), ('u', 'x'))
