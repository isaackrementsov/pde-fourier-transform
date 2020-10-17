# Fourier Transform Wave Equation Solution, based on Steve Brunton's lecture (https://www.youtube.com/watch?v=hDeARtZdq-U)

import numpy as np
import numpy.fft as nft
from scipy.integrate import odeint

from export_utils import gen_animation, graph
from math_utils import aaf, applyeach, separate, join, join_matrix

# Wave velocity
v = 1

# Domain
L = 10
dx = 0.1
x_range = np.arange(-L/2, L/2, dx)

dt = 0.1
t_range = np.arange(0, 5, dt)

# Number of discrete data points
n = int(L/dx)

# Wavenumber vector
omega = 2*np.pi*nft.fftfreq(n, d=dx)

# Initial wave distribution
f = lambda x: np.exp(-x**2)
f = applyeach(f, x_range)
# Intial velocity distribution
g = lambda x: 0
g = applyeach(g, x_range)

# Initial condition Fourier coefficients
f_hat = nft.fft(f)
g_hat = nft.fft(g)
# Separate real and imaginary parts
f_hat_ri = separate(f_hat)
g_hat_ri = separate(g_hat)

# Represent the transformed wave equation as a system of 1st order DEs
def wave_eq(state, t, omega, v):
    # Separate system state
    u_hat_ri = state[:2*n]
    q_ri = state[2*n:]
    # Join back into a complex number
    u_hat = join(u_hat_ri, n)
    q = join(q_ri, n)

    # q is a variable to avoid second derivatives in the equations
    u_hat_t = q
    q_t = -v**2*np.power(omega, 2)*u_hat

    # Separate into real/imaginary again
    u_hat_t_ri = separate(u_hat_t)
    q_t_ri = separate(q_t)

    return np.concatenate((u_hat_t_ri, q_t_ri))

# Solve u_hat ODE
u_hat_ri = odeint(wave_eq, np.concatenate((f_hat_ri, g_hat_ri)), t_range, args=(omega, v))[:,:2*n]
u_hat = join_matrix(u_hat_ri, n)
# Initialize empty solution array
u = np.zeros_like(u_hat)

for k in range(len(t_range)):
    # Construct solution from new Fourier Coefficients
    u[k,:] = nft.ifft(u_hat[k,:])

# Only the real part is needed
u = u.real

# Save mp4 file representing solution
#gen_animation(u, x_range, t_range, (-2,2), ('u', 'x'))
graph(u[-1], x_range, ('u', 'x'))
