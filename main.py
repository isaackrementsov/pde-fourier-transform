# Fourier Transform Wave Equation Solution, based on Steve Brunton's lecture (https://www.youtube.com/watch?v=hDeARtZdq-U)

import time
import multiprocessing as mp
import numpy as np
import numpy.fft as nft
from scipy.integrate import odeint

from export_utils import gen_animation, graph
from math_utils import aaf, applyeach, separate, join, join_matrix, d_dt
from metrics import energy, disorder

try:
    # Wave velocity
    v = 3

    # Domain
    L = (10, 10)
    d = (0.1, 0.1)
    x = np.meshgrid(np.arange(-L[0]/2, L[0]/2, d[0]), np.arange(-L[1]/2, L[1]/2, d[1]))

    dt = 0.01
    t = np.arange(0, 10, dt)

    # Number of discrete data points
    n = (int(L[0]/d[0]), int(L[1]/d[1]))

    # Wavenumber vector
    omega = np.meshgrid(2*np.pi*nft.fftfreq(n[0], d=d[0]), 2*np.pi*nft.fftfreq(n[1], d=d[1]))

    # Initial wave distribution
    f = lambda x, y: 3*np.exp(-(x**2 + y**2))
    # Intial velocity distribution
    g = lambda x, y: 0*x*y

    # Initial condition Fourier coefficients
    f_data = f(*x)
    g_data = g(*x)
    f_hat = nft.fft2(f_data)
    g_hat = nft.fft2(g_data)

    # Represent the transformed wave equation as a system of 1st order DEs
    def wave_eq(state, t, omega_1, omega_2, v, m):
        # Separate system state
        u_hat_ri = state[:2*m]
        q_ri = state[2*m:]
        # Join back into a complex number
        u_hat = join(u_hat_ri, m)
        q = join(q_ri, m)

        # q is a variable to avoid second derivatives in the equations
        u_hat_t = q
        # Since this is the 2d wave equation, the square magnitude of the frequency vector is used
        q_t = -v**2*(np.power(omega_1, 2) + np.power(omega_2, 2))*u_hat

        # Separate into real/imaginary again
        u_hat_t_ri = separate(u_hat_t)
        q_t_ri = separate(q_t)

        return np.concatenate((u_hat_t_ri, q_t_ri))

    # Self-contained ode solution with split parameters
    def solve(f_hat_s, g_hat_s, omega_s1, omega_s2):
        u_hat_list = []

        for j in range(len(omega_s1)):
            # Array of omega vectors
            omega_row = (omega_s1[j], omega_s2[j])
            # Intial condition rows
            f_hat_row = f_hat_s[j]
            g_hat_row = g_hat_s[j]

            # Length of frequency range
            m = len(omega_row[0])

            # Separate real and imaginary parts
            f_hat_ri = separate(f_hat_row)
            g_hat_ri = separate(g_hat_row)

            # Solve u_hat ODE
            u_hat_ri = odeint(wave_eq, np.concatenate((f_hat_ri, g_hat_ri)), t, args=(omega_row[0], omega_row[1], v, m))[:,:2*m]

            u_hat_list.append(join_matrix(u_hat_ri, m))

        return np.stack(u_hat_list, axis=1)

    # Get maximum usable number of threads
    threads = 12
    while n[0]/threads < 1:
        threads -= 1

    pool = mp.Pool(threads)
    # Break down inputs
    f_hats = np.array_split(f_hat, threads)
    g_hats = np.array_split(g_hat, threads)
    omegas = (np.array_split(omega[0], threads), np.array_split(omega[1], threads))

    # Put input sets into their own arrays
    inputs = zip(f_hats, g_hats, omegas[0], omegas[1])

    # Time solution generation
    t1 = time.time()

    # Solve ODE on multiple threads
    u_hats = pool.starmap(solve, inputs)
    # Join solutions together
    u_hat = np.hstack(u_hats)
    print(len(u_hat), len(u_hat[0]), len(u_hat[0][0]))
    # Inverse transform
    def inverse(iter_range):
        # Initialize empty solution array
        u_i = np.zeros((len(iter_range), n[0], n[1]))

        for k in range(len(iter_range)):
            # Construct solution from new Fourier Coefficients
            i = k + iter_range[0]
            u_i[k,:,:] = nft.ifft2(u_hat[i,:,:])

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

    # Get time-dependent system metrics
    u_t = d_dt(u, g_data, dt)

    graph(energy(u, u_t, 1, 1), t, ('t', 'E(t)'), 'energy')
    graph(disorder(u, u_t, 1), t, ('t', 'D(t)'), 'disorder')
    # Save mp4 file representing solution
    #gen_animation(u, x, t, (-1,3), ('x', 'y', 'Wave height u'))
except KeyboardInterrupt:
    print('Stopping...')
    quit()
