# Helpful math functions

from scipy.fft import fft, ifft
from scipy.integrate import simps
import numpy as np

# Apply array of function values as function
def aaf(y_range, x, x_range):
    return y_range[closest_index(x, x_range)]

# Find the index of the closest value in an array
def closest_index(x, x_range):
    return (np.abs(x_range - x)).argmin()

# Apply a function to each value in an array
def applyeach(f, x_range, output='real'):
    f_range = []

    for x in x_range:
        val = f(x)

        if output == 'real':
            val = np.real(val)
        elif output == 'imag':
            val = np.imag(val)

        f_range.append(val)

    return f_range

# Separate real and imaginary parts of a complex vector
def separate(vec):
    return np.concatenate((vec.real, vec.imag))

# Join real and imaginary parts back into complex vector
def join(vec, n):
    return vec[:n] + (1j)*vec[n:]

# Join real and imaginary parts of matrix columns
def join_matrix(matrix, n):
    return matrix[:,:n] + (1j)*matrix[:,n:]

def double_simps(integrand):
    return simps([simps(slice) for slice in integrand])

def d_dt(u, u_t0, dt):
    u_t = np.array([u_t0])

    for i in range(1, len(u)):
        u_slice = u[i]
        du = u[:,:,i] - u[:,:,i - 1]

        np.stack([u_t, [du/dt]])

    return u_t
