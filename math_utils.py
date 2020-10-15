# Helpful math functions

from scipy.fft import fft, ifft
import numpy as np

# Apply array of function values as function
def aaf(y_range, x, x_range):
    return y_range[closest_index(x, x_range)]

# Find the index of the closest value in an array
def closest_index(x, x_range):
    return (np.abs(x_range - x)).argmin()

# Forward or inverse finite Fourier Transform
def transform(f, x_range, mode='forward'):
    y_range = applyeach(f, x_range)

    if mode == 'forward':
        return fft(y_range)
    elif mode == 'inverse':
        return ifft(y_range)

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
