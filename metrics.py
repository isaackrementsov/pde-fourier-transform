from math_utils import double_simps
import numpy as np
import numpy.fft as nft

def energy(u, u_t, weights):
    (w1, w2) = weights
    e = w1*u + w2*u_t
    E = [double_simps(ei) for ei in e]

    return E

def disorder(u, u_t, weight):
    u_hat = [nft.fft2(ui) for ui in u]

    d = weight*u_hat**2
    D = [double_simps(di) for di in d]

    return D
