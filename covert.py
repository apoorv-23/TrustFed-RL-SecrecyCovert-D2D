import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc


def detection_probabilities(tau_over_s0, gamma_w, n):
    c = 1.0 + np.asarray(gamma_w, dtype=np.float64)
    p_fa = gammaincc(n, n * tau_over_s0)
    p_md = gammainc(n, n * tau_over_s0 / c)
    return p_fa, p_md


def min_detection_error(gamma_w, n):
    g = np.maximum(np.asarray(gamma_w, dtype=np.float64), 1e-12)
    ln_c = np.log1p(g)                  
    a = n * (1.0 + g) * ln_c / g        
    b = n * ln_c / g                    
    return gammaincc(n, a) + gammainc(n, b)


def covert_sinr_threshold(n, epsilon):
    return brentq(lambda g: min_detection_error(g, n) - (1.0 - epsilon), 1e-10, 1e3, xtol=1e-14)


def kl_divergence(gamma_w, n):
    g = np.asarray(gamma_w, dtype=np.float64)
    return n * (np.log1p(g) - g / (1.0 + g))


def kl_covert_sinr_threshold(n, epsilon):
    return brentq(lambda g: kl_divergence(g, n) - 2.0 * epsilon ** 2, 1e-10, 1e3, xtol=1e-14)
