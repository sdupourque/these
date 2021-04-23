import numpy as np

def pl_cutoff_PowerSpectrum(k):

    alpha = 11/3
    norm = 1
    k_inje = 1e-4
    k_piv = 1e-3
    k_disp = 1e-1

    return 10**norm*np.exp(-(k_inje/k)**2)*np.exp(-(k/k_disp)**2)*(k/k_piv)**(-alpha)

class Fluctuation:

    def __init__(self, P3D):

        self.ps3d = P3D