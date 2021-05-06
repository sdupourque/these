import numpy as np

def pl_cutoff_PowerSpectrum(k):

    alpha = 11/3
    norm = 1e-8
    k_inje = 1e-4
    k_piv = 1e-3
    k_disp = 1e-1
    k = np.abs(k)

    return 10**norm*np.exp(-(k_inje/k)**2)*np.exp(-(k/k_disp)**2)*(k/k_piv)**(-alpha)

class Fluctuation:

    def __init__(self, P2D):

        self.P2D = P2D

    def gen_complex_field(self,k):

        mods = np.sqrt(np.random.gamma(3, scale=self.P2D(k)/3))
        phase = np.random.uniform(low=0, high=2*np.pi, size=mods.shape)
        mods = np.nan_to_num(mods)

        return mods*(np.cos(phase) + 1j*np.sin(phase))

    def gen_real_field(self, pixsize, shape):

        kx = np.fft.fftfreq(max(shape), d=pixsize)
        ky = np.fft.fftfreq(max(shape), d=pixsize)

        Kx, Ky = np.meshgrid(kx, ky)
        k = np.sqrt(Kx**2 + Ky**2)

        complex_field = self.gen_complex_field(k)
        real_field = np.fft.fftn(complex_field)

        return real_field