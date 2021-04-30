import numpy as np
import iminuit
from scipy.special import gamma, kv, j0
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM
from scipy.special import roots_hermite, roots_legendre, roots_laguerre
from numba import vectorize, njit

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class EtaBetaModel:

    def __init__(self, r500 = 1, model_params = None, z = None):

        #r500 in Mpc

        beta, rc, norm, bkg = model_params
        rc *= cosmo.kpc_proper_per_arcmin(z).value
        p = r500*1000 / rc
        self._t = np.linspace(0, p, 1000)

        def xn_kn(x, n):
            """Computes the function x^n K_n(x).
            """

            new = np.ones_like(x) * np.pi * 2 ** (n - 1.) / np.sin(n * np.pi) / gamma(1. - n)
            new[x != 0] = x[x != 0] ** n * kv(n, x[x != 0])

            return new

        def tfeta(kr, kz):
            prefact = 2 ** (5 / 2 - 3 * beta) * np.pi * rc ** 2 / gamma(3 * beta - 1 / 2)
            return prefact * simps(self._t * j0(2 * np.pi * rc * kr * self._t) * xn_kn(2 * np.pi * rc * kz * np.sqrt(1 + self._t ** 2), 3 * beta - 1 / 2), x=self._t)

        self._vcall = np.vectorize(tfeta)

    def __call__(self, kr, kz):

        return self._vcall(kr, kz)

class P3D_to_P2D:

    def __init__(self, eta):

        order = 60
        _x_legendre, _w_legendre = roots_legendre(n=order)
        self.kr, self.kz = np.geomspace(1e-8, 1e1, 200), np.geomspace(1e-8, 1e1, 200)
        self.kkr, self.kkz = np.meshgrid(self.kr, self.kz)

        @njit
        def P3D(k, k_in, norm, alpha):
            k_piv = 1e-3
            return 10 ** norm * np.exp(-(k_in / k) ** 2) * (k / k_piv) ** (-alpha)

        @vectorize(nopython=True)
        def shifted_P3D(kr1, kr2, kz, k_in, norm, alpha):
            return np.pi * np.sum(
                integrand_shifted_P3D(np.pi * (1 + _x_legendre), kr1, kr2, kz, k_in, norm, alpha) * _w_legendre)

        @njit
        def integrand_shifted_P3D(theta, kr1, kr2, kz, k_in, norm, alpha):
            return P3D(np.sqrt(kr1 ** 2 + kr2 ** 2 + kz ** 2 - 2 * kr1 * kr2 * np.cos(theta)), k_in, norm, alpha)

        self.shifted_P3D = shifted_P3D
        self.P3D = P3D
        self.Peta = eta(self.kkr, self.kkz)**2

        def _call(kr_in, k_in, norm, alpha):

            return 4*simps(simps(self.shifted_P3D(kr_in,
                                                  self.kkr,
                                                  self.kkz,
                                                  k_in,
                                                  norm,
                                                  alpha) * self.Peta,
                                 x=self.kz, axis=-1),
                           x=self.kr, axis=-1)

        self._vcall = np.vectorize(_call)

    def __call__(self, kr_in, k_in, norm, alpha):

        return self._vcall(kr_in, k_in, norm, alpha)

class ChiSquared_P3D:

    def __init__(self, psc, covmat, model):

        self.x = psc.k
        self.y = psc.ps
        self.covmat = np.linalg.inv(covmat)
        self.model = model

    def __call__(self, *par):  # par are a variable number of model parameters

        ym = self.model(self.x, *par)
        y = self.y-ym
        chi2 = y.T@self.covmat@y

        return chi2

class FitterPS:

    def __init__(self, chi2):

        self.cost = chi2
        self.mlike=None
        self.params=None
        self.errors=None
        self.minuit=None
        self.out=None

    def Migrad(self, **kwargs):

        minuit=iminuit.Minuit(self.cost, name=['k_in', 'norm', 'alpha'], **kwargs)
        fmin, param=minuit.migrad()
        print(fmin)
        print(param)
        npar = len(minuit.values)
        outval = np.empty(npar)
        outerr = np.empty(npar)

        for i in range(npar):
            outval[i] = minuit.values[i]
            outerr[i] = minuit.errors[i]

        self.params=minuit.values
        self.errors=minuit.errors
        self.mlike=fmin
        self.minuit=minuit
        self.out=outval