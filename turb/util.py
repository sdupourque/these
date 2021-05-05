import numpy as np
import astropy.units as u
import warnings

from scipy.special import iv, gamma
from scipy.optimize import root_scalar

class PSF_XMM_ft:

    def __init__(self, cosmo, redshift):

        kpc_proper_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec).value

        @np.vectorize
        def psf(k):

            rc = 6.668 * kpc_proper_per_arcsec
            sigma = 139.41 * kpc_proper_per_arcsec
            alpha = 1.748
            R = 2.69e-2

            tf_exp = np.sqrt(np.pi / 2) * R * sigma * np.exp(-np.pi ** 2 * k ** 2 * sigma ** 2)
            tf_king = -np.pi ** alpha * k ** (alpha - 1) * rc ** (alpha + 1) * gamma(1 - alpha) * (
                    iv(1 - alpha, 2 * np.pi * k * rc) - iv(alpha - 1, 2 * np.pi * k * rc))

            return (tf_exp + tf_king)

        self.vfunc = psf
        sol = root_scalar(lambda k : self(k) - 1e-3, bracket=(1e-5, 1), method='bisect', rtol = 1e-14)
        self.sol = sol
        self.k_cut = sol.root

    def __call__(self, k):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            #Overflow happens for big k
            return self.vfunc(k)/self.vfunc(1e-150)

#%%

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    psf = PSF_XMM_ft(cosmo, 0.05)

    k = np.geomspace(1e-5, 1e2, 1000)

    plt.plot(k, psf(k))
    plt.vlines(x=psf.k_cut, ymin = min(psf(k)), ymax = max(psf(k)))
    plt.loglog()
    plt.show()
    
