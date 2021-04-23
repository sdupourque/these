import pyproffit
import os
import numpy as np
import emcee
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import gamma
from .project_model import BetaModel
from .project_ps import EtaBetaModel, P3D_to_P2D, FitterPS,ChiSquared_P3D
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class Cluster:

    def __init__(self, id):

        self.name = id['NAME']
        self.tag = id['TAG']
        self.path = id['PATH']
        self.datalink = os.path.join(id['PATH'], id['datalink'])
        self.explink = os.path.join(id['PATH'],id['explink'])
        self.bkglink = os.path.join(id['PATH'],id['bkglink'])
        self.reg = os.path.join(id['PATH'],id['reg'])
        self.r500 = id['R500']
        self.t500 = id['THETA500']
        self.z = id['REDSHIFT']
        self.r500_arcmin = self.r500/cosmo.kpc_proper_per_arcmin(self.z).value
        self.ra = id['RA']
        self.dec = id['DEC']

        self.dat = pyproffit.Data(self.datalink, explink=self.explink, bkglink=self.bkglink)
        self.wcs = self.dat.wcs_inp
        self.dat.region(self.reg)
        self.nscales = 10

        self.prof = None
        self.mod = None
        self.fitobj = None

    def extract_profile(self):

        self.prof = pyproffit.Profile(self.dat, center_choice='centroid', centroid_region=self.t500/2, center_ra=self.ra, center_dec=self.dec, maxrad=self.t500, binsize=10., cosmo=cosmo)
        self.prof.SBprofile(ellipse_ratio=self.prof.ellratio, rotation_angle=self.prof.ellangle % 180)

    def fit_model(self):

        beta = 2/3
        rc = 2
        norm = np.log10(self.prof.profile.max()/(rc)/(np.sqrt(np.pi)*gamma(3*beta-1/2)/gamma(3*beta)))
        bkg = np.log10(self.prof.profile.min())

        self.mod = BetaModel()
        self.fitobj = pyproffit.Fitter(model=self.mod, profile=self.prof)
        self.fitobj.Migrad(beta=beta,
                           rc=rc,
                           norm=norm,
                           bkg=bkg,
                           limit_rc=(0, 10),
                           limit_bkg=(-20,0))
        self.model_best_fit = self.mod.params.copy()
        self.outmod()
        self.model_best_fit_image = fits.getdata('outmod_{}.fits'.format(self.name), memmap = False)

    def model_mcmc(self, n_samples=1000):

        def lnprob(x):

            res = - self.fitobj.cost(*x)

            if np.isnan(res):
                return -np.inf

            return res

        ndim, nwalkers = self.mod.npar, 8
        pos = [self.mod.params + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        sampler.run_mcmc(pos, n_samples, progress=True)

        self.model_chains = sampler.chain
        self.model_samples = self.model_chains[:, :, :].reshape((-1, ndim))
        self.model_covariance = np.cov(self.model_samples, rowvar=False)

    def outmod(self, params=None):

        if params is not None:
            self.mod.SetParameters(params)
            self.prof.SaveModelImage('outmod_{}.fits'.format(self.name), model=self.mod)
            self.mod.SetParameters(self.model_best_fit)
        else:
            self.prof.SaveModelImage('outmod_{}.fits'.format(self.name), model=self.mod)

    def ps2D(self):

        self.r_size = 2*self.r500 / 1000
        self.outmod(params=self.model_best_fit)
        self.psc = pyproffit.power_spectrum.PowerSpectrum(self.dat, self.prof, nscales=self.nscales, cosmo=cosmo)
        self.psc.MexicanHat(modimg_file='outmod_{}.fits'.format(self.name),
                       z=self.z,
                       region_size=self.r_size,
                       factshift=1.5,
                       path= self.path,
                       poisson = True)

        self.psc.PS(z=self.z, region_size=self.r_size, radius_out=self.r500 / 1000, path=self.path)

        self.ps = np.copy(np.abs(self.psc.ps))
        self.psnoise = np.abs(self.psc.psnoise)

    def ps_mcmc(self, n_samples=100):

        self.ps_samples_poisson = []
        self.ps_samples_profile = []
        self.ps_noise_samples = []

        for i in tqdm(range(n_samples)):

            self.outmod(params=self.model_samples[i,:])

            psc = pyproffit.power_spectrum.PowerSpectrum(self.dat, self.prof, nscales=self.nscales, cosmo=cosmo)
            psc.MexicanHat(modimg_file='outmod_{}.fits'.format(self.name),
                           z=self.z,
                           region_size=self.r_size,
                           factshift=1.5,
                           path=self.path,
                           poisson=False)

            psc.PS(z=self.z, region_size=self.r_size, radius_out=self.r500 / 1000, path=self.path)

            self.ps_samples_profile.append(np.abs(np.copy(psc.ps)))

        self.outmod(params=self.model_best_fit)

        for i in tqdm(range(n_samples)):

            psc = pyproffit.power_spectrum.PowerSpectrum(self.dat, self.prof, nscales=self.nscales, cosmo=cosmo)
            psc.MexicanHat(modimg_file='outmod_{}.fits'.format(self.name),
                           z=self.z,
                           region_size=self.r_size,
                           factshift=1.5,
                           path=self.path,
                           poisson=True)

            psc.PS(z=self.z, region_size=self.r_size, radius_out=self.r500 / 1000, path=self.path)

            self.ps_samples_poisson.append(np.abs(np.copy(psc.ps)))
            self.ps_noise_samples.append(np.abs(np.copy(psc.psnoise)))

        self.ps_cov_poisson = np.cov(self.ps_samples_poisson, rowvar=False)
        self.ps_cov_profile = np.cov(self.ps_samples_profile, rowvar=False)
        self.ps_cov_sample = np.diag((self.psc.ps**2)/3)
        self.ps_covariance = self.ps_cov_poisson + self.ps_cov_profile + self.ps_cov_sample

    def fit_P3D(self):

        eta = EtaBetaModel(r500=self.r500, model_params=self.model_best_fit, z=self.z)
        mod = P3D_to_P2D(eta)
        fitobj = FitterPS(ChiSquared_P3D(self.psc, self.ps_covariance, mod))
        fitobj.Migrad(k_in= 1e-4, norm=-11, alpha=11 / 3, pedantic=False, limit_k_in=(0, 5e-4), fix_k_in=True)

        self.model_P3D = mod
        self.res = fitobj.out
