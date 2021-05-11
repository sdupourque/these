import pyproffit
import os
import numpy as np
import emcee
from tqdm import tqdm
from astropy.io import fits
from scipy.special import gamma
from .project_ps import EtaBetaModel, P3D_to_P2D, FitterPS, ChiSquared_P3D
from .graphics import dashboard
from .util import PSF_XMM_ft
from .project_model import BetaModel
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class Extractor:

    def __init__(self, **kwargs):

        self.name = kwargs.get('NAME')
        self.sample = kwargs.get('SAMPLE')
        self.analysis_path = kwargs.get('analysis_path', kwargs.get('PATH'))
        self.dump_path = os.path.join(self.analysis_path, 'clusters/{}'.format(self.name))
        self.mask_path = os.path.join(self.dump_path, 'mask.fits')
        self.outmod_path = os.path.join(self.dump_path, 'outmod_{}.fits'.format(self.name))

        self.datalink = os.path.join(kwargs.get('PATH'), kwargs.get('datalink_X'))
        self.explink = os.path.join(kwargs.get('PATH'), kwargs.get('explink_X'))
        self.bkglink = os.path.join(kwargs.get('PATH'), kwargs.get('bkglink_X'))
        self.reg = os.path.join(kwargs.get('PATH'), kwargs.get('reg_X'))
        self.r500 = kwargs.get('R500')
        self.t500 = kwargs.get('THETA500')
        self.z = kwargs.get('REDSHIFT')
        self.r500_arcmin = self.r500/cosmo.kpc_proper_per_arcmin(self.z).value
        self.ra = kwargs.get('RA')
        self.dec = kwargs.get('DEC')

        self.ps_region_size = None

        self.dat = None
        self.prof = None
        self.mod = None
        self.fitobj = None

    @classmethod
    def from_catalog_row(cls, row, **kwargs):

        row_dict = dict(zip(row.colnames, row))
        return cls(**row_dict, **kwargs)

    def load_data(self):

        self.dat = pyproffit.Data(self.datalink,
                                  explink=self.explink,
                                  bkglink=self.bkglink)

        self.wcs = self.dat.wcs_inp
        self.dat.region(self.reg)
        self.nscales = 10

    def extract_profile(self):

        self.prof = pyproffit.Profile(self.dat,
                                      center_choice='centroid',
                                      centroid_region=self.t500/2,
                                      center_ra=self.ra,
                                      center_dec=self.dec,
                                      maxrad=2*self.t500,
                                      binsize=10.,
                                      cosmo=cosmo)

        self.prof.SBprofile(ellipse_ratio=self.prof.ellratio,
                            rotation_angle=self.prof.ellangle % 180)

    def fit_model(self):

        self.fitobj = pyproffit.Fitter(model=self.mod,
                                       profile=self.prof)

        beta = 2/3
        rc = 2
        norm = np.log10(self.prof.profile.max()/(rc)/(np.sqrt(np.pi)*gamma(3*beta-1/2)/gamma(3*beta)))
        bkg = np.log10(self.prof.bkgprof.mean())
        self.fitobj.Migrad(beta=beta,
                           rc=rc,
                           norm=norm,
                           bkg=bkg,
                           limit_beta=(0,20),
                           limit_rc=(0, 20),
                           limit_bkg=(-20, 0))


        self.model_best_fit = self.mod.params.copy()
        self.outmod()
        self.model_best_fit_image = fits.getdata(self.outmod_path, memmap = False)

    def model_posterior_sample(self, n_samples=1000):

        def lnprob(x):

            res = - self.fitobj.cost(*x)

            if np.isnan(res):
                return -np.inf

            return res

        ndim, nwalkers = self.mod.npar, 3*self.mod.npar
        pos = [self.mod.params + 1e-3 * np.random.randn(ndim) for _ in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        sampler.run_mcmc(pos, n_samples, progress=True)

        self.model_chains = sampler.chain
        self.model_samples = self.model_chains[:, :, :].reshape((-1, ndim))
        self.model_covariance = np.cov(self.model_samples, rowvar=False)

    def outmod(self, params=None, median_noise=False):

        if params is not None:
            self.mod.SetParameters(params)
            self.prof.SaveModelImage(self.outmod_path,
                                     model=self.mod)
            self.mod.SetParameters(self.model_best_fit)

        else:
            self.prof.SaveModelImage(self.outmod_path,
                                     model=self.mod,
                                     median_noise=median_noise)

    def extract_ps(self, **kwargs):

        psc = pyproffit.power_spectrum.PowerSpectrum(self.dat,
                                                     self.prof,
                                                     nscales=kwargs.get('n_scales', 10),
                                                     cosmo=cosmo)

        psc.MexicanHat(modimg_file=self.outmod_path,
                       z=self.z,
                       region_size=self.ps_region_size,
                       factshift=1.5,
                       path= self.dump_path,
                       poisson = True)

        psc.PS(z=self.z,
               region_size=self.ps_region_size,
               radius_in = self.r500 / 2 / 1000,
               radius_out = self.r500 / 1000,
               path=self.dump_path)

        return np.copy(psc.ps), np.copy(psc.psnoise), np.copy(psc.k), psc

    def ps_posterior_sample(self, **kwargs):

        self.ps_samples = []
        self.ps_noise_samples = []

        for i in tqdm(range(kwargs.get('n_samples', 10))):
            if self.mod is not None:
                self.outmod(params=self.model_samples[-(i+1), :])
            else:
                self.outmod(median_noise=True)

            ps, psnoise, self.k, self.psc = self.extract_ps(**kwargs)

            self.ps_samples.append(ps)
            self.ps_noise_samples.append(psnoise)

        self.ps = np.median(self.ps_samples, axis=0)
        self.ps_cov_poisprof = np.cov(self.ps_samples, rowvar=False)
        self.ps_cov_sample = np.diag((self.ps ** 2) / 3)
        self.ps_covariance = self.ps_cov_poisprof + self.ps_cov_sample
        self.ps_noise = np.median(self.ps_noise_samples, axis=0)
        self.ps_noise_covariance = np.cov(self.ps_noise_samples, rowvar=False)

    def doit(self, **kwargs):

        self.mod = kwargs['profile_model']

        if kwargs['ps_region_size'] == '2r500':
            self.ps_region_size = 2 * self.r500 / 1000

        self.load_data()
        self.extract_profile()

        if self.mod is not None:

            if self.mod == 'BetaModel':
                self.mod = BetaModel()

            self.fit_model()
            self.model_posterior_sample(n_samples=kwargs.get('model_samples', 1000))

        else:
            self.outmod()
            self.model_best_fit_image = fits.getdata(self.outmod_path, memmap=False)

        self.ps_posterior_sample(n_samples=kwargs.get('ps_samples', 10),
                                 n_scales=kwargs.get('ps_scales',10)
                                 )

        self.psf_k_cut = PSF_XMM_ft(cosmo, self.z).k_cut

        dashboard(self, outfile=os.path.join(self.dump_path, self.name + '.html'))
        
        del self.dat
        del self.prof
        del self.mod
        del self.fitobj

        return True

    def ps_mcmc(self, n_samples=100):
        """
        Separate sampling for the two error sources
        """
        print('WARNING:DEPRECATED')
        self.ps_samples_poisson = []
        self.ps_samples_profile = []
        self.ps_noise_samples = []

        for i in tqdm(range(n_samples)):

            self.outmod(params=self.model_samples[i,:])

            psc = pyproffit.power_spectrum.PowerSpectrum(self.dat, self.prof, nscales=self.nscales, cosmo=cosmo)
            psc.MexicanHat(modimg_file='outmod_{}.fits'.format(self.name),
                           z=self.z,
                           region_size=self.ps_region_size,
                           factshift=1.5,
                           path=self.conv_path,
                           poisson=False)

            psc.PS(z=self.z, region_size=self.ps_region_size, radius_out=self.r500 / 1000, path=self.conv_path)

            self.ps_samples_profile.append(np.abs(np.copy(psc.ps)))

        self.outmod(params=self.model_best_fit)

        for _ in tqdm(range(n_samples)):

            psc = pyproffit.power_spectrum.PowerSpectrum(self.dat, self.prof, nscales=self.nscales, cosmo=cosmo)
            psc.MexicanHat(modimg_file='outmod_{}.fits'.format(self.name),
                           z=self.z,
                           region_size=self.ps_region_size,
                           factshift=1.5,
                           path=self.conv_path,
                           poisson=True)

            psc.PS(z=self.z, region_size=self.ps_region_size, radius_out=self.r500 / 1000, path=self.conv_path)

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

        fitobj.Migrad(k_in= 1e-3,
                      norm=-11,
                      alpha=11 / 3,
                      pedantic=False,
                      limit_k_in=(0, 1e-2))

        self.model_P3D = mod
        self.res = fitobj.out
