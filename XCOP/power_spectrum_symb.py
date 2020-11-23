import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.signal import convolve
from scipy.special import gamma
import matplotlib.pyplot as plt
from numba import jit
import logging
import time

epsilon = 1e-3
Yofn = np.pi
alpha = 11. / 3.  # Kolmogorov slope
cf = np.power(2., alpha / 2.) * gamma(3. - alpha / 2.) / gamma(
    3.)  # correction factor for power spectrum, Arevalo et al. Eq. B1
a3d = 0.1  # Fractional perturbations

# Function to Mexican Hat filter images at a given scale sc
def calc_mexicanhat(sc, img, mask, simmod):
    """
    Filter an input image with a Mexican-hat filter

    :param sc: Mexican Hat scale in pixel
    :type sc: float
    :param img: Image to be smoothed
    :type img: class:`numpy.ndarray`
    :param mask: Mask image
    :type mask: class:`numpy.ndarray`
    :param simmod: Model surface brightness image
    :type simmod: class:`numpy.ndarray`
    :return: Mexican Hat convolved image and SB model
    :rtype: class:`numpy.ndarray`
    """
    # Define Gaussian convolution kernel
    gf = np.zeros_like(img.shape)
    cx = int(img.shape[0] / 2 + 0.5)
    cy = int(img.shape[1] / 2 + 0.5)
    gf[cx, cy] = 1.
    gfm = gaussian_filter(gf, sc / np.sqrt(1. + epsilon))
    gfp = gaussian_filter(gf, sc * np.sqrt(1. + epsilon))
    # FFT-convolve image with the two scales
    gsigma1 = convolve(img, gfm, mode='same')
    gsigma2 = convolve(img, gfp, mode='same')
    # FFT-convolve mask with the two scales
    gmask1 = convolve(mask, gfm, mode='same')
    gmask2 = convolve(mask, gfp, mode='same')
    # FFT-convolve model with the two scales
    gbeta1 = convolve(simmod, gfm, mode='same')
    gbeta2 = convolve(simmod, gfp, mode='same')
    # Eq. 6 of Arevalo et al. 2012
    fout1 = np.nan_to_num(np.divide(gsigma1, gmask1))
    fout2 = np.nan_to_num(np.divide(gsigma2, gmask2))
    fout = (fout1 - fout2) * mask
    # Same for simulated model
    bout1 = np.nan_to_num(np.divide(gbeta1, gmask1))
    bout2 = np.nan_to_num(np.divide(gbeta2, gmask2))
    bout = (bout1 - bout2) * mask
    return fout, bout


# Bootstrap function to compute the covariance matrix
def do_bootstrap(vals, nsample):
    """
    Compute the covariance matrix of power spectra by bootstrapping the image

    :param vals: Set of values
    :type vals: class:`numpy.ndarray`
    :param nsample: Number of bootstrap samples
    :type nsample: int
    :return: 2D covariance matrix
    :rtype: class:`numpy.ndarray`
    """
    nval = len(vals[0])
    nsc = len(vals)
    vout = np.zeros([nsc, nsample])
    for ns in range(nsample):
        idb = [int(np.floor(np.random.rand() * nval)) for i in range(nval)]
        for k in range(nsc):
            vout[k, ns] = np.mean(vals[k][idb])
    cov = np.cov(vout)
    return cov

def calc_ps(region, img, mod, kr, nreg):
    """
    Function to compute the power at a given scale kr from the Mexican Hat filtered images

    :param region: Index defining the region from which the power spectrum will be extracted
    :type region: class:`numpy.ndarray`
    :param img: Mexican Hat filtered image
    :type img: class:`numpy.ndarray`
    :param mod: Mexican Hat filtered SB model
    :type mod: class:`numpy.ndarray`
    :param kr: Extraction scale
    :type kr: float
    :param nreg: Number of subregions into which the image should be splitted to perform the bootstrap
    :type nreg: int
    :return:
            - ps (float): Power at scale kr
            - psnoise (float): Noise at scale kr
            - vals (class:`numpy.ndarray`): set of values for bootstrap error calculation
    """
    var = np.var(img[region])  # Eq. 17 of Churazov et al. 2012
    vmod = np.var(mod[region])
    ps = (var - vmod) / epsilon ** 2 / Yofn / kr ** 2
    psnoise = vmod / epsilon ** 2 / Yofn / kr ** 2
    # amp=np.sqrt(ps*2.*np.pi*kr**2)
    # Compute power in subregions
    nptot = len(img[region])
    vals = np.empty(nreg)
    for l in range(nreg):
        step = np.double(nptot / nreg)
        imin = int(l * step)
        imax = int((l + 1) * step - 1)
        vals[l] = (np.var(img[region][imin:imax]) - np.var(mod[region][imin:imax])) / (epsilon ** 2 * Yofn * kr ** 2)
    return ps, psnoise, vals

class PowerSpectrum(object):
    """
    Class to perform fluctuation power spectrum analysis from Poisson count images. This is the code used in Eckert et al. 2017.

    :param data: Object of type :class:`pyproffit.data.Data` including the image, exposure map, background map, and region definition
    :type data: class:`pyproffit.data.Data`
    :param profile: Object of type :class:`pyproffit.profextract.Profile` including the extracted surface brightness profile
    :type profile: class:`pyproffit.profextract.Profile`
    """
    def __init__(self, data, profile):
        """
        Constructor for class PowerSpectrum

        """
        self.data = data
        self.profile = profile

    def MexicanHat(self, modimg_file, z, region_size=1., factshift=1.5):
        """
        Convolve the input image and model image with a set of Mexican Hat filters at various scales. The convolved images are automatically stored into FITS images called conv_scale_xx.fits and conv_beta_xx.fits, with xx the scale in kpc.

        :param modimg_file: Path to a FITS file including the model image, typically produced with :meth:`pyproffit.profextract.Profile.SaveModelImage`
        :type modimg_file: str
        :param z: Source redshift
        :type z: float
        :param region_size: Size of the region of interest in Mpc. Defaults to 1.0
        :type region_size: float
        :param factshift: Size of the border around the region, i.e. a region of size factshift * region_size is used for the computation. Defaults to 1.5
        :type factshift: float
        """
        imgo = self.data.img
        expo = self.data.exposure
        bkg = self.data.bkg
        pixsize = self.data.pixsize
        # Read model image
        fmod = fits.open(modimg_file)
        modimg = fmod[0].data.astype(float)
        # Define the mask
        nonz = np.where(expo > 0.0)
        masko = np.copy(expo)
        masko[nonz] = 1.0
        imgt = np.copy(imgo)
        noexp = np.where(expo == 0.0)
        imgt[noexp] = 0.0
        # Set the region of interest
        x_c = self.profile.cx  # Center coordinates
        y_c = self.profile.cy
        kpcp = cosmo.kpc_proper_per_arcmin(z).value
        Mpcpix = 1000. / kpcp / pixsize  # 1 Mpc in pixel
        regsizepix = region_size * Mpcpix
        self.regsize = regsizepix
        minx = int(np.round(x_c - factshift * regsizepix))
        maxx = int(np.round(x_c + factshift * regsizepix + 1))
        miny = int(np.round(y_c - factshift * regsizepix))
        maxy = int(np.round(y_c + factshift * regsizepix + 1))
        if minx < 0: minx = 0
        if miny < 0: miny = 0
        if maxx > self.data.axes[1]: maxx = self.data.axes[1]
        if maxy > self.data.axes[0]: maxy = self.data.axes[0]
        img = np.nan_to_num(np.divide(imgt[miny:maxy, minx:maxx], modimg[miny:maxy, minx:maxx]))
        mask = masko[miny:maxy, minx:maxx]
        self.size = img.shape
        self.mask = mask
        fmod[0].data = mask
        fmod.writeto('tmp/mask.fits', overwrite=True)
        # Simulate perfect model with Poisson noise
        randmod = np.random.poisson(modimg[miny:maxy, minx:maxx])
        simmod = np.nan_to_num(np.divide(randmod, modimg[miny:maxy, minx:maxx]))
        # Set the scales
        minscale = 2  # minimum scale of 2 pixels
        maxscale = regsizepix / 2. # at least 4 resolution elements on a side
        scale = np.logspace(np.log10(minscale), np.log10(maxscale), 10)  # 10 scale logarithmically spaced
        sckpc = scale * pixsize * kpcp
        # Convolve images
        for i in range(len(scale)):
            sc = scale[i]
            logging.info('MexicanHat: Convolving with scale {}'.format(sc))
            convimg, convmod = calc_mexicanhat(sc, img, mask, simmod)
            # Save image
            fmod[0].data = convimg
            fmod.writeto('tmp/conv_scale_%d_kpc.fits' % (int(np.round(sckpc[i]))), overwrite=True)
            fmod[0].data = convmod
            fmod.writeto('tmp/conv_model_%d_kpc.fits' % (int(np.round(sckpc[i]))), overwrite=True)
        fmod.close()

    #
    def PS(self, z, region_size=1., radius_in=0., radius_out=1.):
        """
        Function to compute the power spectrum from existing Mexican Hat images in a given circle or annulus

        :param z: Source redshift
        :type z: float
        :param region_size: Size of the region of interest in Mpc. Defaults to 1.0. This value must be equal to the region_size parameter used in :meth:`pyproffit.power_spectrum.PowerSpectrum.MexicanHat`.
        :type region_size: float
        :param radius_in: Inner boundary in Mpc of the annulus to be used. Defaults to 0.0
        :type radius_in: float
        :param radius_out: Outer boundary in Mpc of the annulus to be used. Defaults to 1.0
        :type radius_out: float
        """
        kpcp = cosmo.kpc_proper_per_arcmin(z).value
        Mpcpix = 1000. / kpcp / self.data.pixsize  # 1 Mpc in pixel
        regsizepix = region_size * Mpcpix
        ######################
        # Set the scales
        ######################
        minscale = 2  # minimum scale of 2 pixels
        maxscale = regsizepix / 2.
        scale = np.logspace(np.log10(minscale), np.log10(maxscale), 10)  # 10 scale logarithmically spaced
        sckpc = scale * self.data.pixsize * kpcp
        kr = 1. / np.sqrt(2. * np.pi ** 2) * np.divide(1., scale)  # Eq. A5 of Arevalo et al. 2012
        ######################
        # Define the region where the power spectrum will be extracted
        ######################
        fmask = fits.open('tmp/mask.fits')
        mask = fmask[0].data
        data_size = mask.shape
        fmask.close()
        y, x = np.indices(data_size)
        rads = np.hypot(y - data_size[0] / 2., x - data_size[1] / 2.)
        region = np.where(
            np.logical_and(np.logical_and(rads > radius_in * Mpcpix, rads <= radius_out * Mpcpix), mask > 0.0))
        ######################
        # Extract the PS from the various images
        ######################
        nsc = len(scale)
        ps, psnoise, amp, eamp = np.empty(nsc), np.empty(nsc), np.empty(nsc), np.empty(nsc)
        vals = []
        nreg = 20  # Number of subregions for bootstrap calculation
        for i in range(nsc):
            # Read images
            fco = fits.open('tmp/conv_scale_%d_kpc.fits' % (int(np.round(sckpc[i]))))
            convimg = fco[0].data.astype(float)
            fco.close()
            fmod = fits.open('tmp/conv_model_%d_kpc.fits' % (int(np.round(sckpc[i]))))
            convmod = fmod[0].data.astype(float)
            fmod.close()
            logging.info('PS: Computing the power at scale {} kpc'.format(sckpc[i]))
            ps[i], psnoise[i], vps = calc_ps(region, convimg, convmod, kr[i], nreg)
            vals.append(vps)
        # Bootstrap the data and compute covariance matrix
        logging.info('PS: Computing the covariance matrix...')
        nboot = int(1e4)  # number of bootstrap resamplings
        cov = do_bootstrap(vals, nboot)
        # compute eigenvalues of covariance matrix to verify that the matrix is positive definite
        la, v = np.linalg.eig(cov)
        logging.info('PS: Eigenvalues: {}'.format(la))
        eps = np.empty(nsc)
        for i in range(nsc):
            eps[i] = np.sqrt(cov[i, i])
        amp = np.sqrt(np.abs(ps) * 2. * np.pi * kr ** 2 / cf)
        eamp = 1. / 2. * np.power(np.abs(ps) * 2. * np.pi * kr ** 2 / cf, -0.5) * 2. * np.pi * kr ** 2 / cf * eps
        self.kpix = kr
        self.k = 1. / np.sqrt(2. * np.pi ** 2) * np.divide(1., sckpc)
        self.ps = ps
        self.eps = eps
        self.psnoise = psnoise
        self.amp = amp
        self.eamp = eamp
        self.cov = cov

