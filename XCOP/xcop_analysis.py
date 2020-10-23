import pyproffit
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from astropy.io import fits
from matplotlib import rc
from tqdm import tqdm

logging.basicConfig(filename='xcop_analysis.log', filemode='w', level=logging.INFO)

rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':15})
rc('text', usetex=True)

#%%
#'A780'

catalog = fits.getdata('../data/XCOP/XCOP_master_table.fits')
redshift = dict([(x['NAME'], x['REDSHIFT']) for x in catalog])
r500 = dict([(x['NAME'], x['R500_HSE']/1000) for x in catalog])

names = ['A644', 'A2319', 'RXC1825', 'A1644', 'A3158', 'A3266', 'A85', 'A1795', 'A2142', 'ZW1215', 'A2255', 'A2029']
spectrums = {}
espectrums = {}
spectrums3D = {}
espectrums3D = {}

#%%
for name in tqdm(names):

    dat = pyproffit.Data('../data/XCOP/{}/mosaic_{}.fits.gz'.format(name, name.lower()),
                         explink='../data/XCOP/{}/mosaic_{}_expo.fits.gz'.format(name, name.lower()),
                         bkglink='../data/XCOP/{}/mosaic_{}_bkg.fits.gz'.format(name, name.lower()))

    # Constructor of class Profile
    # Here we define the center as the image centroid
    prof = pyproffit.Profile(dat, center_choice='centroid', maxrad=35., binsize=10.)

    # Extract the brightness profile in elliptical annuli
    # prof.ellratio and prof.ellangle contain the major-to-minor-axis ratio and rotation angle of the ellipse
    prof.SBprofile(ellipse_ratio=prof.ellratio, rotation_angle=prof.ellangle+180.)

    #Model fitting

    mod = pyproffit.Model(pyproffit.BetaModel)
    # Construct a Fitter object by specifying model and data
    fitobj = pyproffit.Fitter(mod, prof)

    # Chi squarred
    fitobj.Migrad(beta=0.7, rc=2., norm=-2, bkg=-4, pedantic=False)

    # C stat
    # fitobj.Migrad(method='cstat',beta=0.7,rc=3.,norm=-2,bkg=-4,pedantic=False)

    pars = mod.params

    prof.SaveModelImage(mod, 'test_outmod.fits')

    #%% Power spectrum
    r_size = r500[name]
    psc = pyproffit.power_spectrum.PowerSpectrum(dat, prof)
    psc.MexicanHat(modimg_file='test_outmod.fits', z=redshift[name], region_size=r_size, factshift=1.5)
    psc.PS(z=redshift[name], region_size=r_size)

    spectrums[name] = (psc.k, psc.ps)
    espectrums[name] = (psc.k, psc.ps - psc.eps, psc.ps + psc.eps)

    pfact = psc.ProjectionFactor(z=redshift[name], betaparams=pars, region_size=r_size)
    spectrums3D[name] = (psc.k, psc.ps3d)
    espectrums3D[name] = (psc.k, psc.ps3d - psc.eps3d, psc.ps3d + psc.eps3d)

#%%

plt.figure(figsize=(7, 7))
plt.loglog()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for name, color in zip(names, colors):

    plt.plot(*spectrums[name], color=color, linewidth=2, label=name)
    plt.fill_between(*espectrums[name], color=color, alpha=0.4)

plt.ylabel('P$_{2D}$', fontsize=25)
plt.xlabel('$k$ [kpc$^{-1}$]', fontsize=25)
plt.legend()
plt.show()