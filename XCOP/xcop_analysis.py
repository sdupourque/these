import logging
from astropy.io import fits
from matplotlib import rc
from tqdm import tqdm

logging.basicConfig(filename='xcop.log', filemode='w', level=logging.INFO)

rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':15})
rc('text', usetex=True)

#%%
#'A780'

catalog = fits.getdata('../data/XCOP/XCOP_master_table.fits')
redshift = dict([(x['NAME'], x['REDSHIFT']) for x in catalog])
regions = dict([(x['NAME'], x['R200_HSE'] / 1000) for x in catalog])

names = ['A644', 'A2319', 'RXC1825', 'A1644', 'A3158', 'A3266', 'A85', 'A1795', 'A2142', 'ZW1215', 'A2255', 'A2029']
names = ['A1795']
spectrums = {}
espectrums = {}
spectrums3D = {}
espectrums3D = {}

#%%
for name in tqdm(names):

    dat = modified_pyproffit.Data('../data/XCOP/{}/mosaic_{}.fits.gz'.format(name, name.lower()),
                                  explink='../data/XCOP/{}/mosaic_{}_expo.fits.gz'.format(name, name.lower()),
                                  bkglink='../data/XCOP/{}/mosaic_{}_bkg.fits.gz'.format(name, name.lower()))

    # Constructor of class Profile
    # Here we define the center as the image centroid
    prof = modified_pyproffit.Profile.loadProf(dat, 'profiles/{}.prof'.format(name))

    #Model fitting

    mod = modified_pyproffit.Model(old_code.modified_pyproffit.models.Vikhlinin3D, freeze={'gamma': 3.})
    fitobj = modified_pyproffit.Fitter(mod, prof)
    initialGuess = {'beta': 0.45, 'alpha': 0.8, 'log_rc': -2, 'epsilon': 3., 'gamma': 3., 'log_rs': 0., 'norm': -2,
                    'bkg': -5}
    fitobj.CMA(**initialGuess)
    fitobj.ComputePosterior()
    fitobj.Plot(save=True, path='fitprofile/{}'.format(name), name=name)#, sb_plot=True, diagnostic_plot=True, corner_plot=True)

    pars = mod.params
    prof.SaveModelImage(mod, 'fitprofile/{}/outmod.fits'.format(name))

    #%% Power spectrum
    r_size = 0.5
    psc = old_code.modified_pyproffit.power_spectrum.PowerSpectrum(dat, prof)
    psc.MexicanHat(modimg_file='fitprofile/{}/outmod.fits'.format(name), z=redshift[name], region_size=r_size, factshift=1.5)
    psc.PS(z=redshift[name], region_size=r_size)

    #spectrums[name] = (psc.k, psc.ps)
    #espectrums[name] = (psc.k, psc.ps - psc.eps, psc.ps + psc.eps)

    #%%

    pfact = psc.ProjectionFactor(z=redshift[name], vikhlinin_params=pars, region_size=r_size)
    #psc.Plot(save_plots=False, plot_3d=True)

    #spectrums3D[name] = (psc.k, psc.ps3d)
    #espectrums3D[name] = (psc.k, psc.ps3d - psc.eps3d, psc.ps3d + psc.eps3d)
