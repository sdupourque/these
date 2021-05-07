import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from turb.extract_ps import Extractor
from astropy.io import fits
from simu.random_field import Fluctuation, pl_cutoff_PowerSpectrum
from astropy.cosmology import FlatLambdaCDM

k = np.geomspace(1e-6, 1e1, 1000)

plt.plot(k , pl_cutoff_PowerSpectrum(k))
plt.ylim(bottom=1e-12, top=1e3)
plt.xlabel('k [kpc-1]')
plt.ylabel('P3D')
plt.loglog()
plt.show()

#%%
import matplotlib.colors as colors

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
outmod = fits.getdata('analysis_results/test/clusters/A644/outmod_A644.fits')
z = 0.07
dr = 0.000694444444444445*cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.deg).value

#%%
f = Fluctuation(pl_cutoff_PowerSpectrum)
delta = np.real(f.gen_real_field(dr, outmod.shape))

plt.imshow(delta)
plt.show()

#%%
fig, ax = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(10, 7))
ax[0].imshow(outmod, norm=colors.LogNorm())
imax, jmax = outmod.shape
outmod_turb = outmod*(1+2*delta[0:imax, 0:jmax])
map =ax[1].imshow(outmod_turb, norm=colors.LogNorm())
plt.show()

#%%

plt.imshow(np.log10(outmod_turb/outmod))
plt.colorbar()
plt.show()

#%%
from astropy.table import Table
from turb.extract_ps import Extractor

master_table = Table.read('data/master_table.fits')
master_table = master_table[master_table['NAME']=='A644']

extractor = Extractor.from_catalog_row(master_table[0])

extractor.ps_region_size = 2 * extractor.r500 / 1000
extractor.outmod_path = 'simu/dump/outmod.fits'
extractor.dump_path = 'simu/dump'

extractor.load_data()
extractor.dat.img = outmod_turb
extractor.extract_profile()
extractor.outmod()
extractor.model_best_fit_image = fits.getdata(extractor.outmod_path, memmap=False)
extractor.ps_posterior_sample(n_samples=30, n_scales=20)

#%%