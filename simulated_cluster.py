import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from turb.extract_ps import Extractor
from astropy.io import fits
from simu.random_field import Fluctuation, pl_cutoff_PowerSpectrum
from astropy.cosmology import FlatLambdaCDM

k = np.geomspace(1e-6, 1e1, 1000)

plt.plot(k , pl_cutoff_PowerSpectrum(k))
plt.ylim(bottom=1e-12, top=1e5)
plt.xlabel('k [kpc-1]')
plt.ylabel('P3D')
plt.loglog()
plt.show()

#%%
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
outmod = fits.getdata('analysis_results/test/clusters/A644/outmod_A644.fits')
z = 0.07
dr = 0.000694444444444445*cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.deg).value

f = Fluctuation(pl_cutoff_PowerSpectrum)

fig, ax = plt.subplots(ncols=3, nrows=3, constrained_layout=True, figsize=(10, 10))

for i in range(9):

    map = ax[i // 3, i % 3].imshow(np.real(f.gen_real_field(dr, outmod.shape)))

plt.colorbar(mappable=map, ax=ax)
plt.show()

