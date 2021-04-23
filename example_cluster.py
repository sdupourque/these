import numpy as np
from astropy.io import fits
from turbulence_analysis.cluster import Cluster
import matplotlib.pyplot as plt
from turbulence_analysis import graph
from astropy.table import Table

#id = 22
id = 18
#id = 15
catalog = Table.read('./data/master_table.fits')
cluster = Cluster(catalog[id])

cluster.extract_profile()
cluster.fit_model()

cluster.prof.Plot(model=cluster.mod)
plt.show()

cluster.model_mcmc(n_samples=100)
cluster.ps2D()
cluster.ps_mcmc(n_samples=10)
#cluster.fit_P3D()

#%%
from matplotlib.colors import SymLogNorm, LogNorm

kpcp = cluster.psc.cosmo.kpc_proper_per_arcmin(cluster.z).value
Mpcpix = 1000. / kpcp / cluster.dat.pixsize  # 1 Mpc in pixel
regsizepix = cluster.psc.regsize * Mpcpix

factshift = 1.5
minx = int(np.round(cluster.prof.cx - factshift * regsizepix))
maxx = int(np.round(cluster.prof.cx + factshift * regsizepix + 1))
miny = int(np.round(cluster.prof.cy - factshift * regsizepix))
maxy = int(np.round(cluster.prof.cy + factshift * regsizepix + 1))
if minx < 0: minx = 0
if miny < 0: miny = 0
if maxx > cluster.dat.axes[1]: maxx = cluster.dat.axes[1]
if maxy > cluster.dat.axes[0]: maxy = cluster.dat.axes[0]

outmod = cluster.model_best_fit_image[miny:maxy, minx:maxx]
outmod[cluster.psc.mask.astype(np.int)] = np.nan
image = fits.getdata(cluster.datalink)[miny:maxy, minx:maxx]

plt.figure(figsize=(8,8))
plt.imshow(cluster.dat.img,
           norm=LogNorm(),
           origin='lower')
plt.title('Img')
plt.axis('off')
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(outmod,
           norm=LogNorm(),
           origin='lower')
plt.title('Mean model')
plt.axis('off')
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(cluster.dat.exposure,
           origin='lower')
plt.title('Exposure')
plt.axis('off')
plt.show()


plt.figure(figsize=(8,8))
scale = fits.getdata(cluster.psc.convlink[-1])
scale[cluster.psc.mask.astype(np.int)] = np.nan
plt.imshow(np.log10(scale),
       origin='lower')
plt.title(cluster.psc.convlink[-1])
plt.axis('off')

plt.figure(figsize=(8,8))

for i, file in enumerate(cluster.psc.convlink[1:10]):

    plt.subplot(331+i)
    scale = fits.getdata(file)
    scale[cluster.psc.mask.astype(np.int)] = np.nan
    plt.imshow(np.log10(scale),
           origin='lower')
    plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
plt.plot(cluster.psc.k, cluster.ps)
plt.fill_between(cluster.psc.k, np.abs(cluster.psc.ps)- np.diag(cluster.ps_covariance)**0.5, np.abs(cluster.psc.ps)+ np.diag(cluster.ps_covariance)**0.5, alpha=0.5)
plt.plot(cluster.psc.k, np.diag(np.abs(cluster.ps_cov_poisson))**0.5)
plt.plot(cluster.psc.k, np.diag(np.abs(cluster.ps_cov_profile))**0.5)
plt.xlabel(r'$k$ [kpc$^{-1}$]')
plt.ylabel(r'$P_{2D}$ [kpc$^{4}$]')

#for psnoise in cluster.ps_noise_samples:
#    plt.plot(cluster.psc.k, psnoise, alpha=0.1, color='blue')

plt.loglog()
plt.tight_layout()
plt.show()

#%% 
from importlib import reload
reload(graph)

graph.dashboard(cluster)

#%%

plt.imshow(cluster.dat.img)