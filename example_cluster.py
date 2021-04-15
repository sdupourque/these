import numpy as np
from astropy.io import fits
from turbulence_analysis.cluster import Cluster
import matplotlib.pyplot as plt
from turbulence_analysis import graph
from astropy.table import Table

#id = 22
id = 17
catalog = Table.read('./data/master_table.fits')
cluster = Cluster(catalog[id])

cluster.extract_profile()
cluster.fit_model()

cluster.prof.Plot(model=cluster.mod)
plt.show()

cluster.model_mcmc(n_samples=100)
cluster.ps2D()
cluster.ps_mcmc(n_samples=10)
cluster.fit_P3D()

#%%
from matplotlib.colors import SymLogNorm

regsizepix = cluster.psc.regsize
factshift = 1.5
minx = int(np.round(cluster.prof.cx - factshift * regsizepix))
maxx = int(np.round(cluster.prof.cx + factshift * regsizepix + 1))
miny = int(np.round(cluster.prof.cy - factshift * regsizepix))
maxy = int(np.round(cluster.prof.cy + factshift * regsizepix + 1))

outmod = cluster.model_best_fit_image[miny:maxy, minx:maxx]
image = fits.getdata(cluster.datalink)[miny:maxy, minx:maxx]

plt.figure(figsize=(8,8))
plt.imshow(np.log10(outmod))
plt.axis('off')
plt.show()

residual = (image-outmod)*outmod/outmod

plt.figure(figsize=(8,8))
plt.imshow(residual,
           norm=SymLogNorm(linthresh=0.03,
                           vmin= -np.nanmax(np.abs(residual)),
                           vmax = np.nanmax(np.abs(residual))),
           cmap='bwr')
plt.axis('off')
plt.colorbar()
plt.show()

plt.figure(figsize=(8,8))

for i, file in enumerate(cluster.psc.convlink[0:9]):

    plt.subplot(331+i)
    scale = fits.getdata(file)
    plt.imshow(scale)
    plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
plt.plot(cluster.psc.k, np.abs(cluster.psc.ps))
plt.fill_between(cluster.psc.k, np.abs(cluster.psc.ps)- np.diag(cluster.ps_covariance)**0.5, np.abs(cluster.psc.ps)+ np.diag(cluster.ps_covariance)**0.5, alpha=0.5)
plt.xlabel(r'$k$ [kpc$^{-1}$]')
plt.ylabel(r'$P_{2D}$ [kpc$^{4}$]')
plt.loglog()
plt.tight_layout()
plt.show()

#%% 
from importlib import reload
reload(graph)

graph.dashboard(cluster)