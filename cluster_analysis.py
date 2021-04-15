import os
import sys
import numpy as np
from turbulence_analysis.graph import dashboard
from turbulence_analysis.cluster import Cluster
from astropy.table import Table

if __name__ == "__main__":

    id = int(sys.argv[1])
    catalog = Table.read('./data/master_table.fits')
    cluster = Cluster(catalog[id])

    cluster.extract_profile()
    cluster.fit_model()
    cluster.model_mcmc(n_samples=1000)
    cluster.ps2D()
    cluster.ps_mcmc(n_samples=100)
    cluster.fit_P3D()

    np.savetxt('power_spectrum/ps2d_{}.txt'.format(cluster.name), (cluster.psc.k, cluster.psc.ps))
    np.savetxt('power_spectrum/pscov_{}.txt'.format(cluster.name), cluster.ps_covariance)
    np.savetxt('power_spectrum/res_{}.txt'.format(cluster.name), cluster.res)

    dashboard(cluster, outfile='analysis_overview/{}.html'.format(cluster.name))
