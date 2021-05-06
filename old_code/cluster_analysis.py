import os
import sys
import numpy as np
from turb.graphics import dashboard
from turb.extract_ps import Extractor
from astropy.table import Table

if __name__ == "__main__":

    id = int(sys.argv[1])
    catalog = Table.read('./data/master_table.fits')
    cluster = Extractor(catalog[id])

    cluster.extract_profile()
    cluster.fit_model()
    cluster.model_posterior_sample(n_samples=1000)
    cluster.extract_ps()
    cluster.ps_mcmc(n_samples=100)
    cluster.fit_P3D()

    np.savetxt('power_spectrum/ps2d_{}.txt'.format(cluster.name), (cluster.psc.k, cluster.psc.ps))
    np.savetxt('power_spectrum/pscov_{}.txt'.format(cluster.name), cluster.ps_covariance)
    np.savetxt('power_spectrum/res_{}.txt'.format(cluster.name), cluster.res)

    dashboard(cluster, outfile='analysis_overview/{}.html'.format(cluster.name))
