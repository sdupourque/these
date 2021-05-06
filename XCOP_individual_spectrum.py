import numpy as np
from turb.extract_ps import Extractor
from astropy.table import Table
from turb.project_model import BetaModel
from turb.graphics import dashboard
import matplotlib.pyplot as plt

res = Table.read('analysis_results/XCOP_Median/XCOP_Median.fits')

fig, ax = plt.subplots(ncols=3, nrows=4, constrained_layout=True, figsize=(10, 12))

for i, row in enumerate(res):

    ps_samples = row['ps_samples']
    ps_noise_samples = row['ps_noise_samples']
    k = row['k']
    ax[i % 4, i % 3].loglog()
    ax[i % 4, i % 3].set_title(row['NAME'])
    for j in range(ps_samples.shape[0]):

        ax[i % 4, i % 3].plot(k, ps_samples[j, :], color='lightblue', alpha=0.01)
        ax[i % 4, i % 3].plot(k, ps_noise_samples[j, :], color='lightcoral', alpha=0.01)

plt.show()