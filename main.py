import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.table import Table

catalog = Table.read('data/master_table.fits')

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

#%% Kolmogorov
def cascade(k):

    k_cut = 1e-4
    k_disp = 1e-1
    k_pivot = 1e-2

    return np.exp(-(k_cut/k)**2 - (k/k_disp)**2)*(k/k_pivot)**(-11/3)


k = np.geomspace(1e-5, 1 ,1000)

fig, ax = plt.subplots(figsize=(4,4))
plt.plot(k, cascade(k), color='black')
plt.vlines(7.5e-5, 1e-8, 1e10,linestyles='dotted')
plt.vlines(1.5e-1, 1e-8, 1e10,linestyles='dotted')
plt.loglog()
plt.ylim(top=1e9, bottom=1e-7)
plt.text(2e-4, 3e6, 'Inertial range $\propto -11/3$', fontsize=16,
              rotation=-51,
              rotation_mode='anchor')

plt.text(1.5e-5, 1e-1, 'Injection Scale', fontsize=16,
              rotation=90,
              rotation_mode='anchor')

plt.text(4e-1, 1e-1, 'Dissipation Scale', fontsize=16,
              rotation=90,
              rotation_mode='anchor')

plt.xlabel(r'$k$ [kpc$^{-1}$]', fontsize=14)
plt.ylabel(r'$P_{3D}$ [kpc$^{6}$]', fontsize=14)

plt.tight_layout()

plt.savefig('figures/kolmogorov.png')

plt.show()

#%% Plan R500 Z

fig, ax = plt.subplots(figsize=(4,4))

xcop = catalog[catalog['TAG']=='XCOP']
chexmate = catalog[catalog['TAG']=='CHEXMATE']
lpsz = catalog[catalog['TAG']=='LPSZ']

plt.xlabel(r'Z', fontsize=14)
plt.ylabel(r'$R_{500}$ [kpc]', fontsize=14)

plt.scatter(xcop['REDSHIFT'], xcop['R500'], color='blue', marker="o", label='XCOP')
plt.scatter(chexmate['REDSHIFT'], chexmate['R500'], color='red', marker="D", label='CHEX-MATE')
plt.scatter(lpsz['REDSHIFT'], lpsz['R500'], color='green', marker="*", label='LPSZ')
plt.legend()
plt.tight_layout()

plt.savefig('figures/kolmogorov.png')

plt.show()