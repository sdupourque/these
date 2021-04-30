import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.table import Table
from matplotlib.lines import Line2D

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

catalog = Table.read('data/master_table.fits')

#%%
norm_list = []
alpha_list = []
k_in_list = []
name_list = []

for cluster in catalog:

    try:
        k, ps2d = np.loadtxt('power_spectrum/ps2d_{}.txt'.format(cluster['NAME']))
        cov = np.loadtxt('power_spectrum/pscov_{}.txt'.format(cluster['NAME']))
        k_in, norm, alpha = np.loadtxt('power_spectrum/res_{}.txt'.format(cluster['NAME']))
        if True:#np.max(ps2d) < 1e4 and norm < -8 and alpha <6:
            norm_list.append(norm)
            k_in_list.append(k_in)
            alpha_list.append(alpha)
            name_list.append(cluster['NAME'])

        else:
            norm_list.append(np.nan)
            k_in_list.append(np.nan)
            alpha_list.append(np.nan)
            name_list.append(cluster['NAME'])
    except:

        pass

catalog['norm'] = norm_list
catalog['alpha'] = alpha_list
catalog['k_in'] = k_in_list

xcop = catalog[catalog['TAG'] == 'XCOP']
chexmate = catalog[catalog['TAG'] == 'CHEXMATE']
lpsz = catalog[catalog['TAG'] == 'LPSZ']


#%%
fig = plt.figure(figsize=(4, 4))
for cluster in catalog:
    try:
        k, ps2d = np.loadtxt('power_spectrum/ps2d_{}.txt'.format(cluster['NAME']))
        cov = np.loadtxt('power_spectrum/pscov_{}.txt'.format(cluster['NAME']))
        plt.plot(k, ps2d, color='black', alpha=0.1)

        if cluster['TAG'] == 'XCOP':
            plt.fill_between(k, ps2d - np.diag(cov) ** 0.5, ps2d + np.diag(cov) ** 0.5 , alpha=0.05, color='blue')
        if cluster['TAG'] == 'CHEXMATE':
            plt.fill_between(k, ps2d - np.diag(cov) ** 0.5, ps2d + np.diag(cov) ** 0.5, alpha=0.05, color='red')
        if cluster['TAG'] == 'LPSZ':
            plt.fill_between(k, ps2d - np.diag(cov) ** 0.5, ps2d + np.diag(cov) ** 0.5, alpha=0.05, color='green')
    except:
        print('{} analysis failed'.format(cluster['NAME']))
        pass

plt.loglog()
plt.xlabel('$k$ [kpc$^{-1}$]')
plt.ylabel('$P_{2D}$ [kpc$^{4}$]')
custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4)]
plt.legend(custom_lines, ['XCOP', 'CHEX-MATE', 'LPSZ'])
plt.tight_layout()
plt.savefig('figures/p2d.png', transparent=True, dpi=600)
plt.show()

#%%
fig = plt.figure(figsize=(4, 4))
for cluster in catalog:
    try:
        k, ps2d = np.loadtxt('power_spectrum/ps2d_{}.txt'.format(cluster['NAME']))
        ps2d = np.abs(ps2d)
        cov = np.loadtxt('power_spectrum/pscov_{}.txt'.format(cluster['NAME']))

        if cluster['TAG'] == 'XCOP':
            plt.plot(k, ps2d, color='black', alpha=0.1)
            plt.fill_between(k, ps2d - np.diag(cov) ** 0.5, ps2d + np.diag(cov) ** 0.5 , alpha=0.05, color='blue')
        if cluster['TAG'] == 'CHEXMATE':
            pass
        if cluster['TAG'] == 'LPSZ':
            pass
    except:
        print('{} analysis failed'.format(cluster['NAME']))
        pass

plt.loglog()
plt.xlabel('$k$ [kpc$^{-1}$]')
plt.ylabel('$P_{2D}$ [kpc$^{4}$]')

custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4)]
plt.legend(custom_lines, ['XCOP'])
plt.tight_layout()
plt.savefig('figures/p2d_xcop.png', transparent=True, dpi=600)
plt.show()

#%%
fig = plt.figure(figsize=(4, 4))
for cluster in catalog:
    try:
        k, ps2d = np.loadtxt('power_spectrum/ps2d_{}.txt'.format(cluster['NAME']))
        ps2d = np.abs(ps2d)

        cov = np.loadtxt('power_spectrum/pscov_{}.txt'.format(cluster['NAME']))

        if cluster['TAG'] == 'XCOP':
            pass
        if cluster['TAG'] == 'CHEXMATE':
            plt.plot(k, ps2d, color='black', alpha=0.1)
            plt.fill_between(k, ps2d - np.diag(cov) ** 0.5, ps2d + np.diag(cov) ** 0.5, alpha=0.05, color='red')
        if cluster['TAG'] == 'LPSZ':
            pass
    except:
        print('{} analysis failed'.format(cluster['NAME']))
        pass

plt.loglog()
plt.xlabel('$k$ [kpc$^{-1}$]')
plt.ylabel('$P_{2D}$ [kpc$^{4}$]')
custom_lines = [Line2D([0], [0], color='red', lw=4)]
plt.legend(custom_lines, ['CHEX-MATE'])
plt.tight_layout()
plt.savefig('figures/p2d_chex.png', transparent=True, dpi=600)
plt.show()

#%%
fig = plt.figure(figsize=(4, 4))
for cluster in catalog:
    try:
        k, ps2d = np.loadtxt('power_spectrum/ps2d_{}.txt'.format(cluster['NAME']))
        ps2d = np.abs(ps2d)
        cov = np.loadtxt('power_spectrum/pscov_{}.txt'.format(cluster['NAME']))

        if cluster['TAG'] == 'XCOP':
            pass
        if cluster['TAG'] == 'CHEXMATE':
            pass
        if cluster['TAG'] == 'LPSZ':
            plt.plot(k, ps2d, color='black', alpha=0.1)
            plt.fill_between(k, ps2d - np.diag(cov) ** 0.5, ps2d + np.diag(cov) ** 0.5, alpha=0.05, color='green')
    except:
        print('{} analysis failed'.format(cluster['NAME']))
        pass

plt.loglog()
plt.xlabel('$k$ [kpc$^{-1}$]')
plt.ylabel('$P_{2D}$ [kpc$^{4}$]')
custom_lines = [Line2D([0], [0], color='green', lw=4)]
plt.legend(custom_lines, ['LPSZ'])
plt.tight_layout()
plt.savefig('figures/p2d_lpsz.png', transparent=True, dpi=600)
plt.show()

#%%
fig = plt.figure(figsize=(3, 3))
for cluster in catalog:
    try:
        k, ps2d = np.loadtxt('power_spectrum/ps2d_{}.txt'.format(cluster['NAME']))
        ps2d = np.abs(ps2d)
        cov = np.loadtxt('power_spectrum/pscov_{}.txt'.format(cluster['NAME']))

        if np.max(ps2d) < 1e4:
            plt.plot(k*cluster['R500'], ps2d, color='black', alpha=0.1)
            if cluster['TAG'] == 'XCOP':
                plt.fill_between(k * cluster['R500'],
                                     (ps2d-np.diag(cov)**0.5),
                                     (ps2d+np.diag(cov)**0.5),
                                     alpha=0.05, color='blue')
            if cluster['TAG'] == 'CHEXMATE':
                plt.fill_between(k * cluster['R500'],
                                     (ps2d-np.diag(cov)**0.5),
                                     (ps2d+np.diag(cov)**0.5),
                                     alpha=0.05, color='red')
            if cluster['TAG'] == 'LPSZ':
                plt.fill_between(k * cluster['R500'],
                                     (ps2d-np.diag(cov)**0.5),
                                     (ps2d+np.diag(cov)**0.5),
                                     alpha=0.05, color='green')

    except:
        pass

plt.loglog()
plt.xlabel('$k/k_{500}$')
plt.ylabel('$P_{2D}$')
custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4)]
plt.legend(custom_lines, ['XCOP', 'CHEX-MATE', 'LPSZ'])
plt.tight_layout()
plt.savefig('figures/p2d_renormalized.png',transparent=True,  dpi=600)
plt.show()

#%%
fig = plt.figure(figsize=(4, 4))
for cluster in catalog:
    try:
        k, ps2d = np.loadtxt('power_spectrum/ps2d_{}.txt'.format(cluster['NAME']))
        ps2d = np.abs(ps2d)
        cov = np.loadtxt('power_spectrum/pscov_{}.txt'.format(cluster['NAME']))

        A2d = np.sqrt(2*np.pi*k**2*ps2d)
        dA2d = 2*np.sqrt(2 * np.pi * k ** 2 / ps2d)
        plt.plot(k * cluster['R500'], A2d, color='black',alpha=0.1)

        if cluster['TAG'] == 'XCOP':
            plt.fill_between(k * cluster['R500'],A2d - dA2d,A2d + dA2d,alpha=0.05, color='blue')
        if cluster['TAG'] == 'CHEXMATE':
            plt.fill_between(k * cluster['R500'],A2d - dA2d,A2d + dA2d,alpha=0.05, color='red')
        if cluster['TAG'] == 'LPSZ':
            plt.fill_between(k * cluster['R500'],A2d - dA2d,A2d + dA2d,alpha=0.05, color='green')

    except:
        pass

plt.loglog()
plt.xlabel('$k/k_{500}$')
plt.ylabel('$A_{2D}$')
custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4)]
plt.legend(custom_lines, ['XCOP', 'CHEX-MATE', 'LPSZ'])
plt.tight_layout()
plt.savefig('figures/a2d.png',transparent=True,  dpi=600)
plt.show()

#%%
# definitions for the axes
left, width = 0.13, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(4, 4))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

# use the previously defined function
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
ax.scatter(xcop['norm'], xcop['alpha'], marker='o', color='blue', label='XCOP')
ax.scatter(chexmate['norm'], chexmate['alpha'], marker='D', color='red', label='CHEX-MATE')
ax.scatter(lpsz['norm'], lpsz['alpha'], marker='*', color='green', label='LPSZ')
ax_histx.hist(catalog['norm'], bins=10, color='black')
ax_histy.hist(catalog['alpha'], bins=10, orientation='horizontal', color='black')
ax_histx.yaxis.set_visible(False)
ax_histy.yaxis.set_visible(False)
ax_histx.xaxis.set_visible(False)
ax_histy.xaxis.set_visible(False)
#ax.hlines(11/3, min(norm_list), max(norm_list))

ax.legend(fontsize=6)
ax.set_xlabel('log(norm)')
ax.set_ylabel(r'$\alpha$')
plt.savefig('figures/parameters_dispersion.png', transparent=True, dpi=600)
plt.show()

#%%
import matplotlib.colors as colors
fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3, 3))

norm1=colors.Normalize(vmin=min(catalog['alpha']),vmax=max(catalog['alpha']))
mp1 = axarr[0].scatter(xcop['REDSHIFT'], xcop['R500'], c=xcop['alpha'],marker="o", label='XCOP', norm=norm1)
axarr[0].scatter(chexmate['REDSHIFT'], chexmate['R500'], c=chexmate['alpha'],marker="D", label='CHEX-MATE', norm=norm1)
axarr[0].scatter(lpsz['REDSHIFT'], lpsz['R500'], c=lpsz['alpha'],marker="*", label='LPSZ', norm=norm1)
axarr[0].set_ylabel('$R_{500}$ [kpc]')
#plt.legend()

norm2=colors.Normalize(vmin=min(catalog['norm']),vmax=max(catalog['norm']))
mp2 = axarr[1].scatter(xcop['REDSHIFT'], xcop['R500'], c=xcop['norm'],marker="o", label='XCOP', norm=norm2,cmap='plasma')
axarr[1].scatter(chexmate['REDSHIFT'], chexmate['R500'], c=chexmate['norm'],marker="D", label='CHEX-MATE', norm=norm2,cmap='plasma')
axarr[1].scatter(lpsz['REDSHIFT'], lpsz['R500'], c=lpsz['norm'],marker="*", label='LPSZ', norm=norm2,cmap='plasma')
axarr[1].set_xlabel('z')
axarr[1].set_ylabel('$R_{500}$ [kpc]')

fig.colorbar(mp1, ax=axarr[0], label=r'$\alpha$')
fig.colorbar(mp2, ax=axarr[1], label=r'log(norm) [kpc$^{6}$]')
plt.tight_layout()
plt.savefig('figures/params_rz.png',transparent=True,  dpi=600)
plt.show()