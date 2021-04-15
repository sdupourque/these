import numpy as np
from astropy.table import Table, Column, vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

#%% XCOP
catalog_XCOP = Table.read('data/XCOP/XCOP_master_table.fits')
catalog_XCOP['tag'] = 'XCOP'

# Remove HydraA
catalog_XCOP.remove_row(list(catalog_XCOP['NAME']).index('HydraA'))

catalog_XCOP.keep_columns(['NAME', 'REDSHIFT', 'R500_HSE', 'RA', 'DEC'])
catalog_XCOP.rename_column('R500_HSE', 'R500')

catalog_XCOP['THETA500'] = catalog_XCOP['R500']/cosmo.kpc_proper_per_arcmin(catalog_XCOP['REDSHIFT']).value
catalog_XCOP['THETA500'].unit = 'arcmin'


catalog_XCOP['TAG'] = 'XCOP'
catalog_XCOP['SZ'] = 'None'


#%% ACT

catalog_CHEXMATE = Table.read('data/ACT/CHEXMATE_In_ACTDR4-ymap.fits')
catalog_CHEXMATE.keep_columns(['NAME', 'REDSHIFT', 'R500', 'RA', 'DEC'])

catalog_CHEXMATE['THETA500'] = catalog_CHEXMATE['R500']
catalog_CHEXMATE['THETA500'].unit = 'arcmin'
catalog_CHEXMATE['R500'] = np.round(cosmo.kpc_proper_per_arcmin(catalog_CHEXMATE['REDSHIFT']).value * catalog_CHEXMATE['R500'])
catalog_CHEXMATE['R500'].unit = 'kpc'
catalog_CHEXMATE['SZ'] = 'ACT'

for name in catalog_CHEXMATE['NAME']:

    catalog_CHEXMATE['NAME'][list(catalog_CHEXMATE['NAME']).index(name)] = name.replace(" ", "")

catalog_CHEXMATE['TAG'] = 'CHEXMATE'

#%% NIKA2

catalog_LPSZ = Table.read('data/NIKA2/CHEXMATE_In_NIKA2LPSZ.fits')
catalog_LPSZ.keep_columns(['NAME', 'REDSHIFT', 'R500', 'RA', 'DEC'])

catalog_LPSZ['THETA500'] = catalog_LPSZ['R500']
catalog_LPSZ['THETA500'].unit = 'arcmin'
catalog_LPSZ['R500'] = np.round(cosmo.kpc_proper_per_arcmin(catalog_LPSZ['REDSHIFT']).value * catalog_LPSZ['R500'])
catalog_LPSZ['R500'].unit = 'kpc'
catalog_LPSZ['SZ'] = 'NIKA2'

for name in catalog_LPSZ['NAME']:

    catalog_LPSZ['NAME'][list(catalog_LPSZ['NAME']).index(name)] = name.replace(" ", "")

catalog_LPSZ['TAG'] = 'LPSZ'

#%%

master_table = vstack([catalog_XCOP, catalog_CHEXMATE, catalog_LPSZ])

nb_char = 50
path = Column(name='PATH', dtype=np.dtype(('U', nb_char)) ,  length=len(master_table))
dat = Column(name='datalink', dtype=np.dtype(('U', nb_char)) ,  length=len(master_table))
exp = Column(name='explink', dtype=np.dtype(('U', nb_char)) , length=len(master_table))
bkg = Column(name='bkglink', dtype=np.dtype(('U', nb_char)) , length=len(master_table))
reg = Column(name='reg', dtype=np.dtype(('U', nb_char)) , length=len(master_table))

for i, (name, sz) in enumerate(zip(master_table['NAME'], master_table['SZ'])):

    if sz== 'None':

        path[i] = 'data/XCOP/{}'.format(name)
        dat[i] = 'mosaic_{}.fits.gz'.format(name.lower())
        exp[i] = 'mosaic_{}_expo.fits.gz'.format(name.lower())
        bkg[i] = 'mosaic_{}_bkg.fits.gz'.format(name.lower())
        reg[i] = 'src_ps.reg'

    if sz == 'ACT':

        path[i] = 'data/ACT/{}'.format(name)
        dat[i] = 'epic-obj-im-700-1200.fits.gz'
        exp[i] = 'epic-exp-im-700-1200.fits.gz'
        bkg[i] = 'epic-back-oot-sky-700-1200.fits.gz'
        reg[i] = 'srclist_{}.reg'.format(name)
        
    if sz == 'NIKA2':

        path[i] = 'data/NIKA2/{}'.format(name)
        dat[i] = 'epic-obj-im-700-1200.fits.gz'
        exp[i] = 'epic-exp-im-700-1200.fits.gz'
        bkg[i] = 'epic-back-oot-sky-700-1200.fits.gz'
        reg[i] = 'srclist_{}.reg'.format(name)

master_table.add_columns([path, dat, exp, bkg, reg])
master_table.write('data/master_table.fits', format='fits', overwrite=True)