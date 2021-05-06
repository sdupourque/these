import yaml
import multiprocessing
from turb.analysis import Analysis
from turb.project_model import BetaModel
from astropy.table import Table

config = {'ps_region_size': '2r500',
          'profile_model': BetaModel(),
          'ps_samples' : 1000}

master_table = Table.read('data/master_table.fits')
xcop_table = master_table[master_table['SAMPLE'] == 'XCOP']

if __name__ == '__main__':

    analysis = Analysis(xcop_table, config, 'XCOP_Beta')
    analysis.launch()
