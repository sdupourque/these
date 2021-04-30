import yaml
from turb.analysis import Analysis
from turb.project_model import BetaModel
from astropy.table import Table

config = {'ps_region_size': '2r500',
          'profile_model': BetaModel()}

master_table = Table.read('data/master_table.fits')
analysis = Analysis(master_table[17:19], config)

analysis.launch()