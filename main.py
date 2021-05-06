from turb.analysis import Analysis
from astropy.table import Table
import sys, os

sys.path.append(os.getcwd())

config_test= {'ps_region_size': '2r500',
          'profile_model': 'BetaModel',
          'ps_samples' : 2,
          }

config_XCOP_Beta = {'ps_region_size': '2r500',
          'profile_model': 'BetaModel',
          'ps_samples' : 100,
          }

config_XCOP_Median = {'ps_region_size': '2r500',
          'profile_model': None,
          'ps_samples' : 100,
          }

master_table = Table.read('data/master_table.fits')
test_table = master_table[7:9]
xcop_table = master_table[master_table['SAMPLE'] == 'XCOP']

if __name__ == '__main__':

    #analysis = Analysis(xcop_table, config_XCOP_Median, 'XCOP_Median')
    #analysis.launch()

    #print('Launching Beta analysis')

    analysis = Analysis(test_table, config_test, 'test')
    analysis.launch()
