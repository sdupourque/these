import os
import sys
import numpy as np
from turb.graph import dashboard
from turb.extract_ps import Extractor
from astropy.table import Table

if __name__ == "__main__":

    catalog = Table.read('data/master_table.fits')

    for id in range(len(catalog)):

        os.system('nohup srun -p irap -n 2 -J cluster{} --time=12:00:00 python extract_ps.py {} > {}.out 2>&1 &'.format(id,id,id))