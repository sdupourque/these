import os
import sys
import numpy as np
from turbulence_analysis.graph import dashboard
from turbulence_analysis.cluster import Cluster
from astropy.table import Table

if __name__ == "__main__":

    catalog = Table.read('data/master_table.fits')

    for id in range(len(catalog)):

        os.system('nohup srun -p irap -n 2 -J cluster{} --time=12:00:00 python cluster_analysis.py {} > {}.out 2>&1 &'.format(id,id,id))