import os
import numpy as np
import yaml
from astropy.table import Table
from multiprocessing import Pool
from turb.extract_ps import Extractor
from turb.graphics import report

def run(arg):

    extraction_done = False
    n_try = 0

    while (not extraction_done) or n_try >4:
        #Assuring that the extraction is done
        #Very bad tho
        try :
            cluster, config = arg
            extraction_done = cluster.doit(**config)
        except :
            n_try += 1
            print('{} analysis failed {} time'.format(cluster.name, n_try))

    return cluster

class Analysis(list):

    def __init__(self, master_table, config, name):

        self.clusters_to_analyze = []
        self.master_table = master_table
        self.config = config

        self.name = name
        self.analysis_path = 'analysis_results/{}'.format(name)
        self.overview_path = os.path.join(self.analysis_path, 'overview')
        self.clusters_path = os.path.join(self.analysis_path, 'clusters')
        
        path_list = [self.analysis_path, self.overview_path, self.clusters_path]
        
        for path in path_list:
            if not os.path.exists(path):
                try:
                    os.mkdir(path)
                    print('Directory created at {}'.format(path))
                except:
                    #Handling an error happening in multiproc
                    print('Ignored already exists error')

        for row in master_table:
            
            path = os.path.join(self.clusters_path, row['NAME'])
            self.append(Extractor.from_catalog_row(row, analysis_path = self.analysis_path))
            if not os.path.exists(path):
                try:
                    os.mkdir(path)
                    print('Directory created at {}'.format(path))
                except:
                    print('Ignored already exists error')

        with open(os.path.join(self.analysis_path, 'config.yaml'), 'w+') as file:
            yaml.dump(config, file)

    @classmethod
    def load(cls, name):

        with open('analysis_results/{}/config.yaml'.format(name), 'r') as file:
            config = yaml.safe_load(file)
        master_table = Table.read('analysis_results/{}/{}.fits'.format(name, name))
        return cls(master_table, config, name)

    def launch(self):

        #int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        with Pool() as p:

            arg_list = [(self[i], self.config) for i in range(len(self))]
            #Doing this because multiprocessing create new cluster objects with same properties
            self.clear()
            self.extend(p.map(run, arg_list))

        self.master_table['k'] = [np.zeros_like(self[0].k) for _ in range(len(self))]
        self.master_table['ps'] = [np.zeros_like(self[0].ps) for _ in range(len(self))]
        self.master_table['ps_samples'] = [np.zeros_like(self[0].ps_samples) for _ in range(len(self))]
        self.master_table['ps_covariance'] = [np.zeros_like(self[0].ps_covariance) for _ in range(len(self))]
        self.master_table['ps_noise'] = [np.zeros_like(self[0].ps_noise) for _ in range(len(self))]
        self.master_table['ps_noise_samples'] = [np.zeros_like(self[0].ps_noise_samples) for _ in range(len(self))]
        self.master_table['ps_noise_covariance'] = [np.zeros_like(self[0].ps_noise_covariance) for _ in range(len(self))]
        self.master_table['psf_k_cut'] = [np.copy(self[0].psf_k_cut) for _ in range(len(self))]

        for cluster in self:

            self.master_table['k'][self.master_table['NAME'] == cluster.name] = cluster.k
            self.master_table['ps'][self.master_table['NAME'] == cluster.name] = cluster.ps
            self.master_table['ps_samples'][self.master_table['NAME'] == cluster.name] = cluster.ps_samples
            self.master_table['ps_covariance'][self.master_table['NAME'] == cluster.name] = cluster.ps_covariance
            self.master_table['ps_noise'] = cluster.ps_noise
            self.master_table['ps_noise_samples'][self.master_table['NAME'] == cluster.name] = cluster.ps_noise_samples
            self.master_table['ps_noise_covariance'][self.master_table['NAME'] == cluster.name] = cluster.ps_noise_covariance
            self.master_table['psf_k_cut'][self.master_table['NAME'] == cluster.name] = cluster.psf_k_cut

        self.master_table.write(os.path.join(self.analysis_path,
                                             '{}.fits'.format(self.name)),
                                format='fits',
                                overwrite=True)

        report(self)