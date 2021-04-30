from turb.extract_ps import Extractor

class Analysis:

    def __init__(self, master_table, config):

        self.clusters_to_analyze = []
        self.config = config

        for cluster_row in master_table:

            self.clusters_to_analyze.append(Extractor.from_catalog_row(cluster_row))

    def __iter__(self):

        return iter(self.clusters_to_analyze)

    def __getitem__(self,index):

        return self.clusters_to_analyze[index]

    def launch(self):

        for cluster in self:

            cluster.doit(**self.config)