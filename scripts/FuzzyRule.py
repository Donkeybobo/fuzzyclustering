## This file contains the FuzzyClustering class definition
import numpy as np

class FuzzyRule:
    
    ## Initialize an object 
    def __init__(self, data):
        """This function initialize a FuzzyClustering object and set necessary parameters
        data: 2-dimensional array
        """
        self.data = np.array(data)
        self.cluster_centers = []
        
        ### helper variable
        self.grey_center_index = -1
        
        # normalize data points
        self._normalizeData()
        
        # initialize potentials for all data points
        self._setInitialPotentials()
        
        # find the first cluster center and append it to cluster_centers
        idx = self._getIndexWithMaxPotential()
        self.cluster_centers.append(self.data[idx])
        
        # save p1star
        self.p1star = self.potentials[idx]
        
    
                
        
        
        
        