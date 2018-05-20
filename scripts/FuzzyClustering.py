## This file contains the FuzzyClustering class definition
import math
import numpy as np

class FuzzyClustering:
    ### Class variables
    ra = 1
    rb = 1.25
    
    alpha = 4 / ra / ra
    beta = 4 / rb / rb
    
    elower = 0.15
    eupper = 0.5
    
    ### Making sure all class variables are of class np.array
    
    ## Function to return the index of data points which has the largest potential
    def _getIndexWithMaxPotential(self):
        return(np.where(self.potentials == self.potentials.min())[0].min())
    
    ## Function to simply set k-th data point's potential to 0
    def _setKthPotential(self, k):
        self.potentials[k] = 0
    
    ## Function to calcualte the Euclidean distance between two data points
    def _calclualteDistance(x1, x2):
        # making sure x1 and x2 are of class np.array
        x1 = np.array(x1)
        x2 = np.array(x2)
        
        return(((x1 - x2) ** 2).sum())
    
    ## Function to normalize data 
    def _normalizeData(self):
        
        min_max = (np.amin(self.data, axis = 0), np.amax(self.data, axis = 0))
        
        # set normalizing_factors
        self.normalizing_factors = (2 / (min_max[1] - min_max[0]), -(min_max[1] + min_max[0]) / (min_max[1] - min_max[0]))
        
        # apply normalizing factors to original data
        for j in range(len(self.data[0])):
            self.data[j] = self.normalizing_factors[0][j] * self.data[j] + self.normalizing_factors[1][j]
    
    ## Function to calculate initial potential for data point i
    def _calculateInitialPotential(self, i):
        p = 0
        
        for j in range(len(self.data)):
            p += math.exp(-FuzzyClustering.alpha * FuzzyClustering._calclualteDistance(self.data[i], self.data[j]))
            
        return(p)
    
    ## function to set initial potential for all data points
    def _setInitialPotentials(self):
        ## initialize potentials
        self.potentials = np.zeros(len(self.data))
        
        for i in range(len(self.potentials)):
            self.potentials[i] = self._calculateInitialPotential(i)
    
    ## Function to update potentials for all data points
    def _updatePotential(self):
        # get index (k) of data point with max potential
        k = self._getIndexWithMaxPotential()
        
        # get data point and corresponding potential for k-th data point
        pk = self.potentials[k]
        xk = self.data[k]
        
        # update potentials
        for i in range(len(self.potentials)):
            self.potentials[i] = self.potentials[i] - pk * math.exp(-FuzzyClustering.beta * FuzzyClustering._calclualteDistance(self.data[i], xk))
            
    ## Function to return the shortest distance between data[k] to all previous selected cluster centers
    def _getDMin(self, k):
        dist = []
        
        for center in self.cluster_centers:
            dist.append(FuzzyClustering._calclualteDistance(self.data[k], center))
            
        return(np.min(dist))
            
    
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
        
    ## extract clustering centers
    def extractClusteringCenters(self):
        ## Stop until all criteria met
        while True:
            
            ## update potentials accordingly
            if (self.grey_center_index >= 0):
                self._setKthPotential(self.grey_center_index)
                self.grey_center_index = -1
            else:
                self._updatePotential()
                
            ## Find pkstar
            k = _getIndexWithMaxPotential()
            pkstar = self.data[k]
            
            ## update potentials and determine whether to continue
            if (pkstar > FuzzyClustering.eupper * self.p1star):
                # append the new cluster center into the list
                self.cluster_centers.append(self.data[k])
            elif (pkstar < FuzzyClustering.elower * self.p1star):
                break
            else:
                d_min = self._getDMin(k)
                
                if (d_min / FuzzyClustering.ra + pkstar / p1star >= 1):
                    # append the new cluster center
                    self.cluster_centers.append(self.data[k])
                else:
                    self.grey_center_index = k
                    
    ## Get denormalized cluster centers
    def getDenormalizedClusterCenters(self):
        denormalized = self.data.copy()
        
        # apply normalizing factors to get data into original scales
        for j in range(len(denormalized[0])):
            denormalized[j] = (denormalized[j] - self.normalizing_factors[1][j]) / self.normalizing_factors[0][j] 
            
        return(denormalized)
                
        
        
        
        