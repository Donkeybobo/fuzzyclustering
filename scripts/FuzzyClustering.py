## This file contains the FuzzyClustering class definition
import math
import numpy as np

class FuzzyClustering:
    
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
    
    ## Function to update potentials for all data points
    def _updatePotential(self):
        # get index (k) of data point with max potential
        k = self._getIndexWithMaxPotential()
        
        # get data point and corresponding potential for k-th data point
        pk = self.potentials[k]
        xk = self.data[k]
        
        # update potentials
        for i in range(len(self.potentials)):
            self.potentials[i] = self.potentials[i] - pk * math.exp(-beta * FuzzyClustering._calclualteDistance(self.data[i], xk))
            
    
    ## Initialize an object with parameters
    def __init__(self, 
                 num_particles, 
                 num_generations, 
                 num_dimensions, 
                 parameter_lower_bounds = -2, 
                 parameter_upper_bounds = 2):
        """This function initialize a RuleExtractionQPSO with given inputs.
        num_particles: number of particles to use.
        num_generations: number of generations to iterate when solving an optimization problem.
        num_dimensions: the dimension of the particle, i.e., the number of parameters this QPSO tries to optimize.
        parameter_lower_bounds: lower bounds of the INITIAL parameters.
        parameter_upper_bounds: upper bounds of the INITIAL parameters.
        """
        ## Parameters to initialize QPSO
        self.num_particles = num_particles
        self.num_generations = num_generations
        self.num_dimensions = num_dimensions
        
        self.parameter_lower_bounds = parameter_lower_bounds
        self.parameter_upper_bounds = parameter_upper_bounds
        
        ## Initialize variables needed when running QPSO
        self.gbest = np.zeros(num_dimensions)
        self.best_particles = np.zeros(num_dimensions)
        
        self.MINIMUM = 0