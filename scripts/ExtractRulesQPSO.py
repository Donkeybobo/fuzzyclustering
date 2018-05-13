## This file contains the RuleExtractionQPSO class definition
import math
import numpy as np

class ExtractRulesQPSO:
    
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
        
        
    ## Return exponential membership with given inputs
    def _expMembership(x, m, s):
        return(math.exp(-0.5 * math.pow((x - m) / s, 2)))
    
    ## Return max membership across all rules for one data point (a vector)
    def _maxMembershipFromAllRules(centers, stds, data):
        """This is a utility function to return max membership of all rules.
        centers: 2-dimentional array for all center parameters of the rules, 
        where the first dimension is equal to the number of rules, 
        the second is equal to the number of dimensions of the data.
        
        std: 2-dimentional array for all std parameters of the rules, 
        where the first dimension is equal to the number of rules, 
        the second is equal to the number of dimensions of the data.
        
        data: 1-dimensional array for a given data point, where the length is equal to the dimension"""
        mu = 0
        
        ## For all rules (note that len(centers) and len(stds) should always be the same)
        for m, s in zip(centers, stds):
            curMu = 1
            
            ## For all dimensions of the input data point
            for d in data:
                curMu *= _expMembership(d, m, s)
            
            mu = curMu if curMu > mu else mu
        
        return(mu)
        
        
        