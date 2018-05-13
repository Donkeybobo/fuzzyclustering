## This file contains the RuleExtractionQPSO class definition
import random
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
        
        
        
    def 