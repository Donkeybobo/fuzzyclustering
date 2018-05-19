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
    
    ## Return the overall classification error
    def _classificationError(all_class_centers, all_class_stds, all_class_data):
        """This function calculates the overall classification error that is to be minimized.
        
        all_class_centers: 3-dimentional array where the first dimension is for different class; 
        for each of the classes, the 2-dimensional array is all center parameters for all rules, 
        like centers used in _maxMembershipFromAllRules.
        
        all_class_stds: 3-dimentional array where the first dimension is for different class; 
        for each of the classes, the 2-dimensional array is all std parameters for all rules, 
        like stds used in _maxMembershipFromAllRules.
        
        all_class_data: 3-dimentional array where the first dimension is for different class;
        for each of the classes, the 2-dimensional array is all the available data.
        """
        
        error = 0
        
        ### iterate all available classes
        for k in range(len(all_class_centers)):
            ## parameters for class k and data
            centers_k = all_class_centers[k]
            stds_k = all_class_stds[k]
            data_k = all_class_data[k]
            
            ### iterate number of data points for each class
            for i in range(len(data_k)):
                # max membership for class k
                max_membership_k = _maxMembershipFromAllRules(centers_k, stds_k, data_k[i])
                
                # max membership for classes that are not k
                max_membership_not_k = 0
                
                for kk in range(len(all_class_centers)):
                    if (kk != k):
                        temp_membership = _maxMembershipFromAllRules(all_class_centers[kk], 
                                                                     all_class_stds[kk], 
                                                                     data_k[i])
                        
                        max_membership_not_k = temp_membership if temp_membership > max_membership_not_k else max_membership_not_k
                        
            
                error += 0.5 * math.pow(1 - max_membership_k + max_membership_not_k, 0.5)
        
        return(error)
    
    
    ## arrange particle into stds
    def _arrangeStds(particle, all_class_centers):
        """This function reshapes a 1-dimensional vector (particle) into 2-dimentional stds variable.
        The input all_class_centers is a 3-dimensional array that contains all 
        center parameters for all classes for all rules
        """
        all_class_stds = []
        start_idx = 0
        
        # iterate through classes
        for k in range(len(all_class_centers)):
            num_rules = len(all_class_centers[k])
            num_antecedents = len(all_class_centers[k][0])
            
            # reshape the current trunk of parameters into a 2-dimensional array
            stds = np.reshape(particle[start_idx:(start_idx + num_rules * num_antecedents)], (num_rules, num_antecedents))
            
            # append the std to stds
            all_class_stds.append(stds)
            
            # update start_idx
            start_idx += num_rules * num_antecedents
        
        return(all_class_stds)
    
    
    ## Objective function to minimize in QPSO
    def _objective(particle, all_class_centers, all_class_data):
        """The particle variable is all the parameters (stds) we are optimizing using QPSO.
        
        all_class_centers, all_class_data are the same as the ones we used in functions defined above.
        """
        
        all_class_stds = _arrangeStds(particle, all_class_centers)
        
        class_error = _classificationError(all_class_centers, all_class_stds, all_class_data)
        
        return(class_error)
        
        
        
        
        
        
        
        
        
        
        
        
   
        
        