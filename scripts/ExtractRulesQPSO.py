## This file contains the RuleExtractionQPSO class definition
import math
import numpy as np
import matplotlib.pyplot as plt

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
        
        self.MINIMUM = 0
        
        
    ## Return exponential membership with given inputs
    def _expMembership(x, m, s):
        return(np.exp(-0.5 * np.power((np.array(x) - m) / s, 2)))
    
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
            for j in range(len(data)):
                curMu *= ExtractRulesQPSO._expMembership(data[j], m[j], s[j])
            
            mu = curMu if curMu > mu else mu
        
        return(mu)
    
    def _productMFFromAllRules(centers, stds, data):
        """
        Calculates product of MFs of all input rules (centers + stds)
        """
        mu = 1
        
        for m, s in zip(centers, stds):
            curMu = 1
            
            ## For all dimensions of the input data point
            for j in range(len(data)):
                curMu *= ExtractRulesQPSO._expMembership(data[j], m[j], s[j])
            
            mu *= curMu
        
        return mu
    
    def _returnClassWithMaxMF(all_class_centers, all_class_stds, data_point, class_labels):
        class_idx_with_max_mf = 0
        cur_max_mf = 0
        for k in range(len(all_class_centers)):
            centers_k = all_class_centers[k]
            stds_k = all_class_stds[k]
            
            cur_mf = ExtractRulesQPSO._maxMembershipFromAllRules(centers_k, stds_k, data_point)
            
            if cur_mf > cur_max_mf:
                class_idx_with_max_mf = k
                cur_max_mf = cur_mf
            
        return class_labels[class_idx_with_max_mf]
    
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
                max_membership_k = ExtractRulesQPSO._maxMembershipFromAllRules(centers_k, stds_k, data_k[i])
                
                # max membership for classes that are not k
                max_membership_not_k = 0
                
                for kk in range(len(all_class_centers)):
                    if (kk != k):
                        temp_membership = ExtractRulesQPSO._maxMembershipFromAllRules(all_class_centers[kk], 
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
        
        all_class_stds = ExtractRulesQPSO._arrangeStds(particle, all_class_centers)
        
        class_error = ExtractRulesQPSO._classificationError(all_class_centers, all_class_stds, all_class_data)

        return(class_error)
    
    ## Solver function to find the optimal parameters to minimize _objective defined above
    def solver(self, all_class_centers, all_class_data):
        # random vector
        x = np.random.rand(self.num_particles, self.num_dimensions) * (self.parameter_upper_bounds - self.parameter_lower_bounds) + self.parameter_lower_bounds
        
        # First set pbest = x
        pbest = x.copy()
        
        # calculate the initial objective functions according to x
        f_x = np.array([ExtractRulesQPSO._objective(p, all_class_centers, all_class_data) for p in x])
        f_pbest = f_x.copy()
        
        # find the index which gives the minimum objective
        g = np.where(f_pbest == f_pbest.min())[0].min()
        
        # update gbest and f_gbest
        self.gbest = pbest[g].copy()
        f_gbest = f_pbest[g]
        
        # The main optimization procedure
        for t in range(self.num_generations):
            # step size
            beta = (1 - 0.5) * (self.num_generations - 1 - t) / self.num_generations + 0.5
            
            # mbest: the average of all current particles
            mbest = np.sum(pbest, axis = 0) / self.num_particles
            
            ## iterate through all particles
            for i in range(self.num_particles):
                fi = np.random.rand(self.num_dimensions)
                p = np.array(pbest[i]) * np.array(fi) + (1 - np.array(fi)) * np.array(self.gbest)
                u = np.random.rand(self.num_dimensions)
                b = (np.array(mbest) - np.array(x[i])) * beta
                v = np.log(np.array(u)) * -1
                
                # Compute y
                prob_vector = (-1) ** np.ceil(np.array(np.random.rand(self.num_dimensions) + 0.5)) * b * v
                
                y = p + prob_vector
                
                # add cap to values in y
                y_capped = [yy if abs(yy) <= 3 else 3 for yy in y]
                
                # update x[i] and f_x[i]
                x[i] = y_capped.copy()
                f_x[i] = ExtractRulesQPSO._objective(x[i], all_class_centers, all_class_data)
                
                # update pbest and f_pbest when needed
                if (f_x[i] < f_pbest[i]):
                    pbest[i] = x[i].copy()
                    f_pbest[i] = f_x[i].copy()
            # update gbest and f_gbest    
            g = np.where(f_pbest == f_pbest.min())[0].min()
            self.gbest = pbest[g]
            f_gbest = f_pbest[g]
            self.MINIMUM = f_gbest
            
        self.best_particle = self.gbest

        
        
        
        
        
        
        
        
        
        
        
   
        
        