## This file contains the FuzzyClustering class definition
import numpy as np
import os, sys
sys.path.append(os.getcwd() + '/scripts')
from ExtractRulesQPSO import ExtractRulesQPSO

class FuzzyRule:
    
    ## Initialize an object 
    def __init__(self, centers, stds, class_label):
        """This function initialize a FuzzyRule object
        centers: a vector that has centers of all antecedents
        stds: a vector that has stds of all antecedents
        class_label: A string that represents the class label of this rule
        """
        self.centers = np.array(centers)
        self.stds = np.array(stds)
        self.class_label = class_label
        
        print(os.getcwd())
        
    ## A function that plots the rule
    def plotRule(self):
        ## Plot every antecedent
        plt.plot(xs, ExtractRulesQPSO._expMembership(xs, centers_c2[2][0], qpso.best_particle[10]), label = 'x1')
        plt.plot(xs, ExtractRulesQPSO._expMembership(xs, centers_c2[2][1], qpso.best_particle[11]), label = 'x2')
        plt.legend()
        
    
                
        
        
        
        