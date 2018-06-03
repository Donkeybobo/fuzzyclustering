## This file contains the FuzzyClustering class definition
import numpy as np
import os, sys
import matplotlib.pyplot as plt

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
        self.num_antecedents = len(self.centers)
        
        
    ## A function that plots the rule
    def plotRule(self):
        
        ## set plot parameters
        plt.figure()
        plt.figure(figsize=(15,7))
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        
        ## Plot every antecedent
        for i in range(self.num_antecedents):
            
            # set xs, looking at mean +- 2std
            xs = np.linspace(self.centers[i] - 2 * self.stds[i], self.centers[i] + 2 * self.stds[i], 1000)
            
            plt.subplot(1, self.num_antecedents, i + 1)
            plt.plot(xs, ExtractRulesQPSO._expMembership(xs, self.centers[i], self.stds[i]), 
                     label = 'x' + str(i + 1), linewidth = 5)
            plt.xlabel('x' + str(i + 1), fontsize=18)       
            
            if (i == 0):
                plt.ylabel('Membership', fontsize=18)
            
#         plt.suptitle('Rule for class ' + self.class_label, fontsize = 24)