import numpy as np
import random
import os 
from time import time
from copy import deepcopy

from functional_data import *



class SchillingTestFunctional():
    """
    Implements a two-sample test for functional data based on the approximated L2 norm. 
    Null distribution is approximated by shuffling and resampling from the the dataset.
    
    Reference: https://arxiv.org/abs/1610.06960
    """
    
    def __init__(self, data):
        self.data = data
        N = len(data)
        # array holding the types of each function in the dataset
        self.type_array = np.array([f.type for f in data]).reshape(N,1)
        self.N_k_matrix = None
        self.Null_dist = None  
            
    
    def get_N_k_matrix(self, k=10):
        """
        Computes (N x k) matrix containing the types of the k nearest neighbors.  
        """
        # when computing matrix for first time, make sure that all functions have neighbor lists
        if self.N_k_matrix is None:
            Function.get_nns_all_functions(self.data)
        
        # need data = data1 + data2, not shuffled
        # len(data1) = n, len(data2) = m
        N = len(self.data)
        mat = np.zeros((N, k), dtype='object')
        for i in range(N):

            function = self.data[i]
            # check if every function already has a neigbor list for the given data
            if len(function.neighbors) != (len(self.data)-1):
                print("Error! Not all functions have complete neigbor lists!")

            types = [f.type for f in function.neighbors[:k]]
            mat[i, :] = np.array(types)

        self.N_k_matrix = mat
    
    
    def compute_Schilling_statistic(self, matrix=None):
        """
        Computes Schilling's statistic for functional data. 
        """
        
        if matrix is None:
            # check if N_k matrix is available
            if self.N_k_matrix is None:
                print("Error! No N_k_matrix!")
                return
            matrix = self.N_k_matrix
            
        N, k = matrix.shape
        truth_values = (matrix == self.type_array)
        sum_ = np.sum(truth_values)
        schilling_stat = sum_ * 1/N * 1/k
        return schilling_stat
    
    
    def approximate_null_distribution(self, n_iter=10000):
        """
        Approximate Null distribution by shuffling and resampling from the the dataset.
        This is implemented by shuffling the (N x k) matrix.
        """
        if self.N_k_matrix is None:
            print("Error! No N_k_matrix!")
            return
        
        vals = []
        t0 = time()

        for i in range(n_iter):
            m = deepcopy(self.N_k_matrix)
            np.random.shuffle(m)
            s = self.compute_Schilling_statistic(m)
            vals.append(s)

        t1 = time()
        diff_t = t1-t0

        print(f"Completed in {diff_t:.2f} seconds!")
        self.Null_dist = vals
        
    
    def get_critical_value(self, alpha):
        """
        Returns critical Null distribution value for a given significance level alpha.
        """
        if self.Null_dist is None:
            print("Error! Null distribution has not been approximated yet!")
            return
        return np.percentile(self.Null_dist, 100-alpha)
    
    
    def two_sample_test(self, alpha, bins=70, folder="."):
        """
        Perform two-sample test using the Schilling's statistic for functional data.
        """
        crit_value = self.get_critical_value(alpha)
        test_statistic = self.compute_Schilling_statistic()
        print(f"Testing at significance level {alpha}%...")
        
        # adapt to different bin numbers etc.
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        
        ax.set_title("Approx. $H_0$ distr. and values of $T_{N,k}$")
        ax.set_xlabel("$T_{N,k}$")
        ax.set_ylabel('Density')
        
        ax.hist(self.Null_dist, bins=bins, label='Null distribution', density=True)
        ax.vlines(x=test_statistic, ymin=0, ymax=15, color='red', label='Observed value')
        ax.legend(loc='upper right')
        
        fig.text(.5, .05, f"Testing at significance level {alpha}%", ha='center')
        fig.text(.5, .0, f"Critical value is {crit_value:.2f}, test statistic has value {test_statistic:.2f}", ha='center')
        
        if test_statistic>crit_value:
            fig.text(.5, -.05, f"Null hypothesis is rejected at significance level {alpha}%", ha='center')
        else:
            fig.text(.5, -.05, f"Null hypothesis cannot be rejected at significance level {alpha}%", ha='center')
   
        plt.title("Approx. $H_0$ distr. and values of $T_{N,k}$")
        plt.savefig(os.path.join(folder, f"two_sample_test_alpha{str(alpha).replace('.','')}.pdf"), bbox_inches='tight')
        plt.close()