import os
import numpy as np
import matplotlib.pyplot as plt


def average_K_diffs(datasets, results_dest=".", n_std=1):
    """
    ...
    """
    
    # average over K functions 
    K_data_averaged = []
    plt.figure(figsize=(8,8))
    plt.title(f"Averaged K functions, {n_std} std error envelope")
    
    # e.g. datasets = [data_clca, data_clcb]
    for data in datasets:
        # data[i] = [range_of_t, K_diff, K_ring_diff, clc_type, filename, z]
        # where i are idices over all files processed in the current run
        range_of_t = data[0][0]
        clc_type = data[0][3]
        K_diffs = [d[1] for d in data]

        K_diffs_mean = np.mean(K_diffs, axis=0)
        K_diffs_std = np.std(K_diffs, axis=0) 
        K_data_averaged.append([range_of_t, K_diffs_mean, K_diffs_std])

        plt.plot(range_of_t, K_diffs_mean, label=clc_type)
        plt.fill_between(range_of_t, K_diffs_mean-K_diffs_std*n_std, K_diffs_mean+K_diffs_std*n_std, alpha=0.3)

    plt.xlabel("$t$")
    plt.ylabel("$K(t)$")
    plt.legend(loc="upper right")
    path = os.path.join(results_dest, "K_functions_averaged.pdf")
    plt.savefig(path)
    
    
    # average over K ring functions 
    K_ring_data_averaged = []
    plt.figure(figsize=(8,8))
    plt.title(f"Averaged K ring functions, {n_std} std error envelope")

    # e.g. datasets = [data_clca, data_clcb]
    for data in datasets:

        range_of_t = data[0][0]
        clc_type = data[0][3]
        K_diffs = [d[2] for d in data]

        K_diffs_mean = np.mean(K_diffs, axis=0)
        K_diffs_std = np.std(K_diffs, axis=0) 
        K_ring_data_averaged.append([range_of_t, K_diffs_mean, K_diffs_std])

        plt.plot(range_of_t, K_diffs_mean, label=clc_type)
        plt.fill_between(range_of_t, K_diffs_mean-K_diffs_std*n_std, K_diffs_mean+K_diffs_std*n_std, alpha=0.3)

    plt.xlabel("$t$")
    plt.ylabel("$K_{Ring}(t)$")
    plt.legend(loc="upper right")
    path = os.path.join(results_dest, "K_ring_functions_averaged.pdf")
    plt.savefig(path)

    return K_data_averaged, K_ring_data_averaged
    

    
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# unused functions


def find_nearest_neighbor_function(range_of_t, function1, all_functions):
    """
    From a given dataset all_functions, find the closest neighbor function from all frunction for function1
    with respect to the aprroximated L2 norm. function1 does not count as its own next neigbor.
    """
    next_neighbor = None
    min_distance = np.inf 
    
    # make sure that function1 is not included in list of the other functions
    other_functions = remove_function_from_list(function1, all_functions)
    
    for function2 in other_functions:
        dist = approx_distance_L2(range_of_t, function1, function2)
        if dist < min_distance:
            min_distance = dist
            next_neighbor = function2
            
    return min_distance, next_neighbor



def closest_neighbor_from_same_dataset(range_of_t, function1, data1, data2):
    """
    Returns True, if nearest neighbor of function1 is contained in data1, returns false otherwise.
    """
    min_dist1, _ = find_nearest_neighbor_function(range_of_t, function1, data1)
    min_dist2, _ = find_nearest_neighbor_function(range_of_t, function1, data2)
    return min_dist1 < min_dist2



#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# used functions


def approx_distance_L2(range_of_t, function1, function2):
    """
    Returns the distance between two instances of functional data based on an approximation of the L2 norm.
    """
    sum_ = 0
    
    for i in range(1, len(range_of_t)):
        t0 = range_of_t[i-1]
        t1 = range_of_t[i]
        delta_t = t1-t0
        sum_ += delta_t * (function1[i] - function2[i])**2
        
    return sum_


def function_in_list(function, function_list):
    """
    Checks if a function is an element of a given function list.
    """
    for f in function_list:
        if np.array_equal(f, function):
            return True
    return False


def remove_function_from_list(function, function_list):
    """
    Returns a list containing all elements of function_list, except function.
    """
    new_function_list = []
    
    for f in function_list:
        if not np.array_equal(f, function):
            new_function_list.append(f)
    
    return new_function_list


def get_distance_list(function1, datasets, mode="ring"):
    """
    Returns a list of the (approximated) L2 distances to function 1 for all K functions contained
    in any of the datasets, as well as the information what dataset they are from, e.g.
    
        [[2000, "dataset1"],
         [2200, "dataset3"],
         [1290, "dataset1"],
         ...]  
    """
    dist_list = []
    
    # e.g. datasets = [clca_data, clcb_data]
    for data in datasets:
        # extract data
        # data[i] = [range_of_t, K_diff, K_ring_diff, clc_type, filename, z]
        range_of_t = data[0][0]
        clc_type = data[0][3]

        # K functions to be compared to function1
        if  mode == "ring":
            K_diffs = [d[2] for d in data]
        else:
            K_diffs = [d[1] for d in data]

        # make sure that other functions list does not contain function1 itself
        other_functions = remove_function_from_list(function1, K_diffs)

        for function2 in other_functions:

            dist = approx_distance_L2(range_of_t, function1, function2)
            dist_list.append([dist, clc_type])

    return dist_list


def dist_from_dist_type_pair(dist_type_pair):
    """
    ... used for sorting list.
    """
    dist, type_ = dist_type_pair
    return dist


def get_k_nns(function1, datasets, k, mode='ring'):
    """
    ...
    """
    dist_list = get_distance_list(function1, datasets, mode)
    sorted_dist_list = sorted(dist_list, key=dist_from_dist_type_pair)
    k_nns = sorted_dist_list[0:k]
    return k_nns


def number_k_nns_same_type(function1, type1, k_nns):
    """
    Returns number of k nearest neighbors of function1 are from the same data, i.e. that also have the type of function1, type1.
    """
    # list that contains 1 whenever the rth nearest neighbor matches the type of function1 and 0 otherwise
    truth_values = [1 if type1==x[1] else 0 for x in k_nns]
    sum_ = np.sum(truth_values)
    return sum_


def compute_Schilling_statistic(all_data, datasets, k,  mode="ring"):
    """
    ...
    """
    N = len(all_data)
    sum_ = 0
    
    for range_of_t, K_diff, K_ring_diff, type1, _, _ in all_data:
        
        if mode == "ring":
            function1 = K_ring_diff
        else:
            function1 = K_diff
            
        k_nns = get_k_nns(function1, datasets, k, mode)
        sum_ += number_k_nns_same_type(function1, type1, k_nns)
    
    schilling_stat = sum_ * 1/N * 1/k
    return schilling_stat

