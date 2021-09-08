import os
from time import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# scikit-image
from skimage import io
from tifffile import TiffFile
from skimage import filters
from scipy import signal

from utilities import *
from process_images2D import *
from process_images3D import *



def distance_array1D(dim1, dim2):
    """
    Returns 2-dimensional array where the pixel intensities are equal to the distance from the center
    of the array in one dimension, given by dim1.
    """
    half_dim_1 = int((dim1-1)/2)
    dim_1_range = list(range(1,half_dim_1+1))
    line = dim_1_range[::-1] + [0] + dim_1_range 

    # array where every element is x distance from center
    dist_1_2D = np.array([line, ]*dim2, dtype=np.float32)

    return dist_1_2D


def distance_arr3D(arr, len_x=1, len_y=1, len_z=1, img_name="unnamed", temp_folder="temp", save=False):
    """
    Computes array whose elements are equal to their Euclidean distance from the center.
    
    Args:
        save (bool): if True, store the distance array for future computations
    """
    # if the distance array was already computed previously, it can be loaded from existing .npy file
    # located in temp folder
    dist_array_folder = create_folder("cutting_ellipsoid3D", temp_folder, False)
    file_path = os.path.join(dist_array_folder, img_name+"_distance_array3D.npy")
    if os.path.isfile(file_path):
        print("Distance array already computed, loading .npy file...")
        dist_3D = np.load(file_path).astype(np.float32)
        return dist_3D
    elif save:
        print("No .npy file for distance array found, computing and saving distance array...")
    
    dimz, dimy, dimx = arr.shape

    # array should have equal dimensions, dimension should be odd
    if is_even(dimx) or is_even(dimy) or is_even(dimz):
        print("Error! There should be an odd pixel number in all dimensions!")
        return
        
    # array where every element is x-distance from center
    dist_x_2D = distance_array1D(dimx, dimy)
    dist_x = np.tile(dist_x_2D, (dimz, 1, 1)).astype(np.float32)
    # array where every element is y-distance from center
    dist_y_2D = distance_array1D(dimy, dimx).T
    dist_y = np.tile(dist_y_2D, (dimz, 1, 1)).astype(np.float32)
    
    half_dim_z = int(dimz/2) 
    z_range = list(range(1,half_dim_z+1))
    line_z = z_range[::-1] + [0] + z_range 
    
    # array where every element is z distance from center
    z_arrays = [np.zeros((dimy,dimx), dtype=np.float32)+z_dist for z_dist in line_z]    
    dist_z = np.stack(z_arrays, axis=0).astype(np.float32)
    
    # compute array whose elements are equal to their Euclidean distance from the center
    dist_3D = np.sqrt((dist_x*len_x)**2 + (dist_y*len_y)**2 + (dist_z*len_z)**2).astype(np.float32)

    if not (dist_3D.shape == (dimz, dimy, dimx)):
        print("Error! 3D distance array has wrong shape!")
        return
    
    # save array to be reused in other computations
    if save:
        np.save(file_path, dist_3D)
    
    return dist_3D


def cut_ellipsoid(arr, dist_array, radius):
    """
    Takes a 3D array and sets all elements outside an ellipsoid (around center of image) to zero. 
    """
    #mask = np.ones(arr.shape)
    #mask[dist_array>=radius] = 0
    mask = (dist_array<radius).astype(np.float32)
    return arr.astype(np.float32) * mask


def autocorrelation_3D(img_arr, img_name="unnamed", temp_folder="temp", save=False):
    """
    Computes the autocorrelation of a 3D array. Uses FFT for speed-up.
    
    Args:
        save (bool): if True, store the autocorrelation array for future computations
    """
    # if the distance array was already computed previously, it can be loaded from existing .npy file
    # located in temp folder 
    
    autocorr_folder = create_folder("autocorrelation3D", temp_folder, False)
    file_path = os.path.join(autocorr_folder, img_name+"_autocorrelation3D.npy")
    if os.path.isfile(file_path):
        print("Autocorrelation array already computed, loading .npy file...")
        auto_corr = np.load(file_path).astype(np.float32)
        return auto_corr
    elif save:
        print("No .npy file for autocorrelation array found, computing and saving autocorrelation array...")
     
    auto_corr = signal.correlate(img_arr.astype(np.float32), img_arr.astype(np.float32), method='fft')
    if save:
        np.save(file_path, auto_corr)
    
    return auto_corr



def ripleys_K_3D(img_array, range_of_t, width=1, len_x=1, len_y=1, len_z=1, img_name="unnamed", temp_folder="temp", save=False, printout=True):
    """
    Computes two variants of the pixel-based Ripley's K-function (for a given 3D image):
      1.) For each t, sum up values of the autocorrelation function that lie WITHIN a circle of radius t.
          --> "K-function"
      2.) For each t, sum up values of the autocorrelation function that lie ON a circle of radius t with specified width.
          --> "Ring K-function"
        
    Args:
        len_x, len_y, len_z (float): dimension of each voxel in the autocorrelation function
        save (bool): if True, store the autocorrelation array for future computations
    """
    K_values = []
    K_ring_values = []
    
    if printout:
        print("Computing K function...")
        
    t0=time()
    
    # cut away all slices of the array that only contain zeros in order to speed up
    # the computation time for the autocorrelation
    img_trimmed = cut_away_zeros3D(img_array.astype(np.float32))
    
    if printout:
        print("Computing 3D autocorrelation...")
        print("Image dimensions:", img_trimmed.shape)
    
    # 3D autocorrelation function
    auto_corr_full = autocorrelation_3D(img_trimmed, img_name, temp_folder=temp_folder, save=save)
    if printout:
        print("Computing distance array...")
    
    # 3D array where every element is equal to the distance to the array's centre
    dist_array = distance_arr3D(auto_corr_full, len_x=len_x, len_y=len_y, len_z=len_z, img_name=img_name, temp_folder=temp_folder, save=False)
    
    if printout:
        print("Summing up autocorrelation values...")
    
    # compute K-function value for every radius t
    count=0
    for t in range_of_t:
        count+=1
        prog = count/len(range_of_t)*100
        if prog<100:
            print(f"Progress: {prog:.2f}%, t={t}", end="\r")
        else:
            print(f"Progress: {prog:.2f}%, t={t}")
        
        # sum up values of autocorrelation function that lie within an ellipsoid, whose dimensions are given by radius=t 
        # as well as len_x, len_y and len_z
        auto_corr_t_inner = np.sum(cut_ellipsoid(auto_corr_full, dist_array=dist_array, radius=t))
        auto_corr_t_outer = np.sum(cut_ellipsoid(auto_corr_full, dist_array=dist_array, radius=t+width))
        
        # compute both variants of the K-funtion:
        # 1) cut out ellipsoids with specified width and radius out of autocorrelation function and sum the values
        sum_ring = auto_corr_t_outer-auto_corr_t_inner
        # 2) cut out disks with specified radius out of autocorrelation function and sum the values
        sum_ = auto_corr_t_inner
    
        # TODO: add normalization factor
        K = sum_
        K_ring = sum_ring
    
        K_values.append(K)
        K_ring_values.append(K_ring)
    
    t1=time()
    diff=t1-t0
    if printout:
        print(f"Completed in {diff:.2f} seconds")

    return K_values, K_ring_values


def compute_K_diff_3D(img_real, img_csr, range_of_t, width=1, len_x=1, len_y=1, len_z=1, img_name="unnamed", temp_folder="temp", save=False, printout=True):
    """
    Computes the difference of the K-functions (using two different variants of the K-function) of an image
    and its corresponding CSR image (image showing the same shape, but with a uniform distribution of pixel intensities). 
    """
    # remove all slices consisting of only zeros from the 3D arrays
    # to decrease the computation time
    img_real_trimmed = cut_away_zeros3D(img_real)
    img_csr_trimmed = cut_away_zeros3D(img_csr)

    if printout:
        print("Computing 3D K-functions for real image...")
    K_values_real, K_ring_values_real = ripleys_K_3D(img_real.astype(np.float32), range_of_t, width, len_x, len_y, len_z, "real_"+img_name, temp_folder, save, printout)

    if printout:
        print("Computing 3D K-functions for CSR image...")
    K_values_csr, K_ring_values_csr = ripleys_K_3D(img_csr.astype(np.float32), range_of_t, width, len_x, len_y, len_z, "csr_"+img_name, temp_folder, save, printout)
        
    K_diff = np.array(K_values_real) - np.array(K_values_csr)
    K_ring_diff = np.array(K_ring_values_real) - np.array(K_ring_values_csr)
        
    return K_diff, K_ring_diff



def compute_all_K_functions3D(dataset, data_type, params, temp_folder="temp", check_plot=False, save=True, log_dest=None):
    """
    For all 3D images included in datasets, apply preprocessing (see process_image3D) and compute the K-functions.
    
    Args:
        data_type (str): specifies what sample the given data is from
        dataset (list): contains information on all the images to be processed, the path and the channel to be used. 
        params (list): contains parameters for preprocessing and K-function computation   
    """
    K_function_data = []
    desired_int, mask_params, rm_background_params, K_func_params = params
    
    indices = range(len(dataset))
    
    count=0
    for i in indices:
        count+=1
        filepath, channel, _ = dataset[i]
        img_base_name = get_filename_from_path(filepath)
        img_name = f"{img_base_name}_channel{channel}"
        
        if not log_dest is None:
            write_file_in_log(img_name, log_dest)
        
        print(f"[{data_type} {count}/{len(indices)}] Processing {img_name}...")
        
        img_array, metadata = load_image3D(filepath, channel=channel)
        
        #---------------------------------------------------------------
        # 1. Preprocessing
        print("Preprocessing 3D image...")
        img_real, img_csr = process_image3D(img_array, desired_int, mask_params, rm_background_params, img_name, temp_folder, check_plot, save)
        
        #---------------------------------------------------------------
        # 2. Computing K-functions and their difference (real-CSR)
        print("Computing K functions...")
        K_diff, K_ring_diff = compute_K_diff_3D(img_real, img_csr, *K_func_params, img_name, temp_folder, save, True)
        
        # store information for later, e.g. plotting (same format as for 2D)
        z = None
        range_of_t = K_func_params[0]
        K_function_data.append([range_of_t, K_diff, K_ring_diff, data_type, img_base_name, z])
        
    return K_function_data