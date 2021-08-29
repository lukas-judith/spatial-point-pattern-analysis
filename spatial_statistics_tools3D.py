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



def distance_arr3D(arr, len_x=1, len_y=1, len_z=1, temp_folder="temp", save=False):
    """
    Computes array whose elements are equal to their Euclidean distance from the center.
    """
    # if the distance array was already computed previously, it can be loaded from existing .npy file
    # located in temp folder 
    if not temp_folder in os.listdir("."):
        os.mkdir(temp_folder)
    file_path = os.path.join(temp_folder, "distance_array3D.npy")
    if os.path.isfile(file_path):
        dist_3D = np.load(file_path)
        return dist_3D
    elif save:
        print("No .npy file for distance array found, computing and saving distance array...")
    
    dimz, dimx, dimy = arr.shape
    # array should have equal dimensions, dimension should be odd
    if (dimx != dimy) or is_even(dimx) or is_even(dimz):
        print("Error! Input array has wrong shape! x- and y-dimension need to be identical and there should be an odd pixel number in all dimensions!")
        return
        
    half_dim = int((dimx-1)/2)
    x_range = list(range(1,half_dim+1))
    line = x_range[::-1] + [0] + x_range 

    # array where every element is x distance from center
    dist_x_2D = np.array([line, ]*dimx)
    dist_x = np.tile(dist_x_2D, (dimz, 1, 1))
    
    # array where every element is y distance from center
    dist_y_2D = dist_x_2D.T
    dist_y = np.tile(dist_y_2D, (dimz, 1, 1))
    
    half_dim_z = int(dimz/2) 
    z_range = list(range(1,half_dim_z+1))
    line_z = z_range[::-1] + [0] + z_range 
    
    # array where every element is z distance from center
    z_arrays = [np.zeros((dimx,dimy))+z_dist for z_dist in line_z]    
    dist_z = np.stack(z_arrays, axis=0)
    
    # compute array whose elements are equal to their Euclidean distance from the center
    dist_3D = np.sqrt((dist_x/len_x)**2 + (dist_y/len_y)**2 + (dist_z/len_z)**2)

    if not (dist_3D.shape == (dimz, dimx, dimx)):
        print("Error! 3D distance array has wrong shape!")
        return
    
    # save array to be reused in other computations
    np.save(file_path, dist_3D)
    
    return dist_3D


def cut_ellipsoid(arr, dist_array, radius):
    """
    Takes a 3D array and sets all elements outside an ellipsoid (around center of image) to zero. 
    """
    mask = np.ones(arr.shape)
    mask[dist_array>=radius] = 0
    return arr * mask


def autocorrelation_3D(img_arr, img_name, temp_folder="temp", save=False):
    """
    ...
    save should be disabled if not enough storage space is available
    """
    # if the distance array was already computed previously, it can be loaded from existing .npy file
    # located in temp folder 
    
    autocorr_folder = create_folder("autocorrelation3D", temp_folder, False)
    file_path = os.path.join(autocorr_folder, img_name+"_autocorrelation3D.npy")
    if os.path.isfile(file_path):
        auto_corr = np.load(file_path)
        return auto_corr
    elif save:
        print("No .npy file for autocorrelation array found, computing and saving autocorrelation array...")
     
    auto_corr = signal.correlate(img_arr, img_arr, method='fft')
    if save:
        np.save(file_path, auto_corr)
    
    return auto_corr



def ripleys_K_3D(img_arr, img_name, mask, range_of_t, width, printout=False):
    pass

