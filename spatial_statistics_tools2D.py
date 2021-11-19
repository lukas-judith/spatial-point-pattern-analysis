import os
from time import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.fft import fft, ifft
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

# scikit-image
from skimage import io
from tifffile import TiffFile
from skimage import filters
from scipy import signal

from utilities import *
from process_images2D import *


def find_pixels_in_cell(img, threshold):
    """
    Returns positions of pixels above a certain threshold in intensity.
    """
    pixel_positions = []
    
    # loop over the coordinates of all pixels
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
        
            position = (x,y)
            intensity = img[position] 
            
            if intensity > threshold:
                pixel_positions.append(position) 
                
    return pixel_positions


def K_values_mean_std(img, n_K, range_of_t, mask=None):
    """
    Computes the K values (for given range of t) for n_K images containing random noise.
    Returns mean and std of the K values (arrays with same length as range_of_t).
    """
    K_values = []

    for i in range(n_K):

        img_csr = create_csr_img(img, mask)
        K = ripleys_K_fast(img_csr, range_of_t, printout=False)
        K_values.append(K)

    K_values = np.array(K_values)
    
    # for large-enough n_K, mean_K will have a Gaussian distribution
    # according to the Central Limit Theorem
    mean_K = np.mean(K_values, axis=0)
    std_K = np.std(K_values, axis=0)
    
    return mean_K, std_K


def ripleys_K_slow(pixel_positions, img_arr, t, _lambda):
    """
    Returns Ripley's K based on pixel intensity using for-loops.
    """
    intensity_array = img_arr.astype('float32')
    
    K = 0
    N = len(pixel_positions)
    
    count = 0
    t1 = time()
    
    # here, i and j are pixels
    for i in pixel_positions:
        for j in pixel_positions:
            
            # intensity of pixels i and j in pixel_positions
            x_i = intensity_array[i] 
            x_j = intensity_array[j] 
            
            if d(j, i) < t:
                K += x_i * x_j 
            
            # to keep track of speed
            count += 1
            t2 = time()
            percent_done = count/N**2 * 100
            print(f"t={t}: {percent_done:.5f} percent done in {(t2-t1):.2f} seconds", end="\r")
        
    K = K / _lambda * 1/N
    return K


def ripleys_K_semi_vectorized(pixel_positions, img_arr, t, _lambda):
    """
    Returns Ripley's K based on pixel intensity, semi-vectorized.
    """
    intensity_array = img_arr.astype('float32')

    K = 0
    N = len(pixel_positions)
    
    count = 0
    t1 = time()
    
    # turn list of tuples into array
    point_arr = np.array(pixel_positions)
        
    # loop over all pixels i
    for i in pixel_positions: 

        # compute distances to all other pixels j
        distances = np.sqrt(np.sum(np.square(i - point_arr), axis=1))

        # only use points with distances within radius t
        in_circle = point_arr[(distances<=t) * (distances>0)]
        
        # array of all intensities x_j within circle of radius t
        # -> take intensity for every point in "in_circle"
        intens_circle = np.array([intensity_array[tuple(in_circle[i])] for i in range(len(in_circle))])

        # intensity x_i of point i
        intens_point = intensity_array[tuple(i)]
        
        # sum of {x_i * x_j} over j for d(i,j)<=t; x_k is intensity of pixel k 
        K += np.sum(intens_circle * intens_point)    
        
        # to keep track of speed
        count += 1
        t2 = time()
        percent_done = count/N * 100
        print(f"t={t}: {len(in_circle)} points in circle, {percent_done:.2f} percent done in {(t2-t1):.2f} seconds", end="\r")
    print("", end="\r")
      
    K = K / _lambda * 1/N
    return K



#--------------------------------------
# using auto-correlation in real space
#--------------------------------------


def create_t_mask(t):
    """
    Computes mask to sum up all values in circle of t.
    """
        
    d = 2*t + 1
    
    mask = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            dist = np.sqrt((i-t)**2 + (j-t)**2)
            if (dist<=t):
                mask[i,j] = 1 

    mask[t,t] = 0
    
    return mask


def ripleys_K_autocorr(img_arr, t, lambda_):
    """
    Computes Ripley's K function by convoluting the image by a mask 
    that sums up pixel intensities in ball of radius t.
    """
    
    arr = img_arr.astype('float32')
    N = (arr>0).sum()
    
    t1 = time()
    
    mask = create_t_mask(t)
    new_arr = signal.correlate2d(arr, mask)[t:-t, t:-t]
    
    t2 = time()    
    
    print(f"t={t}: done in {(t2-t1):.2f} seconds")

    sum_ = np.sum(new_arr * arr)
    
    K = sum_ * 1/lambda_ * 1/N 
    
    return K


#-----------------------------------------
# using auto-correlation in Fourier space
#-----------------------------------------


def is_even(number):
    """
    Checks if integer number is even.
    """
    if int(number)!=number: 
        print("Error! Input can only be of type int!")
        return
    return int(number/2) == number/2


def distance_arr(arr, len_x=1, len_y=1):
    """
    Computes array whose elements are equal to their Euclidean distance from the center.
    """
    dimy, dimx = arr.shape
    # array should have equal dimensions, dimension should be odd
    if (dimx != dimy) or is_even(dimx):
        print("Error! Array has wrong shape!")
        return
        
    half_dim = int((dimx-1)/2)
    x_range = list(range(1,half_dim+1))
    line = x_range[::-1] + [0] + x_range 

    # array where every element is x distance from center
    diff_x = np.array([line, ]*dimx)
        
    # array where every element is y distance from center
    diff_y = diff_x.T
        
    # compute array whose elements are equal to their Euclidean distance from the center
    diff_xy = np.sqrt((diff_x*len_x)**2 + (diff_y*len_y)**2)
    
    return diff_xy
    

def cut_circle(arr, radius, diff_xy):
    """
    Takes a 2D array and sets all elements outside a circle (around center of image) with specified radius to zero. 
    """
    mask = np.ones(arr.shape)
    mask[diff_xy>=radius] = 0
    return arr * mask


def auto_correlation_fft(img):
    """
    Fast computation of auto-correlation using FFT.
    """
    dimx, dimy = img.shape
    # zero-padding of the input image
    img_for_fft = np.pad(img, (int(dimx/2),int(dimx/2)), 'constant', constant_values=(0, 0))

    # transform into Fourier space using fft
    # => auto-correlation is ifft of product of transformed image multiplied by its conjugate
    ft = fft2(img_for_fft)
    cc = np.real(ifft2(ft * ft.conj()))
    cc_image = fftshift(cc)
    corr_fft = cc_image.real[1:,1:]
    return corr_fft


def test_auto_correlation_fft():
    """
    Compares fft-based auto-corr. with auto-corr. in real space.
    """
    # compute arbitrary test image
    test_img = np.ones((20,20))
    test_img[10:15, 10:14] = 4
    test_img[11:12, 11:13] = 10
    
    corr = signal.correlate(test_img, test_img)
    corr_fft = auto_correlation_fft(test_img)
    
    diff = corr - corr_fft
    
    # we require equivalence except for small numerical errors
    diff[diff<0.001] = 0
    
    assert np.unique(diff).item() == 0
    

# older implementation of the K-function, see ripleys_K_fast_ring for newer implementation
def ripleys_K_fast(img_arr, mask, range_of_t, printout=False):
    """
    Computes Ripley's K function for a range of t. Utilizes FFT for fast computation of auto-correlation of the image.
    """
    K_values = []
    
    # assure datatype that does not cause errors
    arr = img_arr.astype('float32')
    # number of pixels in desired area
    N = np.sum(mask)
    # sum of all pixel intensities
    total_int = np.sum(arr)
    # here, assume A_pixel = 1, thus A=N
    lambda_ = total_int / N

    t1 = time()
    
    # full array for the auto-correlation of the input image
    # own implementation:
    #full_auto_corr = auto_correlation_fft(arr)
    # library function:
    full_auto_corr = signal.correlate(arr, arr, method='fft')
    
    # array whose elements are equal to their Euclidean distance from the center
    diff_xy = distance_arr(full_auto_corr)

    for t in range_of_t:
        # array containing the auto-correlation up to distance t
        auto_corr_t = cut_circle(full_auto_corr, radius=t, diff_xy=diff_xy)
        
        # sum over all elements
        sum_ = np.sum(auto_corr_t)
        
        K = sum_ * 1/lambda_ * 1/N
        K_values.append(K)
    
    t2 = time()
    if printout:
        print(f"Completed in {(t2-t1):.2f} seconds")
    
    return K_values


def ripleys_K_fast_ring(img_arr, mask, range_of_t, width, len_x=1, len_y=1, printout=False):
    """
    Computes two variants of the pixel-based Ripley's K-function (for a given 2D image):
      1.) For each t, sum up values of the autocorrelation function that lie WITHIN a circle of radius t.
          --> "K-function"
      2.) For each t, sum up values of the autocorrelation function that lie ON a circle of radius t with specified width.
          --> "Ring K-function"
        
    Args:
        len_x, len_y (float): dimension of each pixel in the autocorrelation function
        save (bool): if True, store the autocorrelation array for future computations
    """
    K_values = []
    K_values_ring = []
    
    # assure datatype that does not cause errors
    arr = img_arr.astype('float32')
    # number of pixels in desired area
    N = np.sum(mask)
    # sum of all pixel intensities
    total_int = np.sum(arr)
    # here, assume A_pixel = 1, thus A=N
    lambda_ = total_int / N
    
    t1 = time()
    
    # full array for the auto-correlation of the input image
    # own implementation:
    #full_auto_corr = auto_correlation_fft(arr)
    # library function:
    full_auto_corr = signal.correlate(arr, arr, method='fft')
    
    # array whose elements are equal to their Euclidean distance from the center
    diff_xy = distance_arr(full_auto_corr, len_x, len_y)

    for t in range_of_t:

        # arrays containing the spatial auto-correlation up to distance t (+width)
        auto_corr_t_inner = cut_circle(full_auto_corr, radius=t, diff_xy=diff_xy)
        auto_corr_t_outer = cut_circle(full_auto_corr, radius=t+width, diff_xy=diff_xy)
        
        # compute both variants of the K-funtion:
        # 1) cut out circles with specified width and radius out of autocorrelation function and sum the values
        sum_ring = np.sum(auto_corr_t_outer-auto_corr_t_inner)
        # 2) cut out disks with specified radius out of autocorrelation function and sum the values
        sum_ = np.sum(auto_corr_t_inner)
        
        K = sum_ * 1/lambda_ * 1/N #* 1/(2*np.pi*(t+width))
        K_ring = sum_ring * 1/lambda_ * 1/N #* 1/(2*np.pi*(t+width))
        
        K_values.append(K)
        K_values_ring.append(K_ring)
    
    t2 = time()
    if printout:
        print(f"Completed in {(t2-t1):.2f} seconds")
    
    return K_values, K_values_ring, full_auto_corr



def ripleys_K_fast_normalized(img_arr, mask, range_of_t, width, len_x=1, len_y=1, printout=False):
    """
    SAME AS ABOVE WITH NORMALIZATION
    """
    K_values = []
    K_values_ring = []
    
    # assure datatype that does not cause errors
    arr = img_arr.astype('float32')
    # number of pixels in desired area
    N = np.sum(mask)
    # sum of all pixel intensities
    total_int = np.sum(arr)
    # here, assume A_pixel = 1, thus A=N
    lambda_ = total_int / N
    
    t1 = time()
    
    # full array for the auto-correlation of the input image
    # own implementation:
    #full_auto_corr = auto_correlation_fft(arr)
    # library function:
    full_auto_corr = signal.correlate(arr, arr, method='fft')
    
    # array whose elements are equal to their Euclidean distance from the center
    diff_xy = distance_arr(full_auto_corr, len_x, len_y)

    for t in range_of_t:

        # arrays containing the spatial auto-correlation up to distance t (+width)
        auto_corr_t_inner = cut_circle(full_auto_corr, radius=t, diff_xy=diff_xy)
        auto_corr_t_outer = cut_circle(full_auto_corr, radius=t+width, diff_xy=diff_xy)
        
        # compute both variants of the K-funtion:
        # 1) cut out circles with specified width and radius out of autocorrelation function and sum the values
        sum_ring = np.sum(auto_corr_t_outer-auto_corr_t_inner)
        pixels_in_ring = ((auto_corr_t_outer-auto_corr_t_inner) > 0)
        norm = np.sum(pixels_in_ring)
        
        # 2) cut out disks with specified radius out of autocorrelation function and sum the values
        sum_ = np.sum(auto_corr_t_inner)
        pixels_in_ring = ((auto_corr_t_inner) > 0)
        norm2 = np.sum(pixels_in_ring)
        
        K = sum_ * 1/lambda_ * 1/norm #* 1/(2*np.pi*(t+width))
        K_ring = sum_ring * 1/lambda_ * 1/norm2 #* 1/(2*np.pi*(t+width))
        
        K_values.append(K)
        K_values_ring.append(K_ring)
    
    t2 = time()
    if printout:
        print(f"Completed in {(t2-t1):.2f} seconds")
    
    return K_values, K_values_ring, full_auto_corr




def get_K_diff(mask, img_real, img_csr, range_of_t, width, len_x=1, len_y=1, printout=True):
    """
    Computes the difference of the K-functions (using two different variants of the K-function) of an image
    and its corresponding CSR image (image showing the same shape, but with a uniform distribution of pixel intensities). 
    """
    K_values_real, K_values_ring_real, corr_real = ripleys_K_fast_ring(img_real, mask, range_of_t, width, len_x, len_y, printout)
    K_values_csr, K_values_ring_csr, corr_csr = ripleys_K_fast_ring(img_csr, mask, range_of_t, width, len_x, len_y, printout)
    K_diff = np.array(K_values_real) - np.array(K_values_csr)
    K_ring_diff = np.array(K_values_ring_real) - np.array(K_values_ring_csr)
    return K_diff, K_ring_diff, corr_real, corr_csr


def compute_all_K_functions2D(dataset, data_type, params, results_dest=".", check_plot=False, unit=None):
    """
    For all 2D images included in datasets, apply preprocessing (see process_image3D) and compute the K-functions.
    
    Args:
        data_type (str): specifies what sample the given data is from
        dataset (list): contains information on all the images to be processed, the path and the channel to be used. 
        params (list): contains parameters for preprocessing    
    """
    K_function_data = []
    desired_int, mask_params, rm_background_params, K_func_params = params
    
    indices = range(len(dataset))
    
    count=0
    for i in indices:
        count+=1
        filepath, channel, z = dataset[i]
        filename = os.path.basename(filepath)
        print(f"[{data_type} {count}/{len(indices)}] Processing {filename}...")
        
        # create subfolder to store plots produced while processing the image
        # (only important for check_plot=True)
        name = f"{data_type}_check_plots_{filename[:-4]}"
        sub_results_dest = create_folder(name, os.path.join(results_dest,"check_plots"))

        img_array, metadata = load_image2D(filepath, z, channel=channel)
        
        # 1. Preprocessing
        print("Preprocessing image...")
        img_real, img_csr, cell_mask = process_image2D(img_array, desired_int, mask_params, rm_background_params, sub_results_dest, check_plot)

        # 2. K-functions
        print("Computing K functions...")
        K_diff, K_ring_diff, corr_real, corr_csr = get_K_diff(cell_mask, img_real, img_csr, *K_func_params, True)
        
        if check_plot:
            range_of_t = K_func_params[0]
            width = K_func_params[1]
            
            # works only if number odd, should be given for autocorrelation
            len_ = corr_real.shape[0]
            lim_lower = -int(len_/2)
            lim_upper = int(len_/2)

            fig, ax = plt.subplots(2, 2, figsize=(12, 10))

            ax[0][0].imshow(img_real)
            im1 = ax[0][0].imshow(img_real)
            ax[0][0].set_title("Original (\"real\") image")
            ax[0][1].set_title("(autocorr. real) - (autocor. CSR); negative values in white")
            im2 = ax[0][1].matshow((corr_real-corr_csr), norm=LogNorm(), extent=[lim_lower, lim_upper, lim_lower, lim_upper], origin='lower')    
            ax[1][0].set_title("Autocorr. real image")

            ax[1][0].set_title("K function for (real)-(CSR)")
            ax[1][0].plot(range_of_t, K_diff)
            if not unit is None:
                ax[1][0].set_xlabel(f"$t$ ({unit})")
            else: 
                ax[1][0].set_xlabel("$t$")
                print("No unit for plots specified")
            ax[1][0].set_ylabel("$K(t)$")

            ax[1][1].set_title(f"Ring K function for (real)-(CSR), width={width}")
            ax[1][1].plot(range_of_t, K_ring_diff)
            if not unit is None:
                ax[1][1].set_xlabel(f"$t$ ({unit})")
            else: 
                ax[1][1].set_xlabel("$t$")
                print("No unit for plots specified")
            ax[1][1].set_ylabel("$K(t)$")
    
            plt.colorbar(im1, ax=ax[0, 0])
            plt.colorbar(im2, ax=ax[0, 1])

            #fig.suptitle(title)
            
            k_func_dest = os.path.join(sub_results_dest, "K_function_computation.pdf")
            plt.savefig(k_func_dest)
            plt.close()

        # store information for later, e.g. plotting
        K_function_data.append([range_of_t, K_diff, K_ring_diff, data_type, filename, z])
        
    return K_function_data


def plot_K_functions(data, result_dest, mode="disk", full_legend=False, unit=None):
    """
    Creates plot showing all K-functions contained in data.
    """
    plt.figure(figsize=(10,8))
    plt.title(f"K functions, o = clca, x = clcb")

    for range_of_t, K_disk_diff, K_ring_diff, data_type, filename, _ in data:

        if data_type == "clca":
            marker = "o"
        elif data_type == "clcb":
            marker = "x"
            
        if mode=="disk":
            K_diff = K_disk_diff
            name = "K_functions_comparison"
        elif mode=="ring":
            K_diff = K_ring_diff
            name = "K_ring_functions_comparison"

        if full_legend:
            filename_ = get_filename_from_path(filename)
            plt.plot(range_of_t, K_diff, marker=marker, linestyle="dashed", label=filename_[-14:])
        else:
            plt.plot(range_of_t, K_diff, marker=marker, linestyle="dashed")
    
    if not unit is None:
        plt.xlabel(f"$t$ ({unit})")
    else: 
        plt.xlabel("$t$")
        print("No unit for plots specified")
    plt.ylabel("$K(t)$")
    
    if full_legend:
        k_func_dest = os.path.join(result_dest, name+"_full_legend.pdf")
        plt.legend(loc='upper right')
        plt.savefig(k_func_dest, bbox_inches='tight')
    else:
        plt.plot([], color='black', marker="o", linestyle="dashed", label="Distr. CLCA - CSR")
        plt.plot([], color='black', marker="x", linestyle="dashed", label="Distr. CLCB - CSR")
        plt.legend(loc='upper right')
        k_func_dest = os.path.join(result_dest, name+".pdf")
        plt.savefig(k_func_dest)
    plt.close()
    
    
def average_K_diffs(datasets, results_dest=".", n_std=1, unit=None):
    """
    From samples of K functions, compute and plot the averages and standard deviations.
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

    if not unit is None:
        plt.xlabel(f"$t$ ({unit})")
    else: 
        plt.xlabel("$t$")
        print("No unit for plots specified")
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

    if not unit is None:
        plt.xlabel(f"$t$ ({unit})")
    else: 
        plt.xlabel("$t$")
        print("No unit for plots specified")
    plt.ylabel("$K_{Ring}(t)$")
    plt.legend(loc="upper right")
    path = os.path.join(results_dest, "K_ring_functions_averaged.pdf")
    plt.savefig(path)

    return K_data_averaged, K_ring_data_averaged