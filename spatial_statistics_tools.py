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


def load_image(filename, folder="CLC_Distribution_TIFFs"):
    """
    Returns image and metadata from filepath.
    """
    
    path = os.path.join(folder, filename)

    meta_data = {}
    
    with TiffFile(path) as tif:
        # WARNING: metadata only for first page
        for info in tif.pages[0].tags.values():
            key, value = info.name, info.value
            # toss unneccessary metadata
           # if not key in ["IJMetadataByteCounts", "IJMetadata"]:
            meta_data[key] = value
        im = tif.asarray()

    # print metadata
    #for k in meta_data.keys():
    #    print(k, ":", meta_data[k])
    print("Image shape:", im.shape)
    
    return im, meta_data


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


def ripleys_K(pixel_positions, img_arr, t, _lambda):
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


def distance_arr(arr):
    """
    Computes array whose elements are equal to their Euclidean distance from the center.
    """
    dimx, dimy = arr.shape
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
    diff_xy = np.sqrt(diff_x**2 + diff_y**2)
    
    return diff_xy
    

def cut_circle(arr, radius, diff_xy):
    """
    Takes a 2D array and sets all elements outside a circle (around center of image) with specified radius to zero. 
    """
    mask = np.ones(arr.shape)
    mask[diff_xy>radius] = 0
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
    
    
def ripleys_K_fast(img_arr, range_of_t):
    """
    Computes Ripley's K function for a range of t. Utilizes FFT for fast computation of auto-correlation of the image.
    """
    K_values = []
    
    # assure datatype that does not cause errors
    arr = img_arr.astype('float32')
    # number of pixels in desired area
    N = (arr>0).sum()
    # sum of all pixel intensities
    total_int = np.sum(arr)
    # here, assume A_pixel = 1, thus A=N
    lambda_ = total_int / N


    t1 = time()
    
    # full array for the auto-correlation of the input image
    # own implementation:
    full_auto_corr = auto_correlation_fft(arr)
    # library function:
    #full_auto_corr = signal.correlate(arr,arr, method='fft')
    
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
    print(f"Completed in {(t2-t1):.2f} seconds")
    
    return K_values
