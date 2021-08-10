from spatial_statistics_tools import *
from utilities import *
from skimage import feature, exposure, restoration
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from skimage.filters import gaussian

from matplotlib.colors import LogNorm


def load_image(path):
    """
    Returns image and metadata from filepath.
    """
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
    #print("Image shape:", im.shape)
    return im, meta_data


def scale_image(img, desired_int=2000000000):
    """
    Scales the input image to have a specified total intensity.
    """
    total_int = np.sum(img)
    img_scaled = img * desired_int/total_int
    return img_scaled


def crop_image(img, mask):
    """
    Multiplies mask with image.
    """
    return img * mask


def normalize_image(img):
    """
    Maps image intensities to interval [0,1].
    """
    img_norm = (img-np.min(img))/(np.max(img)-np.min(img))
    return img_norm 


def load_image2D(path, z, channel=0):
    """
    Load 3D cell image from .tif file and extract a desired z slice.
    Returns 2D image array and metadata.
    """
    # get array with all pixel intensities and metadata of image
    im, metadata = load_image(path) 

    # pick desired channel
    zxy_arr = im[:, channel, :, :]
    
    # extract 2D slice
    xy_array = zxy_arr[z, :, :].astype("float32")
    
    return xy_array, metadata


def remove_background(img_array, rolling_ball_radius, folder=".", check_plot=False):
    """
    Removes background using rolling ball algorithm.
    """
    kernel = restoration.ball_kernel(rolling_ball_radius, ndim=2)
    background = restoration.rolling_ball(img_array, kernel=kernel)
    signal = img_array - background
    
    #---------------------------------------------------------------
    # create plots for checking if the method is working correctly
    #---------------------------------------------------------------
    if check_plot:
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].set_title("Original image")
        ax[0].imshow(img_array)
        ax[1].set_title("Signal")
        ax[1].imshow(signal)
        ax[2].set_title("Background")
        ax[2].imshow(background)
        
        name = "check_background_removal.pdf"
        dest = os.path.join(folder, name)
        plt.savefig(dest)
        plt.close()

    return signal 
    
    
def create_mask(img_array, sigma=20, iter_erode=35, iter_dilate=20, folder=".", check_plot=False):
    """
    Creates a mask capturing the silhoutte of the cell on the input image.
    """
    # map image intensities to the interval [0,1]
    img_normalized = normalize_image(img_array)

    # blur image and map resulting intensities to the interval [0,1]
    img_blur = img_normalized.copy()
    img_blur = gaussian(img_blur, sigma=sigma)
    img_blur_norm = normalize_image(img_blur)

    # create histogram of blurred image intensities in order to find optimal cut-off
    # for extracting the cell shape from the image
    counts, bin_edges = np.histogram(img_blur_norm.ravel(), bins=200)
    bin_middles = np.diff(bin_edges)
    bin_middles = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

    # find first local minimum in counts of the histogram, this should separate the peak 
    # from the background intensities and the peak from the intensities within the cell area

    # smooth values to facilitate finding the local minimum
    arr = gaussian(counts, sigma=2)

    # attempt to find first minimum by checking for the first point in the plot
    # at which the count value start increasing after the initial decrease
    # TODO: replace with more sophisticated/robust method
    y_old = arr[0]
    for i in range(1,len(arr)):
        y_new = arr[i]
        if y_new > y_old:
            x = i-1
            break
        y_old = y_new

    # optimal cut-off value above which we consider the pixels to be within the cell
    cut = bin_middles[x]
    mask = img_blur_norm>cut

    # some additional processing:
    # fill holes, erode artifacts from Gaussian blur
    # erode more than necessary, then dilate again to remove smaller objects
    mask = binary_dilation(binary_erosion(binary_fill_holes(mask), iterations=iter_erode), iterations=iter_dilate)
    
    #---------------------------------------------------------------
    # create plots for checking if the method is working correctly
    #---------------------------------------------------------------
    
    if check_plot:
        fig, ax = plt.subplots(3, 2, figsize=(10,15))
        ax[0][0].set_title("Original image")
        ax[0][0].imshow(img_array)
        ax[0][1].set_title("Blurred image")
        ax[0][1].imshow(img_blur)
        ax[1][0].set_title("Smoothed values of histogram counts")
        ax[1][0].plot(bin_middles, counts)
        ax[1][0].scatter([bin_middles[x]], [counts[x]], color='r', marker='x', label="Cut-off value")
        ax[1][0].set_ylabel("Counts")
        ax[1][0].set_xlabel("Pixel intensity")
        ax[1][0].set_yscale('log')
        ax[1][0].legend()
        ax[1][1].set_title("Mask = blurred image after cut-off")
        ax[1][1].imshow(mask)
        ax[2][0].set_title("Overlay of orig. image with mask")
        ax[2][0].imshow(mask-img_normalized*4)
        ax[2][1].set_title("Compare: orig. image, histogram equalized")
        ax[2][1].imshow(exposure.equalize_hist(img_array))
        
        name = "check_cell_mask.pdf"
        dest = os.path.join(folder, name)
        plt.savefig(dest)
        plt.close()

    return mask


def process_image2D(img_array, desired_int, mask_params, rm_background_params, folder=".", check_plot=False):
    """
    ...
    """
    #TODO: add plotting options

    # compute mask that captures the silhouette of the cell.
    cell_mask = create_mask(img_array, *mask_params, folder, check_plot)
    
    # remove background from cell image
    img_signal = remove_background(img_array, *rm_background_params, folder, check_plot)
    
    # crop out the cell using the mask
    img_cropped = crop_image(img_signal, cell_mask)
    
    # compute CSR image, which depicts the cell with uniform intensity
    # and scale real (cropped-out) image and CSR image to have the same intensity
    img_csr = scale_image(cell_mask, desired_int) 
    img_real = scale_image(img_cropped, desired_int)
    
    return img_real, img_csr, cell_mask


def compute_K_values(range_of_t, params, filenames_and_z_slices, clc_type, indices, results_dest=".", check_plot=False):
    """
    ...
    """
    data = []
    desired_int, mask_params, rm_background_params = params
    
    for i in indices:
        filepath, z = filenames_and_z_slices[i]
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        
        # create subfolder to store plots produced while processing the image
        # (only important for check_plot=True)
        name = f"check_plots_{filename[:-4]}"
        sub_results_dest = create_folder(name, os.path.join(results_dest,"check_plots"))
        
        img_array, metadata = load_image2D(filepath, z, channel=0)
        img_real, img_csr, cell_mask = process_image2D(img_array, desired_int, mask_params, rm_background_params, sub_results_dest, check_plot)

        print("Computing K functions...")
        K_values_real = ripleys_K_fast(img_real, cell_mask, range_of_t, printout=False)
        K_values_csr = ripleys_K_fast(img_csr, cell_mask, range_of_t, printout=False)
        
        if check_plot:
            K_diff = np.array(K_values_real)-np.array(K_values_csr)
            plt.plot(range_of_t, K_diff, marker='o', linestyle="dashed")#, label=filename[-14:-4])
            plt.xlabel("t")
            plt.ylabel("K(t)")
            k_func_dest = os.path.join(sub_results_dest, "K_function.pdf")
            plt.savefig(k_func_dest)
            plt.close()

        # store information for later, e.g. plotting
        data.append([range_of_t, K_values_real, K_values_csr, clc_type, filename, z])
        
    return data
        
    
    
#--------------------
# "Ring" K functions
#--------------------


def get_K_diff(mask, img_real, img_csr, range_of_t, width, printout=True):
    """
    ...
    """
    K_values_real, K_values_ring_real, corr_real = ripleys_K_fast_ring(img_real, mask, range_of_t, width, printout)
    K_values_csr, K_values_ring_csr, corr_csr = ripleys_K_fast_ring(img_csr, mask, range_of_t, width, printout)
    K_diff = np.array(K_values_real) - np.array(K_values_csr)
    K_ring_diff = np.array(K_values_ring_real) - np.array(K_values_ring_csr)
    return K_diff, K_ring_diff, corr_real, corr_csr


def compute_K_values_ring(range_of_t, width, params, filenames_and_z_slices, clc_type, indices, results_dest=".", check_plot=False):
    """
    ...
    """
    data = []
    desired_int, mask_params, rm_background_params = params
    
    for i in indices:
        filepath, z = filenames_and_z_slices[i]
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        
        # create subfolder to store plots produced while processing the image
        # (only important for check_plot=True)
        name = f"check_plots_{filename[:-4]}"
        sub_results_dest = create_folder(name, os.path.join(results_dest,"check_plots"))
        
        img_array, metadata = load_image2D(filepath, z, channel=0)
        img_real, img_csr, cell_mask = process_image2D(img_array, desired_int, mask_params, rm_background_params, sub_results_dest, check_plot)

        print("Computing K functions...")
        K_diff, K_ring_diff, corr_real, corr_csr = get_K_diff(cell_mask, img_real, img_csr, range_of_t, width, True)
        
        
        if check_plot:
            
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
            ax[1][0].set_xlabel("$t$")
            ax[1][0].set_ylabel("$K(t)$")

            ax[1][1].set_title(f"Ring K function for (real)-(CSR), width={width}")
            ax[1][1].plot(range_of_t, K_ring_diff)
            ax[1][1].set_xlabel("$t$")
            ax[1][1].set_ylabel("$K_{Ring}(t)$")

            plt.colorbar(im1, ax=ax[0, 0])
            plt.colorbar(im2, ax=ax[0, 1])

            #fig.suptitle(title)
            
            k_func_dest = os.path.join(sub_results_dest, "K_function_computation.pdf")
            plt.savefig(k_func_dest)
            plt.close()

        # store information for later, e.g. plotting
        data.append([range_of_t, K_diff, K_ring_diff, clc_type, filename, z])
        
    return data


def plot_K_functions(data, result_dest, mode="disk", full_legend=False):
    """
    ...
    """
    plt.figure(figsize=(10,8))
    plt.title(f"K functions, o = clca, x = clcb")

    for range_of_t, K_disk_diff, K_ring_diff, clc_type, filename, _ in data:

        if clc_type == "clca":
            marker = "o"
        elif clc_type == "clcb":
            marker = "x"
            
        if mode=="disk":
            K_diff = K_disk_diff
            name = "K_functions_comparison"
        elif mode=="ring":
            K_diff = K_ring_diff
            name = "K_ring_functions_comparison"

        if full_legend:
            plt.plot(range_of_t, K_diff, marker=marker, linestyle="dashed", label=filename[-14:-4])
        else:
            plt.plot(range_of_t, K_diff, marker=marker, linestyle="dashed")
    
    plt.xlabel("t")
    plt.ylabel("K(t)")
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