import open3d as o3d
from process_images2D import *


def load_image3D(path, channel=0):
    """
    Load 3D cell image and metadata from .tif file.
    """
    # get array with all pixel intensities and metadata of image
    im, metadata = load_image(path) 

    # pick desired channel
    zyx_arr = im[:, channel, :, :].astype("float32")

    return zyx_arr, metadata


def plot_3D_array(input_arr, x_scale=1, y_scale=1, z_scale=1):
    """
    Uses open3D library to show binary 3D array as point cloud.
    """
    if len(np.unique(input_arr))>2:
        print("Error! Input should be binary array!")
        return

    # only plot outline, i.e. most outer pixels of the shape to reduce computation time
    arr = input_arr.astype('int') - binary_erosion(input_arr.astype('int'))
    
    zyx = np.where(arr>0)
    # scale all axes
    z_scaled = zyx[0] * z_scale
    x_scaled = zyx[1] * x_scale
    y_scaled = zyx[2] * y_scale
    zyx_scaled = [z_scaled, x_scaled, y_scaled]
    
    # create 3D array containing all coordinates/indices where a point in the outline is =1
    point_coordinates = np.stack(zyx_scaled).T.astype('float64')

    # use open3D library to plot
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_coordinates)
    o3d.visualization.draw_geometries([pcd])
    
    
def cut_away_zeros3D(input_):
    """
    From a given 3D input array, remove all outer xy-, yz- and zx-slices that contain only zeros.
    """
    arr = input_.astype(np.float32)

    only_zeros_z = np.all(arr==0, axis=0)
    only_zeros_y = np.all(arr==0, axis=1)

    non_zeros_x = np.logical_not(np.all(only_zeros_z, axis=1))
    non_zeros_y = np.logical_not(np.all(only_zeros_z, axis=0))
    non_zeros_z = np.logical_not(np.all(only_zeros_y, axis=1))

    # all non-zero indices along one axis
    non_zeros_x_idx = np.where(non_zeros_x)[0]
    non_zeros_y_idx = np.where(non_zeros_y)[0]
    non_zeros_z_idx = np.where(non_zeros_z)[0]

    min_x = non_zeros_x_idx[0]
    max_x = non_zeros_x_idx[-1]+1
    min_y = non_zeros_y_idx[0]
    max_y = non_zeros_y_idx[-1]+1
    min_z = non_zeros_z_idx[0]
    max_z = non_zeros_z_idx[-1]+1

    out = arr[min_z:max_z, min_x:max_x, min_y:max_y].astype(np.float32)

    if not np.isclose(np.sum(out), np.sum(arr), rtol=1e-05):
        print("Error! Sums of input and output are different! Non-zero parts of the array were probably cut off!")
        #print(np.sum(out))
        #print(np.sum(input_))
        return

    return out
    
    
def remove_background3D(zyx_arr, radius, img_name="unnamed", temp_folder="temp", check_plot=True, save=False):
    """
    Removes background from 3D array by applying the rolling ball method to every 2D slice separately.
    """ 
    bg_removal_folder = create_folder("bg_removal3D", temp_folder, False)
    file_path = os.path.join(bg_removal_folder, img_name+"_signal3D.npy")
    if os.path.isfile(file_path):
        print("Signal array already computed, loading .npy file...")
        signal3D = np.load(file_path).astype(np.float32)
        return signal3D
    elif save:
        print("No .npy file for signal array found, computing signal array...")

    # create plots of intermediate results to check if background was removed correctly
    check_plots_folder = create_folder("check_plots", bg_removal_folder, False)

    # folder for intermediate background removal results for respective z slice
    check_plots_subfolder = create_folder(f"check_"+img_name, check_plots_folder, False)
    
    signal_slices = []

    # loop over all z slices (2D) and remove background for each slice
    N_z = zyx_arr.shape[0]
    for z in range(N_z):
        xy_array = zyx_arr[z, :, :].astype("float32")   
        signal2D = remove_background(xy_array, radius, check_plots_subfolder, check_plot, z)
        signal_slices.append(signal2D.astype(np.float32))


    # stack 2D signal slices together to 3D signal array    
    signal3D = np.stack(signal_slices).astype(np.float32)  
    if save:
        np.save(file_path, signal3D)

    return signal3D


def create_mask3D(img, sigma=40, iter_dilation3D=10, iter_erosion3D=10, iter_erosion2D=15, iter_dilation2D=10, sigma_for_finding_minima=0, n_min=1, distance_min_max=0.05, max_position_thresh=0.1, custom_thresholds=None, img_name="unnamed", temp_folder="temp", check_plot=True, save=False):
    """
    Given a 3D image, computes a binary mask capturing the shape of the object(s) in the image. This is done by 
    computing the respective 2D mask for every z-slice of the 3D image (see: process_images2D.create_mask). 
    When check_plot=True, plots of intermediate results during the computation are produced. 
    
    Args postprocessing: 
        img (np.ndarray): 3D input image
        sigma (float): standard deviation for Gaussian blur
        iter_dilation3D (int): number of iterations for binary dilation of 3D mask (postprocessing)
        iter_erosion3D (int): number of iterations for binary erosion of 3D mask (postprocessing)
        iter_erosion2D (int): number of iterations for binary erosion of every z-slice of the 3D mask (postprocessing)
        iter_dilation2D (int): number of iterations for binary dilation of every z-slice of the 3D mask (postprocessing)
    
    Args for finding optimal cut-off value: 
        sigma_for_finding_minima (float): standard deviation for smoothing the function of histogram counts for intensities 
            of the blurred image; can be used to facilitate finding the optimal cut-off value for computing the binary 2D mask
        n_min (int): number of minimum in function of histogram counts for intensities 
            of the blurred image to be selected as the cut-off value for computing the binary 2D mask
        distance_min_max (float): minimum distance (assumes normalized images, i.e. in interval [0,1]) that the minimum
            (of the function of histogram counts for intensities of the blurred image) is allowed to have from the 
            global maximum 
        max_position_thresh (float): maximum position (assumes normalized images, i.e. in interval [0,1]) that the maximum
            (of the function of histogram counts for intensities of the blurred image) is allowed to have 
        custom_thresholds (dict): dictionary specifying for which z-slices a self-selected cut-off value should be used
            (see process_images2D.create_mask)
        
    Returns:
        np.ndarray : 3D binary mask capturing the shape of the object(s) on the input image
    
    """
    # folder to store intermediate results
    cell_mask_folder = create_folder("cell_mask3D", temp_folder, False)
    file_path = os.path.join(cell_mask_folder, img_name+"_cell_mask3D.npy")
    if os.path.isfile(file_path):
        print("3D mask array already computed, loading .npy file...")
        cell_mask3D = np.load(file_path).astype(np.float32)
        return cell_mask3D
    elif save:
        print("No .npy file for cell mask found, computing cell mask array...")

    # create plots of intermediate results to check if background was removed correctly
    check_plots_folder = create_folder("check_plots", cell_mask_folder, False)
    
    N_pixels_2D_slice = img[0].shape[0] * img[0].shape[1]
    
    # folder for intermediate 2D mask results for respective z slice
    check_plots_subfolder = create_folder(f"check_"+img_name, check_plots_folder, False)
    
    masks2D = []
    for z in range(img.shape[0]):
        # custom threshold can be applied for computing 2D masks
        custom_thresh = None
        if not custom_thresholds is None:
            if z-1 in custom_thresholds.keys():
                custom_thresh = custom_thresholds[z-1]
            else:
                custom_thresh = None
        
        mask2D = create_mask(img[z], sigma=sigma, folder=check_plots_subfolder, check_plot=check_plot, postprocessing=False, z=z, sigma_for_finding_minima=sigma_for_finding_minima, n_min=n_min, distance_min_max=distance_min_max, max_position_thresh=max_position_thresh, custom_thresh=custom_thresh)
        
        N_pixels_not_zero = np.sum(mask2D>0)
        ratio = N_pixels_not_zero/N_pixels_2D_slice
        # if too many pixels are >0, this is most likely an error, set mask to all-zero
        if ratio > 0.9:
            mask2D = np.zeros(mask2D.shape, dtype=np.float32)
        
        # if one 2D mask is all-zero, we can set the the following slices to zero as well,
        # as the "top" of the object in the image (e.g. the cell) has been reached
        if len(masks2D)>0:
            if np.sum(masks2D[-1])==0:
                mask2D = np.zeros(mask2D.shape, dtype=np.float32)
        
        masks2D.append(mask2D)

    # stack 2D masks to create full 3D mask
    mask3D = np.stack(masks2D).astype(np.float32)
    
    print("Apply 3D postprocessing...")
    # postprocessing: perform binary dilation and erosion to close holes/gaps and to remove artifacts
    # from Gaussian blur (due to blur, the mask would otherwise appear bigger than the actual cell)
    # use zero-padding to assure that the dilation works properly
    mask_padded = np.pad(mask3D.astype('int'), iter_dilation3D, mode='constant')
    mask3D = binary_erosion(binary_dilation(mask_padded, iterations=iter_dilation3D), iterations=iter_erosion3D)
    # bring mask back to old shape, before zero-padding
    mask3D = mask3D[iter_dilation3D:-iter_dilation3D, iter_dilation3D:-iter_dilation3D, iter_dilation3D:-iter_dilation3D]
    
    # additional erosion that only affects the xy-plane; used to remove artifacts from Gaussian blur,
    # since blur causes the mask's area to be slightly larger than the original cell area
    masks2D_ = []
    for z in range(mask3D.shape[0]):
        mask2D = binary_erosion(mask3D[z], iterations=iter_erosion2D)
        mask2D = binary_dilation(mask2D, iterations=iter_dilation2D)
        masks2D_.append(mask2D.astype(np.float32))
    mask3D = np.stack(masks2D_)
    
    #---------------------------------------------------------------
    # create plots for checking if the method is working correctly
    #---------------------------------------------------------------
    if check_plot:
        print("Create new plots to check after postprocessing...")
        for z in range(mask3D.shape[0]):
            fig, ax = plt.subplots(2, 2, figsize=(10,10))
            ax[0][0].set_title(f"Original image, z={z}")
            ax[0][0].imshow(img[z])
            ax[0][1].set_title(f"Mask, z={z}")
            ax[0][1].imshow(mask3D[z])
            ax[1][0].set_title("Overlay of orig. image with mask")
            img_norm = normalize_image(img[z])
            ax[1][0].imshow(mask3D[z]-img_norm*4)
            ax[1][1].set_title("Compare: orig. image, histogram equalized")
            ax[1][1].imshow(exposure.equalize_hist(img[z]))

            if not z is None:
                z_str = str(z+1)
                # if z is one-digit number, add zero in front, e.g. 1 --> 01
                if len(z_str)==1:
                    z_str = "0"+z_str
                name = f"after_3D_postprocessing_z{z_str}_check_cell_mask.pdf"
            else:
                name = "check_cell_mask.pdf"
            dest = os.path.join(check_plots_subfolder, name)
            plt.savefig(dest)
            plt.close()
            
    if save:
        np.save(file_path, mask3D)

    return mask3D



def process_image3D(img_array, desired_int, mask_params, rm_background_params, img_name="unnamed", temp_folder="temp", check_plot=False, save=False):
    """
    Performs preprocessing of 3D image, including background removal and computation of binary 3D mask
    capturing the object(s) on the image. Returns real and CSR image to be used in K-function computation.
    """

    # compute mask that captures the silhouette of the cell.
    cell_mask = create_mask3D(img_array, *mask_params, img_name=img_name, temp_folder=temp_folder, check_plot=check_plot, save=save)
    
    # remove background from cell image
    img_signal = remove_background3D(img_array.astype(np.float32), *rm_background_params, img_name=img_name, temp_folder=temp_folder, check_plot=check_plot, save=save)
    
    # crop out the cell using the mask
    img_cropped = img_signal * cell_mask
    
    # compute CSR image, which depicts the cell with uniform intensity
    # and scale real (cropped-out) image and CSR image to have the sum of pixel intensities
    img_csr = create_csr_img(cell_mask, desired_int).astype(np.float32) 
    img_real = scale_image(img_cropped, desired_int).astype(np.float32)
    
    # check if sum of pixel intensities is the same for real and CSR image
    if not np.isclose(np.sum(img_real), np.sum(img_csr), rtol=1e-05):
        print("Error! Sum of all pixel intensities is different for real and CSR image!")
        return
    
    return img_real, img_csr



