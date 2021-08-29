import open3d as o3d
from process_images2D import *


def load_image3D(path, channel=0):
    """
    Load 3D cell image and metadata from .tif file.
    """
    # get array with all pixel intensities and metadata of image
    im, metadata = load_image(path) 

    # pick desired channel
    zxy_arr = im[:, channel, :, :].astype("float32")

    return zxy_arr, metadata


def plot_3D_array(input_arr, x_scale=1, y_scale=1, z_scale=1):
    """
    Uses open3D library to show binary 3D array as point cloud.
    """
    if len(np.unique(input_arr))>2:
        print("Error! Input should be binary array!")
        return

    # only plot outline, i.e. most outer pixels of the shape to reduce computation time
    arr = input_arr.astype('int') - binary_erosion(input_arr.astype('int'))
    
    zxy = np.where(arr>0)
    # scale all axes
    z_scaled = zxy[0] * z_scale
    x_scaled = zxy[1] * x_scale
    y_scaled = zxy[2] * y_scale
    zxy_scaled = [z_scaled, x_scaled, y_scaled]
    
    # create 3D array containing all coordinates/indices where a point in the outline is =1
    point_coordinates = np.stack(zxy_scaled).T.astype('float64')

    # use open3D library to plot
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_coordinates)
    o3d.visualization.draw_geometries([pcd])
    

def remove_background3D(zxy_arr, img_name, radius, temp_folder, check_plot=True, save=False):
    """
    Removes background from 3D array by applying the rolling ball method to every 2D slice separately.
    """ 
    bg_removal_folder = create_folder("bg_removal3D", temp_folder, False)
    file_path = os.path.join(bg_removal_folder, img_name+"_signal3D.npy")
    if os.path.isfile(file_path):
        signal3D = np.load(file_path)
        return signal3D
    elif save:
        print("No .npy file for signal array found, computing signal array...")

    # create plots of intermediate results to check if background was removed correctly
    check_plots_folder = create_folder("check_plots", bg_removal_folder, False)

    signal_slices = []

    # loop over all z slices (2D) and remove background for each slice
    N_z = zxy_arr.shape[0]
    for z in range(N_z):
        # folder for intermediate background removal results for respective z slice
        check_plots_subfolder = create_folder(f"check_"+img_name, check_plots_folder, False)
        xy_array = zxy_arr[z, :, :].astype("float32")   
        signal2D = remove_background(xy_array, radius, check_plots_subfolder, check_plot, z)
        signal_slices.append(signal2D)


    # stack 2D signal slices together to 3D signal array    
    signal3D = np.stack(signal_slices)   
    if save:
        np.save(file_path, signal3D)

    return signal3D


def create_mask3D(img, img_name, sigma=40, iter_dilation3D=10, iter_erosion3D=10, iter_erosion2D=5, sigma_for_finding_minima=0, n_min=1, distance_min_max=0.05, temp_folder="temp", check_plot=True, save=False):
    """
    ...
    """
    # folder to store intermediate results
    cell_mask_folder = create_folder("cell_mask3D", temp_folder, False)
    file_path = os.path.join(cell_mask_folder, img_name+"_cell_mask3D.npy")
    if os.path.isfile(file_path):
        cell_mask3D = np.load(file_path)
        return cell_mask3D
    elif save:
        print("No .npy file for cell mask found, computing cell mask array...")

    # create plots of intermediate results to check if background was removed correctly
    check_plots_folder = create_folder("check_plots", cell_mask_folder, False)

    masks2D = []
    for z in range(img.shape[0]):
        mask2D = create_mask(img[z], sigma=sigma, folder=check_plots_folder, check_plot=check_plot, postprocessing=False, z=z, sigma_for_finding_minima=sigma_for_finding_minima, n_min=n_min, distance_min_max=distance_min_max)
        masks2D.append(mask2D)

    # stack 2D masks to create full 3D mask
    mask3D = np.stack(masks2D)

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
    for z in range(img.shape[0]):
        mask2D = binary_erosion(mask3D[z], iterations=iter_erosion2D)
        masks2D_.append(mask2D)
    mask3D = np.stack(masks2D_)
    
    #---------------------------------------------------------------
    # create plots for checking if the method is working correctly
    #---------------------------------------------------------------
    if check_plot:
        print("Create new plots to check after postprocessing...")
        for z in range(img.shape[0]):
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
            dest = os.path.join(check_plots_folder, name)
            plt.savefig(dest)
            plt.close()
            
    if save:
        np.save(file_path, mask3D)

    return mask3D