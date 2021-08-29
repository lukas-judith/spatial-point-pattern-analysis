from spatial_statistics_tools2D import *
from process_images2D import *
from two_sample_testing import *
from utilities import *
#from skimage import feature, restoration, segmentation, exposure, img_as_float
#from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
#from skimage.filters import gaussian
#from datetime import datetime


# create folder to store results of each run
parent_folder = "./Results"
folder_name = "results_" + datetime.today().strftime('%Y-%m-%d')

results_dest = create_folder(folder_name, parent_folder)

# folder in which the .tif files for the analysis are located
#tif_folder_clca = "extracted_tifs/2021-07-13_HA"
#tif_folder_clcb = "extracted_tifs/2021-07-26_HB"
tif_folder_clca = "extracted_tifs/2021-07-27_SA_HB"
tif_folder_clcb = "extracted_tifs/2021-07-27_SA_HB"

# .txt files containing the filenames, channels and z slices of .tif files to be used for the analysis
clca_txt = "#2_clca_files.txt"
clcb_txt = "#2_clcb_files.txt"


#--------
# Setup
#--------

# when set to True, create plots for every step of the pipeline to check if everything works
check_plot = True

# comment for current run
comment = "Test after implementing 3D functions"

# parameters for cropping out cell and creating CSR image
sigma = 25
sigma=20
iter_erode=35
iter_dilate=20

sigma_for_finding_minima=2
n_min=1
distance_min_max=0.02

mask_params = [sigma, iter_erode, iter_dilate, sigma_for_finding_minima, n_min, distance_min_max]

# parameters for background removal
rolling_ball_radius = 7
rm_background_params = [rolling_ball_radius]

# desired total intensity of the scaled images
desired_int = 1000000

# create readme file and write parameters
readme_dest = create_readme(results_dest, comment)
write_params_readme(readme_dest, mask_params, rm_background_params)

# get file paths, z slices and channels of the .tif files to be used in the analysis
clca_filedata = extract_filedata(clca_txt, tif_folder_clca)
clcb_filedata = extract_filedata(clcb_txt, tif_folder_clcb)


#------------------------
# Processing of 2D image
#------------------------
print("")
print("Starting computation of K functions for 2D images...")
print("Results will be stored at", results_dest)

# parameters for K function
range_of_t = np.arange(1, 1200, 40)
width = 1

# specifies the indices for images that should be processed 
# (index = (line number - 1) in the files clca_files.txt and clcb_files.txt)
indices_clca = range(len(clca_filedata))
indices_clcb = range(len(clcb_filedata))
print(f"No. of clca files used: {len(indices_clca)}")
print(f"No. of clcb files used: {len(indices_clcb)}")
print("")

params = [desired_int, mask_params, rm_background_params]

# compute K functions for clca and clcb datasets
# "data" contains range_of_t, K_diff, K_ring_diff, clc_type, filename, z slice
K_function_data_clca = compute_K_values(range_of_t, width, params, clca_filedata, 'clca', results_dest, check_plot)
print("")
K_function_data_clcb = compute_K_values(range_of_t, width, params, clcb_filedata, 'clcb', results_dest, check_plot)
print("")

K_function_data = K_function_data_clca + K_function_data_clcb

plot_K_functions(K_function_data, results_dest, mode="disk", full_legend=False)
plot_K_functions(K_function_data, results_dest, mode="disk", full_legend=True)
plot_K_functions(K_function_data, results_dest, mode="ring", full_legend=False)
plot_K_functions(K_function_data, results_dest, mode="ring", full_legend=True)

# save data for statistical testing
save_file(K_function_data_clca, "data_clca", results_dest)
save_file(K_function_data_clcb, "data_clcb", results_dest)


#----------------------------------------------
# Compute averages and apply two-sample tests
#----------------------------------------------

K_data_averaged, K_ring_data_averaged = average_K_diffs([K_function_data_clca, K_function_data_clcb], results_dest)

#save_file(K_data_averaged, "K_data_averaged", results_dest)
#save_file(K_ring_data_averaged, "K_ring_data_averaged", results_dest)

print("Done!")

