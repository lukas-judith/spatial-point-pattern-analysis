from spatial_statistics_tools import *
from process_images import *
from statistical_testing import *
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
tif_folder_clca = "extracted_tifs/2021-07-13_HA"
tif_folder_clcb = "extracted_tifs/2021-07-26_HB"


#--------
# Setup
#--------

channel = 0

# when set to True, create plots for every step of the pipeline to check if everything works
check_plot = True

# comment for current run
comment = "Test first full run"

# parameters for cropping out cell and creating CSR image
sigma = 20
erode = 35
dilate = 20
mask_params = [sigma, erode, dilate]

# parameters for background removal
rolling_ball_radius = 7
rm_background_params = [rolling_ball_radius]

# desired total intensity of the scaled images
desired_int = 1000000

# create readme file and write parameters
readme_dest = create_readme(results_dest, comment)
write_params_readme(readme_dest, mask_params, rm_background_params)

# get file paths of the .tif files to be used in the analysis
clca_filenames_and_z_slices = extract_filenames("clca_files.txt", tif_folder_clca)
clcb_filenames_and_z_slices = extract_filenames("clcb_files.txt", tif_folder_clcb)


#------------------------
# Processing of 2D image
#------------------------

range_of_t = np.arange(1, 1200, 40)
width = 1

# specifies the indices for images that should be processed 
# (index = (line number - 1) in the files clca_files.txt and clcb_files.txt)
indices_clca = range(len(clca_filenames_and_z_slices))
indices_clcb = range(len(clcb_filenames_and_z_slices))
print(f"No. of clca files used: {len(indices_clca)}")
print(f"No. of clcb files used: {len(indices_clcb)}")

params = [desired_int, mask_params, rm_background_params]

data_clca = compute_K_values_ring(range_of_t, width, params, clca_filenames_and_z_slices, 'clca', indices_clca, results_dest, check_plot)
data_clcb = compute_K_values_ring(range_of_t, width, params, clcb_filenames_and_z_slices, 'clcb', indices_clcb, results_dest, check_plot)
    
data = data_clca + data_clcb

plot_K_functions(data, results_dest, mode="disk", full_legend=False)
plot_K_functions(data, results_dest, mode="disk", full_legend=True)
plot_K_functions(data, results_dest, mode="ring", full_legend=False)
plot_K_functions(data, results_dest, mode="ring", full_legend=True)

save_file(data_clca, "data_clca", results_dest)
save_file(data_clcb, "data_clcb", results_dest)


#----------------------------------------------
# Compute averages and apply two-sample tests
#----------------------------------------------

K_data_averaged, K_ring_data_averaged = average_K_diffs([data_clca, data_clcb], results_dest)

#save_file(K_data_averaged, "K_data_averaged", results_dest)
#save_file(K_ring_data_averaged, "K_ring_data_averaged", results_dest)

print("Done!")

#...