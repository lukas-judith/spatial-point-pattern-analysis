from spatial_statistics_tools3D import *
from process_images3D import *
from two_sample_testing import *
from utilities import *


# create folder to store results of each run
parent_folder = "./Results"
folder_name = "results3D_" + datetime.today().strftime('%Y-%m-%d')

results_dest = create_folder(folder_name, parent_folder)

# folder for intermediate results when save=True (see under Setup)
# WARNING: depending on the number and sizes of the used images, 
# this can require a large amount of available memory (up to ~1000GB)
temp_folder = "/Volumes/Seagate Exp/hci_project/temp"

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
# when set to True, all functions for processing the image or computing the K-function will store intermediate results 
# to speed up future computations; requires a large amount of available storage, depending on image and dataset size
save = True

# comment for current run, will be included in log file
comment = "..."


#----------------------------------------------------
# parameters for computing cell mask and CSR image:
# (see process_images3D.py for documentation)
# sigma for Gaussian blur 
sigma=40
iter_dilation3D=10 
iter_erosion3D=10 
iter_erosion2D=60
iter_dilation2D=55
# parameters for finding optimal intensity cut-off after blur
sigma_for_finding_minima=0
n_min=1
distance_min_max=0.1
max_position_thresh=0.1

# set custom thresholds for computing 2D slices of the 3D mask
# keys: z positions (starting at 1, not at 0)
# values: cut-off value at respective z position
custom_thresholds = {
    # example values:
    #5 : 0.4,
    45 : 0, 
}

# if no custom thresholds should be used:
custom_thresholds = None

mask_params = [sigma, iter_dilation3D, iter_erosion3D, iter_erosion2D, iter_dilation2D, sigma_for_finding_minima, n_min, distance_min_max, max_position_thresh, custom_thresholds]


#----------------------------------------------------
# parameters for background removal
# (see process_images3D.py for documentation)
rolling_ball_radius = 7

rm_background_params = [rolling_ball_radius]


#----------------------------------------------------
# parameters for computing K-function
# the scaling refers to the dimension of each voxel
unit = "microns"
x_scaling = 0.0353
y_scaling = 0.0353
z_scaling = 0.15
t_scaling = x_scaling

# specify the range of radii over which the autocorrelation values will be summed up
min_t = 1
max_t = 900*t_scaling
n_t = 20

range_of_t = np.linspace(min_t, max_t, n_t)
width = 1

K_func_params = [range_of_t, width, x_scaling, y_scaling, z_scaling]


#----------------------------------------------------
# desired total intensity of the scaled images
desired_int = 100000000

# create log file and write parameters
log_dest = create_log(temp_folder, comment)

# TODO: include all parameters
write_params_log3D(log_dest, mask_params, rm_background_params, [unit, min_t, max_t, n_t]+K_func_params)

# get file paths, z slices and channels of the .tif files to be used in the analysis
clca_filedata = extract_filedata(clca_txt, tif_folder_clca)
clcb_filedata = extract_filedata(clcb_txt, tif_folder_clcb)

# all parameters
params = [desired_int, mask_params, rm_background_params, K_func_params]


#------------------------
# Processing of 3D image
#------------------------

print("")
print("Starting computation of K functions for 3D images...")
print("Results will be stored at", results_dest)


# specifies the indices for images that should be processed 
# (index = (line number - 1) in the files clca_files.txt and clcb_files.txt)
indices_clca = range(len(clca_filedata))
indices_clcb = range(len(clcb_filedata))
print(f"No. of clca files used: {len(indices_clca)}")
print(f"No. of clcb files used: {len(indices_clcb)}")
print("")



# compute K functions for clca and clcb datasets
# "data" contains range_of_t, K_diff, K_ring_diff, clc_type, filename, z=None

K_function_data_clca = compute_all_K_functions3D(clca_filedata, "clca", params, temp_folder, check_plot, save, log_dest)
print("")
K_function_data_clcb = compute_all_K_functions3D(clcb_filedata, "clcb", params, temp_folder, check_plot, save, log_dest)
print("")


K_function_data = K_function_data_clca + K_function_data_clcb

plot_K_functions(K_function_data, results_dest, mode="disk", full_legend=False, unit=unit)
plot_K_functions(K_function_data, results_dest, mode="disk", full_legend=True, unit=unit)
plot_K_functions(K_function_data, results_dest, mode="ring", full_legend=False, unit=unit)
plot_K_functions(K_function_data, results_dest, mode="ring", full_legend=True, unit=unit)

# save data for statistical testing
save_file(K_function_data_clca, "data_clca", results_dest)
save_file(K_function_data_clcb, "data_clcb", results_dest)


#---------------------
# Compute averages  
#---------------------

K_data_averaged, K_ring_data_averaged = average_K_diffs([K_function_data_clca, K_function_data_clcb], results_dest, unit=unit)

#save_file(K_data_averaged, "K_data_averaged", results_dest)
#save_file(K_ring_data_averaged, "K_ring_data_averaged", results_dest)

print("Done!")

