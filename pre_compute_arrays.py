from spatial_statistics_tools3D import *
from process_images3D import *
from process_images2D import *
from two_sample_testing import *
from utilities import *
import psutil


# folder for intermediate results
temp_folder = create_folder("temp", "/Volumes/Seagate Exp/hci_project", allow_duplicates=False)

ans = input("Reset/create finished_files.txt? yes/[no] ")
if ans.lower() == "yes":
    # remember for what files computations have been finished already
    f = open(temp_folder+"/finished_files.txt", 'w')
    f.close()

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
comment = ""


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
distance_min_max=0.05
max_position_thresh=0.1

# if computation of cell mask is inaccurate,
# a custom threshold for the 2D slices can be chosen
# keys: respective z slices
# values: custom threshold
custom_thresholds = {
    19 : 0.19
}
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
print("Results will be stored at", temp_folder)

# specifies the indices for images that should be processed 
# (index = (line number - 1) in the files clca_files.txt and clcb_files.txt)
indices_clca = range(len(clca_filedata))
indices_clcb = range(len(clcb_filedata))
print(f"No. of clca files used: {len(indices_clca)}")
print(f"No. of clcb files used: {len(indices_clcb)}")


# loop over all files and save results of background removal, cell mask computation and 3D autocorrelation
for filedata, data_type in [[clca_filedata, 'clca'], [clcb_filedata, 'clcb']]:
    
    indices = range(len(filedata))

    count=0
    for i in indices:
        
        count+=1

        img_path, channel, _ = filedata[i]
        img_base_name = get_filename_from_path(img_path)
        
        filename = os.path.basename(img_path)
       
        print("")
        print(f"[{data_type} {count}/{len(indices)}] Processing {filename}...")
         
        img_name = f"{img_base_name}_channel{channel}"
        
        
        with open(temp_folder+"/finished_files.txt", 'r') as f:
            text = f.read()
        finished_files = text.split("\n")

        if img_name in finished_files:
            print("Computations for this file have already been completed.")
            continue
        
        # load 3D image from .tif file
        im, _ = load_image(img_path)

        # pick desired channel 
        zxy_arr = im[:, channel, :, :]

        
        #------------------------
        # 1) Background removal
        #------------------------
        
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory()[2]
        print(f"CPU usage: {cpu_usage:.2f}%")
        print(f"Memory usage: {mem_usage:.2f}%")

        radius=7

        t0=time()
        print("Remove background...")
        signal_arr = remove_background3D(zxy_arr, *rm_background_params, img_name=img_name, temp_folder=temp_folder, check_plot=True, save=True)
        t1=time()
        diff=t1-t0
        print(f"Completed in {diff:.2f} seconds")

        #------------------------
        # 2) Computing 3D masks
        #------------------------
        
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory()[2]
        print(f"CPU usage: {cpu_usage:.2f}%")
        print(f"Memory usage: {mem_usage:.2f}%")

        sigma = 40
        sigma_for_finding_minima = 0
        n = 1
        distance_min_max = 0.1

        t0=time()
        print("Computing 3D cell mask...") 
        mask3D = create_mask3D(zxy_arr, *mask_params, img_name=img_name, temp_folder=temp_folder, check_plot=True, save=True)
        t1=time()
        diff=t1-t0
        print(f"Completed in {diff:.2f} seconds")

        #-----------------------
        # 3) Autocorrelation
        #-----------------------
        
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory()[2]
        print(f"CPU usage: {cpu_usage:.2f}%")
        print(f"Memory usage: {mem_usage:.2f}%")

        desired_int = 100000000 

        img_real = cut_away_zeros3D(scale_image(signal_arr*mask3D, desired_int))
        img_csr = cut_away_zeros3D(scale_image(mask3D, desired_int))
        
        del mask3D
        del signal_arr
        del zxy_arr

        if not np.isclose(np.sum(img_real), np.sum(img_csr)):
            raise Exception("Error! Pixel sum of real and CSR image are not equal!")
            
        t0=time()
        print("Computing 3D autocorrelation for real image...")
        print("Image dimensions:", img_real.shape)
        auto_corr_real = autocorrelation_3D(img_real, "real_"+img_name, temp_folder=temp_folder, save=True)
        t1=time()
        diff=t1-t0
        print(f"Completed in {diff:.2f} seconds")

        del auto_corr_real
        del img_real

        t0=time()
        print("Computing 3D autocorrelation for CSR image...")
        print("Image dimensions:", img_csr.shape)
        auto_corr_csr = autocorrelation_3D(img_csr, "csr_"+img_name, temp_folder=temp_folder, save=True)
        t1=time()
        diff=t1-t0
        print(f"Completed in {diff:.2f} seconds")

        del auto_corr_csr
        del img_csr   
        
        f = open(temp_folder+"/finished_files.txt", 'a')
        f.write(img_name)
        f.write("\n")
        f.close()
        
        write_file_in_log(img_name, log_dest)
        
        
print("Done!")









