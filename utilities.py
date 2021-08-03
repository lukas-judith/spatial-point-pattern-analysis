# useful functions for storing results etc.

import os
from datetime import datetime



def create_readme(folder_name, comment='-'):
    """
    Create readme file.
    """
    date = datetime.today().strftime('%Y-%m-%d')
    
    readme_content = f"""
Results from the {date}:

Check the folder check_plots to assess wether different steps in the pipeline work as intended.

________________________________________________________________
Comment:
{comment}

"""
    dest = os.path.join(folder_name, 'readme.txt')
    readme = open(dest, 'w')
    readme.write(readme_content[1:])
    readme.close()
    
    return dest
    
    
def write_params_readme(readme_dest, mask_params, rm_background_params):
    """
    Add information about the used parameters into the readme file.
    """
    date = datetime.today().strftime('%Y-%m-%d')
    
    readme_content = f"""
________________________________________________________________
Parameters used:

1. Background removal:
    Rolling ball radius = {rm_background_params[0]}
    
2. Cell mask/cell crop-out:
    Sigma = {mask_params[0]}
    Iterations erode = {mask_params[1]}
    Iterations dilate = {mask_params[2]}

"""
    with open(readme_dest, 'a') as readme:
        readme.write(readme_content[1:])
    
    
def create_folder(folder_name, parent_folder):
    """
    Creates a folder with specified name in specified directory. If folder name already exists, 
    create copies with slightly altered name.
    """
        
    dir_ = os.path.dirname(parent_folder)
    dir_list = os.listdir(dir_)
    # if parent folder does not exist already, create it
    if not os.path.basename(parent_folder) in dir_list:
        os.mkdir(parent_folder)

    # make subfolder
    dir_parent = os.listdir(parent_folder)

    count = 1
    child_folder = folder_name

    while child_folder.lower() in [path.lower() for path in dir_parent]:
        count += 1
        child_folder = folder_name + f"_#{count}"

    dest = os.path.join(parent_folder, child_folder)
    os.mkdir(dest)

    return dest


def extract_filenames(txt_filename, tif_folder):
    """
    ...
    """
    filenames_and_z_slices = []
    
    with open(txt_filename) as file:
        content = file.read()
        lines = content.split("\n")
        for line in lines:
            if len(line)>1:
                filename, z = line.split(" ")
                filepath = os.path.join(tif_folder, filename)
                filenames_and_z_slices.append([filepath, int(z)-1]) # minus 1 neccessary?
    return filenames_and_z_slices



def print_tif_series(filename, folder, destination="tif_series", channel=0):
    """
    From one .tif file, print every xy slice for all z values.
    """
    print(f"Creating series for .tif file {filename}")
    im, _ = load_image(filename, folder) 

    # pick desired channel
    zxy_arr = im[:, channel, :, :]
    
    N_z = zxy_arr.shape[0]

    fig, ax = plt.subplots(N_z, 1, figsize=(5, 5*N_z))
    
    # loop over all possible z values
    for z in range(N_z):
        # extract 2D slices
        xy_array = zxy_arr[z, :, :].astype("float32")
        ax[z].imshow(xy_array)
        ax[z].set_title(f"z = {z}")
    
    #plt.show()
    plt.savefig(destination+"/"+filename[:-4]+"_tif_series")
    print("Done!")