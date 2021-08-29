# useful functions for storing results etc.

import os
import pickle
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
    
    
def create_folder(folder_name, parent_folder, allow_duplicates=True):
    """
    Creates a folder with specified name in specified directory. 
    In case of allow_duplicates: If folder name already exists, create copies with slightly altered name.
    """ 
    # if parent folder does not exist already, create it
    if not parent_folder=='.':
        if os.path.dirname(parent_folder)=='':
            dir_super = '.'
        else:
            dir_super = os.path.dirname(parent_folder)
        dir_super_list = os.listdir(dir_super)
        if not os.path.basename(parent_folder) in dir_super_list:
            os.mkdir(parent_folder)

    # make subfolder
    dir_parent = os.listdir(parent_folder)

    count = 1
    child_folder = folder_name
    
    while child_folder.lower() in [path.lower() for path in dir_parent]:
        # if duplicates are not allowed, do not create a new folder
        if not allow_duplicates:
            return os.path.join(parent_folder, child_folder)
        count += 1
        child_folder = folder_name + f"_#{count}"
   
    dest = os.path.join(parent_folder, child_folder)
    os.mkdir(dest)

    return dest


def get_filename_from_path(path):
    """
    Returns filename without file extension and parent folders.
    """
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]
    return filename


def save_file(file, filename, folder):  
    """
    Saves file using pickle.
    """
    path = os.path.join(folder, filename)
    
    with open(path, "wb") as f:   
        pickle.dump(file, f)

        
def load_file(filename, folder): 
    """
    Loads file using pickle.
    """
    path = os.path.join(folder, filename)
    
    with open(path, "rb") as f:  
        file = pickle.load(f)
    return file



# auxiliary functions to create the .txt files used to load the datasets

def contains_any_word(string, word_list):
    """
    Checks if a string contains any word from a given list.
    """
    for word in word_list:
        if word.lower() in string.lower():
            return True
    return False    
    

def create_txt_for_datasets(folder, txt_file, channel, z_slice, filter_words):
    """
    Auxiliary function for creating .txt files for loading datasets.
    """
    filtered_files = []
    for f in sorted(os.listdir(folder)):
        # filter out any files with any of the specified words in their filename
        if not contains_any_word(f, filter_words):
            filtered_files.append(f)

    txt = open(txt_file, 'w')
    for f in filtered_files:
        txt.write(f+" "+str(channel)+" "+str(z_slice)+"\n")
    txt.close()
    
    
