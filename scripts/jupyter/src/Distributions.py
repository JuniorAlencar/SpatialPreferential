import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  
import glob
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from IPython.display import clear_output
import gzip
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sklearn.model_selection import train_test_split
import shutil
from multiprocessing import Pool
import math
import re


# create folder to results
def make_results_folders():
    path = "../../results"
    # If file in all_files, create check_folder to move files
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+"/alpha_a")
        os.makedirs(path+"/alpha_g")
        os.makedirs(path+"/N")
        os.makedirs(path+"/distributions")
        os.makedirs(path+"/network")
    else:
        pass

# Open the folder string name and return (alpha_a_value, alpha_g_value)
def extract_alpha_values(folder_name):
    pattern = r"alpha_a_(-?\d+\.\d+)_alpha_g_(-?\d+\.\d+)"
    match = re.match(pattern, folder_name)
    if match:
        alpha_a = float(match.group(1))
        alpha_g = float(match.group(2))
        return (alpha_a, alpha_g)
    else:
        return None, None

# List all pair of (alpha_a,alpha_g) folders in (N,dim) folder
def list_all_folders(N,dim):
    directory = f"../../data/N_{N}/dim_{dim}/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    lst_folders = lst_folders[0]
    set_parms = []

    for i in range(len(lst_folders)):
        set_parms.append(extract_alpha_values(lst_folders[i]))

    return set_parms

def distribution(N, dim, alpha_a, alpha_g, degree, save):
    hist, bins_edge = np.histogram(degree, bins=np.arange(0.5,10**4+1.5,1), density=True)
    
    P = hist*np.diff(bins_edge)             # distribution = density*deltaK
    K = bins_edge[:-1]+bins_edge[:1]
    index_remove = []                       # load index with distribution zero
    
    for idk,elements in enumerate(P):
        if(elements==0):
            index_remove.append(idk) 
    # Removing elements in k_mean and distribution with distribution = 0 (empty box)
    p_real = np.delete(P,index_remove)      
    k_real = np.delete(K,index_remove)
    
    if(save == True):
        # Save distribution (or update)
        distri_dataframe = pd.DataFrame(data={"k":k_real,"pk":p_real})
        distri_dataframe['pk'] = distri_dataframe['pk'].apply(lambda x: format(x, '.2e'))
        distri_dataframe.to_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/distri_linear_all.csv",mode="w",index=False)
        return k_real,p_real
    else:
        return k_real,p_real
    

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(N, dim, alpha_a, alpha_g, counter_dict, bin_count, save):

    max_x = np.log10(max(list(counter_dict.keys())))
    max_y = np.log10(max(list(counter_dict.values())))
    max_base = max([max_x,max_y])

    min_x = np.log10(min(drop_zeros(list(counter_dict.keys()))))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    Pk = (np.histogram(list(counter_dict.keys()),bins,weights=list(counter_dict.values()), density = True)[0] / np.histogram(list(counter_dict.keys()),bins)[0])*np.diff(bins)
    k = (np.histogram(list(counter_dict.keys()),bins,weights=list(counter_dict.keys()))[0] / np.histogram(list(counter_dict.keys()),bins)[0])

    k = [x for x in k if str(x) != 'nan']
    Pk = [x for x in Pk if str(x) != 'nan']
    
    if(save==True):
        distri_dataframe = pd.DataFrame(data={"k":k,"pk":Pk})
        distri_dataframe['pk'] = distri_dataframe['pk'].apply(lambda x: format(x, '.2e'))
        distri_dataframe.to_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/distri_log_all.csv",mode="w",index=False)
        return k,Pk
    else:
        return k,Pk

    
def ln_q(k, pk, q, eta):
    k_values = np.zeros(len(k))
    for i in range(len(k)):
        k_values[i] = (1-(1-q)*(k[i]/eta))**(1/(1-q))
    P0 = 1/sum(k_values)
    return ((pk/P0)**(1-q)-1)/(1-q)

def q(alpha_a,d):
    if(0 <= alpha_a/d <= 1):
        return 4/3
    else:
        return round((1/3)*np.exp(1-alpha_a/d)+1.0,4)

def kappa(alpha_a,d):
    if(0 <= alpha_a/d <= 1):
        return 0.3
    else:
        return round(-1.15*np.exp(1-alpha_a/d)+1.45,4)

def find_order_of_magnitude(number):
    order = int(math.floor(math.log10(abs(number))))
    return abs(order)    

# ----------------------------
# FUNCTION MOVED TO TOP LEVEL
# ----------------------------
def process_file(args):
    """Process a single file, extract degree data, and delete the file."""
    i, file, n_lines = args  # Unpack the arguments
    degrees = np.zeros(n_lines, dtype=np.int32)  # Create a fixed array (avoids append)

    with gzip.open(file, 'rt') as gzip_file:
        count = 0
        for line in gzip_file:
            if line.startswith("degree "):  # Faster than strip() and [:6]
                degrees[count] = int(line[7:])  # Extract degree and store in the array
                count += 1
                if count == n_lines:  # Avoid unnecessary processing
                    break

    # Delete the processed file
    os.remove(file)
    return i, degrees, os.path.basename(file)

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def degree_file(N, dim, alpha_a, alpha_g):
    """Process degree files and store the results in a .npy file."""
    # Define file paths
    path_folder = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml"
    degree_file_path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/degree.npy"
    filenames_path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/filenames_degree.csv"
    print(f"Processing to N = {N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")
    # List all .gml.gz files in the directory
    all_files = sorted(glob.glob(os.path.join(path_folder, "*.gml.gz")))  # Sort for consistency
    
    if not all_files:
        print(f"⚠️ Warning: The folder {path_folder} is empty. Skipping processing.")
        return  # Exit function without doing anything
    
    all_filenames = [os.path.basename(file) for file in all_files]

    # Number of rows per file
    n_lines = N  

    # Check if degree.npy already exists
    if os.path.exists(degree_file_path) and os.path.exists(filenames_path):
        # Load the list of already processed files
        existing_filenames = pd.read_csv(filenames_path, header=None).squeeze().tolist()

        # Identify new files that have not been processed yet
        new_files = [file for file in all_files if os.path.basename(file) not in existing_filenames]

        if not new_files:
            print("All files have already been processed. No updates needed.")
            return
        else:
            print(f"Found {len(new_files)} new files. Updating degree.npy...")

            # Load existing data
            existing_data = np.load(degree_file_path)

            # Create a matrix for new files
            n_new_files = len(new_files)
            new_data = np.zeros((n_lines, n_new_files), dtype=np.int32)

            # Process new files in parallel (pass n_lines as argument)
            with Pool() as pool:
                results = pool.map(process_file, [(i, file, n_lines) for i, file in enumerate(new_files)])

            # Insert new results into `new_data`
            for i, degrees, filename in results:
                new_data[:, i] = degrees
                existing_filenames.append(filename)  # Add to the processed file list

            # Concatenate new data with the existing dataset
            updated_data = np.hstack((existing_data, new_data))

            # Save the updated dataset
            np.save(degree_file_path, updated_data)

            # Update `filenames.csv`
            pd.DataFrame(existing_filenames).to_csv(filenames_path, index=False, header=False)

            print(f"Update completed! {len(new_files)} new files were processed and deleted.")

    else:
        print("Degree files not found. Processing all files from scratch...")

        # Create a matrix to store all data
        n_files = len(all_files)
        data = np.zeros((n_lines, n_files), dtype=np.int32)

        # Process all files in parallel (pass n_lines as argument)
        with Pool() as pool:
            results = pool.map(process_file, [(i, file, n_lines) for i, file in enumerate(all_files)])

        # Insert results into `data`
        filenames = []
        for i, degrees, filename in results:
            data[:, i] = degrees
            filenames.append(filename)  # Add file name to the list

        # Save dataset and file names
        np.save(degree_file_path, data)
        pd.DataFrame(filenames).to_csv(filenames_path, index=False, header=False)

        print(f"Processing completed! {n_files} files were processed, saved, and deleted.")
