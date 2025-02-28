import shutil
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for progress bars

# Path to the parent_parent directory containing multiple parent folders
parent_parent_folder_path = "results/raw"

# Get the list of parent folders
parent_folders = [f for f in os.listdir(parent_parent_folder_path) if os.path.isdir(os.path.join(parent_parent_folder_path, f))]

# Wrap the parent folders list with tqdm to show a high-level progress bar
for parent_folder_name in tqdm(parent_folders, desc="Processing Parent Folders", unit="parent-folder"):
    parent_folder_path = os.path.join(parent_parent_folder_path, parent_folder_name)
    
    # Get the list of subfolders inside the current parent folder
    subfolders = [f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))]
    
    # Wrap the subfolders list with tqdm for subfolder-level progress
    for folder_name in tqdm(subfolders, desc=f"Processing Subfolders in {parent_folder_name}", leave=False, unit="folder"):
        folder_path = os.path.join(parent_folder_path, folder_name)
        
        # Loop through all .npy files in the current subfolder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(folder_path, file_name)
                
                # Load the .npy file
                data = np.load(file_path)
                
                # Save as compressed .npz
                compressed_file_path = file_path.replace('.npy', '.npz')
                np.savez_compressed(compressed_file_path, data=data)
                
                # Optional: Remove the original .npy to save space
                os.remove(file_path)
        
        # Compress the current folder as .zip
        zip_output_path = os.path.join(parent_folder_path, f"{folder_name}.zip")
        shutil.make_archive(zip_output_path.replace('.zip', ''), 'zip', folder_path)

        # Remove the original folder to save space
        shutil.rmtree(folder_path)