import os
import glob
import pandas as pd
import numpy as np
from natsort import natsorted
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog


# Function to select sample dictionaries with unique PN's
def pickSamplesWithUniquePN(dataset_dict):
    PNs_found = np.zeros((1,10), dtype=bool)                                                # Create a [1,10] list filled with False values to track if a sample with a specified PN number has been found
    data_used = []                                                                          # Initiate a new list of dictionaries
    for data in dataset_dict:                                                               # Iterate over all dictionaries in the list of dictionaries
        PN = int(data["image_custom_info"]["PN_image"])                                     # Get the number of PN's in the current sample
        if PNs_found[0,PN] == False:                                                        # If no sample with the current PN_number has been found ...
            PNs_found[0,PN] = True                                                          # ... the corresponding entry in the PNs_found array are set to true ...
            data_used.append(data)                                                          # ... and the data_used array is appended with the current sample
    data_used = sorted(data_used, key=lambda x: x["image_custom_info"]["PN_image"])         # Sort the data_used list by the number of dictionaries
    return data_used



# Define the function to return the list of dictionaries with information regarding all images available in the vitrolife dataset
def vitrolife_dataset_function(run_mode="train", debugging=False):
    # Find the folder containing the vitrolife dataset  
    vitrolife_dataset_filepath = os.path.join(os.getenv("DETECTRON2_DATASETS"), "Vitrolife_dataset")    # Get the path to the vitrolife dataset
    
    # Find the metadata file
    metadata_file = os.path.join(vitrolife_dataset_filepath, "metadata.csv")                # Get the csv file with the metadata for all images
    df_data = pd.read_csv(metadata_file)                                                    # Read the csv file 
    df_data = df_data.set_index(["HashKey","Well"])                                         # Set the two columns HashKey and Well as index columns

    # Create the list of dictionaries with information about all images
    img_mask_pair_list = []                                                                 # Initiate the list to store the information about all images
    total_files = len(os.listdir(os.path.join(vitrolife_dataset_filepath, "raw_images")))   # List all the image files in the raw_images directory
    iteration_counter = 0                                                                   # Initiate a iteration counter 
    count = 0                                                                               # Initiate a counter to count the number of images inserted to the dataset
    for img_filename in tqdm(os.listdir(os.path.join(vitrolife_dataset_filepath, "raw_images")),    # Loop through all files in the raw_images folder
            total=total_files, unit="img", postfix="Read the Vitrolife {:s} dataset dictionaries".format(run_mode), leave=True,
            bar_format="{desc}  | {percentage:3.0f}% | {bar:45}| {n_fmt}/{total_fmt} | [Spent: {elapsed}. Remaining: {remaining} | {postfix}]"):  
        iteration_counter += 1                                                              # Increase the counter that counts the number of iterations in the for-loop
        img_filename_wo_ext = os.path.splitext(os.path.basename(img_filename))[0]           # Get the image filename without .jpg extension
        img_filename_wo_ext_parts = img_filename_wo_ext.split("_")                          # Split the filename where the _ is
        hashkey = img_filename_wo_ext_parts[0]                                              # Extract the hashkey from the filename
        well = int(img_filename_wo_ext_parts[1][1:])                                        # Extract the well from the filename
        row = deepcopy(df_data.loc[hashkey,well])                                           # Find the row of the corresponding file in the dataframe
        data_split = row["split"]                                                           # Find the split for the current image, i.e. either train, val or test
        if data_split != run_mode: continue                                                 # If the current image is supposed to be in another split, then continue to the next image
        mask_filename = glob.glob(os.path.join(vitrolife_dataset_filepath, 'masks', img_filename_wo_ext + '*')) # Find the corresponding mask filename
        if len(mask_filename) != 1: continue                                                # Continue only if we find only one mask filename
        mask_filename = os.path.basename(mask_filename[0])                                  # Extract the mask filename from the list
        row["img_file"] = os.path.join(vitrolife_dataset_filepath, "raw_images", img_filename)  # Add the current filename for the input image to the row-variable
        row["mask_file"] = os.path.join(vitrolife_dataset_filepath, "masks", mask_filename) # Add the current filename for the semantic segmentation ground truth mask to the row-variable
        mask = np.asarray(Image.open(row["mask_file"]))                                     # Read the ground truth label mask image
        if len(np.unique(mask)) <= 1: continue                                              # Apparently a test mask had only 0's, even though the corresponding input image was ordinary
        width_img, height_img = Image.open(row["img_file"]).size                            # Get the image size of the img_file
        width_mask, height_mask = mask.shape                                                # Get the size of the ground truth mask
        if not all([width_img==width_mask, height_img==height_mask]): continue              # The image and mask have to be of the same size
        current_pair = {"file_name": row["img_file"],                                       # Initiate the dict of the current image with the full filepath + filename
                        "height": height_img,                                               # Write the image height
                        "width": width_img,                                                 # Write the image width
                        "image_id": img_filename_wo_ext,                                    # A unique key for the current image
                        "sem_seg_file_name": row["mask_file"],                              # The full filepath + filename for the mask ground truth label image
                        "image_custom_info": row}                                           # Add all the info from the current row to the dataset
        img_mask_pair_list.append(current_pair)                                             # Append the dictionary for the current pair to the list of images for the given dataset
        count += 1                                                                          # Increase the sample counter 
        if count > 15: break
    assert len(img_mask_pair_list) >= 1, print("No image/mask pairs found in {:s} subfolders 'raw_image' and 'masks'".format(vitrolife_dataset_filepath))
    img_mask_pair_list = natsorted(img_mask_pair_list)                                      # Sorting the list assures the same every time this function runs
    if debugging==True: img_mask_pair_list=pickSamplesWithUniquePN(img_mask_pair_list)      # If we are debugging, we'll only get one sample with each number of PN's 
    return img_mask_pair_list                                                               # Return the found list of dictionaries

# Function to register the dataset and the meta dataset for each of the three splitshuffleshuffles, [train, val, test]
def register_vitrolife_data_and_metadata_func(debugging=False):
    class_labels = ["Background", "Well", "Zona", "Perivitelline space", "Cell", "PN"]      # Write the class labels
    stuff_id = {ii: ii for ii in range(len(class_labels))}                                  # Create a dictionary with the class_id's as both keys and values
    stuff_colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]     # Set random colors for when the images will be visualized
    for split_mode in ["train", "val", "test"]:                                             # Iterate over the three dataset splits ... 
        DatasetCatalog.register("vitrolife_dataset_{:s}".format(split_mode), lambda split_mode=split_mode: vitrolife_dataset_function(run_mode=split_mode, debugging=debugging))    # Register the dataset
        MetadataCatalog.get("vitrolife_dataset_{:s}".format(split_mode)).set(stuff_classes=class_labels,                    # Set the metadata stuff_class names
                                                                            stuff_colors = stuff_colors,                    # Set the metadata stuff_colors for visualization
                                                                            stuff_dataset_id_to_contiguous_id = stuff_id,   # Set the metadata class indices
                                                                            # ignore_label=0,                               # The model won't be rewarded by predicting background pixels
                                                                            ignore_label=255,                               # No labels will be ignored as 255 >> num_classes ...
                                                                            evaluator_type="sem_seg",                       # Choose the type of evaluator for this dataset
                                                                            num_files_in_dataset=len(DatasetCatalog["vitrolife_dataset_{:}".format(split_mode)]())) # Write the length of the dataset
    assert any(["vitrolife" in x for x in list(MetadataCatalog)]), "Datasets have not been registered correctly"    # Assuring the dataset has been registered correctly

# Test that the function will actually return a list of dicts
# img_mask_list_train = vitrolife_dataset_function(run_mode="train")
# img_mask_list_val = vitrolife_dataset_function(run_mode="val")
# img_mask_list_test = vitrolife_dataset_function(run_mode="test")


