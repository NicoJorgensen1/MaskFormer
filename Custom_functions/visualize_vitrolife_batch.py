import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
from tqdm import tqdm                                                               # Used to set a progress bar
from register_vitrolife_dataset import vitrolife_dataset_function
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.engine.defaults import DefaultPredictor
from custom_goto_trainer_class import custom_augmentation_mapper                    # A function that returns a custom mapper using data augmentation

# Move the figure to the wanted position when displaying
try:
    import pyautogui
    def move_figure_position(fig=plt.figure(), screensize=list(pyautogui.size()),   # Define a function to move a figure ...
                            dpi=100, position=[0.10, 0.09, 0.80, 0.75]):            # ... to a specified position on the screen
        fig = plt.figure(fig)                                                       # Make the wanted figure the current figure again
        # screensize[1] = np.round(np.divide(screensize[1], 1.075))                 # Reduce height resolution as the processbar in the bottom is part of the screen size
        screensize_inches = np.divide(screensize,dpi)                               # Convert the screensize into inches
        fig.set_figheight(position[3] * screensize_inches[1])                       # Set the wanted height of the figure
        fig.set_figwidth(position[2] * screensize_inches[0])                        # Set the wanted width of the figure
        figManager = plt.get_current_fig_manager()                                  # Get the current manager (i.e. window execution commands) of the current figure
        upper_left_corner_position = "+{:.0f}+{:.0f}".format(                       # Define a string with the upper left corner coordinates ...
            screensize[0]*position[0], screensize[1]*position[1])                   # ... which are read from the position inputs
        figManager.window.wm_geometry(upper_left_corner_position)                   # Move the figure to the upper left corner coordinates
        return fig                                                                  # Return the figure handle
except: pass


# Define function to apply a colormap on the images
def apply_colormap(mask, config):
    colors_used = list(MetadataCatalog[config.DATASETS.TEST[0]].stuff_colors)       # Read the colors used in the Metadatacatalog. If no colors are assigned, random colors are used
    if "vitrolife" in config.DATASETS.TEST[0].lower():                              # If we are working on the vitrolife dataset ...
        labels_used = list(MetadataCatalog[config.DATASETS.TEST[0]].stuff_dataset_id_to_contiguous_id.values()) # ... labels_used will be read from the MetadataCatalog
    else: labels_used = list(range(len(MetadataCatalog["ade20k_sem_seg_train"].stuff_classes))) # Else, labels is just 0:num_classes-1
    color_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)       # Allocate a RGB 3D array of zeros
    for label_idx, label in enumerate(labels_used):                                 # Loop through each label from the labels_used found from the MetadataCatalog
        color_array[mask == label] = colors_used[label_idx]                         # Assign all pixels in the mask with the current label_value the colors_used[idx] value
    return color_array                                                              # Return the colored mask


# Define a function to extract numbers from a string
def extractNumbersFromString(str, dtype=float, numbersWanted=1):
    try: vals = dtype(str)                                                          # At first, simply try to convert the string into the wanted dtype
    except:                                                                         # Else, if that is not possible ...
        vals = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", str)]            # Extract all the numbers from the string and put them in a list
        if len(vals) > 0:                                                           # If any numbers is found ...
            for kk in range(len(vals)):                                             # Loop through all the found numbers
                vals[kk] = dtype(vals[kk])                                          # Convert each of the found numbers into the wanted dtype
                if kk+1 == numbersWanted: break                                     # If we have convert all the numbers wanted, we'll stop the loop
            if numbersWanted < len(vals): vals = vals[:numbersWanted]               # Then we'll only use up to 'numbersWanted' found numbers
            if numbersWanted==1: vals = vals[0]                                     # If we only want 1 number, then we'll extract that from the list
        else: vals = np.nan                                                         # ... else if no numbers were found, return NaN
    return vals                                                                     # Return the wanted numbers, either as a type 'dtype' or, if multiple numbers, a list of 'dtypes'


# Define a function to put the latest saved model as the model_weights in the config before creating the dataloader
def putModelWeights(config, delete_remaining=False):
    model_files = [x for x in os.listdir(config.OUTPUT_DIR) if "model" in x.lower() and x.endswith(".pth") and not np.isnan(extractNumbersFromString(x))]   # Find all saved model checkpoints
    if len(model_files) >= 1:                                                       # If any model checkpoint is found, 
        iteration_numbers = [extractNumbersFromString(x, int) for x in model_files] # Find the iteration numbers for when they were saved
        latest_iteration_idx = np.argmax(iteration_numbers)                         # Find the index of the model checkpoint with the latest iteration number
        config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, model_files[latest_iteration_idx])   # Assign the latest model checkpoint to the config
        if delete_remaining==True:                                                  # If the user chose to delete all other but the final model, ... 
            for model_file in model_files:                                          # ... loop through all found model checkpoint files
                if os.path.join(config.OUTPUT_DIR,model_file) != config.MODEL.WEIGHTS:  # If the current model_file is not the newest checkpoint file ...
                    os.remove(os.path.join(config.OUTPUT_DIR,model_file))           # ... remove the current model_file
    return config                                                                   # Return the updated config


# Define a function to predict some label-masks for the dataset
def create_batch_img_ytrue_ypred(config, data_split, FLAGS, data_batch=None):
    config = putModelWeights(config)                                                # Change the config and append the latest model as the used checkpoint
    predictor = DefaultPredictor(cfg=config)                                        # Create a default predictor instance
    Softmax_module = nn.Softmax(dim=2)                                              # Create a module to compute the softmax value along the final, channel, dimension of the predicted images
    if data_batch == None:                                                          # If no batch with data was send to the function ...
        if "vitrolife" in FLAGS.dataset_name.lower():                               # ... and if we are using the vitrolife dataset
            dataset_dicts = vitrolife_dataset_function(data_split, debugging=True)  # ... the list of dataset_dicts from vitrolife is computed.
            dataset_dicts = dataset_dicts[:FLAGS.num_images]                        # We'll maximally show the first FLAGS.num_images images
        else: dataset_dicts = DatasetCatalog.get("ade20k_sem_seg_{:s}".format(data_split))  # Else we use the ADE20K dataset
        config = putModelWeights(config)                                            # Add the newest model weights to the configuration
        if "train" in data_split: data_mapper = custom_augmentation_mapper(config=config, is_train=True)    # Use the custom data augmentation mapper for training images
        else: data_mapper = DatasetMapper(config, is_train=False)                   # Use the regular, default mapper for val+test images
        dataloader = build_detection_train_loader(dataset_dicts, mapper=data_mapper, total_batch_size=np.min([FLAGS.num_images, len(dataset_dicts)]))   # Create the dataloader
        data_batch = next(iter(dataloader))                                         # Extract the next batch from the dataloader
    img_ytrue_ypred = {"input": list(), "y_pred": list(), "y_true": list(), "PN": list()}   # Initiate a dictionary to store the input images, ground truth masks and the predicted masks
    for data in data_batch:                                                         # Iterate over each data sample in the batch from the dataloader
        img = torch.permute(data["image"], (1,2,0)).numpy()                         # Input image [H,W,C]
        y_true = data["sem_seg"].numpy()                                            # Ground truth label mask [H,W]
        y_true_col = apply_colormap(mask=y_true, config=config)                     # Ground truth color mask
        out = predictor.__call__(img)                                               # Predicted output dictionary. The call function needs images in BGR format.
        out_img = torch.permute(out["sem_seg"], (1,2,0))                            # Predicted output image [H,W,C]
        out_img_softmax = Softmax_module(out_img)                                   # Softmax of predicted output image
        y_pred = torch.argmax(out_img_softmax,2).cpu()                              # Predicted output image [H,W]
        y_pred_col = apply_colormap(mask=y_pred, config=config)                     # Predicted colormap for predicted output image
        # Append the input image, y_true and y_pred to the dictionary
        img_ytrue_ypred["input"].append(img)                                        # Append the input image to the dictionary
        img_ytrue_ypred["y_true"].append(y_true_col)                                # Append the ground truth to the dictionary
        img_ytrue_ypred["y_pred"].append(y_pred_col)                                # Append the predicted mask to the dictionary
        if "vitrolife" in FLAGS.dataset_name.lower():                               # If we are visualizing the vitrolife dataset
            img_ytrue_ypred["PN"].append(int(data["image_custom_info"]["PN_image"]))# Read the true number of PN on the current image
    return img_ytrue_ypred, data_batch, FLAGS


# Function to create directories in which the visualization results are saved
def get_save_dirs(config, dataset_split):
    for data_split in ["train", "val", "test"]:
        if "vitrolife" not in config.DATASETS.TRAIN[0].lower() and data_split=="test": continue     # There are no test dataset for the ade20k
        os.makedirs(os.path.join(config.OUTPUT_DIR, "Visualizations", data_split), exist_ok=True)   # Create a folder to store the segmentations of the images
    return os.path.join(config.OUTPUT_DIR, "Visualizations", dataset_split)                         # Return the folder name of the current dataset split


# Function to sort the dictionaries by the number of PN's found
def sort_dictionary_by_PN(data):
    PNs_idx = np.argsort(data["PN"])
    new_data = {}
    for key in data:
        new_data[key] = [data[key][x] for x in PNs_idx]
    return new_data


# Define function to plot the images
def visualize_the_images(config, FLAGS, position=[0.55, 0.08, 0.40, 0.75], epoch_num=None, data_batches=None, model_done_training=False):
    # Get the datasplit and number of images to show
    fig_list, data_batches_final = list(), list()                                   # Initiate the list to store the figures in
    if data_batches==None: data_batches = [None, None, None]                        # If no previous data has been sent, it must be a list of None's...
    data_split_count = 1
    data_split = "train"
    for data_split, data_batch in tqdm(zip(["train", "val", "test"], data_batches), # Iterate through the three splits available
                leave=True, unit="Data_split", total=3, ascii=True,  desc="Dataset split {:d}/{:d}".format(data_split_count, 3),
                bar_format="{desc}  | {percentage:3.0f}% | {bar:35}| {n_fmt}/{total_fmt} [Spent: {elapsed}. Remaining: {remaining}{postfix}]"):      
        data_split_count += 1
        if "vitrolife" not in FLAGS.dataset_name.lower() and data_split=="test": continue   # Only vitrolife has a test dataset. ADE20K doesn't. 
        # Extract information about the dataset used
        img_ytrue_ypred, data_batch, FLAGS = create_batch_img_ytrue_ypred(config=config,    # Create the batch of images that needs to be visualized ...
                data_split=data_split, FLAGS=FLAGS, data_batch=data_batch)                  # ... and return the images in the data_batch dictionary
        if "vitrolife" in FLAGS.dataset_name.lower():                               # If we are working on the vitrolife dataset sort the ...
            data_batch = sorted(data_batch, key=lambda x: x["image_custom_info"]["PN_image"])   # ... data_batch after the number of PN per found image
            img_ytrue_ypred = sort_dictionary_by_PN(data=img_ytrue_ypred)           # And then also sort the data dictionary
        num_rows, num_cols = 3, len(data_batch)                                     # The figure will have three rows (input, y_pred, y_true) and one column per image
        fig = plt.figure(figsize=(int(np.ceil(len(data_batch)*5.5)), 10))           # Create the figure object
        row = 0                                                                     # Initiate the row index counter (all manual indexing could have been avoided by having created img_ytrue_ypred as an OrderedDict)
        for key in img_ytrue_ypred.keys():                                          # Loop through all the keys in the batch dictionary
            if key.lower() not in ['input', 'y_true', 'y_pred']: continue           # If the key is not one of (input, y_pred, y_true), we simply skip to the next one
            for col, img in enumerate(img_ytrue_ypred[key]):                        # Loop through all available images in the dictionary
                plt.subplot(num_rows, num_cols, row*num_cols+col+1)                 # Create the subplot instance
                plt.axis("off")                                                     # Remove axis tickers
                if "vitrolife" in FLAGS.dataset_name.lower():                       # If we are visualizing the vitrolife dataset
                    plt.title("{:s} with {:.0f} PN".format(key, img_ytrue_ypred["PN"][col]))# Create the title for the plot with the number of PN
                else: plt.title("{:s}".format(key))                                 # Otherwise simply put the key, i.e. either input, y_pred or y_true.
                plt.imshow(img, cmap="gray")                                        # Display the image
            row += 1                                                                # Increase the row counter by 1
        try: fig = move_figure_position(fig=fig, position=position)                 # Try and move the figure to the wanted position (only possible on home computer with a display)
        except: pass                                                                # Except, simply just let the figure retain the current position
        fig.tight_layout()                                                          # Assures the subplots are plotted tight around each other
        fig_name_init = "Segmented_{:s}_data_samples_from_".format(data_split)      # Initialize the figure name
        if epoch_num != None: fig_name = "{:s}epoch_{:d}.jpg".format(fig_name_init, epoch_num)                              # If an epoch number has been specified, the figure name will contain that
        else: fig_name = "{:s}{:s}_training.jpg".format(fig_name_init, "after" if model_done_training else "before")        # Otherwise the visualization happens before/after training
        fig.savefig(os.path.join(get_save_dirs(config=config, dataset_split=data_split), fig_name), bbox_inches="tight")    # Save the figure in the correct output directory
        fig_list.append(fig)                                                        # Append the current figure to the list of figures
        data_batches_final.append(data_batch)                                       # Append the current data_batch to the list of data_batches
        fig.show() if FLAGS.display_images==True else plt.close(fig)                # Display the figure if that is the chosen option
    return fig_list, data_batches_final, putModelWeights(config), FLAGS             # Return the figure, the dictionary with the used images and the updated config with a new model checkpoint


