# Import libraries
import os                                                                   # Used to navigate the folder structure in the current os
import numpy as np                                                          # Used for computing the iterations per epoch
import argparse                                                             # Used to parse input arguments through command line
import pickle                                                               # Used to save the history dictionary after training
from datetime import datetime                                               # Used to get the current date and time when starting the process
from shutil import make_archive                                             # Used to zip the directory of the output folder
from detectron2.data import DatasetCatalog, MetadataCatalog                 # Catalogs over registered datasets and metadatas for all datasets available in Detectron2
from detectron2.engine import default_argument_parser                       # Default argument_parser object
from GPU_memory_ranked_assigning import assign_free_gpus                    # Function to assign the script to a given number of GPUs
from register_vitrolife_dataset import register_vitrolife_data_and_metadata_func        # Register the vitrolife dataset and metadata in the Detectron2 dataset catalog
from create_custom_config import createVitrolifeConfiguration, changeConfig_withFLAGS   # Function to create a configuration used for training 


# Define function to log information about the dataset
def printAndLog(input_to_write, logs, print_str=True, write_input_to_log=True, prefix="\n", postfix="", oneline=True, length=None):
    mode = "a" if os.path.isfile(logs) else "w"                             # Whether or not we are appending to the logfile or creating a new logfile
    logs = open(file=logs, mode=mode)                                       # Open the logfile, i.e. making it writable
    
    if isinstance(input_to_write, str):                                     # If the input is of type string ...
        if print_str: print(input_to_write)                                 # ... and it needs to be printed, print it ...
        if write_input_to_log: logs.writelines("{:s}{:s}{:s}".format(prefix, input_to_write, postfix))  # ... if it needs to be logged, write it to log
    if isinstance(input_to_write, list):                                    # If the input is a list ...
        if print_str:                                                       # ... and the list needs to be printed ...
            for string in input_to_write: print(string, end="\t" if oneline else "\n")  # ... print every item in the list
        if write_input_to_log:                                              # ... if the input needs to be logged ...
            if write_input_to_log: logs.writelines("{:s}".format(prefix))   # ... the prefix is written ...
            for string in input_to_write: logs.writelines("{}\n".format(string))    # ... and then each item in the list is written on a new line ...
            logs.writelines("{:s}".format(postfix))                         # ... and then the postfix is written
    if isinstance(input_to_write, dict):                                    # If the input is a dictionary ...
        if write_input_to_log: logs.writelines("{:s}".format(prefix))       # ... and the input must be logged, the prefix is written ...
        if write_input_to_log or print_str:                                 # ... or if the input has to be either logged or printed ...
            for key, val in input_to_write.items():                         # ... and then we loop over all key-value pairs in the dictionary ...
                if isinstance(val, float): val = "{:.4f}".format(val)       # If the value is a float value, round it to four decimals
                else: val = str(val)                                        # Else, simply convert the value to a string
                if val == "": val = "None"                                  # If the value is empty, let it be "None"
                key = key + ": "                                            # Add the colon to the key
                if length != None: key = key.ljust(length)                  # If any length has been chosen by the user, pad the key-name string to that specific length
                string = "{:s}{:s}{:s}".format(key, val.ljust(9), "||".ljust(5) if oneline else "\n")   # Create the string to print and log
                if write_input_to_log: logs.writelines(string)              # ... If the string needs to be logged, log the key-value pairs on the same line ...
                if print_str: print(string, end="\t" if oneline else "")    # ... and print it if chosen ...
        if write_input_to_log: logs.writelines("{:s}".format(postfix))      # ... before writing the postfix 
    
    # Close the logfile again
    logs.close()                                                            # Close the logfile again


# Function to rename the automatically created "inference" directory in the OUTPUT_DIR from "inference" to "validation" before performing actual inference with the test set
def rename_output_inference_folder(config):                                 # Define a function that will only take the config as input
    source_folder = os.path.join(config.OUTPUT_DIR, "inference")            # The source folder is the current inference (i.e. validation) directory
    dest_folder = os.path.join(config.OUTPUT_DIR, "validation")             # The destination folder is in the same parent-directory where inference is changed with validation
    os.rename(source_folder, dest_folder)                                   # Perform the renaming of the folder

# Function to zip the output folder after training has completed
def zip_output(cfg):
    print("\nZipping the output directory {:s} with {:.0f} files".format(os.path.basename(cfg.OUTPUT_DIR), len(os.listdir(cfg.OUTPUT_DIR))))
    make_archive(base_name=os.path.basename(cfg.OUTPUT_DIR), format="zip",  # Instantiate the zipping of the output directory  where the resulting zip file will ...
        root_dir=os.path.dirname(cfg.OUTPUT_DIR), base_dir=os.path.basename(cfg.OUTPUT_DIR))    # ... include the output folder (not just the files from the folder)

# Define a function to convert string values into booleans
def str2bool(v):
    if isinstance(v, bool): return v                                        # If the input argument is already boolean, the given input is returned as-is
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True             # If any signs of the user saying yes is present, the boolean value True is returned
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False          # If any signs of the user saying no is present, the boolean value False is returned
    else: raise argparse.ArgumentTypeError('Boolean value expected.')       # If the user gave another input an error is raised


# Alter the FLAGS input arguments
def changeFLAGS(FLAGS):
    if FLAGS.num_gpus != FLAGS.gpus_used: FLAGS.num_gpus = FLAGS.gpus_used  # As there are two input arguments where the number of GPUs can be assigned, the gpus_used argument is superior
    if "vitrolife" in FLAGS.dataset_name.lower() : FLAGS.num_gpus = 1       # Working with the Vitrolife dataset can only be done using a single GPU for some weird reason...
    if FLAGS.eval_only != FLAGS.inference_only: FLAGS.eval_only = FLAGS.inference_only  # As there are two inputs where "eval_only" can be set, inference_only is the superior
    if FLAGS.min_delta < 1.0: FLAGS.min_delta *= 100                        # As the model outputs metrics multiplied by a factor of 100, the min_delta value must also be scaled accordingly
    if FLAGS.debugging: FLAGS.eval_metric.replace("val", "train")           # The metric used for evaluation will be a training metric, if we are debugging the model
    if FLAGS.use_per_pixel_baseline: FLAGS.resnet_depth = 50                # The per-pixel baseline can only be used with a ResNet depth of 50 
    if FLAGS.inference_only: FLAGS.num_epochs = 1                           # If we are only using inference, then we'll only run through one epoch
    FLAGS.num_queries = 100 if "ade20k" in FLAGS.dataset_name.lower() else 25   # The number of queries will be set to 100 with the ADE20K dataset and 25 with the Vitrolife dataset
    return FLAGS

# Define a function to extract the final results that will be printed in the log file
def getBestEpochResults(history, best_epoch):
    val_to_keep = {}
    best_epoch_idx = np.max(np.argwhere(np.isin(history["val_epoch_num"], best_epoch)))
    keys_to_use = sorted([x for x in history.keys() if all(["val" in x, "_C" not in x, "_"!=x[-3], "_"!=x[-2]]) and any([y in x for y in ["SQ", "RQ", "PQ", "IoU", "ACC", "loss"]])], key=str.lower)
    for key in keys_to_use:
        if "loss" in key: val_to_keep[key] = history[key][best_epoch_idx]
        if "loss" not in key: val_to_keep[key] = history[key][best_epoch-1]
    return val_to_keep

# Save history dictionary
def SaveHistory(historyObject, save_folder, historyName="history"):         # Function to save the dict history in the specified folder 
    hist_file = open(os.path.join(save_folder, historyName+".pkl"), "wb")   # Opens a pickle for saving the history dictionary 
    pickle.dump(historyObject, hist_file)                                   # Saves the history dictionary 
    hist_file.close()                                                       # Close the pickle again  


# Running the parser function. By doing it like this the FLAGS will get out of the main function
parser = default_argument_parser()
start_time = datetime.now().strftime("%H_%M_%d%b%Y").upper()
parser.add_argument("--dataset_name", type=str, default="vitrolife", help="Which datasets to train on. Choose between [ADE20K, Vitrolife]. Default: Vitrolife")
parser.add_argument("--output_dir_postfix", type=str, default=start_time, help="Filename extension to add to the output directory of the current process. Default: now: 'HH_MM_DDMMMYYYY'")
parser.add_argument("--eval_metric", type=str, default="val_fwIoU", help="Metric to use in order to determine the 'best' model weights. Available: val_/train_ prefix to [total_loss, mIoU, fwIoU, mACC, PQ, RQ, SQ]. Default: val_fwIoU")
parser.add_argument("--optimizer_used", type=str, default="ADAMW", help="Optimizer to use. Available [SGD, ADAMW]. Default: ADAMW")
parser.add_argument("--num_workers", type=int, default=6, help="Number of workers to use for training the model. Default: 4")
parser.add_argument("--max_iter", type=int, default=int(1e5), help="Maximum number of iterations to train the model for. <<Deprecated argument. Use 'num_epochs' instead>>. Default: 100000")
parser.add_argument("--img_size_min", type=int, default=500, help="The length of the smallest size of the training images. Default: 500")
parser.add_argument("--img_size_max", type=int, default=500, help="The length of the largest size of the training images. Default: 500")
parser.add_argument("--resnet_depth", type=int, default=50, help="The depth of the feature extracting ResNet backbone. Possible values: [18,34,50,101] Default: 50")
parser.add_argument("--batch_size", type=int, default=1, help="The batch size used for training the model. Default: 1")
parser.add_argument("--num_images", type=int, default=6, help="The number of images to display/segment. Default: 6")
parser.add_argument("--display_rate", type=int, default=1, help="The epoch_rate of how often to display image segmentations. A display_rate of 3 means that every third epoch, visual segmentations are saved. Default: 25")
parser.add_argument("--gpus_used", type=int, default=1, help="The number of GPU's to use for training. Only applicable for training with ADE20K. This input argument deprecates the '--num-gpus' argument. Default: 1")
parser.add_argument("--num_epochs", type=int, default=3, help="The number of epochs to train the model for. Default: 1")
parser.add_argument("--warm_up_epochs", type=int, default=2, help="The number of epochs to warm up the learning rate when training. Will go from 1/100 '--learning_rate' to '--learning_rate' during these warm_up_epochs. Default: 1")
parser.add_argument("--patience", type=int, default=3, help="The number of epochs to accept that the model hasn't improved before lowering the learning rate by a factor '--lr_gamma'. Default: 6")
parser.add_argument("--early_stop_patience", type=int, default=9, help="The number of epochs to accept that the model hasn't improved before terminating training. Default: 15")
parser.add_argument("--dice_loss_weight", type=int, default=10, help="The weighting for the dice loss in the loss function. Default: 10")
parser.add_argument("--mask_loss_weight", type=int, default=20, help="The weighting for the mask loss in the loss function. Default: 20")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="The initial learning rate used for training the model. Default: 1e-3")
parser.add_argument("--lr_gamma", type=float, default=0.15, help="The update factor for the learning rate when the model performance hasn't improved in 'patience' epochs. Will do new_lr=old_lr*lr_gamma. Default 0.15")
parser.add_argument("--backbone_multiplier", type=float, default=0.25, help="The multiplier for the backbone learning rate. Backbone_lr = learning_rate * backbone_multiplier. Default: 0.20")
parser.add_argument("--dropout", type=float, default=0.10, help="The dropout rate used for the linear layers. Default: 0.10")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay used for the model. Default: 1e-4")
parser.add_argument("--min_delta", type=float, default=5e-4, help="The minimum improvement the model must have made in order to be accepted as an actual improvement. Default 5e-4")
parser.add_argument("--crop_enabled", type=str2bool, default=True, help="Whether or not cropping is allowed on the images. Default: True")
parser.add_argument("--hp_optim", type=str2bool, default=True, help="Whether or not we are initiating the training with a hyperparameter optimization. Default: True")
parser.add_argument("--inference_only", type=str2bool, default=False, help="Whether or not training is skipped and only inference is run. This input argument deprecates the '--eval_only' argument. Default: False")
parser.add_argument("--display_images", type=str2bool, default=False, help="Whether or not some random sample images are displayed before training starts. Default: False")
parser.add_argument("--use_checkpoint", type=str2bool, default=False, help="Whether or not we are loading weights from a model checkpoint file before training. Default: False")
parser.add_argument("--use_transformer_backbone", type=str2bool, default=False, help="Whether or now we are using the extended swin_small_transformer backbone. Only applicable if '--use_per_pixel_baseline'=False. Default: False")
parser.add_argument("--use_per_pixel_baseline", type=str2bool, default=False, help="Whether or now we are using the per_pixel_calculating head. Alternative is the MaskFormer (or transformer) heads. Default: True")
parser.add_argument("--debugging", type=str2bool, default=False, help="Whether or not we are debugging the script. Default: False")
# Parse the arguments into a Namespace variable
FLAGS = parser.parse_args()
FLAGS = changeFLAGS(FLAGS)


# Setup functions
assign_free_gpus(max_gpus=FLAGS.num_gpus)                                   # Assigning the running script to the selected amount of GPU's with the largest memory available
if "vitrolife" in FLAGS.dataset_name.lower():                               # If we want to work with the Vitrolife dataset ...
    register_vitrolife_data_and_metadata_func(debugging=FLAGS.debugging)    # ... register the vitrolife dataset
else:                                                                       # Otherwise, if we are working on the ade20k dataset ...
    for split in ["train", "val"]:                                          # ... then we will find the training and the validation set
        MetadataCatalog["ade20k_sem_seg_{:s}".format(split)].num_files_in_dataset = len(DatasetCatalog["ade20k_sem_seg_{:s}".format(split)]())  # ... and create a key-value pair telling the number of files in the dataset

# Create the initial configuration, define FLAGS epoch variables and alter the configuration
cfg = createVitrolifeConfiguration(FLAGS=FLAGS)                             # Create the custom configuration used to e.g. build the model
FLAGS.num_train_files = MetadataCatalog[cfg.DATASETS.TRAIN[0]].num_files_in_dataset # Write the number of training files to the FLAGS namespace
FLAGS.num_val_files = MetadataCatalog[cfg.DATASETS.TEST[0]].num_files_in_dataset    # Write the number of validation files to the FLAGS namespace
if FLAGS.batch_size > FLAGS.num_train_files: FLAGS.batch_size = FLAGS.num_train_files   # The batch size can't be greater than the number of files in the dataset
FLAGS.epoch_iter = int(np.floor(np.divide(FLAGS.num_train_files, FLAGS.batch_size)))# Compute the number of iterations per training epoch
FLAGS.num_classes = len(MetadataCatalog[cfg.DATASETS.TRAIN[0]].stuff_classes)       # Get the number of classes in the current dataset
FLAGS.HPO_trial = 1                                                         # A counter for the number of trials of hyperparameter optimization performed 
cfg = changeConfig_withFLAGS(cfg=cfg, FLAGS=FLAGS)                          # Set the final values for the config


# Create the log file
log_file = os.path.join(cfg.OUTPUT_DIR, "Training_logs.txt")                # Initiate the log filename
if os.path.exists(log_file): os.remove(log_file)                            # Remove an earlier logfile if that already exists (which it shouldn't)

# Initiate the log file
start_time_readable = "{:s}:{:s} {:s}-{:s}-{:s}".format(start_time[:2], start_time[3:5], start_time[6:8], start_time[8:11], start_time[11:])
printAndLog(input_to_write="Training initiated at {:s}".format(start_time_readable).upper(), logs=log_file, prefix="", postfix="\n")
printAndLog(input_to_write="FLAGS input arguments:", logs=log_file)
printAndLog(input_to_write=vars(FLAGS), logs=log_file, oneline=False, length=27)


# Return the values again
def setup_func(): return FLAGS, cfg, log_file

