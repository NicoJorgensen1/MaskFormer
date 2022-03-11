# Import important libraries
import os                                                                   # Used to navigate the folder structure in the current os
import argparse                                                             # Used to parse input arguments through command line
from datetime import datetime                                               # Used to get the current date and time when starting the process
from detectron2.engine import default_argument_parser                       # Default argument_parser object
from custom_setup import main_setup                                         # To setup the system

# Function to rename the automatically created "inference" directory in the OUTPUT_DIR from "inference" to "validation" before performing actual inference with the test set
def rename_output_inference_folder(config):                                 # Define a function that will only take the config as input
    source_folder = os.path.join(config.OUTPUT_DIR, "inference")            # The source folder is the current inference (i.e. validation) directory
    dest_folder = os.path.join(config.OUTPUT_DIR, "validation")             # The destination folder is in the same parent-directory where inference is changed with validation
    os.rename(source_folder, dest_folder)                                   # Perform the renaming of the folder

# Define a function to convert string values into booleans
def str2bool(v):
    if isinstance(v, bool): return v                                        # If the input argument is already boolean, the given input is returned as-is
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True             # If any signs of the user saying yes is present, the boolean value True is returned
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False          # If any signs of the user saying no is present, the boolean value False is returned
    else: raise argparse.ArgumentTypeError('Boolean value expected.')       # If the user gave another input an error is raised

# Running the main function. By doing it like this the FLAGS will get out of the main function
if __name__ == "__main__":
    # Create the input arguments with possible values
    parser = default_argument_parser()
    start_time = datetime.now().strftime("%H_%M_%d%b%Y").upper()
    parser.add_argument("--dataset_name", type=str, default="vitrolife", help="Which datasets to train on. Choose between [ADE20K, Vitrolife]. Default: Vitrolife")
    parser.add_argument("--output_dir_postfix", type=str, default=start_time, help="Filename extension to add to the output directory of the current process. Default: now: 'HH_MM_DDMMMYYYY'")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for training the model. Default: 1")
    parser.add_argument("--max_iter", type=int, default=int(3e1), help="Maximum number of iterations to train the model for. Default: 100")
    parser.add_argument("--img_size_min", type=int, default=500, help="The length of the smallest size of the training images. Default: 500")
    parser.add_argument("--img_size_max", type=int, default=500, help="The length of the largest size of the training images. Default: 500")
    parser.add_argument("--resnet_depth", type=int, default=50, help="The depth of the feature extracting ResNet backbone. Possible values: [18,34,50,101] Default: 50")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size used for training the model. Default: 1")
    parser.add_argument("--num_images", type=int, default=5, help="The number of images to display. Only relevant if --display_images is true. Default: 5")
    parser.add_argument("--gpus_used", type=int, default=1, help="The number of GPU's to use for training. Only applicable for training with ADE20K. This input argument deprecates the '--num-gpus' argument. Default: 1")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="The initial learning rate used for training the model. Default 1e-4")
    parser.add_argument("--crop_enabled", type=str2bool, default=False, help="Whether or not cropping is allowed on the images. Default: False")
    parser.add_argument("--inference_only", type=str2bool, default=False, help="Whether or not training is skipped and only inference is run. This input argument deprecates the '--eval_only' argument. Default: False")
    parser.add_argument("--display_images", type=str2bool, default=True, help="Whether or not some random sample images are displayed before training starts. Default: False")
    parser.add_argument("--use_checkpoint", type=str2bool, default=True, help="Whether or not we are loading weights from a model checkpoint file before training. Only applicable when using ADE20K dataset. Default: False")
    parser.add_argument("--use_transformer_backbone", type=str2bool, default=True, help="Whether or now we are using the extended swin_small_transformer backbone. Only applicable if '--use_per_pixel_baseline'=False. Default: False")
    parser.add_argument("--use_per_pixel_baseline", type=str2bool, default=False, help="Whether or now we are using the per_pixel_calculating head. Alternative is the MaskFormer (or transformer) heads. Default: False")
    parser.add_argument("--debugging", type=str2bool, default=True, help="Whether or not we are debugging the script. Default: False")
    # Parse the arguments into a Namespace variable
    FLAGS = parser.parse_args()
    FLAGS = main_setup(FLAGS)

# Import the other custom libraries
from shutil import make_archive                                             # Used to zip the directory of the output folder
from detectron2.data import DatasetCatalog, MetadataCatalog                 # Catalogs over registered datasets and metadatas for all datasets available in Detectron2
from custom_train_func import launch_custom_training                        # Function to launch the training with custom dataset
from visualize_vitrolife_batch import visualize_the_images                  # Import the function used for visualizing the image batch
from show_learning_curves import show_history                               # Function used to plot the learning curves for the given training


print(FLAGS)

# # Visualize some random images
# fig_list_before, data_batches, cfg, FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS)   # Visualize some segmentations on random images before training

# # Train the model
# launch_custom_training(args=FLAGS, config=cfg)                              # Launch the training loop

# # Visualize the same images, now with a trained model
# fig_list_after, data_batches, cfg, FLAGS = visualize_the_images(            # Visualize the same images ...
#     config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_has_trained=True)  # ... now after training

# # Evaluation on the vitrolife test dataset. There is no ADE20K test dataset.
# if FLAGS.debugging == False and "vitrolife" in FLAGS.dataset_name.lower():  # Inference will only be performed if we are not debugging the model
#     rename_output_inference_folder(config=cfg)                              # Rename the "inference" folder in OUTPUT_DIR to "validation" before doing inference
#     FLAGS.eval_only = True                                                  # Letting the model know we will only perform evaluation here
#     cfg.DATASETS.TEST = ("vitrolife_dataset_test",)                         # The inference will be done on the test dataset
#     launch_custom_training(args=FLAGS, config=cfg)                          # Launch the training (i.e. inference) loop

# # Display learning curves
# fig_learn_curves = show_history(config=cfg, FLAGS=FLAGS)                    # Create and save learning curves

# # Zip the resulting output directory
# make_archive(base_name=os.path.basename(cfg.OUTPUT_DIR), format="zip",      # Instantiate the zipping of the output directory  where the resulting zip file ...
#     root_dir=os.path.dirname(cfg.OUTPUT_DIR), base_dir=os.path.basename(cfg.OUTPUT_DIR))    # ... will include the output folder (not just the files from the folder)



