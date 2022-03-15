# Add the MaskFormer directory to PATH
import os                                                                           # Used to navigate the folder structure in the current os
import sys

from matplotlib.pyplot import hist                                                                          # Used to control the PATH variable
MaskFormer_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")                                                              # Home WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Home windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "MaskFormer")                              # Work WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Work windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "MaskFormer")  # Larac server
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "home_shared", MaskFormer_dir.split(os.path.sep, 2)[2])                                     # Balder server
assert os.path.isdir(MaskFormer_dir), "The MaskFormer directory doesn't exist in the chosen location"
sys.path.append(MaskFormer_dir)                                                     # Add MaskFormer directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "Custom_functions"))                   # Add Custom_functions directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "tools"))                              # Add the tools directory to PATH

# Add the environmental variable DETECTRON2_DATASETS
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")             # Home WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                              # Home windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Datasets")                                      # Work WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                              # Work windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                          # Larac server
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 2)[2])                                              # Balder server
assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"
os.environ["DETECTRON2_DATASETS"] = dataset_dir

# Import important libraries
import shutil
import numpy as np
from custom_setup_func import setup_func, zip_output, SaveHistory, keepAllButLatestAndBestModel # Assign script to GPU, register vitrolife dataset, create config, zip the output_dir, save the history_dict and delete unused models
from custom_train_func import launch_custom_training                                # Function to launch the training with the given dataset
from visualize_vitrolife_batch import putModelWeights, visualize_the_images         # Functions to put model_weights in the config and visualizing the image batch
from show_learning_curves import show_history                                       # Function used to plot the learning curves for the given training
from custom_evaluation_func import evaluateResults                                  # Function to evaluate the metrics for the segmentation


# Get the FLAGS and config variables
FLAGS, cfg = setup_func()

# Create properties
history = None
train_dataset = cfg.DATASETS.TRAIN
val_dataset = cfg.DATASETS.TEST
base_lr = cfg.SOLVER.BASE_LR


# Visualize some random images
fig_list_before, data_batches, cfg, FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS)   # Visualize some segmentations on random images before training

if FLAGS.inference_only == False:
    # Train the model
    for epoch in range(FLAGS.num_epochs):                                           # Iterate over the chosen amount of epochs
        trainer_class = launch_custom_training(FLAGS=FLAGS, config=cfg)             # Launch the training loop for one epoch
        shutil.copyfile(os.path.join(cfg.OUTPUT_DIR, "metrics.json"),               # Rename the metrics.json to train_metricsX.json ...
            os.path.join(cfg.OUTPUT_DIR, "train_metrics_{:d}.json".format(epoch+1)))# ... where X is the current epoch number
        os.remove(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))                     # Remove the original metrics file
        os.rename(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"),                  # Rename the model that is automatically saved after each epoch ...
            os.path.join(cfg.OUTPUT_DIR, "model_epoch_{:d}.pth".format(epoch+1)))   # ... to model_epoch_x (where x is current epoch number)
        eval_train_results = evaluateResults(FLAGS, cfg, data_split="train", trainer=trainer_class) # Evaluate the result metrics on the training set

        # Validation period
        cfg = putModelWeights(cfg)                                                  # Assign the latest model weights to the config
        cfg.DATASETS.TRAIN = val_dataset                                            # Change the 'train_dataset' variable to the validation dataset ...
        cfg.SOLVER.BASE_LR = float(0)                                               # Set the learning rate to 0
        validator_class = launch_custom_training(FLAGS=FLAGS, config=cfg)           # ... and then "train" the model, i.e. compute losses, wi
        shutil.copyfile(os.path.join(cfg.OUTPUT_DIR, "metrics.json"),               # Rename the metrics.json to val_metricsX.json ...
            os.path.join(cfg.OUTPUT_DIR, "val_metrics_{:d}.json".format(epoch+1)))  # ... where X is the current epoch number
        os.remove(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))                     # Remove the original metrics file
        os.remove(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))                  # Remove the model that will be saved after validation
        eval_val_results = evaluateResults(FLAGS, cfg, data_split="val", trainer=trainer_class) # Evaluate the result metrics on the training set
        
        # Prepare for the training phase of the next epoch
        cfg.DATASETS.TRAIN = train_dataset                                          # Set the 'dataset_train' variable back to the training data
        cfg.SOLVER.BASE_LR = base_lr * 0.9                                          # Lower the learning right just a little
        base_lr = cfg.SOLVER.BASE_LR                                                # Save a new value for the base_lr variable
        history = show_history(config=cfg, FLAGS=FLAGS, metrics_train=eval_train_results["sem_seg"], metrics_eval=eval_val_results["sem_seg"], history=history)  # Create and save learning curves 
        cfg = keepAllButLatestAndBestModel(cfg=cfg, history=history, FLAGS=FLAGS)   # Keep only the best and the latest model weights. The rest are deleted.
        if np.mod(np.add(epoch,1), FLAGS.display_rate) == 0:                        # Every 'display_rate' epochs, the model will segment the same images again ...
            _,data_batches,cfg,FLAGS=visualize_the_images(config=cfg,FLAGS=FLAGS,data_batches=data_batches, epoch_num=epoch+1)  # ... segment and save visualizations

    # Visualize the same images, now with a trained model
    cfg = keepAllButLatestAndBestModel(cfg=cfg, history=history, FLAGS=FLAGS, bestOrLatest="best")  # Put the model weights for the best performing model on the config
    SaveHistory(historyObject=history, save_folder=cfg.OUTPUT_DIR)                  # Save the history dictionary after training
    fig_list_after, data_batches, cfg, FLAGS = visualize_the_images(                # Visualize the same images ...
        config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_has_trained=True)  # ... now after the model has trained

# Evaluation on the vitrolife test dataset. There is no ADE20K test dataset.
if FLAGS.debugging == False and "vitrolife" in FLAGS.dataset_name.lower():          # Inference will only be performed if we are not debugging the model and working on the vitrolife dataset
    cfg.DATASETS.TEST = ("vitrolife_dataset_test",)                                 # The inference will be done on the test dataset
    eval_test_results = evaluateResults(FLAGS, cfg, data_split="test")              # Evaluate the result metrics on the validation set with the best performing model


# Zip the resulting output directory
zip_output(cfg)




