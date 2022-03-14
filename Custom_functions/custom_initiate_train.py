# Add the MaskFormer directory to PATH
from curses import putp
import os                                                                   # Used to navigate the folder structure in the current os
import sys                                                                  # Used to control the PATH variable
MaskFormer_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")                                                              # Home WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Home windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "MaskFormer")                              # Work WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Work windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "MaskFormer")  # Larac server
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "home_shared", MaskFormer_dir.split(os.path.sep, 2)[2])                                     # Balder server
assert os.path.isdir(MaskFormer_dir), "The MaskFormer directory doesn't exist in the chosen location"
sys.path.append(MaskFormer_dir)                                             # Add MaskFormer directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "Custom_functions"))           # Add Custom_functions directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "tools"))                      # Add the tools directory to PATH

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
from custom_setup_func import rename_output_inference_folder, setup_func, zip_output    # Functions to rename output dir and assign to GPU, register vitrolife dataset, create config and zip output_dir
from custom_train_func import launch_custom_training                        # Function to launch the training with custom dataset
from visualize_vitrolife_batch import putModelWeights, visualize_the_images # Function to assign the model_weights to the config and a function used for visualizing the image batch
from show_learning_curves import show_history                               # Function used to plot the learning curves for the given training
from custom_evaluation_func import evaluateResults                          # Function to evaluate the metrics for the segmentation

# Get the FLAGS and config variables
FLAGS, cfg = setup_func()


# Visualize some random images
fig_list_before, data_batches, cfg, FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS)   # Visualize some segmentations on random images before training

if FLAGS.inference_only == False:
    # Train the model
    for epoch in range(FLAGS.num_epochs):
        print("Starting epoch {:d}".format(epoch+1))
        trainer_class = launch_custom_training(FLAGS=FLAGS, config=cfg)     # Launch the training loop for one epoch
        cfg = putModelWeights(cfg)                                          # Assign the newest model weights to the config
        eval_train_results = evaluateResults(FLAGS, cfg, data_split="train", trainer=trainer_class) # Evaluate the result metrics on the training set
        eval_val_results = evaluateResults(FLAGS, cfg, data_split="val", trainer=trainer_class)     # Evaluate the result metrics on the training set
        os.rename(os.path.join(cfg.OUTPUT_DIR, "metrics.json"),             # Rename the metrics.json to metricsX.json ...
            os.path.join(cfg.OUTPUT_DIR, "metrics_{:d}.json".format(epoch+1)))  # ... where X is the current epoch number
    

    # Visualize the same images, now with a trained model
    fig_list_after, data_batches, cfg, FLAGS = visualize_the_images(        # Visualize the same images ...
        config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_has_trained=True)  # ... now after training

# Evaluation on the vitrolife test dataset. There is no ADE20K test dataset.
if FLAGS.debugging == False and "vitrolife" in FLAGS.dataset_name.lower():  # Inference will only be performed if we are not debugging the model and working on the vitrolife dataset
    cfg.DATASETS.TEST = ("vitrolife_dataset_test",)                         # The inference will be done on the test dataset
    eval_train_results = evaluateResults(FLAGS, cfg, data_split="test")     # Evaluate the result metrics on the validation set

# Display learning curves
fig_learn_curves = show_history(config=cfg, FLAGS=FLAGS)                    # Create and save learning curves

# Zip the resulting output directory
zip_output(cfg)




