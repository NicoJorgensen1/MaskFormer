# Add the MaskFormer directory to PATH
import os                                                                           # Used to navigate the folder structure in the current os
import sys                                                                          # Used to control the PATH variable
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
import numpy as np                                                                  # Used for algebraic equations
from time import time                                                               # Used to time the epoch/training duration
from custom_setup_func import setup_func, zip_output, SaveHistory, printAndLog, getBestEpochResults # Assign to GPU, register vitrolife dataset, create config, zip output_dir, save history_dict, log results, get best results
from custom_train_func import launch_custom_training                                # Function to launch the training with the given dataset
from visualize_image_batch import visualize_the_images                              # Functions visualize the image batch
from show_learning_curves import show_history, combineDataToHistoryDictionaryFunc   # Function used to plot the learning curves for the given training and to add results to the history dictionary
from custom_evaluation_func import evaluateResults                                  # Function to evaluate the metrics for the segmentation
from custom_callback_functions import early_stopping, lr_scheduler, keepAllButLatestAndBestModel, computeRemainingTime, updateLogsFunc  # Callback functions for model training
from custom_pq_eval_func import pq_evaluation                                       # Used to perform the panoptic quality evaluation on the semantic segmentation results
from visualize_conf_matrix import plot_confusion_matrix                             # Function to plot the available confusion matrixes


# Get the FLAGS and config variables
FLAGS, cfg, log_file = setup_func()

# Create properties
train_trainer, val_trainer = None, None
train_loader, val_loader, train_evaluator, val_evaluator, history = None, None, None, None, None    # Initiates all the loaders, evaluators and history as None type objects
train_mode = "min" if "loss" in FLAGS.eval_metric else "max"                        # Compute the mode of which the performance should be measured. Either a negative or a positive value is better
new_best = np.inf if train_mode=="min" else -np.inf                                 # Initiate the original "best_value" as either infinity or -infinity according to train_mode
best_epoch = 0                                                                      # Initiate the best epoch as being epoch_0, i.e. before doing any model training
train_dataset = cfg.DATASETS.TRAIN                                                  # Get the training dataset name
val_dataset = cfg.DATASETS.TEST                                                     # Get the validation dataset name
lr_update_check = np.zeros((FLAGS.patience, 1), dtype=bool)                         # Preallocating array to determine whether or not the learning rate was updated

# Visualize some random images
fig_list_before, data_batches, cfg, FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS)   # Visualize some segmentations on random images before training

if FLAGS.inference_only == False:
    # Train the model
    train_start_time = time()                                                       # Now the training starts
    for epoch in range(FLAGS.num_epochs):                                           # Iterate over the chosen amount of epochs
        # Training period. Will train the model, correct the metrics files and evaluate performance on the training data
        epoch_start_time = time()                                                   # Now this new epoch starts
        cfg = launch_custom_training(FLAGS=FLAGS, config=cfg, dataset=train_dataset, epoch=epoch, run_mode="train") # Launch the training loop for one epoch
        eval_train_results, train_loader, train_evaluator, conf_matrix_train = evaluateResults(FLAGS, cfg, data_split="train", dataloader=train_loader, evaluator=train_evaluator) # Evaluate the result metrics on the training set
        train_pq_results = pq_evaluation(args=FLAGS, config=cfg, data_split="train")# Evaluate the Panoptic Quality for the training semantic segmentation results

        # Validation period. Will 'train' with lr=0 on validation data, correct the metrics files and evaluate performance on validation data
        cfg = launch_custom_training(FLAGS=FLAGS, config=cfg, dataset=val_dataset, epoch=epoch, run_mode="val") # Launch the training loop for one epoch
        eval_val_results, val_loader, val_evaluator, conf_matrix_val = evaluateResults(FLAGS, cfg, data_split="val", dataloader=val_loader, evaluator=val_evaluator) # Evaluate the result metrics on the training set
        val_pq_results = pq_evaluation(args=FLAGS, config=cfg, data_split="val")    # Evaluate the Panoptic Quality for the validation semantic segmentation results
        
        # Prepare for the training phase of the next epoch. Switch back to training dataset, save history and learning curves and visualize segmentation results
        history = show_history(config=cfg, FLAGS=FLAGS, metrics_train=eval_train_results["sem_seg"],    # Create and save the learning curves ...
            metrics_eval=eval_val_results["sem_seg"], history=history, pq_train=train_pq_results, pq_val=val_pq_results)    # ... including all training and validation metrics
        SaveHistory(historyObject=history, save_folder=cfg.OUTPUT_DIR)              # Save the history dictionary after each epoch
        [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if "events.out.tfevent" in x]
        if np.mod(np.add(epoch,1), FLAGS.display_rate) == 0:                        # Every 'display_rate' epochs, the model will segment the same images again ...
            _,data_batches,cfg,FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS, data_batches=data_batches, epoch_num=epoch+1)    # ... segment and save visualizations
            _ = plot_confusion_matrix(config=cfg, epoch=epoch+1, conf_train=conf_matrix_train, conf_val=conf_matrix_val)
        
        # Performing callbacks
        cfg = keepAllButLatestAndBestModel(cfg=cfg, history=history, FLAGS=FLAGS)   # Keep only the best and the latest model weights. The rest are deleted.
        if epoch+1 >= FLAGS.patience:                                               # If the model has trained for more than 'patience' epochs and we aren't debugging ...
            cfg, lr_update_check = lr_scheduler(cfg=cfg, history=history, FLAGS=FLAGS, lr_updated=lr_update_check)  # ... change the learning rate, if needed
            FLAGS.learning_rate = cfg.SOLVER.BASE_LR                                # Update the FLAGS.learning_rate value
        if epoch+1 >= FLAGS.early_stop_patience:                                    # If the model has trained for more than 'early_stopping_patience' epochs ...
            quit_training = early_stopping(history=history, FLAGS=FLAGS)            # ... perform the early stopping callback
            if quit_training == True: break                                         # If the early stopping callback says we need to quit the training, break the for loop and stop running more epochs
        string1, string2 = computeRemainingTime(epoch=epoch, num_epochs=FLAGS.num_epochs, train_start_time=train_start_time, epoch_start_time=epoch_start_time)
        new_best, best_epoch = updateLogsFunc(log_file=log_file, FLAGS=FLAGS, history=history, best_val=new_best, train_start=train_start_time, epoch_start=epoch_start_time, best_epoch=best_epoch)

    # Visualize the same images, now with a trained model
    cfg = keepAllButLatestAndBestModel(cfg=cfg, history=history, FLAGS=FLAGS, bestOrLatest="best")  # Put the model weights for the best performing model on the config
    fig_list_after, data_batches, cfg, FLAGS = visualize_the_images(                # Visualize the same images ...
        config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_done_training=True)    # ... now after the model has trained

# Evaluation on the vitrolife test dataset. There is no ADE20K test dataset.
if FLAGS.debugging == False and "vitrolife" in FLAGS.dataset_name.lower():          # Inference will only be performed if we are not debugging the model and working on the vitrolife dataset
    cfg.DATASETS.TEST = ("vitrolife_dataset_test",)                                 # The inference will be done on the test dataset
    eval_test_results,_,_,conf_matrix_test = evaluateResults(FLAGS, cfg, data_split="test") # Evaluate the result metrics on the validation set with the best performing model
    _ = plot_confusion_matrix(config=cfg, conf_train=conf_matrix_train, conf_val=conf_matrix_val, conf_test=conf_matrix_test, done_training=True)
    test_pq_results = pq_evaluation(args=FLAGS, config=cfg, data_split="test")      # Evaluate the Panoptic Quality for the test semantic segmentation results
    history = combineDataToHistoryDictionaryFunc(config=cfg, eval_metrics=eval_test_results["sem_seg"], pq_metrics=test_pq_results, data_split="test", history=history)
    test_history = {}
    for key in history.keys():
        if "test" in key:
            test_history[key] = history[key][-1]

# Print and log the best metric results
printAndLog(input_to_write="Final results:".upper(), logs=log_file)
printAndLog(input_to_write="Best validation results:".ljust(30)+"Epoch {:d}: {:s} = {:.3f}\n{:s}".format(best_epoch, FLAGS.eval_metric, new_best, "All best validation results:".upper().ljust(30)), logs=log_file)
printAndLog(input_to_write=getBestEpochResults(history, best_epoch), logs=log_file, prefix="")
if "test" in cfg.DATASETS.TEST[0]:
    printAndLog(input_to_write="All test results:".upper().ljust(30), logs=log_file)
    printAndLog(input_to_write=test_history, logs=log_file, prefix="")

# Remove all metrics.json files, the default log-file and zip the resulting output directory
[os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if "metrics" in x.lower() and x.endswith(".json")]
# os.remove(os.path.join(cfg.OUTPUT_DIR, "log.txt"))
zip_output(cfg)




