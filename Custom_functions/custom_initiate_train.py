# Add the MaskFormer directory to PATH
import os                                                                                           # Used to navigate the folder structure in the current os
import sys                                                                                          # Used to control the PATH variable
MaskFormer_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")                                                              # Home WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Home windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "MaskFormer")                              # Work WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Work windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "MaskFormer")  # Larac server
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "home_shared", MaskFormer_dir.split(os.path.sep, 2)[2])                                     # Balder server
assert os.path.isdir(MaskFormer_dir), "The MaskFormer directory doesn't exist in the chosen location"
sys.path.append(MaskFormer_dir)                                                                     # Add MaskFormer directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "Custom_functions"))                                   # Add Custom_functions directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "tools"))                                              # Add the tools directory to PATH

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
import optuna                                                                                       # Library used to perform hyperparameter optimization 
import hiplot
import numpy as np 
import gc as garb_collect                                                                           # Used for garbage collecting after each hyperparameter trial
from custom_callback_functions import keepAllButLatestAndBestModel                                  # Used for setting model weights on the config
from custom_setup_func import setup_func, zip_output, printAndLog, getBestEpochResults, SaveHistory # Assign to GPU, register vitrolife dataset, create config, zip output_dir, log results, get best results, save dictionary
from custom_train_func import objective_train_func                                                  # Function to launch the training with the given dataset
from visualize_image_batch import visualize_the_images                                              # Functions visualize the image batch
from custom_model_analysis_func import analyze_model_func                                           # Analyze the model FLOPS, number of parameters and activations computed

# Get the FLAGS and config variables and analyze the model
FLAGS, cfg, log_file = setup_func()
model_analysis = analyze_model_func(config=cfg)
printAndLog(input_to_write="Model analysis:".upper(), logs=log_file)
printAndLog(input_to_write=model_analysis, logs=log_file, oneline=False, length=27)

# Visualize some random images before training 
fig_list_before, data_batches, cfg, FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS)           # Visualize some segmentations on random images before training
def object_func(trial):
    new_best = float("nan")
    it_count = 0
    while np.isnan(new_best):                                                                       # Sometimes the iteration fails for some reason. We'll allow three extra retrys before skipping
        try:
            new_best = objective_train_func(trial=trial, FLAGS=FLAGS, cfg=cfg, logs=log_file, data_batches=data_batches, hyperparameter_optimization=True)
        except Exception as ex:
            error_str = "An exception of type {0} occured. Arguments:\n{1!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_str, logs=log_file)
            new_best = float("nan")
        it_count += 1
        if it_count >= 4: break 
    [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if all([x.endswith(".pth"), "model" in x.lower()])]
    [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if all([x.endswith(".json"), "metrics" in x.lower()])]
    [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if "events.out.tfevents" in x]
    FLAGS.HPO_trial += 1
    return new_best

# Make a hyperparameter optimization
if FLAGS.hp_optim:
    warm_ups = FLAGS.warm_up_epochs
    FLAGS.warm_up_epochs = 0
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), study_name="Hyperparameter optimization for {:s} dataset".format(FLAGS.dataset_name), direction="maximize")
    study.optimize(object_func, n_trials=2, callbacks=[lambda study, trial: garb_collect.collect()])
    trial = study.best_trial
    best_params = trial.params
    SaveHistory(historyObject=best_params, save_folder=cfg.OUTPUT_DIR, historyName="best_HPO_params")
    printAndLog(input_to_write="Hyperparameter optimization completed.\nBest val_fwIoU: {:.3f}".format(trial.value), logs=log_file, prefix="\n\n")
    printAndLog(input_to_write="Best hyperparameters: ".ljust(25), logs=log_file)
    printAndLog(input_to_write=best_params, logs=log_file, prefix="", postfix="\n\n", oneline=True, length=15)
    FLAGS.warm_up_epochs = warm_ups
    fig = optuna.visualization.plot_contour(study)

# Train the model with the best found hyperparameters
history, test_history, new_best, best_epoch, cfg = objective_train_func(trial=trial, FLAGS=FLAGS, cfg=cfg, logs=log_file, data_batches=data_batches, hyperparameter_optimization=False)

# Visualize the same images, now after training
cfg = keepAllButLatestAndBestModel(cfg=cfg, history=history, FLAGS=FLAGS, bestOrLatest="best")      # Put the model weights for the best performing model on the config
visualize_the_images(config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_done_training=True)   # Visualize the images again

# Print and log the best metric results
printAndLog(input_to_write="Final results:".upper(), logs=log_file)
if FLAGS.inference_only==False: 
    printAndLog(input_to_write="Best validation results:".ljust(30)+"Epoch {:d}: {:s} = {:.3f}\n{:s}".format(best_epoch, FLAGS.eval_metric, new_best, "All best validation results:".upper().ljust(30)), logs=log_file)
    printAndLog(input_to_write=getBestEpochResults(history, best_epoch), logs=log_file, prefix="")
if "test" in cfg.DATASETS.TEST[0]:
    printAndLog(input_to_write="All test results:".upper().ljust(30), logs=log_file)
    printAndLog(input_to_write=test_history, logs=log_file, prefix="")

# Remove all metrics.json files, the default log-file and zip the resulting output directory
[os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if "metrics" in x.lower() and x.endswith(".json")]
os.remove(os.path.join(cfg.OUTPUT_DIR, "log.txt"))
zip_output(cfg)

