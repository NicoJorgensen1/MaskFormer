# Import libraries
import os
import numpy as np
from natsort import natsorted                                                       # Used to sort the list of model_files saved 
from visualize_vitrolife_batch import extractNumbersFromString                      # Function to extract numbers from a string

# Define a function to commit early stopping
def early_stopping(history, FLAGS, quit_training=False):
    mode = "min" if "loss" in FLAGS.eval_metric.lower() else "max"                  # Whether a lower value or a higher value is better
    metric_monitored = history[FLAGS.eval_metric][-FLAGS.early_stop_patience+1:]    # Getting the last 'early_stop_patience' values of the 'monitor' metric
    if len(history[FLAGS.eval_metric]) >= FLAGS.early_stop_patience:                # If we have run for at least FLAGS.early_stop_patience epochs, we'll continue
        if mode=="max": val_used = np.max(metric_monitored)                         # If we monitor an increasing metric, we want to find the largest value
        if mode=="min": val_used = np.min(metric_monitored)                         # If we monitor a decreasing metric, we want to find the smallest value    
        if mode=="max" and val_used <= metric_monitored[0] + FLAGS.min_delta or mode=="min" and val_used >= metric_monitored[0] - FLAGS.min_delta:  # If the model hasn't improved in the last ...
            quit_training = True                                                    # ... 'early_stop_patience' epochs, the training is terminated
    return quit_training


# Define a function to lower the learning rate
def lr_scheduler(cfg, history, FLAGS, learn_rate):
    mode = "min" if "loss" in FLAGS.eval_metric.lower() else "max"                  # Whether a lower value or a higher value is better
    metric_monitored = history[FLAGS.eval_metric][-FLAGS.patience:]                 # Getting the last 'patience' values of the 'monitor' metric
    if len(history[FLAGS.eval_metric]) >= FLAGS.patience:                           # If we have run for at least FLAGS.patience epochs, we'll continue
        if mode=="max": val_used = np.max(metric_monitored)                         # If we monitor an increasing metric, we want to find the largest value
        if mode=="min": val_used = np.min(metric_monitored)                         # If we monitor a decreasing metric, we want to find the smallest value    
        if mode=="max" and val_used <= metric_monitored[0] + FLAGS.min_delta or mode=="min" and val_used >= metric_monitored[0] - FLAGS.min_delta:  # If the model hasn't improved in the last 'patience' ...
            cfg.SOLVER.BASE_LR = learn_rate * FLAGS.lr_gamma                        # ... epochs the learning rate is lowered
    return cfg


# Define a function to delete all models but the 
def keepAllButLatestAndBestModel(cfg, history, FLAGS, bestOrLatest="latest", delete_leftovers=True):
    model_files = natsorted([x for x in os.listdir(cfg.OUTPUT_DIR) if "model_epoch" in x.lower()])  # Get a list of available models
    if len(model_files) >= 1:
        mode = "min" if "loss" in FLAGS.eval_metric.lower() else "max"      # Whether a lower value or a higher value is better
        epoch_numbers = [extractNumbersFromString(x, dtype=int) for x in model_files]   # Extract which epoch each model are from
        metric_list = history[FLAGS.eval_metric]                            # Read the list of values for the metric chosen to use as 
        metric_list = np.asarray(metric_list)[np.subtract(epoch_numbers,1)].tolist()    # Get the remaining values (i.e. if some models have already been deleted, their corresponding values should be removed)
        best_model_idx = np.argmin(metric_list) if mode=="min" else np.argmax(metric_list)  # Get the idx of the best epoch
        best_model = model_files[best_model_idx]                            # Get the model name of the best model
        latest_model = model_files[np.argmax(epoch_numbers)]                # Get the model name of the latest model
        if delete_leftovers == True:                                        # If we want to delete the leftovers ...
            [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in model_files if not any([x==best_model, x==latest_model])]  # ... remove the models that are neither the best or latest model
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, latest_model if "latest" in bestOrLatest.lower() else best_model)  # Set the model weights as either the best or the latest model
    return cfg                                                              # Return the config where the cfg.MODEL.WEIGHTS are set to the chosen model


