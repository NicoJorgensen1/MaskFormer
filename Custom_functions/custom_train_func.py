# Import libraries 
import shutil                                                                                               # Used to copy/rename the metrics.json file after each training/validation step
import os                                                                                                   # For joining paths
import numpy as np                                                                                          # For algebraic equations 
from time import time                                                                                       # Used to time the epoch/training duration
from copy import  deepcopy                                                                                  # Used to create a new copy in memory
from custom_goto_trainer_class import My_GoTo_Trainer                                                       # To instantiate the Trainer class
from visualize_image_batch import putModelWeights                                                           # Assign the latest model checkpoint to the config model.weights 
from custom_setup_func import SaveHistory, printAndLog                                                      # Save history_dict, log results
from create_custom_config import createVitrolifeConfiguration, changeConfig_withFLAGS                       # Create the config used for hyperparameter optimization 
from visualize_image_batch import visualize_the_images                                                      # Functions visualize the image batch
from show_learning_curves import show_history, combineDataToHistoryDictionaryFunc                           # Function used to plot the learning curves for the given training and to add results to the history dictionary
from custom_evaluation_func import evaluateResults                                                          # Function to evaluate the metrics for the segmentation
from custom_callback_functions import early_stopping, lr_scheduler, keepAllButLatestAndBestModel, updateLogsFunc    # Callback functions for model training
from custom_pq_eval_func import pq_evaluation                                                               # Used to perform the panoptic quality evaluation on the semantic segmentation results
from visualize_conf_matrix import plot_confusion_matrix                                                     # Function to plot the available confusion matrixes


def run_train_func(cfg, run_mode):
    Trainer = My_GoTo_Trainer(cfg)
    Trainer.resume_or_load(resume=False)
    return Trainer.train()

# Function to launch the training
def launch_custom_training(FLAGS, config, dataset, epoch=0, run_mode="train", hyperparameter_opt=False):
    FLAGS.epoch_iter = int(np.floor(np.divide(FLAGS.num_train_files, FLAGS.batch_size)))                    # Compute the number of iterations per training epoch with the given batch size
    config.SOLVER.MAX_ITER = FLAGS.epoch_iter * (7 if all(["train" in run_mode, hyperparameter_opt==False, "vitrolife" in FLAGS.dataset_name.lower()]) else 1)  # Increase training iteration count for precise BN computations
    if all(["train" in run_mode, hyperparameter_opt==True]):
        if "vitrolife" in FLAGS.dataset_name.lower(): config.SOLVER.MAX_ITER = int(FLAGS.epoch_iter * (1 if FLAGS.use_per_pixel_baseline else 2.0)) # ... Transformer and ResNet backbones need a ...
        elif "ade20k" in FLAGS.dataset_name.lower(): config.SOLVER.MAX_ITER = int(FLAGS.epoch_iter * (1 if FLAGS.use_per_pixel_baseline else 2)/6)  # ... little more data to do well while searching...
    if "val" in run_mode and "ade20k" in FLAGS.dataset_name.lower(): config.SOLVER.MAX_ITER = int(np.ceil(np.divide(FLAGS.epoch_iter, 4)))
    if any([x in config.OUTPUT_DIR.lower() for x in ["nico", "wd974261"]]):                                 # If I am working on my own local computer ...
        config.SOLVER.MAX_ITER = int(np.min([FLAGS.epoch_iter, 15]))                                        # ... the maximum number of iterations is lowered 
    config.SOLVER.CHECKPOINT_PERIOD = config.SOLVER.MAX_ITER                                                # Save a new model checkpoint after each epoch
    if "train" in run_mode and hyperparameter_opt==False:                                                   # If we are training ... 
        for idx, item in enumerate(config.custom_key[::-1]):                                                # Iterate over the custom keys in reversed order
            if "epoch_num" in item[0]:                                                                      # If the current item is the tuple with the epoch_number
                config.custom_key[-idx-1] = (item[0], item[1]+1)                                            # The current epoch number is updated 
                break                                                                                       # And the loop is broken out of 
    config = putModelWeights(config)                                                                        # Assign the latest saved model to the config
    if "train" in run_mode.lower():                                                                         # If we are on the training split ...
        config.SOLVER.BASE_LR = FLAGS.learning_rate                                                         # ... set the latest saved learning rate to the config
        config.MODEL.MASK_FORMER.DROPOUT = FLAGS.dropout                                                    # ... set the dropout to the chosen dropout value
    else:                                                                                                   # Else, if we are on the validation set 
        config.SOLVER.BASE_LR = float(0)                                                                    # ... set the learning rate to 0
        config.MODEL.MASK_FORMER.DROPOUT = 0                                                                # ... set the dropout to 0 
    config.DATASETS.TRAIN = dataset                                                                         # Change the config dataset used to the dataset sent along ...
    run_train_func(cfg=config, run_mode=run_mode)                                                           # Run the training for the current epoch
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "metrics.json"),                                        # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, run_mode+"_metrics_{:d}.json".format(epoch+1)))                     # ... where X is the current epoch number
    os.remove(os.path.join(config.OUTPUT_DIR, "metrics.json"))                                              # Remove the original metrics file
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "model_final.pth"),                                     # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, "model_epoch_{:d}.pth".format(epoch+1)))                            # ... where X is the current epoch number    
    [os.remove(os.path.join(config.OUTPUT_DIR, x)) for x in os.listdir(config.OUTPUT_DIR) if all(["model_" in x, "epoch" not in x, x.endswith(".pth")])]    # Remove all irrelevant models
    return config


# Define a function to create the hyper parameters of the trials
def get_HPO_params(config, FLAGS, trial, hpt_opt=False):
    # If we are performing hyperparameter optimization, the config should be updated
    if all([hpt_opt==True, trial is not None, FLAGS.hp_optim==True]):
        # Change the FLAGS parameters and then change the config
        FLAGS.learning_rate = trial.suggest_float(name="learning_rate", low=1e-6, high=1e-3)
        FLAGS.batch_size = trial.suggest_int(name="batch_size", low=1, high=int(np.ceil(np.min(FLAGS.available_mem_info)/1250)))
        FLAGS.optimizer_used = trial.suggest_categorical(name="optimizer_used", choices=["ADAMW", "SGD"])
        FLAGS.weight_decay = trial.suggest_float(name="weight_decay", low=1e-8, high=3e-1)
        FLAGS.backbone_multiplier = trial.suggest_float("backbone_multiplier", low=1e-6, high=0.5) 
        FLAGS.dice_loss_weight = trial.suggest_int(name="dice_loss_weight", low=1, high=25)
        FLAGS.mask_loss_weight = trial.suggest_int(name="mask_loss_weight", low=1, high=25)
        FLAGS.dropout = trial.suggest_float(name="dropout", low=1e-10, high=0.75)
        if "vitrolife" in FLAGS.dataset_name:
            FLAGS.num_queries = trial.suggest_int(name="num_queries", low=10, high=250) 
            FLAGS.use_checkpoint = bool(trial.suggest_categorical(name="use_checkpoint", choices=["True", "False"]))
        if FLAGS.use_transformer_backbone==False:
            FLAGS.resnet_depth = trial.suggest_categorical(name="resnet_depth", choices=[50, 101])
            FLAGS.backbone_freeze_layers = trial.suggest_int(name="backbone_freeze", low=0, high=5)
        del config 
        config = createVitrolifeConfiguration(FLAGS=FLAGS)
        config = changeConfig_withFLAGS(cfg=config, FLAGS=FLAGS)
    elif all([hpt_opt==False, trial is not None, FLAGS.hp_optim==True]):
        # Let the FLAGS parameters take the values of the best found parameters 
        FLAGS.learning_rate = FLAGS.best_params["learning_rate"]
        FLAGS.batch_size = FLAGS.best_params["batch_size"]
        FLAGS.optimizer_used = FLAGS.best_params["optimizer_used"]
        FLAGS.weight_decay = FLAGS.best_params["weight_decay"]
        FLAGS.backbone_multiplier = FLAGS.best_params["backbone_multiplier"] 
        FLAGS.dice_loss_weight = FLAGS.best_params["dice_loss_weight"]
        FLAGS.mask_loss_weight = FLAGS.best_params["mask_loss_weight"]
        FLAGS.dropout = FLAGS.best_params["dropout"]
        if "vitrolife" in FLAGS.dataset_name.lower():
            FLAGS.num_queries = FLAGS.best_params["num_queries"]
            FLAGS.use_checkpoint = bool(FLAGS.best_params["use_checkpoint"])
        if FLAGS.use_transformer_backbone==False:
            FLAGS.resnet_depth = FLAGS.best_params["resnet_depth"]
            FLAGS.backbone_freeze_layers = FLAGS.best_params["backbone_freeze"]
        del config 
        config = createVitrolifeConfiguration(FLAGS=FLAGS)
        config = changeConfig_withFLAGS(cfg=config, FLAGS=FLAGS)
    else: config = deepcopy(config)
    return config, FLAGS


# Create function to train the objective function
def objective_train_func(trial, FLAGS, cfg, logs, data_batches=None, hyperparameter_optimization=False):
    # Setup training variables before starting training
    objective_mode = "training"
    if FLAGS.inference_only: objective_mode = "inference"
    if hyperparameter_optimization: objective_mode = "hyperparameter optimization trial {:d}/{:d}".format(FLAGS.HPO_current_trial+1, FLAGS.num_trials)
    printAndLog(input_to_write="Start {:s}...".format(objective_mode).upper(), logs=logs, postfix="\n")     # Print and log a message saying that a new iteration is now starting
    train_loader, val_loader, train_evaluator, val_evaluator = None, None, None, None                       # Initiates all the loaders, evaluators and history as None type objects
    train_mode = "min" if "loss" in FLAGS.eval_metric else "max"                                            # Compute the mode of which the performance should be measured. Either a negative or a positive value is better
    new_best = np.inf if train_mode=="min" else -np.inf                                                     # Initiate the original "best_value" as either infinity or -infinity according to train_mode
    best_epoch = 0                                                                                          # Initiate the best epoch as being epoch_0, i.e. before doing any model training
    train_pq_results, val_pq_results, eval_train_results = None, None, None                                 # Set train/val PQ results and evaluation results as None values
    conf_matrix_train, conf_matrix_val, conf_matrix_test = None, None, None                                 # Initialize the confusion matrixes as None values 
    train_dataset = cfg.DATASETS.TRAIN                                                                      # Get the training dataset name
    val_dataset = cfg.DATASETS.TEST                                                                         # Get the validation dataset name
    lr_update_check = np.zeros((FLAGS.patience, 1), dtype=bool)                                             # Preallocating validation array to determine whether or not the learning rate was updated
    quit_training = False                                                                                   # Boolean value determining whether or not to commit early stopping
    epochs_to_run = 1 if hyperparameter_optimization else FLAGS.num_epochs                                  # We'll run only 1 epoch if we are performing HPO
    epochs_to_run = 1 if FLAGS.inference_only else epochs_to_run                                            # If we are just performing inference, then we'll only go through the dataset once 
    train_start_time = time()                                                                               # Now the training starts
    epoch_next_display = FLAGS.display_rate - 1                                                             # The next epoch where the images must be visualized
    img_ytrue_ypred = None                                                                                  # Initiate a variable for the predicted images 
    history = deepcopy(FLAGS.history) 

    # Change the FLAGS and config parameters and perform either hyperparameter optimization, use the best found parameters or simply just train
    config, FLAGS = get_HPO_params(config=cfg, FLAGS=FLAGS, trial=trial, hpt_opt=hyperparameter_optimization)
    
    # Train the model 
    for epoch in range(FLAGS.start_epoch, epochs_to_run):                                                   # Iterate over the chosen amount of epochs
        try:
            epoch_start_time = time()                                                                       # Now this new epoch starts
            if FLAGS.inference_only==False:
                config = launch_custom_training(FLAGS=FLAGS, config=config, dataset=train_dataset, epoch=epoch, run_mode="train", hyperparameter_opt=hyperparameter_optimization)   # Launch the training loop for one epoch
                if not hyperparameter_optimization:
                    eval_train_results, train_loader, train_evaluator, conf_matrix_train, _, _ = evaluateResults(FLAGS, config, data_split="train", dataloader=train_loader, evaluator=train_evaluator, hp_optim=hyperparameter_optimization) # Evaluate the result on the training set
                    train_pq_results = pq_evaluation(args=FLAGS, config=config, data_split="train", hp_optim=hyperparameter_optimization)   # Evaluate the Panoptic Quality for the training semantic segmentation results  
            
            # Validation period. Will 'train' with lr=0 on validation data, correct the metrics files and evaluate performance on validation data
            config = launch_custom_training(FLAGS=FLAGS, config=config, dataset=val_dataset, epoch=epoch, run_mode="val", hyperparameter_opt=hyperparameter_optimization)   # Launch the training loop for one epoch
            eval_val_results, val_loader, val_evaluator, conf_matrix_val, _, _ = evaluateResults(FLAGS, config, data_split="val", dataloader=val_loader, evaluator=val_evaluator) # Evaluate the result metrics on the training set
            if not hyperparameter_optimization:
                val_pq_results = pq_evaluation(args=FLAGS, config=config, data_split="val")                 # Evaluate the Panoptic Quality for the validation semantic segmentation results
            
            # Prepare for the training phase of the next epoch. Switch back to training dataset, save history and learning curves and visualize segmentation results
            if any([all([FLAGS.inference_only, "ade20k" in FLAGS.dataset_name.lower()], all([FLAGS.inference_only==False, "vitrolife" in FLAGS.dataset_name.lower()]))]):
                history = show_history(config=config, FLAGS=FLAGS, metrics_train=eval_train_results,            # Create and save the learning curves ...
                    metrics_eval=eval_val_results, history=history, pq_train=train_pq_results, pq_val=val_pq_results)    # ... including all training and validation metrics
                SaveHistory(historyObject=history, save_folder=config.OUTPUT_DIR)                               # Save the history dictionary after each epoch
                [os.remove(os.path.join(config.OUTPUT_DIR, x)) for x in os.listdir(config.OUTPUT_DIR) if "events.out.tfevent" in x]
            
            # Performing callbacks
            if FLAGS.inference_only==False and hyperparameter_optimization==False: 
                config = keepAllButLatestAndBestModel(cfg=config, history=history, FLAGS=FLAGS)             # Keep only the best and the latest model weights. The rest are deleted.
                if epoch+1 >= FLAGS.patience:                                                               # If the model has trained for more than 'patience' epochs and we aren't debugging ...
                    config, lr_update_check = lr_scheduler(cfg=config, history=history, FLAGS=FLAGS, lr_updated=lr_update_check)  # ... change the learning rate, if needed
                    FLAGS.learning_rate = config.SOLVER.BASE_LR                                             # Update the FLAGS.learning_rate value
                if epoch+1 >= FLAGS.early_stop_patience:                                                    # If the model has trained for more than 'early_stopping_patience' epochs ...
                    quit_training = early_stopping(history=history, FLAGS=FLAGS)                            # ... perform the early stopping callback
            earlier_HPO_best = deepcopy(FLAGS.HPO_best_metric)                                              # Read the earlier best HPO value 
            earlier_train_best = deepcopy(new_best)                                                         # Read the earlier best train value 
            used_best_val = earlier_HPO_best if hyperparameter_optimization else earlier_train_best         # If we are performing HPO, use the best HPO_metric as baseline, else best training metric 
            new_best, best_epoch = updateLogsFunc(log_file=logs, FLAGS=FLAGS, history=history, best_val=new_best,
                    train_start=train_start_time, epoch_start=epoch_start_time, best_epoch=best_epoch,
                    cur_epoch=FLAGS.HPO_current_trial if hyperparameter_optimization else epoch)
            metrics_has_improved = all(["loss" in FLAGS.eval_metric, new_best < used_best_val]) or all(["loss" not in FLAGS.eval_metric, new_best > used_best_val])
            HPO_visualize = True if metrics_has_improved and hyperparameter_optimization else False
            train_visualize = True if epoch==epoch_next_display or all([metrics_has_improved, hyperparameter_optimization==False]) else False
            if HPO_visualize or train_visualize:                                                            # At least every 'display_rate' epochs or if the model has improved ...
                _,data_batches,config,FLAGS,img_ytrue_ypred = visualize_the_images(config=config, FLAGS=FLAGS, data_batches=data_batches, epoch_num=epoch+1)    # ... the model will segment and save visualizations ...
                if not hyperparameter_optimization:
                    plot_confusion_matrix(config=config, epoch=epoch+1, conf_train=conf_matrix_train, conf_val=conf_matrix_val) # ... and display confusion matrixes 
                    SaveHistory(historyObject=img_ytrue_ypred, save_folder=config.OUTPUT_DIR, historyName="img_ytrue_ypred_trial_{}".format(FLAGS.HPO_current_trial))
            if quit_training == True:                                                                       # If the early stopping callback says we need to quit the training ...
                printAndLog(input_to_write="Committing early stopping at epoch {:d}. The best {:s} is {:.3f} from epoch {:d}".format(epoch+1, FLAGS.eval_metric, new_best, best_epoch), logs=logs)
                break                                                                                       # break the for loop and stop running more epochs
        except Exception as ex:
            error_string = "An exception of type {} occured while doing epoch {}/{}. Arguments:\n{!r}".format(type(ex).__name__, epoch+1, epochs_to_run, ex.args)
            if isinstance(ex, FloatingPointError):
                printAndLog(input_to_write="Lowering the learning rate as the loss became nan or inf", logs=logs)
                config.SOLVER.BASE_LR = config.SOLVER.BASE_LR * FLAGS.lr_gamma
            printAndLog(input_to_write=error_string, logs=logs, prefix="", postfix="\n")

    # Evaluation on the vitrolife test dataset. There is no ADE20K-test dataset.
    test_history = {}                                                                                       # Initialize the test_history dictionary as an empty dictionary
    if all([FLAGS.debugging == False, "vitrolife" in FLAGS.dataset_name.lower(), hyperparameter_optimization==False]):  # Inference will only be performed when training the Vitrolife model
        config.DATASETS.TEST = ("vitrolife_dataset_test",)                                                  # The inference will be done on the test dataset
        eval_test_results,_,_,conf_matrix_test, PN_test_pred_count, PN_test_true_count = evaluateResults(FLAGS, config, data_split="test")  # Evaluate the result metrics on the validation set with the best performing model
        _ = plot_confusion_matrix(config=config, conf_train=conf_matrix_train, conf_val=conf_matrix_val, conf_test=conf_matrix_test, done_training=True)
        test_pq_results = pq_evaluation(args=FLAGS, config=config, data_split="test")                       # Evaluate the Panoptic Quality for the test semantic segmentation results
        history_test = combineDataToHistoryDictionaryFunc(config=config, eval_metrics=eval_test_results["sem_seg"], pq_metrics=test_pq_results, data_split="test")
        for key in history_test.keys():                                                                     # Iterate over all the keys in the history dictionary
            if "test" in key: test_history[key] = history_test[key][-1]                                     # If "test" is in the key, assign the value to the test_dictionary 

    # Return the results
    if hyperparameter_optimization: return new_best
    else: return history, test_history, new_best, best_epoch, config, PN_test_pred_count, PN_test_true_count
    