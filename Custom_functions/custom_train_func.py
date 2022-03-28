# Import libraries 
import shutil                                                                                               # Used to copy/rename the metrics.json file after each training/validation step
import os
from copy import deepcopy
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from custom_goto_trainer_class import My_GoTo_Trainer
from visualize_image_batch import putModelWeights


def setup(FLAGS):
    cfg = FLAGS.config                                                                                      # Create the custom config as an independent file
    default_setup(cfg, FLAGS)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def run_train_func(FLAGS, run_mode):
    cfg = setup(FLAGS)

    # if trainer is None:
    Trainer = My_GoTo_Trainer(cfg)
    Trainer.resume_or_load(resume=False)
    return Trainer.train()


# Function to launch the training
def launch_custom_training(FLAGS, config, dataset, epoch=0, run_mode="train"):
    config.SOLVER.MAX_ITER = FLAGS.epoch_iter * (25 if all(["train" in run_mode, epoch>0]) else 1)          # Increase training iteration count for precise BN computations
    config.SOLVER.CHECKPOINT_PERIOD = config.SOLVER.MAX_ITER                                                # Save a new model checkpoint after each epoch
    if epoch==0 and "train" in run_mode: config.custom_key.append(tuple(("epoch_num", epoch)))              # Append the current epoch number to the custom_key list in the config ...
    if "train" in run_mode:                                                                                 # If we are training ... 
        for idx, item in enumerate(config.custom_key[::-1]):                                                # Iterate over the custom keys in reversed order
            if "epoch_num" in item[0]:                                                                      # If the current item is the tuple with the epoch_number
                config.custom_key[-idx-1] = (item[0], item[1]+1)                                            # The current epoch number is updated 
                break                                                                                       # And the loop is broken out of 
    FLAGS.config = config                                                                                   # Save the config on the FLAGS argument
    config = putModelWeights(config)                                                                        # Assign the latest saved model to the config
    if "val" in run_mode.lower(): config.SOLVER.BASE_LR = float(0)                                          # If we are on the validation split set the learning rate to 0
    else: config.SOLVER.BASE_LR = FLAGS.learning_rate                                                       # Else, we are on the training split, so assign the latest saved learning rate to the config
    config.DATASETS.TRAIN = dataset                                                                         # Change the config dataset used to the dataset sent along ...
    run_train_func(FLAGS=FLAGS, run_mode=run_mode)                                                          # Run the training for the current epoch
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "metrics.json"),                                        # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, run_mode+"_metrics_{:d}.json".format(epoch+1)))                     # ... where X is the current epoch number
    os.remove(os.path.join(config.OUTPUT_DIR, "metrics.json"))                                              # Remove the original metrics file
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "model_final.pth"),                                     # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, "model_epoch_{:d}.pth".format(epoch+1)))                            # ... where X is the current epoch number    
    [os.remove(os.path.join(config.OUTPUT_DIR, x)) for x in os.listdir(config.OUTPUT_DIR) if "model_" in x and "epoch" not in x and x.endswith(".pth")]  # Remove all irrelevant models
    return config
