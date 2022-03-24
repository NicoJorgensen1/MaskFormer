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


def run_train_func(FLAGS, trainer, run_mode):
    cfg = setup(FLAGS)

    if FLAGS.eval_only:
        model = My_GoTo_Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=FLAGS.resume)
        res = My_GoTo_Trainer.test(cfg, model)
        return res
    
    # if trainer is None:
    trainer = My_GoTo_Trainer(cfg)
    trainer.resume_or_load(resume=False)
    # else:
    #     trainer.resume_or_load(resume=True)                                                               # We'll resume training 
        # trainer.max_iter += FLAGS.epoch_iter+1                                                            # Extend the max_iter with the epoch_iter to continue training
        # trainer.build_writers()                                                                           # For some reason we have to re-build the writers in order to make a new metrics.json file
    #     trainer.iter += 1
    #     trainer.start_iter += 1
    trainer.train()
    return trainer


# Function to launch the training
def launch_custom_training(FLAGS, config, dataset, epoch=0, run_mode="train", trainer=None):
    FLAGS.config = config                                                                                   # Save the config on the FLAGS argument
    config = putModelWeights(config)                                                                        # Assign the latest saved model to the config
    if "val" in run_mode.lower(): config.SOLVER.BASE_LR = float(0)                                          # If we are on the validation split set the learning rate to 0
    else: config.SOLVER.BASE_LR = FLAGS.learning_rate                                                       # Else, we are on the training split, so assign the latest saved learning rate to the config
    config.DATASETS.TRAIN = dataset                                                                         # Change the config dataset used to the dataset sent along ...
    trainer_class = run_train_func(FLAGS=FLAGS, trainer=trainer, run_mode=run_mode)                         # Run the training for the current epoch
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "metrics.json"),                                        # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, run_mode+"_metrics_{:d}.json".format(epoch+1)))                     # ... where X is the current epoch number
    os.remove(os.path.join(config.OUTPUT_DIR, "metrics.json"))                                              # Remove the original metrics file
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "model_final.pth"),                                     # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, "model_epoch_{:d}.pth".format(epoch+1)))                            # ... where X is the current epoch number    
    [os.remove(os.path.join(config.OUTPUT_DIR, x)) for x in os.listdir(config.OUTPUT_DIR) if "model_" in x and "epoch" not in x and x.endswith(".pth")]  # Remove all irrelevant models
    return config, trainer_class
