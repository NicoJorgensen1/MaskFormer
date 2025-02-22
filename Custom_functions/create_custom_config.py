import os                                                                                   # Used to navigate around the current os and folder structure
import torch                                                                                # torch is implemented to check if a GPU is available
import numpy as np                                                                          # Used to compute the step ranges
from sys import path as sys_PATH                                                            # Import the PATH variable
from detectron2.data import MetadataCatalog                                                 # Catalog containing metadata for all datasets available in Detectron2
from detectron2.config import get_cfg                                                       # Function to get the default configuration from Detectron2
from detectron2.projects.deeplab import add_deeplab_config                                  # Used to merge the default config with the deeplab config before training
from mask_former import add_mask_former_config                                              # Used to add the new configuration to the list of possible configurations


# Locate the folder containing other configurations
MaskFormer_dir = [x for x in sys_PATH if x.endswith("MaskFormer")][0]
checkpoint_dir = os.path.join(MaskFormer_dir, "maskformer_model_checkpoints")
config_folder = os.path.join(MaskFormer_dir, "configs", "ade20k-150")


# Define function to get all keys in a nested dictionary
def accumulate_keys(dct):
    key_list = list()
    def accumulate_keys_recursive(dct):
        for key in dct.keys():
            if isinstance(dct[key], dict): accumulate_keys_recursive(dct[key])
            else: key_list.append(key.upper())
    accumulate_keys_recursive(dct)
    return key_list


# Define a function to create a custom configuration in the chosen config_dir and takes a namespace option
def createVitrolifeConfiguration(FLAGS):
    cfg = get_cfg()                                                                         # Get the default configuration from detectron2.
    add_deeplab_config(cfg)                                                                 # Add some deeplab (i.e. sem_seg) config values
    add_mask_former_config(cfg)                                                             # Add some default values used for semantic segmentation to the config and choose datasetmapper
    if FLAGS.use_transformer_backbone==True:                                                # If the user chose the transformer backbone ...
        swin_type = "tiny" if any([x in MaskFormer_dir.lower() for x in ["nico", "wd974261"]]) and not FLAGS.inference_only else "base" # If on home computer, use swin tiny. If on gpucluster, use swin base
        swin_config = [x for x in os.listdir(os.path.join(config_folder, "swin")) if all([swin_type in x, x.endswith(".yaml")])][-1]    # Find the corresponding swin config
        cfg.merge_from_file(os.path.join(config_folder, "swin", swin_config))               # ... merge with the config for the chosen swin transformer
        if FLAGS.use_checkpoint==True:                                                      # If the user choose to start training from a earlier checkpoint ...
            checkpoint_file = [x for x in os.listdir(checkpoint_dir) if all([swin_type in x, x.endswith(".pkl")])][-1]  # Find the checkpoint file for the chosen swin transformer
            cfg.MODEL.WEIGHTS = os.path.join(checkpoint_dir, checkpoint_file)               # Load the swin_transformer checkpoint
    elif FLAGS.use_transformer_backbone==False:                                             # Else-if the user will use regular ResNet backbone ...
        cfg.merge_from_file(os.path.join(config_folder, "maskformer_R"+"{:d}".format(FLAGS.resnet_depth)+"_bs16_160k.yaml"))    # ... merge with the resnet config
        cfg.MODEL.RESNETS.DEPTH = FLAGS.resnet_depth                                        # Assign the depth of the ResNet backbone feature extracting model
        if FLAGS.use_checkpoint==True:                                                      # If the user choose to start training from a earlier checkpoint ...
            cfg.MODEL.WEIGHTS = os.path.join(checkpoint_dir, "maskformer_resnet_backbone_checkpoint.pkl")   # Load the resnet_backbone checkpoint
    if FLAGS.use_per_pixel_baseline==True:                                                  # Otherwise, then the user chose to work with the per_pixel_calculating baselines, so ...
        cfg.merge_from_file(os.path.join(config_folder, "per_pixel_baseline_R"+"{:d}".format(50)+"_bs16_160k.yaml"))    # ... only R50 per-pixel baseline config is available 
        if FLAGS.use_checkpoint==True:                                                      # If the user choose to start training from a earlier checkpoint ...
            cfg.MODEL.WEIGHTS = os.path.join(checkpoint_dir, "maskformer_per_pixel_baseline_checkpoint.pkl")    # Load the per_pixel classification checkpoint
    # cfg.merge_from_file(os.path.join(config_folder, "Base-ADE20K-150.yaml"))                # Merge with the base config for ade20K dataset. This is the config selecting that we use the ADE20K dataset
    if "vitrolife" in FLAGS.dataset_name.lower():                                           # If the vitrolife dataset was chosen ...
        cfg["DATASETS"]["TRAIN"] = ("vitrolife_dataset_train",)                             # ... define the training dataset by using the config as a dictionary
        cfg.DATASETS.TEST = ("vitrolife_dataset_val",)                                      # ... define the validation dataset by using the config as a CfgNode 
    if "false" not in FLAGS.model_weights_used.lower():
        cfg.MODEL.WEIGHTS = FLAGS.model_weights_used 
    cfg.OUTPUT_DIR = os.path.join(MaskFormer_dir, "output_{:s}{:s}".format("vitrolife_" if "vitro" in FLAGS.dataset_name.lower() else "", FLAGS.output_dir_postfix))    # Get MaskFormer directory and name the output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)                                              # Create the output folder, if it doesn't already exist
    return cfg


# Define a function to change the config based on the input arguments
def changeConfig_withFLAGS(cfg, FLAGS):
    if FLAGS.use_checkpoint == False: cfg.MODEL.WEIGHTS = ""                                # If the model must be trained without using earlier checkpoints, any earlier checkpoint must be removed...
    cfg.SOLVER.BASE_LR = FLAGS.learning_rate                                                # Starting learning rate
    cfg.SOLVER.IMS_PER_BATCH = int(FLAGS.batch_size)                                        # Batch size used when training => batch_size pr GPU = batch_size // num_gpus
    cfg.SOLVER.MAX_ITER = FLAGS.epoch_iter                                                  # <<< Deprecated input argument: Use --num_epochs instead >>>
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"                                         # Default learning rate scheduler
    cfg.SOLVER.OPTIMIZER = FLAGS.optimizer_used.upper()                                     # The optimizer to use for training the model
    cfg.SOLVER.NESTEROV = True                                                              # Whether or not the learning algorithm will use Nesterow momentum
    cfg.SOLVER.WEIGHT_DECAY = FLAGS.weight_decay                                            # A small lambda value for the weight decay. It is larger when training with transformers
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True                                                # We need to clip the gradients during training in order to avoid exploding gradients, especially when performing HPO
    cfg.SOLVER.BACKBONE_MULTIPLIER = FLAGS.backbone_multiplier                              # Backbone learning rate = learning_rate * backbone_multiplier
    cfg.TEST.AUG = False                                                                    # No augmentation used for inference
    cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED = False                                          # Disable the panoptic head during inference
    cfg.DATALOADER.NUM_WORKERS = FLAGS.num_workers                                          # Set the number of workers to only 2
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False                                            # We'll simply shuffle the input data, we won't group them after aspect ratios, even though that would be more GPU efficient
    cfg.INPUT.CROP.ENABLED = FLAGS.crop_enabled                                             # We will not allow any cropping of the input images
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() and not FLAGS.inference_only else 'cpu'  # Assign the device on which the model should run
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = FLAGS.dice_loss_weight                              # Set the weight for the dice loss (original 2)
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = FLAGS.mask_loss_weight                              # Set the weight for the mask predictive loss (original 20)
    cfg.MODEL.MASK_FORMER.DROPOUT = FLAGS.dropout                                           # We'll set a dropout probability on 0.10 when training
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = float(0.10)                                    # The loss weight for the "no-object" label
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = float(0.025)                             # The threshold for overlapping masks
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False                                          # Disable the panoptic head for the maskformer 
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = FLAGS.num_queries                            # The number of queries to detect from the Transformer module 
    cfg.MODEL.BACKBONE.FREEZE_AT = FLAGS.backbone_freeze_layers                             # The number of backbone layers to freeze of the backbone 
    cfg.MODEL.RESNETS.DEPTH = FLAGS.resnet_depth                                            # The number of layers in the ResNet backbone 
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1                                                  # Increase loss weight for the sem_seg_head
    cfg.TEST.EVAL_PERIOD = 0                                                                # We won't use the build in evaluation, only the custom evaluation function
    cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.MAX_ITER*1e5)                             # Save a new model checkpoint after each epoch, i.e. after everytime the entire trainining set has been seen by the model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05                                            # Assign the IoU threshold used for the model
    cfg.INPUT.FORMAT = "BGR"                                                                # The input format is set to be BGR, like the visualization method
    if "vitrolife" in FLAGS.dataset_name.lower():                                           # If the vitrolife dataset was chosen ...
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(MetadataCatalog[cfg.DATASETS.TEST[0]].stuff_classes)   # Assign the number of classes for the model to segment
        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = FLAGS.ignore_label                            # If we are training on the Vitrolife dataset, HPO will be able to enable/disable whether or not background should be displayed...
        cfg.INPUT.SIZE_DIVISIBILITY = FLAGS.img_size_min                                    # Resize all images into (500, 500) 
        cfg.INPUT.MIN_SIZE_TRAIN = FLAGS.img_size_min                                       # The minimum size length for one side of the training images
        cfg.INPUT.MAX_SIZE_TRAIN = FLAGS.img_size_max                                       # The maximum size length for one side of the training images
        cfg.INPUT.MIN_SIZE_TEST = FLAGS.img_size_min                                        # The minimum size length for one side of the validation images
        cfg.INPUT.MAX_SIZE_TEST = FLAGS.img_size_max                                        # The maximum size length for one side of the validation images
        cfg.MODEL.PIXEL_MEAN = [100.15, 102.03, 103.89]                                     # Write the correct image mean value for the entire vitrolife dataset
        cfg.MODEL.PIXEL_STD = [57.32, 59.69, 61.93]                                         # Write the correct image standard deviation value for the entire vitrolife dataset
        cfg.SOLVER.STEPS = []                                                               # <<< Deprecated input argument: Use --patience instead >>>
        cfg.SOLVER.GAMMA = 1                                                                # After every "step" iterations the learning rate will be updated, as new_lr = old_lr*gamma
    if FLAGS.debugging==True:                                                               # If we are debugging the model ...
        cfg.SOLVER.WEIGHT_DECAY = float(0)                                                  # ... we don't want any weight decay
        cfg.MODEL.MASK_FORMER.DROPOUT = float(0)                                            # ... we don't wany any dropout

    # Change the config and add the FLAGS input arguments one by one ... Not pretty, but efficient and doesn't cost memory...
    cfg.custom_key = []
    for key, value in vars(FLAGS).items():
        if any([x.lower() in key.lower() for x in ["batch_size"]]):
            value = int(value) 
        cfg.custom_key.append(tuple((key, value)))

    # Return the custom configuration
    return cfg

