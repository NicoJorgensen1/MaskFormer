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
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Alting", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")             # Home WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                              # Home windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Datasets")                                      # Work WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                              # Work windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                          # Larac server
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 2)[2])                                              # Balder server
assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"
os.environ["DETECTRON2_DATASETS"] = dataset_dir

# Import important libraries
import numpy as np 
from custom_callback_functions import keepAllButLatestAndBestModel                                  # Used for setting model weights on the config
from custom_setup_func import printAndLog, getBestEpochResults, zip_output, write_config_to_file, SaveHistory   # Log the results, get metrics from the best epoch, zip output directory and write config to file
from custom_train_func import objective_train_func                                                  # Function to launch the training with the given dataset
from visualize_image_batch import visualize_the_images                                              # Functions visualize the image batch
from custom_model_analysis_func import analyze_model_func                                           # Analyze the model FLOPS, number of parameters and activations computed
from custom_HPO_function import perform_HPO                                                         # Function to perform HPO and read the input variables

# Get the FLAGS, the config and the logfile. 
FLAGS, cfg, trial, log_file = perform_HPO()                                                         # Perform HPO if that is chosen 
write_config_to_file(config=cfg)                                                                    # Save the config file with the final parameters used in the output dir
printAndLog(input_to_write="FLAGS input arguments:", logs=log_file)                                 # Print the new, updated FLAGS ...
printAndLog(input_to_write={key: vars(FLAGS)[key] for key in sorted(vars(FLAGS).keys()) if key != "history"},   # ...  input arguments to the logfile ...
            logs=log_file, oneline=False, length=27)                                                # ... sorted by the key names 

# Analyze the model with the found parameters from the HPO
data_batches = None 
if not FLAGS.inference_only:
    model_analysis, FLAGS = analyze_model_func(config=cfg, args=FLAGS)                              # Analyze the model with the FLAGS input parameters
    printAndLog(input_to_write="Model analysis:".upper(), logs=log_file)                            # Print the model analysis ...
    printAndLog(input_to_write=model_analysis, logs=log_file, oneline=False, length=27)             # ... and write it to the logfile

    # Visualize some random images before training 
    fig_list_before, data_batches, cfg, FLAGS, _ = visualize_the_images(config=cfg, FLAGS=FLAGS)    # Visualize some segmentations on random images before training

# Train the model with the best found hyperparameters
history, test_history, new_best, best_epoch, cfg, PN_pred, PN_true = objective_train_func(trial=trial,  # Start the training with ...
    FLAGS=FLAGS, cfg=cfg, logs=log_file, data_batches=data_batches, hyperparameter_optimization=False)  # ... the optimal hyper parameters

# Visualize the same images, now after training
if not FLAGS.inference_only:
    cfg = keepAllButLatestAndBestModel(cfg=cfg, history=history, FLAGS=FLAGS, bestOrLatest="best")  # Put the model weights for the best performing model on the config
write_config_to_file(config=cfg)                                                                    # Save the config file with the final parameters used in the output dir
visualize_the_images(config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_done_training=True)   # Visualize the images again

# Print and log the best metric results
printAndLog(input_to_write="Final results:".upper(), logs=log_file)
if FLAGS.inference_only==False: 
    printAndLog(input_to_write="Best validation results:".ljust(30)+"Epoch {:d}: {:s} = {:.3f}\n{:s}".
        format(best_epoch, FLAGS.eval_metric, new_best, "All best validation results:".upper().ljust(30)), logs=log_file)
    printAndLog(input_to_write=getBestEpochResults(history, best_epoch), logs=log_file, prefix="", length=15)
if "vitrolife" in FLAGS.dataset_name.lower():                                                       # As only the Vitrolife dataset includes a test set...
    printAndLog(input_to_write="All test results:".upper().ljust(30), logs=log_file)
    PN_accuracy = np.divide(np.sum(np.asarray(PN_pred) == np.asarray(PN_true)), len(PN_true))       # Compute the accuracy of computed PNs 
    printAndLog(input_to_write=test_history, logs=log_file, prefix="", length=15)
    test_history["PN_pred"] = PN_pred                                                               # Assign the list of predicted PNs to the test history
    test_history["PN_true"] = PN_true                                                               # Assign the list of true PNs to the test history 
    printAndLog(input_to_write="The PN counts of the test dataset has an accuracy of {:.3f}".format(PN_accuracy), logs=log_file, postfix="\n")
SaveHistory(historyObject=test_history, save_folder=cfg.OUTPUT_DIR, historyName="test_history")     # Save the test history to the output folder

# Remove all metrics.json files, the default log-file and zip the resulting output directory
[os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if "metrics" in x.lower() and x.endswith(".json")]

zip_output(cfg)




# from detectron2.modeling import build_model
# from copy import deepcopy 
# from detectron2.structures import ImageList
# from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader

# from create_custom_config import createVitrolifeConfiguration, changeConfig_withFLAGS
# from custom_goto_trainer_class import custom_augmentation_mapper


# FLAGS.use_transformer_backbone = True    
# FLAGS.use_per_pixel_baseline = True 
# FLAGS.resnet_depth = 101
# FLAGS.use_checkpoint = False
# FLAGS.num_queries = 80

# del config, dataloader, model_test, features, outputs 

# config = createVitrolifeConfiguration(FLAGS)
# config_folder = os.path.join(MaskFormer_dir, "configs", "ade20k-150")
# if FLAGS.use_per_pixel_baseline:
#     config.merge_from_file(os.path.join(config_folder, "per_pixel_baseline_R"+"{:d}".format(50)+"_bs16_160k.yaml"))    # ... only R50 per-pixel baseline config is available 
# if FLAGS.use_transformer_backbone:
#     swin_type = "base" 
#     swin_config = [x for x in os.listdir(os.path.join(config_folder, "swin")) if all([swin_type in x, x.endswith(".yaml")])][-1] 
#     config.merge_from_file(os.path.join(config_folder, "swin", swin_config))             # ... merge with the config for the chosen swin transformer
# config = changeConfig_withFLAGS(cfg=config, FLAGS=FLAGS)
# if FLAGS.use_transformer_backbone:
#     config.MODEL.DEVICE = "cpu"

# model_test = build_model(cfg=config)
# dataloader = iter(build_detection_train_loader(DatasetCatalog.get(config.DATASETS.TRAIN[0]),                    # ... create the dataloader for evaluation ...
#     mapper=custom_augmentation_mapper(config, is_train=True), total_batch_size=1, num_workers=2))          # ... with batch_size = 1 and no augmentation on the mapper

# batched_inputs = next(dataloader) 

# images = [x["image"].to(model_test.device) for x in batched_inputs]
# images = [(x - model_test.pixel_mean) / model_test.pixel_std for x in images]
# images = ImageList.from_tensors(images, model_test.size_divisibility)

# features = model_test.backbone(images.tensor)
# outputs = model_test.sem_seg_head(features)

# pred_logits = outputs["pred_logits"]
# pred_masks = outputs["pred_masks"]

# model_test.backbone
# print("\nInput img.shape: {}. res4.shape: {}. res5.shape: {}.\npred_logits.shape: {}, pred_masks.shape: {}\n".format(batched_inputs[0]["image"].numpy().shape, features["res4"].detach().cpu().numpy().shape, features["res5"].detach().cpu().numpy().shape, pred_logits.shape, pred_masks.shape))


