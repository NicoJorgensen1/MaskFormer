# Import the libraries and functions used here
import sys
import os
import torch
import numpy as np
from copy import deepcopy
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.evaluation import SemSegEvaluator
from detectron2.engine.defaults import DefaultPredictor
from tqdm import tqdm


# Create a custom 'process' function, where the mask GT image, that is send with the input dictionary is actually used...
class My_Evaluator(SemSegEvaluator):
    def process(self, inputs, outputs, gts):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output, gt in zip(inputs, outputs, gts):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            gt = np.asarray(gt.numpy()).astype(np.uint8)
            gt[gt == self._ignore_label] = self._num_classes                                                # All pixels in the "ignore_label" class will be set to the "auxillary" class
            self._conf_matrix += np.bincount((self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),   # 
                minlength=self._conf_matrix.size,).reshape(self._conf_matrix.shape)                         #
            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))                    # 


# Define the evaluation function
def evaluateResults(FLAGS, cfg, data_split="train",  dataloader=None, evaluator=None, hp_optim=False):
    # Get the correct properties
    dataset_name = cfg.DATASETS.TRAIN[0] if "train" in data_split.lower() else cfg.DATASETS.TEST[0]         # Get the name of the dataset that will be evaluated
    total_runs = FLAGS.num_train_files if "train" in data_split.lower() else FLAGS.num_val_files            # Get the number of files 
    meta_data = MetadataCatalog.get(dataset_name)
    if "train" in data_split and hp_optim==True: total_runs = 10                                            # If we are performing hyperparameter optimization, only 10 train samples will be evaluated
    if "ade20k" in FLAGS.dataset_name.lower() and hp_optim: total_runs = int(np.ceil(np.divide(total_runs, 4))) # If we are on the ADE20k dataset, then only 1/4 of the dataset will be evaluated during HPO

    pred_out_dir = os.path.join(cfg.OUTPUT_DIR, "Predictions", data_split)                                  # The path of where to store the resulting evaluation
    os.makedirs(pred_out_dir, exist_ok=True)                                                                # Create the evaluation folder, if it doesn't already exist

    # Build the dataloader if no dataloader has been sent to the function as an input
    if dataloader is None:                                                                                  # If no dataloader has been inputted to the function ...
        dataloader = iter(build_detection_train_loader(DatasetCatalog.get(dataset_name),                    # ... create the dataloader for evaluation ...
            mapper=DatasetMapper(cfg, is_train=False, augmentations=[]), total_batch_size=1, num_workers=2))    # ... with batch_size = 1 and no augmentation on the mapper
    
    # Create the predictor and evaluator instances
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_weights.pth") 
    predictor = DefaultPredictor(cfg=cfg)
    if evaluator is None: evaluator = My_Evaluator(dataset_name=dataset_name, output_dir=pred_out_dir)      # If no evaluator has been sent into the function, crate a new one
    evaluator.reset()                                                                                       # Reset the evaluator, i.e. remove all earlier computations and confusion matrixes

    # Create a progress bar to keep track on the evaluation
    PN_count_pred, PN_count_true = list(), list()
    with tqdm(total=total_runs, iterable=None, postfix="Evaluating the {:s} dataset".format(data_split), unit="img",  
            file=sys.stdout, desc="Image {:d}/{:d}".format(1, total_runs), colour="green", leave=True, ascii=True, 
            bar_format="{desc}  | {percentage:3.0f}% | {bar:35}| {n_fmt}/{total_fmt} | [Spent: {elapsed}. Remaining: {remaining} | {postfix}]") as tepoch:
        
        # Predict all the files in the dataset
        for kk, data_batch in enumerate(dataloader):                                                        # Iterate through all batches in the dataloader
            outputs, gt_mask = list(), list()                                                               # Initiate lists to store the predicted arrays and the ground truth tensors
            for data in data_batch:                                                                         # Iterate over all dataset dictionaries in the list
                img = torch.permute(data["image"], (1,2,0)).numpy()                                         # Load the image and convert it into a [H,W,C] numpy array
                gt_mask.append(data["sem_seg"])                                                             # Get the ground truth corresponding to the current image
                out_pred = predictor.__call__(img)                                                          # Make a prediction for the input image
                outputs.append(out_pred)                                                                    # Append the current output to the list of outputs
                if "vitrolife" in dataset_name.lower() and "test" in data_split.lower():                    # If we are working on the Vitrolife test dataset ... 
                    out_img = torch.nn.functional.softmax(torch.permute(out_pred["sem_seg"], (1,2,0)), dim=-1)          # Get the softmax output of the predicted image
                    out_pred_img = torch.argmax(out_img, dim=-1).cpu().numpy()                                          # Convert the predicted image into a numpy mask 
                    PN_pred_area = np.sum(out_pred_img == np.where(np.in1d(meta_data.stuff_classes, "PN"))[0].item())   # Count the area of predicted PNs
                    PN_count_pred.append(int(np.ceil(np.divide(PN_pred_area, FLAGS.PN_mean_pixel_area))))               # Compute the predicted number of PN's 
                    PN_count_true.append(int(data["image_custom_info"]["PN_image"]))                                    # Get the true number of PN's 
            evaluator.process(data_batch, outputs, gt_mask)                                                 # Process the results by adding the results to the confusion matrix
            tepoch.desc = "Image {:d}/{:d} ".format(kk+1, total_runs)                                       # Update the description of the progress bar
            tepoch.update(1)                                                                                # Update the progress bar
            if kk+1 >= total_runs: break                                                                    # When all images in the dataset has been processed, break the for loop
    
    # Evaluate the dataset results
    eval_metrics_results = evaluator.evaluate()                                                             # Compute the evaluation metrics, all IoU's and pixel accuracies

    # Return the results
    return eval_metrics_results, dataloader, evaluator, evaluator._conf_matrix, PN_count_pred, PN_count_true

