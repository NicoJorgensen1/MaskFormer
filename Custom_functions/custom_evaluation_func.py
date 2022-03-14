# Import the libraries and functions used here
import sys
import torch
from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.evaluation import SemSegEvaluator
from detectron2.engine.defaults import DefaultPredictor
from visualize_vitrolife_batch import putModelWeights
from custom_goto_trainer_class import My_GoTo_Trainer
from tqdm import tqdm

def evaluateResults(FLAGS, cfg, data_split="train", trainer=My_GoTo_Trainer):
    # Get the correct properties
    cfg = putModelWeights(config=cfg)
    dataset_name = cfg.DATASETS.TRAIN[0] if "train" in data_split.lower() else cfg.DATASETS.TEST[0]         # Get the name of the dataset that will be evaluated
    dataset_num_files = FLAGS.num_train_files if "train" in data_split.lower() else FLAGS.num_val_files     # Get the number of files 

    # Create the predictor and evaluator instances
    predictor = DefaultPredictor(cfg=cfg)
    evaluator = SemSegEvaluator(dataset_name=dataset_name, output_dir=cfg.OUTPUT_DIR)
    evaluator.reset()
    model = trainer.build_model(cfg)

    # Build the dataloader.
    dataloader = iter(build_detection_train_loader(DatasetCatalog.get(dataset_name),
        mapper=DatasetMapper(cfg, is_train=False), total_batch_size=1, num_workers=1))

    with tqdm(total=dataset_num_files, iterable=None, postfix=None, unit="img", ascii=True, 
    file=sys.stdout, desc="Image {:d}/{:d}".format(1, dataset_num_files), colour="green", leave=True,
    bar_format="{desc}{percentage:3.0f}% | {bar:35}| {n_fmt}/{total_fmt} [Spent: {elapsed}. Remaining: {remaining}{postfix}]") as tepoch:     
        
        # Predict all the files in the dataset
        for kk, data_batch in enumerate(dataloader):
            outputs, gt_mask = list(), list()
            for data in data_batch:
                img = torch.permute(data["image"], (1,2,0)).numpy()
                gt_mask.append(data["sem_seg"])
                out_pred = predictor.__call__(img)
                outputs.append(out_pred)
            evaluator.process(data_batch, outputs)
            tepoch.desc = "Image {:d}/{:d} ".format(kk+1, dataset_num_files)
            tepoch.update(1)
            if kk+1 >= dataset_num_files:
                break
    
    # Evaluate the dataset results
    eval_metrics_results = evaluator.evaluate()

    # Return the results
    return eval_metrics_results

