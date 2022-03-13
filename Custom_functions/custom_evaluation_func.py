# Import the libraries and functions used here
import torch
from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.evaluation import SemSegEvaluator
from detectron2.engine.defaults import DefaultPredictor

def evaluateResults(FLAGS, cfg, data_split="train"):
    # Get the correct properties
    dataset_name = cfg.DATASETS.TRAIN[0] if "train" in data_split.lower() else cfg.DATASETS.TEST[0]         # Get the name of the dataset that will be evaluated
    dataset_num_files = FLAGS.num_train_files if "train" in data_split.lower() else FLAGS.num_val_files     # Get the number of files 

    # Create the predictor and evaluator instances
    predictor = DefaultPredictor(cfg=cfg)
    evaluator = SemSegEvaluator(dataset_name=dataset_name, output_dir=cfg.OUTPUT_DIR)
    evaluator.reset()

    # Build the dataloader.
    dataloader = iter(build_detection_train_loader(DatasetCatalog.get(dataset_name),
        mapper=DatasetMapper(cfg, is_train=False), total_batch_size=1, num_workers=1))
    
    # Predict all the files in the dataset
    for kk, data_batch in enumerate(dataloader):
        outputs = list()
        for data in data_batch:
            img = torch.permute(data["image"], (1,2,0)).numpy()
            out_pred = predictor.__call__(img)
            outputs.append(out_pred)
        evaluator.process(data_batch, outputs)
        print(kk)
        
        if kk+1 >= dataset_num_files:
            break
    
    # Evaluate the dataset results
    eval_metrics_results = evaluator.evaluate()

    # Return the results
    return eval_metrics_results



