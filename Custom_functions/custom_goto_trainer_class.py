import itertools
import os
import torch
import numpy as np
import weakref
import logging
from copy import copy
from detectron2.utils import comm
from typing import Any, Dict, List, Set
from detectron2.data import build_detection_train_loader, MetadataCatalog
from detectron2.data.samplers.distributed_sampler import TrainingSampler
from detectron2.engine import DefaultTrainer, hooks
from detectron2.engine.hooks import PeriodicWriter
from detectron2.engine.train_loop import SimpleTrainer
import math
from detectron2.utils.events import EventWriter
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.file_io import PathManager
from detectron2.engine.defaults import create_ddp_model
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.data import transforms as T
from mask_former import MaskFormerSemanticDatasetMapper
from detectron2.solver.lr_scheduler import LRMultiplier
from fvcore.common.param_scheduler import CosineParamScheduler
from fvcore.nn.precise_bn import get_bn_modules
from PIL import Image

# Define a function that will return a list of augmentations to use for training
def custom_augmentation_mapper(config, is_train=True):
    if "val" in config.DATASETS.TRAIN[0].lower(): transform_list = []       # If we are validating the images, we won't use data augmentation
    else:
        transform_list = [                                                  # Initiate the list of image data augmentations to use
            T.Resize((500,500), Image.BILINEAR),                            # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.Resize
            T.RandomBrightness(0.8, 1.5),                                   # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomBrightness
            T.RandomLighting(0.7),                                          # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomLighting
            T.RandomContrast(0.7, 1.3),                                     # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomContrast
            T.RandomSaturation(0.85, 1.15),                                 # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomSaturation
            T.RandomRotation(angle=[-45, 45]),                              # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomRotation
            T.RandomFlip(prob=0.25, horizontal=True, vertical=False),       # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomLighting 
            T.RandomFlip(prob=0.25, horizontal=False, vertical=True),       # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomLighting  
            T.RandomCrop("relative", (0.75, 0.75)),                         # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomCrop
            T.Resize((500,500), Image.BILINEAR)]                            # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.Resize
    custom_mapper = MaskFormerSemanticDatasetMapper(config, is_train=is_train, augmentations=transform_list)    # Create the mapping from data dictionary to augmented training image
    return custom_mapper

# Cosine lr_scheduler
class CosineParamScheduler2(CosineParamScheduler):
    def __init__(self, start_value, end_value):
        self._start_value = float(start_value)
        self._end_value = float(end_value)
    
    def __call__(self, where):
        where = float(where)
        new_lr = float(self._end_value + 0.5 * (self._start_value - self._end_value) * (1 + math.cos(math.pi * where)))
        return new_lr


# Custom Trainer class build on the DefaultTrainer class. This is mostly copied from the train_net.py
class My_GoTo_Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler2(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, save_to_disk = "train" in cfg.DATASETS.TRAIN[0], trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())


    # We'll only use the custom evaluation, not any build-in method...
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return None

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        mapper = custom_augmentation_mapper(config=cfg)
        return build_detection_train_loader(cfg, mapper=mapper, sampler=TrainingSampler(size=MetadataCatalog[cfg.DATASETS.TRAIN[0]].num_files_in_dataset, shuffle=True))

    @classmethod
    def build_lr_scheduler2(cls, cfg, optimizer, start_val=1, end_val=0.25):
        sched = CosineParamScheduler2(start_value=start_val, end_value=end_val)
        scheduler = LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
        return scheduler
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and clip_norm_val > 0.0)

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                cfg.SOLVER.MAX_ITER,                # Run precise_BN after each epoch
                self.model,                         # Assign the current model that must be used for the precise BN
                self.build_train_loader(cfg),       # Build a new data loader to not affect training
                cfg.SOLVER.MAX_ITER,                # The number of iterations used to compute the precise values
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency, some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            write_period = int(np.min([20, MetadataCatalog[self.cfg.DATASETS.TRAIN[0]].num_files_in_dataset/np.min([25, MetadataCatalog[self.cfg.DATASETS.TRAIN[0]].num_files_in_dataset])]))
            ret.append(PeriodicWriter(self.build_writers(), period=write_period))
        return ret

        