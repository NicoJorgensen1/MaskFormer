# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py

from distutils.command.config import config
import logging
import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)
from detectron2.utils.logger import setup_logger

# fmt: off
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

logger = logging.getLogger("detectron2")


def setup(args, config):
    setup_logger(name="fvcore", output=os.path.join(config.OUTPUT_DIR, "analyze_model_output.txt"))
    setup_logger()
    return config


def do_flop(cfg):
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        if args.use_fixed_input_size and isinstance(cfg, CfgNode):
            import torch
            crop_size = cfg.INPUT.CROP.SIZE[0]
            data[0]["image"] = torch.zeros((3, crop_size, crop_size))
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )
    return "Total GFlops: {:.1f} ±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)


def do_activation(cfg):
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))
    # logger.info(
    #     "(Million) Activations for Each Type of Operators:\n"
    #     + str([(k, v / idx) for k, v in counts.items()])
    # )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )
    return "Total Activations: {:.3f}M ±{}".format(np.mean(total_activations), np.std(total_activations))


def do_parameter(cfg):
    model = build_model(cfg)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))
    # return "Parameter Count:\n" + parameter_count_table(model, max_depth=5)
    return parameter_count_table(model, max_depth=5)


def do_structure(cfg):
    model = build_model(cfg)
    logger.info("Model Structure:\n" + str(model))
    return "Model Structure:\n" + str(model)


# if __name__ == "__main__":
def analyze_model_func(config):
    if __name__ == "__main__":
        parser = default_argument_parser()
        parser.add_argument(
            "--tasks",
            choices=["flop", "activation", "parameter", "structure"],
            default=["flop", "activation", "parameter"],
            required=False,
            nargs="+",
        )
        parser.add_argument(
            "-n",
            "--num-inputs",
            default=1,
            type=int,
            help="number of inputs used to compute statistics for flops/activations, "
            "both are data dependent.",
        )
        parser.add_argument(
            "--use-fixed-input-size",
            action="store_true",
            help="use fixed input size when calculating flops",
        )
        args = parser.parse_args()
        assert not args.eval_only
        assert args.num_gpus == 1

        config = setup(args, config)
        
        # for task in args.tasks:
        res = { "Flops": do_flop(config),
                "Parameters": "".join([x.strip() for x in do_parameter(config).split("|")[8]]),
                "Activations": do_activation(config)}
    return res

# from copy import deepcopy
# res = analyze_model_func(config=deepcopy(cfg))