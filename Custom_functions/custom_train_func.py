# Import libraries 
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from custom_goto_trainer_class import My_GoTo_Trainer


def setup(FLAGS):
    cfg = FLAGS.config                           # Create the custom config as an independent file
    default_setup(cfg, FLAGS)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def run_train_func(FLAGS):
    cfg = setup(FLAGS)

    if FLAGS.eval_only:
        model = My_GoTo_Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=FLAGS.resume)
        res = My_GoTo_Trainer.test(cfg, model)
        return res

    trainer = My_GoTo_Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer


# Function to launch the training
def launch_custom_training(FLAGS, config):
    FLAGS.config = config
    trainer_class = run_train_func(FLAGS=FLAGS)
    return trainer_class
