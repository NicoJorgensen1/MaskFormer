# Import libraries 
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup, launch
from detectron2.evaluation import verify_results
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
        if cfg.TEST.AUG.ENABLED:
            res.update(My_GoTo_Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = My_GoTo_Trainer(cfg)
    trainer.resume_or_load(resume=FLAGS.resume)
    trainer.train()
    return


# Function to launch the training
def launch_custom_training(FLAGS, config):
    print("Command Line FLAGS:", FLAGS)
    FLAGS.config = config
    launch(
        run_train_func,
        FLAGS.num_gpus,
        num_machines=FLAGS.num_machines,
        machine_rank=FLAGS.machine_rank,
        dist_url=FLAGS.dist_url,
        FLAGS=(FLAGS,),
    )
