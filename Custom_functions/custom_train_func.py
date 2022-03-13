# Import libraries 
from custom_goto_trainer_class import My_GoTo_Trainer


def setup(args):
    cfg = args.config                           # Create the custom config as an independent file
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = My_GoTo_Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = My_GoTo_Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(My_GoTo_Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = My_GoTo_Trainer(cfg)
    # val_loss_hook = ValLossHook(cfg=cfg)
    # trainer.register_hooks([val_loss_hook])
    # Implement BestCheckpointer hook somehow...
    # best_checkpoint_hook = BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, checkpointer=Checkpointer, val_metric="total_loss", mode="min", file_prefix="model_best")
    # periodic_writer_hook = [hook for hook in trainer._hooks if isinstance(hook, PeriodicWriter)]
    # all_other_hooks = [hook for hook in trainer._hooks if not isinstance(hook, PeriodicWriter) and not isinstance(hook, PeriodicCheckpointer)]
    # trainer.register_hooks(hooks=all_other_hooks + periodic_writer_hook)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


# Function to launch the training
def launch_custom_training(args, config):
    print("Command Line Args:", args)
    args.config = config
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
