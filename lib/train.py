from dataclasses import dataclass
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers

from lib.config import config


# **** descriptor ****

@dataclass
class TrainDescriptor:
    model: pl.LightningModule
    num_epochs: int
    pbar: bool
    accumulate_grad_batches: int = 1
    device: str = "cpu"
    val_check_interval: float = 1.0

    def __getstate__(self):
        d = self.__dict__
        del d['model']
        return d


def _Train():
    def transformer(func, sub_proc_wrapped):
        trainer_info = sub_proc_wrapped[1]

        model = trainer_info.model

        set_pbar = hasattr(trainer_info, "pbar")

        def wrapper(tvt_data_loaders):
            # setup tensorboard
            logdir = os.path.join(config['app_path'], "logs/tensorboard")
            os.makedirs(logdir, exist_ok=True)
            logger = loggers.TensorBoardLogger(name=model.__class__.__name__, save_dir=logdir)

            trainer = pl.Trainer(max_epochs=trainer_info.num_epochs, logger=logger,
                                 accumulate_grad_batches=trainer_info.accumulate_grad_batches,
                                 accelerator=trainer_info.device, precision=16,
                                 num_sanity_val_steps=0, val_check_interval=trainer_info.val_check_interval)

            trainer.fit(model, tvt_data_loaders[0], tvt_data_loaders[1])
            trainer.test(model, tvt_data_loaders[2])

        return wrapper

    return transformer
