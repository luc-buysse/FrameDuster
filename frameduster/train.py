from dataclasses import dataclass
import torch
import torch.utils.data
import os
import time
from datetime import datetime
import torch.utils.tensorboard
from typing import Callable
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score)
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.config import config


# **** metrics ****

class EvaluationMetrics(object):
    def __init__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.f1_score = f1_score(y_true, y_pred)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)

    def log_test(self):
        logger.info(f"TEST accuracy {self.accuracy:.3f}")
        logger.info(f"TEST precision {self.precision:.3f}")
        logger.info(f"TEST recall {self.recall:.3f}")
        logger.info(f"TEST f1 score {self.f1_score:.3f}")
        logger.info(f"TEST confusion matrix {self.confusion_matrix}")
        display = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix)
        display.plot()
        plt.show()


def write_metrics(
        writer: SummaryWriter,
        epoch: int, metrics:
        EvaluationMetrics,
        descriptor: str = "val"):
    writer.add_scalar("Accuracy/{}_acc".format(descriptor), metrics.accuracy, epoch)
    writer.add_scalar(
        "Classification_metrics/{}_precision".format(descriptor), metrics.precision, epoch
    )
    writer.add_scalar(
        "Classification_metrics/{}_recall".format(descriptor), metrics.recall, epoch
    )
    writer.add_scalar(
        "Classification_metrics/{}_f1score".format(descriptor), metrics.f1_score, epoch
    )


# **** descriptor descriptions ****

@dataclass
class ModelDescriptor:
    model: torch.nn.Module
    device: str
    model_name: str
    model_path: str


@dataclass
class TrainDescriptor:
    loss_fun: Callable
    num_epochs: int
    pbar: bool
    optimizer: Callable
    scheduler: Callable

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['loss_fun']
        return d


# **** model ****

def save_model(
        model: torch.nn.Module,
        model_name: str,
        optimizer,
        scheduler,
        epoch: int,
        seed: int):
    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "seed": seed,
    }
    path = f"{os.environ['APP_PATH']}/data/models/{model_name}_{epoch}"
    torch.save(state_dict, path)
    logger.success(f"Model saved at {path}.")


def _Train(composition=None, split=None, batch_size=50):
    def transformer(func, sub_proc_wrapped):
        model_info = sub_proc_wrapped[1]
        total_train = int(sum(composition.values()) * split[0] / batch_size)
        total_val = int(sum(composition.values()) * split[1] / batch_size)
        total_test = int(sum(composition.values()) * split[2] / batch_size)

        model = model_info.model
        device = model_info.device
        model = model.to(device)
        num_epochs = model_info.num_epochs
        loss_fun = model_info.loss_fun
        optimizer = model_info.optimizer
        scheduler = model_info.scheduler
        try:
            batch_preprocess = model_info.__class__.batch_preprocess
        except:
            batch_preprocess = None
        start_epoch = 0

        set_pbar = hasattr(model_info, "pbar")

        def wrapper(tvt_data_loaders):

            # setup tensorboard
            logdir = os.path.join(config['app_path'], "logs")
            os.makedirs(logdir, exist_ok=True)
            writer = torch.utils.tensorboard.SummaryWriter(
                os.path.join(
                    logdir,
                    datetime.now().strftime("%d%m%y%H%M%S") + ".log"))

            best_acc = 0.0
            best_epoch = 0

            start = time.time()

            for epoch in range(start_epoch, model_info.num_epochs):
                logger.info(f'Epoch {epoch+1}/{model_info.num_epochs}')

                train_epoch(model,
                            tvt_data_loaders[0],
                            loss_fun,
                            writer,
                            optimizer,
                            scheduler,
                            epoch,
                            num_epochs,
                            device,
                            set_pbar,
                            total=total_train,
                            batch_preprocess=batch_preprocess)

                eval_tuple = test(model,
                                  tvt_data_loaders[1],
                                  loss_fun,
                                  writer,
                                  device,
                                  total_val,
                                  batch_preprocess,
                                  set_pbar,
                                  epoch,
                                  descriptor="Validation",
                                  )

                save_model(
                    model=model,
                    model_name=model_info.model_name,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    seed=config['train_seed'])

                if eval_tuple.accuracy > best_acc:
                    best_epoch = epoch
                    best_acc = eval_tuple.accuracy

            # kill persistent workers
            del tvt_data_loaders[0]._iterator, tvt_data_loaders[1]._iterator

            # end of training
            time_elapsed = time.time() - start  # slight error
            logger.success(
                'Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                      time_elapsed % 3600 % 60))
            logger.info("Best Val Acc: {:.4f} in epoch {}".format(best_acc, best_epoch))

            # test
            test_metrics = test(
                model=model,
                data_loader=tvt_data_loaders[2],
                loss_fun=loss_fun,
                writer=writer,
                device=model_info.device,
                set_pbar=model_info.pbar,
                descriptor="Test",
                total=total_test,
                batch_preprocess=batch_preprocess
                )

            test_metrics.log_test()

            save_model(
                model=model,
                model_name=model_info.model_name,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=model_info.num_epochs,
                seed=config['train_seed'])

        return wrapper

    return transformer


def train_epoch(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fun,
        writer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
        num_epochs: int,
        device: str,
        set_pbar: bool,
        total: int,
        batch_preprocess):
    model.zero_grad()
    model.train()
    p_bar = tqdm(total=total) if set_pbar else None

    correct = 0
    total_viewed = 0

    for i, batch in enumerate(data_loader):

        if batch_preprocess:
            batch_images = batch_preprocess(batch[0])
            batch = batch_images, batch[1]

        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # outputs = torch.argmax(outputs, dim=-1)

        loss = loss_fun(outputs, labels)

        loss.backward()
        optimizer.step()

        if p_bar is not None:
            p_bar.set_description(
                f"Train Epoch: {epoch+1}/{num_epochs}. "
                f"Iter: {i + 1}/{total} "
                f"LR: {scheduler.get_last_lr()[0]:.10f}."
            )
            p_bar.update()

        # Tensorboard
        pred_labels = torch.argmax(outputs, dim=-1)
        true_labels = labels
        total_viewed += labels.size(0)
        correct += pred_labels.eq(true_labels).sum().item()
        it = epoch * total + i
        writer.add_scalar('Loss/train', loss.item(), it)
        writer.add_scalar('Accuracy/train', 100. * correct / total_viewed, it)

    scheduler.step()

    if p_bar is not None:
        p_bar.close()


def test(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fun: Callable,
        writer,
        device,
        total: int,
        batch_preprocess,
        set_pbar: bool = False,
        epoch: int = 0,
        openclip_model: torch.nn.Module = None,
        descriptor: str = "test",
) -> EvaluationMetrics:
    model.eval()

    pred_labels = []
    true_labels = []

    p_bar = tqdm(total=total) if set_pbar else None

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in enumerate(data_loader):

            if batch_preprocess:
                batch_images = batch_preprocess(batch[0])
                batch = batch_images, batch[1]

            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Output
            outputs = model(inputs)
            # loss = loss_fun(outputs, targets)

            pred_labels.append(torch.argmax(outputs, dim=-1))
            true_labels.append(targets)

            # Compute metrics
            acc = accuracy_score(torch.argmax(outputs, dim=-1).cpu(), targets.cpu())

            if set_pbar:
                p_bar.set_description(
                    f"{descriptor}. "
                    f"{epoch}. "
                    f"Iter: {i + 1}/{total} "
                    f"acc: {acc}."
                )
                p_bar.update()

    if set_pbar:
        p_bar.close()

    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)

    eval_tuple = EvaluationMetrics(true_labels.cpu(), pred_labels.cpu())

    logger.info(eval_tuple.log_test())

    write_metrics(writer, epoch, eval_tuple)

    return eval_tuple
