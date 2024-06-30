import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import torch
from datetime import datetime
import logging
from torch.nn.modules.loss import _Loss
import mlflow
import torch.distributed as dist

logger = logging.getLogger(
    __name__
)
logging.basicConfig(
    filename='logs/training.log',
    encoding='utf-8',
    level=logging.DEBUG
)

def get_latest_checkpoint(checkpoint_path):
    file_list = list(filter(lambda x: not x.startswith("."),os.listdir(checkpoint_path)))
    if not file_list:
        return None
    return f"{checkpoint_path}/{max(file_list)}"

class Trainer():
    def __init__(
        self,
        model,
        device: str,
        epochs: int,
        loss_function: _Loss,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        save_path: str,
        save_interval: int
    ):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model
        self.device = device
        self.initial_epoch = 0
        self.epochs = epochs
        self.loss_function = loss_function
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.save_interval = save_interval
                    
    def _log(self,name,value,step):
        logging.info(
            f"{name}: {value}"
        )
        mlflow.log_metric(name, value, step=step)

    def _run_batch(self, input, target):
            self.optimizer.zero_grad()
            # Forward pass
            outputs = torch.nn.functional.softmax(
                self.model(**input)["logits"],
                dim=None
            )
            # Compute loss
            loss = self.loss_function(
                outputs,
                target
            )
            # Compute gradients
            loss.backward()
            # Update weights
            self.optimizer.step()
            return loss

    def _run_epoch(self):
        running_loss = 0.0
        n_batches = len(self.train_dataloader)
        for i_batch, data in enumerate(self.train_dataloader):
            inputs = data["text"]
            labels = data["label"]
            inputs.to(f"{self.device}:{self.local_rank}")
            batch_loss = self._run_batch(
                inputs,
                labels 
            )
            running_loss += batch_loss
        if self.scheduler:
            self.scheduler.step()
        epoch_loss = running_loss/n_batches
        return epoch_loss

    def _get_accuracy(self, prediction, ground_truth):
        predicted_labels = prediction.max(dim=1).indices
        true_labels = ground_truth.max(dim=1).indices
        n = true_labels.size(0)
        return (true_labels==predicted_labels).sum()/n

    def _validate(self, epoch):
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            for i, val_data in enumerate(self.val_dataloader):
                val_inputs = val_data["text"]
                val_labels = val_data["label"]
                val_outputs = torch.nn.functional.softmax(
                    self.model(**val_inputs)["logits"]
                )
                val_loss = self.loss_function(val_outputs, val_labels)
                running_val_loss += val_loss.item()
                running_val_accuracy += self._get_accuracy(val_outputs, val_labels)

        avg_val_loss = running_val_loss / (i + 1)
        avg_val_accuracy = running_val_accuracy / (i + 1)

        #dist.reduce(avg_val_loss,0,op=dist.reduce_op.AVG)
        #dist.reduce(avg_val_accuracy,0,op=dist.reduce_op.AVG)
        if self.global_rank == 0:
            self._log("EPOCH VAL LOSS", avg_val_loss, step = epoch)
            self._log("EPOCH VAL ACC", avg_val_accuracy, step = epoch)

    def train(self):
        for epoch in range(self.initial_epoch,self.epochs):
            epoch_loss = self._run_epoch()
            self._log(
                "EPOCH LOSS", epoch_loss, step=epoch
            )
            self._validate(
                epoch
            )
            dist.barrier()
            
            if epoch%self.save_interval == 0:
                self.save_snapshot(epoch)
    
    def load_snapshot(self):
        snapshot_path = get_latest_checkpoint(
            self.save_path
        )
        if snapshot_path:
            snapshot = torch.load(snapshot_path)
            self.model.load_state_dict(
                snapshot["model"]
            )
            self.optimizer.load_state_dict(
                snapshot["optimizer"]
            )
            self.scheduler.load_state_dict(
                snapshot["scheduler"]
            )
            logger.info(
                f"Loaded snapshot {snapshot_path}"
            )
        else:
            logger.info(
                "No SNAPSHOT loaded. Training from scratch."
            )

    def save_snapshot(self, step):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        snapshot_name = f"checkpoint_{timestamp}"
        torch.save(
            {
                "step": step,
                "model":self.model.state_dict(),
                "optimizer":self.optimizer.state_dict(),
                "scheduler":self.scheduler.state_dict()
            },
            f"{self.save_path}/{snapshot_name}"
        )