import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from clickbait_detector.datasets.ClickbaitDataset import ClickbaitDataset
from clickbait_detector.models.model import ClickBaitDetectorModel
import torch.multiprocessing as mp
import json
import os
from transformers import AutoTokenizer
from torch.utils.data import DistributedSampler
from datetime import datetime
import logging
import sys
import mlflow
from Trainer import Trainer
import argparse

logger = logging.getLogger(
    __name__
)
logging.basicConfig(
    filename='logs/training.log',
    encoding='utf-8',
    level=logging.DEBUG
)


def setup():
    log_experiment_params()
    dist.init_process_group(backend="gloo")


def cleanup():
    dist.destroy_process_group()


def get_or_create_experiment(name: str) -> str:
    client = mlflow.tracking.MlflowClient()
    searched_experiment = client.search_experiments(
        filter_string=f"attribute.name = '{name}'"
    )
    if searched_experiment:
        return searched_experiment[0].experiment_id
    else:
        return client.create_experiment(
            "clickbait-detector",
            tags=experiment_tags
        )


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def log_experiment_params():
    mlflow.log_param("BATCH_SIZE", int(os.getenv("BATCH_SIZE")))
    mlflow.log_param("NUM_EPOCHS", int(os.getenv("NUM_EPOCHS")))
    mlflow.log_param("INITIAL LEARNING_RATE",
                     float(os.getenv("LEARNING_RATE")))


def train():

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    device = get_device()
    logging.info(f"Device {device} available.")

    model = ClickBaitDetectorModel(
        os.environ["BASE_MODEL_PATH"],
        f"{device}:{local_rank}"
    )

    if device == "cuda":
        ddp_model = DDP(model.model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model.model)

    optimizer = torch.optim.Adam(
        ddp_model.parameters(),
        lr=float(os.environ["LEARNING_RATE"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    loss_fn = torch.nn.BCELoss()

    if global_rank == 0:
        mlflow.set_tag("Scheduler", "StepLR")
        mlflow.set_tag("Step Size", 5)
        mlflow.set_tag("Gamma", 0.1)

    tokenizer = AutoTokenizer.from_pretrained(
        "m-newhauser/distilbert-political-tweets"
    )

    training_data = ClickbaitDataset(
        os.environ["DATA_PATH"],
        tokenizer,
        int(os.environ["DATA_LIMIT"])
    )

    train_dataset, val_dataset = torch.utils.data.random_split(training_data, [
                                                               0.9, 0.1])

    sampler = DistributedSampler(
        train_dataset,
        rank=global_rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=global_rank,
        shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(os.environ["BATCH_SIZE"]),
        sampler=sampler
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(os.environ["VAL_BATCH_SIZE"]),
        sampler=val_sampler
    )

    trainer = Trainer(
        ddp_model,
        device,
        int(os.getenv("NUM_EPOCHS")),
        loss_fn,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        os.getenv("CHECKPOINT_PATH"),
        int(os.getenv("CHECKPOINT_INTERVAL"))
    )

    trainer.load_snapshot()

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Argument parser for training script.')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size used for training.')
    parser.add_argument('--batch_size_val', type=int, default=32,
                        help='Batch size used for training validation.')
    parser.add_argument('--data_path', type=str, default="data/cleansed_data",
                        help='Path where training data is stored.')
    parser.add_argument('--data_limit', type=int, default=-1,
                        help='Path where training data is stored.')
    parser.add_argument('--base_model_path', type=str, default="m-newhauser/distilbert-political-tweets",
                        help='Location of the base model.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to run training.')
    parser.add_argument('--checkpoint_path', type=str, default="data/checkpoints/",
                        help='Path where checkpoint data will be stored.')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Number of epochs between checkpointing.')
    parser.add_argument('--logging_interval', type=int, default=1,
                        help='Number of epochs between logging.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate for optimizer.')

    args = parser.parse_args()

    os.environ["BATCH_SIZE"] = str(args.batch_size)
    os.environ["VAL_BATCH_SIZE"] = str(args.batch_size_val)
    os.environ["DATA_PATH"] = args.data_path
    os.environ["DATA_LIMIT"] = str(args.data_limit)
    os.environ["BASE_MODEL_PATH"] = args.base_model_path
    os.environ["NUM_EPOCHS"] = str(args.num_epochs)
    os.environ["CHECKPOINT_PATH"] = str(args.checkpoint_path)
    os.environ["CHECKPOINT_INTERVAL"] = str(args.checkpoint_interval)
    os.environ["LOGGING_INTERVAL"] = str(args.logging_interval)
    os.environ["LEARNING_RATE"] = str(args.learning_rate)

    experiment_tags = {
        "BATCH_SIZE": os.getenv("BATCH_SIZE"),
        "NUM_EPOCHS": os.getenv("NUM_EPOCHS"),
        "LEARNING_RATE": os.getenv("LEARNING_RATE")
    }

    if os.environ["RANK"] == "0":
        experiment_id = get_or_create_experiment(
            "clickbait-detector"
        )

        mlflow.autolog()

        with mlflow.start_run(experiment_id=experiment_id):
            setup()
            train()
            cleanup()
    else:
        setup()
        train()
        cleanup()
