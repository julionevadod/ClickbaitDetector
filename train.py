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

logger = logging.getLogger(
    __name__
)
logging.basicConfig(
    filename='logs/training.log',
    encoding='utf-8',
    level=logging.DEBUG
)

def setup():
    dist.init_process_group(backend="gloo")

def cleanup():
    dist.destroy_process_group()

def get_latest_checkpoint(checkpoint_path):
    file_list = list(filter(lambda x: not x.startswith("."),os.listdir(checkpoint_path)))
    if not file_list:
        return None
    return f"{checkpoint_path}/{max(file_list)}"

def get_accuracy(prediction, ground_truth):
    predicted_labels = prediction.max(dim=1).indices
    true_labels = ground_truth.max(dim=1).indices
    n = true_labels.size(0)
    return (true_labels==predicted_labels).sum()/n

def save_checkpoint():
    pass

def get_or_create_experiment(name: str) -> str:
    client = mlflow.tracking.MlflowClient()
    searched_experiment = client.search_experiments(
        filter_string = f"attribute.name = '{name}'"
    )
    if searched_experiment:
        return searched_experiment[0].experiment_id
    else:
        return client.create_experiment(
            "clickbait-detector",
            tags=experiment_tags
        )



def train():

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    if global_rank==0:
        mlflow.log_param("BATCH_SIZE",config["batch_size"])
        mlflow.log_param("NUM_EPOCHS",config["num_epochs"])
        mlflow.log_param("INITIAL LEARNING_RATE",config["learning_rate"])

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logging.info(f"Device {device} available.")

    model = ClickBaitDetectorModel(
        os.environ["BASE_MODEL_PATH"],
        f"{device}:{local_rank}"
    )

    if device == "cuda":
        ddp_model = DDP(model.model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model.model)

    if os.environ["CHECKPOINT_PATH"]:
        latest_checkpoint = get_latest_checkpoint(
            os.environ["CHECKPOINT_PATH"]
        )
        if latest_checkpoint:
            logging.info(
                f"Loading {latest_checkpoint} checkpoint."
            )
            ddp_model.load_state_dict(
                torch.load(latest_checkpoint)
            )
        else:
            logging.warning(
                "Checkpoint path was specified but no checkpoints were found."
            )

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=float(os.environ["LEARNING_RATE"]))
    
    if os.environ["OPTIMIZER_PATH"]:
        latest_checkpoint = get_latest_checkpoint(
            os.environ["OPTIMIZER_PATH"]
        )
        if latest_checkpoint:
            logging.info(
                f"Loading {latest_checkpoint} optimizer."
            )
            optimizer.load_state_dict(
                torch.load(latest_checkpoint)
            )
        else:
            logging.warning(
                "Optimizer path was specified but no checkpoints were found."
            )

    loss_fn = torch.nn.BCELoss()

    tokenizer = AutoTokenizer.from_pretrained(
        "m-newhauser/distilbert-political-tweets"
    )
    training_data = ClickbaitDataset(
        os.environ["DATA_PATH"],
        tokenizer
    )

    train_dataset, val_dataset = torch.utils.data.random_split(training_data, [0.9, 0.1])

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

    for epoch in range(int(os.environ["NUM_EPOCHS"])):
        logger.info(f"Starting epoch {epoch} in device {global_rank}")
        running_loss = 0.0
        for i_batch, data in enumerate(train_dataloader):
            inputs = data["text"]
            labels = data["label"]

            inputs.to(f"{device}:{local_rank}")

            # By default PyTorch accumulates gradients.
            #  This accumulation happens when .backwards() is called.
            #  We reset them at the beginning of the batch
            #  to avoid their accumulation.
            #  We can set_to_none parameter to True
            #  which is more memory efficient.
            #  but it needs to be handled.
            optimizer.zero_grad()

            # Forward pass
            outputs = torch.nn.functional.softmax(
                ddp_model(**inputs)["logits"],
                dim=None
            )

            # Compute loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            if global_rank==0:
                if i_batch % int(os.environ["LOGGING_INTERVAL"]) == 0:
                    loss_id = f"EPOCH.BATCH-{i_batch} Loss"
                    accuracy_id = f"EPOCH.BATCH-{i_batch} Acc"
                    logging.info(
                        f"{loss_id}: {loss.item()}"
                    )
                    mlflow.log_metric(loss_id,loss.item(), step = (epoch+1)*i_batch)
                    mlflow.log_metric(accuracy_id,get_accuracy(outputs,labels), step = (epoch+1)*i_batch)


        if global_rank == 0:
            loss_id = f"EPOCH Loss"
            accuracy_id = f"EPOCH Acc"
            logger.info(f"{loss_id}: {str(running_loss/i_batch)}")
            mlflow.log_metric(loss_id,running_loss/i_batch, step = epoch)
            mlflow.log_metric(accuracy_id,get_accuracy(outputs,labels), step = epoch)
            if epoch % int(os.environ["CHECKPOINT_INTERVAL"]) == 0:
                logger.info(f"Saving for epoch {epoch}...")
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
                save_path = (
                    f"{os.environ['CHECKPOINT_PATH']}/"
                    f"checkpoint_{timestamp}_{epoch}"
                )
                torch.save(
                    ddp_model.state_dict(),
                    save_path
                )
                logger.info(
                    f"Checkpoint checkpoint_{timestamp}_{epoch} saved!"
                )

        with torch.no_grad():
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            for i, val_data in enumerate(val_dataloader):
                val_inputs = val_data["text"]
                val_labels = val_data["label"]
                val_outputs = torch.nn.functional.softmax(
                    ddp_model(**val_inputs)["logits"]
                )
                val_loss = loss_fn(val_outputs, val_labels)
                running_val_loss += val_loss.item()
                running_val_accuracy += get_accuracy(val_outputs, val_labels)

        avg_val_loss = running_val_loss / (i + 1)
        avg_val_accuracy = running_val_accuracy / (i + 1)

        #dist.reduce(avg_val_loss,0,op=dist.reduce_op.AVG)
        #dist.reduce(avg_val_accuracy,0,op=dist.reduce_op.AVG)
        if global_rank == 0:
            mlflow.log_metric("EPOCH VAL ACC",avg_val_accuracy, step=epoch)
            mlflow.log_metric("EPOCH VAL LOSS",avg_val_loss, step=epoch)
        dist.barrier()

    if global_rank == 0:
        logger.info(f"Saving for epoch {epoch}...")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        save_path = (
            f"{os.environ['CHECKPOINT_PATH']}/"
            f"checkpoint_{timestamp}_final"
        )
        save_path_optimizer = (
            f"{os.environ['OPTIMIZER_PATH']}/"
            f"checkpoint_{timestamp}_final"
        )
        torch.save(
            ddp_model.state_dict(),
            save_path
        )
        torch.save(
            optimizer.state_dict(),
            save_path_optimizer
        )
        logger.info(
            f"Checkpoint checkpoint_{timestamp}_final saved!"
        )

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)

    os.environ["BATCH_SIZE"] = str(config["batch_size"])
    os.environ["VAL_BATCH_SIZE"] = str(config["val_batch_size"])
    os.environ["DATA_PATH"] = config["data_path"]
    os.environ["BASE_MODEL_PATH"] = config["base_model_path"]
    os.environ["NUM_EPOCHS"] = str(config["num_epochs"])
    os.environ["CHECKPOINT_PATH"] = config["checkpoint_path"]
    os.environ["OPTIMIZER_PATH"] = config["optimizer_path"]
    os.environ["CHECKPOINT_INTERVAL"] = str(config["checkpoint_interval"])
    os.environ["LOGGING_INTERVAL"] = str(config["logging_interval"])
    os.environ["LEARNING_RATE"] = str(config["learning_rate"])

    experiment_tags = {
        "BATCH_SIZE": os.getenv("BATCH_SIZE"),
        "NUM_EPOCHS": os.getenv("NUM_EPOCHS"),
        "LEARNING_RATE": os.getenv("LEARNING_RATE")
    }

    if os.environ["RANK"] == "0":
        experiment_id = get_or_create_experiment(
            "clickbait-detector"
        )
        with mlflow.start_run(experiment_id=experiment_id):
            setup()
            train()
            cleanup()
    else:
        setup()
        train()
        cleanup()