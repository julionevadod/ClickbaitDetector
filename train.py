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

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log/training.log",
    encoding='utf-8',
    level=logging.DEBUG
)


def get_latest_checkpoint(checkpoint_path):
    file_list = os.listdir(checkpoint_path)
    if not file_list:
        return None
    return f"{checkpoint_path}/{max(file_list)}"


def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logging.info(f"Device {device} available.")

    model = ClickBaitDetectorModel(
        os.environ["BASE_MODEL_PATH"],
        f"{device}:{rank}"
    )

    if device == "cuda":
        ddp_model = DDP(model.model, device_ids=[rank])
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

    optimizer = torch.optim.Adam(ddp_model.parameters())
    loss_fn = torch.nn.BCELoss()

    tokenizer = AutoTokenizer.from_pretrained(
        "m-newhauser/distilbert-political-tweets"
    )
    training_data = ClickbaitDataset(
        os.environ["DATA_PATH"],
        tokenizer
    )

    sampler = DistributedSampler(
        training_data,
        rank=rank,
        shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        training_data,
        batch_size=int(os.environ["BATCH_SIZE"]),
        sampler=sampler
    )

    for epoch in range(int(os.environ["NUM_EPOCHS"])):
        logger.info(f"Starting epoch {epoch}")
        for _, data in enumerate(train_dataloader):
            inputs = data["text"]
            labels = data["label"]

            inputs.to(f"{device}:{rank}")

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
                ddp_model(**inputs)["logits"]
            )

            # Compute loss
            loss = loss_fn(outputs, labels)

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

    if rank == 0:
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


if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)

    os.environ["BATCH_SIZE"] = str(config["batch_size"])
    os.environ["DATA_PATH"] = config["data_path"]
    os.environ["BASE_MODEL_PATH"] = config["base_model_path"]
    os.environ["NUM_EPOCHS"] = str(config["num_epochs"])
    os.environ["CHECKPOINT_PATH"] = config["checkpoint_path"]
    os.environ["CHECKPOINT_INTERVAL"] = str(config["checkpoint_interval"])

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    world_size = config["world_size"]
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
