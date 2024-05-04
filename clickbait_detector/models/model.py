from transformers import AutoModelForSequenceClassification
import torch

BATCHES_TO_REPORT = 1000


class ClickBaitDetectorModel():

    def __init__(self, model_path: str, device: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(
            device
        )

        self.device = device
    
    def load_pretrained(self, trained_path):
        self.model.load_state_dict(
            torch.load(trained_path)
        )

    def parameters(self):
        return self.model.parameters()
