import torch
from torch.utils.data import Dataset
import pandas as pd


class ClickbaitDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.clickbait_dataframe = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.clickbait_dataframe.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        post_text = self.clickbait_dataframe.loc[idx, "postText"]

        post_text_tokenized = self.tokenizer(
            post_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=200,
            padding="max_length"
        )

        post_text_tokenized["input_ids"] = torch.squeeze(
            post_text_tokenized["input_ids"]
        )
        post_text_tokenized["attention_mask"] = torch.squeeze(
            post_text_tokenized["attention_mask"]
        )

        post_label = self.clickbait_dataframe.loc[idx, "truthClass"]
        post_label_tensor = torch.nn.functional.one_hot(
            torch.tensor(int(post_label)),
            num_classes=2
        ).float()

        sample = {"text": post_text_tokenized, "label": post_label_tensor}

        return sample
