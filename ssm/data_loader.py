import pickle
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


class MOSEIDataset(Dataset):
    def __init__(self, text, vision, audio, labels):
        self.text = torch.tensor(text, dtype=torch.float32)
        self.vision = torch.tensor(vision, dtype=torch.float32)
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "vision": self.vision[idx],
            "audio": self.audio[idx],
            "label": self.labels[idx],
        }


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except TypeError:
            return pickle.load(f, encoding="latin1")


def infer_dims(split_dict: Dict) -> Tuple[int, int, int]:
    text_dim = int(split_dict["text"].shape[-1])
    vision_dim = int(split_dict["vision"].shape[-1])
    audio_dim = int(split_dict["audio"].shape[-1])
    return text_dim, vision_dim, audio_dim


def load_mosei_from_pkl(data_path: str):
    data = _safe_pickle_load(data_path)

    train_dataset = MOSEIDataset(
        data["train"]["text"],
        data["train"]["vision"],
        data["train"]["audio"],
        data["train"]["regression_labels"],
    )
    valid_dataset = MOSEIDataset(
        data["valid"]["text"],
        data["valid"]["vision"],
        data["valid"]["audio"],
        data["valid"]["regression_labels"],
    )
    test_dataset = MOSEIDataset(
        data["test"]["text"],
        data["test"]["vision"],
        data["test"]["audio"],
        data["test"]["regression_labels"],
    )

    dims = infer_dims(data["train"])
    meta = {
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "test_size": len(test_dataset),
        "text_dim": dims[0],
        "vision_dim": dims[1],
        "audio_dim": dims[2],
    }
    return train_dataset, valid_dataset, test_dataset, meta
