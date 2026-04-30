import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MOSIDataset(Dataset):
    def __init__(self, text, vision, audio, labels):
        self.text = torch.as_tensor(text, dtype=torch.float32)
        self.vision = torch.as_tensor(vision, dtype=torch.float32)
        self.audio = torch.as_tensor(audio, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32).view(-1)

        self.text = torch.nan_to_num(self.text, nan=0.0, posinf=0.0, neginf=0.0)
        self.vision = torch.nan_to_num(self.vision, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = torch.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)

        self.text_mask = self._build_mask(self.text)
        self.vision_mask = self._build_mask(self.vision)
        self.audio_mask = self._build_mask(self.audio)

    @staticmethod
    def _build_mask(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [N, T, D], got {tuple(x.shape)}")
        mask = x.abs().sum(dim=-1) > eps
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask = mask.clone()
            mask[empty] = True
        return mask

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "vision": self.vision[idx],
            "audio": self.audio[idx],
            "label": self.labels[idx],
            "text_mask": self.text_mask[idx],
            "vision_mask": self.vision_mask[idx],
            "audio_mask": self.audio_mask[idx],
        }



def _read_pickle(data_path: Path):
    with open(data_path, "rb") as f:
        try:
            return pickle.load(f)
        except TypeError:
            return pickle.load(f, encoding="latin1")



def _compute_stats(x: torch.Tensor, mask: torch.Tensor):
    flat = x[mask]
    if flat.numel() == 0:
        feat_dim = x.size(-1)
        return torch.zeros(feat_dim, dtype=x.dtype), torch.ones(feat_dim, dtype=x.dtype)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-4)
    return mean, std



def _normalize(x: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, clip: float):
    out = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    out = out.clamp(min=-clip, max=clip)
    return out * mask.unsqueeze(-1).to(out.dtype)



def load_mosi_from_pkl(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"MOSI data file not found: {data_path}")

    data = _read_pickle(data_path)
    required_top = {"train", "valid", "test"}
    if not required_top.issubset(set(data.keys())):
        raise KeyError(f"Expected top-level keys {sorted(required_top)}, got {list(data.keys())}")

    required = {"text", "vision", "audio", "regression_labels"}
    splits = {}
    for split_name in ["train", "valid", "test"]:
        split = data[split_name]
        missing = required - set(split.keys())
        if missing:
            raise KeyError(f"Split {split_name} missing keys: {sorted(missing)}")
        splits[split_name] = {
            "text": torch.as_tensor(split["text"], dtype=torch.float32),
            "vision": torch.as_tensor(split["vision"], dtype=torch.float32),
            "audio": torch.as_tensor(split["audio"], dtype=torch.float32),
            "labels": split["regression_labels"],
        }
        for key in ["text", "vision", "audio"]:
            splits[split_name][key] = torch.nan_to_num(splits[split_name][key], nan=0.0, posinf=0.0, neginf=0.0)

    train_text, train_vision, train_audio = splits["train"]["text"], splits["train"]["vision"], splits["train"]["audio"]
    t_mask = MOSIDataset._build_mask(train_text)
    v_mask = MOSIDataset._build_mask(train_vision)
    a_mask = MOSIDataset._build_mask(train_audio)

    t_mean, t_std = _compute_stats(train_text, t_mask)
    v_mean, v_std = _compute_stats(train_vision, v_mask)
    a_mean, a_std = _compute_stats(train_audio, a_mask)

    for split_name, clip_map in [("train", (4.0, 3.5, 3.5)), ("valid", (4.0, 3.5, 3.5)), ("test", (4.0, 3.5, 3.5))]:
        s = splits[split_name]
        s_t_mask = MOSIDataset._build_mask(s["text"])
        s_v_mask = MOSIDataset._build_mask(s["vision"])
        s_a_mask = MOSIDataset._build_mask(s["audio"])
        s["text"] = _normalize(s["text"], s_t_mask, t_mean, t_std, clip_map[0])
        s["vision"] = _normalize(s["vision"], s_v_mask, v_mean, v_std, clip_map[1])
        s["audio"] = _normalize(s["audio"], s_a_mask, a_mean, a_std, clip_map[2])

    train_dataset = MOSIDataset(
        splits["train"]["text"], splits["train"]["vision"], splits["train"]["audio"], splits["train"]["labels"]
    )
    valid_dataset = MOSIDataset(
        splits["valid"]["text"], splits["valid"]["vision"], splits["valid"]["audio"], splits["valid"]["labels"]
    )
    test_dataset = MOSIDataset(
        splits["test"]["text"], splits["test"]["vision"], splits["test"]["audio"], splits["test"]["labels"]
    )

    meta = {
        "dataset": "MOSI",
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "test_size": len(test_dataset),
        "text_dim": int(train_dataset.text.shape[-1]),
        "vision_dim": int(train_dataset.vision.shape[-1]),
        "audio_dim": int(train_dataset.audio.shape[-1]),
        "seq_len": int(train_dataset.text.shape[1]),
    }
    return train_dataset, valid_dataset, test_dataset, meta


# backward-compatible alias
load_mosei_from_pkl = load_mosi_from_pkl
