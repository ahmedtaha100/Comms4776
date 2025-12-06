import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_samples(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "super_class", "sub_class"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV at {csv_path} must contain columns: {required}")
        for row in reader:
            rows.append({"image": row["image"], "super_class": row["super_class"], "sub_class": row["sub_class"]})
    return rows


def load_mapping(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    mapping_path = Path(path)
    if not mapping_path.exists():
        return None
    rows = []
    with mapping_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["index"]), row["class"]))
    rows.sort(key=lambda x: x[0])
    return [name for _, name in rows]


def build_label_maps(train_samples: List[Dict[str, str]], val_samples: List[Dict[str, str]], super_map_path: Optional[str], sub_map_path: Optional[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    super_names = load_mapping(super_map_path) or sorted({s["super_class"] for s in train_samples + val_samples})
    sub_names = load_mapping(sub_map_path) or sorted({s["sub_class"] for s in train_samples + val_samples})
    super_to_idx = {name: i for i, name in enumerate(super_names)}
    sub_to_idx = {name: i for i, name in enumerate(sub_names)}
    return super_to_idx, sub_to_idx


class HierarchicalImageDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], root: str, transform, super_to_idx: Dict[str, int], sub_to_idx: Dict[str, int]):
        self.samples = samples
        self.root = Path(root)
        self.transform = transform
        self.super_to_idx = super_to_idx
        self.sub_to_idx = sub_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = self.root / sample["image"]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        y_super = self.super_to_idx[sample["super_class"]]
        y_sub = self.sub_to_idx[sample["sub_class"]]
        return {"image": image, "super_class": torch.tensor(y_super, dtype=torch.long), "sub_class": torch.tensor(y_sub, dtype=torch.long)}


def build_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, val_transform


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    train_samples = load_samples(cfg.data.train_csv)
    val_samples = load_samples(cfg.data.val_csv)
    super_to_idx, sub_to_idx = build_label_maps(train_samples, val_samples, cfg.data.superclass_mapping, cfg.data.subclass_mapping)
    train_tf, val_tf = build_transforms(cfg.data.image_size)
    train_dataset = HierarchicalImageDataset(train_samples, cfg.data.root, train_tf, super_to_idx, sub_to_idx)
    val_dataset = HierarchicalImageDataset(val_samples, cfg.data.root, val_tf, super_to_idx, sub_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)
    return train_loader, val_loader, super_to_idx, sub_to_idx
