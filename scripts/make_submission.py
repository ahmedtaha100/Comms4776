import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from config import load_config
from data import build_transforms, load_mapping
from models import HierarchicalClassifier
from utils import get_device


class TestImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "name": path.name}


def compute_novel_thresholds(model, val_loader, device, super_percentile=5.0, sub_percentile=5.0):
    model.eval()
    super_confs = []
    sub_confs = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            super_logits, sub_logits = model(images)
            super_probs = torch.softmax(super_logits, dim=1)
            sub_probs = torch.softmax(sub_logits, dim=1)
            super_conf, _ = torch.max(super_probs, dim=1)
            sub_conf, _ = torch.max(sub_probs, dim=1)
            super_confs.extend(super_conf.cpu().tolist())
            sub_confs.extend(sub_conf.cpu().tolist())

    super_threshold = float(np.percentile(super_confs, super_percentile))
    sub_threshold = float(np.percentile(sub_confs, sub_percentile))
    return super_threshold, sub_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/Released_Data_NNDL_2025/test_images/test_images")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    super_classes = load_mapping(cfg.data.superclass_mapping)
    sub_classes = load_mapping(cfg.data.subclass_mapping)
    num_super = len(super_classes)
    num_sub = len(sub_classes)

    novel_super_label = num_super - 1
    novel_sub_label = num_sub - 1

    model = HierarchicalClassifier(cfg.model.name, cfg.model.pretrained, cfg.model.freeze_encoder, cfg.model.dropout, num_super, num_sub)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    threshold_super = ckpt.get("threshold_super")
    threshold_sub = ckpt.get("threshold_sub")
    model.eval()

    _, val_tf = build_transforms(cfg.data.image_size)
    image_paths = sorted(Path(args.test_dir).glob("*.jpg"))
    dataset = TestImageDataset(image_paths, val_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    records = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            names = batch["name"]
            super_logits, sub_logits = model(images)

            super_probs = torch.softmax(super_logits, dim=1)
            sub_probs = torch.softmax(sub_logits, dim=1)

            super_conf, super_pred = torch.max(super_probs, dim=1)
            sub_conf, sub_pred = torch.max(sub_probs, dim=1)

            is_novel_super = super_conf < threshold_super
            is_novel_sub = sub_conf < threshold_sub
            is_novel_sub = is_novel_sub | is_novel_super

            super_preds_with_novel = super_pred.clone()
            sub_preds_with_novel = sub_pred.clone()

            super_preds_with_novel[is_novel_super] = novel_super_label
            sub_preds_with_novel[is_novel_sub] = novel_sub_label

            super_preds = super_preds_with_novel.cpu().tolist()
            sub_preds = sub_preds_with_novel.cpu().tolist()

            for name, s, sub in zip(names, super_preds, sub_preds):
                records.append({"image": name, "superclass_index": s, "subclass_index": sub})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "superclass_index", "subclass_index"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote predictions to {out_path} from {len(records)} images")


if __name__ == "__main__":
    main()
