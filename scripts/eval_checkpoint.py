import argparse
from pathlib import Path
import torch
from config import load_config
from data import build_dataloaders
from models import HierarchicalClassifier
from utils import get_device
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    _, val_loader, super_to_idx, sub_to_idx = build_dataloaders(cfg)
    num_super = len(super_to_idx)
    num_sub = len(sub_to_idx)

    model = HierarchicalClassifier(cfg.model.name, cfg.model.pretrained, cfg.model.freeze_encoder, cfg.model.dropout, num_super, num_sub)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    metrics = evaluate(model, val_loader, device, num_super, num_sub)
    print(f"checkpoint: {args.checkpoint}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
