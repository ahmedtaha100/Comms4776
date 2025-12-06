import argparse
from pathlib import Path
import torch
from config import load_config
from data import build_dataloaders
from models import HierarchicalClassifier
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--novel-subclass-list", type=str, default=None, help="Path to text file listing novel sub_class names (one per line)")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    _, val_loader, super_to_idx, sub_to_idx = build_dataloaders(cfg)
    num_super = len(super_to_idx)
    num_sub = len(sub_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = HierarchicalClassifier(cfg.model.name, cfg.model.pretrained, cfg.model.freeze_encoder, cfg.model.dropout, num_super, num_sub)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    novel_set = None
    if args.novel_subclass_list:
        lines = Path(args.novel_subclass_list).read_text().splitlines()
        novel_set = {line.strip() for line in lines if line.strip()}
        novel_indices = {sub_to_idx[name] for name in novel_set if name in sub_to_idx}

    metrics = evaluate(model, val_loader, device, num_super, num_sub)
    print("overall:", metrics)

    if novel_set:
        from torch.utils.data import Subset
        dataset = val_loader.dataset
        novel_indices_list = [i for i, sample in enumerate(dataset.samples) if sample["sub_class"] in novel_set]
        novel_subset = Subset(dataset, novel_indices_list)
        novel_loader = torch.utils.data.DataLoader(novel_subset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=False)
        novel_metrics = evaluate(model, novel_loader, device, num_super, num_sub)
        print("novel_subclasses:", novel_metrics)


if __name__ == "__main__":
    main()
