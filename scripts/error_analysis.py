import argparse
import csv
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
    parser.add_argument("--output", type=str, default="outputs/error_analysis.csv")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    _, val_loader, super_to_idx, sub_to_idx = build_dataloaders(cfg)
    num_super = len(super_to_idx)
    num_sub = len(sub_to_idx)
    idx_to_super = {v: k for k, v in super_to_idx.items()}
    idx_to_sub = {v: k for k, v in sub_to_idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = HierarchicalClassifier(cfg.model.name, cfg.model.pretrained, cfg.model.freeze_encoder, cfg.model.dropout, num_super, num_sub)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    metrics = evaluate(model, val_loader, device, num_super, num_sub)
    print("metrics:", metrics)

    records = []
    base_idx = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            y_super = batch["super_class"].to(device)
            y_sub = batch["sub_class"].to(device)
            super_logits, sub_logits = model(images)
            super_preds = super_logits.argmax(dim=1)
            sub_preds = sub_logits.argmax(dim=1)
            for i in range(images.size(0)):
                sample_idx = base_idx + i
                super_ok = int(super_preds[i].item() == y_super[i].item())
                sub_ok = int(sub_preds[i].item() == y_sub[i].item())
                if not (super_ok and sub_ok):
                    rec = {
                        "image": val_loader.dataset.samples[sample_idx]["image"],
                        "true_super": idx_to_super[y_super[i].item()],
                        "pred_super": idx_to_super[super_preds[i].item()],
                        "true_sub": idx_to_sub[y_sub[i].item()],
                        "pred_sub": idx_to_sub[sub_preds[i].item()],
                        "super_correct": super_ok,
                        "sub_correct": sub_ok,
                    }
                    records.append(rec)
            base_idx += images.size(0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true_super", "pred_super", "true_sub", "pred_sub", "super_correct", "sub_correct"])
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} misclassified rows to {out_path}")


if __name__ == "__main__":
    main()
