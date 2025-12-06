import json
import sys
from pathlib import Path
import torch
from config import load_config
from data import build_dataloaders
from models import HierarchicalClassifier
from train import evaluate

root = Path(__file__).resolve().parent.parent
outputs_dir = root / "outputs"
configs_dir = root / "configs"
results = []
for final_path in outputs_dir.glob("*/final.pth"):
    cfg_name = final_path.parent.name + ".yaml"
    cfg_path = configs_dir / cfg_name
    if not cfg_path.exists():
        continue
    cfg = load_config(str(cfg_path))
    cfg.data.num_workers = 0
    _, val_loader, super_to_idx, sub_to_idx = build_dataloaders(cfg)
    num_super = len(super_to_idx)
    num_sub = len(sub_to_idx)
    model = HierarchicalClassifier(cfg.model.name, cfg.model.pretrained, cfg.model.freeze_encoder, cfg.model.dropout, num_super, num_sub)
    ckpt = torch.load(final_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate(model, val_loader, torch.device("cpu"), num_super, num_sub)
    results.append({"run": final_path.parent.name, **metrics})
print(json.dumps(results, indent=2))
