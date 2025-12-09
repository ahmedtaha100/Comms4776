import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR
from tqdm import tqdm
from config import load_config
from data import build_dataloaders
from logging_utils import setup_logging
from models import HierarchicalClassifier
from utils import set_seed, get_device, save_checkpoint, mixup_data
from metrics import macro_f1


def create_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    else:
        main_sched = LambdaLR(optimizer, lambda epoch: 1.0)
    if train_cfg.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=train_cfg.warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, main_sched], milestones=[train_cfg.warmup_epochs])
    else:
        scheduler = main_sched
    return scheduler


def mixup_loss(logits, targets, criterion, lam):
    if isinstance(targets, tuple):
        y_a, y_b = targets
        return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    return criterion(logits, targets)


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, logger):
    model.train()
    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    correct_super = 0
    correct_sub = 0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device)
        y_super = batch["super_class"].to(device)
        y_sub = batch["sub_class"].to(device)
        if cfg.train.mixup_alpha > 0:
            images, y_super_mix, y_sub_mix, lam = mixup_data(images, y_super, y_sub, cfg.train.mixup_alpha)
        else:
            y_super_mix, y_sub_mix, lam = y_super, y_sub, 1.0
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=device.type == "cuda"):
            super_logits, sub_logits = model(images)
            loss_super = mixup_loss(super_logits, y_super_mix, ce, lam)
            loss_sub = mixup_loss(sub_logits, y_sub_mix, ce, lam)
            loss = (loss_super + loss_sub) * 0.5
        scaler.scale(loss).backward()
        if cfg.train.clip_grad_norm and cfg.train.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds_super = super_logits.argmax(dim=1)
        preds_sub = sub_logits.argmax(dim=1)
        correct_super += (preds_super == y_super).sum().item()
        correct_sub += (preds_sub == y_sub).sum().item()
        if (step + 1) % cfg.logging.log_interval == 0:
            logger.info(f"step {step + 1} loss {running_loss / total:.4f} super_acc {correct_super / total:.4f} sub_acc {correct_sub / total:.4f}")
    return running_loss / total, correct_super / total, correct_sub / total


def evaluate(model, loader, device, num_super, num_sub):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    correct_super = 0
    correct_sub = 0
    joint_correct = 0
    super_preds_all = []
    super_targets_all = []
    sub_preds_all = []
    sub_targets_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images = batch["image"].to(device)
            y_super = batch["super_class"].to(device)
            y_sub = batch["sub_class"].to(device)
            super_logits, sub_logits = model(images)
            loss = (ce(super_logits, y_super) + ce(sub_logits, y_sub)) * 0.5
            loss_sum += loss.item() * images.size(0)
            total += images.size(0)
            preds_super = super_logits.argmax(dim=1)
            preds_sub = sub_logits.argmax(dim=1)
            correct_super += (preds_super == y_super).sum().item()
            correct_sub += (preds_sub == y_sub).sum().item()
            joint_correct += ((preds_super == y_super) & (preds_sub == y_sub)).sum().item()
            super_preds_all.extend(preds_super.cpu().tolist())
            super_targets_all.extend(y_super.cpu().tolist())
            sub_preds_all.extend(preds_sub.cpu().tolist())
            sub_targets_all.extend(y_sub.cpu().tolist())
    avg_loss = loss_sum / total
    super_acc = correct_super / total
    sub_acc = correct_sub / total
    joint_acc = joint_correct / total
    super_f1 = macro_f1(super_preds_all, super_targets_all, num_classes=num_super)
    sub_f1 = macro_f1(sub_preds_all, sub_targets_all, num_classes=num_sub)
    return {
        "loss": avg_loss,
        "super_acc": super_acc,
        "sub_acc": sub_acc,
        "joint_acc": joint_acc,
        "super_f1": super_f1,
        "sub_f1": sub_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    device = get_device()
    logger = setup_logging(cfg.experiment.output_dir, cfg.experiment.exp_name)
    logger.info(f"using device {device}")
    train_loader, val_loader, super_to_idx, sub_to_idx = build_dataloaders(cfg)
    num_super = len(super_to_idx)
    num_sub = len(sub_to_idx)
    logger.info(f"super classes: {num_super} sub classes: {num_sub}")
    if cfg.model.super_classes != num_super or cfg.model.sub_classes != num_sub:
        logger.info("overriding model class counts with dataset-derived values")
    model = HierarchicalClassifier(cfg.model.name, cfg.model.pretrained, cfg.model.freeze_encoder, cfg.model.dropout, num_super, num_sub)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = create_scheduler(optimizer, cfg.train)
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    best_super = 0.0
    best_sub = 0.0

    for epoch in range(1, cfg.train.epochs + 1):
        logger.info(f"epoch {epoch}/{cfg.train.epochs}")
        train_loss, train_super_acc, train_sub_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, logger)
        scheduler.step()
        if epoch % cfg.logging.val_interval == 0:
            val_metrics = evaluate(model, val_loader, device, num_super, num_sub)
            logger.info(
                "val loss %.4f super_acc %.4f sub_acc %.4f joint_acc %.4f super_f1 %.4f sub_f1 %.4f",
                val_metrics["loss"],
                val_metrics["super_acc"],
                val_metrics["sub_acc"],
                val_metrics["joint_acc"],
                val_metrics["super_f1"],
                val_metrics["sub_f1"],
            )
            if val_metrics["super_acc"] > best_super:
                best_super = val_metrics["super_acc"]
            if val_metrics["sub_acc"] > best_sub:
                best_sub = val_metrics["sub_acc"]
        if epoch % cfg.logging.checkpoint_interval == 0:
            ckpt_path = Path(cfg.experiment.output_dir) / f"epoch_{epoch}.pth"
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "super_to_idx": super_to_idx,
                "sub_to_idx": sub_to_idx,
                "best_super": best_super,
                "best_sub": best_sub,
            }, ckpt_path)
        logger.info(f"train loss {train_loss:.4f} super_acc {train_super_acc:.4f} sub_acc {train_sub_acc:.4f}")
    final_path = Path(cfg.experiment.output_dir) / "final.pth"
    save_checkpoint({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "super_to_idx": super_to_idx,
        "sub_to_idx": sub_to_idx,
        "best_super": best_super,
        "best_sub": best_sub,
    }, final_path)
    logger.info("training complete")


if __name__ == "__main__":
    main()
