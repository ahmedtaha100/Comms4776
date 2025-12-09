# Hierarchical Transfer Learning Project

Simple PyTorch/TIMM pipeline for hierarchical image classification (super: bird/dog/reptile; sub: 88 labels). We trained CNN and vision-language models, ran freeze vs. finetune, augmentation sweeps, and picked a final model.

## Quickstart
1) Python 3.10+ virtualenv, then `pip install -r requirements.txt`.
2) Data under `data/` (CSV columns: `image,super_class,sub_class`); default uses the released dataset.
3) Check env: `python scripts/check_env.py`.
4) Train any config: `python src/train.py --config <config>`. Runs on MPS (Apple GPU) or CUDA if available.

## Trained Models (outputs/)
- Final pick: ResNet-50 baseline — `configs/resnet50_baseline.yaml`, `outputs/resnet50_baseline/final.pth`.
- Alternates:
  - ResNet: frozen (`resnet50_frozen`), aug/mixup (`resnet50_aug`).
  - CLIP ViT-B/32: frozen (`clip_vitb32`), finetune (`clip_vitb32_finetune`), aug (`clip_vitb32_aug`).
  - SigLIP ViT-B/16: frozen (`siglip_vitb16`), finetune (`siglip_vitb16_finetune`), aug (`siglip_vitb16_aug`).

## Metrics (val, sub_acc/joint_acc)
- ResNet-50 baseline: 0.9785 / 0.9785 (sub_f1 0.9673) — best.
- ResNet-50 aug: 0.9785 / 0.9785 (sub_f1 0.9626).
- CLIP ViT-B/32 finetune: 0.9719 / 0.9719 (sub_f1 0.9566).
- SigLIP ViT-B/16 finetune: 0.9769 / 0.9769 (sub_f1 0.9591).
- CLIP frozen: 0.9554; SigLIP frozen: 0.6865; other variants in `scripts/eval_all.py`.

## Evaluation / Analysis
- Single eval: `PYTHONPATH=src python scripts/eval_checkpoint.py --config <config> --checkpoint <ckpt> [--num-workers 0]`
- Robustness (optionally with novel subclasses): `PYTHONPATH=src python scripts/robustness_eval.py --config <config> --checkpoint <ckpt> [--novel-subclass-list file] --num-workers 0`
- Error CSV (misclassifications): `PYTHONPATH=src python scripts/error_analysis.py --config <config> --checkpoint <ckpt> --output outputs/errors.csv --num-workers 0`
- Compare all runs: `PYTHONPATH=src python scripts/eval_all.py`

## Report
- NeurIPS-style LaTeX: `report.tex` + `references.bib` (compile to PDF). Final model and alternates listed in `outputs/README.md`.

## Submissions
- Evaluate a checkpoint on val: `PYTHONPATH=src python scripts/eval_checkpoint.py --config <config> --checkpoint <ckpt> --num-workers 0`.
- Make a leaderboard CSV from test images: `PYTHONPATH=src python scripts/make_submission.py --config <config> --checkpoint <ckpt> --test-dir data/Released_Data_NNDL_2025/test_images/test_images --val-dir data/val.csv --output submission.csv` (a `submission.csv` from the final model is already included).
- Metrics logged: super/sub accuracy, joint accuracy (both correct), and macro F1 for each level.

## Layout

- `configs/` YAML configs for runs.
- `scripts/` helper scripts (env check, data prep).
- `src/` dataloaders, models, training loop, utils.
- `outputs/` checkpoints and logs (created on run).
