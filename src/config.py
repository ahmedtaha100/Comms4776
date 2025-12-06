from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ExperimentConfig:
    seed: int
    output_dir: str
    exp_name: str

@dataclass
class DataConfig:
    root: str
    train_csv: str
    val_csv: str
    superclass_mapping: str
    subclass_mapping: str
    num_workers: int
    batch_size: int
    image_size: int

@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    freeze_encoder: bool
    dropout: float
    super_classes: int
    sub_classes: int

@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    lr_scheduler: str
    clip_grad_norm: float
    mixup_alpha: float

@dataclass
class LoggingConfig:
    log_interval: int
    val_interval: int
    checkpoint_interval: int

@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    logging: LoggingConfig


def load_config(path: str) -> Config:
    cfg_dict = yaml.safe_load(Path(path).read_text())
    exp = ExperimentConfig(**cfg_dict["experiment"])
    data = DataConfig(**cfg_dict["data"])
    model = ModelConfig(**cfg_dict["model"])
    train = TrainConfig(**cfg_dict["train"])
    log_cfg = LoggingConfig(**cfg_dict["logging"])
    output_dir = Path(exp.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return Config(exp, data, model, train, log_cfg)
