import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def mixup_data(x, y_super, y_sub, alpha):
    if alpha <= 0:
        return x, y_super, y_sub, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_super_a, y_super_b = y_super, y_super[index]
    y_sub_a, y_sub_b = y_sub, y_sub[index]
    return mixed_x, (y_super_a, y_super_b), (y_sub_a, y_sub_b), lam


def load_image_paths_from_csv(csv_path: Path) -> list[Path]:
    import csv
    image_paths = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path(row["image"])
            if not p.is_absolute():
                p = csv_path.parent / p
            image_paths.append(p)
    return image_paths
