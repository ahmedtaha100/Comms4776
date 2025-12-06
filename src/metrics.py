from typing import Iterable, List, Optional
import numpy as np
from sklearn.metrics import f1_score


def accuracy(preds: Iterable[int], targets: Iterable[int]) -> float:
    preds_arr = np.asarray(list(preds))
    targets_arr = np.asarray(list(targets))
    if len(preds_arr) == 0:
        return 0.0
    return float((preds_arr == targets_arr).mean())


def macro_f1(preds: Iterable[int], targets: Iterable[int], num_classes: Optional[int] = None) -> float:
    preds_arr = np.asarray(list(preds))
    targets_arr = np.asarray(list(targets))
    if num_classes is not None:
        labels = list(range(num_classes))
    else:
        labels = None
    return float(f1_score(targets_arr, preds_arr, labels=labels, average="macro", zero_division=0))
