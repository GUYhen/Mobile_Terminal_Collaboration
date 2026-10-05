from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)


def critical_classes(per_class: Sequence[float], threshold: float) -> List[int]:
    return [c for c, f in enumerate(per_class) if f < threshold]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    Y: Optional[Mapping[str, Sequence[int]]] = None,
    rare_modalities: Sequence[str] = (),
) -> Dict[str, Any]:
    pc = per_class_f1(y_true, y_pred, num_classes)
    out: Dict[str, Any] = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(pc)),
        "per_class_f1": [float(v) for v in pc],
    }
    if Y:
        per_modality = {m: float(np.mean(pc[list(Y_m)])) if len(Y_m) else float("nan") for m, Y_m in Y.items()}
        out["per_modality_f1"] = per_modality
        out["rare_mod_f1"] = float(np.mean([per_modality[m] for m in rare_modalities if m in per_modality]))
    return out


def time_to_accuracy(history: Sequence[Mapping[str, Any]], threshold: float, key: str = "macro_f1") -> Optional[Dict[str, float]]:
    for record in history:
        if key in record and record[key] >= threshold:
            return {"rounds": record["round"], "wall_clock": record["wall_clock"], "cum_energy": record["cum_energy"]}
    return None
