"""
src/evaluation/sentiment_metrics.py

Bộ metric chuyên dụng cho bài toán ABSA VLSP2018 Restaurant.

Schema nhãn mặc định:
    0 = none
    1 = positive
    2 = negative
    3 = neutral

Mỗi mẫu có A aspect, vì vậy:
    y_true.shape == y_pred.shape == (N, A)

Metric chính:
1. Aspect detection:
   Đánh giá aspect có được nhắc tới hay không, bỏ qua polarity.

2. Aspect + Polarity:
   Một dự đoán chỉ là true positive khi đúng cả aspect và polarity.
   Đây là metric end-to-end quan trọng nhất để chọn model.

3. Per-aspect metrics:
   Precision/Recall/F1 riêng cho từng aspect, 4-class Macro-F1 và support.

4. Sentiment/polarity diagnostics:
   Đánh giá positive/negative/neutral trên các aspect có nhãn thật.

5. Calibration metrics:
   ECE, Brier score và NLL cho presence head và polarity head nếu truyền
   presence_probs/polarity_probs.

Module này chỉ tính metric; không phụ thuộc PyTorch và không tham gia training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


DEFAULT_ASPECT_NAMES: tuple[str, ...] = (
    "AMBIENCE#GENERAL",
    "DRINKS#PRICES",
    "DRINKS#QUALITY",
    "DRINKS#STYLE&OPTIONS",
    "FOOD#PRICES",
    "FOOD#QUALITY",
    "FOOD#STYLE&OPTIONS",
    "LOCATION#GENERAL",
    "RESTAURANT#GENERAL",
    "RESTAURANT#MISCELLANEOUS",
    "RESTAURANT#PRICES",
    "SERVICE#GENERAL",
)

DEFAULT_LABEL_NAMES: tuple[str, ...] = (
    "none",
    "positive",
    "negative",
    "neutral",
)


# =============================================================================
# VALIDATION AND LOW-LEVEL HELPERS
# =============================================================================

def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    """
    Chuyển list/NumPy/Torch tensor sang NumPy.

    Không import torch để module metric nhẹ và độc lập.
    """
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()

    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} không được rỗng.")
    return array


def _validate_labels(
    y_true: Any,
    y_pred: Any,
    *,
    num_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    true = _to_numpy(y_true, name="y_true")
    pred = _to_numpy(y_pred, name="y_pred")

    if true.ndim != 2 or pred.ndim != 2:
        raise ValueError(
            "y_true và y_pred phải là ma trận 2 chiều shape (N, num_aspects). "
            f"Nhận được {true.shape=} và {pred.shape=}."
        )

    if true.shape != pred.shape:
        raise ValueError(
            f"y_true và y_pred phải cùng shape; nhận được "
            f"{true.shape} và {pred.shape}."
        )

    if not np.issubdtype(true.dtype, np.number):
        raise TypeError("y_true phải chứa label số nguyên.")
    if not np.issubdtype(pred.dtype, np.number):
        raise TypeError("y_pred phải chứa label số nguyên.")

    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true/y_pred chứa NaN hoặc Inf.")

    if not np.all(true == np.floor(true)):
        raise ValueError("y_true phải chứa label số nguyên.")
    if not np.all(pred == np.floor(pred)):
        raise ValueError("y_pred phải chứa label số nguyên.")

    true = true.astype(np.int64, copy=False)
    pred = pred.astype(np.int64, copy=False)

    valid_min = 0
    valid_max = num_labels - 1
    if np.any((true < valid_min) | (true > valid_max)):
        invalid = np.unique(true[(true < valid_min) | (true > valid_max)])
        raise ValueError(
            f"y_true có label ngoài [{valid_min}, {valid_max}]: "
            f"{invalid.tolist()}."
        )
    if np.any((pred < valid_min) | (pred > valid_max)):
        invalid = np.unique(pred[(pred < valid_min) | (pred > valid_max)])
        raise ValueError(
            f"y_pred có label ngoài [{valid_min}, {valid_max}]: "
            f"{invalid.tolist()}."
        )

    return true, pred


def _resolve_names(
    num_aspects: int,
    aspect_names: Optional[Sequence[str]],
    label_names: Optional[Sequence[str]],
) -> tuple[list[str], list[str]]:
    if aspect_names is None:
        if num_aspects <= len(DEFAULT_ASPECT_NAMES):
            aspects = list(DEFAULT_ASPECT_NAMES[:num_aspects])
        else:
            aspects = [
                *DEFAULT_ASPECT_NAMES,
                *[
                    f"ASPECT_{index}"
                    for index in range(
                        len(DEFAULT_ASPECT_NAMES),
                        num_aspects,
                    )
                ],
            ]
    else:
        aspects = [str(name) for name in aspect_names]
        if len(aspects) != num_aspects:
            raise ValueError(
                f"aspect_names phải có {num_aspects} phần tử, "
                f"nhưng nhận được {len(aspects)}."
            )

    labels = (
        list(DEFAULT_LABEL_NAMES)
        if label_names is None
        else [str(name) for name in label_names]
    )
    if len(labels) < 2:
        raise ValueError("label_names phải có ít nhất 2 nhãn.")

    return aspects, labels


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _balanced_accuracy_present_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Balanced accuracy = trung bình recall của các lớp thực sự có trong y_true.

    Tự tính để không phát sinh warning khi y_pred chứa lớp không xuất hiện
    trong y_true của một aspect hiếm.
    """
    present_labels = np.unique(y_true)
    if present_labels.size == 0:
        return 0.0

    recalls: list[float] = []
    for label in present_labels:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))

    return float(np.mean(recalls))


def _prf_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float | int]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(
        2.0 * precision * recall,
        precision + recall,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _binary_set_metrics(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> dict[str, float | int]:
    tp = int(np.sum(true_mask & pred_mask))
    fp = int(np.sum(~true_mask & pred_mask))
    fn = int(np.sum(true_mask & ~pred_mask))
    tn = int(np.sum(~true_mask & ~pred_mask))

    metrics = _prf_from_counts(tp, fp, fn)
    metrics["tn"] = tn
    metrics["support"] = int(np.sum(true_mask))
    metrics["predicted"] = int(np.sum(pred_mask))
    return metrics


def _joint_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    none_label: int,
) -> dict[str, float | int]:
    true_mentioned = y_true != none_label
    pred_mentioned = y_pred != none_label

    exact_correct = (
        (y_true == y_pred)
        & true_mentioned
        & pred_mentioned
    )

    tp = int(np.sum(exact_correct))
    fp = int(np.sum(pred_mentioned)) - tp
    fn = int(np.sum(true_mentioned)) - tp

    metrics = _prf_from_counts(tp, fp, fn)
    metrics["support"] = int(np.sum(true_mentioned))
    metrics["predicted"] = int(np.sum(pred_mentioned))
    return metrics


def _sample_jaccard(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> float:
    intersection = np.sum(true_mask & pred_mask, axis=1)
    union = np.sum(true_mask | pred_mask, axis=1)

    # Hai tập rỗng được xem là khớp hoàn toàn.
    scores = np.where(
        union == 0,
        1.0,
        intersection / np.maximum(union, 1),
    )
    return float(np.mean(scores))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


# =============================================================================
# CORE ABSA METRICS
# =============================================================================

def compute_absa_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    none_label: int = 0,
    aspect_names: Optional[Sequence[str]] = None,
    label_names: Optional[Sequence[str]] = None,
    include_details: bool = True,
) -> dict[str, Any]:
    """
    Tính bộ metric ABSA đầy đủ.

    Parameters
    ----------
    y_true, y_pred:
        Ma trận nhãn shape (N, A).

    none_label:
        ID của nhãn không nhắc đến aspect. Mặc định 0.

    aspect_names:
        Danh sách tên A aspect. Bỏ trống để dùng schema VLSP2018.

    label_names:
        Tên nhãn theo thứ tự ID. Mặc định:
        ["none", "positive", "negative", "neutral"].

    include_details:
        Nếu True, trả thêm per_aspect, per_label và confusion matrix.

    Returns
    -------
    dict
        Giữ các key tương thích với evaluator.py hiện tại:
        - aspect_precision, aspect_recall, aspect_f1
        - polarity_precision, polarity_recall, polarity_f1
        - macro_f1
        - label_accuracy
        - exact_row_match
        - per_aspect_f1

        Đồng thời bổ sung các nhóm metric chi tiết hơn.
    """
    labels = (
        list(DEFAULT_LABEL_NAMES)
        if label_names is None
        else [str(name) for name in label_names]
    )
    true, pred = _validate_labels(
        y_true,
        y_pred,
        num_labels=len(labels),
    )
    aspects, labels = _resolve_names(
        true.shape[1],
        aspect_names,
        labels,
    )

    if none_label < 0 or none_label >= len(labels):
        raise ValueError(
            f"none_label={none_label} nằm ngoài khoảng label hợp lệ."
        )

    true_mentioned = true != none_label
    pred_mentioned = pred != none_label

    aspect_detection = _binary_set_metrics(
        true_mentioned,
        pred_mentioned,
    )
    joint = _joint_metrics(
        true,
        pred,
        none_label=none_label,
    )

    flattened_true = true.ravel()
    flattened_pred = pred.ravel()
    all_label_ids = list(range(len(labels)))
    sentiment_label_ids = [
        label_id
        for label_id in all_label_ids
        if label_id != none_label
    ]

    label_accuracy = float(
        accuracy_score(flattened_true, flattened_pred)
    )
    exact_row_match = float(
        np.mean(np.all(true == pred, axis=1))
    )
    aspect_set_exact_match = float(
        np.mean(np.all(true_mentioned == pred_mentioned, axis=1))
    )
    aspect_set_jaccard = _sample_jaccard(
        true_mentioned,
        pred_mentioned,
    )

    macro_f1_4class = float(
        f1_score(
            flattened_true,
            flattened_pred,
            labels=all_label_ids,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1_4class = float(
        f1_score(
            flattened_true,
            flattened_pred,
            labels=all_label_ids,
            average="weighted",
            zero_division=0,
        )
    )
    sentiment_macro_f1_end_to_end = float(
        f1_score(
            flattened_true,
            flattened_pred,
            labels=sentiment_label_ids,
            average="macro",
            zero_division=0,
        )
    )

    # Đánh giá polarity tại các vị trí có nhãn thật.
    gold_mask = true_mentioned
    gold_true = true[gold_mask]
    gold_pred = pred[gold_mask]

    if gold_true.size:
        sentiment_on_gold_macro_f1 = float(
            f1_score(
                gold_true,
                gold_pred,
                labels=sentiment_label_ids,
                average="macro",
                zero_division=0,
            )
        )
        sentiment_on_gold_accuracy = float(
            np.mean(gold_true == gold_pred)
        )
    else:
        sentiment_on_gold_macro_f1 = 0.0
        sentiment_on_gold_accuracy = 0.0

    # Polarity conditional: chỉ xem các vị trí cả nhãn thật và dự đoán đều
    # cho rằng aspect được nhắc tới. Metric này là diagnostic, không dùng thay
    # metric end-to-end vì nó không phạt missed/spurious aspect.
    both_mentioned = true_mentioned & pred_mentioned
    if np.any(both_mentioned):
        conditional_polarity_accuracy = float(
            np.mean(true[both_mentioned] == pred[both_mentioned])
        )
        conditional_polarity_macro_f1 = float(
            f1_score(
                true[both_mentioned],
                pred[both_mentioned],
                labels=sentiment_label_ids,
                average="macro",
                zero_division=0,
            )
        )
    else:
        conditional_polarity_accuracy = 0.0
        conditional_polarity_macro_f1 = 0.0

    result: dict[str, Any] = {
        # Backward-compatible keys used by trainer/evaluator.
        "aspect_precision": aspect_detection["precision"],
        "aspect_recall": aspect_detection["recall"],
        "aspect_f1": aspect_detection["f1"],
        "polarity_precision": joint["precision"],
        "polarity_recall": joint["recall"],
        "polarity_f1": joint["f1"],
        "macro_f1": macro_f1_4class,
        "label_accuracy": label_accuracy,
        "exact_row_match": exact_row_match,

        # Clear grouped metrics.
        "aspect_detection": aspect_detection,
        "aspect_polarity_joint": joint,
        "classification": {
            "label_accuracy": label_accuracy,
            "macro_f1_4class": macro_f1_4class,
            "weighted_f1_4class": weighted_f1_4class,
            "sentiment_macro_f1_end_to_end": (
                sentiment_macro_f1_end_to_end
            ),
            "exact_row_match": exact_row_match,
            "aspect_set_exact_match": aspect_set_exact_match,
            "aspect_set_jaccard": aspect_set_jaccard,
        },
        "polarity_diagnostics": {
            "gold_aspect_accuracy": sentiment_on_gold_accuracy,
            "gold_aspect_macro_f1": sentiment_on_gold_macro_f1,
            "conditional_accuracy_when_both_mentioned": (
                conditional_polarity_accuracy
            ),
            "conditional_macro_f1_when_both_mentioned": (
                conditional_polarity_macro_f1
            ),
            "gold_mentions": int(np.sum(true_mentioned)),
            "both_mentioned": int(np.sum(both_mentioned)),
        },
        "metadata": {
            "num_samples": int(true.shape[0]),
            "num_aspects": int(true.shape[1]),
            "num_slots": int(true.size),
            "none_label": int(none_label),
            "aspect_names": aspects,
            "label_names": labels,
        },
    }

    per_aspect: dict[str, dict[str, Any]] = {}
    per_aspect_f1: dict[str, float] = {}

    for aspect_index, aspect_name in enumerate(aspects):
        true_column = true[:, aspect_index]
        pred_column = pred[:, aspect_index]
        true_presence = true_column != none_label
        pred_presence = pred_column != none_label

        presence_metrics = _binary_set_metrics(
            true_presence,
            pred_presence,
        )
        joint_metrics = _joint_metrics(
            true_column.reshape(-1, 1),
            pred_column.reshape(-1, 1),
            none_label=none_label,
        )

        four_class_macro_f1 = float(
            f1_score(
                true_column,
                pred_column,
                labels=all_label_ids,
                average="macro",
                zero_division=0,
            )
        )
        per_aspect_f1[aspect_name] = four_class_macro_f1

        true_gold = true_column[true_presence]
        pred_gold = pred_column[true_presence]
        gold_macro_f1 = (
            float(
                f1_score(
                    true_gold,
                    pred_gold,
                    labels=sentiment_label_ids,
                    average="macro",
                    zero_division=0,
                )
            )
            if true_gold.size
            else 0.0
        )

        per_aspect[aspect_name] = {
            "support": int(np.sum(true_presence)),
            "predicted": int(np.sum(pred_presence)),
            "prevalence": float(np.mean(true_presence)),
            "aspect_detection": presence_metrics,
            "aspect_polarity_joint": joint_metrics,
            "accuracy_4class": float(
                accuracy_score(true_column, pred_column)
            ),
            "balanced_accuracy_4class": (
                _balanced_accuracy_present_classes(
                    true_column,
                    pred_column,
                )
            ),
            "macro_f1_4class": four_class_macro_f1,
            "sentiment_macro_f1_on_gold_mentions": gold_macro_f1,
        }

    result["per_aspect_f1"] = per_aspect_f1

    if include_details:
        precision, recall, f1, support = (
            precision_recall_fscore_support(
                flattened_true,
                flattened_pred,
                labels=all_label_ids,
                zero_division=0,
            )
        )

        per_label = {}
        predicted_counts = np.bincount(
            flattened_pred,
            minlength=len(labels),
        )
        for label_id, label_name in enumerate(labels):
            per_label[label_name] = {
                "label_id": int(label_id),
                "precision": float(precision[label_id]),
                "recall": float(recall[label_id]),
                "f1": float(f1[label_id]),
                "support": int(support[label_id]),
                "predicted": int(predicted_counts[label_id]),
            }

        result["per_aspect"] = per_aspect
        result["per_label"] = per_label
        result["confusion_matrix"] = confusion_matrix(
            flattened_true,
            flattened_pred,
            labels=all_label_ids,
        ).astype(int).tolist()

    return _json_safe(result)


# Backward-friendly aliases.
compute_sentiment_metrics = compute_absa_metrics
calculate_metrics = compute_absa_metrics


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def expected_calibration_error(
    confidences: Any,
    correctness: Any,
    *,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE càng gần 0 càng tốt. Confidence cao không đồng nghĩa model được
    calibration tốt.
    """
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    corr = np.asarray(correctness, dtype=np.float64).reshape(-1)

    if conf.shape != corr.shape:
        raise ValueError(
            "confidences và correctness phải cùng shape."
        )
    if conf.size == 0:
        return 0.0
    if n_bins < 2:
        raise ValueError("n_bins phải >= 2.")
    if np.any(~np.isfinite(conf)) or np.any(~np.isfinite(corr)):
        raise ValueError("ECE input chứa NaN/Inf.")

    conf = np.clip(conf, 0.0, 1.0)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for index in range(n_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]

        if index == 0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf > lower) & (conf <= upper)

        if not np.any(mask):
            continue

        bin_accuracy = float(np.mean(corr[mask]))
        bin_confidence = float(np.mean(conf[mask]))
        bin_weight = float(np.mean(mask))
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_calibration_metrics(
    y_true: Any,
    *,
    presence_probs: Optional[Any] = None,
    polarity_probs: Optional[Any] = None,
    thresholds: float | Sequence[float] = 0.5,
    none_label: int = 0,
    n_bins: int = 15,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Tính ECE/Brier/NLL cho hai head.

    presence_probs:
        Shape (N, A), xác suất aspect được nhắc tới.

    polarity_probs:
        Shape (N, A, 3), xác suất positive/negative/neutral.
        Polarity calibration được tính tại các gold-mentioned aspect.

    thresholds:
        Scalar hoặc vector A phần tử để tạo presence prediction.
    """
    true = _to_numpy(y_true, name="y_true")
    if true.ndim != 2:
        raise ValueError("y_true phải có shape (N, A).")
    true = true.astype(np.int64, copy=False)

    output: dict[str, Any] = {
        "n_bins": int(n_bins),
    }

    if presence_probs is not None:
        probs = np.asarray(presence_probs, dtype=np.float64)
        if probs.shape != true.shape:
            raise ValueError(
                "presence_probs phải cùng shape với y_true; "
                f"nhận được {probs.shape} và {true.shape}."
            )
        if np.any(~np.isfinite(probs)):
            raise ValueError("presence_probs chứa NaN/Inf.")

        probs = np.clip(probs, eps, 1.0 - eps)
        targets = (true != none_label).astype(np.float64)

        threshold_array = np.asarray(
            thresholds,
            dtype=np.float64,
        )
        if threshold_array.ndim == 0:
            threshold_array = np.full(
                true.shape[1],
                float(threshold_array),
            )
        if threshold_array.shape != (true.shape[1],):
            raise ValueError(
                "thresholds phải là scalar hoặc vector num_aspects."
            )

        predictions = (
            probs >= threshold_array.reshape(1, -1)
        )
        correctness = (
            predictions == targets.astype(bool)
        ).astype(np.float64)

        output["presence"] = {
            "ece": expected_calibration_error(
                np.where(predictions, probs, 1.0 - probs),
                correctness,
                n_bins=n_bins,
            ),
            "brier_score": float(np.mean((probs - targets) ** 2)),
            "nll": float(
                -np.mean(
                    targets * np.log(probs)
                    + (1.0 - targets) * np.log(1.0 - probs)
                )
            ),
            "accuracy": float(np.mean(correctness)),
            "num_predictions": int(probs.size),
        }

    if polarity_probs is not None:
        probs = np.asarray(polarity_probs, dtype=np.float64)
        expected_shape = (
            true.shape[0],
            true.shape[1],
            3,
        )
        if probs.shape != expected_shape:
            raise ValueError(
                "polarity_probs phải có shape (N, A, 3); "
                f"nhận được {probs.shape}, mong đợi {expected_shape}."
            )
        if np.any(~np.isfinite(probs)):
            raise ValueError("polarity_probs chứa NaN/Inf.")

        gold_mask = true != none_label
        if np.any(gold_mask):
            selected_probs = probs[gold_mask]
            selected_probs = np.clip(
                selected_probs,
                eps,
                1.0,
            )
            selected_probs = (
                selected_probs
                / selected_probs.sum(axis=1, keepdims=True)
            )

            targets = true[gold_mask] - 1
            if np.any((targets < 0) | (targets > 2)):
                raise ValueError(
                    "Polarity label phải nằm trong {1, 2, 3}."
                )

            predicted = np.argmax(selected_probs, axis=1)
            confidence = np.max(selected_probs, axis=1)
            correctness = (
                predicted == targets
            ).astype(np.float64)

            one_hot = np.eye(3, dtype=np.float64)[targets]
            output["polarity_on_gold_aspects"] = {
                "ece": expected_calibration_error(
                    confidence,
                    correctness,
                    n_bins=n_bins,
                ),
                "brier_score": float(
                    np.mean(
                        np.sum(
                            (selected_probs - one_hot) ** 2,
                            axis=1,
                        )
                    )
                ),
                "nll": float(
                    -np.mean(
                        np.log(
                            selected_probs[
                                np.arange(len(targets)),
                                targets,
                            ]
                        )
                    )
                ),
                "accuracy": float(np.mean(correctness)),
                "num_predictions": int(len(targets)),
            }
        else:
            output["polarity_on_gold_aspects"] = {
                "ece": 0.0,
                "brier_score": 0.0,
                "nll": 0.0,
                "accuracy": 0.0,
                "num_predictions": 0,
            }

    return _json_safe(output)


# =============================================================================
# REPORT EXPORT
# =============================================================================

def save_metrics_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = "sentiment_metrics",
) -> dict[str, str]:
    """
    Lưu:
    - JSON đầy đủ
    - CSV theo aspect
    - CSV theo label
    - CSV confusion matrix

    Hàm import pandas cục bộ để module metric chính vẫn nhẹ.
    """
    import pandas as pd

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    json_path = directory / f"{prefix}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(dict(report)),
            file,
            ensure_ascii=False,
            indent=2,
        )
    saved["json"] = str(json_path)

    per_aspect = report.get("per_aspect", {})
    if isinstance(per_aspect, Mapping) and per_aspect:
        rows = []
        for aspect, values in per_aspect.items():
            presence = values.get("aspect_detection", {})
            joint = values.get("aspect_polarity_joint", {})
            rows.append(
                {
                    "aspect": aspect,
                    "support": values.get("support", 0),
                    "predicted": values.get("predicted", 0),
                    "prevalence": values.get("prevalence", 0.0),
                    "aspect_precision": presence.get("precision", 0.0),
                    "aspect_recall": presence.get("recall", 0.0),
                    "aspect_f1": presence.get("f1", 0.0),
                    "joint_precision": joint.get("precision", 0.0),
                    "joint_recall": joint.get("recall", 0.0),
                    "joint_f1": joint.get("f1", 0.0),
                    "accuracy_4class": values.get(
                        "accuracy_4class",
                        0.0,
                    ),
                    "macro_f1_4class": values.get(
                        "macro_f1_4class",
                        0.0,
                    ),
                    "sentiment_macro_f1_on_gold_mentions": (
                        values.get(
                            "sentiment_macro_f1_on_gold_mentions",
                            0.0,
                        )
                    ),
                }
            )

        aspect_path = directory / f"{prefix}_per_aspect.csv"
        pd.DataFrame(rows).to_csv(
            aspect_path,
            index=False,
            encoding="utf-8-sig",
        )
        saved["per_aspect_csv"] = str(aspect_path)

    per_label = report.get("per_label", {})
    if isinstance(per_label, Mapping) and per_label:
        rows = []
        for label_name, values in per_label.items():
            rows.append(
                {
                    "label": label_name,
                    **dict(values),
                }
            )

        label_path = directory / f"{prefix}_per_label.csv"
        pd.DataFrame(rows).to_csv(
            label_path,
            index=False,
            encoding="utf-8-sig",
        )
        saved["per_label_csv"] = str(label_path)

    matrix = report.get("confusion_matrix")
    metadata = report.get("metadata", {})
    if matrix is not None:
        label_names = metadata.get(
            "label_names",
            list(DEFAULT_LABEL_NAMES),
        )
        matrix_path = (
            directory / f"{prefix}_confusion_matrix.csv"
        )
        pd.DataFrame(
            matrix,
            index=[f"true_{name}" for name in label_names],
            columns=[f"pred_{name}" for name in label_names],
        ).to_csv(
            matrix_path,
            encoding="utf-8-sig",
        )
        saved["confusion_matrix_csv"] = str(matrix_path)

    return saved


def format_metrics_summary(
    report: Mapping[str, Any],
) -> str:
    """Tạo phần tổng kết ngắn để in vào log/console."""
    return (
        "Aspect detection "
        f"P/R/F1={report.get('aspect_precision', 0.0):.4f}/"
        f"{report.get('aspect_recall', 0.0):.4f}/"
        f"{report.get('aspect_f1', 0.0):.4f} | "
        "Aspect+Polarity "
        f"P/R/F1={report.get('polarity_precision', 0.0):.4f}/"
        f"{report.get('polarity_recall', 0.0):.4f}/"
        f"{report.get('polarity_f1', 0.0):.4f} | "
        f"Macro-F1={report.get('macro_f1', 0.0):.4f} | "
        f"Exact row={report.get('exact_row_match', 0.0):.4f}"
    )


__all__ = [
    "DEFAULT_ASPECT_NAMES",
    "DEFAULT_LABEL_NAMES",
    "compute_absa_metrics",
    "compute_sentiment_metrics",
    "calculate_metrics",
    "expected_calibration_error",
    "compute_calibration_metrics",
    "save_metrics_report",
    "format_metrics_summary",
]