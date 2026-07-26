"""
src/modeling/evaluator.py

Evaluator cho PhoBERT ABSA v4 hai bước:
- presence_logits -> aspect được nhắc tới hay không
- polarity_logits -> positive/negative/neutral nếu aspect được nhắc tới
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src import config
from src.modeling.model import get_aspect_names, get_polarity_names


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def decode_predictions(presence_probs: np.ndarray, polarity_probs: np.ndarray, thresholds=None) -> np.ndarray:
    """
    presence_probs: (N, A)
    polarity_probs: (N, A, 3)
    thresholds: scalar hoặc list độ dài A.
    return y_pred shape (N, A), label 0/1/2/3.
    """
    presence_probs = np.asarray(presence_probs)
    polarity_probs = np.asarray(polarity_probs)
    num_aspects = presence_probs.shape[1]

    if thresholds is None:
        thresholds = np.full(num_aspects, float(getattr(config, "DEFAULT_ABSA_THRESHOLD", 0.5)))
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full(num_aspects, float(thresholds))

    mentioned = presence_probs >= thresholds.reshape(1, -1)
    polarity = np.argmax(polarity_probs, axis=-1) + 1
    return np.where(mentioned, polarity, 0).astype(int)


def compute_absa_metrics(y_true: np.ndarray, y_pred: np.ndarray, none_label: int = 0) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    aspect_names = get_aspect_names()
    num_aspects = y_true.shape[1]

    true_mentioned = y_true != none_label
    pred_mentioned = y_pred != none_label

    tp_aspect = int(np.sum(true_mentioned & pred_mentioned))
    fp_aspect = int(np.sum(~true_mentioned & pred_mentioned))
    fn_aspect = int(np.sum(true_mentioned & ~pred_mentioned))

    aspect_precision = _safe_div(tp_aspect, tp_aspect + fp_aspect)
    aspect_recall = _safe_div(tp_aspect, tp_aspect + fn_aspect)
    aspect_f1 = _safe_div(2 * aspect_precision * aspect_recall, aspect_precision + aspect_recall)

    exact = (y_true == y_pred) & true_mentioned
    tp_pol = int(np.sum(exact))
    fp_pol = int(np.sum(pred_mentioned)) - tp_pol
    fn_pol = int(np.sum(true_mentioned)) - tp_pol

    polarity_precision = _safe_div(tp_pol, tp_pol + fp_pol)
    polarity_recall = _safe_div(tp_pol, tp_pol + fn_pol)
    polarity_f1 = _safe_div(2 * polarity_precision * polarity_recall, polarity_precision + polarity_recall)

    per_aspect_f1 = {}
    for i, name in enumerate(aspect_names[:num_aspects]):
        per_aspect_f1[name] = float(f1_score(y_true[:, i], y_pred[:, i], average="macro", zero_division=0))

    label_accuracy = float(np.mean(y_true == y_pred))
    exact_row_match = float(np.mean(np.all(y_true == y_pred, axis=1)))
    macro_f1 = float(np.mean(list(per_aspect_f1.values()))) if per_aspect_f1 else 0.0

    return {
        "aspect_precision": aspect_precision,
        "aspect_recall": aspect_recall,
        "aspect_f1": aspect_f1,
        "polarity_precision": polarity_precision,
        "polarity_recall": polarity_recall,
        "polarity_f1": polarity_f1,
        "macro_f1": macro_f1,
        "label_accuracy": label_accuracy,
        "exact_row_match": exact_row_match,
        "per_aspect_f1": per_aspect_f1,
    }


@torch.no_grad()
def collect_outputs(model, dataloader, device=None) -> dict:
    model.eval()
    device = device or next(model.parameters()).device

    all_presence, all_polarity, all_labels = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        if outputs.loss is not None:
            total_loss += float(outputs.loss.item())
        n_batches += 1

        all_presence.append(torch.sigmoid(outputs.presence_logits).cpu().numpy())
        all_polarity.append(torch.softmax(outputs.polarity_logits, dim=-1).cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())

    return {
        "presence_probs": np.concatenate(all_presence, axis=0),
        "polarity_probs": np.concatenate(all_polarity, axis=0),
        "y_true": np.concatenate(all_labels, axis=0),
        "loss": total_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, device=None, thresholds=None) -> dict:
    collected = collect_outputs(model, dataloader, device=device)
    y_pred = decode_predictions(collected["presence_probs"], collected["polarity_probs"], thresholds=thresholds)
    metrics = compute_absa_metrics(collected["y_true"], y_pred)
    metrics.update(collected)
    metrics["y_pred"] = y_pred
    return metrics


def _fbeta(precision: float, recall: float, beta: float = 0.7) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def tune_thresholds_from_probs(
    y_true: np.ndarray,
    presence_probs: np.ndarray,
    polarity_probs: np.ndarray,
    grid: Optional[Iterable[float]] = None,
    beta: float = 0.85,
    min_threshold: float = 0.30,
) -> list[float]:
    """
    Tìm threshold riêng cho từng aspect.

    Bản v4 không ép threshold quá cao một chiều. Mục tiêu là cân bằng:
    - giảm spam aspect (precision tối thiểu nếu có thể),
    - nhưng vẫn tránh bỏ sót aspect rõ (recall tối thiểu nếu có thể),
    - metric chính vẫn là Aspect+Polarity F-beta trên dev.
    """
    y_true = np.asarray(y_true)
    num_aspects = y_true.shape[1]
    if grid is None:
        max_threshold = float(getattr(config, "MAX_THRESHOLD", 0.90))
        grid = np.round(np.arange(min_threshold, max_threshold + 1e-9, 0.02), 2)

    polarity_pred = np.argmax(polarity_probs, axis=-1) + 1
    thresholds = []

    min_precision = float(getattr(config, "MIN_ASPECT_PRECISION", 0.45))
    min_recall = float(getattr(config, "MIN_ASPECT_RECALL", 0.25))
    fp_penalty = float(getattr(config, "THRESHOLD_FP_PENALTY", 0.02))
    fn_penalty = float(getattr(config, "THRESHOLD_FN_PENALTY", 0.01))
    default_t = float(getattr(config, "DEFAULT_ABSA_THRESHOLD", 0.55))

    for i in range(num_aspects):
        true_mentioned = y_true[:, i] != 0
        n_pos = int(true_mentioned.sum())
        n_neg = int((~true_mentioned).sum())

        best_any = (default_t, -1e9)
        best_constrained = (None, -1e9)

        for t in grid:
            pred_mentioned = presence_probs[:, i] >= t
            y_pred_i = np.where(pred_mentioned, polarity_pred[:, i], 0)

            exact = (y_pred_i == y_true[:, i]) & true_mentioned
            tp = int(np.sum(exact))
            fp = int(np.sum(pred_mentioned)) - tp
            fn = int(np.sum(true_mentioned)) - tp
            p = _safe_div(tp, tp + fp)
            r = _safe_div(tp, tp + fn)

            # Aspect+Polarity F-beta, kèm penalty nhỏ cho FP/FN để threshold ổn định hơn.
            score = _fbeta(p, r, beta=beta)
            score -= fp_penalty * _safe_div(fp, max(n_neg, 1))
            score -= fn_penalty * _safe_div(fn, max(n_pos, 1))

            # Nếu aspect có quá ít mẫu, không đặt yêu cầu precision/recall quá cứng.
            feasible_precision = p >= min_precision or n_pos < 10
            feasible_recall = r >= min_recall or n_pos < 10
            if feasible_precision and feasible_recall and score > best_constrained[1]:
                best_constrained = (float(t), score)

            if score > best_any[1]:
                best_any = (float(t), score)

        chosen = best_constrained[0] if best_constrained[0] is not None else best_any[0]
        thresholds.append(float(chosen))

    return thresholds


def save_thresholds(thresholds, output_dir: str, filename: str = "absa_thresholds.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    payload = {
        "thresholds": [float(x) for x in thresholds],
        "aspect_names": get_aspect_names(),
        "note": "Presence thresholds tuned on dev set. v4 cân bằng precision/recall và giảm spam aspect.",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_thresholds(model_dir: str, default: float = 0.55):
    path = os.path.join(model_dir, "absa_thresholds.json")
    if not os.path.exists(path):
        return [default] * len(get_aspect_names())
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    values = data.get("thresholds", data if isinstance(data, list) else None)
    if not values:
        return [default] * len(get_aspect_names())
    return [float(x) for x in values]


def save_classification_report(y_true, y_pred, filename: str = "classification_report.json"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics_dir = getattr(config, "METRICS_DIR", "outputs/metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    aspect_names = get_aspect_names()
    polarity_names = get_polarity_names()
    num_aspects = y_true.shape[1]

    full_report = {}
    for i, aspect in enumerate(aspect_names[:num_aspects]):
        labels_present = sorted(set(y_true[:, i].tolist()) | set(y_pred[:, i].tolist()))
        target_names = [polarity_names[j] for j in labels_present]
        report = classification_report(
            y_true[:, i], y_pred[:, i],
            labels=labels_present,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        full_report[aspect] = report

        print(f"\n=== {aspect} ===")
        print(classification_report(
            y_true[:, i], y_pred[:, i],
            labels=labels_present,
            target_names=target_names,
            zero_division=0,
        ))

    overall = compute_absa_metrics(y_true, y_pred)
    full_report["_overall"] = overall

    path = os.path.join(metrics_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print("\n=== TỔNG QUAN ABSA ===")
    print(f"Aspect detection   - P: {overall['aspect_precision']:.4f}  R: {overall['aspect_recall']:.4f}  F1: {overall['aspect_f1']:.4f}")
    print(f"Aspect + Polarity  - P: {overall['polarity_precision']:.4f}  R: {overall['polarity_recall']:.4f}  F1: {overall['polarity_f1']:.4f}")
    print(f"Label accuracy: {overall['label_accuracy']:.4f}")
    print(f"Exact row match: {overall['exact_row_match']:.4f}")
    print(f"Đã lưu classification report vào: {path}")
    return full_report


def plot_confusion_matrix(y_true, y_pred, filename: str = "confusion_matrix.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    figures_dir = getattr(config, "FIGURES_DIR", "outputs/figures")
    os.makedirs(figures_dir, exist_ok=True)

    aspect_names = get_aspect_names()
    polarity_names = get_polarity_names()
    num_aspects = y_true.shape[1]

    n_cols = 4
    n_rows = int(np.ceil(num_aspects / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.7 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, aspect in enumerate(aspect_names[:num_aspects]):
        ax = axes[i]
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=list(range(len(polarity_names))))
        ax.imshow(cm)
        ax.set_title(aspect, fontsize=9)
        ax.set_xticks(np.arange(len(polarity_names)))
        ax.set_yticks(np.arange(len(polarity_names)))
        ax.set_xticklabels(polarity_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(polarity_names, fontsize=7)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=7)

    for j in range(num_aspects, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    path = os.path.join(figures_dir, filename)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Đã lưu confusion matrix vào: {path}")
    return path


def save_error_analysis(y_true, y_pred, texts=None, filename: str = "error_analysis.csv"):
    import pandas as pd

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    aspect_names = get_aspect_names()
    polarity_names = get_polarity_names()

    rows = []
    for n in range(y_true.shape[0]):
        for a in range(y_true.shape[1]):
            if y_true[n, a] != y_pred[n, a]:
                rows.append({
                    "row_id": n,
                    "aspect": aspect_names[a],
                    "true_id": int(y_true[n, a]),
                    "pred_id": int(y_pred[n, a]),
                    "true_label": polarity_names[int(y_true[n, a])],
                    "pred_label": polarity_names[int(y_pred[n, a])],
                    "text": "" if texts is None else str(texts[n]),
                })

    metrics_dir = getattr(config, "METRICS_DIR", "outputs/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu error analysis vào: {path}")
    return path
