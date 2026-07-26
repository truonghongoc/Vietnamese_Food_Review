"""
src/train.py

Train PhoBERT ABSA v2 hai bước cho VLSP2018-ABSA-Restaurant.

Chạy từ thư mục gốc project:
    python -m src.train --reprocess --eval_test
hoặc:
    python src/train.py --reprocess --eval_test
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import logging
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore", message=r".*You are using a model of type.*")
warnings.filterwarnings("ignore", message=r".*You are sending unauthenticated requests.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.modeling.model import get_aspect_names, load_model, load_model_from_checkpoint
from src.modeling.trainer import Trainer
from src.modeling.evaluator import (
    evaluate,
    load_thresholds,
    plot_confusion_matrix,
    save_classification_report,
    save_error_analysis,
)

TEXT_CANDIDATES = (
    "phobert_text",
    "processed_comments",
    "Review",
    "review",
    "text",
    "comment",
    "content",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhoBERT ABSA v2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--clean_checkpoints", action="store_true", help="Xóa checkpoint cũ trước khi train")
    return parser.parse_args()


def apply_overrides(args):
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.max_len is not None:
        config.MAX_LEN = args.max_len


def _get_dir(name: str, default: str):
    return getattr(config, name, default)


def _get_best_checkpoint_path() -> str:
    return os.path.join(_get_dir("SAVED_MODEL_DIR", "models/saved_models"), "best_model")


def _infer_text_column(df: pd.DataFrame) -> Optional[str]:
    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            return col
    return None


def _ensure_processed_data(pipeline: PreprocessingPipeline, force_reprocess: bool = False):
    processed_dir = getattr(getattr(config, "path", None), "processed_dir", None)
    raw_dir = getattr(getattr(config, "path", None), "raw_dir", None)

    has_processed = False
    if processed_dir and os.path.exists(processed_dir):
        for name in os.listdir(processed_dir):
            if name.lower().endswith((".parquet", ".csv", ".json", ".jsonl", ".xlsx", ".xls")):
                has_processed = True
                break

    if force_reprocess or not has_processed:
        print("Đang chạy pipeline để tạo dữ liệu processed từ raw...")
        pipeline.process_dataset(
            raw_dir=str(raw_dir) if raw_dir else None,
            processed_dir=str(processed_dir) if processed_dir else None,
            keep_intermediate=False,
            output_column="phobert_text",
            aspect_output_column="aspect_sentiment",
        )


def _load_training_data(pipeline: PreprocessingPipeline) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    split_data = pipeline.load_processed_split()
    if split_data and "train" in split_data:
        train_df = split_data["train"]
        val_df = split_data.get("dev", split_data.get("val"))
        test_df = split_data.get("test")
        if val_df is not None and not train_df.empty and not val_df.empty:
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), None if test_df is None else test_df.reset_index(drop=True)

    df = pipeline.load_processed_data()
    if "type" in df.columns:
        splits = pipeline.split_by_type(df)
        train_df = splits.get("train")
        val_df = splits.get("dev", splits.get("val"))
        test_df = splits.get("test")
        if train_df is not None and val_df is not None:
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), None if test_df is None else test_df.reset_index(drop=True)

    print("[WARN] Không tìm thấy cột type hoặc file split. Tự chia 90/10, không stratify.")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=getattr(config, "RANDOM_SEED", 42))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), None


def _validate_aspect_columns(df: pd.DataFrame, aspect_cols):
    missing = [c for c in aspect_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột aspect: {missing}")


def _clean_old_checkpoints():
    for p in [_get_dir("CHECKPOINT_DIR", "models/checkpoints"), _get_best_checkpoint_path()]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
            print(f"Đã xóa checkpoint cũ: {p}")


def _evaluate_split(name, model, loader, thresholds, texts=None):
    print(f"\nĐánh giá {name} bằng threshold tối ưu: {[round(x, 2) for x in thresholds]}")
    metrics = evaluate(model, loader, thresholds=thresholds)
    save_classification_report(metrics["y_true"], metrics["y_pred"], filename=f"{name}_classification_report.json")
    plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], filename=f"{name}_confusion_matrix.png")
    save_error_analysis(metrics["y_true"], metrics["y_pred"], texts=texts, filename=f"{name}_error_analysis.csv")
    return metrics


def main():
    args = parse_args()
    apply_overrides(args)

    if args.clean_checkpoints:
        _clean_old_checkpoints()

    pipeline = PreprocessingPipeline()
    _ensure_processed_data(pipeline, force_reprocess=args.reprocess)

    print("Đang load dữ liệu đã xử lý từ pipeline...")
    train_df, val_df, test_df = _load_training_data(pipeline)
    print(f"Train: {len(train_df)} | Dev/Val: {len(val_df)} | Test: {0 if test_df is None else len(test_df)}")

    text_col = _infer_text_column(train_df)
    if text_col is None:
        raise ValueError("Không tìm thấy cột text để train.")

    aspect_cols = get_aspect_names()
    _validate_aspect_columns(train_df, aspect_cols)
    _validate_aspect_columns(val_df, aspect_cols)
    if test_df is not None:
        _validate_aspect_columns(test_df, aspect_cols)

    print(f"Text column: {text_col}")
    print(f"Aspect columns: {aspect_cols}")

    print("Đang load tokenizer + PhoBERT ABSA v2...")
    tokenizer, model = load_model(freeze_base=args.freeze_base)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        text_col=text_col,
        aspect_cols=aspect_cols,
    )
    best_score = trainer.train()
    print(f"\nHoàn tất training. Best val Aspect+Polarity F1: {best_score:.4f}")

    best_dir = _get_best_checkpoint_path()
    if os.path.exists(best_dir):
        print(f"Đang load best model: {best_dir}")
        _, eval_model = load_model_from_checkpoint(best_dir)
        thresholds = load_thresholds(best_dir)
    else:
        print("[WARN] Không tìm thấy best_model. Dùng model hiện tại.")
        eval_model = model
        thresholds = trainer.best_thresholds

    _evaluate_split("val", eval_model, trainer.val_loader, thresholds, texts=trainer.val_texts)

    if args.eval_test and test_df is not None:
        test_trainer = Trainer(
            model=eval_model,
            tokenizer=tokenizer,
            train_df=train_df.iloc[:1].copy(),
            val_df=test_df,
            text_col=text_col,
            aspect_cols=aspect_cols,
        )
        test_texts = test_df[text_col].fillna("").astype(str).tolist()
        _evaluate_split("test", eval_model, test_trainer.val_loader, thresholds, texts=test_texts)


if __name__ == "__main__":
    main()
