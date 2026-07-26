"""
src/modeling/trainer.py

Trainer ABSA v2 cho PhoBERTForABSA two-stage:
- BCE cho aspect presence.
- CE cho polarity chỉ trên aspect được nhắc tới.
- Threshold tuning trên dev set dùng F-beta với beta < 1 để giảm dự đoán thừa aspect.
"""

from __future__ import annotations

import logging
import os
import random
import contextlib
import io
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from src import config
from src.modeling.model import get_aspect_names, get_device
from src.modeling.evaluator import (
    collect_outputs,
    compute_absa_metrics,
    decode_predictions,
    evaluate,
    save_thresholds,
    tune_thresholds_from_probs,
)

logger = logging.getLogger(__name__)


def _make_grad_scaler(use_amp: bool):
    """Tương thích PyTorch mới/cũ, tránh FutureWarning torch.cuda.amp.GradScaler."""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def _autocast_context(use_amp: bool):
    """Tương thích PyTorch mới/cũ, tránh FutureWarning torch.cuda.amp.autocast."""
    try:
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.autocast(enabled=use_amp)


@contextlib.contextmanager
def _maybe_quiet_hf_save():
    """
    Ẩn progress 'Writing model shards' khi save_pretrained.
    Nếu muốn hiện progress, đặt config.SUPPRESS_HF_WARNINGS = False.
    """
    if not bool(getattr(config, "SUPPRESS_HF_WARNINGS", True)):
        yield
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(getattr(config, "CUDNN_DETERMINISTIC", False))
    torch.backends.cudnn.benchmark = bool(getattr(config, "CUDNN_BENCHMARK", True))


class ABSADataset(Dataset):
    def __init__(self, texts, aspect_labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.aspect_labels = np.asarray(aspect_labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.aspect_labels[idx], dtype=torch.long)
        return item


class EarlyStopping:
    def __init__(self, patience: int = 4, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True
        improved = score > self.best_score + self.min_delta if self.mode == "max" else score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_df,
        val_df,
        text_col: str = "phobert_text",
        aspect_cols: Optional[List[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_device()
        self.aspect_cols = list(aspect_cols) if aspect_cols else get_aspect_names()
        self.text_col = text_col

        self.batch_size = int(getattr(config, "BATCH_SIZE", 16))
        self.epochs = int(getattr(config, "EPOCHS", 10))
        self.lr = float(getattr(config, "LEARNING_RATE", 2e-5))
        self.head_lr_multiplier = float(getattr(config, "HEAD_LR_MULTIPLIER", 3.0))
        self.max_len = int(getattr(config, "MAX_LEN", 192))
        self.weight_decay = float(getattr(config, "WEIGHT_DECAY", 0.01))
        self.warmup_ratio = float(getattr(config, "WARMUP_RATIO", 0.08))
        self.patience = int(getattr(config, "EARLY_STOPPING_PATIENCE", 4))
        self.grad_clip_norm = float(getattr(config, "GRAD_CLIP_NORM", 1.0))
        self.seed = int(getattr(config, "RANDOM_SEED", 42))
        self.threshold_beta = float(getattr(config, "THRESHOLD_BETA", 0.85))

        self.checkpoint_dir = getattr(config, "CHECKPOINT_DIR", "models/checkpoints")
        self.saved_model_dir = getattr(config, "SAVED_MODEL_DIR", "models/saved_models")
        self.log_dir = getattr(config, "LOG_DIR", "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.saved_model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        set_seed(self.seed)
        self._setup_logging()

        self._validate_df(train_df, text_col)
        self._validate_df(val_df, text_col)

        self._set_loss_weights(train_df)

        self.train_loader = self._build_dataloader(train_df, text_col, shuffle=True)
        self.val_loader = self._build_dataloader(val_df, text_col, shuffle=False)
        self.val_texts = val_df[text_col].fillna("").astype(str).tolist()

        self.optimizer = self._build_optimizer()
        total_steps = max(1, len(self.train_loader) * self.epochs)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )
        self.early_stopping = EarlyStopping(patience=self.patience, mode="max")
        self.best_thresholds = [float(getattr(config, "DEFAULT_ABSA_THRESHOLD", 0.55))] * len(self.aspect_cols)

    def _setup_logging(self):
        log_file = os.path.join(self.log_dir, f"train_absa_v4_{datetime.now():%Y%m%d_%H%M%S}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )

    def _validate_df(self, df, text_col):
        if text_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột text '{text_col}'. Các cột hiện có: {list(df.columns)}")
        missing = [c for c in self.aspect_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame thiếu cột aspect: {missing}")
        for c in self.aspect_cols:
            bad = sorted(set(df[c].dropna().astype(int).unique()) - {0, 1, 2, 3})
            if bad:
                raise ValueError(f"Cột {c} có label ngoài 0/1/2/3: {bad}")

    def _set_loss_weights(self, train_df):
        labels = train_df[self.aspect_cols].astype(int).values
        mentioned = labels != 0

        pos = mentioned.sum(axis=0).astype(float)
        neg = labels.shape[0] - pos

        # Presence: dùng sqrt + clip nhẹ hơn để học aspect hiếm nhưng không spam aspect.
        presence_pos_weight = np.sqrt((neg + 1.0) / (pos + 1.0))
        presence_pos_weight = np.clip(
            presence_pos_weight,
            1.0,
            float(getattr(config, "MAX_PRESENCE_POS_WEIGHT", 2.8)),
        )

        # Polarity: trọng số RIÊNG THEO ASPECT, shape (num_aspects, 3).
        # Bản cũ dùng trọng số toàn cục [pos,neg,neutral], dễ làm aspect này ảnh hưởng aspect khác.
        max_pol_w = float(getattr(config, "MAX_POLARITY_CLASS_WEIGHT", 2.8))
        polarity_weight = np.ones((labels.shape[1], 3), dtype=float)
        for a in range(labels.shape[1]):
            counts = np.array([(labels[:, a] == i).sum() for i in [1, 2, 3]], dtype=float)
            if counts.sum() <= 0:
                polarity_weight[a] = 1.0
                continue
            max_count = max(counts.max(), 1.0)
            w = np.sqrt(max_count / np.maximum(counts, 1.0))
            # Nếu một lớp chỉ có vài mẫu, không để weight quá lớn vì dễ học vẹt.
            polarity_weight[a] = np.clip(w, 1.0, max_pol_w)

        if hasattr(self.model, "set_loss_weights"):
            self.model.set_loss_weights(presence_pos_weight, polarity_weight)

        logger.info(f"Presence pos_weight: {np.round(presence_pos_weight, 3).tolist()}")
        logger.info("Polarity class_weight per aspect [pos, neg, neutral]:")
        for aspect, w in zip(self.aspect_cols, polarity_weight):
            logger.info(f"  {aspect}: {np.round(w, 3).tolist()}")

    def _build_dataloader(self, df, text_col, shuffle: bool) -> DataLoader:
        dataset = ABSADataset(
            texts=df[text_col].fillna("").astype(str).tolist(),
            aspect_labels=df[self.aspect_cols].astype(int).values,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _build_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            target = backbone_params if name.startswith("backbone.") else head_params
            target.append((name, param))

        def split_decay(named_params, lr):
            return [
                {
                    "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                    "lr": lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
                    "lr": lr,
                    "weight_decay": 0.0,
                },
            ]

        groups = []
        groups.extend(split_decay(backbone_params, self.lr))
        groups.extend(split_decay(head_params, self.lr * self.head_lr_multiplier))
        groups = [g for g in groups if g["params"]]
        return AdamW(groups)

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        use_amp = bool(getattr(config, "USE_AMP", torch.cuda.is_available())) and self.device.type == "cuda"
        scaler = _make_grad_scaler(use_amp)

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            with _autocast_context(use_amp):
                outputs = self.model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()

            total_loss += float(loss.item())
            if step % 50 == 0:
                logger.info(f"  step {step}/{len(self.train_loader)} - loss: {loss.item():.4f}")

        return total_loss / max(len(self.train_loader), 1)

    def _evaluate_and_tune(self):
        collected = collect_outputs(self.model, self.val_loader, self.device)
        thresholds = tune_thresholds_from_probs(
            collected["y_true"],
            collected["presence_probs"],
            collected["polarity_probs"],
            beta=self.threshold_beta,
            min_threshold=float(getattr(config, "MIN_THRESHOLD", 0.30)),
        )
        y_pred = decode_predictions(collected["presence_probs"], collected["polarity_probs"], thresholds)
        metrics = compute_absa_metrics(collected["y_true"], y_pred)
        metrics.update(collected)
        metrics["y_pred"] = y_pred
        metrics["thresholds"] = thresholds
        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False, thresholds=None):
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        if is_best:
            best_dir = os.path.join(self.saved_model_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            with _maybe_quiet_hf_save():
                self.model.save_pretrained(
                    best_dir,
                    safe_serialization=False,
                    max_shard_size=getattr(config, "SAVE_MAX_SHARD_SIZE", "10GB"),
                )
                self.tokenizer.save_pretrained(best_dir)
            if thresholds is not None:
                save_thresholds(thresholds, best_dir)
            logger.info(f"New best model saved to: {best_dir}")

    def train(self):
        logger.info(f"Bắt đầu training ABSA v4 balanced-precision trên device: {self.device}")
        logger.info(
            f"Epochs={self.epochs}, batch_size={self.batch_size}, lr={self.lr}, "
            f"head_lr_multiplier={self.head_lr_multiplier}, max_len={self.max_len}"
        )

        best_score = -1.0
        for epoch in range(1, self.epochs + 1):
            logger.info(f"\n===== Epoch {epoch}/{self.epochs} =====")
            train_loss = self._train_one_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")

            val_metrics = self._evaluate_and_tune()
            thresholds = val_metrics["thresholds"]
            score = val_metrics["polarity_f1"]

            logger.info(
                f"Val - loss: {val_metrics['loss']:.4f} | "
                f"Aspect P/R/F1: {val_metrics['aspect_precision']:.4f}/"
                f"{val_metrics['aspect_recall']:.4f}/{val_metrics['aspect_f1']:.4f} | "
                f"Aspect+Polarity P/R/F1: {val_metrics['polarity_precision']:.4f}/"
                f"{val_metrics['polarity_recall']:.4f}/{val_metrics['polarity_f1']:.4f} | "
                f"Exact row: {val_metrics['exact_row_match']:.4f}"
            )
            logger.info(f"Thresholds: {np.round(thresholds, 2).tolist()}")

            is_best = self.early_stopping.step(score)
            self.save_checkpoint(epoch, is_best=is_best, thresholds=thresholds)
            if is_best:
                best_score = score
                self.best_thresholds = thresholds

            if self.early_stopping.should_stop:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Best val Aspect+Polarity F1: {best_score:.4f}")
        return best_score
