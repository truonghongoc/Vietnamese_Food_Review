"""
src/modeling/model.py

PhoBERT ABSA v4: two-stage aspect-based sentiment analysis.

Ý tưởng chính:
- Không ép mỗi aspect thành bài toán 4 lớp trực tiếp như trước.
- Tách thành 2 bước trong cùng một model:
    1) Presence head: aspect này có được nhắc tới không?  (binary)
    2) Polarity head: nếu có nhắc tới thì positive/negative/neutral?
- Dùng aspect-aware attention: mỗi aspect có một vector query riêng để nhìn vào
  các token liên quan trong review. Cách này giảm lỗi "thiếu aspect" và giảm
  việc một câu chung chung bị gán nhầm nhiều aspect.

Encoding label gốc VLSP2018:
    0 = none
    1 = positive
    2 = negative
    3 = neutral
"""

from __future__ import annotations

import os
# Giảm log/warning nhiễu từ HuggingFace khi load/save model.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import math
import contextlib
import io
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers import logging as transformers_logging
from transformers.modeling_outputs import ModelOutput

from src import config

# Tắt các log cảnh báo không ảnh hưởng đến kết quả:
# - "unauthenticated requests to HF Hub"
# - "unexpected lm_head.*" khi nạp checkpoint masked-LM vào RobertaModel
# - "You are using a model of type ..." khi nạp custom PreTrainedModel
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=r".*You are using a model of type.*")
warnings.filterwarnings("ignore", message=r".*Some weights of.*were not used.*")
warnings.filterwarnings("ignore", message=r".*You are sending unauthenticated requests.*")


def _quiet_call(func, *args, **kwargs):
    """
    Gọi các hàm HuggingFace trong chế độ yên lặng để output train sạch hơn.
    Nếu muốn xem đầy đủ warning/log, đặt config.SUPPRESS_HF_WARNINGS = False.
    """
    if not bool(getattr(config, "SUPPRESS_HF_WARNINGS", True)):
        return func(*args, **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return func(*args, **kwargs)


def _from_pretrained_quiet(cls, model_name_or_path, **kwargs):
    """
    Ưu tiên load từ cache local để tránh warning HF Hub. Nếu cache chưa có,
    tự fallback sang online download.
    """
    prefer_local = bool(getattr(config, "HF_PREFER_LOCAL_CACHE", True))
    is_remote_id = isinstance(model_name_or_path, str) and not os.path.exists(model_name_or_path)

    if prefer_local and is_remote_id:
        try:
            return _quiet_call(cls.from_pretrained, model_name_or_path, local_files_only=True, **kwargs)
        except Exception:
            return _quiet_call(cls.from_pretrained, model_name_or_path, **kwargs)

    return _quiet_call(cls.from_pretrained, model_name_or_path, **kwargs)



DEFAULT_ASPECT_NAMES: List[str] = [
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
]

DEFAULT_POLARITY_NAMES: List[str] = ["none", "positive", "negative", "neutral"]


def get_aspect_names() -> List[str]:
    return list(getattr(config, "ASPECT_NAMES", DEFAULT_ASPECT_NAMES))


def get_polarity_names() -> List[str]:
    return list(getattr(config, "POLARITY_NAMES", DEFAULT_POLARITY_NAMES))


def get_device() -> torch.device:
    wanted = getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(wanted, str) and wanted.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(wanted)


class PhoBERTABSAConfig(PretrainedConfig):
    model_type = "phobert_absa_v4"

    def __init__(
        self,
        base_model_name: str = "vinai/phobert-base",
        aspect_names: Optional[List[str]] = None,
        polarity_names: Optional[List[str]] = None,
        hidden_dropout_prob: float = 0.25,
        presence_loss_weight: float = 1.0,
        polarity_loss_weight: float = 1.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.aspect_names = list(aspect_names) if aspect_names else list(DEFAULT_ASPECT_NAMES)
        self.polarity_names = list(polarity_names) if polarity_names else list(DEFAULT_POLARITY_NAMES)
        self.num_aspects = len(self.aspect_names)
        self.num_polarities = len(self.polarity_names)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.presence_loss_weight = presence_loss_weight
        self.polarity_loss_weight = polarity_loss_weight


@dataclass
class ABSAModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    # logits tương thích dạng 4 lớp: none/positive/negative/neutral, dùng cho debug.
    logits: Optional[torch.FloatTensor] = None
    # logits chính của model.
    presence_logits: Optional[torch.FloatTensor] = None       # (batch, aspects)
    polarity_logits: Optional[torch.FloatTensor] = None       # (batch, aspects, 3)


class PhoBERTForABSA(PreTrainedModel):
    config_class = PhoBERTABSAConfig

    def __init__(self, config: PhoBERTABSAConfig):
        super().__init__(config)

        backbone_config = AutoConfig.from_pretrained(config.base_model_name)
        self.backbone = AutoModel.from_config(backbone_config)
        hidden = self.backbone.config.hidden_size
        self.hidden_size = hidden

        self.aspect_embeddings = nn.Embedding(config.num_aspects, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.token_proj = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.presence_head = nn.Linear(hidden, 1)
        self.polarity_head = nn.Linear(hidden, 3)  # positive/negative/neutral only

        # Sẽ được Trainer cập nhật bằng thống kê train set.
        self.register_buffer("presence_pos_weight", torch.ones(config.num_aspects))
        # Có thể là vector (3,) hoặc matrix (num_aspects, 3). Trainer v4 dùng matrix
        # để mỗi aspect có trọng số polarity riêng, tránh một lớp hiếm ở aspect này
        # làm lệch polarity của aspect khác.
        self.register_buffer("polarity_class_weight", torch.ones(config.num_aspects, 3))

        self.post_init()

    def set_loss_weights(self, presence_pos_weight=None, polarity_class_weight=None):
        if presence_pos_weight is not None:
            w = torch.as_tensor(presence_pos_weight, dtype=torch.float)
            self.presence_pos_weight = w.to(self.presence_pos_weight.device)
        if polarity_class_weight is not None:
            w = torch.as_tensor(polarity_class_weight, dtype=torch.float)
            # Hỗ trợ cả dạng cũ (3,) và dạng mới (A,3).
            if w.ndim == 1:
                w = w.unsqueeze(0).expand(self.config.num_aspects, -1).contiguous()
            self.polarity_class_weight = w.to(self.polarity_class_weight.device)

    @staticmethod
    def _asymmetric_bce_with_logits(logits, targets, pos_weight=None):
        """
        Asymmetric focal BCE cho presence:
        - Giảm ảnh hưởng của negative quá dễ (none quá nhiều).
        - Không đẩy pos_weight quá mạnh để tránh spam aspect hiếm.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        gamma_pos = float(getattr(config, "ASL_GAMMA_POS", 0.0))
        gamma_neg = float(getattr(config, "ASL_GAMMA_NEG", 2.0))
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        gamma = torch.where(targets > 0.5, torch.full_like(targets, gamma_pos), torch.full_like(targets, gamma_neg))
        focal = torch.pow((1.0 - pt).clamp(min=1e-4), gamma)
        return (bce * focal).mean()

    def _aspect_attention(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden: (B, T, H)
        return: aspect_context (B, A, H)
        """
        batch_size = last_hidden.size(0)
        aspect_ids = torch.arange(self.config.num_aspects, device=last_hidden.device)
        aspect_emb = self.aspect_embeddings(aspect_ids)  # (A, H)

        q = self.query_proj(aspect_emb).unsqueeze(0).expand(batch_size, -1, -1)  # (B, A, H)
        k = self.token_proj(last_hidden)  # (B, T, H)
        scores = torch.einsum("bah,bth->bat", q, k) / math.sqrt(self.hidden_size)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).bool()  # (B, 1, T)
            scores = scores.masked_fill(~mask, -1e4)

        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bat,bth->bah", weights, last_hidden)
        return context

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        backbone_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**backbone_kwargs)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0, :]

        aspect_context = self._aspect_attention(last_hidden, attention_mask)

        batch_size = last_hidden.size(0)
        aspect_ids = torch.arange(self.config.num_aspects, device=last_hidden.device)
        aspect_emb = self.aspect_embeddings(aspect_ids).unsqueeze(0).expand(batch_size, -1, -1)
        cls_expand = cls.unsqueeze(1).expand(-1, self.config.num_aspects, -1)

        features = torch.cat([aspect_context, aspect_emb, cls_expand], dim=-1)
        features = self.feature_mlp(features)
        features = self.dropout(features)

        presence_logits = self.presence_head(features).squeeze(-1)  # (B, A)
        polarity_logits = self.polarity_head(features)              # (B, A, 3)

        # 4-class compatibility logits. None logit được suy ra từ negative presence.
        none_logit = (-presence_logits).unsqueeze(-1)
        compat_logits = torch.cat([none_logit, polarity_logits], dim=-1)

        loss = None
        if labels is not None:
            labels = labels.long()
            mention_targets = (labels != 0).float()

            # Presence loss: asymmetric focal BCE, giảm over-predict aspect nhưng vẫn học được aspect hiếm.
            if bool(getattr(config, "USE_ASYMMETRIC_PRESENCE_LOSS", True)):
                bce = self._asymmetric_bce_with_logits(
                    presence_logits,
                    mention_targets,
                    pos_weight=self.presence_pos_weight,
                )
            else:
                bce = F.binary_cross_entropy_with_logits(
                    presence_logits,
                    mention_targets,
                    pos_weight=self.presence_pos_weight,
                    reduction="mean",
                )

            # Polarity loss chỉ tính trên aspect thật sự được mention.
            # Dùng aspect-specific class weights (A,3) để không làm lệch polarity giữa các aspect.
            mask = labels != 0
            if mask.any():
                polarity_targets = labels[mask] - 1  # 1/2/3 -> 0/1/2
                flat_logits = polarity_logits[mask]
                per_item_ce = F.cross_entropy(
                    flat_logits,
                    polarity_targets,
                    reduction="none",
                    label_smoothing=float(getattr(config, "LABEL_SMOOTHING", 0.02)),
                )

                aspect_index = torch.arange(labels.size(1), device=labels.device).unsqueeze(0).expand_as(labels)
                aspect_index = aspect_index[mask]
                item_weights = self.polarity_class_weight[aspect_index, polarity_targets]
                ce = (per_item_ce * item_weights).sum() / item_weights.sum().clamp(min=1e-6)
            else:
                ce = torch.zeros((), device=presence_logits.device)

            loss = (
                self.config.presence_loss_weight * bce
                + self.config.polarity_loss_weight * ce
            )

        return ABSAModelOutput(
            loss=loss,
            logits=compat_logits,
            presence_logits=presence_logits,
            polarity_logits=polarity_logits,
        )


def load_model(model_name: str = None, aspect_names: List[str] = None, polarity_names: List[str] = None, freeze_base: bool = False):
    model_name = model_name or getattr(config, "MODEL_NAME", "vinai/phobert-base")
    aspect_names = aspect_names or get_aspect_names()
    polarity_names = polarity_names or get_polarity_names()

    tokenizer = _from_pretrained_quiet(AutoTokenizer, model_name, use_fast=False)
    absa_config = PhoBERTABSAConfig(
        base_model_name=model_name,
        aspect_names=aspect_names,
        polarity_names=polarity_names,
        hidden_dropout_prob=float(getattr(config, "HIDDEN_DROPOUT_PROB", 0.25)),
        presence_loss_weight=float(getattr(config, "PRESENCE_LOSS_WEIGHT", 1.0)),
        polarity_loss_weight=float(getattr(config, "POLARITY_LOSS_WEIGHT", 1.2)),
    )
    model = PhoBERTForABSA(absa_config)

    # Nạp pretrained backbone khi bắt đầu train từ đầu.
    pretrained_backbone = _from_pretrained_quiet(AutoModel, model_name)
    model.backbone.load_state_dict(pretrained_backbone.state_dict())
    del pretrained_backbone

    if freeze_base:
        for param in model.backbone.parameters():
            param.requires_grad = False

    model.to(get_device())
    return tokenizer, model


def load_model_from_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        cfg = _quiet_call(PhoBERTABSAConfig.from_pretrained, checkpoint_path)
        tokenizer = _from_pretrained_quiet(AutoTokenizer, checkpoint_path, use_fast=False)
        model = _quiet_call(PhoBERTForABSA.from_pretrained, checkpoint_path, config=cfg)
    else:
        model_name = getattr(config, "MODEL_NAME", "vinai/phobert-base")
        tokenizer, model = load_model(model_name=model_name)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)

    model.to(get_device())
    model.eval()
    return tokenizer, model
