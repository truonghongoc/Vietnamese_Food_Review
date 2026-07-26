# src/config.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


# =========================================================
# PROJECT ROOT
# =========================================================

def get_project_root() -> Path:
    """
    Tim thu muc goc cua project tu dong.
    Uu tien tim parent co ca 'src' va 'dictionaries'
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "src").exists() and (parent / "dictionaries").exists():
            return parent

    # fallback an toan
    return current.parents[1]


PROJECT_ROOT: Path = get_project_root()


# =========================================================
# HELPERS
# =========================================================

def _to_path(value: str | Path | None, default: Path) -> Path:
    """
    Chuyen gia tri ve Path
    """
    if value is None:
        return default
    if isinstance(value, Path):
        return value.resolve()
    return Path(value).resolve()


def _serialize(value: Any) -> Any:
    """
    Chuyen Path va nested structure sang kieu an toan de debug/json.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


# =========================================================
# PATH CONFIG
# =========================================================

@dataclass(slots=True)
class PathConfig:
    """
    Cau hinh duong dan cho du an
    """

    project_root: Path | str = field(default_factory=get_project_root)

    # base dirs
    src_dir: Path = field(init=False)
    preprocessing_dir: Path = field(init=False)

    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    splits_dir: Path = field(init=False)

    models_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    dictionaries_dir: Path = field(init=False)

    # raw data
    raw_train_path: Path = field(init=False)
    raw_val_path: Path = field(init=False)
    raw_test_path: Path = field(init=False)

    # processed data
    processed_train_path: Path = field(init=False)
    processed_val_path: Path = field(init=False)
    processed_test_path: Path = field(init=False)

    # artifacts
    label_map_path: Path = field(init=False)
    vocabulary_path: Path = field(init=False)
    metrics_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = _to_path(self.project_root, get_project_root())

        self.src_dir = self.project_root / "src"
        self.preprocessing_dir = self.src_dir / "preprocessing"

        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.data_dir / "splits"

        self.models_dir = self.project_root / "models"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.outputs_dir = self.project_root / "outputs"
        self.logs_dir = self.project_root / "logs"

        self.dictionaries_dir = self.project_root / "dictionaries"

        self.raw_train_path = self.raw_dir / "train.csv"
        self.raw_val_path = self.raw_dir / "val.csv"
        self.raw_test_path = self.raw_dir / "test.csv"

        self.processed_train_path = self.processed_dir / "train.parquet"
        self.processed_val_path = self.processed_dir / "val.parquet"
        self.processed_test_path = self.processed_dir / "test.parquet"

        self.label_map_path = self.data_dir / "label_map.json"
        self.vocabulary_path = self.outputs_dir / "vocab.txt"
        self.metrics_path = self.outputs_dir / "metrics.json"

    def ensure_dirs(self) -> None:
        """
        Tao san cac thu muc can thiet
        """
        dirs = [
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.splits_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.outputs_dir,
            self.logs_dir,
            self.dictionaries_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, str]:
        """
        Tra ve duong dan duoi dang string.
        """
        return {
            f.name: str(getattr(self, f.name))
            for f in fields(self)
            if isinstance(getattr(self, f.name), Path)
        }


# =========================================================
# DATASET CONFIG
# =========================================================

@dataclass(slots=True)
class DatasetConfig:
    """
    Cau hinh lien quan den dataset
    """

    text_column: str = "review"
    label_column: str = "label"
    id_column: str = "id"

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    random_state: int = 42
    stratify: bool = True


# =========================================================
# PREPROCESS CONFIG
# =========================================================

@dataclass(slots=True)
class PreprocessConfig:
    """
    Cau hinh tien xu ly tong.
    """

    use_rule_based: bool = True
    use_dictionary_based: bool = True
    use_tokenizer: bool = True

    save_intermediate: bool = True
    save_tokenized_text: bool = True

    output_format: str = "parquet"  # csv | parquet | jsonl
    overwrite: bool = True


# =========================================================
# RULE-BASED CONFIG
# =========================================================

@dataclass(slots=True)
class RuleBasedConfig:
    """
    Cau hinh cho rule-based cleaning
    """

    max_repeated_chars: int = 2

    normalize_unicode: bool = True
    normalize_html_entities: bool = True

    remove_urls: bool = True
    remove_emails: bool = True
    remove_mentions: bool = True

    remove_invisible_chars: bool = True
    normalize_whitespace: bool = True
    normalize_repeated_chars: bool = True


# =========================================================
# DICTIONARY-BASED CONFIG
# =========================================================

@dataclass(slots=True)
class DictionaryConfig:
    """
    Cau hinh cho dictionary-based normalization.
    """

    project_root: Path | str = field(default_factory=get_project_root)

    teencode_path: str = "dictionaries/teencode.json"
    abbreviation_path: str = "dictionaries/abbreviation.json"
    english_food_path: str = "dictionaries/english_food.json"
    emoji_path: str = "dictionaries/emoji.json"
    emoticon_path: str = "dictionaries/emoticon.json"
    negation_path: str = "dictionaries/negation.json"

    use_teencode: bool = True
    use_abbreviation: bool = True
    use_english_food: bool = True
    use_emoji: bool = True
    use_emoticon: bool = True
    use_negation: bool = True

    def __post_init__(self) -> None:
        self.project_root = _to_path(self.project_root, get_project_root())


# =========================================================
# TOKENIZER / TEXT CONFIG
# =========================================================

@dataclass(slots=True)
class TokenizerConfig:
    """
    Cau hinh cho tokenizer.
    """

    use_sentence_tokenization: bool = True
    use_word_tokenization: bool = True
    use_phrase_merge: bool = True

    keep_punctuation: bool = True

    # PhoBERT style: giua cac cau co 2 khoang trang
    sentence_separator: str = "  "

    # path den phrase dictionary
    phrase_dict_path: Optional[str] = None

    # project root
    project_root: Path | str = field(default_factory=get_project_root)

    def __post_init__(self) -> None:
        self.project_root = _to_path(self.project_root, get_project_root())


# =========================================================
# MODEL CONFIG
# =========================================================

@dataclass(slots=True)
class ModelConfig:
    """
    Cau hinh cho model.
    """

    model_name: str = "vinai/phobert-base"
    num_labels: int = 3

    max_length: int = 256
    dropout: float = 0.1

    train_batch_size: int = 16
    eval_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    num_epochs: int = 5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    seed: int = 42
    use_fp16: bool = True
    num_workers: int = 2

    early_stopping_patience: int = 2
    save_best_model: bool = True
    metric_for_best_model: str = "f1"


# =========================================================
# INFERENCE CONFIG
# =========================================================

@dataclass(slots=True)
class InferenceConfig:
    """
    Cau hinh suy luan.
    """

    batch_size: int = 32
    threshold: float = 0.5
    return_probabilities: bool = True


# =========================================================
# LABEL MAP CONFIG
# =========================================================

@dataclass(slots=True)
class LabelConfig:
    """
    Mapping nhan sentiment.
    """

    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            0: "negative",
            1: "neutral",
            2: "positive",
        }
    )

    label2id: Dict[str, int] = field(
        default_factory=lambda: {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }
    )


# =========================================================
# MASTER CONFIG
# =========================================================

@dataclass(slots=True)
class AppConfig:
    """
    Cau hinh tong cho toan du an.
    """

    path: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    rule_based: RuleBasedConfig = field(default_factory=RuleBasedConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    label: LabelConfig = field(default_factory=LabelConfig)

    def ensure_dirs(self) -> None:
        self.path.ensure_dirs()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "dataset": asdict(self.dataset),
            "preprocess": asdict(self.preprocess),
            "rule_based": asdict(self.rule_based),
            "dictionary": _serialize(asdict(self.dictionary)),
            "tokenizer": _serialize(asdict(self.tokenizer)),
            "model": asdict(self.model),
            "inference": asdict(self.inference),
            "label": {
                "id2label": self.label.id2label,
                "label2id": self.label.label2id,
            },
        }

    # =====================================================
    # FACTORY HELPERS
    # =====================================================

    def build_rule_based_config(self) -> RuleBasedConfig:
        return RuleBasedConfig(**asdict(self.rule_based))

    def build_dictionary_config(self) -> DictionaryConfig:
        cfg = asdict(self.dictionary)
        cfg["project_root"] = self.path.project_root
        return DictionaryConfig(**cfg)

    def build_tokenizer_config(self) -> TokenizerConfig:
        cfg = asdict(self.tokenizer)
        cfg["project_root"] = self.path.project_root

        # mac dinh lay phrase dictionary trong thu muc dictionaries
        if cfg.get("phrase_dict_path") in (None, ""):
            cfg["phrase_dict_path"] = str(self.path.dictionaries_dir / "food_phrases.json")

        return TokenizerConfig(**cfg)

    def build_preprocess_pipeline_flags(self) -> Dict[str, bool]:
        return {
            "use_rule_based": self.preprocess.use_rule_based,
            "use_dictionary_based": self.preprocess.use_dictionary_based,
            "use_tokenizer": self.preprocess.use_tokenizer,
        }


# =========================================================
# DEFAULT INSTANCE
# =========================================================

CONFIG = AppConfig()
CONFIG.ensure_dirs()


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MODEL:", CONFIG.model.model_name)
    print("RAW TRAIN:", CONFIG.path.raw_train_path)
    print("PROCESSED TRAIN:", CONFIG.path.processed_train_path)
    print("PHRASE FILE:", CONFIG.path.dictionaries_dir / "food_phrases.json")
    print("NEGATION FILE:", CONFIG.path.dictionaries_dir / "negation.json")