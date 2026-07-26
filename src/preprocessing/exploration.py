# src/exploration.py
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =========================================================
# PATH BOOTSTRAP
# =========================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG  # noqa: E402

# =========================================================
# OPTIONAL IMPORTS
# =========================================================

try:
    from underthesea import sent_tokenize, word_tokenize
    HAS_UNDERTHESEA = True
except Exception:
    sent_tokenize = None
    word_tokenize = None
    HAS_UNDERTHESEA = False

try:
    from src.preprocessing.rule_based import RuleBasedNormalizer  # noqa: E402
except Exception:
    RuleBasedNormalizer = None

try:
    from src.preprocessing.dictionary_based import DictionaryBasedNormalizer  # noqa: E402
except Exception:
    DictionaryBasedNormalizer = None

# =========================================================
# REGEX
# =========================================================

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(?xi)\b(?:https?://\S+|www\.\S+)\b")
_EMAIL_RE = re.compile(r"(?xi)\b[\w.%+-]+@[\w.-]+\.\w{2,}\b")
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    re.UNICODE,
)
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}", re.UNICODE)
_WORD_RE = re.compile(r"\b[\wÀ-ỹà-ỹ]+\b", re.UNICODE)
_TEENCODE_HINTS = {
    "ko", "k", "hok", "hông", "hong", "hem", "hẻm", "đc", "dc", "j", "vkl",
    "vl", "lm", "lắm", "thik", "thjk", "ik", "ib", "wa", "quáaaa", "ngonnn",
}
_EMOTICON_PATTERNS = [
    r":-\)", r":\)", r"\(:", r":D", r":-D", r"=\)", r":\(", r":-\(",
    r";\)", r";-\)", r":P", r":-P", r":O", r":-O", r"XD", r"xD",
    r"T_T", r"ToT", r"TT_TT", r":3", r":v", r"\^_\^", r"<3",
]
_EMOTICON_RE = re.compile("|".join(_EMOTICON_PATTERNS))

# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class ExplorationConfig:
    raw_dir: Path = CONFIG.path.data_dir / "raw"
    processed_dir: Path = CONFIG.path.data_dir / "processed"
    outputs_dir: Path = CONFIG.path.outputs_dir / "exploration"

    text_columns: Tuple[str, ...] = ("review", "text", "content", "comment", "sentence", "raw_text")
    label_columns: Tuple[str, ...] = ("label", "sentiment", "target", "class", "rating")

    sample_size_for_manual_plot: int = 5_000
    max_rows_per_file_to_scan: Optional[int] = None

    save_png: bool = True
    dpi: int = 160

# =========================================================
# UTILITIES
# =========================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    if suffix == ".json":
        try:
            return pd.read_json(path, orient="records")
        except Exception:
            return pd.DataFrame()

    if suffix == ".jsonl":
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.DataFrame()

    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def list_dataset_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    supported = {".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported)


def infer_text_column(df: pd.DataFrame, preferred: Sequence[str]) -> Optional[str]:
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            return col
    return None


def load_texts_from_folder(folder: Path, preferred_columns: Sequence[str]) -> List[str]:
    texts: List[str] = []
    for path in list_dataset_files(folder):
        df = read_table(path)
        if df.empty:
            continue
        col = infer_text_column(df, preferred_columns)
        if col is None:
            continue
        series = df[col].fillna("").astype(str)
        texts.extend(series.tolist())
    return texts

# =========================================================
# FEATURE COUNTERS
# =========================================================

def token_count(text: str) -> int:
    text = normalize_whitespace(text)
    if not text:
        return 0
    if HAS_UNDERTHESEA and word_tokenize is not None:
        try:
            return len([t for t in word_tokenize(text, format="list") if t and t.strip()])
        except Exception:
            pass
    return len(_WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    text = normalize_whitespace(text)
    if not text:
        return 0
    if HAS_UNDERTHESEA and sent_tokenize is not None:
        try:
            sents = sent_tokenize(text)
            sents = [s for s in sents if s and str(s).strip()]
            return len(sents) if sents else 1
        except Exception:
            pass
    return max(1, len(re.findall(r"[.!?…]+", text)) or 1)


def count_emojis(text: str) -> int:
    return len(_EMOJI_RE.findall(text or ""))


def count_emoticons(text: str) -> int:
    return len(_EMOTICON_RE.findall(text or ""))


def count_urls(text: str) -> int:
    return len(_URL_RE.findall(text or ""))


def count_emails(text: str) -> int:
    return len(_EMAIL_RE.findall(text or ""))


def count_mentions(text: str) -> int:
    return len(_MENTION_RE.findall(text or ""))


def count_repeated_chars(text: str) -> int:
    return len(_REPEAT_CHAR_RE.findall(text or ""))


def count_teencode(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"\b[\wÀ-ỹà-ỹ]+\b", text.lower(), flags=re.UNICODE)
    return sum(1 for tok in tokens if tok in _TEENCODE_HINTS)


def looks_english_token(token: str) -> bool:
    token = token.strip()
    if not token or len(token) <= 1:
        return False
    if any(ch.isdigit() for ch in token) or "_" in token:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", token))


def count_english_tokens(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"\b[A-Za-z][A-Za-z\-']*\b", text)
    return sum(1 for tok in tokens if looks_english_token(tok))


def load_phrase_map(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_phrase_dictionary() -> Dict[str, Any]:
    candidates = [
        CONFIG.path.dictionaries_dir / "food_phrases.json",
        CONFIG.path.dictionaries_dir / "teencode.json",
        CONFIG.path.dictionaries_dir / "abbreviation.json",
        CONFIG.path.dictionaries_dir / "emoji.json",
        CONFIG.path.dictionaries_dir / "emoticon.json",
    ]
    merged: Dict[str, Any] = {}
    for p in candidates:
        merged.update(load_phrase_map(p))
    return merged


def count_phrase_hits(text: str, phrase_map: Dict[str, Any]) -> int:
    if not text or not phrase_map:
        return 0
    lowered = normalize_whitespace(text).casefold()
    keys = [normalize_whitespace(k).casefold() for k in phrase_map.keys() if " " in normalize_whitespace(k)]
    keys.sort(key=lambda s: len(s.split()), reverse=True)
    count = 0
    for phrase in keys:
        if phrase and phrase in lowered:
            count += lowered.count(phrase)
    return count


def compute_stats(texts: Sequence[str], phrase_map: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    texts = [normalize_whitespace(t) for t in texts if normalize_whitespace(t)]
    if not texts:
        return {
            "num_rows": 0.0,
            "avg_chars": 0.0,
            "avg_words": 0.0,
            "avg_sentences": 0.0,
            "avg_emojis": 0.0,
            "avg_emoticons": 0.0,
            "avg_urls": 0.0,
            "avg_emails": 0.0,
            "avg_mentions": 0.0,
            "avg_repeated_chars": 0.0,
            "avg_teencode": 0.0,
            "avg_english_tokens": 0.0,
            "avg_phrase_hits": 0.0,
        }

    phrase_map = phrase_map or {}
    n = len(texts)
    return {
        "num_rows": float(n),
        "avg_chars": float(sum(len(t) for t in texts) / n),
        "avg_words": float(sum(token_count(t) for t in texts) / n),
        "avg_sentences": float(sum(sentence_count(t) for t in texts) / n),
        "avg_emojis": float(sum(count_emojis(t) for t in texts) / n),
        "avg_emoticons": float(sum(count_emoticons(t) for t in texts) / n),
        "avg_urls": float(sum(count_urls(t) for t in texts) / n),
        "avg_emails": float(sum(count_emails(t) for t in texts) / n),
        "avg_mentions": float(sum(count_mentions(t) for t in texts) / n),
        "avg_repeated_chars": float(sum(count_repeated_chars(t) for t in texts) / n),
        "avg_teencode": float(sum(count_teencode(t) for t in texts) / n),
        "avg_english_tokens": float(sum(count_english_tokens(t) for t in texts) / n),
        "avg_phrase_hits": float(sum(count_phrase_hits(t, phrase_map) for t in texts) / n),
    }


def build_row_metrics(texts: Sequence[str], dataset_name: str, phrase_map: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for t in texts:
        rows.append({
            "dataset": dataset_name,
            "chars": len(t),
            "words": token_count(t),
            "sentences": sentence_count(t),
            "emojis": count_emojis(t),
            "emoticons": count_emoticons(t),
            "urls": count_urls(t),
            "emails": count_emails(t),
            "mentions": count_mentions(t),
            "repeated_chars": count_repeated_chars(t),
            "teencode": count_teencode(t),
            "english_tokens": count_english_tokens(t),
            "phrase_hits": count_phrase_hits(t, phrase_map),
        })
    return pd.DataFrame(rows)

# =========================================================
# PLOTTING
# =========================================================

def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def save_fig(fig: plt.Figure, path: Path, save_png: bool, dpi: int = 160) -> None:
    if save_png:
        ensure_dir(path.parent)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.show()
    plt.close(fig)


def plot_summary_bar(summary_df: pd.DataFrame, out_dir: Path, save_png: bool, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    melted = summary_df.melt(id_vars=["dataset"], var_name="metric", value_name="value")
    sns.barplot(data=melted, x="metric", y="value", hue="dataset", ax=ax)
    ax.set_title("So sánh thống kê trung bình giữa raw và processed")
    ax.set_xlabel("")
    ax.set_ylabel("Giá trị")
    ax.tick_params(axis="x", rotation=35)
    save_fig(fig, out_dir / "summary_bar.png", save_png, dpi)


def plot_distribution(df: pd.DataFrame, column: str, title: str, out_path: Path, save_png: bool, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df, x=column, hue="dataset", kde=True, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(title)
    save_fig(fig, out_path, save_png, dpi)


def plot_boxplot(df: pd.DataFrame, column: str, title: str, out_path: Path, save_png: bool, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="dataset", y=column, ax=ax)
    ax.set_title(title)
    save_fig(fig, out_path, save_png, dpi)


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, save_png: bool, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue="dataset", alpha=0.5, ax=ax)
    ax.set_title(title)
    save_fig(fig, out_path, save_png, dpi)


def plot_heatmap(df: pd.DataFrame, out_path: Path, save_png: bool, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.drop(columns=["dataset"]).corr(numeric_only=True)
    sns.heatmap(corr, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Heatmap tương quan các đặc trưng")
    save_fig(fig, out_path, save_png, dpi)


def plot_top_tokens(texts: Sequence[str], out_path: Path, save_png: bool, dpi: int) -> None:
    stop = {
        "và", "là", "của", "cho", "một", "những", "các", "thì", "mình", "bạn", "tôi", "anh", "chị",
        "ở", "tại", "rất", "khá", "hơi", "đã", "đang", "sẽ", "cũng", "này", "kia", "ấy"
    }
    counter = Counter()
    for text in texts:
        toks = re.findall(r"\b[\wÀ-ỹà-ỹ]+\b", normalize_whitespace(text).lower(), flags=re.UNICODE)
        for tok in toks:
            if tok not in stop and len(tok) > 1:
                counter[tok] += 1
    top = pd.DataFrame(counter.most_common(25), columns=["token", "count"])
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=top, y="token", x="count", ax=ax)
    ax.set_title("Top token xuất hiện nhiều nhất")
    save_fig(fig, out_path, save_png, dpi)

# =========================================================
# REPORTING
# =========================================================

def print_table(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)


def save_tables(summary_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    try:
        summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    except Exception:
        pass

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    config = ExplorationConfig()
    ensure_dir(config.outputs_dir)
    set_style()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw dir: {config.raw_dir}")
    print(f"Processed dir: {config.processed_dir}")
    print(f"Outputs dir: {config.outputs_dir}")
    print(f"Underthesea available: {HAS_UNDERTHESEA}")

    raw_texts = load_texts_from_folder(config.raw_dir, config.text_columns)
    processed_texts = load_texts_from_folder(config.processed_dir, config.text_columns)

    if config.max_rows_per_file_to_scan is not None:
        raw_texts = raw_texts[:config.max_rows_per_file_to_scan]
        processed_texts = processed_texts[:config.max_rows_per_file_to_scan]

    if not raw_texts:
        print("Khong tim thay du lieu trong data/raw.")
        return
    if not processed_texts:
        print("Khong tim thay du lieu trong data/processed.")
        return

    phrase_map = get_phrase_dictionary()
    raw_stats = compute_stats(raw_texts, phrase_map)
    processed_stats = compute_stats(processed_texts, phrase_map)

    summary_df = pd.DataFrame([
        {"dataset": "raw", **raw_stats},
        {"dataset": "processed", **processed_stats},
    ])
    print_table(summary_df, "TONG HOP THONG KE CHINH")
    save_tables(summary_df, config.outputs_dir)

    raw_metrics = build_row_metrics(raw_texts, "raw", phrase_map)
    processed_metrics = build_row_metrics(processed_texts, "processed", phrase_map)
    merged_metrics = pd.concat([raw_metrics, processed_metrics], ignore_index=True)

    raw_metrics.to_csv(config.outputs_dir / "raw_metrics.csv", index=False, encoding="utf-8-sig")
    processed_metrics.to_csv(config.outputs_dir / "processed_metrics.csv", index=False, encoding="utf-8-sig")
    merged_metrics.to_csv(config.outputs_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")

    feature_order = [
        "chars", "words", "sentences", "emojis", "emoticons", "urls", "emails",
        "mentions", "repeated_chars", "teencode", "english_tokens", "phrase_hits",
    ]
    feature_table = pd.DataFrame({
        "feature": feature_order,
        "raw_avg": [raw_metrics[c].mean() for c in feature_order],
        "processed_avg": [processed_metrics[c].mean() for c in feature_order],
    })
    print_table(feature_table, "THONG KE CHI TIET THEO DAC TRUNG")
    feature_table.to_csv(config.outputs_dir / "feature_table.csv", index=False, encoding="utf-8-sig")

    # Plots
    plot_summary_bar(summary_df, config.outputs_dir, config.save_png, config.dpi)
    plot_distribution(merged_metrics, "words", "Phan phoi so tu moi dong", config.outputs_dir / "words_distribution.png", config.save_png, config.dpi)
    plot_distribution(merged_metrics, "sentences", "Phan phoi so cau moi dong", config.outputs_dir / "sentences_distribution.png", config.save_png, config.dpi)
    plot_distribution(merged_metrics, "teencode", "Phan phoi so tu viet teencode", config.outputs_dir / "teencode_distribution.png", config.save_png, config.dpi)
    plot_distribution(merged_metrics, "english_tokens", "Phan phoi so tu tieng Anh", config.outputs_dir / "english_distribution.png", config.save_png, config.dpi)
    plot_distribution(merged_metrics, "phrase_hits", "Phan phoi so cum tu ghep", config.outputs_dir / "phrase_distribution.png", config.save_png, config.dpi)
    plot_boxplot(merged_metrics, "words", "Boxplot so tu", config.outputs_dir / "words_boxplot.png", config.save_png, config.dpi)
    plot_scatter(merged_metrics, "words", "sentences", "Quan he giua so tu va so cau", config.outputs_dir / "words_sentences_scatter.png", config.save_png, config.dpi)
    plot_heatmap(merged_metrics, config.outputs_dir / "correlation_heatmap.png", config.save_png, config.dpi)
    plot_top_tokens(raw_texts, config.outputs_dir / "top_tokens_raw.png", config.save_png, config.dpi)
    plot_top_tokens(processed_texts, config.outputs_dir / "top_tokens_processed.png", config.save_png, config.dpi)

    print("\nDa xong. Ket qua da luu tai:")
    print(config.outputs_dir)
    print("Files goi y:")
    print(" - summary.csv")
    print(" - feature_table.csv")
    print(" - raw_metrics.csv")
    print(" - processed_metrics.csv")
    print(" - words_distribution.png")
    print(" - sentences_distribution.png")
    print(" - teencode_distribution.png")
    print(" - english_distribution.png")
    print(" - phrase_distribution.png")
    print(" - correlation_heatmap.png")
    print(" - top_tokens_raw.png")
    print(" - top_tokens_processed.png")


if __name__ == "__main__":
    main()
