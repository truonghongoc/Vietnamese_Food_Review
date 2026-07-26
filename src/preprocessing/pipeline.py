# src/preprocessing/pipeline.py
"""
Pipeline tiền xử lý cho Food_Review_NLP + VLSP2018 ABSA Restaurant.

Mục tiêu của file này:
- Xử lý 1 câu review thành `phobert_text`.
- Xử lý toàn bộ dataset trong data/raw và xuất data/processed.
- Giữ nguyên các cột aspect của VLSP2018 để train ABSA thật sự.
- Tạo thêm các cột debug khi cần: cleaned, normalized, tokenized, aspect_sentiment.
- Hỗ trợ tách train/dev/test bằng cột `type`.

Lưu ý quan trọng:
- Không xóa dấu câu, không xóa emoji/emoticon, không lowercase toàn bộ.
- PhoBERT cần text đã word-segment, ví dụ: "phục_vụ", "bò_bít_tết".
"""

from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import CONFIG
except Exception:  # pragma: no cover
    CONFIG = None


def _default_project_root() -> str:
    if CONFIG is not None and getattr(CONFIG, "path", None) is not None:
        pr = getattr(CONFIG.path, "project_root", None)
        if pr:
            return str(pr)
    return str(PROJECT_ROOT)


def _default_raw_dir() -> Path:
    if CONFIG is not None and getattr(CONFIG, "path", None) is not None:
        raw_dir = getattr(CONFIG.path, "raw_dir", None)
        if raw_dir:
            return Path(raw_dir)
    return PROJECT_ROOT / "data" / "raw"


def _default_processed_dir() -> Path:
    if CONFIG is not None and getattr(CONFIG, "path", None) is not None:
        processed_dir = getattr(CONFIG.path, "processed_dir", None)
        if processed_dir:
            return Path(processed_dir)
    return PROJECT_ROOT / "data" / "processed"


def _build_config_instance(config_cls, project_root: str):
    """Tạo config linh hoạt: chỉ truyền project_root nếu class hỗ trợ."""
    if config_cls is None:
        return None
    try:
        return config_cls(project_root=project_root)
    except TypeError:
        try:
            return config_cls()
        except TypeError:
            return config_cls


try:
    from .rule_based import RuleBasedNormalizer, RuleBasedConfig
    from .dictionary_based import DictionaryBasedNormalizer, DictionaryConfig
    from .tokenizer import VietnameseTokenizer, TokenizerConfig
except ImportError:  # pragma: no cover
    from src.preprocessing.rule_based import RuleBasedNormalizer, RuleBasedConfig
    from src.preprocessing.dictionary_based import DictionaryBasedNormalizer, DictionaryConfig
    from src.preprocessing.tokenizer import VietnameseTokenizer, TokenizerConfig


__all__ = ["PreprocessingPipelineConfig", "PreprocessingPipeline"]


@dataclass(slots=True)
class PreprocessingPipelineConfig:
    project_root: Optional[str] = None
    rule_based_config: Optional[RuleBasedConfig] = None
    dictionary_config: Optional[DictionaryConfig] = None
    tokenizer_config: Optional[TokenizerConfig] = None

    use_rule_based: bool = True
    use_dictionary_based: bool = True
    use_tokenizer: bool = True

    # Khi True, process_dataset sẽ chuẩn hóa aspect về int 0..3.
    validate_absa_labels: bool = True


class PreprocessingPipeline:
    """
    Luồng xử lý:
        raw text
            -> rule_based.clean()
            -> dictionary_based.normalize()
            -> tokenizer.process()
            -> phobert_text

    File này cũng quản lý dataset-level API để train ABSA.
    """

    _TEXT_COLUMN_CANDIDATES = [
        "text", "review", "Review", "comment", "content", "sentence",
        "review_text", "comment_text", "noi_dung", "binh_luan", "cmt",
        "description", "nội dung", "bình luận",
    ]

    def __init__(self, config: Optional[PreprocessingPipelineConfig] = None) -> None:
        self.config = config or PreprocessingPipelineConfig()
        project_root = self.config.project_root or _default_project_root()

        if self.config.use_rule_based:
            rule_config = self.config.rule_based_config or _build_config_instance(RuleBasedConfig, project_root)
            if hasattr(rule_config, "project_root") and getattr(rule_config, "project_root", None) is None:
                rule_config.project_root = project_root
            self.rule_normalizer = RuleBasedNormalizer(rule_config)
        else:
            self.rule_normalizer = None

        if self.config.use_dictionary_based:
            dict_config = self.config.dictionary_config or _build_config_instance(DictionaryConfig, project_root)
            if hasattr(dict_config, "project_root") and getattr(dict_config, "project_root", None) is None:
                dict_config.project_root = project_root
            self.dictionary_normalizer = DictionaryBasedNormalizer(dict_config)
        else:
            self.dictionary_normalizer = None

        if self.config.use_tokenizer:
            tok_config = self.config.tokenizer_config or _build_config_instance(TokenizerConfig, project_root)
            if hasattr(tok_config, "project_root") and getattr(tok_config, "project_root", None) is None:
                tok_config.project_root = project_root
            self.tokenizer = VietnameseTokenizer(tok_config)
        else:
            self.tokenizer = None

    # =====================================================
    # SINGLE TEXT PIPELINE
    # =====================================================

    @staticmethod
    def _safe_text(text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, float) and pd.isna(text):
            return ""
        return unicodedata.normalize("NFC", str(text)).strip()

    def clean_text(self, text: Any) -> str:
        text = self._safe_text(text)
        if not text:
            return ""
        if self.rule_normalizer is None:
            return text
        return self.rule_normalizer.clean(text)

    def normalize_text(self, text: Any) -> str:
        if not text:
            return ""
        text = self.clean_text(text)
        if self.dictionary_normalizer is not None:
            text = self.dictionary_normalizer.normalize(text)
        return text

    def tokenize_text(self, text: Any) -> List[Dict[str, List[str]]]:
        normalized = self.normalize_text(text)
        if not normalized:
            return []
        if self.tokenizer is None:
            return [{"sentence": normalized, "tokens": normalized.split()}]
        return self.tokenizer.process(normalized)

    def to_phobert_text(self, text: Any) -> str:
        """
        Xuất chuỗi đã word-segment cho PhoBERT.
        Ghép câu bằng 1 khoảng trắng để không sinh token rác kiểu "_Món".
        """
        normalized = self.normalize_text(text)
        if not normalized:
            return ""

        if self.tokenizer is None:
            return normalized

        processed = self.tokenizer.process(normalized)
        sentence_texts: List[str] = []
        for item in processed:
            tokens = item.get("tokens", [])
            if tokens:
                sentence_texts.append(" ".join(str(t).strip() for t in tokens if str(t).strip()))

        return " ".join(sentence_texts).strip()

    def process_text(self, text: Any) -> Dict[str, Any]:
        return self.process(text)

    def process(self, text: Any) -> Dict[str, Any]:
        raw = self._safe_text(text)
        cleaned = self.clean_text(raw)
        normalized = self.normalize_text(raw)
        tokenized = self.tokenize_text(raw)
        phobert_text = self.to_phobert_text(raw)

        return {
            "raw": raw,
            "cleaned": cleaned,
            "normalized": normalized,
            "tokenized": tokenized,
            "phobert_text": phobert_text,
        }

    def process_batch(self, texts: Iterable[Any]) -> List[Dict[str, Any]]:
        return [self.process(text) for text in texts]

    def to_phobert_batch(self, texts: Iterable[Any]) -> List[str]:
        return [self.to_phobert_text(text) for text in texts]

    # =====================================================
    # DATASET-LEVEL API
    # =====================================================

    @classmethod
    def _guess_text_column(cls, columns: Iterable[str]) -> Optional[str]:
        lowered = {str(c).strip().lower(): c for c in columns}
        for cand in cls._TEXT_COLUMN_CANDIDATES:
            key = cand.lower()
            if key in lowered:
                return lowered[key]
        return None

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                return pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="utf-8-sig")
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path, orient="records")
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
            return pd.DataFrame({"text": lines})
        raise ValueError(f"Định dạng file không được hỗ trợ: {path}")

    @staticmethod
    def _list_supported_files(folder: Path, file_pattern: str = "*") -> List[Path]:
        if not folder.exists():
            return []
        supported_ext = {".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls", ".txt"}
        if folder.is_file():
            return [folder] if folder.suffix.lower() in supported_ext else []
        return sorted(
            p for p in folder.glob(file_pattern)
            if p.is_file() and p.suffix.lower() in supported_ext
        )

    @staticmethod
    def get_aspect_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
        exclude_set = set(exclude or []) | {"type", "dataset", "_source_file", "phobert_text"}
        return [c for c in df.columns if c not in exclude_set and "#" in str(c)]

    @staticmethod
    def _coerce_aspect_labels(df: pd.DataFrame, aspect_columns: List[str]) -> pd.DataFrame:
        """
        Chuẩn hóa nhãn aspect về int 0..3.
        NaN/rỗng -> 0 vì trong VLSP 0 nghĩa là aspect không xuất hiện.
        """
        out = df.copy()
        for col in aspect_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
            invalid_mask = ~out[col].isin([0, 1, 2, 3])
            if invalid_mask.any():
                bad_values = sorted(out.loc[invalid_mask, col].unique().tolist())
                raise ValueError(
                    f"Cột aspect '{col}' có nhãn ngoài 0..3: {bad_values}. "
                    "Hãy kiểm tra lại dataset."
                )
        return out

    def build_aspect_sentiment_column(
        self,
        df: pd.DataFrame,
        aspect_columns: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Gộp các cột aspect thành JSON, chỉ giữ aspect != 0 để dễ debug.
        Ví dụ: {"FOOD#QUALITY": 1, "SERVICE#GENERAL": 2}
        """
        cols = aspect_columns if aspect_columns is not None else self.get_aspect_columns(df)

        if not cols or df.empty:
            return pd.Series([json.dumps({}, ensure_ascii=False)] * len(df), index=df.index)

        def _to_json_safe(value: Any) -> Any:
            if pd.isna(value):
                return None
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return int(value) if float(value).is_integer() else float(value)
            return value

        def _row_to_json(row: pd.Series) -> str:
            scores = {}
            for col in cols:
                value = _to_json_safe(row[col])
                if value not in (None, "", 0, "0"):
                    scores[col] = value
            return json.dumps(scores, ensure_ascii=False)

        return df[cols].apply(_row_to_json, axis=1)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: Optional[str] = None,
        output_column: str = "phobert_text",
        keep_intermediate: bool = False,
        aspect_output_column: Optional[str] = "aspect_sentiment",
        aspect_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        if text_column is None:
            text_column = self._guess_text_column(df.columns)
            if text_column is None:
                raise ValueError(
                    "Không tự đoán được cột chứa text review. "
                    "Hãy truyền text_column='ten_cot_cua_ban'."
                )

        if text_column not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{text_column}' trong dataset.")

        cols = aspect_columns if aspect_columns is not None else self.get_aspect_columns(df, exclude=[text_column])
        out = df.copy()
        if self.config.validate_absa_labels and cols:
            out = self._coerce_aspect_labels(out, cols)

        raw_texts = out[text_column].fillna("").astype(str).tolist()

        cleaned_list: List[str] = []
        normalized_list: List[str] = []
        tokenized_json_list: List[str] = []
        phobert_list: List[str] = []

        total = len(raw_texts)
        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=total, desc="Tiền xử lý", unit="dòng")
        except Exception:
            progress = None

        for i, text in enumerate(raw_texts, start=1):
            result = self.process(text)
            cleaned_list.append(result["cleaned"])
            normalized_list.append(result["normalized"])
            tokenized_json_list.append(json.dumps(result["tokenized"], ensure_ascii=False))
            phobert_list.append(result["phobert_text"])

            if progress is not None:
                progress.update(1)
            elif total >= 500 and i % 500 == 0:
                print(f"  ... đã xử lý {i}/{total} dòng")

        if progress is not None:
            progress.close()

        out[output_column] = phobert_list

        if keep_intermediate:
            out["cleaned"] = cleaned_list
            out["normalized"] = normalized_list
            out["tokenized"] = tokenized_json_list

        if aspect_output_column and cols:
            out[aspect_output_column] = self.build_aspect_sentiment_column(out, cols)

        return out

    def load_raw_data(
        self,
        raw_dir: Optional[str] = None,
        file_pattern: str = "*",
    ) -> pd.DataFrame:
        raw_path = Path(raw_dir) if raw_dir else _default_raw_dir()
        if not raw_path.exists():
            raise FileNotFoundError(f"Không tìm thấy raw: {raw_path.resolve()}")

        files = self._list_supported_files(raw_path, file_pattern)
        if not files:
            raise FileNotFoundError(f"Không tìm thấy file hợp lệ trong {raw_path.resolve()}")

        frames: List[pd.DataFrame] = []
        for file_path in files:
            try:
                df = self._read_table(file_path)
            except Exception as e:
                print(f"[WARN] Không đọc được {file_path.name}: {e}")
                continue
            if df.empty:
                continue
            df = df.copy()
            df["_source_file"] = file_path.name
            frames.append(df)

        if not frames:
            raise ValueError(f"Không đọc được dữ liệu nào từ {raw_path.resolve()}")

        return pd.concat(frames, ignore_index=True)

    def process_dataset(
        self,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        text_column: Optional[str] = None,
        output_column: str = "phobert_text",
        keep_intermediate: bool = False,
        file_pattern: str = "*",
        aspect_output_column: Optional[str] = "aspect_sentiment",
        aspect_columns: Optional[List[str]] = None,
        overwrite: bool = True,
    ) -> List[str]:
        """
        Đọc dataset raw, tiền xử lý, ghi .parquet sang data/processed.

        Với VLSP2018, file output vẫn giữ đủ 12 cột aspect + cột type train/dev/test.
        """
        raw_path = Path(raw_dir) if raw_dir else _default_raw_dir()
        processed_path = Path(processed_dir) if processed_dir else _default_processed_dir()

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy raw: {raw_path.resolve()}\n"
                f"CWD hiện tại: {Path.cwd()}"
            )

        processed_path.mkdir(parents=True, exist_ok=True)
        files = self._list_supported_files(raw_path, file_pattern)

        if not files:
            print(f"Không tìm thấy file hợp lệ trong {raw_path.resolve()}")
            return []

        exported: List[str] = []

        for file_path in files:
            print(f"\n>>> Đang xử lý: {file_path.name}")
            try:
                df = self._read_table(file_path)
            except Exception as e:
                print(f"  Lỗi đọc file {file_path.name}: {e}")
                continue

            if df.empty:
                print(f"  File {file_path.name} rỗng, bỏ qua.")
                continue

            col_for_file = text_column or self._guess_text_column(df.columns)
            if col_for_file is None and file_path.suffix.lower() == ".txt":
                col_for_file = "text"

            try:
                out_df = self.process_dataframe(
                    df,
                    text_column=col_for_file,
                    output_column=output_column,
                    keep_intermediate=keep_intermediate,
                    aspect_output_column=aspect_output_column,
                    aspect_columns=aspect_columns,
                )
            except ValueError as e:
                print(f"  Lỗi: {e}")
                continue

            out_path = processed_path / (file_path.stem + ".parquet")
            if out_path.exists() and not overwrite:
                print(f"  Bỏ qua vì đã tồn tại: {out_path}")
                exported.append(str(out_path))
                continue

            try:
                out_df.to_parquet(out_path, index=False)
            except ImportError as e:
                raise ImportError("Cần cài pyarrow để xuất parquet: pip install pyarrow") from e

            print(f"  Đã lưu: {out_path} ({len(out_df)} dòng, cột: {list(out_df.columns)})")
            exported.append(str(out_path))

        print(f"\nHoàn tất. Đã xuất {len(exported)} file vào {processed_path.resolve()}")
        return exported

    def load_processed_data(
        self,
        processed_dir: Optional[str] = None,
        file_pattern: str = "*",
    ) -> pd.DataFrame:
        processed_path = Path(processed_dir) if processed_dir else _default_processed_dir()
        if not processed_path.exists():
            raise FileNotFoundError(f"Không tìm thấy processed: {processed_path.resolve()}")

        files = self._list_supported_files(processed_path, file_pattern)
        if not files:
            raise FileNotFoundError(f"Không tìm thấy file processed hợp lệ trong {processed_path.resolve()}")

        frames: List[pd.DataFrame] = []
        for file_path in files:
            try:
                df = self._read_table(file_path)
            except Exception as e:
                print(f"[WARN] Không đọc được {file_path.name}: {e}")
                continue
            if df.empty:
                continue
            df = df.copy()
            df["_source_file"] = file_path.name
            frames.append(df)

        if not frames:
            raise ValueError(f"Không đọc được dữ liệu nào từ {processed_path.resolve()}")

        return pd.concat(frames, ignore_index=True)

    def load_processed_split(
        self,
        processed_dir: Optional[str] = None,
        file_pattern: str = "*",
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Nếu data/processed có file riêng train/val/dev/test, trả về dict.
        Nếu chỉ có 1 file VLSP có cột type thì hàm này trả None;
        train.py sẽ dùng split_by_type().
        """
        processed_path = Path(processed_dir) if processed_dir else _default_processed_dir()
        if not processed_path.exists():
            return None

        files = self._list_supported_files(processed_path, file_pattern)
        if not files:
            return None

        buckets: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "dev": [], "test": []}

        def _bucket_name(stem: str) -> Optional[str]:
            low = stem.lower()
            if "train" in low and "valid" not in low:
                return "train"
            if "dev" in low:
                return "dev"
            if "val" in low or "valid" in low:
                return "val"
            if "test" in low:
                return "test"
            return None

        for file_path in files:
            bucket = _bucket_name(file_path.stem)
            if bucket is None:
                continue
            df = self._read_table(file_path)
            if df.empty:
                continue
            df = df.copy()
            df["_source_file"] = file_path.name
            buckets[bucket].append(df)

        result: Dict[str, pd.DataFrame] = {}
        for name, frames in buckets.items():
            if frames:
                result[name] = pd.concat(frames, ignore_index=True)

        return result or None

    # =====================================================
    # ABSA HELPERS
    # =====================================================

    def split_by_type(self, df: pd.DataFrame, split_col: str = "type") -> Dict[str, pd.DataFrame]:
        if split_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{split_col}' để tách train/dev/test.")

        result = {}
        for value in df[split_col].dropna().unique():
            key = str(value).strip().lower()
            result[key] = df[df[split_col].astype(str).str.lower() == key].reset_index(drop=True)
        return result

    def prepare_vlsp_absa(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        keep_intermediate: bool = False,
    ) -> pd.DataFrame:
        """
        Tiền xử lý riêng cho VLSP2018-ABSA-Restaurant.csv và trả về DataFrame
        sẵn sàng cho train ABSA: Review + 12 aspect + type + phobert_text.
        """
        if input_path is None:
            input_path = str(_default_raw_dir() / "VLSP2018-ABSA-Restaurant.csv")
        input_path = str(input_path)
        df = self._read_table(Path(input_path))

        out = self.process_dataframe(
            df,
            text_column="Review" if "Review" in df.columns else None,
            output_column="phobert_text",
            keep_intermediate=keep_intermediate,
            aspect_output_column="aspect_sentiment",
        )

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix.lower() == ".csv":
                out.to_csv(output_path, index=False, encoding="utf-8-sig")
            else:
                out.to_parquet(output_path, index=False)
            print(f"Đã lưu VLSP processed: {output_path}")

        return out

    def show_stats(self) -> None:
        print("\n========== PREPROCESSING PIPELINE ==========")
        print(f"Rule-based:        {'ON' if self.rule_normalizer is not None else 'OFF'}")
        print(f"Dictionary-based:  {'ON' if self.dictionary_normalizer is not None else 'OFF'}")
        print(f"Tokenizer:         {'ON' if self.tokenizer is not None else 'OFF'}")
        print("===========================================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing pipeline cho Food Review NLP / VLSP ABSA")
    parser.add_argument("--dataset", action="store_true", help="Xử lý toàn bộ dataset trong data/raw")
    parser.add_argument("--vlsp", action="store_true", help="Xử lý riêng VLSP2018-ABSA-Restaurant.csv")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--output-column", type=str, default="phobert_text")
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args()

    pipeline = PreprocessingPipeline()

    if args.vlsp:
        output = _default_processed_dir() / "VLSP2018-ABSA-Restaurant.parquet"
        pipeline.prepare_vlsp_absa(output_path=str(output), keep_intermediate=args.keep_intermediate)
    elif args.dataset:
        pipeline.process_dataset(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            text_column=args.text_column,
            output_column=args.output_column,
            keep_intermediate=args.keep_intermediate,
        )
    else:
        text = "Quán này ngon lắm!!!!!! Món bò bít tết rất ngon, phục vụ hơi chậm."
        result = pipeline.process(text)
        print("RAW:", result["raw"])
        print("CLEANED:", result["cleaned"])
        print("NORMALIZED:", result["normalized"])
        print("PHOBERT:", result["phobert_text"])
        pipeline.show_stats()
