"""
src/modeling/inference.py

Inference ABSA v3 precision mode.

Các điểm chính:
- Input luôn chạy qua PreprocessingPipeline trước khi đưa vào PhoBERT.
- Sửa lỗi tiền xử lý kiểu "Đồ ăn_không_ngon" -> "đồ_ăn không_ngon".
- Không dùng keyword để tự thêm aspect.
- Keyword/evidence chỉ dùng để CHẶN bớt false positive khi model dự đoán yếu.
- Có heuristic polarity override nhẹ cho các mẫu rất rõ như "không_ngon", "phục_vụ chậm".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", message=r".*You are using a model of type.*")
warnings.filterwarnings("ignore", message=r".*You are sending unauthenticated requests.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src import config
from src.modeling.evaluator import load_thresholds
from src.modeling.model import (
    get_aspect_names,
    get_device,
    get_polarity_names,
    load_model_from_checkpoint,
)

_tokenizer = None
_model = None
_device = None
_thresholds = None
_preprocessor = None


# ============================================================
# Console UI + progress bar
# ============================================================

class _Console:
    """Output terminal gọn, đẹp và vẫn chạy tốt khi terminal không hỗ trợ màu."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"

    use_color = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

    @classmethod
    def style(cls, value: Any, *codes: str) -> str:
        text = str(value)
        if not cls.use_color:
            return text
        return "".join(codes) + text + cls.RESET

    @classmethod
    def banner(cls, title: str, width: int = 92) -> None:
        line = "═" * width
        print("\n" + cls.style(f"╔{line}╗", cls.CYAN))
        print(
            cls.style("║", cls.CYAN)
            + cls.style(title.center(width), cls.BOLD, cls.CYAN)
            + cls.style("║", cls.CYAN)
        )
        print(cls.style(f"╚{line}╝", cls.CYAN))

    @classmethod
    def section(cls, title: str, width: int = 94) -> None:
        suffix = "─" * max(1, width - len(title) - 4)
        print(
            "\n"
            + cls.style(f"┌─ {title} ", cls.BOLD, cls.BLUE)
            + cls.style(suffix, cls.BLUE)
        )

    @classmethod
    def success(cls, message: str) -> None:
        print(cls.style("✓ ", cls.BOLD, cls.GREEN) + message)

    @classmethod
    def warning(cls, message: str) -> None:
        print(cls.style("⚠ ", cls.BOLD, cls.YELLOW) + message)

    @classmethod
    def error(cls, message: str) -> None:
        print(cls.style("✗ ", cls.BOLD, cls.RED) + message)


def _fmt_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    if digits == 0:
        return f"{int(round(number)):,}"
    return f"{number:,.{digits}f}"


def _print_key_values(items: Iterable[Tuple[str, Any]], width: int = 38) -> None:
    for key, value in items:
        print(f"  {key:<{width}} : {value}")


def _print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_width: int = 28,
    max_rows: Optional[int] = None,
) -> None:
    if not rows:
        print("  (Không có dữ liệu)")
        return

    shown = list(rows if max_rows is None else rows[:max_rows])
    widths = [len(str(header)) for header in headers]

    def compact(value: Any, width: int) -> str:
        raw = str(value).replace("\n", " ")
        return raw if len(raw) <= width else raw[: max(1, width - 1)] + "…"

    for row in shown:
        for index, value in enumerate(row):
            widths[index] = min(max(widths[index], len(str(value))), max_width)

    top = "  ┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    middle = "  ├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "  └" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    print(top)
    print(
        "  │"
        + "│".join(
            f" {compact(headers[index], widths[index]).ljust(widths[index])} "
            for index in range(len(widths))
        )
        + "│"
    )
    print(middle)
    for row in shown:
        print(
            "  │"
            + "│".join(
                f" {compact(row[index], widths[index]).ljust(widths[index])} "
                for index in range(len(widths))
            )
            + "│"
        )
    print(bottom)

    if max_rows is not None and len(rows) > max_rows:
        print(f"  … còn {len(rows) - max_rows:,} dòng")


try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


class _FallbackProgress:
    """Thanh progress dự phòng, không yêu cầu cài tqdm."""

    def __init__(self, total: int, description: str, width: int = 30) -> None:
        self.total = max(int(total), 1)
        self.description = description
        self.width = width
        self.done = 0
        self.started_at = time.perf_counter()
        self._render()

    def update(self, amount: int) -> None:
        self.done = min(self.total, self.done + int(amount))
        self._render()

    def _render(self) -> None:
        ratio = self.done / self.total
        filled = int(self.width * ratio)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        speed = self.done / elapsed
        eta = (self.total - self.done) / speed if speed > 0 else 0.0
        print(
            f"\r  {self.description:<15} [{bar}] "
            f"{self.done:,}/{self.total:,} {ratio * 100:6.2f}% | "
            f"{speed:6.2f} review/s | ETA {eta:6.1f}s",
            end="",
            flush=True,
        )

    def close(self) -> None:
        print()


def _create_progress(total: int, description: str):
    if _tqdm is not None:
        return _tqdm(
            total=total,
            desc=description,
            unit="review",
            dynamic_ncols=True,
            smoothing=0.08,
        )
    return _FallbackProgress(total, description)


# Những phrase này chỉ dùng để sửa output tiền xử lý lỗi, không thay thế tokenizer.
DOMAIN_PHRASE_REPAIRS: Dict[str, str] = {
    "đồ ăn": "đồ_ăn",
    "món ăn": "món_ăn",
    "thức ăn": "thức_ăn",
    "đồ uống": "đồ_uống",
    "thức uống": "thức_uống",
    "trà sữa": "trà_sữa",
    "cà phê": "cà_phê",
    "bò bít tết": "bò_bít_tết",
    "phục vụ": "phục_vụ",
    "nhân viên": "nhân_viên",
    "không gian": "không_gian",
    "giá cả": "giá_cả",
    "nhà hàng": "nhà_hàng",
    "quán ăn": "quán_ăn",
    "hóa đơn": "hóa_đơn",
    "thanh toán": "thanh_toán",
    "giao hàng": "giao_hàng",
    "đậu xe": "đậu_xe",
    "gửi xe": "gửi_xe",
}


# Evidence được chia theo aspect. Các từ quá chung chung được loại khỏi aspect dễ nhầm.
ASPECT_EVIDENCE: Dict[str, List[str]] = {
    "AMBIENCE#GENERAL": [
        "không_gian", "view", "trang_trí", "ấm_cúng", "ồn", "yên_tĩnh",
        "nhạc", "sạch", "bẩn", "thoáng", "chỗ_ngồi", "bàn_ghế",
    ],
    "DRINKS#PRICES": [
        "giá_nước", "giá đồ_uống", "giá thức_uống", "nước đắt", "nước rẻ",
        "trà_sữa đắt", "cà_phê đắt", "đồ_uống rẻ", "đồ_uống hợp_lý",
    ],
    "DRINKS#QUALITY": [
        "đồ_uống", "thức_uống", "nước_uống", "trà_sữa", "cà_phê",
        "cafe", "sinh_tố", "matcha", "nước ép", "trà", "đậm", "nhạt",
    ],
    "DRINKS#STYLE&OPTIONS": [
        "menu_nước", "menu đồ_uống", "nhiều loại nước", "ít loại nước",
        "nhiều đồ_uống", "ít đồ_uống", "trà_sữa", "cà_phê", "option nước",
    ],
    "FOOD#PRICES": [
        "giá", "giá_món", "giá đồ_ăn", "giá thức_ăn", "giá hợp_lý",
        "hợp_lý", "rẻ", "đắt", "mắc", "phần_ăn", "suất", "tiền",
    ],
    "FOOD#QUALITY": [
        "đồ_ăn", "món_ăn", "thức_ăn", "món", "bò_bít_tết", "cơm",
        "phở", "bún", "ngon", "không_ngon", "dở", "tệ", "mặn", "nhạt",
        "nguội", "nóng", "tươi", "ôi", "khó_ăn",
    ],
    "FOOD#STYLE&OPTIONS": [
        "menu", "nhiều_món", "ít_món", "đa_dạng", "lựa_chọn", "combo",
        "size", "option", "món mới", "nhiều lựa_chọn", "ít lựa_chọn",
        "nhiều_vị", "ít_vị", "nhiều hương",
    ],
    "LOCATION#GENERAL": [
        "vị_trí", "địa_điểm", "đường", "hẻm", "gần", "xa", "dễ_tìm",
        "khó_tìm", "đậu_xe", "gửi_xe", "bãi_xe", "trung_tâm",
    ],
    "RESTAURANT#GENERAL": [
        "quán", "nhà_hàng", "chỗ_này", "ở_đây", "trải_nghiệm",
        "lần_sau", "quay_lại", "recommend", "đáng_thử", "khuyên",
    ],
    "RESTAURANT#MISCELLANEOUS": [
        "wifi", "khuyến_mãi", "voucher", "đặt_bàn", "hóa_đơn",
        "thanh_toán", "máy_lạnh", "wc", "toilet", "dịch_vụ_khác",
    ],
    "RESTAURANT#PRICES": [
        "giá_cả", "mức_giá", "bill", "hóa_đơn", "tổng_tiền",
        "chi_phí", "giá quán", "quán đắt", "quán rẻ",
    ],
    "SERVICE#GENERAL": [
        "phục_vụ", "nhân_viên", "order", "gọi_món", "chờ", "lâu",
        "chậm", "nhanh", "thái_độ", "nhiệt_tình", "vui_vẻ", "khó_chịu",
    ],
}

DRINK_TERMS = {"đồ_uống", "thức_uống", "nước_uống", "trà_sữa", "cà_phê", "cafe", "sinh_tố", "matcha", "trà"}
RESTAURANT_PRICE_TERMS = {"giá_cả", "mức_giá", "bill", "hóa_đơn", "tổng_tiền", "chi_phí"}
FOOD_PRICE_TERMS = {"giá", "giá_món", "hợp_lý", "rẻ", "đắt", "mắc", "suất", "phần_ăn"}
FOOD_QUALITY_NEG = {"không_ngon", "dở", "tệ", "khó_ăn", "mặn", "nhạt", "ôi", "nguội"}
SERVICE_NEG = {"chậm", "lâu", "khó_chịu", "tệ", "thái_độ_kém"}
PRICE_POS = {"hợp_lý", "rẻ", "vừa_túi_tiền", "đáng_tiền"}
PRICE_NEG = {"đắt", "mắc", "chát", "không_hợp_lý"}


def _get_default_checkpoint() -> str:
    return os.path.join(getattr(config, "SAVED_MODEL_DIR", "models/saved_models"), "best_model")


def _load_if_needed(checkpoint_path: Optional[str] = None) -> None:
    global _tokenizer, _model, _device, _thresholds
    if _model is not None:
        return

    checkpoint_path = checkpoint_path or _get_default_checkpoint()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy model tại '{checkpoint_path}'. "
            "Hãy train trước bằng: python -m src.train --reprocess --clean_checkpoints"
        )

    _tokenizer, _model = load_model_from_checkpoint(checkpoint_path)
    _device = get_device()
    _thresholds = load_thresholds(
        checkpoint_path,
        default=float(getattr(config, "DEFAULT_ABSA_THRESHOLD", 0.62)),
    )
    if os.getenv("INFERENCE_QUIET", "0") != "1":
        _Console.success(f"Đã tải model: {checkpoint_path}")
        print(f"  Device     : {_device}")
        print(f"  Thresholds : {[round(x, 2) for x in _thresholds]}")


def _get_preprocessor(required: bool = True):
    global _preprocessor
    if _preprocessor is not None:
        return _preprocessor

    try:
        from src.preprocessing.pipeline import PreprocessingPipeline
        _preprocessor = PreprocessingPipeline()
        return _preprocessor
    except Exception as exc:
        if required:
            raise RuntimeError(
                "Không load được PreprocessingPipeline. Kiểm tra src/preprocessing/pipeline.py, "
                "dictionary_based.py, tokenizer.py, rule_based.py và đường dẫn dictionaries/."
            ) from exc
        return None


def repair_phobert_text(text: Any) -> str:
    """Sửa lỗi nối phrase + phủ định sau pipeline."""
    if text is None:
        return ""
    repaired = str(text).strip()
    if not repaired:
        return ""

    neg_words = r"(?:không|chưa|chẳng|chả|đừng|ko|k|kh|hok|hong|hông)"
    for phrase, merged in DOMAIN_PHRASE_REPAIRS.items():
        words = phrase.split()
        if len(words) < 2:
            continue

        space_phrase = r"\s+".join(re.escape(w) for w in words)
        repaired = re.sub(
            rf"(?<!\w){space_phrase}_(?={neg_words}_)",
            f"{merged} ",
            repaired,
            flags=re.IGNORECASE | re.UNICODE,
        )

        underscore_phrase = re.escape(merged)
        repaired = re.sub(
            rf"(?<!\w){underscore_phrase}_(?={neg_words}_)",
            f"{merged} ",
            repaired,
            flags=re.IGNORECASE | re.UNICODE,
        )

    for phrase, merged in DOMAIN_PHRASE_REPAIRS.items():
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"(?!\w)"
        repaired = re.sub(pattern, merged, repaired, flags=re.IGNORECASE | re.UNICODE)

    return re.sub(r"\s+", " ", repaired).strip()


def _extract_text_from_pipeline_output(output: Any, fallback: str) -> str:
    if output is None:
        return fallback

    if isinstance(output, str):
        return output.strip() or fallback

    if isinstance(output, dict):
        # Chỉ ưu tiên phobert_text. Không ưu tiên normalized/cleaned vì đó chưa phải input PhoBERT.
        for key in ("phobert_text", "processed_comments", "tokenized_text"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        tokenized = output.get("tokenized")
        if isinstance(tokenized, list):
            parts: List[str] = []
            for item in tokenized:
                if isinstance(item, dict) and isinstance(item.get("tokens"), list):
                    joined = " ".join(str(t).strip() for t in item["tokens"] if str(t).strip())
                    if joined:
                        parts.append(joined)
            if parts:
                return " ".join(parts).strip()

    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get("tokens"), list):
                joined = " ".join(str(t).strip() for t in item["tokens"] if str(t).strip())
                if joined:
                    parts.append(joined)
        if parts:
            return " ".join(parts).strip()

    return fallback


def preprocess_for_model(text: str) -> str:
    """
    Bắt buộc chạy input qua PreprocessingPipeline trước khi đưa vào tokenizer/model.
    Hàm này cũng sửa lỗi "Đồ ăn_không_ngon" -> "đồ_ăn không_ngon".
    """
    raw_text = (text or "").strip()
    if not raw_text:
        return ""

    preprocessor = _get_preprocessor(required=True)

    if hasattr(preprocessor, "process"):
        output = preprocessor.process(raw_text)
        processed = _extract_text_from_pipeline_output(output, raw_text)
    elif hasattr(preprocessor, "to_phobert_text"):
        processed = _extract_text_from_pipeline_output(preprocessor.to_phobert_text(raw_text), raw_text)
    elif callable(preprocessor):
        processed = _extract_text_from_pipeline_output(preprocessor(raw_text), raw_text)
    else:
        raise RuntimeError("PreprocessingPipeline không có process/to_phobert_text.")

    # Nếu pipeline mới có repair_phobert_text thì dùng; nếu không có thì dùng repair local.
    if hasattr(preprocessor, "repair_phobert_text"):
        processed = preprocessor.repair_phobert_text(processed)
    processed = repair_phobert_text(processed)

    if not processed:
        raise ValueError("Pipeline trả về chuỗi rỗng sau tiền xử lý, không thể inference.")
    return processed


def _tokens(text: str) -> set[str]:
    return set((text or "").casefold().split())


def _keyword_hits(text: str, aspect: str) -> List[str]:
    low = (text or "").casefold()
    hits = []
    for kw in ASPECT_EVIDENCE.get(aspect, []):
        if kw.casefold() in low:
            hits.append(kw)
    return hits


def _has_contextual_evidence(text: str, aspect: str) -> bool:
    low = (text or "").casefold()
    toks = _tokens(low)

    hits = _keyword_hits(low, aspect)
    if not hits:
        return False

    if aspect.startswith("DRINKS#"):
        return bool(toks & DRINK_TERMS) or any(k in low for k in DRINK_TERMS)

    if aspect == "RESTAURANT#PRICES":
        return bool(toks & RESTAURANT_PRICE_TERMS) or any(k in low for k in RESTAURANT_PRICE_TERMS)

    if aspect == "FOOD#PRICES":
        return bool(toks & FOOD_PRICE_TERMS) or "giá" in low

    if aspect == "FOOD#STYLE&OPTIONS":
        style_terms = {"menu", "nhiều_món", "ít_món", "đa_dạng", "lựa_chọn", "combo", "size", "option"}
        return bool(toks & style_terms) or any(k in low for k in style_terms)

    return True



# ============================================================
# Clause-aware evidence + polarity heuristics
# ============================================================

FOOD_ANCHORS = {
    "đồ_ăn", "món_ăn", "thức_ăn", "món", "cơm", "phở", "bún", "bánh",
    "bò_bít_tết", "gà", "thịt", "hải_sản", "lẩu", "pizza", "sushi",
}
DRINK_ANCHORS = {
    "đồ_uống", "thức_uống", "nước_uống", "trà_sữa", "cà_phê", "cafe",
    "sinh_tố", "matcha", "trà", "nước ép", "nước_ép",
}
SERVICE_ANCHORS = {"phục_vụ", "nhân_viên", "order", "gọi_món", "chờ", "thái_độ"}
AMBIENCE_ANCHORS = {"không_gian", "view", "trang_trí", "bàn_ghế", "chỗ_ngồi", "nhạc"}
LOCATION_ANCHORS = {"vị_trí", "địa_điểm", "đường", "hẻm", "đậu_xe", "gửi_xe", "bãi_xe"}
RESTAURANT_ANCHORS = {"quán", "nhà_hàng", "chỗ_này", "ở_đây", "trải_nghiệm"}

FOOD_QUALITY_POS = {
    "ngon", "rất_ngon", "ngon_mê_mẩn", "mê_mẩn", "tasty", "đậm_đà",
    "vừa_miệng", "tươi", "nóng", "giòn", "thơm", "xuất_sắc", "tuyệt",
}
FOOD_QUALITY_NEG = {
    "không_ngon", "dở", "tệ", "khó_ăn", "mặn", "nhạt", "ôi", "nguội",
    "tan_h", "khét", "hôi", "chán", "ngán", "không_tươi",
}
SERVICE_POS = {"nhanh", "nhiệt_tình", "vui_vẻ", "dễ_thương", "lịch_sự", "chu_đáo", "tận_tình"}
SERVICE_NEG = {"chậm", "lâu", "khó_chịu", "thái_độ_kém", "cọc", "thô_lỗ", "lơ", "bực"}
PRICE_POS = {"hợp_lý", "rẻ", "vừa_túi_tiền", "đáng_tiền", "phải_chăng"}
PRICE_NEG = {"đắt", "mắc", "chát", "không_hợp_lý", "quá_giá", "đắt_đỏ"}
AMBIENCE_POS = {"đẹp", "thoáng", "ấm_cúng", "sạch", "yên_tĩnh", "dễ_chịu", "xinh", "sang", "rộng"}
AMBIENCE_NEG = {"bẩn", "ồn", "chật", "nóng", "bí", "hôi", "tối", "khó_chịu"}
AMBIENCE_NEU = {"bình_thường", "tạm", "ổn"}
LOCATION_POS = {"dễ_tìm", "gần", "trung_tâm", "tiện", "dễ_đi", "dễ_đậu_xe"}
LOCATION_NEG = {"khó_tìm", "xa", "hẻm_sâu", "khó_đậu_xe", "khó_gửi_xe"}

CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:[,.;!?]+|\bnhưng\b|\btuy_nhiên\b|\btuy nhiên\b|\bmà\b|\bcòn\b)\s*",
    re.IGNORECASE | re.UNICODE,
)


def _clauses(text: str) -> List[str]:
    low = (text or "").casefold()
    parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(low) if p and p.strip()]
    return parts or [low]


def _contains_any(text: str, terms) -> bool:
    low = (text or "").casefold()
    toks = _tokens(low)
    return bool(toks & set(terms)) or any(str(t).casefold() in low for t in terms)


def _find_terms(text: str, terms) -> List[str]:
    low = (text or "").casefold()
    toks = _tokens(low)
    found = []
    for term in terms:
        t = str(term).casefold()
        if t in toks or t in low:
            found.append(term)
    return found


def _best_local_clause(aspect: str, text: str) -> str:
    """Lấy clause có evidence mạnh nhất cho aspect, tránh lấy sentiment ở clause khác."""
    clauses = _clauses(text)
    anchor_map = {
        "FOOD#QUALITY": FOOD_ANCHORS,
        "FOOD#PRICES": FOOD_ANCHORS | FOOD_PRICE_TERMS | {"giá"},
        "FOOD#STYLE&OPTIONS": {"menu", "nhiều_món", "ít_món", "đa_dạng", "lựa_chọn", "combo", "size", "option"},
        "DRINKS#QUALITY": DRINK_ANCHORS,
        "DRINKS#PRICES": DRINK_ANCHORS | {"giá_nước", "giá đồ_uống", "giá"},
        "DRINKS#STYLE&OPTIONS": DRINK_ANCHORS | {"menu_nước", "nhiều loại nước", "ít loại nước"},
        "SERVICE#GENERAL": SERVICE_ANCHORS,
        "AMBIENCE#GENERAL": AMBIENCE_ANCHORS,
        "LOCATION#GENERAL": LOCATION_ANCHORS,
        "RESTAURANT#GENERAL": RESTAURANT_ANCHORS,
        "RESTAURANT#PRICES": RESTAURANT_PRICE_TERMS,
        "RESTAURANT#MISCELLANEOUS": set(ASPECT_EVIDENCE.get("RESTAURANT#MISCELLANEOUS", [])),
    }
    anchors = anchor_map.get(aspect, set(ASPECT_EVIDENCE.get(aspect, [])))
    best = ""
    best_score = -1
    for cl in clauses:
        score = len(_find_terms(cl, anchors))
        if score > best_score:
            best = cl
            best_score = score
    return best or (text or "").casefold()


def _explicit_evidence_polarity(aspect: str, text: str) -> Tuple[bool, Optional[str], float, List[str]]:
    """
    Trả về evidence rất rõ ràng từ cùng một clause:
      (has_explicit_evidence, polarity, strength, terms)
    Hàm này chỉ rescue những trường hợp chắc, không dùng keyword chung để spam aspect.
    """
    low = (text or "").casefold()
    clause = _best_local_clause(aspect, low)
    terms: List[str] = []

    def collect(*groups):
        out = []
        for g in groups:
            out.extend(_find_terms(clause, g))
        return out

    if aspect == "FOOD#QUALITY":
        anchor = _contains_any(clause, FOOD_ANCHORS)
        neg = _contains_any(clause, FOOD_QUALITY_NEG)
        pos = _contains_any(clause, FOOD_QUALITY_POS)
        terms = collect(FOOD_ANCHORS, FOOD_QUALITY_NEG if neg else FOOD_QUALITY_POS)
        if anchor and neg:
            return True, "negative", 0.97, terms
        if anchor and pos:
            return True, "positive", 0.96, terms

    if aspect == "SERVICE#GENERAL":
        anchor = _contains_any(clause, SERVICE_ANCHORS)
        neg = _contains_any(clause, SERVICE_NEG) or bool(re.search(r"(hơi|quá|rất)?\s*chậm", clause))
        pos = _contains_any(clause, SERVICE_POS)
        terms = collect(SERVICE_ANCHORS, SERVICE_NEG if neg else SERVICE_POS)
        if anchor and neg:
            return True, "negative", 0.97, terms
        if anchor and pos:
            return True, "positive", 0.95, terms

    if aspect == "AMBIENCE#GENERAL":
        anchor = _contains_any(clause, AMBIENCE_ANCHORS)
        neg = _contains_any(clause, AMBIENCE_NEG)
        pos = _contains_any(clause, AMBIENCE_POS)
        neu = _contains_any(clause, AMBIENCE_NEU)
        terms = collect(AMBIENCE_ANCHORS, AMBIENCE_NEG if neg else AMBIENCE_POS if pos else AMBIENCE_NEU)
        if anchor and neg:
            return True, "negative", 0.96, terms
        if anchor and pos:
            return True, "positive", 0.96, terms
        if anchor and neu:
            return True, "neutral", 0.90, terms

    if aspect in {"FOOD#PRICES", "DRINKS#PRICES", "RESTAURANT#PRICES"}:
        if aspect == "FOOD#PRICES":
            anchor = _contains_any(clause, FOOD_ANCHORS | FOOD_PRICE_TERMS | {"giá"})
        elif aspect == "DRINKS#PRICES":
            anchor = _contains_any(clause, DRINK_ANCHORS) and ("giá" in clause or _contains_any(clause, PRICE_POS | PRICE_NEG))
        else:
            anchor = _contains_any(clause, RESTAURANT_PRICE_TERMS)
        neg = _contains_any(clause, PRICE_NEG)
        pos = _contains_any(clause, PRICE_POS)
        terms = collect(FOOD_ANCHORS | DRINK_ANCHORS | RESTAURANT_PRICE_TERMS | {"giá"}, PRICE_NEG if neg else PRICE_POS)
        if anchor and neg:
            return True, "negative", 0.93, terms
        if anchor and pos:
            return True, "positive", 0.93, terms

    if aspect == "LOCATION#GENERAL":
        anchor = _contains_any(clause, LOCATION_ANCHORS)
        neg = _contains_any(clause, LOCATION_NEG)
        pos = _contains_any(clause, LOCATION_POS)
        terms = collect(LOCATION_ANCHORS, LOCATION_NEG if neg else LOCATION_POS)
        if anchor and neg:
            return True, "negative", 0.94, terms
        if anchor and pos:
            return True, "positive", 0.94, terms

    if aspect == "DRINKS#QUALITY":
        anchor = _contains_any(clause, DRINK_ANCHORS)
        neg = _contains_any(clause, FOOD_QUALITY_NEG)
        pos = _contains_any(clause, FOOD_QUALITY_POS)
        terms = collect(DRINK_ANCHORS, FOOD_QUALITY_NEG if neg else FOOD_QUALITY_POS)
        if anchor and neg:
            return True, "negative", 0.92, terms
        if anchor and pos:
            return True, "positive", 0.92, terms

    return False, None, 0.0, terms


def _heuristic_polarity(aspect: str, text: str) -> Optional[str]:
    """Polarity dựa trên evidence cùng clause; chỉ dùng khi rõ ràng."""
    ok, label, strength, _ = _explicit_evidence_polarity(aspect, text)
    if ok and label:
        return label
    return None



def detect_sarcasm(text: str) -> dict:
    low = (text or "").casefold()
    signals = []
    if re.search(r"(ngon|tốt|hay|tuyệt|xuất sắc).{0,40}(nhưng|mà|tuy nhiên).{0,80}(tệ|dở|chậm|lâu|đắt|bẩn|không)", low):
        signals.append("positive_negative_contrast")
    if re.search(r"(quá ngon|tuyệt vời|xuất sắc).{0,40}(luôn|ghê|sợ|chịu|ha|hả)", low):
        signals.append("sarcastic_positive_phrase")
    if "!!!" in low or "???" in low:
        signals.append("strong_punctuation")
    return {
        "sarcasm_detected": bool(signals),
        "signals": signals,
        "note": "Chỉ là heuristic cảnh báo. Muốn phát hiện sarcasm chắc hơn cần dataset có nhãn sarcasm.",
    }


def _decode(
    presence_probs,
    polarity_probs,
    text: str,
    *,
    precision_mode: bool = True,
    strict_filter: Optional[bool] = None,
) -> Tuple[Dict[str, dict], List[dict]]:
    aspect_names = get_aspect_names()
    polarity_names = get_polarity_names()
    thresholds = _thresholds or [float(getattr(config, "DEFAULT_ABSA_THRESHOLD", 0.62))] * len(aspect_names)

    if strict_filter is None:
        strict_filter = bool(getattr(config, "INFERENCE_STRICT_FILTER", True))

    no_evidence_margin = float(getattr(config, "NO_EVIDENCE_MARGIN", 0.18))
    min_threshold = float(getattr(config, "INFERENCE_MIN_THRESHOLD", 0.55))
    rescue_min_prob = float(getattr(config, "EVIDENCE_RESCUE_MIN_PROB", 0.22))
    rescue_strong_prob = float(getattr(config, "EVIDENCE_RESCUE_STRONG_PROB", 0.92))
    force_explicit_rescue = bool(getattr(config, "FORCE_EXPLICIT_EVIDENCE_RESCUE", True))

    aspects: Dict[str, dict] = {}
    mentioned: List[dict] = []

    for i, aspect in enumerate(aspect_names):
        p_mention = float(presence_probs[i])
        base_threshold = max(float(thresholds[i]), min_threshold if precision_mode else 0.0)

        hits = _keyword_hits(text, aspect)
        has_evidence = _has_contextual_evidence(text, aspect)
        explicit_ok, explicit_label, explicit_strength, explicit_terms = _explicit_evidence_polarity(aspect, text)

        effective_threshold = base_threshold
        evidence_status = "supported" if has_evidence else "no_contextual_evidence"

        # Không có evidence ngữ cảnh thì threshold tăng cao để chặn false positive.
        if strict_filter and not has_evidence and not explicit_ok:
            effective_threshold = min(0.97, base_threshold + no_evidence_margin)

        is_mentioned = p_mention >= effective_threshold
        aspect_source = "model_threshold"

        # Rescue chỉ cho evidence rất rõ ràng cùng clause, ví dụ:
        # "đồ_ăn ngon_mê_mẩn", "không_gian đẹp", "phục_vụ chậm".
        if explicit_ok and explicit_label:
            rescue_allowed = (
                p_mention >= rescue_min_prob
                or explicit_strength >= rescue_strong_prob
                or force_explicit_rescue
            )
            if rescue_allowed and not is_mentioned:
                is_mentioned = True
                aspect_source = "explicit_evidence_rescue"
                evidence_status = "explicit_evidence_rescue"
                effective_threshold = min(effective_threshold, max(0.01, p_mention))
            elif is_mentioned:
                evidence_status = "explicit_evidence_supported"

        pol_idx = int(polarity_probs[i].argmax())
        label_id = pol_idx + 1
        label = polarity_names[label_id]
        pol_conf = float(polarity_probs[i][pol_idx])
        polarity_source = "model"

        # Polarity override khi evidence cùng clause rõ hơn model.
        if is_mentioned and explicit_ok and explicit_label in polarity_names:
            if (
                explicit_strength >= float(getattr(config, "EXPLICIT_POLARITY_OVERRIDE_STRENGTH", 0.90))
                or pol_conf < float(getattr(config, "HEURISTIC_OVERRIDE_MAX_MODEL_CONF", 0.96))
                or explicit_label == label
            ):
                label = explicit_label
                label_id = polarity_names.index(explicit_label)
                polarity_source = "explicit_evidence_override"

        final_id = label_id if is_mentioned else 0
        final_label = label if is_mentioned else "none"

        # Gộp keyword_hits thường + evidence terms để debug rõ lý do model giữ aspect.
        debug_hits = sorted(set(list(hits) + [str(x) for x in explicit_terms]))

        aspects[aspect] = {
            "label": final_label,
            "label_id": int(final_id),
            "presence_prob": p_mention,
            "threshold": float(base_threshold),
            "effective_threshold": float(effective_threshold),
            "evidence_status": evidence_status,
            "aspect_source": aspect_source,
            "explicit_evidence": bool(explicit_ok),
            "explicit_evidence_strength": float(explicit_strength),
            "polarity_probs": {
                polarity_names[j + 1]: float(polarity_probs[i][j]) for j in range(3)
            },
            "keyword_hits": debug_hits,
            "polarity_source": polarity_source,
        }

        if is_mentioned:
            mentioned.append({
                "aspect": aspect,
                "polarity": label,
                "confidence": p_mention,
                "polarity_confidence": pol_conf,
                "effective_threshold": float(effective_threshold),
                "evidence_status": evidence_status,
                "aspect_source": aspect_source,
                "polarity_source": polarity_source,
                "keyword_hits": debug_hits,
            })

    # Ưu tiên hiển thị aspect có evidence rõ, sau đó confidence.
    mentioned.sort(key=lambda x: (x["evidence_status"].startswith("explicit"), x["confidence"]), reverse=True)
    return aspects, mentioned


def predict(
    text: str,
    checkpoint_path: Optional[str] = None,
    max_len: Optional[int] = None,
    strict_filter: Optional[bool] = None,
    precision_mode: bool = True,
) -> dict:
    """
    Dự đoán ABSA cho 1 review.

    Luồng:
        raw text -> PreprocessingPipeline.process -> repair -> PhoBERT tokenizer -> ABSA model
    """
    _load_if_needed(checkpoint_path)
    max_len = max_len or int(getattr(config, "MAX_LEN", 192))

    processed = preprocess_for_model(text)

    encoding = _tokenizer(
        processed,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    encoding = {k: v.to(_device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = _model(**encoding)
        presence_probs = torch.sigmoid(outputs.presence_logits).squeeze(0).cpu().numpy()
        polarity_probs = F.softmax(outputs.polarity_logits, dim=-1).squeeze(0).cpu().numpy()

    aspects, mentioned = _decode(
        presence_probs,
        polarity_probs,
        processed,
        precision_mode=precision_mode,
        strict_filter=strict_filter,
    )

    return {
        "text": text,
        "preprocessed_text": processed,
        "aspects": aspects,
        "mentioned_aspects": mentioned,
        "sarcasm_warning": detect_sarcasm(text),
    }


def predict_batch(
    texts: List[str],
    checkpoint_path: Optional[str] = None,
    max_len: Optional[int] = None,
    batch_size: int = 32,
    strict_filter: Optional[bool] = None,
    precision_mode: bool = True,
    show_progress: bool = False,
    progress_desc: str = "Đang dự đoán",
) -> List[dict]:
    """
    Dự đoán ABSA theo batch.

    Args:
        texts: Danh sách review.
        checkpoint_path: Model checkpoint; mặc định dùng best_model.
        max_len: Độ dài token tối đa.
        batch_size: Số review trong mỗi batch model.
        strict_filter: Bật/tắt contextual evidence filtering.
        precision_mode: Bật/tắt precision mode.
        show_progress: Hiện progress bar. Mặc định False để tránh progress lồng
            khi hàm được gọi từ tests/test.py.
        progress_desc: Nhãn hiển thị trên progress bar.

    Returns:
        Danh sách kết quả theo đúng thứ tự input.
    """
    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    _load_if_needed(checkpoint_path)
    max_len = max_len or int(getattr(config, "MAX_LEN", 192))
    results: List[dict] = []
    progress = _create_progress(len(texts), progress_desc) if show_progress else None

    try:
        for start in range(0, len(texts), batch_size):
            raw_batch = texts[start:start + batch_size]
            processed_batch = [preprocess_for_model(text) for text in raw_batch]

            encoding = _tokenizer(
                processed_batch,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            encoding = {key: value.to(_device) for key, value in encoding.items()}

            with torch.inference_mode():
                outputs = _model(**encoding)
                presence_probs = torch.sigmoid(
                    outputs.presence_logits
                ).cpu().numpy()
                polarity_probs = F.softmax(
                    outputs.polarity_logits,
                    dim=-1,
                ).cpu().numpy()

            for raw, processed, pp, polp in zip(
                raw_batch,
                processed_batch,
                presence_probs,
                polarity_probs,
            ):
                aspects, mentioned = _decode(
                    pp,
                    polp,
                    processed,
                    precision_mode=precision_mode,
                    strict_filter=strict_filter,
                )
                results.append(
                    {
                        "text": raw,
                        "preprocessed_text": processed,
                        "aspects": aspects,
                        "mentioned_aspects": mentioned,
                        "sarcasm_warning": detect_sarcasm(raw),
                    }
                )

            if progress is not None:
                progress.update(len(raw_batch))
    finally:
        if progress is not None:
            progress.close()

    return results



# ============================================================
# Pretty output + CLI
# ============================================================

def _final_polarity_confidence(result: dict, aspect: str, fallback: Any) -> float:
    detail = (result.get("aspects", {}) or {}).get(aspect, {}) or {}
    label = str(detail.get("label", "none"))
    probs = detail.get("polarity_probs", {}) or {}
    try:
        return float(probs.get(label, fallback))
    except (TypeError, ValueError):
        return float("nan")


def print_prediction(result: dict, *, title: str = "KẾT QUẢ PHOBERT ABSA") -> None:
    """In kết quả một review dưới dạng bảng dễ đọc."""
    _Console.banner(title)

    _Console.section("REVIEW GỐC")
    print("  " + str(result.get("text", "")).replace("\n", "\n  "))

    _Console.section("SAU TIỀN XỬ LÝ")
    print("  " + str(result.get("preprocessed_text", "")))

    mentioned = result.get("mentioned_aspects", []) or []
    _Console.section("ASPECT ĐƯỢC PHÁT HIỆN")

    rows: List[List[Any]] = []
    decision_values: List[float] = []
    for item in mentioned:
        aspect = str(item.get("aspect", ""))
        presence = float(item.get("confidence", float("nan")))
        polarity = _final_polarity_confidence(
            result,
            aspect,
            item.get("polarity_confidence"),
        )
        decision = min(presence, polarity) if math.isfinite(polarity) else presence
        decision_values.append(decision)
        rows.append(
            [
                aspect,
                item.get("polarity", ""),
                _fmt_number(presence, 4),
                _fmt_number(polarity, 4),
                _fmt_number(decision, 4),
                _fmt_number(item.get("effective_threshold"), 3),
                item.get("aspect_source", ""),
                item.get("polarity_source", ""),
            ]
        )

    _print_table(
        [
            "Aspect",
            "Polarity",
            "Presence",
            "Polarity conf.",
            "Decision",
            "Threshold",
            "Aspect source",
            "Polarity source",
        ],
        rows,
        max_width=26,
    )

    _Console.section("TÓM TẮT")
    _print_key_values(
        [
            ("Số aspect được phát hiện", len(mentioned)),
            (
                "Decision confidence trung bình",
                _fmt_number(np_mean(decision_values), 4),
            ),
            (
                "Decision confidence thấp nhất",
                _fmt_number(min(decision_values) if decision_values else None, 4),
            ),
        ]
    )

    sarcasm = result.get("sarcasm_warning", {}) or {}
    if sarcasm.get("sarcasm_detected"):
        _Console.warning(
            "Có tín hiệu sarcasm/đối lập: "
            + ", ".join(str(value) for value in sarcasm.get("signals", []))
        )
    else:
        _Console.success("Không phát hiện tín hiệu sarcasm theo heuristic.")


def np_mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def summarize_batch(results: Sequence[dict], elapsed_seconds: float) -> dict:
    aspect_counts: Dict[str, int] = {}
    polarity_counts: Dict[str, int] = {}
    presence_values: List[float] = []
    polarity_values: List[float] = []
    decision_values: List[float] = []
    no_aspect = 0

    for result in results:
        mentioned = result.get("mentioned_aspects", []) or []
        if not mentioned:
            no_aspect += 1

        for item in mentioned:
            aspect = str(item.get("aspect", ""))
            label = str(item.get("polarity", ""))
            aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            polarity_counts[label] = polarity_counts.get(label, 0) + 1

            presence = float(item.get("confidence", float("nan")))
            polarity = _final_polarity_confidence(
                result,
                aspect,
                item.get("polarity_confidence"),
            )
            if math.isfinite(presence):
                presence_values.append(presence)
            if math.isfinite(polarity):
                polarity_values.append(polarity)
            if math.isfinite(presence) and math.isfinite(polarity):
                decision_values.append(min(presence, polarity))

    total = len(results)
    total_aspects = sum(aspect_counts.values())
    return {
        "total_reviews": total,
        "total_aspects": total_aspects,
        "no_aspect_reviews": no_aspect,
        "aspects_per_review": total_aspects / total if total else 0.0,
        "presence_mean": np_mean(presence_values),
        "polarity_mean": np_mean(polarity_values),
        "decision_mean": np_mean(decision_values),
        "elapsed_seconds": elapsed_seconds,
        "reviews_per_second": total / elapsed_seconds if elapsed_seconds > 0 else None,
        "aspect_counts": aspect_counts,
        "polarity_counts": polarity_counts,
    }


def print_batch_summary(results: Sequence[dict], elapsed_seconds: float) -> dict:
    summary = summarize_batch(results, elapsed_seconds)
    _Console.banner("TỔNG KẾT BATCH INFERENCE")

    _Console.section("THỐNG KÊ CHÍNH")
    _print_key_values(
        [
            ("Tổng review", f"{summary['total_reviews']:,}"),
            ("Tổng aspect phát hiện", f"{summary['total_aspects']:,}"),
            ("Aspect trung bình/review", _fmt_number(summary["aspects_per_review"], 3)),
            ("Review không có aspect", f"{summary['no_aspect_reviews']:,}"),
            ("Presence confidence TB", _fmt_number(summary["presence_mean"], 4)),
            ("Polarity confidence TB", _fmt_number(summary["polarity_mean"], 4)),
            ("Decision confidence TB", _fmt_number(summary["decision_mean"], 4)),
            ("Thời gian", f"{_fmt_number(summary['elapsed_seconds'], 2)} giây"),
            ("Tốc độ", f"{_fmt_number(summary['reviews_per_second'], 2)} review/giây"),
        ]
    )

    _Console.section("TẦN SUẤT ASPECT")
    aspect_rows = [
        [aspect, count]
        for aspect, count in sorted(
            summary["aspect_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    _print_table(["Aspect", "Số lần"], aspect_rows)

    _Console.section("PHÂN BỐ POLARITY")
    polarity_rows = [
        [label, count]
        for label, count in sorted(
            summary["polarity_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    _print_table(["Polarity", "Số lần"], polarity_rows)
    return summary


def _read_texts_from_file(path: Path, text_column: Optional[str]) -> Tuple[List[str], Optional[Any]]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        texts = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return texts, None

    if suffix not in {".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}:
        raise ValueError("Chỉ hỗ trợ CSV, Excel, Parquet, JSON/JSONL hoặc TXT.")

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Cần pandas để đọc dataset batch.") from exc

    if suffix == ".csv":
        dataframe = pd.read_csv(
            path,
            encoding="utf-8-sig",
            quotechar='"',
            keep_default_na=False,
        )
    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    elif suffix == ".parquet":
        dataframe = pd.read_parquet(path)
    elif suffix == ".jsonl":
        dataframe = pd.read_json(path, lines=True)
    else:
        dataframe = pd.read_json(path)

    candidates = ["Comment", "comment", "Review", "review", "text", "Text", "content"]
    column = text_column
    if column is None:
        column = next((name for name in candidates if name in dataframe.columns), None)
    if column is None or column not in dataframe.columns:
        raise ValueError(
            f"Không xác định được cột review. Các cột: {list(dataframe.columns)}. "
            "Hãy dùng --text-column."
        )

    texts = dataframe[column].fillna("").astype(str).tolist()
    return texts, dataframe


def _save_batch_results(
    results: Sequence[dict],
    output_path: Path,
    dataframe: Optional[Any] = None,
) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Cần pandas để xuất kết quả batch.") from exc

    rows: List[dict] = []
    for index, result in enumerate(results):
        mentioned = result.get("mentioned_aspects", []) or []
        presence_values = [float(item.get("confidence", float("nan"))) for item in mentioned]
        polarity_values = [
            _final_polarity_confidence(result, str(item.get("aspect", "")), item.get("polarity_confidence"))
            for item in mentioned
        ]
        decisions = [
            min(presence, polarity)
            for presence, polarity in zip(presence_values, polarity_values)
            if math.isfinite(presence) and math.isfinite(polarity)
        ]
        rows.append(
            {
                "raw_text": result.get("text", ""),
                "preprocessed_text": result.get("preprocessed_text", ""),
                "predicted_aspect_count": len(mentioned),
                "predicted_aspects": " | ".join(
                    f"{item.get('aspect')}:{item.get('polarity')}" for item in mentioned
                ),
                "mean_presence_confidence": np_mean(presence_values),
                "mean_polarity_confidence": np_mean(polarity_values),
                "mean_decision_confidence": np_mean(decisions),
                "mentioned_aspects_json": json.dumps(mentioned, ensure_ascii=False),
                "sarcasm_detected": bool((result.get("sarcasm_warning", {}) or {}).get("sarcasm_detected", False)),
            }
        )

    output_df = pd.DataFrame(rows)
    if dataframe is not None and len(dataframe) == len(output_df):
        output_df = pd.concat(
            [dataframe.reset_index(drop=True), output_df],
            axis=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    _Console.success(f"Đã lưu kết quả: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhoBERT ABSA inference với progress bar và output đẹp.")
    parser.add_argument("--text", type=str, default=None, help="Một review cần dự đoán.")
    parser.add_argument("--input", type=Path, default=None, help="Dataset CSV/XLSX/Parquet/JSON/TXT.")
    parser.add_argument("--text-column", type=str, default=None, help="Tên cột review trong dataset.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "predictions" / "inference_predictions.csv")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-samples", type=int, default=3)
    return parser.parse_args()


# Test ở đây

def _main() -> None:
    args = _parse_args()
    demo = (
        "Quán này ngon quá "
        
    )

    if args.input is None:
        text = args.text or demo
        started = time.perf_counter()
        result = predict_batch(
            [text],
            checkpoint_path=args.checkpoint,
            max_len=args.max_len,
            batch_size=1,
            strict_filter=args.strict_filter,
            precision_mode=args.precision_mode,
            show_progress=True,
        )[0]
        print_prediction(result)
        print(f"\n  Thời gian xử lý: {_fmt_number(time.perf_counter() - started, 3)} giây")
        return

    input_path = args.input
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {input_path}")

    texts, dataframe = _read_texts_from_file(input_path, args.text_column)
    if args.limit is not None:
        texts = texts[: max(0, args.limit)]
        if dataframe is not None:
            dataframe = dataframe.head(len(texts)).copy()

    _Console.banner("BATCH PHOBERT ABSA INFERENCE")
    _print_key_values(
        [
            ("Dataset", input_path),
            ("Số review", f"{len(texts):,}"),
            ("Batch size", args.batch_size),
            ("Strict filter", args.strict_filter),
            ("Precision mode", args.precision_mode),
        ]
    )

    started = time.perf_counter()
    results = predict_batch(
        texts,
        checkpoint_path=args.checkpoint,
        max_len=args.max_len,
        batch_size=args.batch_size,
        strict_filter=args.strict_filter,
        precision_mode=args.precision_mode,
        show_progress=True,
    )
    elapsed = time.perf_counter() - started
    print_batch_summary(results, elapsed)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    _save_batch_results(results, output_path, dataframe)

    if args.show_samples > 0:
        for index, result in enumerate(results[: args.show_samples], start=1):
            print_prediction(result, title=f"DỰ ĐOÁN MẪU {index}")


if __name__ == "__main__":
    try:
        _main()
    except KeyboardInterrupt:
        print()
        _Console.warning("Đã dừng chương trình.")
        raise SystemExit(130)
    except Exception as exc:
        print()
        _Console.error(f"{type(exc).__name__}: {exc}")
        raise
