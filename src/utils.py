# src/utils.py

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None


# =========================================================
# FILE IO
# =========================================================

def load_json(path: str | Path, default: Any = None) -> Any:
    """
    Đọc file json.
    """
    path = Path(path)

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 4,
    ensure_ascii: bool = False,
) -> None:
    """
    Ghi file json.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )


# =========================================================
# TEXT
# =========================================================

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """
    Chuẩn hóa khoảng trắng.
    """
    if not text:
        return ""

    return _WHITESPACE_RE.sub(" ", text).strip()


def safe_lower(text: str) -> str:
    """
    Lower unicode-safe.
    """
    if not text:
        return ""

    return text.casefold()


def is_punctuation(token: str) -> bool:
    """
    Kiểm tra token có phải dấu câu không.
    """
    if not token:
        return True

    return all(not c.isalnum() for c in token)


# =========================================================
# RANDOM / REPRODUCIBILITY
# =========================================================

def set_seed(seed: int = 42) -> None:
    """
    Cố định random seed.
    """
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except Exception:
        pass


# =========================================================
# PATH
# =========================================================

def ensure_dir(path: str | Path) -> Path:
    """
    Tạo thư mục nếu chưa tồn tại.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Tự động tìm thư mục gốc project.
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "src").exists():
            return parent

    return current.parent


# =========================================================
# STATS
# =========================================================

def print_separator(
    title: Optional[str] = None,
    length: int = 50,
) -> None:
    """
    In separator đẹp.
    """
    if title:
        print("\n" + "=" * length)
        print(title)
        print("=" * length)
    else:
        print("=" * length)


def percentage(
    value: int,
    total: int,
    digits: int = 2,
) -> float:
    """
    Tính phần trăm.
    """
    if total == 0:
        return 0.0

    return round(value * 100 / total, digits)


# =========================================================
# DEBUG
# =========================================================

def pretty_dict(
    data: Dict[str, Any],
    indent: int = 2,
) -> None:
    """
    Print dict đẹp.
    """
    print(
        json.dumps(
            data,
            indent=indent,
            ensure_ascii=False,
        )
    )