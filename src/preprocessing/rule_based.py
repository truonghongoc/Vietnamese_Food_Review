# src/preprocessing/rule_based.py

from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG

rule_cfg = CONFIG.build_rule_based_config()
dict_cfg = CONFIG.build_dictionary_config()
tok_cfg = CONFIG.build_tokenizer_config()

import html
import re
import unicodedata
from dataclasses import dataclass

__all__ = ["RuleBasedConfig", "RuleBasedNormalizer"]


@dataclass(slots=True)
class RuleBasedConfig:
    # Số ký tự lặp tối đa được giữ lại
    # EX: "ngonnnn" -> "ngonn"
    max_repeated_chars: int = 2

    # Bật/Tắt chuẩn hóa Unicode
    normalize_unicode: bool = True

    # Bật/Tắt chuẩn hóa HTML entities
    normalize_html_entities: bool = True

    # Bật/Tắt xóa các ký tự vô hình
    normalize_invisible_chars: bool = True

    # Bật/Tắt xóa URL / email / mention
    remove_urls: bool = True
    remove_emails: bool = True
    remove_mentions: bool = True

    # Bật/Tắt chuẩn hóa ký tự lặp
    normalize_repeated_chars: bool = True

    # Bật/Tắt chuẩn hóa dấu câu cảm xúc
    normalize_emotional_punctuation: bool = True

    # Bật/Tắt chuẩn hóa dấu ba chấm
    normalize_ellipsis: bool = True

    # Bật/Tắt chuẩn hóa khoảng trắng
    normalize_whitespace: bool = True


class RuleBasedNormalizer:
    """
    Xử lý tiền xử lý dữ liệu bằng rule-based.
    Không phụ thuộc vào dictionary.
    """

    _URL_RE = re.compile(
        r"""(?xi)
        \b(
            https?://\S+
            |
            www\.\S+
        )
        """
    )

    _EMAIL_RE = re.compile(
        r"""(?xi)
        \b[\w.%+-]+@[\w.-]+\.\w{2,}\b
        """
    )

    _MENTION_RE = re.compile(r"(?<!\w)@\w+")

    _WHITESPACE_RE = re.compile(r"\s+")

    # Chỉ xử lý ký tự chữ lặp, KHÔNG xử lý dấu câu ở đây
    # Ví dụ:
    #   ngonnnn -> ngonn
    #   depiiii -> depii
    _REPEATED_CHAR_RE = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)

    # Invisible / zero-width chars
    _INVISIBLE_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")

    # Emotional punctuation
    _EXCLAMATION_RE = re.compile(r"!{4,}")
    _QUESTION_RE = re.compile(r"\?{4,}")
    _DUPLICATE_PUNCT_RE = re.compile(r"([!?])\1{2,}")

    # Ellipsis
    _ELLIPSIS_RE = re.compile(r"\.{4,}")

    # Quote normalization (optional, nhẹ nhàng)
    _QUOTE_TRANSLATION = str.maketrans({
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    })

    def __init__(self, config: RuleBasedConfig | None = None) -> None:
        self.config = config or RuleBasedConfig()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Chuẩn hóa về NFC để đồng nhất dữ liệu Unicode tiếng Việt.
        """
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def normalize_html_entities(text: str) -> str:
        """
        Giải mã HTML entities.
        EX: &amp; -> &
        """
        if not text:
            return ""
        return html.unescape(text)

    def remove_invisible_chars(self, text: str) -> str:
        """
        Xóa ký tự vô hình / zero-width chars.
        EX: \\u200b, \\u200d, \\u2060, \\ufeff
        """
        if not text:
            return ""
        return self._INVISIBLE_RE.sub("", text)

    def remove_urls(self, text: str) -> str:
        """
        Xóa URL khỏi văn bản.
        """
        if not text:
            return ""
        return self._URL_RE.sub(" ", text)

    def remove_emails(self, text: str) -> str:
        """
        Xóa email khỏi văn bản.
        """
        if not text:
            return ""
        return self._EMAIL_RE.sub(" ", text)

    def remove_mentions(self, text: str) -> str:
        """
        Xóa mention dạng @username.
        """
        if not text:
            return ""
        return self._MENTION_RE.sub(" ", text)

    def normalize_quotes(self, text: str) -> str:
        """
        Chuẩn hóa các kiểu dấu ngoặc kép / nháy.
        """
        if not text:
            return ""
        return text.translate(self._QUOTE_TRANSLATION)

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """
        Gom nhiều khoảng trắng liên tiếp thành 1 khoảng trắng.
        """
        if not text:
            return ""
        return cls._WHITESPACE_RE.sub(" ", text).strip()

    def normalize_repeated_chars(self, text: str, max_repeat: int | None = None) -> str:
        """
        Rút gọn ký tự chữ lặp quá nhiều.
        EX:
            - ngonnnnn -> ngonn
            - depiiiiii -> depii
        """
        if not text:
            return ""

        limit = self.config.max_repeated_chars if max_repeat is None else max_repeat
        if limit < 1:
            return text

        def _replace(match: re.Match[str]) -> str:
            ch = match.group(1)
            return ch * limit

        return self._REPEATED_CHAR_RE.sub(_replace, text)

    def normalize_emotional_punctuation(self, text: str) -> str:
        """
        Chuẩn hóa dấu câu cảm xúc:
            - !!!!!!!! -> !!!
            - ???????? -> ???
            - !?!?!?!?! -> ?!
        """
        if not text:
            return ""

        text = self._DUPLICATE_PUNCT_RE.sub(lambda m: m.group(1) * 2, text)
        text = self._EXCLAMATION_RE.sub(" !!! ", text)
        text = self._QUESTION_RE.sub(" ??? ", text)

        return text

    def normalize_ellipsis(self, text: str) -> str:
        """
        Chuẩn hóa dấu ba chấm dài:
            - .......... -> ...
        """
        if not text:
            return ""

        return self._ELLIPSIS_RE.sub(" ... ", text)

    def clean(self, text: str) -> str:
        """
        Pipeline rule-based cơ bản:
            Unicode -> HTML -> Invisible chars -> URL -> Email -> Mention
            -> Repeated chars -> Emotional punctuation -> Ellipsis(không có)
            -> Quotes(không có) -> Whitespace
        """
        if not text:
            return ""

        if self.config.normalize_unicode:
            text = self.normalize_unicode(text)

        if self.config.normalize_html_entities:
            text = self.normalize_html_entities(text)

        if self.config.normalize_invisible_chars:
            text = self.remove_invisible_chars(text)

        if self.config.remove_urls:
            text = self.remove_urls(text)

        if self.config.remove_emails:
            text = self.remove_emails(text)

        if self.config.remove_mentions:
            text = self.remove_mentions(text)

        if self.config.normalize_repeated_chars:
            text = self.normalize_repeated_chars(text)

        if self.config.normalize_emotional_punctuation:
            text = self.normalize_emotional_punctuation(text)

        if self.config.normalize_ellipsis:
            text = self.normalize_ellipsis(text)

        text = self.normalize_quotes(text)

        if self.config.normalize_whitespace:
            text = self.normalize_whitespace(text)

        return text


# TESTING

if __name__ == "__main__":
    normalizer = RuleBasedNormalizer()

    text = """
    Quán này ngon lắm!!!!  Xem thêm tại https://example.com
    Email: abc@gmail.com, mention: @user123
    Ngonnnnnn quá trời luôn.......... 
    Invisible: Quán\u200b này\u200d rất\u2060 ngon\ufeff!!!
    """

    print(normalizer.clean(text))