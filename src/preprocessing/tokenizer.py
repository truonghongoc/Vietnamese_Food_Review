# src/preprocessing/tokenizer.py

from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from src.config import CONFIG

rule_cfg = CONFIG.build_rule_based_config()
dict_cfg = CONFIG.build_dictionary_config()
tok_cfg = CONFIG.build_tokenizer_config()

try:
    from underthesea import sent_tokenize, word_tokenize
    HAS_UNDERTHESEA = True
except Exception:
    sent_tokenize = None
    word_tokenize = None
    HAS_UNDERTHESEA = False

__all__ = ["TokenizerConfig", "VietnameseTokenizer"]

# =========================================================
# REGEX / GLOBAL HELPERS
# =========================================================

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_ONLY_RE = re.compile(r"^[^\wÀ-ỹà-ỹ]+$", re.UNICODE)
_TOKENIZE_FALLBACK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_NUMBER_LIKE_RE = re.compile(r"^[\d.,]+[kKmMđĐ]?$", re.UNICODE)

_SENT_SPLIT_FALLBACK_RE = re.compile(
    r"(?<=[.!?…])\s+(?=[^\W\d_])",
    re.UNICODE,
)

# Dấu hiệu cho thấy text đã được tokenize kiểu underthesea format="text"
_PRETOKENIZED_HINT_RE = re.compile(
    r"\b\w+_\w+\b|(?:\s[!?,.;:]\s)",
    re.UNICODE,
)

_DEFAULT_PHRASES: Dict[str, str] = {
    "bò bít tết": "bò_bít_tết",
    "trà táo": "trà_táo",
    "trà sữa": "trà_sữa",
    "bánh mì": "bánh_mì",
    "cơm tấm": "cơm_tấm",
    "bánh tráng trộn": "bánh_tráng_trộn",
    "gà rán": "gà_rán",
    "món ăn": "món_ăn",
    "đồ ăn": "đồ_ăn",
    "thức ăn": "thức_ăn",
    "nhà hàng": "nhà_hàng",
    "quán ăn": "quán_ăn",
    "phục vụ": "phục_vụ",
    "nhân viên": "nhân_viên",
    "không gian": "không_gian",
    "giá cả": "giá_cả",
    "đơn hàng": "đơn_hàng",
    "giao hàng": "giao_hàng",
    "hóa đơn": "hóa_đơn",
    "cà phê": "cà_phê",
    "mì xào": "mì_xào",
    "bún bò": "bún_bò",
    "bún chả": "bún_chả",
    "phở bò": "phở_bò",
    "phở gà": "phở_gà",
}

def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()

def _normalize_underthesea_token(token: str) -> str:
    """
    underthesea.word_tokenize(..., format="list") có thể trả về một từ ghép
    (VD: "phục vụ") dưới dạng MỘT token nhưng vẫn còn khoảng trắng bên trong,
    thay vì nối bằng "_" như khi dùng format="text".

    Nếu không chuẩn hoá bước này, token đó sẽ:
    - không khớp được với dictionary food_phrases.json (vì trie so khớp theo
      từng âm tiết tách rời, không so khớp với 1 token có khoảng trắng), và
    - khi join tokens lại bằng " ", nó sẽ trông giống 2 từ rời rạc
      ("phục vụ") thay vì 1 cụm từ ghép ("phục_vụ").

    Do đó ta thay mọi khoảng trắng bên trong token bằng "_" ngay sau khi
    underthesea tokenize, trước khi đưa vào bước merge phrase theo dictionary.
    """
    token = token.strip()
    if not token:
        return token
    return _WHITESPACE_RE.sub("_", token)

@lru_cache(maxsize=100_000)
def _cached_sentence_split(text: str) -> tuple[str, ...]:
    text = _normalize_whitespace(text)
    if not text:
        return tuple()

    if HAS_UNDERTHESEA and sent_tokenize is not None:
        try:
            sentences = sent_tokenize(text)
            return tuple(s.strip() for s in sentences if s and s.strip())
        except Exception:
            pass

    parts = _SENT_SPLIT_FALLBACK_RE.split(text)
    return tuple(p.strip() for p in parts if p.strip())

@lru_cache(maxsize=200_000)
def _cached_word_tokenize(sentence: str, keep_punctuation: bool) -> tuple[str, ...]:
    sentence = _normalize_whitespace(sentence)
    if not sentence:
        return tuple()

    if HAS_UNDERTHESEA and word_tokenize is not None:
        try:
            tokens = word_tokenize(sentence, format="list")
            return tuple(_normalize_underthesea_token(t) for t in tokens if t and t.strip())
        except Exception:
            pass

    if keep_punctuation:
        tokens = _TOKENIZE_FALLBACK_RE.findall(sentence)
        return tuple(t for t in tokens if t and t.strip())

    return tuple(sentence.split())

# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class TokenizerConfig:
    use_sentence_tokenization: bool = True
    use_word_tokenization: bool = True

    # Merge phrase domain sau word segmentation
    use_phrase_merge: bool = True

    # Join negation scope
    join_negation: bool = False

    keep_punctuation: bool = True
    max_negation_scope: int = 4
    negation_join_separator: str = "_"

    # ranh giới giữa các câu khi xuất PhoBERT text.
    # LƯU Ý: đây phải là dấu cách " ", KHÔNG phải "_" — "_" chỉ dùng để nối
    # các từ ghép bên TRONG một cụm từ (VD: "phục_vụ"), không dùng để nối
    # giữa hai câu khác nhau. Nếu để "_" ở đây, câu 1 kết thúc bằng "!" và
    # câu 2 bắt đầu bằng "Món" sẽ bị dính thành "!_Món" (sai).
    phobert_sentence_separator: str = " "

    # phrase dictionary
    use_default_phrases: bool = True
    phrase_dict_path: Optional[str] = None

    # Nếu True: coi TOÀN BỘ input truyền vào tokenize_sentence/process/... là
    # text đã được tokenize sẵn (kiểu output underthesea format="text", ví dụ
    # do chính bạn gọi word_tokenize(text, format="text") từ bên ngoài rồi mới
    # đưa vào đây) -> sẽ chỉ split theo khoảng trắng, KHÔNG chạy lại underthesea.
    #
    # Mặc định là False: luôn chạy underthesea thật trên input.
    #
    # QUAN TRỌNG: KHÔNG bật flag này cho pipeline bình thường (rule_based ->
    # dictionary_based -> tokenizer), kể cả khi dictionary_based đã tự nối
    # sẵn một vài cụm từ domain bằng "_" (VD: "phở_bò"). Nếu bật True trong
    # trường hợp đó, toàn bộ phần còn lại của câu (VD: "nho nhỏ") sẽ KHÔNG
    # được underthesea xử lý nữa mà chỉ tách theo dấu cách -> đây chính là
    # nguyên nhân "underthesea không hoạt động" khi chạy qua pipeline.py.
    treat_input_as_pretokenized: bool = False

    # negation dictionary
    negation_dict_path: Optional[str] = None

    project_root: Optional[str] = None

# =========================================================
# CORE TOKENIZER
# =========================================================

class VietnameseTokenizer:
    """
    Tokenizer cho dự án Review tiếng Việt.

    Pipeline:
    - Tách câu
    - Underthesea word segmentation
    - Phrase merge theo domain
    - Ghép phủ định theo scope
    - Xuất text phù hợp cho PhoBERT
    """

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config or TokenizerConfig()

        self.project_root = self._resolve_project_root(self.config.project_root)
        self.negation_dict_path = self._resolve_negation_path(self.config.negation_dict_path)
        self.phrase_dict_path = self._resolve_phrase_path(self.config.phrase_dict_path)

        self.negation_words: Set[str] = set()
        self.negation_skip_tokens: Set[str] = set()
        self.phrase_map: Dict[str, str] = {}

        self._load_negation_config()
        self._load_phrase_config()

        self._negation_words_lower = {x.casefold() for x in self.negation_words}
        self._negation_skip_lower = {x.casefold() for x in self.negation_skip_tokens}

        self._phrase_trie = self._build_phrase_trie(self.phrase_map)
        self._stats = Counter()

    # =====================================================
    # PATH HANDLING
    # =====================================================

    @staticmethod
    def _resolve_project_root(project_root: Optional[str]) -> Path:
        if project_root:
            return Path(project_root).resolve()
        return Path(__file__).resolve().parents[2]

    def _resolve_negation_path(self, negation_dict_path: Optional[str]) -> Path:
        if negation_dict_path:
            return Path(negation_dict_path).resolve()
        return self.project_root / "dictionaries" / "negation.json"

    def _resolve_phrase_path(self, phrase_dict_path: Optional[str]) -> Path:
        if phrase_dict_path:
            return Path(phrase_dict_path).resolve()
        return self.project_root / "dictionaries" / "food_phrases.json"

    # =====================================================
    # LOAD CONFIGS
    # =====================================================

    def _load_json(self, path: Path) -> Any:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_negation_config(self) -> None:
        default_negation_words = {
            "không", "ko", "k", "kh", "kg", "hok", "hông", "hong",
            "hem", "hẻm", "chưa", "chẳng", "chả", "đừng", "đâu", "chớ",
        }

        default_skip_tokens = {
            "chỉ", "phải", "có", "còn", "là", "những", "mà", "quá", "rất",
            "được", "nữa", "thể", "đến", "bao", "nhiêu", "ít", "nhiều",
            "hề", "hơi", "lắm", "thì", "đã", "đang", "sẽ", "vẫn", "cũng",
        }

        if not self.negation_dict_path.exists():
            self.negation_words = default_negation_words
            self.negation_skip_tokens = default_skip_tokens
            return

        try:
            data = self._load_json(self.negation_dict_path)
            if not isinstance(data, dict):
                raise ValueError("negation config must be a dict")

            neg_words = (
                data.get("negation_words")
                or data.get("words")
                or data.get("negations")
                or []
            )
            skip_tokens = (
                data.get("skip_tokens")
                or data.get("skip")
                or data.get("exceptions")
                or []
            )

            self.negation_words = {
                str(x).strip().lower() for x in neg_words if str(x).strip()
            }
            self.negation_skip_tokens = {
                str(x).strip().lower() for x in skip_tokens if str(x).strip()
            }

            if not self.negation_words:
                self.negation_words = default_negation_words
            if not self.negation_skip_tokens:
                self.negation_skip_tokens = default_skip_tokens

        except Exception:
            self.negation_words = default_negation_words
            self.negation_skip_tokens = default_skip_tokens

    def _load_phrase_config(self) -> None:
        phrase_map: Dict[str, str] = dict(_DEFAULT_PHRASES) if self.config.use_default_phrases else {}

        if self.phrase_dict_path.exists():
            try:
                data = self._load_json(self.phrase_dict_path)

                if isinstance(data, dict):
                    for raw_key, raw_value in data.items():
                        phrase = _normalize_whitespace(str(raw_key)).strip()
                        if not phrase:
                            continue
                        replacement = self._extract_phrase_replacement(raw_value, phrase)
                        phrase_map[phrase] = replacement

                elif isinstance(data, list):
                    for item in data:
                        phrase = _normalize_whitespace(str(item)).strip()
                        if phrase:
                            phrase_map[phrase] = phrase.replace(" ", "_")

            except Exception:
                pass

        self.phrase_map = phrase_map

    @staticmethod
    def _extract_phrase_replacement(value: Any, phrase: str) -> str:
        if isinstance(value, str):
            repl = value.strip()
            return repl.replace(" ", "_") if repl else phrase.replace(" ", "_")

        if isinstance(value, dict):
            for field in ("vi", "text", "normalized", "replacement"):
                repl = value.get(field)
                if repl and str(repl).strip():
                    return str(repl).strip().replace(" ", "_")

        return phrase.replace(" ", "_")

    def _build_phrase_trie(self, phrase_map: Dict[str, str]) -> dict:
        trie: dict = {}
        for phrase, replacement in phrase_map.items():
            tokens = [t.casefold() for t in phrase.split() if t.strip()]
            if not tokens:
                continue

            node = trie
            for tok in tokens:
                node = node.setdefault(tok, {})
            node["__end__"] = replacement
        return trie

    # =====================================================
    # BASIC HELPERS
    # =====================================================

    @staticmethod
    def _is_punctuation_only(token: str) -> bool:
        if not token:
            return True
        return bool(_PUNCT_ONLY_RE.match(token))

    @staticmethod
    def _is_number_like(token: str) -> bool:
        if not token:
            return False
        return bool(_NUMBER_LIKE_RE.fullmatch(token))

    @staticmethod
    def _looks_pre_tokenized_text(text: str) -> bool:
        """
        Heuristic tham khảo: đoán xem text CÓ THỂ đã được tokenize sẵn hay
        chưa (VD: output cũ của word_tokenize(text, format="text")).

        LƯU Ý: hàm này KHÔNG còn được dùng để tự động quyết định bỏ qua
        underthesea nữa (trước đây từng gây lỗi: chỉ cần văn bản chứa 1 token
        dạng "xxx_yyy" - ví dụ do dictionary_based normalize sẵn 1 cụm từ -
        là toàn bộ câu bị coi nhầm là "đã tokenize", khiến các từ khác như
        "nho nhỏ" không được underthesea xử lý nữa).

        Việc bật/tắt chế độ pre-tokenized giờ phải khai báo tường minh qua
        `TokenizerConfig.treat_input_as_pretokenized`. Hàm này chỉ còn giữ
        lại như một tiện ích để tự kiểm tra/debug nếu cần.
        """
        if not text:
            return False
        return bool(_PRETOKENIZED_HINT_RE.search(text))

    @staticmethod
    def _looks_sentence_start_token(token: str) -> bool:
        if not token:
            return False

        token = token.lstrip('\'"“”‘’([{')
        if token.startswith("_") and len(token) > 1:
            token = token[1:]

        if not token:
            return False

        return token[0].isupper() or token[0].isdigit()

    @staticmethod
    def _is_sentence_end_token(token: str) -> bool:
        return token in {".", "!", "?", "…"}

    def _split_pre_tokenized_sentences(self, text: str) -> List[str]:
        """
        Tách câu từ text đã tokenize kiểu underthesea(format="text").
        Ví dụ:
            Quán này ngon lắm ! ! ! ! ! ! Món bò_bít_tết ...
        -> 2 câu
        """
        text = _normalize_whitespace(text)
        if not text:
            return []

        tokens = [t for t in text.split() if t.strip()]
        if not tokens:
            return []

        sentences: List[List[str]] = []
        current: List[str] = []

        for i, tok in enumerate(tokens):
            # Nếu token có dạng _Món do có boundary marker từ output cũ
            if tok.startswith("_") and len(tok) > 1:
                if current:
                    sentences.append(current)
                    current = []
                tok = tok[1:]
                if not tok:
                    continue

            current.append(tok)

            next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
            if next_tok and self._is_sentence_end_token(tok) and self._looks_sentence_start_token(next_tok):
                sentences.append(current)
                current = []

        if current:
            sentences.append(current)

        return [" ".join(s) for s in sentences if s]

    # =====================================================
    # SENTENCE SPLIT
    # =====================================================

    def split_sentences_fallback(self, text: str) -> List[str]:
        if not text:
            return []

        text = _normalize_whitespace(text)
        if not text:
            return []

        parts = _SENT_SPLIT_FALLBACK_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def split_sentences(self, text: str) -> List[str]:
        if not text:
            return []

        text = _normalize_whitespace(text)
        if not text:
            return []

        # Chỉ dùng splitter dành cho text-đã-tokenize khi được khai báo
        # tường minh qua config, KHÔNG tự đoán bằng heuristic (dễ bị
        # trigger nhầm khi dictionary_based đã nối sẵn 1-2 cụm từ bằng "_").
        if self.config.treat_input_as_pretokenized:
            return self._split_pre_tokenized_sentences(text)

        if self.config.use_sentence_tokenization:
            try:
                return list(_cached_sentence_split(text))
            except Exception:
                return self.split_sentences_fallback(text)

        return [text]

    # =====================================================
    # WORD TOKENIZATION
    # =====================================================

    def word_tokenize_sentence(self, sentence: str) -> List[str]:
        if not sentence:
            return []

        sentence = _normalize_whitespace(sentence)
        if not sentence:
            return []

        # Chỉ split theo khoảng trắng khi được khai báo tường minh là input
        # đã pre-tokenized. Mặc định luôn chạy underthesea thật.
        if self.config.treat_input_as_pretokenized:
            return [t for t in sentence.split() if t and t.strip()]

        if self.config.use_word_tokenization:
            try:
                return list(_cached_word_tokenize(sentence, self.config.keep_punctuation))
            except Exception:
                pass

        if self.config.keep_punctuation:
            tokens = _TOKENIZE_FALLBACK_RE.findall(sentence)
            return [t for t in tokens if t and t.strip()]

        return sentence.split()

    # =====================================================
    # PHRASE MERGE
    # =====================================================

    def merge_phrases(self, tokens: Sequence[str]) -> List[str]:
        """
        Merge custom multiword phrases with longest-match-first.
        """
        if not tokens:
            return []

        if not self.config.use_phrase_merge or not self._phrase_trie:
            return [t for t in tokens if t and str(t).strip()]

        result: List[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            current = tokens[i]

            if self._is_punctuation_only(current) or self._is_number_like(current):
                result.append(current)
                i += 1
                continue

            node = self._phrase_trie
            best_replacement: Optional[str] = None
            best_end: Optional[int] = None

            j = i
            while j < n:
                tok = tokens[j]

                if self._is_punctuation_only(tok) or self._is_number_like(tok):
                    break

                key = tok.casefold()
                if key not in node:
                    break

                node = node[key]
                if "__end__" in node:
                    best_replacement = node["__end__"]
                    best_end = j

                j += 1

            if best_replacement is not None and best_end is not None:
                result.append(best_replacement)
                self._stats["phrase_merged"] += 1
                i = best_end + 1
            else:
                result.append(current)
                i += 1

        return result

    # =====================================================
    # NEGATION
    # =====================================================

    def join_negation(self, tokens: Sequence[str]) -> List[str]:
        """
        Ghép phủ định theo scope.
        """
        if not tokens:
            return []

        if not self.config.join_negation or not self.negation_words:
            return [t for t in tokens if t and str(t).strip()]

        result: List[str] = []
        i = 0
        n = len(tokens)
        scope_limit = max(1, int(self.config.max_negation_scope))
        sep = self.config.negation_join_separator or "_"

        while i < n:
            current = tokens[i]
            current_cf = current.casefold()

            # Chỉ xử lý nếu token hiện tại thật sự là phủ định
            if (
                current_cf not in self._negation_words_lower
                or self._is_punctuation_only(current)
                or current.startswith("_")
            ):
                result.append(current)
                i += 1
                continue

            phrase = [current]
            j = i + 1
            skipped = 0
            joined = False

            while j < n and skipped < scope_limit:
                nxt = tokens[j]
                nxt_cf = nxt.casefold()

                # Gặp phủ định mới -> dừng
                if nxt_cf in self._negation_words_lower:
                    break

                # Gặp dấu câu / số / token boundary -> dừng
                if (
                    self._is_punctuation_only(nxt)
                    or self._is_number_like(nxt)
                    or nxt.startswith("_")
                ):
                    break

                phrase.append(nxt)

                if nxt_cf not in self._negation_skip_lower:
                    result.append(sep.join(phrase))
                    self._stats["negation_merged"] += 1
                    i = j + 1
                    joined = True
                    break

                j += 1
                skipped += 1

            if joined:
                continue

            result.append(current)
            i += 1

        return result

    # =====================================================
    # PIPELINE
    # =====================================================

    def tokenize_sentence(self, sentence: str) -> List[str]:
        """
        Tách từ 1 câu và áp dụng:
        underthesea -> phrase merge -> negation join
        """
        tokens = self.word_tokenize_sentence(sentence)

        if self.config.use_phrase_merge:
            tokens = self.merge_phrases(tokens)

        if self.config.join_negation:
            tokens = self.join_negation(tokens)

        return tokens

    def process(self, text: str) -> List[Dict[str, List[str]]]:
        if not text:
            return []

        text = _normalize_whitespace(text)
        if not text:
            return []

        # Chỉ dùng nhánh pre-tokenized khi được khai báo tường minh qua
        # config (treat_input_as_pretokenized), không tự đoán bằng heuristic.
        if self.config.treat_input_as_pretokenized:
            sentences = self._split_pre_tokenized_sentences(text)
        else:
            sentences = self.split_sentences(text)

        output: List[Dict[str, List[str]]] = []
        for sentence in sentences:
            tokens = self.tokenize_sentence(sentence)
            output.append(
                {
                    "sentence": sentence,
                    "tokens": tokens,
                }
            )
            self._stats["sentences"] += 1
            self._stats["tokens"] += len(tokens)

        return output

    def process_batch(self, texts: Iterable[str]) -> List[List[Dict[str, List[str]]]]:
        return [self.process(text) for text in texts]

    def tokenize(self, text: str) -> List[Dict[str, List[str]]]:
        return self.process(text)

    def tokenize_batch(self, texts: Iterable[str]) -> List[List[Dict[str, List[str]]]]:
        return self.process_batch(texts)

    def flatten_tokens(self, text: str) -> List[str]:
        processed = self.process(text)
        flat: List[str] = []
        for item in processed:
            flat.extend(item["tokens"])
        return flat

    def to_phobert_text(self, text: str) -> str:
        """
        Xuất text cho PhoBERT, các câu được nối lại bằng dấu cách để ra đúng dạng:
        Quán này ngon lắm ! ! ! ! ! ! Món bò_bít_tết rất ngon , phục_vụ hơi chậm .
        """
        if not text:
            return ""

        processed = self.process(text)
        if not processed:
            return ""

        sentence_texts = [
            _normalize_whitespace(" ".join(item["tokens"]))
            for item in processed
        ]

        sep = self.config.phobert_sentence_separator or " "
        return sentence_texts[0] + "".join(f"{sep}{s}" for s in sentence_texts[1:])

    def analyze_text(self, text: str) -> Dict[str, int]:
        processed = self.process(text)
        flat = [tok for item in processed for tok in item["tokens"]]

        neg_join_count = 0
        for tok in flat:
            tok_cf = tok.casefold()
            if "_" in tok and any(tok_cf.startswith(w + "_") for w in self._negation_words_lower):
                neg_join_count += 1

        return {
            "sentences": len(processed),
            "tokens": len(flat),
            "compound_tokens": sum(1 for t in flat if "_" in t),
            "negation_joined_tokens": neg_join_count,
        }

    # =====================================================
    # STATS / DEBUG
    # =====================================================

    def show_stats(self) -> None:
        sent = _cached_sentence_split.cache_info()
        word = _cached_word_tokenize.cache_info()

        print("\n========== TOKENIZER STATS ==========")
        print(f"Underthesea:            {HAS_UNDERTHESEA}")
        print(f"Sentence tokenize:      {self.config.use_sentence_tokenization}")
        print(f"Word tokenize:          {self.config.use_word_tokenization}")
        print(f"Phrase merge:           {self.config.use_phrase_merge}")
        print(f"Join negation:          {self.config.join_negation}")
        print(f"Negation words:         {len(self.negation_words)}")
        print(f"Negation skip tokens:    {len(self.negation_skip_tokens)}")
        print(f"Phrase entries:         {len(self.phrase_map)}")
        print("-------------------------------------")
        print("Sentence cache:")
        print(f"  Hits:      {sent.hits}")
        print(f"  Misses:    {sent.misses}")
        print(f"  Current:   {sent.currsize}")
        print(f"  Max size:   {sent.maxsize}")
        print("-------------------------------------")
        print("Word cache:")
        print(f"  Hits:      {word.hits}")
        print(f"  Misses:    {word.misses}")
        print(f"  Current:   {word.currsize}")
        print(f"  Max size:   {word.maxsize}")
        print("-------------------------------------")
        print("Runtime stats:")
        print(f"  Sentences:       {self._stats['sentences']}")
        print(f"  Tokens:          {self._stats['tokens']}")
        print(f"  Phrase merged:   {self._stats['phrase_merged']}")
        print(f"  Negation merged: {self._stats['negation_merged']}")
        print("=====================================\n")

if __name__ == "__main__":
    tokenizer = VietnameseTokenizer()

    text = "Quán này ngon lắm!!!!!! Món bò bít tết rất ngon, phục vụ hơi chậm."

    print("=== INPUT ===")
    print(text)

    print("\n=== Bước 1: Tách câu ===")
    sentences = tokenizer.split_sentences(text)
    for s in sentences:
        print(" -", s)

    print("\n=== Bước 2: underthesea word_tokenize (trước khi tra dictionary) ===")
    for s in sentences:
        raw_tokens = tokenizer.word_tokenize_sentence(s)
        print(" -", raw_tokens)

    print("\n=== Bước 3: Áp dụng dictionary food_phrases.json (merge_phrases) ===")
    for s in sentences:
        raw_tokens = tokenizer.word_tokenize_sentence(s)
        merged_tokens = tokenizer.merge_phrases(raw_tokens)
        print(" -", merged_tokens)

    print("\n=== process() — kết quả từng câu (đầy đủ pipeline) ===")
    print(tokenizer.process(text))

    print("\n=== to_phobert_text() — kết quả cuối cùng cho PhoBERT ===")
    print(tokenizer.to_phobert_text(text))

    # tokenizer.show_stats()
