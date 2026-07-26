# src/preprocessing/dictionary_based.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from src.config import CONFIG

rule_cfg = CONFIG.build_rule_based_config()
dict_cfg = CONFIG.build_dictionary_config()
tok_cfg = CONFIG.build_tokenizer_config()

__all__ = ["DictionaryBasedNormalizer", "DictionaryConfig"]


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class DictionaryConfig:
    project_root: Optional[str] = None

    teencode_path: str = "dictionaries/teencode.json"
    abbreviation_path: str = "dictionaries/abbreviation.json"
    english_food_path: str = "dictionaries/english_food.json"
    emoji_path: str = "dictionaries/emoji.json"
    emoticon_path: str = "dictionaries/emoticon.json"
    negation_path: str = "dictionaries/negation.json"

    # Negation scope handling
    max_negation_scope: int = 4

    # Nếu dictionary emoji/emoticon không có vi/text/normalized,
    # sẽ fallback thành token sentiment-friendly
    use_emoji_label_fallback: bool = True
    use_emoticon_label_fallback: bool = True

    # Bật/tắt kiểm tra dictionary
    validate_dictionaries: bool = True


# =========================================================
# CORE NORMALIZER
# =========================================================

class DictionaryBasedNormalizer:
    """
    Dictionary-based normalization with:
    - boundary matching
    - longest match first
    - regex precompilation
    - multiword phrase support
    - case-insensitive matching
    - dictionary validation
    - replacement statistics
    - batch normalize
    - emoji/emoticon repetition preservation
    - negation scope handling
    """

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, config: Optional[DictionaryConfig] = None):
        self.config = config or DictionaryConfig()
        self.project_root = self._resolve_root(self.config.project_root)

        # Load dictionaries
        self.teencode_map = self._load_dict(self.config.teencode_path)
        self.abbrev_map = self._load_dict(self.config.abbreviation_path)
        self.english_food_map = self._load_dict(self.config.english_food_path)
        self.emoji_map = self._load_dict(self.config.emoji_path)
        self.emoticon_map = self._load_dict(self.config.emoticon_path)
        self.negation_data = self._load_json(self.config.negation_path)

        # Negation config parsed from file
        self.negation_words, self.negation_skip_tokens = self._parse_negation_data(self.negation_data)
        self.negation_canonical = "không"

        # Replacement statistics
        self.replacement_stats = {
            "teencode": Counter(),
            "abbreviation": Counter(),
            "english_food": Counter(),
            "emoji": Counter(),
            "emoticon": Counter(),
            "negation": Counter(),
            "negation_scope": Counter(),
        }

        # Validation report
        self.validation_report = self.validate_dictionaries() if self.config.validate_dictionaries else {
            "errors": [],
            "warnings": [],
            "collisions": {},
            "summary": {},
        }

        # Precompile regex engines
        self._compile_engines()

    # =====================================================
    # PATH HANDLING
    # =====================================================

    def _resolve_root(self, project_root: Optional[str]) -> Path:
        if project_root:
            return Path(project_root).resolve()
        return Path(__file__).resolve().parents[2]

    def _resolve_path(self, relative_path: str) -> Path:
        return (self.project_root / relative_path).resolve()

    # =====================================================
    # LOADING
    # =====================================================

    def _load_json(self, path: str) -> dict:
        file_path = self._resolve_path(path)

        if not file_path.exists():
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_dict(self, path: str) -> Dict[str, Any]:
        data = self._load_json(path)
        if not isinstance(data, dict):
            return {}
        return {str(k).strip(): v for k, v in data.items() if str(k).strip()}

    def _parse_negation_data(self, data: Any) -> tuple[list[str], list[str]]:
        """
        Supports:
        {
            "negation_words": [...],
            "skip_tokens": [...]
        }
        """
        if not isinstance(data, dict):
            return [], []

        neg_words = data.get("negation_words", [])
        skip_tokens = data.get("skip_tokens", [])

        if not isinstance(neg_words, list):
            neg_words = []
        if not isinstance(skip_tokens, list):
            skip_tokens = []

        neg_words_clean = [str(x).strip() for x in neg_words if str(x).strip()]
        skip_tokens_clean = [str(x).strip() for x in skip_tokens if str(x).strip()]
        return neg_words_clean, skip_tokens_clean

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate_dictionaries(self) -> dict[str, Any]:
        """
        Validate loaded dictionaries and detect basic issues.
        Returns a report dict.
        """
        report = {
            "errors": [],
            "warnings": [],
            "collisions": {},
            "summary": {},
        }

        sources = {
            "teencode": self.teencode_map,
            "abbreviation": self.abbrev_map,
            "english_food": self.english_food_map,
            "emoji": self.emoji_map,
            "emoticon": self.emoticon_map,
        }

        # Basic checks
        for name, mapping in sources.items():
            if not isinstance(mapping, dict):
                report["errors"].append(f"{name}: dictionary is not a dict.")
                continue

            invalid_keys = 0
            empty_values = 0
            nested_dict_values = 0

            for k, v in mapping.items():
                if not isinstance(k, str) or not k.strip():
                    invalid_keys += 1
                if v is None:
                    empty_values += 1
                if isinstance(v, dict):
                    nested_dict_values += 1

            report["summary"][name] = {
                "entries": len(mapping),
                "invalid_keys": invalid_keys,
                "empty_values": empty_values,
                "nested_dict_values": nested_dict_values,
            }

            if invalid_keys:
                report["warnings"].append(f"{name}: {invalid_keys} invalid keys found.")
            if empty_values:
                report["warnings"].append(f"{name}: {empty_values} empty values found.")

            # Case-insensitive collisions inside each dictionary
            cf_groups = defaultdict(list)
            for k in mapping.keys():
                cf_groups[k.casefold()].append(k)
            internal_collisions = {
                cf: keys for cf, keys in cf_groups.items() if len(keys) > 1
            }
            if internal_collisions:
                report["collisions"][name] = internal_collisions
                report["warnings"].append(
                    f"{name}: case-insensitive collisions detected ({len(internal_collisions)} groups)."
                )

        # Cross-dictionary collisions
        all_keys = defaultdict(list)
        for name, mapping in sources.items():
            if not isinstance(mapping, dict):
                continue
            for k in mapping.keys():
                all_keys[k.casefold()].append(name)

        cross_collisions = {
            k: sources_list for k, sources_list in all_keys.items() if len(sources_list) > 1
        }
        if cross_collisions:
            report["collisions"]["cross_dictionary"] = cross_collisions
            report["warnings"].append(
                f"cross_dictionary: {len(cross_collisions)} overlapping keys across dictionaries."
            )

        # Negation validation
        if self.negation_data and not isinstance(self.negation_data, dict):
            report["warnings"].append("negation: negation.json is not a dict.")
        elif self.negation_data:
            if not isinstance(self.negation_data.get("negation_words", []), list):
                report["warnings"].append("negation: negation_words should be a list.")
            if not isinstance(self.negation_data.get("skip_tokens", []), list):
                report["warnings"].append("negation: skip_tokens should be a list.")

        return report

    # =====================================================
    # REPLACEMENT ENGINE
    # =====================================================

    def _sorted_unique_keys(self, keys: Iterable[str]) -> list[str]:
        """
        Longest match first.
        """
        seen = set()
        unique_keys = []
        for k in keys:
            if not k:
                continue
            ck = str(k)
            if ck not in seen:
                seen.add(ck)
                unique_keys.append(ck)
        unique_keys.sort(key=len, reverse=True)
        return unique_keys

    def _build_pattern(
        self,
        keys: Iterable[str],
        *,
        boundary: bool,
        case_insensitive: bool,
    ) -> Optional[re.Pattern]:
        """
        Precompile one regex per dictionary.
        """
        cleaned_keys = self._sorted_unique_keys(keys)
        if not cleaned_keys:
            return None

        escaped = [re.escape(k) for k in cleaned_keys]
        joined = "|".join(escaped)

        if boundary:
            pattern = rf"(?<!\w)(?:{joined})(?!\w)"
        else:
            pattern = rf"(?:{joined})"

        flags = re.UNICODE
        if case_insensitive:
            flags |= re.IGNORECASE

        return re.compile(pattern, flags)

    def _get_replacement(self, value: Any, key: str, kind: str) -> str:
        """
        Supports:
        - str -> str
        - dict -> vi/text/normalized fallback
        - emoji/emoticon fallback -> sentiment-friendly token
        """
        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            for field in ("vi", "text", "normalized", "replacement"):
                if value.get(field):
                    return str(value[field])

            # Fallback for emoji/emoticon dictionaries that contain label/score/name/block
            if kind == "emoji" and self.config.use_emoji_label_fallback:
                label = value.get("label")
                if label:
                    return f"emoji_{str(label).strip().lower()}"

            if kind == "emoticon" and self.config.use_emoticon_label_fallback:
                label = value.get("label")
                if label:
                    return f"emoticon_{str(label).strip().lower()}"

            return key

        return key

    def _apply_mapping(
        self,
        text: str,
        *,
        pattern: Optional[re.Pattern],
        lookup: Dict[str, Any],
        kind: str,
        wrap_with_spaces: bool = False,
        count_stats_key: Optional[str] = None,
    ) -> str:
        if not text or not pattern or not lookup:
            return text

        stats_key = count_stats_key or kind

        def _replace(match: re.Match) -> str:
            raw = match.group(0)
            lookup_key = raw.casefold() if kind in {"teencode", "abbreviation", "english_food", "negation"} else raw
            value = lookup.get(lookup_key)
            replacement = self._get_replacement(value, raw, kind)

            # replacement statistics
            self.replacement_stats[stats_key][lookup_key] += 1

            if wrap_with_spaces:
                return f" {replacement} "
            return replacement

        return pattern.sub(_replace, text)

    # =====================================================
    # PRECOMPILED ENGINES
    # =====================================================

    def _compile_engines(self) -> None:
        # Word/phrase dictionaries: case-insensitive + boundary matching
        self._teencode_lookup = {k.casefold(): v for k, v in self.teencode_map.items()}
        self._abbrev_lookup = {k.casefold(): v for k, v in self.abbrev_map.items()}
        self._english_food_lookup = {k.casefold(): v for k, v in self.english_food_map.items()}

        self._teencode_pattern = self._build_pattern(self.teencode_map.keys(), boundary=True, case_insensitive=True)
        self._abbrev_pattern = self._build_pattern(self.abbrev_map.keys(), boundary=True, case_insensitive=True)
        self._english_food_pattern = self._build_pattern(self.english_food_map.keys(), boundary=True, case_insensitive=True)

        # Symbol dictionaries: exact literal matching, longest match first
        self._emoji_lookup = dict(self.emoji_map)
        self._emoticon_lookup = dict(self.emoticon_map)
        self._emoji_pattern = self._build_pattern(self.emoji_map.keys(), boundary=False, case_insensitive=False)
        self._emoticon_pattern = self._build_pattern(self.emoticon_map.keys(), boundary=False, case_insensitive=False)

        # Negation words
        self._negation_lookup = {w.casefold(): self.negation_canonical for w in self.negation_words}
        self._negation_pattern = self._build_pattern(self.negation_words, boundary=True, case_insensitive=True)

        self._negation_words_set = {w.casefold() for w in self.negation_words}
        self._negation_skip_set = {w.casefold() for w in self.negation_skip_tokens}

    # =====================================================
    # TOKEN HELPERS
    # =====================================================

    @staticmethod
    def _is_punctuation_token(token: str) -> bool:
        return bool(re.fullmatch(r"[^\w\s]+", token, flags=re.UNICODE))

    def _tokenize_for_negation(self, text: str) -> list[str]:
        if not text:
            return []
        return self._TOKEN_RE.findall(text)

    # =====================================================
    # INDIVIDUAL NORMALIZERS
    # =====================================================

    def normalize_teencode(self, text: str) -> str:
        return self._apply_mapping(
            text,
            pattern=self._teencode_pattern,
            lookup=self._teencode_lookup,
            kind="teencode",
        )

    def normalize_abbreviation(self, text: str) -> str:
        return self._apply_mapping(
            text,
            pattern=self._abbrev_pattern,
            lookup=self._abbrev_lookup,
            kind="abbreviation",
        )

    def normalize_english_food(self, text: str) -> str:
        return self._apply_mapping(
            text,
            pattern=self._english_food_pattern,
            lookup=self._english_food_lookup,
            kind="english_food",
        )

    def normalize_emoji(self, text: str) -> str:
        # Keep repetition by replacing each occurrence separately with spaces
        return self._apply_mapping(
            text,
            pattern=self._emoji_pattern,
            lookup=self._emoji_lookup,
            kind="emoji",
            wrap_with_spaces=True,
        )

    def normalize_emoticon(self, text: str) -> str:
        # Keep repetition by replacing each occurrence separately with spaces
        return self._apply_mapping(
            text,
            pattern=self._emoticon_pattern,
            lookup=self._emoticon_lookup,
            kind="emoticon",
            wrap_with_spaces=True,
        )

    def normalize_negation(self, text: str) -> str:
        """
        Normalize negation words into canonical form: 'không'
        """
        if not text or not self._negation_pattern:
            return text

        return self._apply_mapping(
            text,
            pattern=self._negation_pattern,
            lookup=self._negation_lookup,
            kind="negation",
        )


    def normalize_negation_scope(self, text: str) -> str:
        """
        Best-effort negation scope handling:
        - canonical negation word is 'không'
        - skips tokens in negation.json skip_tokens
        - joins negation with next content word as 'không_ngon'
        """
        if not text or not self.negation_words:
            return text

        tokens = self._tokenize_for_negation(text)
        if not tokens:
            return text

        result: list[str] = []
        i = 0
        scope_window = max(1, int(self.config.max_negation_scope))

        while i < len(tokens):
            tok = tokens[i]
            tok_cf = tok.casefold()

            # Trigger on canonical or original negation tokens
            if tok_cf in self._negation_words_set or tok_cf == self.negation_canonical:
                skipped_tokens: list[str] = []
                j = i + 1
                steps = 0
                target_index = None

                while j < len(tokens) and steps < scope_window:
                    candidate = tokens[j]
                    candidate_cf = candidate.casefold()

                    if candidate_cf in self._negation_skip_set:
                        skipped_tokens.append(candidate)
                        j += 1
                        steps += 1
                        continue

                    if self._is_punctuation_token(candidate):
                        break

                    target_index = j
                    break

                if target_index is not None:
                    target = tokens[target_index]
                    # keep skip tokens, but join the target
                    result.extend(skipped_tokens)
                    result.append(f"{self.negation_canonical}_{target}")
                    self.replacement_stats["negation_scope"][self.negation_canonical] += 1
                    i = target_index + 1
                    continue

                # no target found -> keep canonical negation
                result.append(self.negation_canonical)
                i += 1
                continue

            result.append(tok)
            i += 1

        return " ".join(result)

    # =====================================================
    # PIPELINE METHODS
    # =====================================================

    def normalize(self, text: str) -> str:
        """
        Full dictionary-based normalization pipeline.
        """
        if not text:
            return ""

        # lexical
        text = self.normalize_teencode(text)
        text = self.normalize_abbreviation(text)
        text = self.normalize_english_food(text)

        # emotive symbols
        text = self.normalize_emoji(text)
        text = self.normalize_emoticon(text)

        # negation normalization + scope
        text = self.normalize_negation(text)
        text = self.normalize_negation_scope(text)

        # cleanup
        text = self._WHITESPACE_RE.sub(" ", text).strip()
        return text

    def batch_normalize(self, texts: Iterable[str]) -> list[str]:
        """
        Normalize a batch of texts.
        """
        return [self.normalize(text) for text in texts]

    # Backward-compatible alias
    def normalize_batch(self, texts: Iterable[str]) -> list[str]:
        return self.batch_normalize(texts)

    # =====================================================
    # STATS / DEBUG
    # =====================================================

    def get_replacement_stats(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.replacement_stats.items()}

    def show_stats(self, top_k: int = 5) -> None:
        print("=== Loaded dictionaries ===")
        print("Teencode:", len(self.teencode_map))
        print("Abbreviation:", len(self.abbrev_map))
        print("English food:", len(self.english_food_map))
        print("Emoji:", len(self.emoji_map))
        print("Emoticon:", len(self.emoticon_map))
        print("Negation words:", len(self.negation_words))
        print("Negation skip tokens:", len(self.negation_skip_tokens))

        print("\n=== Replacement statistics ===")
        for name, counter in self.replacement_stats.items():
            total = sum(counter.values())
            print(f"{name}: {total}")
            if total and top_k > 0:
                top_items = counter.most_common(top_k)
                pretty = ", ".join(f"{k}={v}" for k, v in top_items)
                print(f"  top {top_k}: {pretty}")

        print("\n=== Validation report ===")
        errors = self.validation_report.get("errors", [])
        warnings = self.validation_report.get("warnings", [])
        collisions = self.validation_report.get("collisions", {})
        print("Errors:", len(errors))                                      # Lưu các lỗi nghiêm trọng, có thể làm cho normalization hoạt động sai.
        print("Warnings:", len(warnings))                                  # Lưu các cảnh báo, dictionary vẫn chạy được nhưng có thể gây giảm chất lượng.
        print("Collision groups:", len(collisions))                        # Lưu các trường hợp xung đột dictionary.

    # =====================================================
    # OPTIONAL TEST HELPERS
    # =====================================================

    def step_by_step(self, text: str) -> dict[str, str]:
        """
        Convenient for notebook/debugging.
        Returns output of each stage.
        """
        if not text:
            return {
                "raw": "",
                "teencode": "",
                "abbreviation": "",
                "english_food": "",
                "emoji": "",
                "emoticon": "",
                "negation": "",
                "negation_scope": "",
                "final": "",
            }

        t1 = self.normalize_teencode(text)
        t2 = self.normalize_abbreviation(t1)
        t3 = self.normalize_english_food(t2)
        t4 = self.normalize_emoji(t3)
        t5 = self.normalize_emoticon(t4)
        t6 = self.normalize_negation(t5)
        t7 = self.normalize_negation_scope(t6)
        final = self._WHITESPACE_RE.sub(" ", t7).strip()

        return {
            "raw": text,
            "teencode": t1,
            "abbreviation": t2,
            "english_food": t3,
            "emoji": t4,
            "emoticon": t5,
            "negation": t6,
            "negation_scope": t7,
            "final": final,
        }


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":
    normalizer = DictionaryBasedNormalizer()

    print("\n==============================")
    print("📌 DICTIONARY LOADING CHECK")
    print("==============================")
    normalizer.show_stats()

    test_texts = [
        "Quán này ngon lắm!!!",
        "ko ngon lắm, service vcl 😂😂",
        "burger ở đây tasty nhưng giá hơi cao",
        "đc cái ship nhanh, nhưng phục vụ hơi chậm",
        "pizza 🍕 rất ngon nhưng ko rẻ",
        "không rất ngon nhưng phục vụ ok",
        "ngonnnnnn quá!!!! 😂😂😂",
    ]

    print("\n==============================")
    print("📌 STEP-BY-STEP NORMALIZATION TEST")
    print("==============================")

    for i, text in enumerate(test_texts, 1):
        print(f"\n================ TEST {i} ================")
        print("RAW:            ", text)

        out = normalizer.step_by_step(text)
        print("TEENCODE:       ", out["teencode"])
        print("ABBR:           ", out["abbreviation"])
        print("ENGLISH:        ", out["english_food"])
        print("EMOJI:          ", out["emoji"])
        print("EMOTICON:       ", out["emoticon"])
        print("NEGATION:       ", out["negation"])
        print("NEGATION_SCOPE: ", out["negation_scope"])
        print("------------------------------")
        print("FINAL:          ", out["final"])

    print("\n==============================")
    print(" STEP-BY-STEP TEST COMPLETED")
    print("==============================")