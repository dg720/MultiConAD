import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
PHASE1_ROOT = PROCESSED_ROOT / "phase1"
TABLES_PHASE1_ROOT = PROJECT_ROOT / "tables" / "01-baselines" / "feature-baselines"
TABLES_PHASE1_TABLES_ROOT = TABLES_PHASE1_ROOT / "result-tables"
TABLES_PHASE1_RESULT_TABLES = TABLES_PHASE1_TABLES_ROOT / "csv"
TABLES_PHASE1_SUMMARIES = TABLES_PHASE1_ROOT / "summaries"
LOGS_ROOT = TABLES_PHASE1_ROOT / "logs"

PHASE1_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_PHASE1_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_PHASE1_TABLES_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_PHASE1_RESULT_TABLES.mkdir(parents=True, exist_ok=True)
TABLES_PHASE1_SUMMARIES.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)


LANG_NAME_TO_CODE = {
    "english": "en",
    "spanish": "es",
    "chinese": "zh",
    "greek": "el",
}

RAW_LANG_TO_CODE = {
    "eng": "en",
    "en": "en",
    "spa": "es",
    "cat": "es",
    "zh": "zh",
    "el": "el",
    "greek": "el",
    "spanish": "es",
    "english": "en",
    "chinese": "zh",
}


FUNCTION_WORDS = {
    "en": {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "can", "will", "just", "is", "am", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "our",
        "this", "that", "these", "those",
    },
    "es": {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "si", "porque", "como",
        "de", "del", "al", "a", "en", "con", "sin", "sobre", "entre", "para", "por", "desde", "hasta",
        "durante", "antes", "después", "arriba", "abajo", "más", "menos", "muy", "no", "ni", "solo",
        "también", "ya", "aquí", "allí", "cuando", "donde", "porqué", "cómo", "es", "son", "era", "eran",
        "ser", "estar", "ha", "han", "había", "yo", "tú", "usted", "él", "ella", "nosotros", "ellos",
        "me", "te", "se", "mi", "tu", "su", "nuestro", "esta", "este", "eso", "esa", "estos", "esas",
    },
    "el": {
        "και", "ή", "αλλά", "αν", "για", "με", "σε", "στο", "στη", "στην", "στον", "του", "της", "των",
        "το", "τα", "ο", "η", "οι", "ένα", "μια", "να", "που", "πως", "όπως", "όταν", "εδώ", "εκεί",
        "είναι", "ήταν", "ήμαι", "είμαι", "είμαστε", "είστε", "έχω", "έχει", "είχε", "δεν", "μη", "μην",
        "εγώ", "εσύ", "αυτός", "αυτή", "αυτό", "εμείς", "αυτοί", "μου", "σου", "μας", "σας", "τους",
        "αυτός", "αυτή", "αυτά",
    },
    "zh": {
        "的", "了", "在", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那", "有", "和",
        "跟", "与", "也", "就", "都", "而", "很", "吗", "呢", "啊", "吧", "把", "被", "给", "对", "上", "下",
        "里", "外", "到", "从", "让", "着", "过",
    },
}


FILLERS = {
    "en": {"um", "uh", "erm", "hmm", "mm", "mhm"},
    "es": {"eh", "em", "este", "pues"},
    "el": {"ε", "εε", "εμμ", "λοιπόν"},
    "zh": {"嗯", "呃", "啊", "这个", "那个"},
}


def make_logger(name: str):
    log_path = LOGS_ROOT / f"{name}.log"

    def log(message: str) -> None:
        line = message.encode("ascii", "backslashreplace").decode("ascii")
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return log


def load_cleaned_full_corpus() -> pd.DataFrame:
    cleaned_root = PROCESSED_ROOT / "cleaned"
    frames = []
    for lang_name in ("english", "spanish", "chinese", "greek"):
        for split in ("train", "test"):
            path = cleaned_root / f"{split}_{lang_name}.jsonl"
            df = pd.read_json(path, lines=True)
            df["legacy_split"] = split
            df["source_language_name"] = lang_name
            frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    return merged


def normalize_language(raw_language: str, source_language_name: str) -> str:
    if source_language_name in LANG_NAME_TO_CODE:
        return LANG_NAME_TO_CODE[source_language_name]
    return RAW_LANG_TO_CODE.get(str(raw_language).strip().lower(), "unknown")


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_analysis_text(row: pd.Series) -> str:
    participant = clean_text(row.get("Text_participant", ""))
    combined = clean_text(row.get("Text_interviewer_participant", ""))
    return participant if participant else combined


def split_utterances(text: str, language: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    if language == "zh":
        pieces = re.split(r"[。！？；]+", text)
    else:
        pieces = re.split(r"[.!?;:]+", text)

    utterances = [piece.strip() for piece in pieces if piece.strip()]
    return utterances if utterances else [text]


def tokenize(text: str, language: str):
    text = clean_text(text).lower()
    if not text:
        return []

    if language == "zh":
        try:
            import jieba  # type: ignore

            return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
        except Exception:
            return [ch for ch in text if ch.strip()]

    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def count_syllables(tokens: list[str], language: str) -> int:
    if language == "zh":
        return len(tokens)

    vowels = "aeiouyáéíóúàèìòùäëïöüåãõ"
    total = 0
    for token in tokens:
        token = token.lower()
        matches = re.findall(rf"[{vowels}]+", token)
        total += max(1, len(matches)) if token else 0
    return total


def mattr(tokens: list[str], window: int) -> float:
    if not tokens:
        return np.nan
    if len(tokens) < window:
        return len(set(tokens)) / max(len(tokens), 1)
    values = []
    for idx in range(len(tokens) - window + 1):
        span = tokens[idx: idx + window]
        values.append(len(set(span)) / window)
    return float(np.mean(values)) if values else np.nan


def brunet_index(tokens: list[str]) -> float:
    n = len(tokens)
    v = len(set(tokens))
    if n == 0 or v == 0:
        return np.nan
    return float(n ** (v ** -0.165))


def honore_statistic(tokens: list[str]) -> float:
    n = len(tokens)
    counts = Counter(tokens)
    v = len(counts)
    v1 = sum(1 for count in counts.values() if count == 1)
    if n == 0 or v == 0 or v == v1:
        return np.nan
    return float(100.0 * math.log(max(n, 1)) / (1.0 - (v1 / v)))


def stable_id(*parts: str) -> str:
    joined = "||".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator / denominator)


def feature_group_columns(feature_names: list[str]) -> list[str]:
    groups = sorted({name.split("_", 1)[0] for name in feature_names if "_" in name})
    columns = []
    for group in groups:
        columns.append(f"fg_{group}_available")
        columns.append(f"fg_{group}_num_missing")
    return columns


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
