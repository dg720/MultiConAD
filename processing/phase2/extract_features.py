import json
import math
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.phase1.common import (
    FILLERS,
    FUNCTION_WORDS,
    clean_text,
    count_syllables,
    make_logger,
    safe_div,
    split_utterances,
    tokenize,
    write_json,
)
from processing.phase1.extract_features import (
    collect_chat_tier_stats,
    collect_doc_stats,
    dependency_depths_from_heads,
    get_stanza_pipeline,
    immediate_repetition_count,
    repeated_ngram_ratio,
)
from processing.phase2.common import LOGS_PHASE2_ROOT, PHASE2_ROOT, RESOURCES_PHASE2_ROOT, TABLES_PHASE2_ROOT


MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "phase1" / "phase1_manifest.jsonl"
PHASE1_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "phase1" / "phase1_features.csv"
FEATURES_PATH = PHASE2_ROOT / "phase2_features.csv"
METADATA_PATH = PHASE2_ROOT / "phase2_feature_metadata.csv"
SUMMARY_PATH = TABLES_PHASE2_ROOT / "phase2_manifest_summary.json"

PICTURE_LEXICONS_PATH = RESOURCES_PHASE2_ROOT / "picture_prompt_lexicons.json"
READING_REFERENCES_PATH = RESOURCES_PHASE2_ROOT / "reading_references.json"
STORY_REFERENCES_PATH = RESOURCES_PHASE2_ROOT / "story_references.json"
FLUENCY_RESOURCES_PATH = RESOURCES_PHASE2_ROOT / "fluency_resources.json"


def _safe_stat(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def stats_from_array(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurtosis": np.nan,
        }
    return {
        f"{prefix}_mean": _safe_stat(np.mean(values)),
        f"{prefix}_std": _safe_stat(np.std(values)),
        f"{prefix}_skew": _safe_stat(skew(values, bias=False)) if values.size > 2 else np.nan,
        f"{prefix}_kurtosis": _safe_stat(kurtosis(values, fisher=True, bias=False)) if values.size > 3 else np.nan,
    }


def cosine_between_texts(text_a: str, text_b: str) -> float:
    text_a = clean_text(text_a)
    text_b = clean_text(text_b)
    if not text_a or not text_b:
        return np.nan
    try:
        matrix = TfidfVectorizer().fit_transform([text_a, text_b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        return np.nan


def utterance_similarity_matrix(utterances: list[str]) -> np.ndarray | None:
    utterances = [clean_text(utt) for utt in utterances if clean_text(utt)]
    if len(utterances) < 2:
        return None
    try:
        matrix = TfidfVectorizer().fit_transform(utterances)
        return cosine_similarity(matrix)
    except ValueError:
        return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_text_stats_cache(manifest: pd.DataFrame, log) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    for language, group in manifest.groupby("language"):
        subset = group[group["analysis_text"].astype(str).str.strip() != ""]
        if subset.empty:
            continue

        log(f"Phase2 parsing language={language} rows={len(subset)}")
        pipeline = None
        for processed, (_, row) in enumerate(subset.iterrows(), start=1):
            sample_id = row["sample_id"]
            text = clean_text(row["analysis_text"])
            transcript_path = str(row.get("transcript_path", ""))
            stats = collect_chat_tier_stats(transcript_path, text, language)
            if stats is None:
                try:
                    if pipeline is None:
                        pipeline = get_stanza_pipeline(language)
                    stats = collect_doc_stats(text, language, pipeline(text))
                except Exception:
                    stats = collect_doc_stats(text, language, None)
            cache[sample_id] = stats
            if processed % 100 == 0 or processed == len(subset):
                log(f"Phase2 parsing language={language} progress={processed}/{len(subset)}")
    return cache


def build_language_frequency_tables(text_stats_cache: dict[str, dict[str, object]], manifest: pd.DataFrame) -> dict[str, Counter]:
    language_lookup = manifest.set_index("sample_id")["language"].to_dict()
    counters: dict[str, Counter] = {}
    for sample_id, stats in text_stats_cache.items():
        language = language_lookup.get(sample_id, "unknown")
        counters.setdefault(language, Counter())
        counters[language].update(stats.get("lemmas", []))
    return counters


def empty_pd_features() -> dict[str, float]:
    return {
        "pd_unique_units_count": np.nan,
        "pd_unique_units_ratio": np.nan,
        "pd_total_unit_mentions": np.nan,
        "pd_content_density": np.nan,
        "pd_object_units_ratio": np.nan,
        "pd_action_units_ratio": np.nan,
        "pd_keyword_to_nonkeyword_ratio": np.nan,
        "pd_repeated_content_unit_ratio": np.nan,
        "pd_semantic_similarity_to_unit_list": np.nan,
        "pd_num_unique_keywords": np.nan,
        "pd_num_total_keywords": np.nan,
        "pd_unique_unit_density": np.nan,
        "pd_total_unit_density": np.nan,
        "pd_unique_unit_efficiency": np.nan,
        "pd_total_unit_efficiency": np.nan,
        "pd_percentage_units_mentioned": np.nan,
        "pd_keyword_ttr": np.nan,
        "pd_mean_utterance_to_unit_similarity": np.nan,
        "pd_global_prompt_coherence": np.nan,
        "pd_detected_prompt_score": np.nan,
    }


def empty_rd_features() -> dict[str, float]:
    return {
        "rd_reference_token_coverage": np.nan,
        "rd_reference_bigram_coverage": np.nan,
        "rd_prompt_similarity": np.nan,
        "rd_sequence_match_ratio": np.nan,
        "rd_reference_order_score": np.nan,
        "rd_omission_ratio": np.nan,
        "rd_insertion_ratio": np.nan,
        "rd_repetition_ratio": np.nan,
        "rd_pause_per_reference_token": np.nan,
        "rd_content_word_recall_ratio": np.nan,
    }


def empty_fc_features() -> dict[str, float]:
    return {
        "fc_topic_coherence": np.nan,
        "fc_topic_switch_rate": np.nan,
        "fc_embedding_dispersion": np.nan,
        "fc_named_entity_density": np.nan,
        "fc_repetition_rate": np.nan,
        "fc_mean_utterance_similarity": np.nan,
        "fc_first_last_similarity": np.nan,
        "fc_topic_return_ratio": np.nan,
    }


def empty_ft_features() -> dict[str, float]:
    return {
        "ft_item_count": np.nan,
        "ft_unique_item_count": np.nan,
        "ft_repetition_count": np.nan,
        "ft_intrusion_count": np.nan,
        "ft_valid_item_count": np.nan,
        "ft_valid_item_ratio": np.nan,
        "ft_items_per_second": np.nan,
        "ft_cluster_count": np.nan,
        "ft_mean_cluster_size": np.nan,
        "ft_switch_count": np.nan,
        "ft_letter_valid_count": np.nan,
        "ft_letter_violation_count": np.nan,
    }


def empty_sr_features() -> dict[str, float]:
    return {
        "sr_propositions_recalled_count": np.nan,
        "sr_propositions_recalled_ratio": np.nan,
        "sr_key_event_coverage": np.nan,
        "sr_event_order_score": np.nan,
        "sr_intrusion_count": np.nan,
        "sr_semantic_similarity_to_reference_story": np.nan,
        "sr_repetition_rate": np.nan,
        "sr_content_density": np.nan,
    }


def empty_pr_features() -> dict[str, float]:
    names = [
        "pr_np_count",
        "pr_vp_count",
        "pr_pp_count",
        "pr_adjp_count",
        "pr_advp_count",
        "pr_np_ratio",
        "pr_vp_ratio",
        "pr_pp_ratio",
        "pr_adjp_ratio",
        "pr_advp_ratio",
        "pr_np_mean_len",
        "pr_vp_mean_len",
        "pr_pp_mean_len",
        "pr_adjp_mean_len",
        "pr_advp_mean_len",
        "pr_np_rate",
        "pr_vp_rate",
        "pr_pp_rate",
        "pr_chunk_density",
        "pr_chunk_diversity",
        "pr_det_noun_ratio",
        "pr_adj_noun_ratio",
        "pr_pron_verb_ratio",
        "pr_noun_verb_ratio",
        "pr_verb_det_ratio",
        "pr_adp_noun_ratio",
        "pr_aux_verb_ratio",
        "pr_verb_adv_ratio",
        "pr_adv_verb_ratio",
        "pr_noun_adp_ratio",
        "pr_verb_pron_ratio",
        "pr_cconj_pron_ratio",
        "pr_sconj_pron_ratio",
        "pr_adp_det_ratio",
        "pr_det_adj_ratio",
        "pr_det_adj_noun_ratio",
        "pr_det_noun_verb_ratio",
        "pr_pron_aux_verb_ratio",
        "pr_pron_verb_det_ratio",
        "pr_adp_det_noun_ratio",
        "pr_noun_aux_verb_ratio",
        "pr_root_verb_ratio",
        "pr_root_noun_ratio",
        "pr_root_pron_ratio",
        "pr_root_adj_ratio",
    ]
    return {name: np.nan for name in names}


def empty_sx_features() -> dict[str, float]:
    names = [
        "sx_sentence_length_q25",
        "sx_sentence_length_q50",
        "sx_sentence_length_q75",
        "sx_sentence_length_cv",
        "sx_dep_length_q25",
        "sx_dep_length_q50",
        "sx_dep_length_q75",
        "sx_tree_depth_std",
        "sx_tree_depth_q75",
        "sx_upos_entropy",
        "sx_dep_entropy",
        "sx_clause_per_100_tokens",
        "sx_coordination_density",
        "sx_modifier_density",
        "sx_nominal_density",
        "sx_verbal_density",
        "sx_np_like_ratio",
        "sx_pp_like_ratio",
        "sx_vp_like_ratio",
        "sx_subject_object_balance",
        "sx_function_word_ratio_lexicon",
        "sx_content_word_ratio_lexicon",
        "sx_mean_log_lemma_freq",
        "sx_std_log_lemma_freq",
        "sx_rare_lemma_ratio",
        "sx_very_rare_lemma_ratio",
        "sx_adjacent_similarity_min",
        "sx_adjacent_similarity_max",
        "sx_pairwise_similarity_std",
        "sx_low_similarity_frac_05",
        "sx_low_similarity_frac_03",
    ]
    return {name: np.nan for name in names}


def empty_par_features() -> dict[str, float]:
    base = {}
    for idx in range(1, 14):
        for stat in ("mean", "std", "skew", "kurtosis"):
            base[f"par_mfcc_{idx}_{stat}"] = np.nan
        for stat in ("mean", "std"):
            base[f"par_delta_mfcc_{idx}_{stat}"] = np.nan
    for name in ("spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness", "rms", "zcr"):
        for stat in ("mean", "std", "skew", "kurtosis"):
            base[f"par_{name}_{stat}"] = np.nan
    for name in ("f0",):
        for stat in ("mean", "std", "min", "max", "median", "q25", "q75", "iqr"):
            base[f"par_{name}_{stat}"] = np.nan
    base["par_voiced_frame_ratio"] = np.nan
    base["par_nonzero_rms_ratio"] = np.nan
    return base


def prompt_keywords_for_family(lexicon: dict, language: str) -> dict[str, list[str]]:
    language_block = lexicon.get("languages", {}).get(language, {})
    return {key: [clean_text(value).lower() for value in values] for key, values in language_block.items()}


def shallow_chunks(upos_sequence: list[str]) -> list[tuple[str, int]]:
    chunks: list[tuple[str, int]] = []
    idx = 0
    while idx < len(upos_sequence):
        pos = upos_sequence[idx]
        if pos in {"DET", "ADJ", "NUM", "NOUN", "PROPN", "PRON"}:
            start = idx
            seen_nominal = False
            while idx < len(upos_sequence) and upos_sequence[idx] in {"DET", "ADJ", "NUM", "NOUN", "PROPN", "PRON"}:
                if upos_sequence[idx] in {"NOUN", "PROPN", "PRON"}:
                    seen_nominal = True
                idx += 1
            if seen_nominal:
                chunks.append(("NP", idx - start))
                continue
            idx = start
        if pos in {"AUX", "VERB", "ADV", "PART"}:
            start = idx
            seen_verb = False
            while idx < len(upos_sequence) and upos_sequence[idx] in {"AUX", "VERB", "ADV", "PART"}:
                if upos_sequence[idx] == "VERB":
                    seen_verb = True
                idx += 1
            if seen_verb:
                chunks.append(("VP", idx - start))
                continue
            idx = start
        if pos == "ADP":
            start = idx
            idx += 1
            while idx < len(upos_sequence) and upos_sequence[idx] in {"DET", "ADJ", "NUM", "NOUN", "PROPN", "PRON"}:
                idx += 1
            chunks.append(("PP", idx - start))
            continue
        if pos == "ADJ":
            start = idx
            while idx < len(upos_sequence) and upos_sequence[idx] == "ADJ":
                idx += 1
            chunks.append(("ADJP", idx - start))
            continue
        if pos == "ADV":
            start = idx
            while idx < len(upos_sequence) and upos_sequence[idx] == "ADV":
                idx += 1
            chunks.append(("ADVP", idx - start))
            continue
        idx += 1
    return chunks


def pos_ngram_ratios(upos_sequences: list[list[str]]) -> dict[str, float]:
    bigram_templates = [
        ("DET", "NOUN"),
        ("ADJ", "NOUN"),
        ("PRON", "VERB"),
        ("NOUN", "VERB"),
        ("VERB", "DET"),
        ("ADP", "NOUN"),
        ("AUX", "VERB"),
        ("VERB", "ADV"),
        ("ADV", "VERB"),
        ("NOUN", "ADP"),
        ("VERB", "PRON"),
        ("CCONJ", "PRON"),
        ("SCONJ", "PRON"),
        ("ADP", "DET"),
        ("DET", "ADJ"),
    ]
    trigram_templates = [
        ("DET", "ADJ", "NOUN"),
        ("DET", "NOUN", "VERB"),
        ("PRON", "AUX", "VERB"),
        ("PRON", "VERB", "DET"),
        ("ADP", "DET", "NOUN"),
        ("NOUN", "AUX", "VERB"),
    ]
    bigrams = []
    trigrams = []
    roots = []
    for seq in upos_sequences:
        if seq:
            roots.append(seq[0])
        bigrams.extend(zip(seq[:-1], seq[1:]))
        trigrams.extend(zip(seq[:-2], seq[1:-1], seq[2:]))
    bigram_counter = Counter(bigrams)
    trigram_counter = Counter(trigrams)
    root_counter = Counter(roots)
    total_bigrams = len(bigrams)
    total_trigrams = len(trigrams)
    total_roots = len(roots)
    features = {}
    for left, right in bigram_templates:
        features[f"pr_{left.lower()}_{right.lower()}_ratio"] = safe_div(bigram_counter[(left, right)], total_bigrams)
    for first, second, third in trigram_templates:
        features[f"pr_{first.lower()}_{second.lower()}_{third.lower()}_ratio"] = safe_div(
            trigram_counter[(first, second, third)],
            total_trigrams,
        )
    for root in ("VERB", "NOUN", "PRON", "ADJ"):
        features[f"pr_root_{root.lower()}_ratio"] = safe_div(root_counter[root], total_roots)
    return features


def production_phrase_features(row: pd.Series, stats: dict[str, object]) -> dict[str, float]:
    features = empty_pr_features()
    upos_sequences = [seq for seq in stats.get("upos_sequences", []) if seq]
    if not upos_sequences:
        return features

    all_chunks: list[tuple[str, int]] = []
    for seq in upos_sequences:
        all_chunks.extend(shallow_chunks(seq))

    if not all_chunks:
        features.update(pos_ngram_ratios(upos_sequences))
        return features

    chunk_counter = Counter(chunk_type for chunk_type, _ in all_chunks)
    chunk_lengths: dict[str, list[int]] = {}
    for chunk_type, length in all_chunks:
        chunk_lengths.setdefault(chunk_type, []).append(length)

    utterance_count = max(len(stats.get("utterances", [])), 1)
    total_chunks = len(all_chunks)
    for chunk_type in ("NP", "VP", "PP", "ADJP", "ADVP"):
        lower = chunk_type.lower()
        features[f"pr_{lower}_count"] = float(chunk_counter.get(chunk_type, 0))
        features[f"pr_{lower}_ratio"] = safe_div(chunk_counter.get(chunk_type, 0), total_chunks)
        lengths = np.array(chunk_lengths.get(chunk_type, []), dtype=float)
        features[f"pr_{lower}_mean_len"] = _safe_stat(np.mean(lengths)) if lengths.size else np.nan
        features[f"pr_{lower}_rate"] = safe_div(chunk_counter.get(chunk_type, 0), utterance_count)

    features["pr_chunk_density"] = safe_div(total_chunks, sum(len(seq) for seq in upos_sequences))
    features["pr_chunk_diversity"] = safe_div(len(chunk_counter), total_chunks)
    features.update(pos_ngram_ratios(upos_sequences))
    return features


def count_keyword_mentions(tokens: list[str], text: str, keywords: list[str]) -> int:
    token_counter = Counter(tokens)
    count = 0
    for keyword in keywords:
        if not keyword:
            continue
        if " " in keyword:
            count += text.count(keyword)
        else:
            count += token_counter.get(keyword, 0)
    return count


def detect_prompt_family(row: pd.Series, stats: dict[str, object], picture_lexicons: dict) -> tuple[str, float]:
    prompt_id = str(row.get("prompt_id", "")).strip().lower()
    if prompt_id == "cookie_theft":
        return "cookie_theft", 1.0

    language = row["language"]
    dataset_name = str(row.get("dataset_name", ""))
    if dataset_name in {"ADReSS-M", "DS5", "DS7"} and language == "el":
        return "lion_scene", 0.75

    tokens = [clean_text(token).lower() for token in stats.get("tokens", [])]
    text = " ".join(tokens)
    best_family = ""
    best_score = 0.0
    for family_name, family in picture_lexicons.items():
        lang_keywords = prompt_keywords_for_family(family, language)
        if not lang_keywords:
            continue
        category_hits = 0
        total_mentions = 0
        for keywords in lang_keywords.values():
            mentions = count_keyword_mentions(tokens, text, keywords)
            total_mentions += mentions
            category_hits += int(mentions > 0)
        score = category_hits + (0.1 * total_mentions)
        if score > best_score:
            best_family = family_name
            best_score = score
    if best_score < 2.0:
        return "", 0.0
    return best_family, float(best_score)


def picture_description_features(
    row: pd.Series,
    stats: dict[str, object],
    base_row: dict[str, float],
    picture_lexicons: dict,
) -> tuple[dict[str, float], str]:
    features = empty_pd_features()
    if row["task_type"] != "PD_CTP":
        return features, ""

    family_name, family_score = detect_prompt_family(row, stats, picture_lexicons)
    if not family_name:
        return features, ""

    family = picture_lexicons[family_name]
    keywords_by_category = prompt_keywords_for_family(family, row["language"])
    if not keywords_by_category:
        return features, ""

    tokens = [clean_text(token).lower() for token in stats.get("tokens", []) if clean_text(token)]
    utterances = [clean_text(utt).lower() for utt in stats.get("utterances", []) if clean_text(utt)]
    text = " ".join(tokens)
    token_count = len(tokens)
    duration = base_row.get("len_audio_duration", np.nan)

    category_mentions: dict[str, int] = {}
    matched_keywords: list[str] = []
    for category, keywords in keywords_by_category.items():
        mentions = count_keyword_mentions(tokens, text, keywords)
        category_mentions[category] = mentions
        if mentions > 0:
            matched_keywords.extend(keywords)

    object_categories = family.get("object_categories", [])
    action_categories = family.get("action_categories", [])
    unique_units = sum(1 for category, mentions in category_mentions.items() if mentions > 0 and category in object_categories + action_categories)
    total_mentions = sum(category_mentions.values())
    unique_keywords = len(set(matched_keywords))
    unique_units_possible = len(set(object_categories + action_categories))
    keyword_doc = " ".join(
        keyword
        for category in object_categories + action_categories
        for keyword in keywords_by_category.get(category, [])
    )

    utterance_sims = [
        cosine_between_texts(utterance, keyword_doc)
        for utterance in utterances
    ]
    utterance_sims = [value for value in utterance_sims if pd.notna(value)]

    features.update(
        {
            "pd_unique_units_count": float(unique_units),
            "pd_unique_units_ratio": safe_div(unique_units, unique_units_possible),
            "pd_total_unit_mentions": float(total_mentions),
            "pd_content_density": safe_div(total_mentions, token_count),
            "pd_object_units_ratio": safe_div(sum(category_mentions.get(cat, 0) for cat in object_categories), total_mentions),
            "pd_action_units_ratio": safe_div(sum(category_mentions.get(cat, 0) for cat in action_categories), total_mentions),
            "pd_keyword_to_nonkeyword_ratio": safe_div(total_mentions, max(token_count - total_mentions, 1)),
            "pd_repeated_content_unit_ratio": safe_div(sum(max(0, mentions - 1) for mentions in category_mentions.values()), total_mentions),
            "pd_semantic_similarity_to_unit_list": cosine_between_texts(text, keyword_doc),
            "pd_num_unique_keywords": float(unique_keywords),
            "pd_num_total_keywords": float(total_mentions),
            "pd_unique_unit_density": safe_div(unique_units, token_count),
            "pd_total_unit_density": safe_div(total_mentions, token_count),
            "pd_unique_unit_efficiency": safe_div(unique_units, duration),
            "pd_total_unit_efficiency": safe_div(total_mentions, duration),
            "pd_percentage_units_mentioned": safe_div(unique_units, unique_units_possible),
            "pd_keyword_ttr": safe_div(unique_keywords, total_mentions),
            "pd_mean_utterance_to_unit_similarity": float(np.mean(utterance_sims)) if utterance_sims else np.nan,
            "pd_global_prompt_coherence": cosine_between_texts(" ".join(utterances), keyword_doc),
            "pd_detected_prompt_score": family_score,
        }
    )
    return features, family_name


def reading_features(row: pd.Series, stats: dict[str, object], base_row: dict[str, float], reading_refs: dict) -> dict[str, float]:
    features = empty_rd_features()
    if row["task_type"] != "READING":
        return features

    reference_text = clean_text(reading_refs.get(str(row.get("prompt_id", "")).strip().lower(), {}).get(row["language"], ""))
    if not reference_text:
        return features

    tokens = [clean_text(token).lower() for token in stats.get("tokens", []) if clean_text(token)]
    reference_tokens = tokenize(reference_text, row["language"])
    if not tokens or not reference_tokens:
        return features

    candidate_bigrams = set(zip(tokens[:-1], tokens[1:])) if len(tokens) > 1 else set()
    reference_bigrams = set(zip(reference_tokens[:-1], reference_tokens[1:])) if len(reference_tokens) > 1 else set()
    candidate_vocab = set(tokens)
    reference_vocab = set(reference_tokens)
    content_reference = [token for token in reference_tokens if token not in FUNCTION_WORDS.get(row["language"], set())]

    matcher = SequenceMatcher(a=reference_tokens, b=tokens)
    matched_blocks = sum(block.size for block in matcher.get_matching_blocks())

    features.update(
        {
            "rd_reference_token_coverage": safe_div(len(reference_vocab & candidate_vocab), len(reference_vocab)),
            "rd_reference_bigram_coverage": safe_div(len(reference_bigrams & candidate_bigrams), len(reference_bigrams)),
            "rd_prompt_similarity": cosine_between_texts(" ".join(tokens), reference_text),
            "rd_sequence_match_ratio": float(matcher.ratio()),
            "rd_reference_order_score": safe_div(matched_blocks, len(reference_tokens)),
            "rd_omission_ratio": safe_div(len(reference_vocab - candidate_vocab), len(reference_vocab)),
            "rd_insertion_ratio": safe_div(len(candidate_vocab - reference_vocab), len(candidate_vocab)),
            "rd_repetition_ratio": repeated_ngram_ratio(tokens, 1),
            "rd_pause_per_reference_token": safe_div(base_row.get("pause_total_duration", np.nan), len(reference_tokens)),
            "rd_content_word_recall_ratio": safe_div(len(set(content_reference) & candidate_vocab), len(set(content_reference))),
        }
    )
    return features


def story_features(row: pd.Series, stats: dict[str, object], prompt_family: str, story_refs: dict) -> dict[str, float]:
    features = empty_sr_features()
    reference_block = story_refs.get(prompt_family, {}).get(row["language"], {})
    reference_text = clean_text(reference_block.get("reference_text", ""))
    if not reference_text:
        return features

    tokens = [clean_text(token).lower() for token in stats.get("tokens", []) if clean_text(token)]
    utterances = [clean_text(utt).lower() for utt in stats.get("utterances", []) if clean_text(utt)]
    reference_tokens = tokenize(reference_text, row["language"])
    if not tokens or not reference_tokens:
        return features

    matcher = SequenceMatcher(a=reference_tokens, b=tokens)
    matched_blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    matched_size = sum(block.size for block in matched_blocks)
    reference_vocab = set(reference_tokens)
    token_vocab = set(tokens)
    temporal_markers = {"then", "after", "before", "when", "while", "next", "finally", "because", "and"}
    if row["language"] == "es":
        temporal_markers = {"entonces", "despues", "antes", "cuando", "mientras", "luego", "finalmente", "porque", "y"}
    elif row["language"] == "el":
        temporal_markers = {"μετα", "πριν", "οταν", "ενω", "υστερα", "τελικα", "γιατι", "και"}
    elif row["language"] == "zh":
        temporal_markers = {"然后", "之后", "之前", "当", "最后", "因为", "和"}

    event_order_hits = sum(1 for token in tokens if token in temporal_markers)
    features.update(
        {
            "sr_propositions_recalled_count": float(len(reference_vocab & token_vocab)),
            "sr_propositions_recalled_ratio": safe_div(len(reference_vocab & token_vocab), len(reference_vocab)),
            "sr_key_event_coverage": safe_div(matched_size, len(reference_tokens)),
            "sr_event_order_score": safe_div(event_order_hits, max(len(utterances), 1)),
            "sr_intrusion_count": float(len(token_vocab - reference_vocab)),
            "sr_semantic_similarity_to_reference_story": cosine_between_texts(" ".join(tokens), reference_text),
            "sr_repetition_rate": repeated_ngram_ratio(tokens, 1),
            "sr_content_density": safe_div(len(reference_vocab & token_vocab), len(tokens)),
        }
    )
    return features


def fluency_features(row: pd.Series, stats: dict[str, object], base_row: dict[str, float], fluency_resources: dict) -> dict[str, float]:
    features = empty_ft_features()
    if row["task_type"] != "FLUENCY":
        return features

    prompt_id = str(row.get("prompt_id", "")).strip().lower()
    if "animal" in prompt_id:
        resource_key = "animals"
    elif "letter_f" in prompt_id or "phon_f" in prompt_id or prompt_id.endswith("_f"):
        resource_key = "letter_f"
    else:
        return features

    resource = fluency_resources.get(resource_key, {})
    valid_items = {clean_text(item).lower() for item in resource.get("language_items", {}).get(row["language"], []) if clean_text(item)}
    if not valid_items:
        return features

    tokens = [clean_text(token).lower() for token in stats.get("tokens", []) if clean_text(token)]
    duration = base_row.get("len_audio_duration", np.nan)
    if not tokens:
        return features

    counts = Counter(tokens)
    valid_mentions = [token for token in tokens if token in valid_items]
    valid_counter = Counter(valid_mentions)
    unique_valid = list(valid_counter.keys())
    repetitions = sum(max(0, count - 1) for count in valid_counter.values())
    intrusions = [token for token in tokens if token not in valid_items and token not in FILLERS.get(row["language"], set())]

    switch_count = 0
    prev_initial = None
    for token in unique_valid:
        initial = token[:1]
        if prev_initial is not None and initial != prev_initial:
            switch_count += 1
        prev_initial = initial

    letter_valid_count = np.nan
    letter_violation_count = np.nan
    if resource_key == "letter_f":
        letter_valid_count = float(sum(1 for token in valid_mentions if token.startswith("f")))
        letter_violation_count = float(sum(1 for token in valid_mentions if not token.startswith("f")))

    cluster_sizes = []
    if unique_valid:
        current_cluster = 1
        prev_initial = unique_valid[0][:1]
        for token in unique_valid[1:]:
            initial = token[:1]
            if initial == prev_initial:
                current_cluster += 1
            else:
                cluster_sizes.append(current_cluster)
                current_cluster = 1
                prev_initial = initial
        cluster_sizes.append(current_cluster)

    features.update(
        {
            "ft_item_count": float(len(tokens)),
            "ft_unique_item_count": float(len(set(tokens))),
            "ft_repetition_count": float(repetitions),
            "ft_intrusion_count": float(len(intrusions)),
            "ft_valid_item_count": float(len(valid_mentions)),
            "ft_valid_item_ratio": safe_div(len(valid_mentions), len(tokens)),
            "ft_items_per_second": safe_div(len(valid_mentions), duration),
            "ft_cluster_count": float(len(cluster_sizes)),
            "ft_mean_cluster_size": _safe_stat(np.mean(cluster_sizes)) if cluster_sizes else np.nan,
            "ft_switch_count": float(switch_count),
            "ft_letter_valid_count": letter_valid_count,
            "ft_letter_violation_count": letter_violation_count,
        }
    )
    return features


def conversation_features(row: pd.Series, stats: dict[str, object]) -> dict[str, float]:
    features = empty_fc_features()
    if row["task_type"] != "CONVERSATION":
        return features

    utterances = [clean_text(utt).lower() for utt in stats.get("utterances", []) if clean_text(utt)]
    sims = utterance_similarity_matrix(utterances)
    tokens = [clean_text(token).lower() for token in stats.get("tokens", []) if clean_text(token)]
    token_count = len(tokens)
    if sims is None or token_count == 0:
        return features

    adjacent = [float(sims[i, i + 1]) for i in range(len(utterances) - 1)]
    pairwise = [float(sims[i, j]) for i in range(len(utterances)) for j in range(i + 1, len(utterances))]
    long_range = [
        float(sims[i, j])
        for i in range(len(utterances))
        for j in range(i + 2, len(utterances))
    ]

    features.update(
        {
            "fc_topic_coherence": float(np.mean(adjacent)) if adjacent else np.nan,
            "fc_topic_switch_rate": safe_div(sum(value < 0.2 for value in adjacent), len(adjacent)),
            "fc_embedding_dispersion": float(np.std(pairwise)) if pairwise else np.nan,
            "fc_named_entity_density": safe_div(stats.get("upos_counts", Counter()).get("PROPN", 0), token_count),
            "fc_repetition_rate": safe_div(immediate_repetition_count(tokens), token_count),
            "fc_mean_utterance_similarity": float(np.mean(pairwise)) if pairwise else np.nan,
            "fc_first_last_similarity": float(sims[0, -1]) if len(utterances) > 1 else np.nan,
            "fc_topic_return_ratio": safe_div(sum(value > 0.3 for value in long_range), len(long_range)),
        }
    )
    return features


def richer_syntax_features(
    row: pd.Series,
    stats: dict[str, object],
    language_frequency: Counter,
) -> dict[str, float]:
    features = empty_sx_features()

    tokens = [clean_text(token).lower() for token in stats.get("tokens", []) if clean_text(token)]
    lemmas = [clean_text(token).lower() for token in stats.get("lemmas", []) if clean_text(token)]
    utterances = [clean_text(utt).lower() for utt in stats.get("utterances", []) if clean_text(utt)]
    upos_counts: Counter = stats.get("upos_counts", Counter())
    dep_counts: Counter = stats.get("dep_counts", Counter())
    dep_lengths = np.array(stats.get("dep_lengths", []), dtype=float)
    tree_depths = np.array(stats.get("tree_depths", []), dtype=float)
    token_count = len(tokens)
    dep_total = sum(dep_counts.values())
    if token_count == 0:
        return features

    utterance_lengths = np.array([len(tokenize(utt, row["language"])) for utt in utterances if utt], dtype=float)
    function_lexicon = FUNCTION_WORDS.get(row["language"], set())
    function_hits = sum(1 for token in tokens if token in function_lexicon)
    lemma_total = max(sum(language_frequency.values()), 1)
    lemma_log_freqs = np.array(
        [math.log10((language_frequency.get(lemma, 0) + 1) / lemma_total) for lemma in lemmas],
        dtype=float,
    ) if lemmas else np.array([], dtype=float)
    sims = utterance_similarity_matrix(utterances)
    adjacent = [float(sims[i, i + 1]) for i in range(len(utterances) - 1)] if sims is not None else []
    pairwise = [float(sims[i, j]) for i in range(len(utterances)) for j in range(i + 1, len(utterances))] if sims is not None else []

    content_like = token_count - function_hits
    subject_count = dep_counts.get("nsubj", 0) + dep_counts.get("csubj", 0)
    object_count = dep_counts.get("obj", 0) + dep_counts.get("iobj", 0)

    features.update(
        {
            "sx_sentence_length_q25": _safe_stat(np.percentile(utterance_lengths, 25)) if utterance_lengths.size else np.nan,
            "sx_sentence_length_q50": _safe_stat(np.percentile(utterance_lengths, 50)) if utterance_lengths.size else np.nan,
            "sx_sentence_length_q75": _safe_stat(np.percentile(utterance_lengths, 75)) if utterance_lengths.size else np.nan,
            "sx_sentence_length_cv": safe_div(float(np.std(utterance_lengths)) if utterance_lengths.size else np.nan, float(np.mean(utterance_lengths)) if utterance_lengths.size else np.nan),
            "sx_dep_length_q25": _safe_stat(np.percentile(dep_lengths, 25)) if dep_lengths.size else np.nan,
            "sx_dep_length_q50": _safe_stat(np.percentile(dep_lengths, 50)) if dep_lengths.size else np.nan,
            "sx_dep_length_q75": _safe_stat(np.percentile(dep_lengths, 75)) if dep_lengths.size else np.nan,
            "sx_tree_depth_std": _safe_stat(np.std(tree_depths)) if tree_depths.size else np.nan,
            "sx_tree_depth_q75": _safe_stat(np.percentile(tree_depths, 75)) if tree_depths.size else np.nan,
            "sx_upos_entropy": _safe_stat(entropy(np.array(list(upos_counts.values()), dtype=float))) if upos_counts else np.nan,
            "sx_dep_entropy": _safe_stat(entropy(np.array(list(dep_counts.values()), dtype=float))) if dep_counts else np.nan,
            "sx_clause_per_100_tokens": safe_div((upos_counts.get("VERB", 0) + upos_counts.get("AUX", 0)) * 100.0, token_count),
            "sx_coordination_density": safe_div(dep_counts.get("conj", 0) + upos_counts.get("CCONJ", 0), token_count),
            "sx_modifier_density": safe_div(dep_counts.get("amod", 0) + dep_counts.get("advmod", 0), token_count),
            "sx_nominal_density": safe_div(upos_counts.get("NOUN", 0) + upos_counts.get("PROPN", 0), token_count),
            "sx_verbal_density": safe_div(upos_counts.get("VERB", 0) + upos_counts.get("AUX", 0), token_count),
            "sx_np_like_ratio": safe_div(upos_counts.get("DET", 0) + upos_counts.get("ADJ", 0) + upos_counts.get("NOUN", 0) + upos_counts.get("PROPN", 0), token_count),
            "sx_pp_like_ratio": safe_div(upos_counts.get("ADP", 0), token_count),
            "sx_vp_like_ratio": safe_div(upos_counts.get("VERB", 0) + upos_counts.get("AUX", 0) + upos_counts.get("ADV", 0), token_count),
            "sx_subject_object_balance": safe_div(subject_count - object_count, dep_total),
            "sx_function_word_ratio_lexicon": safe_div(function_hits, token_count),
            "sx_content_word_ratio_lexicon": safe_div(content_like, token_count),
            "sx_mean_log_lemma_freq": _safe_stat(np.mean(lemma_log_freqs)) if lemma_log_freqs.size else np.nan,
            "sx_std_log_lemma_freq": _safe_stat(np.std(lemma_log_freqs)) if lemma_log_freqs.size else np.nan,
            "sx_rare_lemma_ratio": safe_div(int(np.sum(lemma_log_freqs < -4.0)), len(lemma_log_freqs)) if lemma_log_freqs.size else np.nan,
            "sx_very_rare_lemma_ratio": safe_div(int(np.sum(lemma_log_freqs < -5.0)), len(lemma_log_freqs)) if lemma_log_freqs.size else np.nan,
            "sx_adjacent_similarity_min": float(np.min(adjacent)) if adjacent else np.nan,
            "sx_adjacent_similarity_max": float(np.max(adjacent)) if adjacent else np.nan,
            "sx_pairwise_similarity_std": float(np.std(pairwise)) if pairwise else np.nan,
            "sx_low_similarity_frac_05": safe_div(sum(value < 0.5 for value in pairwise), len(pairwise)),
            "sx_low_similarity_frac_03": safe_div(sum(value < 0.3 for value in pairwise), len(pairwise)),
        }
    )
    return features


def richer_acoustic_features(audio_path: str) -> dict[str, float]:
    features = empty_par_features()
    path = Path(audio_path)
    if not audio_path or not path.exists():
        return features

    try:
        import librosa  # type: ignore

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return features

        hop_length = 512
        frame_length = 2048
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

        try:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, frame_length=frame_length, hop_length=hop_length)
            f0 = f0[np.isfinite(f0)]
        except Exception:
            f0 = np.array([], dtype=float)

        for idx in range(13):
            coeff = mfcc[idx]
            delta = delta_mfcc[idx]
            features.update(stats_from_array(f"par_mfcc_{idx + 1}", coeff))
            features[f"par_delta_mfcc_{idx + 1}_mean"] = _safe_stat(np.mean(delta))
            features[f"par_delta_mfcc_{idx + 1}_std"] = _safe_stat(np.std(delta))

        for name, values in {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "spectral_flatness": spectral_flatness,
            "rms": rms,
            "zcr": zcr,
        }.items():
            features.update(stats_from_array(f"par_{name}", np.asarray(values, dtype=float)))

        if f0.size:
            features["par_f0_mean"] = _safe_stat(np.mean(f0))
            features["par_f0_std"] = _safe_stat(np.std(f0))
            features["par_f0_min"] = _safe_stat(np.min(f0))
            features["par_f0_max"] = _safe_stat(np.max(f0))
            features["par_f0_median"] = _safe_stat(np.median(f0))
            features["par_f0_q25"] = _safe_stat(np.percentile(f0, 25))
            features["par_f0_q75"] = _safe_stat(np.percentile(f0, 75))
            features["par_f0_iqr"] = _safe_stat(np.percentile(f0, 75) - np.percentile(f0, 25))

        features["par_voiced_frame_ratio"] = safe_div(len(f0), len(rms))
        features["par_nonzero_rms_ratio"] = safe_div(int(np.sum(rms > 0)), len(rms))
    except Exception:
        return features

    return features


def build_phase2_metadata(phase1_metadata: pd.DataFrame, new_feature_columns: list[str]) -> pd.DataFrame:
    rows = phase1_metadata.to_dict("records")
    for name in new_feature_columns:
        group = name.split("_", 1)[0]
        rows.append(
            {
                "feature_name": name,
                "feature_group": group,
                "task_specific": int(group in {"pd", "rd", "fc"}),
                "valid_task_types": {
                    "pd": "PD_CTP",
                    "rd": "READING",
                    "fc": "CONVERSATION",
                    "ft": "FLUENCY",
                    "sr": "STORY_NARRATIVE|PICTURE_RECALL|PROMPT_FAMILY_STORY",
                    "pr": "ALL",
                    "sx": "ALL",
                    "par": "ALL",
                }.get(group, "ALL"),
                "requires_text": int(group != "par"),
                "requires_audio": int(group == "par"),
                "language_dependent_tooling": int(group in {"pd", "rd", "fc", "ft", "sr", "pr", "sx"}),
                "description": name.replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def finalize_group_availability(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    groups = sorted({name.split("_", 1)[0] for name in feature_columns if "_" in name})
    for group in groups:
        group_cols = [name for name in feature_columns if name.startswith(f"{group}_")]
        df[f"fg_{group}_available"] = (~df[group_cols].isna().all(axis=1)).astype(int)
        df[f"fg_{group}_num_missing"] = df[group_cols].isna().sum(axis=1)
    return df


def main() -> None:
    log = make_logger("phase2_extract_features")
    log("Loading phase1 manifest and feature table")
    manifest = pd.read_json(MANIFEST_PATH, lines=True)
    phase1_df = pd.read_csv(PHASE1_FEATURES_PATH)
    phase1_df = phase1_df.drop(columns=[column for column in phase1_df.columns if column.startswith("fg_")], errors="ignore")

    picture_lexicons = load_json(PICTURE_LEXICONS_PATH)
    reading_refs = load_json(READING_REFERENCES_PATH)
    story_refs = load_json(STORY_REFERENCES_PATH)
    fluency_resources = load_json(FLUENCY_RESOURCES_PATH)

    text_stats_cache = build_text_stats_cache(manifest, log)
    language_frequency_tables = build_language_frequency_tables(text_stats_cache, manifest)

    rows = []
    prompt_rows = []
    for idx, row in manifest.iterrows():
        if idx % 100 == 0:
            log(f"Phase2 feature progress: {idx}/{len(manifest)}")

        sample_id = row["sample_id"]
        base_row = phase1_df.loc[phase1_df["sample_id"] == sample_id].iloc[0].to_dict()
        stats = text_stats_cache.get(sample_id)
        if stats is None:
            text = clean_text(row.get("analysis_text", ""))
            stats = collect_doc_stats(text, row["language"], None)

        rich_row = {"sample_id": sample_id}
        pd_features, detected_prompt_family = picture_description_features(row, stats, base_row, picture_lexicons)
        rich_row.update(pd_features)
        rich_row.update(reading_features(row, stats, base_row, reading_refs))
        rich_row.update(story_features(row, stats, detected_prompt_family, story_refs))
        rich_row.update(fluency_features(row, stats, base_row, fluency_resources))
        rich_row.update(conversation_features(row, stats))
        rich_row.update(production_phrase_features(row, stats))
        rich_row.update(richer_syntax_features(row, stats, language_frequency_tables.get(row["language"], Counter())))
        rich_row.update(richer_acoustic_features(str(row.get("audio_path", ""))))
        rows.append(rich_row)
        prompt_rows.append(
            {
                "sample_id": sample_id,
                "prompt_id": row.get("prompt_id", ""),
                "phase2_prompt_family": detected_prompt_family,
            }
        )

    rich_df = pd.DataFrame(rows)
    prompt_df = pd.DataFrame(prompt_rows)
    phase2_df = phase1_df.merge(prompt_df, on="sample_id", how="left").merge(rich_df, on="sample_id", how="left")

    id_columns = {
        "sample_id",
        "group_id",
        "dataset_name",
        "language",
        "task_type",
        "diagnosis_mapped",
        "binary_label",
        "prompt_id",
        "phase2_prompt_family",
    }
    feature_columns = [column for column in phase2_df.columns if column not in id_columns and not column.startswith("fg_")]
    phase2_df = finalize_group_availability(phase2_df, feature_columns)

    phase1_metadata = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "phase1" / "phase1_feature_metadata.csv")
    new_feature_columns = [column for column in rich_df.columns if column != "sample_id"]
    metadata_df = build_phase2_metadata(phase1_metadata, new_feature_columns)

    phase2_df.to_csv(FEATURES_PATH, index=False)
    metadata_df.to_csv(METADATA_PATH, index=False)

    availability_rows = []
    for group in sorted({name.split("_", 1)[0] for name in feature_columns if "_" in name}):
        group_cols = [name for name in feature_columns if name.startswith(f"{group}_")]
        availability_rows.append(
            {
                "feature_group": group,
                "num_features": len(group_cols),
                "mean_available_fraction": float((~phase2_df[group_cols].isna().all(axis=1)).mean()),
                "mean_missing_fraction": float(phase2_df[group_cols].isna().mean().mean()),
            }
        )
    pd.DataFrame(availability_rows).to_csv(TABLES_PHASE2_ROOT / "feature_group_availability.csv", index=False)

    prompt_summary = (
        phase2_df.groupby(["task_type", "prompt_id", "phase2_prompt_family"])
        .size()
        .reset_index(name="n")
        .sort_values(["task_type", "n"], ascending=[True, False])
    )
    prompt_summary.to_csv(TABLES_PHASE2_ROOT / "phase2_prompt_family_counts.csv", index=False)

    write_json(
        SUMMARY_PATH,
        {
            "num_rows": int(len(phase2_df)),
            "num_phase1_features": int(len(phase1_metadata)),
            "num_new_phase2_features": int(len(new_feature_columns)),
            "num_total_features": int(len(feature_columns)),
            "task_counts": {str(key): int(value) for key, value in phase2_df["task_type"].value_counts().to_dict().items()},
            "prompt_family_counts": {
                str(key): int(value)
                for key, value in phase2_df["phase2_prompt_family"].fillna("").replace("", "UNRESOLVED").value_counts().to_dict().items()
            },
        },
    )
    log(f"Wrote phase2 feature matrix to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
