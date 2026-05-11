import re
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from processing.phase1.common import (
    PHASE1_ROOT,
    TABLES_PHASE1_ROOT,
    FILLERS,
    brunet_index,
    clean_text,
    count_syllables,
    honore_statistic,
    make_logger,
    mattr,
    safe_div,
    split_utterances,
    tokenize,
)


MANIFEST_PATH = PHASE1_ROOT / "phase1_manifest.jsonl"
FEATURES_PATH = PHASE1_ROOT / "phase1_features.csv"
METADATA_PATH = PHASE1_ROOT / "phase1_feature_metadata.csv"

_STANZA_PIPELINES = {}
_OPENSMILE = None

SUBORDINATE_DEPS = {"acl", "advcl", "ccomp", "csubj", "xcomp", "parataxis"}
CONTENT_UPOS = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"}
FUNCTION_UPOS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
OPEN_CLASS_UPOS = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"}
CLOSED_CLASS_UPOS = {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"}
CHAT_POS_TO_UPOS = {
    "adj": "ADJ",
    "adp": "ADP",
    "adv": "ADV",
    "aux": "AUX",
    "cconj": "CCONJ",
    "det": "DET",
    "intj": "INTJ",
    "n": "NOUN",
    "noun": "NOUN",
    "num": "NUM",
    "part": "PART",
    "prep": "ADP",
    "pron": "PRON",
    "propn": "PROPN",
    "q": "PUNCT",
    "sconj": "SCONJ",
    "sym": "SYM",
    "v": "VERB",
    "verb": "VERB",
}


def sanitize_feature_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def repeated_ngram_ratio(tokens: list[str], n: int) -> float:
    if len(tokens) < n or n <= 0:
        return np.nan
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return safe_div(repeated, len(ngrams))


def immediate_repetition_count(tokens: list[str]) -> int:
    return int(sum(1 for idx in range(1, len(tokens)) if tokens[idx] == tokens[idx - 1]))


def utterance_similarity_features(utterances: list[str]) -> dict[str, float]:
    if len(utterances) < 2:
        return {
            "disc_adjacent_utterance_similarity_mean": np.nan,
            "disc_adjacent_utterance_similarity_std": np.nan,
            "disc_mean_pairwise_utt_similarity": np.nan,
            "disc_semantic_drift": np.nan,
            "disc_local_coherence": np.nan,
        }

    try:
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(utterances)
        sims = cosine_similarity(matrix)
    except ValueError:
        return {
            "disc_adjacent_utterance_similarity_mean": np.nan,
            "disc_adjacent_utterance_similarity_std": np.nan,
            "disc_mean_pairwise_utt_similarity": np.nan,
            "disc_semantic_drift": np.nan,
            "disc_local_coherence": np.nan,
        }

    adjacent = [float(sims[i, i + 1]) for i in range(len(utterances) - 1)]
    pairwise = []
    for i in range(len(utterances)):
        for j in range(i + 1, len(utterances)):
            pairwise.append(float(sims[i, j]))

    local_coherence = float(np.mean(adjacent)) if adjacent else np.nan
    return {
        "disc_adjacent_utterance_similarity_mean": float(np.mean(adjacent)) if adjacent else np.nan,
        "disc_adjacent_utterance_similarity_std": float(np.std(adjacent)) if adjacent else np.nan,
        "disc_mean_pairwise_utt_similarity": float(np.mean(pairwise)) if pairwise else np.nan,
        "disc_semantic_drift": 1.0 - local_coherence if not np.isnan(local_coherence) else np.nan,
        "disc_local_coherence": local_coherence,
    }


def graph_features(tokens: list[str]) -> dict[str, float]:
    if len(tokens) < 2:
        return {
            "graph_node_count": float(len(set(tokens))),
            "graph_edge_count": np.nan,
            "graph_density": np.nan,
            "graph_self_loop_ratio": np.nan,
            "graph_largest_component_ratio": np.nan,
            "graph_average_degree": np.nan,
        }

    graph = nx.DiGraph()
    for left, right in zip(tokens[:-1], tokens[1:]):
        graph.add_edge(left, right, weight=graph.get_edge_data(left, right, {"weight": 0})["weight"] + 1)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    self_loops = nx.number_of_selfloops(graph)
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected)) if undirected.number_of_nodes() else []
    largest_component = max((len(component) for component in components), default=0)

    return {
        "graph_node_count": float(node_count),
        "graph_edge_count": float(edge_count),
        "graph_density": float(nx.density(graph)) if node_count > 1 else np.nan,
        "graph_self_loop_ratio": safe_div(self_loops, edge_count),
        "graph_largest_component_ratio": safe_div(largest_component, node_count),
        "graph_average_degree": safe_div(sum(dict(graph.degree()).values()), node_count),
    }


def get_stanza_pipeline(language: str):
    if language in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[language]
    import stanza  # type: ignore

    pipeline = stanza.Pipeline(
        language,
        processors="tokenize,pos,lemma,depparse",
        use_gpu=False,
        logging_level="ERROR",
    )
    _STANZA_PIPELINES[language] = pipeline
    return pipeline


def get_opensmile():
    global _OPENSMILE
    if _OPENSMILE is None:
        import opensmile  # type: ignore

        _OPENSMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _OPENSMILE


def sentence_tree_depth(sentence) -> list[int]:
    heads = {}
    for word in sentence.words:
        if isinstance(word.id, int):
            heads[word.id] = int(word.head)

    depths = []
    for word in sentence.words:
        if not isinstance(word.id, int):
            continue
        seen = set()
        node = word.id
        depth = 0
        while node in heads and heads[node] > 0 and node not in seen:
            seen.add(node)
            node = heads[node]
            depth += 1
        depths.append(depth)
    return depths


def dependency_depths_from_heads(heads: dict[int, int]) -> list[int]:
    depths = []
    for node in heads:
        seen = set()
        current = node
        depth = 0
        while current in heads and heads[current] > 0 and current not in seen:
            seen.add(current)
            current = heads[current]
            depth += 1
        depths.append(depth)
    return depths


def collect_doc_stats(text: str, language: str, doc) -> dict[str, object]:
    utterances = []
    tokens = []
    lemmas = []
    token_lengths = []
    upos_sequences = []
    upos_counts = Counter()
    dep_counts = Counter()
    dep_lengths = []
    tree_depths = []

    if doc is None:
        fallback_tokens = tokenize(text, language)
        fallback_utterances = split_utterances(text, language)
        return {
            "utterances": fallback_utterances,
            "tokens": fallback_tokens,
            "lemmas": fallback_tokens,
            "token_lengths": [len(token) for token in fallback_tokens],
            "upos_sequences": [],
            "upos_counts": upos_counts,
            "dep_counts": dep_counts,
            "dep_lengths": dep_lengths,
            "tree_depths": tree_depths,
        }

    for sentence in doc.sentences:
        sent_tokens = []
        sent_upos = []
        tree_depths.extend(sentence_tree_depth(sentence))
        for word in sentence.words:
            if word.upos == "PUNCT":
                continue
            token_text = clean_text(word.text).lower()
            lemma_text = clean_text(word.lemma or word.text).lower()
            if not token_text:
                continue
            sent_tokens.append(token_text)
            tokens.append(token_text)
            lemmas.append(lemma_text or token_text)
            token_lengths.append(len(token_text))
            upos_counts[word.upos] += 1
            sent_upos.append(word.upos)
            dep_label = str(word.deprel or "").split(":", 1)[0]
            if dep_label:
                dep_counts[dep_label] += 1
            if isinstance(word.id, int) and int(word.head) > 0:
                dep_lengths.append(abs(int(word.id) - int(word.head)))
        if sent_tokens:
            utterances.append(" ".join(sent_tokens))
            upos_sequences.append(sent_upos)

    return {
        "utterances": utterances if utterances else split_utterances(text, language),
        "tokens": tokens,
        "lemmas": lemmas if lemmas else tokens,
        "token_lengths": token_lengths,
        "upos_sequences": upos_sequences,
        "upos_counts": upos_counts,
        "dep_counts": dep_counts,
        "dep_lengths": dep_lengths,
        "tree_depths": tree_depths,
    }


def collect_chat_tier_stats(transcript_path: str, text: str, language: str) -> dict[str, object] | None:
    path = Path(transcript_path)
    if not transcript_path or path.suffix.lower() != ".cha" or not path.exists():
        return None

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if "%mor:" not in raw or "%gra:" not in raw:
        return None

    utterances = []
    tokens = []
    lemmas = []
    token_lengths = []
    upos_sequences = []
    upos_counts = Counter()
    dep_counts = Counter()
    dep_lengths = []
    tree_depths = []

    active_par = False
    current_upos_sequence = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("*PAR:"):
            if current_upos_sequence:
                upos_sequences.append(current_upos_sequence)
                current_upos_sequence = []
            active_par = True
            utterance_text = clean_text(stripped.replace("*PAR:", "", 1))
            if utterance_text:
                utterances.append(utterance_text.lower())
            continue
        if stripped.startswith("*") and not stripped.startswith("*PAR:"):
            if current_upos_sequence:
                upos_sequences.append(current_upos_sequence)
                current_upos_sequence = []
            active_par = False
            continue
        if not active_par:
            continue

        if stripped.startswith("%mor:"):
            body = stripped.split(":", 1)[1].strip()
            for item in body.split():
                if "|" not in item:
                    continue
                tag, rest = item.split("|", 1)
                upos = CHAT_POS_TO_UPOS.get(tag.lower())
                lemma = clean_text(rest.split("|", 1)[0]).lower()
                lemma = lemma.split("-", 1)[0]
                lemma = lemma.replace("~", "")
                if lemma and lemma not in {".", ",", "?", "!"}:
                    tokens.append(lemma)
                    lemmas.append(lemma)
                    token_lengths.append(len(lemma))
                if upos and upos != "PUNCT":
                    upos_counts[upos] += 1
                    current_upos_sequence.append(upos)

        if stripped.startswith("%gra:"):
            body = stripped.split(":", 1)[1].strip()
            heads = {}
            for item in body.split():
                parts = item.split("|")
                if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
                    continue
                token_id = int(parts[0])
                head = int(parts[1])
                dep = parts[2].split(":", 1)[0].lower()
                dep_counts[dep] += 1
                heads[token_id] = head
                if head > 0:
                    dep_lengths.append(abs(token_id - head))
            tree_depths.extend(dependency_depths_from_heads(heads))

    if not tokens and not utterances:
        return None

    if current_upos_sequence:
        upos_sequences.append(current_upos_sequence)

    return {
        "utterances": utterances if utterances else split_utterances(text, language),
        "tokens": tokens,
        "lemmas": lemmas if lemmas else tokens,
        "token_lengths": token_lengths,
        "upos_sequences": upos_sequences,
        "upos_counts": upos_counts,
        "dep_counts": dep_counts,
        "dep_lengths": dep_lengths,
        "tree_depths": tree_depths,
    }


def features_from_stats(stats: dict[str, object], language: str) -> dict[str, float]:
    utterances = stats["utterances"]
    tokens = stats["tokens"]
    lemmas = stats["lemmas"]
    token_lengths = stats["token_lengths"]
    upos_counts = stats["upos_counts"]
    dep_counts = stats["dep_counts"]
    dep_lengths = stats["dep_lengths"]
    tree_depths = stats["tree_depths"]

    token_count = len(tokens)
    type_count = len(set(tokens))
    lemma_type_count = len(set(lemmas))
    utterance_lengths = [len(tokenize(utt, language)) for utt in utterances if utt]
    syllable_count = count_syllables(tokens, language)
    filler_words = FILLERS.get(language, set())
    filler_count = sum(1 for token in tokens if token in filler_words)
    counts = Counter(tokens)
    hapax_count = sum(1 for count in counts.values() if count == 1)

    function_count = sum(upos_counts[pos] for pos in FUNCTION_UPOS)
    content_count = sum(upos_counts[pos] for pos in CONTENT_UPOS)
    open_class_count = sum(upos_counts[pos] for pos in OPEN_CLASS_UPOS)
    closed_class_count = sum(upos_counts[pos] for pos in CLOSED_CLASS_UPOS)
    verb_like_count = upos_counts["VERB"] + upos_counts["AUX"]
    subordinate_count = sum(dep_counts[label] for label in SUBORDINATE_DEPS)
    dep_total = sum(dep_counts.values())

    feature_row = {
        "len_token_count": float(token_count),
        "len_type_count": float(type_count),
        "len_utterance_count": float(len(utterances)),
        "len_mean_utterance_length": float(np.mean(utterance_lengths)) if utterance_lengths else np.nan,
        "len_std_utterance_length": float(np.std(utterance_lengths)) if utterance_lengths else np.nan,
        "len_tokens_per_utterance": safe_div(token_count, len(utterances)),
        "len_syllable_count": float(syllable_count),
        "lex_type_token_ratio": safe_div(type_count, token_count),
        "lex_lemma_type_token_ratio": safe_div(lemma_type_count, token_count),
        "lex_mattr_10": mattr(tokens, 10),
        "lex_mattr_20": mattr(tokens, 20),
        "lex_hapax_ratio": safe_div(hapax_count, token_count),
        "lex_repetition_rate": safe_div(sum(count - 1 for count in counts.values() if count > 1), token_count),
        "lex_mean_token_length": float(np.mean(token_lengths)) if token_lengths else np.nan,
        "lex_std_token_length": float(np.std(token_lengths)) if token_lengths else np.nan,
        "lex_function_word_ratio": safe_div(function_count, token_count),
        "lex_content_word_ratio": safe_div(content_count, token_count),
        "lex_brunet": brunet_index(tokens),
        "lex_honore": honore_statistic(tokens),
        "pause_filled_count": float(filler_count),
        "pause_filled_per_100_words": safe_div(filler_count * 100.0, token_count),
        "disc_repeated_unigram_ratio": repeated_ngram_ratio(tokens, 1),
        "disc_repeated_bigram_ratio": repeated_ngram_ratio(tokens, 2),
        "disc_repeated_trigram_ratio": repeated_ngram_ratio(tokens, 3),
        "disc_immediate_repetition_count": float(immediate_repetition_count(tokens)),
        "syn_noun_ratio": safe_div(upos_counts["NOUN"], token_count),
        "syn_verb_ratio": safe_div(upos_counts["VERB"], token_count),
        "syn_adj_ratio": safe_div(upos_counts["ADJ"], token_count),
        "syn_adv_ratio": safe_div(upos_counts["ADV"], token_count),
        "syn_adp_ratio": safe_div(upos_counts["ADP"], token_count),
        "syn_aux_ratio": safe_div(upos_counts["AUX"], token_count),
        "syn_pronoun_ratio": safe_div(upos_counts["PRON"], token_count),
        "syn_determiner_ratio": safe_div(upos_counts["DET"], token_count),
        "syn_cconj_ratio": safe_div(upos_counts["CCONJ"], token_count),
        "syn_sconj_ratio": safe_div(upos_counts["SCONJ"], token_count),
        "syn_interjection_ratio": safe_div(upos_counts["INTJ"], token_count),
        "syn_propn_ratio": safe_div(upos_counts["PROPN"], token_count),
        "syn_part_ratio": safe_div(upos_counts["PART"], token_count),
        "syn_num_ratio": safe_div(upos_counts["NUM"], token_count),
        "syn_noun_verb_ratio": safe_div(upos_counts["NOUN"], verb_like_count),
        "syn_pronoun_noun_ratio": safe_div(upos_counts["PRON"], upos_counts["NOUN"]),
        "syn_determiner_noun_ratio": safe_div(upos_counts["DET"], upos_counts["NOUN"]),
        "syn_aux_verb_ratio": safe_div(upos_counts["AUX"], upos_counts["VERB"]),
        "syn_content_function_ratio": safe_div(content_count, function_count),
        "syn_open_closed_ratio": safe_div(open_class_count, closed_class_count),
        "syn_clause_density": safe_div(verb_like_count, len(utterances)),
        "syn_subordination_ratio": safe_div(subordinate_count, len(utterances)),
        "syn_nsubj_ratio": safe_div(dep_counts["nsubj"], dep_total),
        "syn_obj_ratio": safe_div(dep_counts["obj"], dep_total),
        "syn_obl_ratio": safe_div(dep_counts["obl"], dep_total),
        "syn_advmod_ratio": safe_div(dep_counts["advmod"], dep_total),
        "syn_amod_ratio": safe_div(dep_counts["amod"], dep_total),
        "syn_mean_dependency_length": float(np.mean(dep_lengths)) if dep_lengths else np.nan,
        "syn_std_dependency_length": float(np.std(dep_lengths)) if dep_lengths else np.nan,
        "syn_max_dependency_length": float(np.max(dep_lengths)) if dep_lengths else np.nan,
        "syn_mean_tree_depth": float(np.mean(tree_depths)) if tree_depths else np.nan,
        "syn_max_tree_depth": float(np.max(tree_depths)) if tree_depths else np.nan,
        "syn_mean_utterance_length_words": float(np.mean(utterance_lengths)) if utterance_lengths else np.nan,
        "syn_utterance_count": float(len(utterances)),
    }

    feature_row.update(utterance_similarity_features(utterances))
    feature_row.update(graph_features(tokens))
    return feature_row


def parsed_text_features(text: str, language: str, doc) -> dict[str, float]:
    text = clean_text(text)
    stats = collect_doc_stats(text, language, doc)
    return features_from_stats(stats, language)


def opensmile_features(audio_path: str) -> dict[str, float]:
    if not audio_path or not Path(audio_path).exists():
        return {}
    try:
        smile = get_opensmile()
        frame = smile.process_file(audio_path)
        if frame.empty:
            return {}
        values = {}
        for column, value in frame.iloc[0].items():
            values[f"ac_egemaps_{sanitize_feature_name(column)}"] = float(value) if pd.notna(value) else np.nan
        return values
    except Exception:
        return {}


def librosa_audio_features(audio_path: str) -> dict[str, float]:
    base = {
        "ac_duration": np.nan,
        "ac_rms_mean": np.nan,
        "ac_rms_std": np.nan,
        "ac_zcr_mean": np.nan,
        "ac_zcr_std": np.nan,
        "ac_f0_mean": np.nan,
        "ac_f0_std": np.nan,
    }
    for idx in range(1, 6):
        base[f"ac_mfcc_{idx}_mean"] = np.nan
        base[f"ac_mfcc_{idx}_std"] = np.nan

    if not audio_path or not Path(audio_path).exists():
        return base

    try:
        import librosa  # type: ignore

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return base

        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)

        try:
            f0 = librosa.yin(
                y,
                fmin=50,
                fmax=400,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            f0 = f0[np.isfinite(f0)]
        except Exception:
            f0 = np.array([])

        base["ac_duration"] = float(len(y) / sr)
        base["ac_rms_mean"] = float(np.mean(rms))
        base["ac_rms_std"] = float(np.std(rms))
        base["ac_zcr_mean"] = float(np.mean(zcr))
        base["ac_zcr_std"] = float(np.std(zcr))
        base["ac_f0_mean"] = float(np.mean(f0)) if f0.size else np.nan
        base["ac_f0_std"] = float(np.std(f0)) if f0.size else np.nan
        for idx in range(5):
            base[f"ac_mfcc_{idx + 1}_mean"] = float(np.mean(mfcc[idx]))
            base[f"ac_mfcc_{idx + 1}_std"] = float(np.std(mfcc[idx]))
    except Exception:
        return base

    return base


def pauses_from_audio(audio_path: str) -> dict[str, float]:
    pause = {
        "pause_short_count": np.nan,
        "pause_medium_count": np.nan,
        "pause_long_count": np.nan,
        "pause_total_duration": np.nan,
        "pause_mean_duration": np.nan,
        "pause_median_duration": np.nan,
        "pause_std_duration": np.nan,
        "pause_iqr_duration": np.nan,
        "len_audio_duration": np.nan,
        "len_speech_duration": np.nan,
        "pause_silence_ratio": np.nan,
        "pause_speaking_to_total_ratio": np.nan,
    }

    if not audio_path or not Path(audio_path).exists():
        return pause

    try:
        import librosa  # type: ignore

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return pause

        hop_length = 512
        frame_length = 2048
        duration = float(len(y) / sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = max(float(np.max(rms)) * 0.02, 1e-4)
        voiced = rms > threshold
        frame_seconds = hop_length / sr
        silence_runs = []
        run = 0
        for is_voiced in voiced:
            if not is_voiced:
                run += 1
            elif run:
                silence_runs.append(run * frame_seconds)
                run = 0
        if run:
            silence_runs.append(run * frame_seconds)

        silence_runs = np.array(silence_runs, dtype=float)
        short_count = int(np.sum((silence_runs >= 0.2) & (silence_runs < 0.5)))
        medium_count = int(np.sum((silence_runs >= 0.5) & (silence_runs < 1.0)))
        long_count = int(np.sum(silence_runs >= 1.0))
        total_pause = float(np.sum(silence_runs)) if silence_runs.size else 0.0
        speech_duration = max(duration - total_pause, 0.0)

        pause["pause_short_count"] = short_count
        pause["pause_medium_count"] = medium_count
        pause["pause_long_count"] = long_count
        pause["pause_total_duration"] = total_pause
        pause["pause_mean_duration"] = float(np.mean(silence_runs)) if silence_runs.size else 0.0
        pause["pause_median_duration"] = float(np.median(silence_runs)) if silence_runs.size else 0.0
        pause["pause_std_duration"] = float(np.std(silence_runs)) if silence_runs.size else 0.0
        pause["pause_iqr_duration"] = (
            float(np.percentile(silence_runs, 75) - np.percentile(silence_runs, 25)) if silence_runs.size else 0.0
        )
        pause["len_audio_duration"] = duration
        pause["len_speech_duration"] = speech_duration
        pause["pause_silence_ratio"] = safe_div(total_pause, duration)
        pause["pause_speaking_to_total_ratio"] = safe_div(speech_duration, duration)
    except Exception:
        return pause

    return pause


def pauses_from_tsv(transcript_path: str) -> dict[str, float]:
    pause = {
        "pause_short_count": np.nan,
        "pause_medium_count": np.nan,
        "pause_long_count": np.nan,
        "pause_total_duration": np.nan,
        "pause_mean_duration": np.nan,
        "pause_median_duration": np.nan,
        "pause_std_duration": np.nan,
        "pause_iqr_duration": np.nan,
        "len_audio_duration": np.nan,
        "len_speech_duration": np.nan,
        "pause_silence_ratio": np.nan,
        "pause_speaking_to_total_ratio": np.nan,
    }
    path = Path(transcript_path)
    if not transcript_path or path.suffix.lower() != ".tsv" or not path.exists():
        return pause

    try:
        df = pd.read_csv(path, sep="\t")
        if not {"start_time", "end_time", "speaker", "value"}.issubset(df.columns):
            return pause

        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
        df = df.dropna(subset=["start_time", "end_time"]).copy()
        if df.empty:
            return pause

        speaker_series = df["speaker"].astype(str).str.lower()
        value_series = df["value"].astype(str).str.lower()
        silence_mask = (
            speaker_series.isin({"sil", "<noise>", "<deaf>"})
            | value_series.isin({"sil", "<noise>", "<deaf>"})
        )

        durations = (df["end_time"] - df["start_time"]).clip(lower=0)
        silence_runs = durations[silence_mask].to_numpy(dtype=float)
        total_duration = float(df["end_time"].max() - df["start_time"].min())
        speech_duration = float(durations[~silence_mask].sum())
        total_pause = float(silence_runs.sum()) if silence_runs.size else 0.0

        short_count = int(np.sum((silence_runs >= 0.2) & (silence_runs < 0.5)))
        medium_count = int(np.sum((silence_runs >= 0.5) & (silence_runs < 1.0)))
        long_count = int(np.sum(silence_runs >= 1.0))

        pause["pause_short_count"] = short_count
        pause["pause_medium_count"] = medium_count
        pause["pause_long_count"] = long_count
        pause["pause_total_duration"] = total_pause
        pause["pause_mean_duration"] = float(np.mean(silence_runs)) if silence_runs.size else 0.0
        pause["pause_median_duration"] = float(np.median(silence_runs)) if silence_runs.size else 0.0
        pause["pause_std_duration"] = float(np.std(silence_runs)) if silence_runs.size else 0.0
        pause["pause_iqr_duration"] = (
            float(np.percentile(silence_runs, 75) - np.percentile(silence_runs, 25)) if silence_runs.size else 0.0
        )
        pause["len_audio_duration"] = total_duration
        pause["len_speech_duration"] = speech_duration
        pause["pause_silence_ratio"] = safe_div(total_pause, total_duration)
        pause["pause_speaking_to_total_ratio"] = safe_div(speech_duration, total_duration)
    except Exception:
        return pause

    return pause


def acoustic_pause_features(audio_path: str, transcript_path: str) -> dict[str, float]:
    features = {}
    features.update(opensmile_features(audio_path))
    features.update(librosa_audio_features(audio_path))
    pause = pauses_from_audio(audio_path)
    if pd.isna(pause["len_audio_duration"]):
        pause = pauses_from_tsv(transcript_path)
    features.update(pause)
    return features


def compute_derived_rates(feature_row: dict[str, float]) -> dict[str, float]:
    token_count = feature_row.get("len_token_count", np.nan)
    audio_duration = feature_row.get("len_audio_duration", np.nan)
    speech_duration = feature_row.get("len_speech_duration", np.nan)
    syllable_count = feature_row.get("len_syllable_count", np.nan)
    pause_total = feature_row.get("pause_total_duration", np.nan)
    long_count = feature_row.get("pause_long_count", np.nan)
    short_count = feature_row.get("pause_short_count", np.nan)
    medium_count = feature_row.get("pause_medium_count", np.nan)
    total_pause_count = np.nansum([short_count, medium_count, long_count])

    feature_row["len_tokens_per_second"] = safe_div(token_count, audio_duration)
    feature_row["len_syllables_per_second"] = safe_div(syllable_count, audio_duration)
    feature_row["pause_pause_to_word_ratio"] = safe_div(total_pause_count, token_count)
    feature_row["pause_long_ratio"] = safe_div(long_count, total_pause_count)
    feature_row["pause_to_speech_ratio"] = safe_div(pause_total, speech_duration)
    feature_row["pause_speech_rate_words_per_second"] = safe_div(token_count, audio_duration)
    feature_row["pause_articulation_rate_words_per_speaking_second"] = safe_div(token_count, speech_duration)
    return feature_row


def finalize_group_availability(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    groups = sorted({name.split("_", 1)[0] for name in feature_columns if "_" in name})
    for group in groups:
        group_cols = [name for name in feature_columns if name.startswith(f"{group}_")]
        df[f"fg_{group}_available"] = (~df[group_cols].isna().all(axis=1)).astype(int)
        df[f"fg_{group}_num_missing"] = df[group_cols].isna().sum(axis=1)
    return df


def build_feature_metadata(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for name in feature_columns:
        group = name.split("_", 1)[0]
        rows.append(
            {
                "feature_name": name,
                "feature_group": group,
                "task_specific": False,
                "valid_task_types": "ALL",
                "requires_text": int(group not in {"ac"}),
                "requires_audio": int(group in {"ac", "pause"} or name in {"len_audio_duration", "len_speech_duration"}),
                "language_dependent_tooling": int(group in {"lex", "syn", "disc", "graph"}),
                "description": name.replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def build_text_feature_cache(manifest: pd.DataFrame, log) -> dict[str, dict[str, float]]:
    cache = {}
    for language, group in manifest.groupby("language"):
        text_rows = group[group["analysis_text"].astype(str).str.strip() != ""]
        if text_rows.empty:
            continue

        log(f"Parsing syntax for language={language} rows={len(text_rows)}")
        pipeline = None
        indices = text_rows.index.tolist()
        for processed, idx in enumerate(indices, start=1):
            text = clean_text(manifest.at[idx, "analysis_text"])
            sample_id = manifest.at[idx, "sample_id"]
            transcript_path = str(manifest.at[idx, "transcript_path"])
            tier_stats = collect_chat_tier_stats(transcript_path, text, language)
            if tier_stats is not None:
                cache[sample_id] = features_from_stats(tier_stats, language)
            elif language == "en" and manifest.at[idx, "dataset_name"] in {"WLS", "TAUKADIAL"}:
                continue
            else:
                if pipeline is None:
                    pipeline = get_stanza_pipeline(language)
                doc = pipeline(text)
                cache[sample_id] = parsed_text_features(text, language, doc)
            if processed % 100 == 0 or processed == len(indices):
                log(f"Parsed syntax language={language} progress={processed}/{len(indices)}")

    return cache


def main() -> None:
    log = make_logger("phase1_extract_features")
    log("Loading phase 1 manifest")
    manifest = pd.read_json(MANIFEST_PATH, lines=True)
    text_cache = build_text_feature_cache(manifest, log)

    rows = []
    for idx, row in manifest.iterrows():
        if idx % 100 == 0:
            log(f"Feature extraction progress: {idx}/{len(manifest)}")

        feature_row = {
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "dataset_name": row["dataset_name"],
            "language": row["language"],
            "task_type": row["task_type"],
            "diagnosis_mapped": row["diagnosis_mapped"],
            "binary_label": row["binary_label"],
        }

        text = clean_text(row.get("analysis_text", ""))
        text_features = text_cache.get(row["sample_id"])
        if text_features is None:
            text_features = parsed_text_features(text, row["language"], None)
        feature_row.update(text_features)

        audio_features = acoustic_pause_features(str(row.get("audio_path", "")), str(row.get("transcript_path", "")))
        feature_row.update(audio_features)
        feature_row = compute_derived_rates(feature_row)
        rows.append(feature_row)

    feature_df = pd.DataFrame(rows)
    id_columns = {"sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"}
    feature_columns = [column for column in feature_df.columns if column not in id_columns]
    feature_df = finalize_group_availability(feature_df, feature_columns)

    feature_df.to_csv(FEATURES_PATH, index=False)
    metadata_df = build_feature_metadata(feature_columns)
    metadata_df.to_csv(METADATA_PATH, index=False)

    availability = []
    for group in sorted({name.split("_", 1)[0] for name in feature_columns if "_" in name}):
        group_cols = [name for name in feature_columns if name.startswith(f"{group}_")]
        availability.append(
            {
                "feature_group": group,
                "num_features": len(group_cols),
                "mean_available_fraction": float((~feature_df[group_cols].isna().all(axis=1)).mean()),
                "mean_missing_fraction": float(feature_df[group_cols].isna().mean().mean()),
            }
        )
    pd.DataFrame(availability).to_csv(TABLES_PHASE1_ROOT / "feature_group_availability.csv", index=False)
    log(f"Wrote feature matrix to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
